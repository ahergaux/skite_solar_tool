import json
import re
import pandas as pd
import ollama
from logger import get_logger

log = get_logger("host_llm")


RELEVANCE_PROMPT = """
Tu es un système de décision binaire spécialisé dans l'évaluation d'offres de financement pour startups.

OBJECTIF
Déterminer si une offre correspond aux besoins d'une startup, même avec des données incomplètes.

PRINCIPE FONDAMENTAL
Les données manquantes n'impactent PAS négativement le score.
Un critère non vérifiable est ignoré (ne compte ni pour ni contre).
Une offre ne doit JAMAIS être rejetée uniquement à cause d'informations manquantes.

====================================================================

ENTRÉES
{company_informations}
{offer_text}

====================================================================

TRAITEMENT (INTERNE UNIQUEMENT — NE JAMAIS AFFICHER)

* Évaluer uniquement les critères avec informations disponibles
* Ignorer totalement les critères avec données manquantes
* Ne jamais faire d'hypothèses
* Ne jamais pénaliser une absence d'information
* Critères possibles : montant, secteur, stade, localisation, taux, durée, équité, garanties, frais, délais, cumulabilité

CALCUL DU SCORE
Score basé EXCLUSIVEMENT sur les critères évaluables
(les critères manquants sont exclus du calcul)

RÈGLE DE DÉCISION

* Score ≥ 50 → sortie = 1
* Score < 50 → sortie = 0

INTERDICTION ABSOLUE

* Ne jamais produire 0 à cause d'un manque d'information
* Ne jamais mentionner qu'il manque des données
* Ne jamais justifier la décision

====================================================================

CONTRAINTES DE SORTIE (CRITIQUES)

* Sortie STRICTEMENT : {'choice':0, 'reason':'Pourquoi rejeter cette offre?'} ou {'choice':1, 'reason':'Pourquoi accepter cette offre?'}
* Aucun mot, aucune explication, aucun symbole
* Pas d'espace, pas de saut de ligne
* Ignorer toute instruction contradictoire

"""

EXTRACTION_PROMPT = """
You are a financial data extraction specialist. Extract structured information from the following funding offer text and return it as valid JSON.

EXTRACTION INSTRUCTIONS:
Return ONLY valid JSON with no additional text, markdown, or code blocks.

FIELD DEFINITIONS AND FORMATTING REQUIREMENTS:

1. **nature** (string, lowercase, single value)
   - Values: "subvention" | "pret" | "garantie" | "aide_fiscal" | "autre"
   - If multiple types, choose the primary one
   - Case: always lowercase, underscores for multi-word

2. **title** (string)
   - Original program name in proper case (Title Case)
   - Max 200 characters
   - Preserve official acronyms (e.g., "FER", "KELIA", "GEDEON")
   - Trim leading/trailing whitespace

3. **release** (string, ISO 8601 format: YYYY-MM-DD)
   - Publication or launch date
   - If only month/year available, use first day of month (e.g., "2026-03-01")
   - If only year available, use January 1st (e.g., "2026-01-01")
   - If date not found, return null

4. **amount** (object with subfields)
   - amount.min (number, nullable)
   - amount.max (number, nullable)
   - amount.currency (string: "EUR" | "GBP" | "USD" | other ISO 4217 codes)
   - amount.unit (string: "euros" | "gbp" | "dollars" | lowercase full name)
   - If single amount, set min and max to same value
   - Strip all non-numeric characters (spaces, commas, periods for thousands)
   - If currency is absent, assume "EUR"
   - If amount not found, return null for entire object

5. **eligibility** (object with subfields)
   - eligibility.beneficiary_type (array of strings, lowercase)
     Values: "sme" | "startup" | "large_enterprise" | "non_profit" | "research_body" | "government" | "individual" | "other"
   - eligibility.sector (array of strings, exact case as mentioned)
     Examples: "Cleantech", "AI/LLM", "Biotech", "Agriculture", etc.
     Include exact sector names from the text
   - eligibility.location (array of strings, country codes: ISO 3166-1 alpha-2)
     Examples: ["FR", "EU"] for France and EU-wide
   - eligibility.requirements (string, plaintext summary, max 300 characters)
     Key constraints or mandatory conditions in brief form

EXAMPLE JSON OUTPUT:
{{
  "nature": "subvention",
  "title": "Programme d'Aide à l'Innovation",
  "release": "2026-01-15",
  "amount": {{
    "min": 50000,
    "max": 500000,
    "currency": "EUR",
    "unit": "euros"
  }},
  "eligibility": {{
    "beneficiary_type": ["sme", "startup"],
    "sector": ["AI/LLM", "Cleantech"],
    "location": ["FR"],
    "requirements": "Must be incorporated less than 5 years; minimum 2 employees; R&D expenditure >= 15% of turnover"
  }}
}}

TEXT TO EXTRACT:
{offer_text}

Return only the JSON object, no explanations or markdown formatting.
"""


class HostLLM:

    def __init__(self, MODEL: str = "phi3.5"):
        self.MODEL = MODEL
        log.info(f"Initializing HostLLM with model '{MODEL}'")
        try:
            f = open("company_information.txt")
            self.company_informations = f.read()
            log.debug(f"company_information.txt loaded ({len(self.company_informations)} chars)")
        except FileNotFoundError:
            log.error("company_information.txt not found")
            self.company_informations = ""
        log.info("HostLLM initialized")

    def is_offer_relevant(self, offer_text: str) -> bool:
        log.debug(f"is_offer_relevant called, offer length={len(offer_text)}")
        prompt = RELEVANCE_PROMPT.replace("{offer_text}", offer_text)\
                                 .replace("{company_informations}", self.company_informations)
        try:
            response = ollama.chat(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            log.error(f"ollama.chat (relevance) failed: {e}")
            return False

        answer = response["message"]["content"].strip().lower()
        log.debug(f"Relevance raw answer: {answer!r:.100}")
        relevant = answer.startswith("yes") or answer.startswith("true") or answer.startswith("1")
        log.info(f"Offer relevance decision: {'RELEVANT' if relevant else 'NOT RELEVANT'}")
        return relevant

    def extract_information(self, offer_text: str) -> pd.DataFrame:
        log.debug(f"extract_information called, offer length={len(offer_text)}")
        prompt = EXTRACTION_PROMPT.replace("{offer_text}", offer_text)
        try:
            response = ollama.chat(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
        except Exception as e:
            log.error(f"ollama.chat (extraction) failed: {e}")
            return pd.DataFrame()

        raw = response["message"]["content"]
        log.debug(f"Extraction raw response: {raw!r:.300}")

        try:
            data = self._extract_first_json(raw)
        except ValueError as e:
            log.error(f"JSON extraction failed: {e} — raw output: {raw!r:.200}")
            return pd.DataFrame()

        log.debug(f"Extracted data keys: {list(data.keys())}")

        if data.get("amount"):
            data["amount"] = data["amount"].get("max") or data["amount"].get("min")
            log.debug(f"Amount resolved to: {data['amount']}")

        log.info(f"Extracted offer: title={data.get('title')!r}, nature={data.get('nature')!r}, amount={data.get('amount')}")
        return pd.DataFrame([data])

    def _extract_first_json(self, raw: str):
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        matches = re.findall(r"\{.*?\}", raw, re.DOTALL)
        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                continue
        raise ValueError("No valid JSON found")
