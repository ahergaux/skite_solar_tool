import json
import re
import pandas as pd
import ollama
from logger import get_logger

log = get_logger("host_llm")

# Canonical flat schema — every CSV row must have exactly these columns in this order
CANONICAL_COLUMNS = [
    "nature",
    "title",
    "institution",
    "release",
    "amount",
    "beneficiary_type",
    "sector",
    "location",
    "requirements",
]

# Threads alloués à chaque appel ollama (i5 = 4 cœurs / 8 threads avec HT)
OLLAMA_NUM_THREAD = 8

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
You are a financial data extraction specialist. Extract structured information from the following funding offer text and return it as a single flat JSON object.

EXTRACTION INSTRUCTIONS:
Return ONLY valid JSON with no additional text, markdown, or code blocks.
All fields must be at the TOP LEVEL of the JSON object — no nested objects.

FIELD DEFINITIONS:

1. **nature** (string, lowercase)
   Values: "subvention" | "pret" | "garantie" | "aide_fiscal" | "autre"

2. **title** (string, Title Case, max 200 chars)
   Preserve official acronyms (e.g., "FER", "KELIA", "GEDEON")

3. **institution** (string)
   Name of the funding body / organisation behind the offer. null if unknown.

4. **release** (string, ISO 8601: YYYY-MM-DD)
   Publication or launch date. If only month/year: first day of month. null if not found.

5. **amount** (number)
   Maximum funding amount as a plain integer (no currency symbol, no thousands separator).
   Use the minimum if no maximum is given. null if not found.

6. **beneficiary_type** (array of strings, lowercase)
   Values: "sme" | "startup" | "large_enterprise" | "non_profit" | "research_body" | "government" | "individual" | "other"

7. **sector** (array of strings, exact case from text)
   Examples: "Cleantech", "AI/LLM", "Biotech", "Agriculture"

8. **location** (array of strings, ISO 3166-1 alpha-2 country codes)
   Examples: ["FR"] for France, ["FR", "EU"] for EU-wide

9. **requirements** (string, max 300 chars)
   Key eligibility conditions in plain text. null if none found.

EXAMPLE OUTPUT:
{{
  "nature": "subvention",
  "title": "Programme d'Aide à l'Innovation",
  "institution": "BPI France",
  "release": "2026-01-15",
  "amount": 500000,
  "beneficiary_type": ["sme", "startup"],
  "sector": ["AI/LLM", "Cleantech"],
  "location": ["FR"],
  "requirements": "Incorporated less than 5 years; minimum 2 employees; R&D >= 15% of turnover"
}}

TEXT TO EXTRACT:
{offer_text}

Return only the JSON object, no explanations or markdown formatting.
"""


class HostLLM:

    def __init__(
        self,
        model_extract: str = "phi3.5",
        model_relevance: str = "qwen2.5:0.5b",
    ):
        self.MODEL_EXTRACT = model_extract
        self.MODEL_RELEVANCE = model_relevance
        log.info(f"Initializing HostLLM — extract={model_extract!r}, relevance={model_relevance!r}")
        try:
            f = open("company_information.txt")
            self.company_informations = f.read()
            log.debug(f"company_information.txt loaded ({len(self.company_informations)} chars)")
        except FileNotFoundError:
            log.error("company_information.txt not found")
            self.company_informations = ""
        self._verify_models()
        log.info("HostLLM initialized")

    def _verify_models(self):
        """Log a warning if a required model is not pulled yet."""
        try:
            available = {m.model for m in ollama.list().models}
            for model in (self.MODEL_EXTRACT, self.MODEL_RELEVANCE):
                if model not in available:
                    log.error(
                        f"Model {model!r} not found locally — run: ollama pull {model}"
                    )
        except Exception as e:
            log.error(f"Could not list ollama models: {e}")

    def is_offer_relevant(self, offer_text: str) -> bool:
        log.debug(f"is_offer_relevant called with model={self.MODEL_RELEVANCE!r}, offer length={len(offer_text)}")
        prompt = RELEVANCE_PROMPT.replace("{offer_text}", offer_text)\
                                 .replace("{company_informations}", self.company_informations)
        try:
            response = ollama.chat(
                model=self.MODEL_RELEVANCE,
                messages=[{"role": "user", "content": prompt}],
                options={"num_thread": OLLAMA_NUM_THREAD},
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
        log.debug(f"extract_information called with model={self.MODEL_EXTRACT!r}, offer length={len(offer_text)}")
        prompt = EXTRACTION_PROMPT.replace("{offer_text}", offer_text)
        try:
            response = ollama.chat(
                model=self.MODEL_EXTRACT,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"num_thread": OLLAMA_NUM_THREAD},
            )
        except Exception as e:
            log.error(f"ollama.chat (extraction) failed: {e}")
            return pd.DataFrame()

        raw = response["message"]["content"]
        log.debug(f"Extraction raw response: {raw!r:.300}")

        try:
            data = self._parse_json(raw)
        except ValueError as e:
            log.error(f"JSON extraction failed: {e} — raw output: {raw!r:.200}")
            return pd.DataFrame()

        # Flatten legacy nested eligibility if the LLM still returns it
        if isinstance(data.get("eligibility"), dict):
            elig = data.pop("eligibility")
            for key in ("beneficiary_type", "sector", "location", "requirements"):
                if key not in data:
                    data[key] = elig.get(key)

        # Flatten legacy nested amount if the LLM still returns it
        if isinstance(data.get("amount"), dict):
            amt = data["amount"]
            data["amount"] = amt.get("max") or amt.get("min")

        # Enforce canonical schema: keep only known columns, fill missing with None
        row = {col: data.get(col) for col in CANONICAL_COLUMNS}

        log.debug(f"Extracted data: {row}")
        log.info(
            f"Extracted offer: title={row.get('title')!r}, "
            f"nature={row.get('nature')!r}, amount={row.get('amount')}"
        )
        return pd.DataFrame([row], columns=CANONICAL_COLUMNS)

    def _parse_json(self, raw: str) -> dict:
        """Parse the largest valid JSON object from a raw LLM string."""
        # Strip markdown code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)

        # Try the whole string first (fast path, avoids regex on nested structures)
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: find all {...} spans, pick the largest that parses
        candidates = []
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw[start:i + 1])

        for span in sorted(candidates, key=len, reverse=True):
            try:
                result = json.loads(span)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON object found in LLM output")
