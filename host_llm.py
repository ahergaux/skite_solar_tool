import json
import re
import pandas as pd
import ollama


RELEVANCE_PROMPT = """
Tu es un expert financier chevronné spécialisé dans le financement des startups.
Ton rôle: Évaluer si une offre de financement (subvention, programme, ou prêt) correspond aux besoins d'une startup, même si certaines informations manquent.
IMPORTANT: Les données manquantes ne bloquent PAS l'analyse. Tu évalues avec les infos disponibles et tu signales simplement ce qui manque.
====================================================================

INFORMATIONS DE L'ENTREPRISE:
{company_informations}
====================================================================

OFFRE DE FINANCEMENT À ANALYSER:
{offer_text}
====================================================================

ANALYSE REQUISE:

ADÉQUATION GÉNÉRALE

Cette offre correspond-elle aux besoins de la startup?
Pourcentage de match (0-100%)?


POINTS POSITIFS

Quels aspects de l'offre sont particulièrement avantageux?


POINTS NÉGATIFS OU RISQUES

Y a-t-il des incompatibilités ou des risques?


DONNÉES MANQUANTES

Quelles informations seraient utiles pour affiner l'analyse?


RECOMMANDATION

Faut-il postuler? Pourquoi?
Actions à prendre avant de postuler?



====================================================================

INSTRUCTIONS FINALES:

Sois direct et pragmatique
Ne rejette pas une offre juste parce que des données manquent
Utilise les infos disponibles pour faire une évaluation honnête
Signale clairement ce qui manque, mais continue l'analyse
Donne des recommandations actionnables

RESTRICTIONS:

Restrictions particulières:
Cumulable avec autres offres?:

====================================================================

ANALYSE REQUISE:
Analyse l'offre selon le profil de l'entreprise.
NE PAS REJETER L'ANALYSE si des informations manquent.

À la place, continuer l'analyse avec les données disponibles

et signaler ce qui manque dans une section "Données manquantes".
STRUCTURE D'ANALYSE:

CORRESPONDANCE GLOBALE

Score: X/100

Basé sur le match avec les critères évaluables
CRITÈRES VÉRIFIÉS

Pour chaque critère applicable:

✓ Respecté

✗ Non respecté

? Donné insuffisant / manquant
Format:

Critère | Résultat | Détail
Critères à évaluer (si applicable):

Montant (min/max vs besoin)
Secteur (accepté/refusé)
Stade (création/amorçage/croissance)
Localisation (pays/région acceptable)
Taux d'intérêt (si prêt, acceptable?)
Durée (si prêt, ok?)
Équité (si accélérateur, acceptable?)
Garanties (réalistes pour startup?)
Frais (élevés?)
Délai de déblocage
Cumulabilité avec autres

Si Score > 70%:

✅ RECOMMANDÉ DE POSTULER

Si Score 50-70%:

⚠️ À CONSIDÉRER

Actions préalables:

Clarifier point X avant de postuler
Évaluer impact de Y
Vérifier si Z est réaliste

Si Score < 50%:

❌ NON RECOMMANDÉ

====================================================================

CRITÈRES D'ÉVALUATION DÉTAILLÉS:
FINANCEMENT:

• Montant demandé: Correspond-il à l'offre min/max?

• Flexibilité: Offre permet ajustement?

• Timing: Déblocage assez rapide?

SECTEUR & STADE:

• Secteur accepté? (pas dans liste noire?)

• Stade correspond?

• Alignement avec phase de développement actuelle?

CONDITIONS FINANCIÈRES:

• Taux d'intérêt compétitif si prêt?

• Durée de remboursement viable?

• Dilution acceptable en cas de prise de capital?

• Ensemble des frais gérables?

GARANTIES & SÉCURITÉ:

• Exigences de garanties réalistes pour le stade?

• Possibilité de constituer hypothèques ou nantissements?

• Coûts associés aux garanties justifiés?

ZONE GÉOGRAPHIQUE:

• Localisation de la startup compatible avec critères d'éligibilité?

• Limites territoriales qui pourraient disqualifier?
DÉLAIS:

• Calendrier de candidature faisable pour constituer le dossier?

• Déblocage des fonds suffisamment rapide?

• Capacité d'attente de la startup réaliste?

OBLIGATIONS CONTRACTUELLES:

• Reporting demandé gérable au quotidien?

• Audits fréquents et coûteux?

• Restrictions sur l'utilisation des fonds problématiques?

• Conditions de sortie trop contraignantes?

PROFIL CIBLE:

• Fondateurs correspondent-ils au profil recherché?

• Expérience requise compatible avec équipe actuelle?

• Taille minimale d'équipe satisfaite?
====================================================================

RÈGLES IMPORTANTES:

NE JAMAIS BLOQUER SUR DONNÉES MANQUANTES

Évaluer avec ce qu'on a
Signaler ce qui manque
Faire une recommandation quand même


ÊTRE OBJECTIF ET FACTUEL

Baser sur les données fournies
Éviter les suppositions
Si doute: signaler comme "donné insuffisant"


ÊTRE CONSTRUCTIF

Actions concrètes, pas juste opinions
Pistes pour clarifier points flous
Alternatives si pas bon match


DISTINGUER LES NIVEAUX

"Critère non respecté" ≠ "Information manquante"
"Offre refusée" ≠ "À clarifier"


CONTEXTUALISER

Une startup en amorçage ≠ scaling
Un prêt diffère d'une subvention
Les critères s'adaptent selon le contexte



====================================================================

FORMAT DE RÉPONSE:
Renvoie un simple booléen, 1 si l'offre est intéressante, 0 sinon.
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

    def __init__(self, MODEL:str = "phi3.5"):
        self.MODEL = MODEL
        f = open("company_information.txt")
        self.company_informations = f.read()

    def is_offer_relevant(self, offer_text: str) -> bool:
        """Returns a bool to acknowledge the offer is relevant to the company."""
        prompt = RELEVANCE_PROMPT.format(offer_text=offer_text, company_informations = self.company_informations)
        response = ollama.chat(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"].strip().lower()
        return answer.startswith("yes") or answer.startswith("true") or answer.startswith("1")

    def extract_information(self,  offer_text: str) -> pd.DataFrame:
        """
        Return pd.DataFrame that corresponds to the format expected by offer_processed.csv and opportunities.csv.
        Expected columns: nature, title, release, amount, eligibility
        """
        prompt = EXTRACTION_PROMPT.format(offer_text=offer_text)
        response = ollama.chat(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        raw = response["message"]["content"]

        # Strip markdown code fences if the model still wraps output
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)

        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])

        return pd.DataFrame([data])
