import re
from datetime import datetime

import numpy as np
import pandas as pd
import ollama

from host_llm import HostLLM

EMBED_MODEL = "phi3.5"

# Similarity thresholds
TITLE_HIGH_SIM = 0.90   # title alone is sufficient to declare a match
TITLE_MED_SIM  = 0.80   # title needs one corroborating field
INST_SIM       = 0.80   # institution similarity threshold
AMOUNT_TOL     = 0.05   # 5% relative tolerance on numeric amounts


class ProcessCheck:

    def __init__(self):
        self.offer_processed = pd.read_csv("offer_processed.csv")
        self.opportunities = pd.read_csv("opportunities.csv")
        self.llm = HostLLM()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def filter(self, offers: list):
        for offer in offers:
            if self.match(self.llm.extract_information(offer), self.opportunities):
                continue
            self.add_offer_processed(offer)
            if self.llm.is_offer_relevant(offer):
                self.add_opportunity(offer)

    def match(self, offer: pd.DataFrame, db: pd.DataFrame) -> bool:
        """Return True if the offer is already present in db.

        Matching rules (any one is sufficient):
          1. title similarity >= 0.90  (strong title match alone)
          2. title similarity >= 0.80  AND  institution similarity >= 0.80
          3. title similarity >= 0.80  AND  amount within 5%
          4. institution similarity >= 0.80  AND  same release date  AND  amount within 5%
        """
        if db.empty:
            return False

        row = offer.iloc[0]

        title_emb = self._embed(row.get("title"))
        inst_emb  = self._embed(row.get("institution"))

        for _, db_row in db.iterrows():
            title_sim = self._cosine(title_emb, self._embed(db_row.get("title")))

            # Rule 1: high title similarity alone
            if title_sim >= TITLE_HIGH_SIM:
                return True

            if title_sim >= TITLE_MED_SIM:
                inst_sim = self._cosine(inst_emb, self._embed(db_row.get("institution")))

                # Rule 2: title + institution
                if inst_sim >= INST_SIM:
                    return True

                # Rule 3: title + amount
                if self._amount_match(row.get("amount"), db_row.get("amount")):
                    return True

            # Rule 4: institution + release date + amount (no title needed)
            inst_sim = self._cosine(inst_emb, self._embed(db_row.get("institution")))
            if (inst_sim >= INST_SIM
                    and self._date_match(row.get("release"), db_row.get("release"))
                    and self._amount_match(row.get("amount"), db_row.get("amount"))):
                return True

        return False

    def add_opportunity(self, offer: pd.DataFrame):
        self.opportunities = pd.concat([self.opportunities, offer], ignore_index=True)
        print(f"New opportunity:{offer}")

    def add_offer_processed(self, offer: pd.DataFrame):
        self.offer_processed = pd.concat([self.offer_processed, offer], ignore_index=True)
        print(f"New offer processed:{offer}")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text) -> np.ndarray:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return np.zeros(1)
        response = ollama.embed(model=EMBED_MODEL, input=str(text))
        return np.array(response.embeddings[0])

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Field-specific comparison helpers
    # ------------------------------------------------------------------

    def _parse_amount(self, text) -> float | None:
        """Parse a monetary amount string to float, handling European and SI formats."""
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return None
        # Strip everything except digits, comma, dot
        s = re.sub(r"[^\d.,]", "", str(text))
        if not s:
            return None
        # Detect European thousand-separator: "50.000" or "1.500.000"
        parts = s.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None

    def _amount_match(self, a, b) -> bool:
        va, vb = self._parse_amount(a), self._parse_amount(b)
        if va is None or vb is None:
            return False
        if va == 0 and vb == 0:
            return True
        return abs(va - vb) / max(abs(va), abs(vb)) <= AMOUNT_TOL

    def _parse_date(self, text) -> datetime | None:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(str(text).strip(), fmt)
            except ValueError:
                continue
        return None

    def _date_match(self, a, b) -> bool:
        da, db = self._parse_date(a), self._parse_date(b)
        return da is not None and db is not None and da == db
