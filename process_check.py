import re
from datetime import datetime

import numpy as np
import pandas as pd
import ollama

from host_llm import HostLLM
from logger import get_logger

log = get_logger("process_check")

EMBED_MODEL = "phi3.5"

# Similarity thresholds
TITLE_HIGH_SIM = 0.90
TITLE_MED_SIM  = 0.80
INST_SIM       = 0.80
AMOUNT_TOL     = 0.05


class ProcessCheck:

    def __init__(self):
        log.info("Initializing ProcessCheck")
        try:
            self.offer_processed = pd.read_csv("offer_processed.csv")
            log.debug(f"offer_processed.csv loaded ({len(self.offer_processed)} rows)")
        except FileNotFoundError:
            log.error("offer_processed.csv not found — starting with empty DataFrame")
            self.offer_processed = pd.DataFrame()

        try:
            self.opportunities = pd.read_csv("opportunities.csv")
            log.debug(f"opportunities.csv loaded ({len(self.opportunities)} rows)")
        except FileNotFoundError:
            log.error("opportunities.csv not found — starting with empty DataFrame")
            self.opportunities = pd.DataFrame()

        self.llm = HostLLM()
        log.info("ProcessCheck initialized")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def filter(self, offers: list) -> int:
        log.info(f"filter() called with {len(offers)} offers")
        count = 0
        for i, offer in enumerate(offers):
            log.debug(f"Processing offer {i + 1}/{len(offers)}: {str(offer)!r:.100}")

            extracted = self.llm.extract_information(offer)
            if extracted.empty:
                log.error(f"Extraction returned empty DataFrame for offer {i + 1} — skipping")
                continue

            if self.match(extracted, self.opportunities):
                log.info(f"Offer {i + 1} already in opportunities — skipping")
                continue

            self.add_offer_processed(extracted)

            if self.llm.is_offer_relevant(offer):
                self.add_opportunity(extracted)
                count += 1
                log.info(f"Offer {i + 1} is relevant and added as opportunity (total so far: {count})")
            else:
                log.info(f"Offer {i + 1} is not relevant — not added to opportunities")

        log.info(f"filter() done — {count} new opportunities added")
        return count

    def match(self, offer: pd.DataFrame, db: pd.DataFrame) -> bool:
        """Return True if the offer is already present in db."""
        if db.empty:
            log.debug("match(): db is empty — no match possible")
            return False

        row = offer.iloc[0]
        title = row.get("title")
        log.debug(f"match(): checking offer title={title!r} against {len(db)} db rows")

        title_emb = self._embed(row.get("title"))
        inst_emb  = self._embed(row.get("institution"))

        for idx, db_row in db.iterrows():
            title_sim = self._cosine(title_emb, self._embed(db_row.get("title")))
            log.debug(f"  db row {idx}: title_sim={title_sim:.3f}")

            if title_sim >= TITLE_HIGH_SIM:
                log.info(f"match(): MATCH via rule 1 (title_sim={title_sim:.3f}) with db row {idx}")
                return True

            if title_sim >= TITLE_MED_SIM:
                inst_sim = self._cosine(inst_emb, self._embed(db_row.get("institution")))
                log.debug(f"  db row {idx}: inst_sim={inst_sim:.3f}")

                if inst_sim >= INST_SIM:
                    log.info(f"match(): MATCH via rule 2 (title+inst) with db row {idx}")
                    return True

                if self._amount_match(row.get("amount"), db_row.get("amount")):
                    log.info(f"match(): MATCH via rule 3 (title+amount) with db row {idx}")
                    return True

            inst_sim = self._cosine(inst_emb, self._embed(db_row.get("institution")))
            if (inst_sim >= INST_SIM
                    and self._date_match(row.get("release"), db_row.get("release"))
                    and self._amount_match(row.get("amount"), db_row.get("amount"))):
                log.info(f"match(): MATCH via rule 4 (inst+date+amount) with db row {idx}")
                return True

        log.debug(f"match(): no match found for title={title!r}")
        return False

    def add_opportunity(self, offer: pd.DataFrame):
        title = offer.iloc[0].get("title") if not offer.empty else "?"
        log.info(f"Adding new opportunity: {title!r}")
        self.opportunities = pd.concat([self.opportunities, offer], ignore_index=True)
        self.opportunities.to_csv("opportunities.csv", sep=",", index=False)
        log.debug("opportunities.csv updated")

    def add_offer_processed(self, offer: pd.DataFrame):
        title = offer.iloc[0].get("title") if not offer.empty else "?"
        log.info(f"Adding to offer_processed: {title!r}")
        self.offer_processed = pd.concat([self.offer_processed, offer], ignore_index=True)
        self.offer_processed.to_csv("offer_processed.csv", sep=",", index=False)
        log.debug("offer_processed.csv updated")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text) -> np.ndarray:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            log.debug("_embed(): received None/NaN — returning zero vector")
            return np.zeros(1)
        log.debug(f"_embed(): embedding text={str(text)!r:.60}")
        try:
            response = ollama.embed(model=EMBED_MODEL, input=str(text))
            return np.array(response.embeddings[0])
        except Exception as e:
            log.error(f"_embed() failed for text={str(text)!r:.60}: {e}")
            return np.zeros(1)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Field-specific comparison helpers
    # ------------------------------------------------------------------

    def _parse_amount(self, text) -> float | None:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return None
        s = re.sub(r"[^\d.,]", "", str(text))
        if not s:
            return None
        parts = s.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            log.debug(f"_parse_amount(): could not parse {text!r}")
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
        log.debug(f"_parse_date(): unrecognized date format: {text!r}")
        return None

    def _date_match(self, a, b) -> bool:
        da, db = self._parse_date(a), self._parse_date(b)
        return da is not None and db is not None and da == db
