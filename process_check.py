import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import ollama

from host_llm import HostLLM, OLLAMA_NUM_THREAD, CANONICAL_COLUMNS
from logger import get_logger

log = get_logger("process_check")

EMBED_MODEL = "phi3.5"

# Similarity thresholds
TITLE_HIGH_SIM = 0.90
TITLE_MED_SIM  = 0.80
INST_SIM       = 0.80
AMOUNT_TOL     = 0.05

# Parallel workers — 3 is a good balance on an i5 to leave headroom for the OS
FILTER_WORKERS = 3


class ProcessCheck:

    def __init__(self):
        log.info("Initializing ProcessCheck")
        self.offer_processed = self._load_csv("offer_processed.csv")
        self.opportunities = self._load_csv("opportunities.csv")

        self.llm = HostLLM()

        # Thread-safe lock for CSV writes and DataFrame mutations
        self._write_lock = threading.Lock()

        # Embedding cache — avoids re-computing the same text twice
        self._embed_cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()

        self._warmup_embed_cache()
        log.info("ProcessCheck initialized")

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            # Drop any columns that don't belong to the canonical schema
            extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
            if extra:
                log.error(f"{path}: dropping unknown columns {extra} — schema mismatch detected")
                df = df.drop(columns=extra)
            # Add any missing canonical columns
            for col in CANONICAL_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df = df[CANONICAL_COLUMNS]
            log.debug(f"{path} loaded ({len(df)} rows)")
            return df
        except FileNotFoundError:
            log.info(f"{path} not found — creating empty file with canonical schema")
            df = pd.DataFrame(columns=CANONICAL_COLUMNS)
            df.to_csv(path, index=False)
            return df

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def filter(self, offers: list) -> int:
        log.info(f"filter() called with {len(offers)} offers — {FILTER_WORKERS} parallel workers")
        count = 0

        with ThreadPoolExecutor(max_workers=FILTER_WORKERS) as executor:
            futures = {
                executor.submit(self._process_one, i, offer): i
                for i, offer in enumerate(offers)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    count += future.result()
                except Exception as e:
                    log.error(f"Unhandled error processing offer {idx + 1}: {e}")

        log.info(f"filter() done — {count} new opportunities added")
        return count

    def _process_one(self, i: int, offer: str) -> int:
        log.debug(f"Processing offer {i + 1}: {str(offer)!r:.100}")

        extracted = self.llm.extract_information(offer)
        if extracted.empty:
            log.error(f"Extraction returned empty DataFrame for offer {i + 1} — skipping")
            return 0

        if self._is_blank(extracted.iloc[0]):
            log.error(
                f"Offer {i + 1}: extraction returned only null/empty fields — "
                f"discarding without saving to CSV (raw offer: {str(offer)!r:.150})"
            )
            return 0

        with self._write_lock:
            already_known = self.match(extracted, self.opportunities)

        if already_known:
            log.info(f"Offer {i + 1} already in opportunities — skipping")
            return 0

        with self._write_lock:
            self.add_offer_processed(extracted)

        if self.llm.is_offer_relevant(offer):
            with self._write_lock:
                self.add_opportunity(extracted)
            log.info(f"Offer {i + 1} is relevant — added as new opportunity")
            return 1

        log.info(f"Offer {i + 1} is not relevant — not added to opportunities")
        return 0

    @staticmethod
    def _is_blank(row: pd.Series) -> bool:
        """True if every field of the extracted row is null/None/empty."""
        for v in row:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, (list, str)) and len(v) == 0:
                continue
            return False
        return True

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

    def _warmup_embed_cache(self):
        """Pre-compute embeddings for all values already in the DB."""
        if self.opportunities.empty:
            return
        cols = [c for c in ("title", "institution") if c in self.opportunities.columns]
        values = set()
        for col in cols:
            values.update(self.opportunities[col].dropna().unique())
        log.info(f"Warming up embedding cache for {len(values)} unique DB values...")
        for val in values:
            self._embed(str(val))
        log.info(f"Embedding cache warm — {len(self._embed_cache)} entries")

    def _embed(self, text) -> np.ndarray:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return np.zeros(1)
        key = str(text)
        with self._cache_lock:
            if key in self._embed_cache:
                log.debug(f"_embed(): cache hit for {key!r:.60}")
                return self._embed_cache[key]

        log.debug(f"_embed(): computing embedding for {key!r:.60}")
        try:
            vec = np.array(
                ollama.embed(
                    model=EMBED_MODEL,
                    input=key,
                    options={"num_thread": OLLAMA_NUM_THREAD},
                ).embeddings[0]
            )
        except Exception as e:
            log.error(f"_embed() failed for {key!r:.60}: {e}")
            return np.zeros(1)

        with self._cache_lock:
            self._embed_cache[key] = vec
        return vec

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
