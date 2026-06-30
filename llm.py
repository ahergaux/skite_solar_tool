import json
import os

from mistralai.client import Mistral
from logger import get_logger

log = get_logger("llm")


class MistralConnection:

    def __init__(self):
        log.info("Initializing MistralConnection")
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            log.error("MISTRAL_API_KEY is not set")
        self.client = Mistral(api_key=api_key)
        log.info("MistralConnection initialized")

    def query_for(self, text: str) -> list[str] | None:
        """
        Returns a list of offer strings scraped by the Mistral agent,
        or None if nothing was found.
        """
        if not text:
            log.info("query_for called with empty text — skipping")
            return None

        agent_id = os.environ.get("idAgent")
        log.info(f"Sending query to Mistral agent (agent_id={agent_id}), text length={len(text)}")
        log.debug(f"Query text: {text!r:.200}")

        try:
            response = self.client.beta.conversations.start(
                agent_id=agent_id,
                inputs=text,
            )
        except Exception as e:
            log.error(f"Mistral API call failed: {e}")
            return None

        log.debug(f"Mistral response received, {len(response.outputs)} output entries")

        for entry in response.outputs:
            if entry.type == "message.output":
                for chunk in entry.content:
                    if not hasattr(chunk, "text"):
                        continue

                    raw = chunk.text.strip()
                    log.debug(f"Raw Mistral output: {raw!r:.300}")

                    parsed = self._parse_offers(raw)
                    if parsed is not None:
                        log.info(f"Mistral query_for returned {len(parsed)} offers")
                        return parsed
                    else:
                        log.error("Could not extract a list of offers from Mistral output")

        log.info("query_for: no exploitable output found in Mistral response")
        return None

    def _parse_offers(self, raw: str) -> list[str] | None:
        """
        Parses the Mistral output into a list of offer strings.

        Accepts:
          - {"data": [...]}        → returns the list
          - {"offers": [...]}      → returns the list
          - [...]                  → returns the list directly
          - {"data": [{"title":...}, ...]}  → each dict is serialised as a string
        """
        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.splitlines()[:-1])
        raw = raw.strip()

        # Find the outermost JSON by depth counting (handles nested objects)
        candidates = []
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch in ("{", "["):
                if depth == 0:
                    start = i
                depth += 1
            elif ch in ("}", "]"):
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw[start:i + 1])

        for span in sorted(candidates, key=len, reverse=True):
            try:
                parsed = json.loads(span)
            except json.JSONDecodeError:
                continue

            # Already a list
            if isinstance(parsed, list):
                return [json.dumps(item) if isinstance(item, dict) else str(item)
                        for item in parsed]

            # Dict with a known list key
            if isinstance(parsed, dict):
                for key in ("data", "offers", "results", "items"):
                    if isinstance(parsed.get(key), list):
                        items = parsed[key]
                        log.debug(f"_parse_offers: found {len(items)} items under key '{key}'")
                        return [json.dumps(item) if isinstance(item, dict) else str(item)
                                for item in items]

        log.error(f"_parse_offers: no list found in output — raw: {raw!r:.200}")
        return None
