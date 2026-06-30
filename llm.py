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

    def query_for(self, text: str):
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
                    if hasattr(chunk, "text"):
                        text_output = chunk.text.strip()
                        log.debug(f"Raw Mistral output: {text_output!r:.300}")

                        start = text_output.find("{")
                        end = text_output.rfind("}") + 1

                        if start != -1 and end != -1:
                            result = text_output[start:end].replace("\n", "")
                            log.info(f"Mistral query_for returned JSON of length {len(result)}")
                            return result
                        else:
                            log.error("Mistral output does not contain a valid JSON object")

        log.info("query_for: no exploitable output found in Mistral response")
        return None
