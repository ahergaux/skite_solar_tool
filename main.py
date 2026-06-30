import os
import subprocess
from dotenv import load_dotenv
from mail import iCloudConnection
from llm import MistralConnection
from host_llm import HostLLM
from process_check import ProcessCheck
from logger import get_logger

log = get_logger("main")

# Models used — pull them if missing
MODELS_REQUIRED = ["phi3.5", "qwen2.5:0.5b"]


def ensure_models():
    """Pull any missing ollama model before the run starts."""
    import ollama as _ollama
    try:
        available = {m.model for m in _ollama.list().models}
    except Exception as e:
        log.error(f"Could not query ollama model list: {e}")
        return
    for model in MODELS_REQUIRED:
        if model not in available:
            log.info(f"Model {model!r} not found locally — pulling...")
            try:
                subprocess.run(["ollama", "pull", model], check=True)
                log.info(f"Model {model!r} pulled successfully")
            except subprocess.CalledProcessError as e:
                log.error(f"Failed to pull model {model!r}: {e}")
        else:
            log.debug(f"Model {model!r} already available")


def main():
    log.info("=== Skite Solar Tool starting ===")

    if not load_dotenv():
        log.error("Failed to load .env file — aborting")
        raise RuntimeError(".env file not found or empty")
    log.debug(".env loaded successfully")

    # These env vars configure the ollama *server* daemon.
    # Set them before starting the ollama service, or add them to your shell profile.
    # OLLAMA_NUM_PARALLEL=3   — concurrent requests the server handles
    # OLLAMA_NUM_THREAD=8     — CPU threads per request (i5 = 8 with HT)
    # OLLAMA_FLASH_ATTENTION=1 — enable Flash Attention if supported by model

    ensure_models()

    log.info("Initializing clients...")
    mail_client = iCloudConnection()
    mistral_client = MistralConnection()
    check = ProcessCheck()
    log.info("All clients initialized")

    new_op = 0

    # --- Mails ---
    log.info("Starting mail processing")
    mails = mail_client.get_all()
    log.info(f"{len(mails)} new offers found in mailbox")
    mail_new = check.filter([f"{mail.header} - {mail.body}" for mail in mails])
    new_op += mail_new
    log.info(f"Mail processing done — {mail_new} new opportunities from mails")

    # --- Scrap ---
    log.info("Starting scrap processing")
    try:
        with open("to_check.txt") as f:
            urls_text = f.read()
        log.debug(f"to_check.txt content: {urls_text!r}")
        new_offers = mistral_client.query_for(urls_text)
        if not new_offers:
            log.info("Mistral returned no offers from scrap")
            scrap_new = 0
        else:
            log.info(f"{len(new_offers)} offers found via scrap")
            scrap_new = check.filter(new_offers)
        new_op += scrap_new
        log.info(f"Scrap processing done — {scrap_new} new opportunities from scrap")
    except FileNotFoundError:
        log.error("to_check.txt not found — skipping scrap processing")

    log.info(f"=== Run complete — {new_op} total new opportunities ===")


if __name__ == "__main__":
    main()
