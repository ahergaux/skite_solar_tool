import os
from dotenv import load_dotenv
from mail import iCloudConnection
from llm import MistralConnection
from host_llm import HostLLM
from process_check import ProcessCheck
from logger import get_logger

log = get_logger("main")

def main():
    log.info("=== Skite Solar Tool starting ===")

    if not load_dotenv():
        log.error("Failed to load .env file — aborting")
        raise RuntimeError(".env file not found or empty")
    log.debug(".env loaded successfully")

    log.info("Initializing clients...")
    mail_client = iCloudConnection()
    mistral_client = MistralConnection()
    host_llm = HostLLM()
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
        if new_offers is None:
            log.info("Mistral returned no offers from scrap")
            scrap_new = 0
        else:
            log.info(f"{len(new_offers)} new offers found via scrap")
            scrap_new = check.filter(new_offers)
        new_op += scrap_new
        log.info(f"Scrap processing done — {scrap_new} new opportunities from scrap")
    except FileNotFoundError:
        log.error("to_check.txt not found — skipping scrap processing")

    log.info(f"=== Run complete — {new_op} total new opportunities ===")

if __name__ == "__main__":
    main()
