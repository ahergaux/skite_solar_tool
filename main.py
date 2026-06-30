import os
from dotenv import load_dotenv
from mail import iCloudConnection
from llm import MistralConnection
from host_llm import HostLLM
from process_check import ProcessCheck

def main():
    assert load_dotenv()

    mail_client = iCloudConnection()
    mistral_client = MistralConnection()

    host_llm = HostLLM()

    check = ProcessCheck()

    # Mails
    mails = mail_client.get_all()
    check.filter([f"{mail.header} - {mail.body}" for mail in mails])
    
    # Scrap
    with open("to_check.txt") as f: 
        new_offers = mistral_client.query_for(f.read())
        check.filter(new_offers)


if __name__=="__main__":
    main()