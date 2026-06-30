import os

from mistralai.client import Mistral


class MistralConnection:

    def __init__(self):
        self.client = Mistral(api_key=(os.environ.get("MISTRAL_API_KEY")))



    def query_for(self, text:str):
        if text=="" or text is None:
            return
        
        response = self.client.beta.conversations.start(
            agent_id=os.environ.get("idAgent") ,
            inputs=text,
        )

        return response
    


