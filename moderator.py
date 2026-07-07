"""Modérateur de questions pour détecter les tentatives de prompt injection."""


import json
from groq import Groq
from config import SG_LLM, PROMPT_MODERATOR_SYSTEM_FILE


class Moderator:

    def __init__(self, client : Groq, prompt_system_file=PROMPT_MODERATOR_SYSTEM_FILE):
        """
        Initialise le modérateur avec le fichier de prompt système spécifié.
        """
        self.client = client
        self.prompt_system_file = prompt_system_file

    def read_prompt_system_file(self):
        with open(self.prompt_system_file, "r", encoding="utf-8") as file:
            content = file.read()
            return content
        

    def moderate(self, question):
        chat_completion = self.client.chat.completions.create(
            model=SG_LLM,
            messages=[
                {"role": "system", "content": self.read_prompt_system_file()},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object", "schema": {"is_prompt_injection": "boolean"}},
            temperature=0
        )

        moderation_response = json.loads(chat_completion.choices[0].message.content) # Déserialisation JSON -> Python dict 

        return moderation_response