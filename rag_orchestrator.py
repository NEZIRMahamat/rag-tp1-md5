""" Orchestrateur de RAG"""


import json
from groq import Groq
from config import GROQ_API_KEY, RAG_LLM, PROMPT_RAG_SYSTEM_FILE
from moderator import Moderator
from vector_base import VectorDB


class RagOrchestrator:

    def __init__(self):
        
        self.client = Groq(api_key=GROQ_API_KEY) # Client Groq
        self.vector_base = VectorDB() # Base vectorielle        
        self.moderator = Moderator(client=self.client) # Modérateur de questions

    
    def read_prompt_system_file(self):
        with open(PROMPT_RAG_SYSTEM_FILE, "r", encoding="utf-8") as file:
            content = file.read()
            return content
        
    # Complète le prompt système avec les chunks et complétion valeur du marqueur
    def compose_prompt_with_chunks(self, list_chunks):
        """
        Compose le prompt système en complètant la valeur du marqueur "Chunks : { LIST_CHUNKS }" par les chunks fournis.
        LIST_CHUNKS : à remplacer par la liste des chunks.
        Args:
            list_chunks (list): Liste des chunks à insérer dans le prompt système.

        Returns:
            str: Le prompt système complété avec les chunks.
        """
        system_prompt_template = self.read_prompt_system_file()
        chunks_text = "\n".join(list_chunks)
        composed_prompt = system_prompt_template.replace("LIST_CHUNKS", chunks_text)
        
        return composed_prompt

    

    # Appel LLM RAG (cas positif, pas d'injection)
    def get_response_from_llm(self, list_chunks, user_prompt):
        chat_completion = self.client.chat.completions.create(
            model=RAG_LLM,
            messages=[
                {"role": "system", "content": self.compose_prompt_with_chunks(list_chunks)},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        response = json.loads(chat_completion.choices[0].message.content) # Python dict 
        
        return response 
    
    def answer_question(self, question, top_k=3):
        # Vérification question => modération
        moderation_result = self.moderator.moderate(question)
        if moderation_result.get("is_prompt_injection"):
            return {"error": "Tentative de prompt injection détectée."}
        
        # Si pas d'injection, on continue le pipeline
        vector_db_results = self.vector_base.retrieve(question, top_k) # top_k pour récupérer les chunks les plus proches (variable)
        list_chunks = list(vector_db_results['documents'][0])
        print(f"Type de list_chunks : {type(list_chunks)}")
        print(f"Chunks récupérés depuis la base vectorielle : {list_chunks}")
        results_llm_rag = self.get_response_from_llm(list_chunks=list_chunks, user_prompt=question)
        
        return results_llm_rag





