from chromadb import PersistentClient

from sentence_transformers import SentenceTransformer
import pandas as pd

from config import EMBEDDING_MODEL_NAME


class VectorDB:
    def __init__(self, path : str = "vector_db", chunks : pd.DataFrame = None):
       # Si base existe => charger; sinon, si on a des chunks => créer la base; sinon => erreur
       self.path = path
       self.chunks = chunks

       try:
           self.client = PersistentClient(path=self.path)
           self.collection = self.client.get_or_create_collection(name="mes_collections")
       except Exception as exc:
           raise RuntimeError(f"Initialisation VectorDB impossible: {exc}") from exc
       
       # 1. base existe deja => only recharge, sans reinsertion
       try:
           existing_count = self.collection.count()
           if existing_count > 0:
               metadatas = self.collection.get(include=["metadatas"])["metadatas"]
               model_name_in_metadata = metadatas[0].get("model_name") if metadatas else None
               if not model_name_in_metadata:
                     raise ValueError(
                          "La base vectorielle existe, mais ses métadonnées ne contiennent pas 'model_name'. "
                          "Supprimez le dossier vector_db puis recréez l'index avec cette version du code."
                     )
               self.model_embedding = SentenceTransformer(model_name_in_metadata)
               if model_name_in_metadata != EMBEDDING_MODEL_NAME:
                     raise ValueError(
                          f"Rechargement, erreur : le modèle {model_name_in_metadata} est différent du modèle embedding actuel {EMBEDDING_MODEL_NAME}, "
                     )
               print(f"Base existante chargée avec succès. Nombre de chunks existants : {existing_count}")
               return
               
       except Exception as exc:
           raise RuntimeError(f"Impossible de lire le contenu de la collection: {exc}") from exc

       
       # 2. base vide + chunks fournis => creation de l'index
       if self.chunks is not None and len(self.chunks) > 0:
           try:
               self.model_embedding = SentenceTransformer(EMBEDDING_MODEL_NAME)
               embeddings = self.model_embedding.encode(
                   self.chunks["text"].tolist(),
                   normalize_embeddings=True,
                   show_progress_bar=True,
                   batch_size=8,
               )
               self.collection.add(
                   ids=self.chunks["id"].tolist(),
                   documents=self.chunks["text"].tolist(),
                   embeddings=embeddings.tolist(), 
                   metadatas=[
                       {
                           "source": self.chunks["source"].iloc[i],
                           "categorie": self.chunks["categorie"].iloc[i],
                           "model_name": EMBEDDING_MODEL_NAME,
                        
                        } for i in range(len(self.chunks))
                   ],
               )
               print(f"Première création de l'index vectoriel réussie avec {len(self.chunks)} chunks.")
           except Exception as exc:
               raise RuntimeError(f"Creation de l'index vectoriel impossible: {exc}") from exc
           return

       # 3. base vide, pas de chunks fournis => Erreur explicite
       raise ValueError(
           "Aucune base existante et aucun chunk fourni. Impossible de créer l'index vectoriel."
       )


    # Recherche sémantique 
    def retrieve(self, question, top_k):
        if not question or not isinstance(question, str):
            raise ValueError("La question doit être une chaîne de caractères non vide.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k doit être un entier strictement positif.")

        try:
            question_embedding = self.model_embedding.encode(
                [question], normalize_embeddings=True
            )
            results = self.collection.query(
                query_embeddings=question_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas"],
            )
            return results
        except Exception as exc:
            raise RuntimeError(f"Erreur pendant la recherche des chunks: {exc}") from exc
        


    