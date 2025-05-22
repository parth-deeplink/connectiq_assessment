import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


from dotenv import load_dotenv
import os

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

class LocalVectorStore:
    def __init__(self, embedding_model_name="text-embedding-3-small"): 
        self.embedding_model = "text-embedding-3-small" # For OpenAI
        self.embedding_model_name = embedding_model_name
        self.documents = [] # Stores {'text': ..., 'page_number': ..., 'embedding': np.array}
        self.embeddings = [] # Stores just the embeddings for quick search

    def _get_embedding(self, text):
        """Generates embedding using the chosen LLM API."""
        try:
            #For OpenAI
            response = client.embeddings.create(input=text,
            model=self.embedding_model)
            return np.array(response.data[0].embedding)


        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def add_documents(self, chunks):
        """Adds processed text chunks to the store with their embeddings."""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        for chunk in chunks:
            embedding = self._get_embedding(chunk['text'])
            if embedding is not None:
                self.documents.append({
                    'text': chunk['text'],
                    'page_number': chunk['page_number'],
                    'embedding': embedding
                })
                self.embeddings.append(embedding)
        self.embeddings = np.array(self.embeddings) # Convert to numpy array for efficient operations
        print(f"Added {len(self.documents)} documents to the store.")

    def search(self, query_text, k=3):
        """Performs a similarity search for the query."""
        query_embedding = self._get_embedding(query_text)
        if query_embedding is None or len(self.embeddings) == 0:
            return []

        # Reshape for sklearn's cosine_similarity
        query_embedding = query_embedding.reshape(1, -1) 

        # Calculate cosine similarity between query embedding and all document embeddings
        # This is where numpy and sklearn come in handy
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get indices of top-k most similar documents
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Retrieve the actual documents
        results = []
        for idx in top_k_indices:
            results.append(self.documents[idx])
        return results

