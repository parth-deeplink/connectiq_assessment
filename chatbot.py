from openai import OpenAI
from dotenv import load_dotenv
import os



load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


class RAGChatbot:
    def __init__(self, vector_store, llm_model_name="gpt-4o-mini"):
        self.vector_store = vector_store
        self.llm_model = "gpt-4o-mini" # For OpenAI
        self.llm_model_name = llm_model_name
        self.conversation_history = [] # Stores [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

    def _generate_response_from_llm(self, user_question, retrieved_context):
        """
        Constructs the prompt and calls the LLM.
        """
        context_str = "\n---\n".join([
            f"Page {c['page_number']}:\n{c['text']}" for c in retrieved_context
        ])

        system_prompt = f"""You are a helpful assistant that answers questions about the book "The Hard Thing About Hard Things" by Ben Horowitz.
Use only the provided context to answer the question. If the answer is not in the context, state that you don't know.
For each piece of information you provide, you MUST cite the page number(s) from the context.
Format your answer clearly, including the answer and then the references at the end.

Example Reference Format:
[Excerpt: "...", Page: N]

Context:
{context_str}
"""

        messages = [{"role": "user", "content": system_prompt + f"\nUser Question: {user_question}"}]
        
        # For multi-turn, add history before the current query
        for message in self.conversation_history:
            messages.append(message)
        
        # Add the current user question again to the messages if it wasn't already added in the history
        if not self.conversation_history or self.conversation_history[-1]['content'] != user_question:
             messages.append({"role": "user", "content": user_question})


        try:
            # For OpenAI
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0 # Make it deterministic
            )
            return response.choices[0].message.content

 

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "I apologize, but I'm unable to generate a response at this moment."

    def ask(self, user_question):
        """
        Main function to ask a question to the chatbot.
        """
        # Step 1: Retrieve relevant chunks
        retrieved_context = self.vector_store.search(user_question, k=5) # Retrieve top 5 chunks

        if not retrieved_context:
            return "I couldn't find any relevant information in the book for your question."

        # Step 2: Generate response using LLM with context
        llm_answer = self._generate_response_from_llm(user_question, retrieved_context)
        
        # Step 3: Update conversation history
        self.conversation_history.append({"role": "user", "content": user_question})
        self.conversation_history.append({"role": "assistant", "content": llm_answer})

        return llm_answer

    def reset_chat(self):
        self.conversation_history = []
        print("Chat history cleared.")

