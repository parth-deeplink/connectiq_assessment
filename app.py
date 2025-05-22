import gradio as gr
from chatbot import RAGChatbot
from pdf_processor import extract_text_from_pdf, chunk_text
from embedding_store import LocalVectorStore
import os
from openai import OpenAI
from dotenv import load_dotenv
# --- Global Initialization (Run once when the app starts) ---
PDF_PATH = "./the_hard_thing_about_hard_things.pdf"
CHATBOT = None
INITIALIZED = False
OPENAI_CLIENT = None 

def initialize_openai_client():
    """Initializes the OpenAI client and handles API key loading."""
    global OPENAI_CLIENT
    load_dotenv() # Load environment variables
    
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
        print("Please ensure your .env file exists and contains OPENAI_API_KEY=YOUR_KEY.")
        # You might want to raise an exception or exit here
        return False # Indicate failure
    
    try:
        OPENAI_CLIENT = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized successfully.")
        return True # Indicate success
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return False # Indicate failure

def initialize_chatbot():
    """
    Initializes the chatbot components. This can take some time due to embedding generation.
    """
    global CHATBOT, INITIALIZED
    if INITIALIZED:
        return "Chatbot is already initialized."

    print("Initializing chatbot components. This may take a while for embedding generation...")
    if not os.path.exists(PDF_PATH):
        return f"Error: PDF file not found at {PDF_PATH}. Please place the PDF in the project directory."
    
    raw_documents = extract_text_from_pdf(PDF_PATH)
    processed_chunks = chunk_text(raw_documents, chunk_size=2000, overlap=200) # Match chunking used in chatbot.py

    # Initialize with Google Gemini embedding model
    vector_store = LocalVectorStore(embedding_model_name="models/text-embedding-3-small") 
    vector_store.add_documents(processed_chunks)

    # Initialize chatbot with Google Gemini Pro
    CHATBOT = RAGChatbot(vector_store, llm_model_name="gpt-4o-mini") 
    INITIALIZED = True
    print("Chatbot initialized successfully!")
    return "Chatbot is ready! You can now ask questions."

def predict(message, history):
    """
    Gradio interface function to handle chat messages.
    """
    global CHATBOT
    if not CHATBOT:
        yield "Chatbot is not initialized yet. Please wait or click 'Initialize Chatbot'."
        return
    
    # Gradio's history is a list of lists: [[user_msg, bot_msg], ...]
    # We need to convert it to the format our chatbot expects.
    # For multi-turn, ensure the chatbot's internal history is updated correctly.
    # In this simplified setup, we'll let `chatbot.ask` manage its own history.
    # If you want Gradio's history to fully control the chatbot's history,
    # you'd need to clear and re-populate `chatbot.conversation_history` based on `history`.
    
    # For a simple multi-turn using chatbot's internal history:
    response = CHATBOT.ask(message)
    # The chatbot's `ask` method already updates its internal `conversation_history`.
    
    # Gradio expects the full conversation for the next turn.
    # `history` represents the past, `message` is the current user input.
    # We yield the final response.
    yield response


def reset_gradio_chat():
    """Resets the Gradio chat interface and the chatbot's internal history."""
    global CHATBOT
    if CHATBOT:
        CHATBOT.reset_chat()
    return [], "Chat history cleared. You can start a new conversation." # Clear Gradio history and provide a message


# Gradio Interface Setup
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # "The Hard Thing About Hard Things" RAG Chatbot
        Ask me anything about Ben Horowitz's book!
        """
    )

    # Initialize button and message
    init_btn = gr.Button("Initialize Chatbot (Click once)")
    init_status = gr.Textbox(label="Initialization Status", interactive=False)
    
    # Chat interface
    chatbot_interface = gr.ChatInterface(
        fn=predict,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask a question about the book...", container=False, scale=7),
        clear_btn="Clear Chat", # This will call the predict function with empty history
        submit_btn="Send",
        # title="The Hard Thing About Hard Things Chatbot",
        # description="Ask anything about the book by Ben Horowitz.",
        # theme="soft",
        # examples=["What is the hardest thing about building a company?", "Tell me about the good product manager, bad product manager concept.", "What are the key takeaways from the book about leadership?"]
    )
    
    init_btn.click(initialize_chatbot, inputs=None, outputs=init_status)
    # Override Gradio's default clear functionality to also reset our chatbot's history
    chatbot_interface.clear_btn.click(reset_gradio_chat, outputs=[chatbot_interface.chatbot, init_status])


if __name__ == "__main__":
    # Ensure you have your API key set up (e.g., in config.py or environment variables)
    # Before launching, you might want to pre-initialize the chatbot to avoid delay
    # if you're not using the "Initialize Chatbot" button.
    # For seamless local execution, uncomment the following line:
    # initialize_chatbot() 
    demo.launch(share=False) # Set share=True to get a public link (temporarily)