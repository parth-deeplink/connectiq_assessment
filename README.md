# "The Hard Thing About Hard Things" RAG Chatbot

This is a RAG (Retrieval Augmented Generation) chatbot that can answer questions about the book "The Hard Thing About Hard Things" by Ben Horowitz. It uses a local, in-memory "vector store" for retrieval and an LLM for generation.


Using in memory as vector store it is simple for execution. for production we can go for other vector DB's

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/].....


    ```
    **For OpenAI:**


    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt


## How to Run the Chatbot

1.  **Navigate to the project directory:**
    ```bash
    cd connectiq_assessment
    ```

2.  **Run the Gradio application:**
    ```bash
    python app.py
    ```

3.  **Access the Chatbot:**
    * Open your web browser and go to the URL provided by Gradio (usually `http://127.0.0.1:7860`).

4.  **Initialize the Chatbot:**
    * On the Gradio interface, click the "Initialize Chatbot" button. This step will load the PDF, chunk its content, and generate embeddings for all chunks. This process can take a few minutes have a coffe.
    * Wait for the "Initialization Status" to confirm that the chatbot is ready.

5.  **Start Asking Questions:**
    * Once initialized, you can type your questions about "The Hard Thing About Hard Things" into the chat box and press "Send" or Enter.
    * The chatbot will respond with answers and cite relevant excerpts along with their page numbers from the book.

## Features

* **RAG (Retrieval Augmented Generation):** Retrieves relevant passages from the book before generating an answer.
* **Local "Vector Store":** Stores embeddings and performs similarity search locally without an external vector database.
* **Contextual Answers:** Answers are grounded in the book's content.
* **References:** Each answer includes specific excerpts and page numbers as references.
* **Multi-Turn Conversation:** Supports follow-up questions within the same chat session.
* **Gradio UI:** User-friendly web interface.