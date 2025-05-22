import pypdf


def extract_text_from_pdf(pdf_path):
    """
    Extracts text and page numbers from a PDF.
    Returns a list of dictionaries: [{'text': '...', 'page_number': N}, ...]
    """
    documents = []
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append({'text': text, 'page_number': i + 1})
    return documents

def chunk_text(documents, chunk_size=2000, overlap=200):
    """
    Splits documents into smaller chunks with overlap and retains page numbers.
    """
    chunks = []
    for doc in documents:
        text = doc['text']
        page_number = doc['page_number']
        
        # Simple character-based chunking
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip(): # Ensure chunk is not empty
                chunks.append({'text': chunk, 'page_number': page_number})
    return chunks

if __name__ == "__main__":
    pdf_path = "the_hard_thing_about_hard_things.pdf" # Make sure this PDF is in your project directory
    raw_documents = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_documents)} pages.")
    
    processed_chunks = chunk_text(raw_documents, chunk_size=500, overlap=100)
    print(f"Generated {len(processed_chunks)} chunks.")
    # print(processed_chunks[0])