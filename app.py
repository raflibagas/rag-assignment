import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

# Initialize Groq
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    # api_key="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",
    model_name="llama-3.1-8b-instant"
)

def create_rag_system():
    try:
        # 1. Load Documents
        pdf_loader = PyPDFLoader("smarthome_hub_documentation_Final.pdf")
        pdf_docs = pdf_loader.load()
        
        # 2. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(pdf_docs)
        
        # 3. Create Embeddings and Vector Store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
    except Exception as e:
        print(f"Error creating RAG system: {e}")
        return None

# Initialize the RAG system
vectorstore = create_rag_system()

def respond(message, history):
    try:
        if vectorstore is None:
            return "Sorry, the system is currently unavailable. Please try again later."
            
        # Create QA chain for each query
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            )
        )
        
        # Get response

        rag_prompt = f"""Jawablah pertanyaan berikut dalam Bahasa Indonesia: {message}"""
        response = qa_chain.run(rag_prompt)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="SmartHome Hub X1000 Assistant",
    description="Ask me anything about SmartHome Hub X1000!",
    examples=[
        "Apa saja fitur utama dari SmartHome Hub X1000?",
        "Bagaimana cara menginstall SmartHome Hub X1000?",
        "Jelaskan sistem keamanan SmartHome Hub X1000"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()