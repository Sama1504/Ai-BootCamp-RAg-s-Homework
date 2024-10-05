import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from langchain_community.vectorstores import Chroma


load_dotenv()

# Configure the Google Gemini API
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
os.environ["GOOGLE_API_KEY"] = google_api_key

def load_document(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vector_store(chunks: List[str]) -> Chroma:
    embeddings = GooglePalmEmbeddings()
    return Chroma.from_texts(chunks, embeddings)

def setup_rag_pipeline(vector_store: Chroma) -> RetrievalQA:
    llm = GooglePalm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

def generate_answer(qa_chain: RetrievalQA, question: str) -> Dict[str, str]:
    result = qa_chain({"query": question})
    return {
        "question": question,
        "answer": result["result"],
        "contexts": [doc.page_content for doc in result["source_documents"]]
    }

def implement_rag():
    # Load and process the document
    doc_path = "docs/intro-to-llms-karpathy.txt"
    document = load_document(doc_path)
    chunks = split_text(document)
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    
    # Set up RAG pipeline
    qa_chain = setup_rag_pipeline(vector_store)
    
    return qa_chain

def generate_answers(questions: List[Dict[str, str]], qa_chain: RetrievalQA) -> List[Dict[str, str]]:
    return [generate_answer(qa_chain, q["question"]) for q in questions]

if __name__ == "__main__":
    # Implement RAG
    qa_chain = implement_rag()
    
    # Load questions from questions.json
    with open('questions.json', 'r') as f:
        questions = json.load(f)
    
    # Generate answers
    answers = generate_answers(questions, qa_chain)
    
    # Save answers to a file
    with open('answers.json', 'w') as f:
        json.dump(answers, f, indent=2)

    print("Answers generated and saved to answers.json")