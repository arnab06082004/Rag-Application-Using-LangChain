# 📄 PDF Q&A Chatbot (RAG) — LangChain + Streamlit + Chroma + Groq

A professional **Retrieval-Augmented Generation (RAG)** based PDF Question Answering chatbot built using **LangChain** and **Streamlit**.  
Users can upload a PDF document and ask questions — the system retrieves the most relevant PDF context and generates accurate answers using an LLM.

---

## 📌 Overview

This application implements a complete RAG pipeline:

1. Upload a PDF  
2. Extract text from the document  
3. Split text into semantic chunks  
4. Convert chunks into vector embeddings  
5. Store embeddings in Chroma Vector Database  
6. Retrieve relevant chunks based on user query  
7. Generate an answer using Groq LLM (Llama 3.3 70B)  
8. Maintain chat history for follow-up questions  

---

## 🚀 Key Features

- PDF upload and processing using Streamlit
- Text extraction using `pypdf`
- Chunking using `RecursiveCharacterTextSplitter`
- Embeddings using HuggingFace: `all-MiniLM-L6-v2`
- Vector storage and retrieval using ChromaDB
- LLM response generation using Groq: `llama-3.3-70b-versatile`
- Conversation memory using `RunnableWithMessageHistory`
- Answers are generated strictly from PDF context

---

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API
- PyPDF
- python-dotenv

---

## 📂 Project Structure


RAG_Application_langchain/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Project dependencies
├── .gitignore           # Git ignore rules
├── .env.example         # Example environment variables
└── README.md            # Documentation

⚙️ Clone the Repository
git clone https://github.com/arnab06082004/Rag-Application-Using-LangChain
cd RRag-Application-Using-LangChain
