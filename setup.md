# What we're building

Building Lab Archives in Your ELN: Essential Steps | Labguru

Step 1 — Upload documents

Upload PDF or TXT files through Streamlit.

🧠 Vector Database & Semantic Search

Step 2 — Index the documents

Split text into chunks, create embeddings, and store them in FAISS.

Building a Local RAG Chatbot: Documents, Embeddings, and Retrieval | local-llm.net

Step 3 — Retrieve and answer

LangGraph retrieves relevant chunks and sends them with the question to the hosted model.

GitHub - gautamnaik1994/GenAI_DocumentQA_Bot: AI-Powered Document Q&A is an AI-powered document question-answering application built with Streamlit, LangChain, HuggingFace, and FAISS. It allows users to upload multiple PDF documents and interactively ask questions, receiving concise answers along with relevant source citations.

Step 4 — Display grounded answers

Show the answer and its source-document references.

# Technology choices

Component

Technology

Purpose

UI

Streamlit

Upload files and chat

Workflow

LangGraph

Connect retrieval and generation

LLM

Hugging Face Inference Providers

Generate answers

Embeddings

Sentence Transformers

Convert text into vectors

Vector database

FAISS

Find semantically relevant chunks

Document parsing

PyPDF

Read PDF files

This design uses the hosted API for text generation. Embeddings run locally, so you don't need a separate paid embedding API.

# Scaffold Project

mkdir huggingface-langgraph-rag
cd huggingface-langgraph-rag

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Run app

streamlit run app.py
