# app.py
# Finance RAG Assistant: FastAPI + LangGraph + OpenAI + pgvector

import os
import uuid
import json
from typing import TypedDict, List, Optional

import psycopg2
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from openai import OpenAI
from langgraph.graph import StateGraph, END

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "finance_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Finance AI RAG Assistant")


# =========================
# DATABASE
# =========================

def get_db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS finance_documents (
        id UUID PRIMARY KEY,
        file_name TEXT,
        chunk_text TEXT,
        metadata JSONB,
        embedding vector(1536)
    );
    """)

    conn.commit()
    cur.close()
    conn.close()


init_db()


# =========================
# MODELS
# =========================

class QueryRequest(BaseModel):
    question: str


class FinanceState(TypedDict):
    question: str
    rewritten_question: Optional[str]
    context: Optional[str]
    answer: Optional[str]


# =========================
# PDF EXTRACTION
# =========================

def extract_pdf_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# =========================
# CLEANING
# =========================

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# =========================
# CHUNKING
# =========================

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# =========================
# EMBEDDINGS
# =========================

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


# =========================
# STORE IN PGVECTOR
# =========================

def store_chunks(file_name: str, chunks: List[str]):
    conn = get_db()
    cur = conn.cursor()

    for index, chunk in enumerate(chunks):
        emb = get_embedding(chunk)

        metadata = {
            "file_name": file_name,
            "chunk_index": index
        }

        cur.execute("""
        INSERT INTO finance_documents
        (id, file_name, chunk_text, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s)
        """, (
            str(uuid.uuid4()),
            file_name,
            chunk,
            json.dumps(metadata),
            emb
        ))

    conn.commit()
    cur.close()
    conn.close()


# =========================
# RETRIEVAL
# =========================

def retrieve_context(question: str, top_k: int = 5) -> str:
    query_embedding = get_embedding(question)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT chunk_text, metadata
    FROM finance_documents
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """, (query_embedding, top_k))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    context = ""
    for chunk, metadata in rows:
        context += f"\n\n{chunk}"

    return context


# =========================
# LANGGRAPH NODES
# =========================

def rewrite_question_node(state: FinanceState) -> FinanceState:
    question = state["question"]

    prompt = f"""
Rewrite the user question into a clear financial analysis query.

Question:
{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You rewrite finance questions clearly."},
            {"role": "user", "content": prompt}
        ]
    )

    state["rewritten_question"] = response.choices[0].message.content
    return state


def retrieve_node(state: FinanceState) -> FinanceState:
    question = state["rewritten_question"] or state["question"]
    context = retrieve_context(question)
    state["context"] = context
    return state


def answer_node(state: FinanceState) -> FinanceState:
    question = state["question"]
    context = state["context"]

    prompt = f"""
You are a professional Finance AI Assistant.

Answer the question using only the provided context.

If the answer is not available in the context, say:
"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful finance assistant. Be accurate and concise."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    state["answer"] = response.choices[0].message.content
    return state


# =========================
# LANGGRAPH WORKFLOW
# =========================

def build_graph():
    graph = StateGraph(FinanceState)

    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("rewrite_question")

    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


rag_graph = build_graph()


# =========================
# API ROUTES
# =========================

@app.get("/")
def home():
    return {
        "message": "Finance AI RAG Assistant is running"
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {
            "error": "Only PDF files are supported"
        }

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    raw_text = extract_pdf_text(file_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    store_chunks(file.filename, chunks)

    return {
        "message": "File uploaded and indexed successfully",
        "file_name": file.filename,
        "total_chunks": len(chunks)
    }


@app.post("/ask")
def ask_question(request: QueryRequest):
    initial_state: FinanceState = {
        "question": request.question,
        "rewritten_question": None,
        "context": None,
        "answer": None
    }

    result = rag_graph.invoke(initial_state)

    return {
        "question": request.question,
        "rewritten_question": result["rewritten_question"],
        "answer": result["answer"]
    }


# =========================
# RUN COMMAND
# =========================
# uvicorn app:app --reload