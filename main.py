from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal
import os

from embeddings import preprocess_texts, get_embeddings, cosine_similarity_matrix

app = FastAPI()

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Literal["MiniLM", "OpenAI"]

class SimilarityRequest(BaseModel):
    texts: List[str]
    model: Literal["MiniLM", "OpenAI"]

@app.get("/")
def read_root():
    return {"message": "Plagiarism Detector API is running."}

@app.post("/preprocess")
def preprocess_api(req: EmbeddingRequest):
    processed = preprocess_texts(req.texts)
    return {"processed": processed}

@app.post("/embeddings")
def embeddings_api(req: EmbeddingRequest):
    try:
        processed = preprocess_texts(req.texts)
        embeddings = get_embeddings(processed, req.model)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity")
def similarity_api(req: SimilarityRequest):
    try:
        processed = preprocess_texts(req.texts)
        embeddings = get_embeddings(processed, req.model)
        matrix = cosine_similarity_matrix(embeddings)
        return {"similarity_matrix": matrix}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
