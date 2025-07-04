from typing import List, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load MiniLM model once
minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Set OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "text-embedding-3-small"


def preprocess_texts(texts: List[str]) -> List[str]:
    """Basic preprocessing: strip and remove empty texts."""
    return [t.strip() for t in texts if t.strip()]


def get_minilm_embeddings(texts: List[str]) -> List[List[float]]:
    return minilm_model.encode(texts, convert_to_numpy=True).tolist()


def get_openai_embeddings(texts: List[str]) -> List[List[float]]:
    # Assumes OPENAI_API_KEY is set in the environment
    response = openai.embeddings.create(
        model=OPENAI_MODEL,
        input=texts
    )
    return [d.embedding for d in response.data]


def cosine_similarity_matrix(embeddings: List[List[float]]) -> List[List[float]]:
    arr = np.array(embeddings)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    sim_matrix = np.dot(arr, arr.T) / (norm * norm.T + 1e-8)
    return sim_matrix.tolist()


def get_embeddings(texts: List[str], model: Literal["MiniLM", "OpenAI"]) -> List[List[float]]:
    if model == "MiniLM":
        return get_minilm_embeddings(texts)
    elif model == "OpenAI":
        return get_openai_embeddings(texts)
    else:
        raise ValueError(f"Unknown model: {model}")
