# Embedding Model Comparison Report

## Overview
This report compares the performance and characteristics of two embedding models used in the Plagiarism Detector:
- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`)
- **OpenAI** (`text-embedding-3-small`)

## Comparison Criteria
- **Speed**: MiniLM runs locally and is generally faster for small to medium input sizes. OpenAI requires API calls, which may introduce network latency.
- **Accuracy**: Both models provide high-quality semantic embeddings, but OpenAI's model may capture more nuanced relationships for certain text types.
- **Cost**: MiniLM is free and open-source. OpenAI model usage may incur API costs.
- **Privacy**: MiniLM processes data locally. OpenAI sends data to external servers.

## Usage in App
- Users can select either model in the frontend.
- Similarity matrices and clone detection are computed for both models.

## Recommendations
- Use **MiniLM** for fast, private, and cost-free analysis.
- Use **OpenAI** if you need potentially higher accuracy and are comfortable with API usage/costs.

---

*For further details, see the app documentation or experiment with both models in the UI.*
