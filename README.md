# W4D3-Plagiarism-Detector

## User Guide

1. **Install all required dependencies using the following command:**

   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Backend**
   - Run the FastAPI backend: `uvicorn main:app --reload`
3. **Start the Frontend**
   - Run the Streamlit app: `streamlit run frontend.py`
4. **Usage**
   - Enter or paste multiple texts using the dynamic input boxes.
   - Select the embedding model (MiniLM or OpenAI) from the dropdown.
   - Adjust the clone detection threshold slider as needed.
   - View the real-time similarity matrix and color-coded clone detection results.
   - Expand the "Model Comparison Report" for details on embedding models.

## How Embeddings are Used for Plagiarism Detection

- **Text Embeddings**: Each input text is converted to a high-dimensional vector using the selected embedding model.
- **Semantic Similarity**: Cosine similarity is computed between all pairs of embeddings, producing a similarity matrix.
- **Clone Detection**: Text pairs with similarity above the user-defined threshold are flagged as potential clones (possible plagiarism).
- **Supported Models**:
  - **MiniLM**: Fast, open-source, runs locally.
  - **OpenAI**: Cloud-based, may provide higher accuracy, requires API key.

## API Documentation

### POST `/preprocess`
- **Input:** `{ "texts": [list of str], "model": "MiniLM" | "OpenAI" }`
- **Output:** `{ "processed": [list of str] }`

### POST `/embeddings`
- **Input:** `{ "texts": [list of str], "model": "MiniLM" | "OpenAI" }`
- **Output:** `{ "embeddings": [[float, ...], ...] }`

### POST `/similarity`
- **Input:** `{ "texts": [list of str], "model": "MiniLM" | "OpenAI" }`
- **Output:** `{ "similarity_matrix": [[float, ...], ...] }`

---

For more details, see `model_report.md` or use the app's built-in help and comparison features.