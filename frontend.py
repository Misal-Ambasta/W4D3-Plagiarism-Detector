import streamlit as st
import requests
import numpy as np

API_URL = "http://localhost:8000"  # Adjust if backend runs elsewhere

st.set_page_config(page_title="Plagiarism Detector", layout="wide")
st.title("Plagiarism Detector - Semantic Similarity Analyzer")

st.markdown("""
Enter multiple texts below. Similarity matrix and clone detection will update in real time.
""")

# --- Dynamic input boxes ---
if "texts" not in st.session_state:
    st.session_state.texts = [""]

st.write("### Input Texts")

reset_col, add_col = st.columns([1, 7])
with reset_col:
    if st.button("Reset All"):
        st.session_state.texts = [""]
        st.session_state.names = ["Text 1"]
        st.session_state.compare_clicked = False
        st.rerun()
with add_col:
    if st.button("Add Text"):
        st.session_state.texts.append("")
        st.session_state.names.append(f"Text {len(st.session_state.texts)}")

if "names" not in st.session_state or len(st.session_state.names) != len(st.session_state.texts):
    st.session_state.names = [f"Text {i+1}" for i in range(len(st.session_state.texts))]

delete_idx = None
for i, text in enumerate(st.session_state.texts):
    cols = st.columns([3, 7, 1])
    with cols[0]:
        st.session_state.names[i] = st.text_input(f"Name for Text {i+1}", st.session_state.names[i], key=f"name_{i}")
    with cols[1]:
        st.session_state.texts[i] = st.text_area(f"Text {i+1}", text, key=f"text_{i}")
    with cols[2]:
        if i >= 2:
            if st.button("🗑️", key=f"delete_{i}"):
                delete_idx = i
if delete_idx is not None and len(st.session_state.texts) > 1:
    st.session_state.texts.pop(delete_idx)
    st.session_state.names.pop(delete_idx)
    st.rerun()

# Remove empty or duplicate texts and names
texts = [t for t in st.session_state.texts if t.strip()]
names = [n if n.strip() else f"Text {i+1}" for i, n in enumerate(st.session_state.names) if st.session_state.texts[i].strip()]

# --- Embedding Model Selection ---
st.write("### Embedding Models")
model_col1, model_col2, compare_col = st.columns([1, 1, 2])
with model_col1:
    use_minilm = st.checkbox("MiniLM", value=True)
with model_col2:
    use_openai = st.checkbox("OpenAI", value=False)
with compare_col:
    if 'compare_clicked' not in st.session_state:
        st.session_state.compare_clicked = False
    if st.button('Compare'):
        st.session_state.compare_clicked = True

selected_models = []
if use_minilm:
    selected_models.append("MiniLM")
if use_openai:
    selected_models.append("OpenAI")

if not selected_models:
    st.warning("Please select at least one embedding model.")

# --- Threshold Slider ---
threshold = st.slider("Clone Detection Threshold (%)", min_value=50, max_value=100, value=80, step=1)

# --- Model Comparison & Report ---
with st.expander("Model Comparison Report", expanded=False):
    st.markdown("""
    **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`):
    - Runs locally, fast, free, private
    
    **OpenAI** (`text-embedding-3-small`):
    - Cloud-based, may be slower, costs may apply, potentially higher accuracy
    
    For a detailed report, see [model_report.md](./model_report.md).
    """)

# --- Comparison Button ---
sim_matrix = None
sim_matrices = {}
error = None

if st.session_state.compare_clicked and len(texts) >= 2 and selected_models:
    sim_matrices = {}
    error = None
    for model in selected_models:
        try:
            payload = {"texts": texts, "model": model}
            resp = requests.post(f"{API_URL}/similarity", json=payload, timeout=30)
            resp.raise_for_status()
            sim_matrices[model] = np.array(resp.json()["similarity_matrix"]) * 100  # Convert to %
        except Exception as e:
            error = str(e)
    sim_matrix = None
    if sim_matrices:
        sim_matrix = list(sim_matrices.values())[0]

if error:
    st.error(f"Error: {error}")
elif sim_matrix is not None:
    for model, matrix in sim_matrices.items():
        st.write(f"### Similarity Matrix (%) for {model}")
        # Custom color rendering
        def color_sim(val):
            if val >= threshold:
                # Red for high similarity
                return f"background-color: rgba(255,0,0,{min((val-threshold)/20,1):.2f}); color: white;"
            else:
                # Green for low similarity
                return f"background-color: rgba(0,200,0,{min((100-val)/100,0.7):.2f}); color: white;"
        import pandas as pd
        df = pd.DataFrame(matrix, columns=names, index=names)
        styled = df.style.applymap(color_sim).format("{:.1f}")
        st.dataframe(styled, height=50+40*len(texts))

        # --- Highlight clones ---
        clones = set()
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if matrix[i][j] >= threshold:
                    clones.add(i)
                    clones.add(j)
        if clones:
            st.warning(f"Potential clones detected by {model}: {', '.join(names[i] for i in sorted(clones))}")
        else:
            st.success(f"No clones detected above threshold by {model}.")

        # --- Pairwise Similarity Summary ---
        st.write(f"#### Pairwise Similarity Summary for {model}")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                st.info(f"{model} calculated similarity between '{names[i]}' and '{names[j]}' is {matrix[i][j]:.1f}%")
else:
    st.info("Enter at least two texts to compute similarity.")
