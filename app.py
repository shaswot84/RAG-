
import streamlit as st
from utils.embedding import generate_embeddings
from utils.chunking import chunk_text
from utils.retrieval import load_faiss_index, retrieve_chunks
from utils.prompt import build_prompt
from utils.completion import generate_completion

st.title("Carlos Alzaraz Life Story")
st.write("Ask questions grounded in the life  of Carlos Alcaraz..")

query = st.text_input("Enter your question here")

if query:
    index, chunk_mapping = load_faiss_index()
    top_chunks = retrieve_chunks(query, index, chunk_mapping)
    prompt = build_prompt(top_chunks, query)
    response = generate_completion(prompt)
    
    st.subheader("Answer")
    st.write(response)

    with st.expander("Retrieved Chunks"):
        for chunk in top_chunks:
            st.markdown(f"- {chunk}")
