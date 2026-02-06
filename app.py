from dotenv import load_dotenv
import streamlit as st
import os
from rag_engine import PolicyRAG

load_dotenv()

st.set_page_config(page_title="Policy Document Navigator")

st.title("📘 Policy Document Navigator")
st.write("Upload a policy PDF and ask questions in plain English.")

uploaded_file = st.file_uploader(
    "Upload Policy PDF",
    type=["pdf"]
)

@st.cache_resource
def init_rag():
    return PolicyRAG()

rag = init_rag()

if uploaded_file:
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = f"temp_uploads/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing policy document..."):
        rag.load_pdf(file_path)

    st.success("Policy loaded successfully!")

    question = st.text_area("Ask a question")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Analyzing policy..."):
                answer = rag.ask(question)
            st.markdown("### ✅ Answer")
            st.write(answer)
        else:
            st.warning("Please enter a question.")
