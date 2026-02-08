from dotenv import load_dotenv
import streamlit as st
import os
from rag_engine import PolicyRAG
import base64

load_dotenv()

st.set_page_config(
    page_title="Policy Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "current_pdf_path" not in st.session_state:
    st.session_state.current_pdf_path = None

if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False

if "selected_page" not in st.session_state:
    st.session_state.selected_page = 1

@st.cache_resource
def init_rag():
    return PolicyRAG()

rag = init_rag()

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload Policy PDF",
        type=["pdf"],
        help="Upload a policy document to start chatting"
    )
    
    if uploaded_file:
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = f"temp_uploads/{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        if not st.session_state.pdf_loaded:
            with st.spinner("Processing document..."):
                rag.load_pdf(file_path)
                st.session_state.pdf_loaded = True
                # Add welcome message
                if len(st.session_state.messages) == 0:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📘 I've loaded **{uploaded_file.name}**. Ask me anything about this policy!"
                    })
            st.success("✅ Document loaded!")
        else:
            st.info(f"📄 **{uploaded_file.name}** is loaded")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.pdf_loaded:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Chat cleared! Ask me anything about the policy."
            })
        st.rerun()
    
    st.divider()

# Main chat interface
st.title("💬 Policy Chat Assistant")
st.caption("Your AI-powered policy document chatbot")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if st.session_state.pdf_loaded:
    if prompt := st.chat_input("Ask me anything about the policy..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.ask(prompt)
                answer_text = response['answer']
                
                # Display the answer
                st.markdown(answer_text)
                
                # Display relevant pages if available
                if response.get('relevant_pages'):
                    pages = response['relevant_pages']
                    st.caption(f"📄 Referenced pages: {', '.join(map(str, pages))}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer_text + (f"\n\n*Referenced pages: {', '.join(map(str, response['relevant_pages']))}" if response.get('relevant_pages') else "")
        })
        st.rerun()
else:
    st.info("👈 Please upload a policy document from the sidebar to start chatting!")
