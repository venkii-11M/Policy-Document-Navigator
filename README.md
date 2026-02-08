# � Policy Chat Assistant

An AI-powered chatbot that makes understanding complex policy documents effortless. Built for the GenAI For GenZ Challenge, this tool uses Retrieval-Augmented Generation (RAG) to help users extract information from policy PDFs through natural, conversational interactions.

## 🌟 Features

- **💬 Chat Interface**: Interactive chat experience with conversation history
- **📄 PDF Upload**: Simply upload any policy document in PDF format via sidebar
- **🗨️ Natural Conversations**: Ask questions in plain English - no legal jargon needed
- **✅ Accurate Answers**: Get precise answers extracted directly from your policy document
- **⚡ Fast Semantic Search**: Powered by FAISS vector similarity search for lightning-fast retrieval
- **🗜️ Smart Compression**: Automatically compresses lengthy documents while preserving key information
- **🎨 Modern UI**: Clean chat interface with message bubbles and conversation flow
- **🗑️ Clear Chat**: Reset conversation anytime while keeping your document loaded

## 🛠️ Tech Stack

- **Python 3.8+**
- **Google Gemini 2.5 Flash**: Advanced AI model for intelligent response generation
- **FAISS**: Facebook AI Similarity Search for efficient vector retrieval
- **Sentence Transformers**: Semantic embeddings using all-MiniLM-L6-v2
- **Streamlit**: Interactive web application framework
- **PyPDF**: PDF text extraction
- **LangChain**: Text splitting utilities

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd policy-navigator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   To get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key
   - Copy it to your `.env` file

## 💻 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload a policy document**
   - Use the sidebar on the left to upload a PDF file
   - Wait for the document to be processed
   - You'll see a welcome message when ready

3. **Start chatting**
   - Type your question in the chat input at the bottom
   - Press Enter or click send
   - View the conversation history as you chat
   - Click "Clear Chat" to reset the conversation anytime

## 📖 Example Questions

- "What is the remote work policy?"
- "How many vacation days do employees get?"
- "What are the eligibility criteria for health benefits?"
- "What is the procedure for requesting time off?"
- "Are there any guidelines about confidentiality?"

## 🏗️ Project Structure

```
policy-navigator/
│
├── app.py                  # Streamlit chatbot interface
├── rag_engine.py           # RAG implementation with FAISS and Gemini
├── scaledown.py            # Text compression utility
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
├── temp_uploads/           # Temporary PDF storage
└── README.md              # Project documentation
```

## 🔧 How It Works

1. **Document Processing**:
   - PDF is uploaded and text is extracted
   - Text is compressed to optimize processing
   - Document is split into manageable chunks (500 chars with 50 char overlap)

2. **Vector Indexing**:
   - Each text chunk is converted to embeddings using Sentence Transformers
   - Embeddings are stored in a FAISS index for fast similarity search

3. **Question Answering**:
   - User question is converted to an embedding
   - Top 10 most relevant text chunks are retrieved
   - Context and question are sent to Gemini AI
   - AI generates an accurate answer based only on the provided context

## �️ System Architecture

```
┌─────────────────────────┐
│   Streamlit UI          │
│   (User Interface)      │
│   - PDF Upload          │
│   - Question Input      │
│   - Answer Display      │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   PyPDF Reader          │
│   (Text Extraction)     │
│   - PDF parsing         │
│   - Page text extract   │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   ScaleDown API         │
│   (Compression Layer)   │
│   - 75% size reduction  │
│   - Fallback: truncate  │
│   (scaledown.py)        │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   RecursiveTextSplitter │
│   (Chunking Layer)      │
│   - chunk_size: 500     │
│   - overlap: 50         │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   SentenceTransformer   │
│   (Embedding Model)     │
│   - all-MiniLM-L6-v2    │
│   - 384-dim vectors     │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   FAISS Index           │
│   (Vector Database)     │
│   - IndexFlatL2         │
│   - L2 distance search  │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   RAG Engine            │
│   (Retrieval Logic)     │
│   - Question embedding  │
│   - Top-10 retrieval    │
│   - Context assembly    │
│   (rag_engine.py)       │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   Google Gemini 2.5     │
│   (LLM Generation)      │
│   - Context-based QA    │
│   - Policy grounding    │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   Streamlit UI          │
│   (Answer Display)      │
│   - Formatted response  │
└─────────────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Layer** | Streamlit | User interaction, file upload, Q&A interface |
| **Parser** | PyPDF | Extract text from policy PDF files |
| **Compression** | ScaleDown API | Reduce text size by 75% (cost optimization) |
| **Chunker** | LangChain Splitter | Split into 500-char chunks with 50-char overlap |
| **Embedder** | SentenceTransformer | Convert text to 384-dim vectors |
| **Vector DB** | FAISS | Store and search embeddings efficiently |
| **RAG Logic** | Custom (rag_engine) | Orchestrate retrieval + generation |
| **LLM** | Gemini 2.5 Flash | Generate grounded answers from context |

### Data Flow

**Indexing Pipeline (PDF → Vectors):**
```
PDF → PyPDF → Raw Text → ScaleDown → Compressed Text → 
Chunking → Chunks → Embedding → Vectors → FAISS
```

**Query Pipeline (Question → Answer):**
```
Question → Embedding → Vector → FAISS Search → Top-10 Chunks → 
Context + Question → Gemini → Answer → Display
```

## �🎯 Use Cases

- **Employee Handbook Navigation**: Quickly find HR policies and procedures
- **Student Policy Queries**: Understand university guidelines and regulations
- **Legal Document Review**: Extract key information from contracts and agreements
- **Compliance Checks**: Verify policy adherence and requirements
- **Onboarding Support**: Help new members understand organizational policies

## 📝 Configuration

The following parameters can be adjusted in `rag_engine.py`:

- **Chunk size**: Currently set to 500 characters
- **Chunk overlap**: Currently set to 50 characters
- **Top-k retrieval**: Currently retrieves 10 most relevant chunks
- **Embedding model**: Currently using "all-MiniLM-L6-v2"
- **LLM model**: Currently using "gemini-2.5-flash"

## 📚 Documentation

* **Technical Documentation**:https://docs.google.com/document/d/1qNuQ6BL30BAtPhf0OWW5CabJsSnqQwb_IW4KAvZp7eQ/edit?usp=sharing



---
## 🔒 Privacy & Security

- Uploaded PDFs are stored temporarily in `temp_uploads/` directory
- Documents are processed locally
- Only question context is sent to Gemini API
- Consider adding cleanup logic for production use

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is created for the GenAI For GenZ Challenge.

## 👨‍💻 Author

Built with ❤️ by venkii for the GenAI For GenZ Challenge

## 🙏 Acknowledgments

- Google Gemini for powerful AI capabilities
- Streamlit for the amazing web framework
- FAISS for efficient similarity search
- The open-source community
