import os
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scaledown import compress

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class PolicyRAG:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []

    def load_pdf(self, pdf_path: str):
        reader = PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() for page in reader.pages)

        compressed = compress(full_text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.text_chunks = splitter.split_text(compressed)

        vectors = self.embedder.encode(self.text_chunks)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.array(vectors))

    def ask(self, question: str) -> str:
        q_vector = self.embedder.encode([question])
        _, indices = self.index.search(q_vector, k=10)

        context = "\n\n".join(
            self.text_chunks[i] for i in indices[0]
        )

        prompt = f"""
Answer the question using ONLY the policy text below.
If the policy does not explicitly mention it, say so clearly.
You may briefly explain implied alignment, if any, without assuming facts.

POLICY TEXT:
{context}

QUESTION:
{question}
"""

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.text
