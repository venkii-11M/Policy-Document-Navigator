import os
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from scaledown import compress

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class PolicyRAG:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []
        self.chunk_metadata = []  # Store page numbers for each chunk
        self.pdf_path = None
        self.page_texts = []  # Store original page texts

    def load_pdf(self, pdf_path: str):
        self.pdf_path = pdf_path
        reader = PdfReader(pdf_path)
        
        # Store original page texts and track page numbers
        self.page_texts = []
        page_chunks = []
        
        # Extract all page texts first
        all_page_texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            self.page_texts.append(page_text)
            all_page_texts.append(page_text)
        
        # Compress all pages in parallel
        from scaledown import compress_batch
        compressed_pages = compress_batch(all_page_texts)
        
        # Process compressed pages into chunks
        for page_num, (compressed_page, original_text) in enumerate(zip(compressed_pages, all_page_texts), start=1):
            # Split into chunks while preserving page number
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_text(compressed_page)
            
            for chunk in chunks:
                page_chunks.append({
                    'text': chunk,
                    'page': page_num,
                    'original_text': original_text[:200]  # Store snippet of original text
                })
        
        # Extract chunks and metadata
        self.text_chunks = [item['text'] for item in page_chunks]
        self.chunk_metadata = [{'page': item['page'], 'original': item['original_text']} 
                               for item in page_chunks]
        
        # Create embeddings and index
        vectors = self.embedder.encode(self.text_chunks)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.array(vectors))

    def ask(self, question: str) -> dict:
        """
        Ask a question and get answer with citations.
        Returns: dict with 'answer', 'citations', and 'relevant_pages'
        """
        q_vector = self.embedder.encode([question])
        _, indices = self.index.search(q_vector, k=15)

        # Collect context with metadata
        context_parts = []
        cited_pages = set()
        citations = []
        
        for idx in indices[0]:
            chunk_text = self.text_chunks[idx]
            page_num = self.chunk_metadata[idx]['page']
            cited_pages.add(page_num)
            
            context_parts.append(f"[Page {page_num}]\n{chunk_text}")
            citations.append({
                'page': page_num,
                'text': chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text
            })
        
        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the question using ONLY the policy text below.
If the policy does not explicitly mention it, say so clearly.

IMPORTANT: 
- Do NOT include page numbers or page references like "(On page X)" in your answer
- Keep answers SHORT and IMPACTFUL (3-5 bullet points max)
- Use clear, direct language
- Focus on key information only
- Start with the main point, then add essential details

POLICY TEXT:
{context}

QUESTION:
{question}
"""

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        
        return {
            'answer': response.text,
            'citations': citations,
            'relevant_pages': sorted(list(cited_pages))
        }
