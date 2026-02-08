import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()


API_KEY = os.getenv("SCALEDOWN_API_KEY")

def compress(text: str) -> str:
    if not API_KEY:
        return text[: int(len(text) * 0.5)]

    try:
        response = requests.post(
            "https://api.scaledown.xyz/compress/raw/",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "text": text,
                "compression_ratio": 0.5
            },
            timeout=3
        )
        response.raise_for_status()
        return response.json().get("compressed_text", text)
    except (requests.exceptions.RequestException, Exception) as e:
        # Fallback to simple compression if API fails
        print(f"ScaleDown API error: {e}. Using fallback compression.")
        return text[: int(len(text) * 0.5)]

def compress_batch(texts: list[str], max_workers: int = 5) -> list[str]:
    """Compress multiple texts in parallel for faster processing."""
    if not texts:
        return []
    
    results = [None] * len(texts)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(compress, text): i for i, text in enumerate(texts)}
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Error compressing text {index}: {e}")
                results[index] = texts[index][: int(len(texts[index]) * 0.25)]
    
    return results
