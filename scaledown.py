import os
import requests
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("SCALEDOWN_API_KEY")

def compress(text: str) -> str:
    if not API_KEY:
        return text[: int(len(text) * 0.25)]

    try:
        response = requests.post(
            "https://api.scaledown.xyz/compress/raw/",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "text": text,
                "compression_ratio": 0.25
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("compressed_text", text)
    except (requests.exceptions.RequestException, Exception) as e:
        # Fallback to simple compression if API fails
        print(f"ScaleDown API error: {e}. Using fallback compression.")
        return text[: int(len(text) * 0.25)]
