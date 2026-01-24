
import httpx
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings

# BGE-Large-EN-v1.5 API URL on HF
HF_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_embedding(text: str) -> List[float]:
    """
    Generates embedding for a single text using HF Inference API.
    Retries on failure.
    """
    headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}
    payload = {"inputs": text}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(HF_API_URL, headers=headers, json=payload, timeout=10.0)
        
        if response.status_code != 200:
            raise Exception(f"HF API Error: {response.status_code} - {response.text}")
            
        result = response.json()
        
        # HF Inference API for feature-extraction usually returns a list of floats (if 1 input)
        # or a list of list of floats.
        # Check format carefully.
        if isinstance(result, list):
             # For a single string input, it might return just the list of floats
             # OR a list containing the list of floats.
             if len(result) > 0 and isinstance(result[0], float):
                 return result
             if len(result) > 0 and isinstance(result[0], list):
                 return result[0]
                 
        raise Exception(f"Unexpected response format: {result}")

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts.
    Ideally, we should rely on the API's batch capability, but free tier can be flaky with large batches.
    Simple implementation: iterate. 
    Optimization: Use gather for concurrent requests (with semaphore to avoid rate limit).
    """
    # For simplicity and robustness on free tier, let's do sequential or small chunks.
    # We'll rely on the caller to batch reasonably.
    embeddings = []
    for text in texts:
        emb = await generate_embedding(text)
        embeddings.append(emb)
    return embeddings
