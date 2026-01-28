
import httpx
import asyncio
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings

# BGE-Large-EN-v1.5 API URL on HF
HF_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"

# Semaphore to limit concurrent requests (HF free tier rate limiting)
EMBEDDING_SEMAPHORE = asyncio.Semaphore(5)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_embedding(
    text: str,
    client: Optional[httpx.AsyncClient] = None
) -> List[float]:
    """
    Generates embedding for a single text using HF Inference API.
    Uses semaphore to limit concurrent requests.
    
    Args:
        text: The text to embed
        client: Optional persistent HTTP client. If None, creates a new one.
    """
    headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}
    payload = {"inputs": text}
    
    async with EMBEDDING_SEMAPHORE:
        if client:
            response = await client.post(HF_API_URL, headers=headers, json=payload, timeout=30.0)
        else:
            # Fallback for cases where client isn't available
            async with httpx.AsyncClient() as fallback_client:
                response = await fallback_client.post(HF_API_URL, headers=headers, json=payload, timeout=30.0)
        
        if response.status_code != 200:
            raise Exception(f"HF API Error: {response.status_code} - {response.text}")
            
        result = response.json()
        
        # HF Inference API for feature-extraction usually returns a list of floats (if 1 input)
        # or a list of list of floats.
        if isinstance(result, list):
             # For a single string input, it might return just the list of floats
             # OR a list containing the list of floats.
             if len(result) > 0 and isinstance(result[0], float):
                 return result
             if len(result) > 0 and isinstance(result[0], list):
                 return result[0]
                 
        raise Exception(f"Unexpected response format: {result}")


async def generate_embeddings_batch(
    texts: List[str],
    client: Optional[httpx.AsyncClient] = None
) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts concurrently.
    Uses asyncio.gather with semaphore-controlled concurrency for ~10x speedup.
    
    Args:
        texts: List of texts to embed
        client: Optional persistent HTTP client
    """
    if not texts:
        return []
    
    # Create concurrent tasks with semaphore control
    tasks = [generate_embedding(text, client) for text in texts]
    embeddings = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions - retry failed ones or return None
    result = []
    for i, emb in enumerate(embeddings):
        if isinstance(emb, Exception):
            # Log and append None for failed embeddings
            import logging
            logging.getLogger(__name__).error(f"Embedding failed for text {i}: {emb}")
            result.append(None)
        else:
            result.append(emb)
    
    return result

