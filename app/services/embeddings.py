
import asyncio
import numpy as np
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import InferenceClient
from app.core.config import settings

# Model configuration
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Semaphore to limit concurrent requests (HF free tier rate limiting)
EMBEDDING_SEMAPHORE = asyncio.Semaphore(5)

# Global client instance (reused across requests)
_hf_client: Optional[InferenceClient] = None


def get_hf_client() -> InferenceClient:
    """Get or create the global HuggingFace InferenceClient instance."""
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(token=settings.HF_TOKEN)
    return _hf_client


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_embedding(
    text: str,
    client: Optional[InferenceClient] = None
) -> List[float]:
    """
    Generates embedding for a single text using HuggingFace Hub InferenceClient.
    Uses semaphore to limit concurrent requests.
    
    Args:
        text: The text to embed
        client: Optional InferenceClient. If None, uses global client.
    
    Returns:
        List of floats representing the embedding vector (1024 dimensions for BGE-Large)
    """
    if client is None:
        client = get_hf_client()
    
    async with EMBEDDING_SEMAPHORE:
        # Run the synchronous feature_extraction in a thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: client.feature_extraction(text, model=MODEL_NAME)
        )
        
        # Convert numpy array to list if needed
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif isinstance(embedding, list):
            # Handle nested list structure
            if len(embedding) > 0 and isinstance(embedding[0], (list, np.ndarray)):
                # Batch response for single item
                first_item = embedding[0]
                if isinstance(first_item, np.ndarray):
                    return first_item.tolist()
                return first_item
            return embedding
        
        raise Exception(f"Unexpected embedding format: {type(embedding)}")


async def generate_embeddings_batch(
    texts: List[str],
    client: Optional[InferenceClient] = None
) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts concurrently.
    Uses asyncio.gather with semaphore-controlled concurrency for ~10x speedup.
    
    Args:
        texts: List of texts to embed
        client: Optional InferenceClient. If None, uses global client.
    
    Returns:
        List of embedding vectors (None for failed items)
    """
    if not texts:
        return []
    
    if client is None:
        client = get_hf_client()
    
    # Create concurrent tasks with semaphore control
    tasks = [generate_embedding(text, client) for text in texts]
    embeddings = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions - log and return None for failed embeddings
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

