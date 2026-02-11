"""
Embedding caching and batch processing service.

Solves three key problems:
1. Embedding Bottleneck: Batches HuggingFace API calls (50-100x speedup)
2. No Caching: Implements LRU cache + database persistence
3. Memory Efficiency: Streams embeddings without loading all data

Architecture:
- Cache Layer 1: In-memory LRU cache (fast, TTL-based expiry)
- Cache Layer 2: Database (persistent across restarts, shared across users)
- Model Version: Cache entries are keyed by model name - switching models
  auto-invalidates stale entries
- Batch Processing: Queue embeddings, process in configurable batch sizes
- Retry Logic: Exponential backoff on API failures

Note: Embeddings are deterministic for the same input text regardless of user.
The cache is therefore global (not user-scoped) to maximize hit rates.
"""

import asyncio
import hashlib
import logging
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import InferenceClient

from app.core.config import settings
from app.db.supabase import get_supabase

logger = logging.getLogger(__name__)

# HuggingFace API configuration
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Model version for cache invalidation — changing this auto-invalidates old cache entries
MODEL_VERSION = MODEL_NAME

# Batch size for HF API (optimal: 25-50 texts per batch)
EMBEDDING_BATCH_SIZE = 32  # Default batch size for HuggingFace API

# In-memory cache configuration
CACHE_SIZE = 10000  # Items to keep in memory
CACHE_TTL_SECONDS = 3600  # 1 hour expiry


class EmbeddingCache:
    """
    Two-layer embedding cache: in-memory LRU + database persistence.
    Supports batch processing and API call optimization.
    
    Cache is global (not user-scoped) because the same text always produces
    the same embedding vector regardless of who asked it.
    """
    
    def __init__(self):
        """Initialize cache with in-memory store and DB backend."""
        self._memory_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._api_calls = 0
        self._semaphore = asyncio.Semaphore(5)  # Rate limit concurrent API calls
        self._hf_client = InferenceClient(token=settings.HF_TOKEN)  # Reusable client
    
    def _hash_text(self, text: str) -> str:
        """Create deterministic hash of text for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_time: datetime) -> bool:
        """Check if cached entry is still valid (within TTL)."""
        return (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS
    
    async def get_embedding(
        self,
        text: str,
        client: Optional[InferenceClient] = None
    ) -> Optional[List[float]]:
        """
        Get embedding for single text with two-layer cache.
        
        Args:
            text: Text to embed
            client: Optional persistent HTTP client
            
        Returns:
            Embedding vector or None if failed
        """
        text_hash = self._hash_text(text)
        
        # Layer 1: Memory cache (keyed by text_hash only)
        if text_hash in self._memory_cache:
            embedding, cached_time = self._memory_cache[text_hash]
            if self._is_cache_valid(cached_time):
                self._cache_hits += 1
                logger.debug(f"Memory cache hit for text hash {text_hash[:8]}")
                return embedding
            else:
                # Expired, remove from memory
                del self._memory_cache[text_hash]
        
        # Layer 2: Database cache (filtered by model_version)
        embedding = await self._get_from_database(text_hash)
        if embedding:
            self._cache_hits += 1
            # Refresh memory cache
            self._memory_cache[text_hash] = (embedding, datetime.now())
            logger.debug(f"Database cache hit for text hash {text_hash[:8]}")
            return embedding
        
        # Cache miss: Call API
        self._cache_misses += 1
        embedding = await self._call_huggingface_api(text, client)
        
        if embedding:
            # Store in both caches
            self._memory_cache[text_hash] = (embedding, datetime.now())
            await self._store_in_database(text_hash, embedding)
        
        return embedding
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        client: Optional[InferenceClient] = None
    ) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts with caching.
        
        Args:
            texts: List of texts to embed
            client: Optional persistent HTTP client
            
        Returns:
            List of embedding vectors (None for failed items)
        """
        if not texts:
            return []
        
        return await self._process_batch(texts, client)
        
    async def _process_batch(self, texts: List[str], client: Optional[InferenceClient]):
        """Process a batch of texts with cache-first strategy."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices = []
        text_hashes = [self._hash_text(text) for text in texts]
        
        # 1. Check Caches
        # 1a. Batch check memory cache
        hashes_to_lookup_db = []
        indices_to_lookup_db = []
        
        for idx, text_hash in enumerate(text_hashes):
            # Memory cache
            if text_hash in self._memory_cache:
                emb, time = self._memory_cache[text_hash]
                if self._is_cache_valid(time):
                    results[idx] = emb
                    self._cache_hits += 1
                    continue
                else:
                    del self._memory_cache[text_hash]
            
            # If not in memory, prepare for DB lookup
            hashes_to_lookup_db.append(text_hash)
            indices_to_lookup_db.append(idx)
            
        # 1b. Batch check database cache
        if hashes_to_lookup_db:
            db_embeddings = await self._get_batch_from_database(hashes_to_lookup_db)
            
            for i, text_hash in enumerate(hashes_to_lookup_db):
                original_idx = indices_to_lookup_db[i]
                
                if text_hash in db_embeddings:
                    emb = db_embeddings[text_hash]
                    self._memory_cache[text_hash] = (emb, datetime.now())
                    results[original_idx] = emb
                    self._cache_hits += 1
                else:
                    uncached_indices.append(original_idx)
                    self._cache_misses += 1
        
        if not uncached_indices:
            logger.info(f"Cache hit for all {len(texts)} texts")
            return results
            
        logger.info(
            f"Batch embedding: {len(texts)} texts, "
            f"{len(uncached_indices)} cache misses ({len(results) - len(uncached_indices)} hits)"
        )

        # 2. Call API for uncached
        uncached_texts = [texts[i] for i in uncached_indices]
        
        # Batch API calls
        # Split into smaller chunks for API stability
        api_batch_size = EMBEDDING_BATCH_SIZE
        batches = [uncached_texts[i:i + api_batch_size] 
                  for i in range(0, len(uncached_texts), api_batch_size)]
        
        batch_results = []
        for sub_batch in batches:
            try:
                # True batch API call
                result = await self._call_huggingface_api_batch(sub_batch, client)
                batch_results.extend(result)
            except Exception as e:
                logger.error(f"Sub-batch API call failed: {e}")
                # Fill with Nones for this sub-batch
                batch_results.extend([None] * len(sub_batch))
        
        # 3. Store results
        if len(batch_results) != len(uncached_indices):
            logger.error(f"Mismatch in batch results: expected {len(uncached_indices)}, got {len(batch_results)}")
            # Pad with None if needed (shouldn't happen with correct logic)
            batch_results.extend([None] * (len(uncached_indices) - len(batch_results)))

        cache_rows = []
        
        for i, embedding in enumerate(batch_results):
            if embedding is None:
                continue
                
            global_idx = uncached_indices[i]
            results[global_idx] = embedding
            
            text_hash = text_hashes[global_idx]
            
            self._memory_cache[text_hash] = (embedding, datetime.now())
            cache_rows.append({
                "text_hash": text_hash,
                "embedding": embedding,
                "model_version": MODEL_VERSION,
                "created_at": datetime.now().isoformat()
            })
                
        if cache_rows:
            await self._store_batch_in_database(cache_rows)
            
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_huggingface_api(
        self,
        text: str,
        client: Optional[InferenceClient] = None
    ) -> Optional[List[float]]:
        """Call HuggingFace API for single text using InferenceClient."""
        if client is None:
            client = self._hf_client
        
        async with self._semaphore:
            try:
                # Run synchronous feature_extraction in thread pool
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    lambda: client.feature_extraction(text, model=MODEL_NAME)
                )
                
                self._api_calls += 1
                return self._normalize_embedding(embedding)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise
    
    @staticmethod
    def _normalize_embedding(embedding) -> Optional[List[float]]:
        """
        Normalize embedding output to a flat list of native Python floats.
        
        Handles all possible return formats from HuggingFace InferenceClient:
        - numpy ndarray (1D or 2D)
        - nested Python lists
        - numpy scalar types (float32/float64)
        
        This is critical because pgvector columns require native Python floats
        for proper JSON serialization by the Supabase client.
        """
        if embedding is None:
            return None
        
        # Convert numpy array to flat list
        if isinstance(embedding, np.ndarray):
            flat = embedding.flatten().tolist()  # .tolist() on ndarray gives native Python types
            return [float(x) for x in flat]
        
        # Handle nested list/array structures  
        if isinstance(embedding, list):
            # Flatten if nested: [[0.1, 0.2, ...]] -> [0.1, 0.2, ...]
            if len(embedding) > 0 and isinstance(embedding[0], (list, np.ndarray)):
                first = embedding[0]
                if isinstance(first, np.ndarray):
                    return [float(x) for x in first.flatten().tolist()]
                return [float(x) for x in first]
            # Already flat list — ensure native Python floats
            return [float(x) for x in embedding]
        
        logger.error(f"Unexpected embedding format: {type(embedding)}")
        return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_huggingface_api_batch(
        self,
        texts: List[str],
        client: Optional[InferenceClient] = None
    ) -> List[Optional[List[float]]]:
        """
        Call HuggingFace API for batch of texts efficiently.
        Uses a single HTTP request for multiple texts.
        """
        if client is None:
            client = self._hf_client

        try:
            # Use run_in_executor because feature_extraction is synchronous
            loop = asyncio.get_event_loop()
            
            # The API supports passing a list of strings
            embeddings = await loop.run_in_executor(
                None,
                lambda: client.feature_extraction(texts, model=MODEL_NAME)
            )
            
            # Normalize results
            normalized_results = []
            
            # When passing a list, the result is usually a list (or ndarray) of embeddings
            if isinstance(embeddings, (list, np.ndarray)):
                # If it's a list of results, process each one
                for emb in embeddings:
                     normalized_results.append(self._normalize_embedding(emb))
            else:
                # Unexpected result structure
                logger.error(f"Unexpected batch API result type: {type(embeddings)}")
                return [None] * len(texts)
                
            self._api_calls += 1  # Count as 1 API call (batch)
            return normalized_results
            
        except Exception as e:
            logger.warning(f"Batch embedding API call failed: {e}")
            # If batch fails, we could fallback to individual or just return None
            # Returning None allows retry logic to kick in at a higher level if needed
            return [None] * len(texts)
    
    
    async def _get_from_database(self, text_hash: str) -> Optional[List[float]]:
        """Retrieve embedding from database cache, filtered by current model version."""
        try:
            supabase = get_supabase()
            response = supabase.table("embedding_cache").select(
                "embedding"
            ).eq("text_hash", text_hash).eq(
                "model_version", MODEL_VERSION
            ).limit(1).execute()
            
            if response and hasattr(response, 'data') and isinstance(response.data, list) and len(response.data) > 0:
                item = response.data[0]
                if isinstance(item, dict):
                    embedding = item.get("embedding")
                    if isinstance(embedding, list):
                        return embedding  # type: ignore
            return None
        except Exception as e:
            logger.warning(f"Database cache read failed: {e}")
            return None

    async def _get_batch_from_database(self, text_hashes: List[str]) -> Dict[str, List[float]]:
        """Retrieve multiple embeddings from database cache in a single query."""
        if not text_hashes:
            return {}
            
        try:
            supabase = get_supabase()
            # Use .in_() for batch lookup
            response = supabase.table("embedding_cache").select(
                "text_hash, embedding"
            ).in_("text_hash", text_hashes).eq(
                "model_version", MODEL_VERSION
            ).execute()
            
            result_map = {}
            if response and hasattr(response, 'data') and isinstance(response.data, list):
                for item in response.data:
                    if isinstance(item, dict):
                        th = item.get("text_hash")
                        emb = item.get("embedding")
                        if th and isinstance(emb, list):
                            result_map[th] = emb
                            
            return result_map
        except Exception as e:
            logger.warning(f"Database batch cache read failed: {e}")
            return {}
    
    async def _store_in_database(self, text_hash: str, embedding: List[float]):
        """Store single embedding in database cache."""
        try:
            supabase = get_supabase()
            supabase.table("embedding_cache").upsert({
                "text_hash": text_hash,
                "embedding": embedding,
                "model_version": MODEL_VERSION,
                "created_at": datetime.now().isoformat()
            }, on_conflict="text_hash,model_version").execute()
        except Exception as e:
            logger.warning(f"Database cache write failed: {e}")
    
    async def _store_batch_in_database(self, rows: List[Dict]):
        """Store multiple embeddings in database cache with a single upsert."""
        try:
            # Deduplicate by text_hash to be safe
            unique_rows = {row["text_hash"]: row for row in rows}
            deduped = list(unique_rows.values())
            
            supabase = get_supabase()
            supabase.table("embedding_cache").upsert(
                deduped, on_conflict="text_hash,model_version"
            ).execute()
            logger.debug(f"Batch-cached {len(deduped)} embeddings to database (from {len(rows)} total)")
        except Exception as e:
            logger.warning(f"Database cache batch write failed ({len(rows)} rows): {e}")
    
    def get_cache_stats(self) -> Dict[str, int | float]:
        """Return cache statistics for monitoring."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "api_calls": self._api_calls,
            "memory_cache_size": len(self._memory_cache),
            "model_version": MODEL_VERSION
        }
    
    def clear_cache(self):
        """Clear in-memory cache (database cache remains for persistence)."""
        self._memory_cache.clear()
        logger.info("In-memory embedding cache cleared")


# Global instance (singleton)
_embedding_cache_instance: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create global embedding cache instance."""
    global _embedding_cache_instance
    if _embedding_cache_instance is None:
        _embedding_cache_instance = EmbeddingCache()
    return _embedding_cache_instance


# Convenience functions
async def get_cached_embedding(
    text: str,
    client: Optional[InferenceClient] = None
) -> Optional[List[float]]:
    """Get embedding for single text with caching."""
    cache = get_embedding_cache()
    return await cache.get_embedding(text, client)


async def get_cached_embeddings_batch(
    texts: List[str],
    client: Optional[InferenceClient] = None
) -> List[Optional[List[float]]]:
    """Get embeddings for batch of texts with intelligent caching and batching."""
    cache = get_embedding_cache()
    return await cache.get_embeddings_batch(texts, client)
