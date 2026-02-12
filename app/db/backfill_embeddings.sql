-- RPC Function: Server-side embedding backfill
-- Copies embeddings from embedding_cache → prompts using content hash matching.
-- Vectors never leave the database — avoids PostgREST bytea serialization issues.
--
-- Usage (from Python): supabase.rpc('backfill_embeddings_from_cache', {'target_user_id': user_id})
-- Returns: number of prompts updated

-- Requires pgcrypto for digest()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION backfill_embeddings_from_cache(target_user_id uuid)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  updated_count integer;
BEGIN
  UPDATE prompts p
  SET embedding = ec.embedding
  FROM embedding_cache ec
  WHERE p.user_id = target_user_id
    AND p.embedding IS NULL
    AND ec.text_hash = encode(digest(p.content, 'sha256'), 'hex')
    AND ec.model_version = 'BAAI/bge-large-en-v1.5'
  ;
  GET DIAGNOSTICS updated_count = ROW_COUNT;
  RETURN updated_count;
END;
$$;
