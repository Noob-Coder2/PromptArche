-- Migration: Add model_version to embedding_cache for cache invalidation
-- Applied against actual table schema: text_hash (PK), embedding, created_at
-- 
-- This migration:
-- 1. Adds model_version column (auto-invalidates when embedding model changes)
-- 2. Drops old text_hash-only PK, replaces with composite (text_hash, model_version)
-- 3. Backfills existing rows with current model name

-- Add model_version column with default matching current model
ALTER TABLE embedding_cache 
ADD COLUMN model_version text NOT NULL DEFAULT 'BAAI/bge-large-en-v1.5';

-- Drop old PK
ALTER TABLE embedding_cache DROP CONSTRAINT embedding_cache_pkey;

-- Add new composite PK
ALTER TABLE embedding_cache ADD CONSTRAINT embedding_cache_pkey 
    PRIMARY KEY (text_hash, model_version);

-- Index for fast lookups by model version (useful if cleaning old versions)
CREATE INDEX IF NOT EXISTS idx_embedding_cache_model 
    ON embedding_cache(model_version);
