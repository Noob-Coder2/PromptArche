-- Migration: Allow duplicate prompts (Event-Based Deduplication)
-- Purpose: Switch unique constraint from content-based to event-based ID
-- This allows storing the same prompt text multiple times (preserving history)
-- while preventing re-ingestion of the same export file.

-- Step 1: Add external_id column to store provider's message ID
ALTER TABLE prompts 
ADD COLUMN IF NOT EXISTS external_id text;

-- Step 2: Drop existing unique constraints
-- We need to drop potential constraints from previous migrations
ALTER TABLE prompts DROP CONSTRAINT IF EXISTS unique_prompt_per_user;
ALTER TABLE prompts DROP CONSTRAINT IF EXISTS unique_prompt_hash_per_user;

-- Step 3: Create new unique constraint on (user_id, source, external_id)
-- Note: external_id needs to be nullable because old data won't have it
-- For rows with NULL external_id, uniqueness won't be enforced (which is fine for legacy data)
-- For new rows, we'll enforce it via application logic or a partial index if needed
-- But standard unique constraint allows multiple NULLs, so we use a partial unique index for non-nulls

CREATE UNIQUE INDEX IF NOT EXISTS idx_prompts_unique_external_id 
ON prompts (user_id, source, external_id) 
WHERE external_id IS NOT NULL;

-- Step 4: Add index on external_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_prompts_external_id ON prompts(external_id);

-- Verify
SELECT 
    indexname, 
    indexdef 
FROM pg_indexes 
WHERE tablename = 'prompts' 
AND indexname LIKE 'idx_prompts_unique%';
