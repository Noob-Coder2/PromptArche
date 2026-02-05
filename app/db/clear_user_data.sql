-- Clear all user-related data while keeping the schema and user accounts
-- This script removes:
-- - All prompts
-- - All clusters
-- - All insights  
-- - All ingestion jobs
-- - All embedding cache
-- User accounts in auth.users remain untouched

BEGIN;

-- Clear in order of dependencies (foreign keys)
-- First, clear insights (depends on clusters)
DELETE FROM insights;

-- Clear prompts (has foreign key to clusters)
DELETE FROM prompts;

-- Clear clusters
DELETE FROM clusters;

-- Clear ingestion jobs
DELETE FROM ingestion_jobs;

-- Clear embedding cache
DELETE FROM embedding_cache;

-- Reset sequences/auto-increment if any
-- (These aren't used in this schema as UUID is used, but included for completeness)

COMMIT;

-- Verify all tables are empty
SELECT 'prompts' as table_name, count(*) as row_count FROM prompts
UNION ALL
SELECT 'clusters', count(*) FROM clusters
UNION ALL
SELECT 'insights', count(*) FROM insights
UNION ALL
SELECT 'ingestion_jobs', count(*) FROM ingestion_jobs
UNION ALL
SELECT 'embedding_cache', count(*) FROM embedding_cache;
