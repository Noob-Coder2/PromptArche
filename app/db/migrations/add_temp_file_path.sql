-- Migration: Add temp_file_path column to ingestion_jobs table
-- Run this in your Supabase SQL Editor

-- Add the missing column
ALTER TABLE ingestion_jobs 
ADD COLUMN IF NOT EXISTS temp_file_path text;

-- Verify the column was added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ingestion_jobs' 
ORDER BY ordinal_position;
