-- Analytics Views for Prompt Statistics
-- Purpose: Provide aggregated views for prompt length distribution and analytics
-- These views query the prompts table on-demand using the metadata->>'prompt_length' field

-- ============================================================================
-- View 1: User Prompt Statistics (Per-User Summary)
-- ============================================================================
CREATE OR REPLACE VIEW user_prompt_statistics AS
SELECT 
    user_id,
    COUNT(*) as total_prompts,
    ROUND(AVG((metadata->>'prompt_length')::int)) as avg_prompt_length,
    MIN((metadata->>'prompt_length')::int) as min_prompt_length,
    MAX((metadata->>'prompt_length')::int) as max_prompt_length,
    COUNT(DISTINCT source) as sources_used,
    ROUND(AVG((metadata->>'prompt_length')::int) FILTER (WHERE source = 'chatgpt')) as avg_length_chatgpt,
    ROUND(AVG((metadata->>'prompt_length')::int) FILTER (WHERE source = 'claude')) as avg_length_claude,
    ROUND(AVG((metadata->>'prompt_length')::int) FILTER (WHERE source = 'grok')) as avg_length_grok,
    MIN(created_at) as first_prompt_date,
    MAX(created_at) as last_prompt_date
FROM prompts
WHERE metadata->>'prompt_length' IS NOT NULL
GROUP BY user_id;

-- ============================================================================
-- View 2: Prompt Length Distribution (Histogram Buckets)
-- ============================================================================
CREATE OR REPLACE VIEW prompt_length_distribution AS
WITH buckets AS (
    SELECT 
        user_id,
        CASE 
            WHEN (metadata->>'prompt_length')::int < 50 THEN '0-50'
            WHEN (metadata->>'prompt_length')::int < 100 THEN '50-100'
            WHEN (metadata->>'prompt_length')::int < 200 THEN '100-200'
            WHEN (metadata->>'prompt_length')::int < 500 THEN '200-500'
            WHEN (metadata->>'prompt_length')::int < 1000 THEN '500-1K'
            WHEN (metadata->>'prompt_length')::int < 2000 THEN '1K-2K'
            WHEN (metadata->>'prompt_length')::int < 5000 THEN '2K-5K'
            ELSE '5K+'
        END as length_bucket,
        (metadata->>'prompt_length')::int as prompt_length
    FROM prompts
    WHERE metadata->>'prompt_length' IS NOT NULL
)
SELECT 
    user_id,
    length_bucket,
    COUNT(*) as bucket_count,
    ROUND(AVG(prompt_length)) as avg_length_in_bucket,
    MIN(prompt_length) as min_length_in_bucket,
    MAX(prompt_length) as max_length_in_bucket
FROM buckets
GROUP BY user_id, length_bucket
ORDER BY user_id, 
    CASE length_bucket
        WHEN '0-50' THEN 1
        WHEN '50-100' THEN 2
        WHEN '100-200' THEN 3
        WHEN '200-500' THEN 4
        WHEN '500-1K' THEN 5
        WHEN '1K-2K' THEN 6
        WHEN '2K-5K' THEN 7
        WHEN '5K+' THEN 8
    END;

-- ============================================================================
-- View 3: Prompt Stats by Source Provider
-- ============================================================================
CREATE OR REPLACE VIEW prompt_stats_by_source AS
SELECT 
    user_id,
    source,
    COUNT(*) as prompt_count,
    ROUND(AVG((metadata->>'prompt_length')::int)) as avg_length,
    MIN((metadata->>'prompt_length')::int) as min_length,
    MAX((metadata->>'prompt_length')::int) as max_length,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (metadata->>'prompt_length')::int) as median_length,
    MIN(created_at) as first_prompt,
    MAX(created_at) as last_prompt,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as prompts_last_30_days
FROM prompts
WHERE metadata->>'prompt_length' IS NOT NULL
GROUP BY user_id, source
ORDER BY user_id, prompt_count DESC;

-- ============================================================================
-- View 4: Temporal Trends (Daily Aggregates)
-- ============================================================================
CREATE OR REPLACE VIEW prompt_trends_daily AS
SELECT 
    user_id,
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as prompts_count,
    ROUND(AVG((metadata->>'prompt_length')::int)) as avg_length,
    COUNT(DISTINCT source) as sources_used,
    COUNT(*) FILTER (WHERE source = 'chatgpt') as chatgpt_count,
    COUNT(*) FILTER (WHERE source = 'claude') as claude_count,
    COUNT(*) FILTER (WHERE source = 'grok') as grok_count
FROM prompts
WHERE metadata->>'prompt_length' IS NOT NULL
GROUP BY user_id, DATE_TRUNC('day', created_at)
ORDER BY user_id, date DESC;

-- ============================================================================
-- View 5: Extreme Prompts (Longest and Shortest)
-- ============================================================================
CREATE OR REPLACE VIEW extreme_prompts AS
WITH ranked_prompts AS (
    SELECT 
        id,
        user_id,
        source,
        (metadata->>'prompt_length')::int as length,
        LEFT(content, 100) as preview,
        created_at,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY (metadata->>'prompt_length')::int DESC) as rank_longest,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY (metadata->>'prompt_length')::int ASC) as rank_shortest
    FROM prompts
    WHERE metadata->>'prompt_length' IS NOT NULL
)
SELECT 
    user_id,
    id,
    source,
    length,
    preview,
    created_at,
    CASE 
        WHEN rank_longest <= 10 THEN 'longest'
        WHEN rank_shortest <= 10 THEN 'shortest'
    END as category
FROM ranked_prompts
WHERE rank_longest <= 10 OR rank_shortest <= 10
ORDER BY user_id, category, length DESC;

-- ============================================================================
-- Indexes for Performance (if not already present)
-- ============================================================================

-- Index on metadata->>'prompt_length' for faster filtering and aggregation
CREATE INDEX IF NOT EXISTS idx_prompts_metadata_length 
ON prompts (((metadata->>'prompt_length')::int));

-- Composite index for temporal queries
CREATE INDEX IF NOT EXISTS idx_prompts_user_created 
ON prompts(user_id, created_at DESC);

-- Index for source-based queries
CREATE INDEX IF NOT EXISTS idx_prompts_user_source 
ON prompts(user_id, source);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to get analytics summary for a specific user
CREATE OR REPLACE FUNCTION get_user_analytics(target_user_id UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'summary', (SELECT row_to_json(t) FROM (
            SELECT * FROM user_prompt_statistics WHERE user_id = target_user_id
        ) t),
        'distribution', (SELECT json_agg(row_to_json(t)) FROM (
            SELECT * FROM prompt_length_distribution WHERE user_id = target_user_id
        ) t),
        'by_source', (SELECT json_agg(row_to_json(t)) FROM (
            SELECT * FROM prompt_stats_by_source WHERE user_id = target_user_id
        ) t),
        'recent_trends', (SELECT json_agg(row_to_json(t)) FROM (
            SELECT * FROM prompt_trends_daily 
            WHERE user_id = target_user_id 
            ORDER BY date DESC 
            LIMIT 30
        ) t),
        'extremes', (SELECT json_agg(row_to_json(t)) FROM (
            SELECT * FROM extreme_prompts WHERE user_id = target_user_id
        ) t)
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION get_user_analytics(UUID) TO authenticated;

-- ============================================================================
-- Usage Examples
-- ============================================================================

-- Example 1: Get your own analytics summary
-- SELECT get_user_analytics(auth.uid());

-- Example 2: View your prompt length distribution
-- SELECT * FROM prompt_length_distribution WHERE user_id = auth.uid();

-- Example 3: Compare sources
-- SELECT * FROM prompt_stats_by_source WHERE user_id = auth.uid();

-- Example 4: View trends over last 30 days
-- SELECT * FROM prompt_trends_daily WHERE user_id = auth.uid() ORDER BY date DESC LIMIT 30;
