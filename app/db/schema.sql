-- Enable Vector extension
create extension if not exists vector;

-- Prompts Table
create table if not exists prompts (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null default auth.uid(), -- Link to Supabase Auth User
  content text not null,
  source text, -- e.g. 'chatgpt', 'claude'
  metadata jsonb default '{}'::jsonb, -- Store extra info like 'success_score'
  embedding vector(1024), -- bge-large-en-v1.5
  cluster_id uuid, -- Check foreign key constraint after creating clusters table
  created_at timestamptz default now(),
  constraint unique_prompt_per_user unique (user_id, content)
);

-- Indexes for Prompts (Performance & Deduplication)
create index if not exists idx_prompts_user_id on prompts(user_id);
create index if not exists idx_prompts_embedding on prompts using hnsw (embedding vector_cosine_ops);


-- Clusters Table
create table if not exists clusters (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null default auth.uid(),
  label text,
  description text,
  centroid vector(1024),
  created_at timestamptz default now()
);

-- Indexes for Clusters
create index if not exists idx_clusters_user_id on clusters(user_id);

-- Insights Table
create table if not exists insights (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null default auth.uid(),
  cluster_id uuid references clusters(id) on delete cascade,
  content text, -- The 'brutally honest' insight
  created_at timestamptz default now()
);

-- Indexes for Insights
create index if not exists idx_insights_cluster_id on insights(cluster_id);


-- Embedding Cache Table (NEW)
-- Stores embeddings for duplicate detection and caching
create table if not exists embedding_cache (
  text_hash varchar(64) primary key,
  embedding vector(1024),
  created_at timestamptz default now()
);

-- Index for cache lookups
create index if not exists idx_embedding_cache_hash on embedding_cache(text_hash);

-- Add Foreign Key to prompts for clusters
alter table prompts 
add constraint fk_cluster 
foreign key (cluster_id) 
references clusters(id) 
on delete set null;

-- Row Level Security (RLS) Policies
-- Ensure RLS is enabled
alter table prompts enable row level security;
alter table clusters enable row level security;
alter table insights enable row level security;

-- Policies for Prompts
create policy "Users can view their own prompts" 
on prompts for select 
using (auth.uid() = user_id);

create policy "Users can insert their own prompts" 
on prompts for insert 
with check (auth.uid() = user_id);

create policy "Users can update their own prompts" 
on prompts for update 
using (auth.uid() = user_id);

create policy "Users can delete their own prompts" 
on prompts for delete 
using (auth.uid() = user_id);

-- Policies for Clusters
create policy "Users can view their own clusters" 
on clusters for select 
using (auth.uid() = user_id);

create policy "Users can insert their own clusters" 
on clusters for insert 
with check (auth.uid() = user_id);

-- Policies for Insights
create policy "Users can view their own insights" 
on insights for select 
using (auth.uid() = user_id);

create policy "Users can insert their own insights" 
on insights for insert 
with check (auth.uid() = user_id);

-- Performance Views
create or replace view prompt_stats_daily as
select 
  date_trunc('day', created_at) as day, 
  user_id,
  count(*) as count
from prompts
group by 1, 2;
