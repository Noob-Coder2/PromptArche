-- Ingestion Jobs Table for tracking upload progress
create table if not exists ingestion_jobs (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null,
  status text not null default 'PENDING', -- PENDING, PARSING, EMBEDDING, COMPLETED, FAILED
  provider text not null,
  current_count int default 0,
  total_count int default 0,
  error_message text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Index for efficient job lookup
create index if not exists idx_jobs_user_status on ingestion_jobs(user_id, status);
create index if not exists idx_jobs_user_created on ingestion_jobs(user_id, created_at desc);

-- Row Level Security
alter table ingestion_jobs enable row level security;

create policy "Users can view their own jobs" 
on ingestion_jobs for select 
using (auth.uid() = user_id);

create policy "Users can insert their own jobs" 
on ingestion_jobs for insert 
with check (auth.uid() = user_id);

create policy "Users can update their own jobs" 
on ingestion_jobs for update 
using (auth.uid() = user_id);
