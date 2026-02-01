-- Function to get information about public tables
-- Used by the application health check to verify schema readiness without separate permission checks
create or replace function get_tables_info()
returns json
language plpgsql
security definer
as $$
begin
  return (
    select json_agg(to_json(t))
    from (
      select tablename
      from pg_tables
      where schemaname = 'public'
    ) t
  );
end;
$$;
