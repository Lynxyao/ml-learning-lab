create extension if not exists pgcrypto;

create table if not exists public.usage_events (
  id bigint generated always as identity primary key,
  participant_id uuid not null,
  user_id uuid null references auth.users(id) on delete set null,
  event_name text not null check (event_name in ('module_view')),
  module text not null check (module in ('wfm', 'ecg', 'motion', 'resistance')),
  created_at timestamptz not null default now()
);

create table if not exists public.assessment_responses (
  id bigint generated always as identity primary key,
  participant_id uuid not null,
  user_id uuid null references auth.users(id) on delete set null,
  module text not null check (module in ('wfm', 'ecg', 'motion', 'resistance')),
  phase text not null check (phase in ('pre', 'post')),
  knowledge_score integer not null check (knowledge_score between 0 and 2),
  confidence integer not null check (confidence between 1 and 5),
  answers jsonb not null,
  created_at timestamptz not null default now()
);

alter table public.usage_events enable row level security;
alter table public.assessment_responses enable row level security;

grant insert on public.usage_events to anon, authenticated;
grant insert on public.assessment_responses to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated;

create policy "consented clients can insert usage events"
on public.usage_events for insert
to anon, authenticated
with check (participant_id is not null);

create policy "consented clients can insert assessment responses"
on public.assessment_responses for insert
to anon, authenticated
with check (participant_id is not null);

create or replace function public.public_participant_count()
returns bigint
language sql
security definer
set search_path = public
as $$
  select count(distinct participant_id) from public.usage_events;
$$;

revoke all on function public.public_participant_count() from public;
grant execute on function public.public_participant_count() to anon, authenticated;
