create extension if not exists pgcrypto;

create table if not exists public.participants (
  user_id uuid primary key references auth.users(id) on delete cascade,
  participant_id uuid not null unique,
  consent_version text not null,
  consented_at timestamptz not null,
  last_seen_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  constraint participant_matches_user check (participant_id = user_id)
);

create table if not exists public.usage_events (
  id bigint generated always as identity primary key,
  participant_id uuid not null references public.participants(participant_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  event_name text not null check (event_name = 'module_view'),
  module text not null check (module in ('wfm', 'ecg', 'motion', 'resistance')),
  created_at timestamptz not null default now(),
  constraint usage_identity_matches check (participant_id = user_id)
);

create table if not exists public.assessment_responses (
  id bigint generated always as identity primary key,
  participant_id uuid not null references public.participants(participant_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  module text not null check (module in ('wfm', 'ecg', 'motion', 'resistance')),
  phase text not null check (phase in ('pre', 'post')),
  knowledge_score integer not null check (knowledge_score between 0 and 2),
  confidence integer not null check (confidence between 1 and 5),
  answers jsonb not null,
  created_at timestamptz not null default now(),
  constraint assessment_identity_matches check (participant_id = user_id)
);

create table if not exists public.quiz_attempts (
  id bigint generated always as identity primary key,
  participant_id uuid not null references public.participants(participant_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  module text not null check (module in ('wfm', 'ecg', 'motion', 'resistance')),
  score integer not null check (score >= 0),
  total_questions integer not null check (total_questions > 0),
  answers jsonb not null,
  created_at timestamptz not null default now(),
  constraint quiz_score_within_total check (score <= total_questions),
  constraint quiz_identity_matches check (participant_id = user_id)
);

create table if not exists public.reflections (
  participant_id uuid not null references public.participants(participant_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  module text not null check (module in ('wfm', 'ecg', 'motion', 'resistance')),
  response_text text not null check (char_length(response_text) between 20 and 5000),
  word_count integer not null check (word_count > 0),
  updated_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  primary key (participant_id, module),
  constraint reflection_identity_matches check (participant_id = user_id)
);

alter table public.participants enable row level security;
alter table public.usage_events enable row level security;
alter table public.assessment_responses enable row level security;
alter table public.quiz_attempts enable row level security;
alter table public.reflections enable row level security;

grant select, insert, update on public.participants to authenticated;
grant insert on public.usage_events to authenticated;
grant insert on public.assessment_responses to authenticated;
grant insert on public.quiz_attempts to authenticated;
grant select, insert, update on public.reflections to authenticated;
grant usage, select on all sequences in schema public to authenticated;

drop policy if exists "participant can create own record" on public.participants;
create policy "participant can create own record"
on public.participants for insert
to authenticated
with check (user_id = auth.uid() and participant_id = auth.uid());

drop policy if exists "participant can read own record" on public.participants;
create policy "participant can read own record"
on public.participants for select
to authenticated
using (user_id = auth.uid());

drop policy if exists "participant can update own record" on public.participants;
create policy "participant can update own record"
on public.participants for update
to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid() and participant_id = auth.uid());

drop policy if exists "participant can insert own usage" on public.usage_events;
drop policy if exists "consented clients can insert usage events" on public.usage_events;
create policy "participant can insert own usage"
on public.usage_events for insert
to authenticated
with check (user_id = auth.uid() and participant_id = auth.uid());

drop policy if exists "participant can insert own assessment" on public.assessment_responses;
drop policy if exists "consented clients can insert assessment responses" on public.assessment_responses;
create policy "participant can insert own assessment"
on public.assessment_responses for insert
to authenticated
with check (user_id = auth.uid() and participant_id = auth.uid());

drop policy if exists "participant can insert own quiz attempt" on public.quiz_attempts;
create policy "participant can insert own quiz attempt"
on public.quiz_attempts for insert
to authenticated
with check (user_id = auth.uid() and participant_id = auth.uid());

drop policy if exists "participant can create own reflection" on public.reflections;
create policy "participant can create own reflection"
on public.reflections for insert
to authenticated
with check (user_id = auth.uid() and participant_id = auth.uid());

drop policy if exists "participant can read own reflection" on public.reflections;
create policy "participant can read own reflection"
on public.reflections for select
to authenticated
using (user_id = auth.uid());

drop policy if exists "participant can update own reflection" on public.reflections;
create policy "participant can update own reflection"
on public.reflections for update
to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid() and participant_id = auth.uid());

create or replace function public.public_participant_count()
returns bigint
language sql
security definer
set search_path = public
as $$
  select count(*) from public.participants;
$$;

revoke all on function public.public_participant_count() from public;
grant execute on function public.public_participant_count() to anon, authenticated;
