-- Overall pilot totals
select
  (select count(*) from public.participants) as participants,
  (select count(*) from public.usage_events) as module_views,
  (select count(*) from public.assessment_responses) as pre_post_submissions,
  (select count(*) from public.quiz_attempts) as quiz_attempts,
  (select count(*) from public.reflections) as saved_reflections;

-- Module engagement
select
  module,
  count(*) as views,
  count(distinct participant_id) as unique_participants
from public.usage_events
group by module
order by module;

-- Quiz performance by module
select
  module,
  count(*) as attempts,
  count(distinct participant_id) as participants,
  round(avg(100.0 * score / total_questions), 1) as mean_percent
from public.quiz_attempts
group by module
order by module;

-- Latest matched pre/post change for each participant and module
with ranked as (
  select
    participant_id,
    module,
    phase,
    knowledge_score,
    confidence,
    row_number() over (
      partition by participant_id, module, phase
      order by created_at desc
    ) as rank
  from public.assessment_responses
),
paired as (
  select
    pre.participant_id,
    pre.module,
    pre.knowledge_score as pre_score,
    post.knowledge_score as post_score,
    pre.confidence as pre_confidence,
    post.confidence as post_confidence
  from ranked pre
  join ranked post
    on post.participant_id = pre.participant_id
   and post.module = pre.module
  where pre.phase = 'pre'
    and post.phase = 'post'
    and pre.rank = 1
    and post.rank = 1
)
select
  module,
  count(*) as paired_participants,
  round(avg(post_score - pre_score), 2) as mean_score_change,
  round(avg(post_confidence - pre_confidence), 2) as mean_confidence_change
from paired
group by module
order by module;

-- Latest saved reflections for authorized qualitative review
select
  upper(left(participant_id::text, 8)) as participant_code,
  module,
  word_count,
  response_text,
  updated_at
from public.reflections
order by updated_at desc;
