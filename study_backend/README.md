# Anonymous Pilot Backend

The website can use Supabase anonymous Auth to create one de-identified
participant account per browser. Email, name, and OSU username are not required.

## Setup

1. Create a Supabase project approved for the pilot.
2. In **Authentication > Providers**, enable **Anonymous Sign-Ins**.
3. Consider enabling CAPTCHA before a public launch to reduce automated signups.
4. Open the Supabase SQL editor and run `schema.sql`.
5. Copy `website/study-config.example.js` to `website/study-config.js`.
6. Set `enabled: true`, the project URL, and the public anon/publishable key.
7. Add the website URL to the Supabase redirect/site URL allowlist.
8. Deploy the website and test with a new private browser window.

Never place the service-role key in website JavaScript. The public key is safe
to expose only because Row Level Security requires every uploaded record to match
the current anonymous Auth user.

The schema is intended for a fresh pilot project. If an earlier draft of the
schema was already installed, remove its empty draft tables before running this
version. Do not drop tables after real participant collection has begun.

## What Is Recorded

- one participant row after consent and `Begin Learning`
- one module-view event per module per page session
- submitted pre/post concept score, confidence, and selected answer indices
- every submitted quiz attempt, including score and per-question answers
- the latest explicitly saved reflection for each participant and module

The website does not upload imported motion files, unsaved draft reflections,
names, email addresses, or OSU identifiers.

## Viewing Results

Use the Supabase table editor for detailed authorized review:

- `participants`: de-identified participant count and consent version
- `usage_events`: module visits
- `assessment_responses`: pre/post learning records
- `quiz_attempts`: every quiz score and per-question response
- `reflections`: the latest saved reflection for each module

The public website can call `public_participant_count()` but cannot select
individual rows. Export or analyze individual records only from an authorized
administrator account.

Run `admin_queries.sql` in the Supabase SQL editor for ready-made aggregate
queries and an authorized reflection review table.

## Study Boundary

Finalize the consent language, retention period, deletion procedure, accessibility
review, and any required course or human-subjects review before central collection
is enabled. Anonymous browser accounts reduce direct identifiers but do not by
themselves make a study exempt from institutional requirements.
