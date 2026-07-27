# Optional Study Backend

The public GitHub Pages site works without this backend. In its default mode,
participation codes, visits, and pre/post assessments remain in browser storage.

Enable the backend only after the instructional team confirms consent language,
accessibility, privacy, retention, and any course or human-subject research review.

## Supabase setup

1. Create a Supabase project controlled by the instructional team.
2. Run `schema.sql` in the Supabase SQL editor.
3. Enable email magic-link authentication and add the GitHub Pages URL to the
   allowed redirect URLs.
4. Copy `website/study-config.example.js` to `website/study-config.js`.
5. Set `enabled: true`, the project URL, and the public anonymous key.
6. Do not place a service-role key in the website.

The browser sends records only when the student checks the consent box. The public
client can insert but cannot read individual records. The exposed RPC returns only a
distinct participant count.

## Important limitations

- A public anonymous endpoint can be spammed, so the count is a usage estimate rather
  than a research-quality enrollment count.
- Formal analysis should deduplicate records, document exclusions, and use an approved
  export path controlled by the instructional team.
- Do not collect names, health details, raw motion files, or free-text medical content
  through these tables.
- Set and document a retention/deletion schedule before enabling data collection.

