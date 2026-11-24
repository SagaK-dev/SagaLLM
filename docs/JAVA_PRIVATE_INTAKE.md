# M2 private Java benchmark intake (preflight only)

Keep newly authored task text, cases, permissions and reviews outside Git and
outside shared CI. Do not send private tasks to model training, model selection
or an assistant before the evaluation freeze. This tool runs locally, without
network, model loading, inference or Java execution. No human task or approval
is supplied by this change.

Use the suite format in JAVA_INDEPENDENT_IO_BENCHMARK.md. Prepare 5–12 genuine
human tasks and obtain two real reviews for every task as required by #36.

## Private task provenance JSON

Schema: `sagallm-private-java-intake-v1`; top-level keys: schema,
suite_sha256, tasks. Each tasks entry must contain exactly:

- task_id, task_sha256 (canonical task object SHA-256 using
  `task_sha256` in the module; it covers prompt, family and all I/O)
- author_id, authored_on (YYYY-MM-DD), revision, source_url
- license_id, permission_url, allowed_uses (evaluate and redistribute only;
  training is prohibited; an empty list means permission outstanding)
- held_back (true only with actual evidence)
- rights_review, io_review, overlap_review (PENDING or ATTESTED; do not
  attest without genuine dated per-task reviews)

The entire suite, provenance and inventory are separately SHA-256 pinned.
The tool does not authenticate source URLs, authors or permissions.

## Private exposure inventory JSON

Schema: `sagallm-private-java-exposure-v1`; top-level keys schema, files.
Each file entry has group (train, validation, test, development, pretraining),
path (relative JSONL path inside exposure root), sha256, format (chat-jsonl
for messages with role/content; text-jsonl for text; prompt-jsonl for prompt).
List the exact actual exposures, including experiments and pretraining, not
merely arbitrary fixtures. Missing groups are explicit blockers. Files are
hashed over the same bytes that are scanned. No network or source download.

Example command (replace every hash with the real local value):

    PYTHONPATH=src python scripts/audit_java_private_intake.py \
      --suite /private/suite.json --expected-suite-sha256 SUITE_SHA \
      --provenance /private/provenance.json --expected-provenance-sha256 PROVENANCE_SHA \
      --inventory /private/inventory.json --expected-inventory-sha256 INVENTORY_SHA \
      --exposure-root /private/exposure --output /private/intake-report.json

## What the report means

The screen uses historical synthetic family names from current modules and
nine exposed public/draft tasks, plus exact family/ID, Unicode NFKC and
case/whitespace-normalized text, numeric-template and sufficiently long
prompt-substring comparisons against supplied JSONL records. Within-suite
duplicate family/prompt/template matches are review leads. Reports contain
task indices, group, match category, counts and hashes, not task text,
source/permission URLs, reviewer identities, corpus paths or Java source.

Even a clean PREFLIGHT_ONLY is not approval. All outcomes remain
REVIEW_REQUIRED, independence_proven=false, behavior=NOT_RUN. A supplied
inventory may omit exposures; the tool cannot know its completeness.
Lexical comparisons miss paraphrases, translations, structural/semantic
similarity and memorized code. Numeric matches may be false positives.
Use a real independent human review of the exact source, I/O, rights and
complete exposure history before freezing the suite. This preflight is a
separate preparation step, not yet a mandatory runtime gate.

Limits: 64 input files, 100 MB/file, 512 MB total, 500,000 JSONL rows,
256 KiB/line; no silent truncation. Larger corpora require a separately
reviewed scalable screening method; never substitute a sample and claim
full coverage. CI uses only artificial fixtures; do not upload private data.
