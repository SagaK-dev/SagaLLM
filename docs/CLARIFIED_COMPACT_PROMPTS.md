# Clarified compact prompt candidates (M2.6-j)

Based on PR #113 head d4fa647b87598ac02344c9950658ed37928a57bd. Dedicated run
46130826759 and CPU CI 46130826734 succeeded; 8 Java compilations and 80
boundary comparisons were recorded in artifact 15258415001 (ZIP digest
c51ac98265fb8a75f26b1d9e7ab7be54812be2d2cb82ca33ae8c4c1e2b06ea40).
This stage reopens the pinned compact and boundary archives, binds all 16
rows, eight source implementations, and 80 historical case hashes. It does not
execute Java again or claim fresh execution.

Seven Japanese rows have fixed, ordered author-written alternatives:
two range-sum, two interval-union, rolling edit distance, and two graph-distance
prompts. The nine other prompts are unchanged. Explicit inclusive ranges,
integer endpoints, rolling 1-D DP and shortest edge counts are the review focus.
The shorter alternatives use mathematical notation; human review must confirm
the intended semantics, including the full algorithm name where abbreviated.
The first option that fits the unchanged 384-position context and has no static
canonical matches is selected as a technical proposal only. If none fits,
report a blocked ID; never silently reuse the ambiguous prior text or truncate
an answer. The full membership hash is emitted only for a complete set.
Every option is measured with the pinned tokenizer; eligible encodings are
checked for exact x/y labels, EOS, roundtrip, and boundary rejection. The
reference source and supervised response token count cannot change.
The report contains hashes and metadata, not new training JSONL.

## Review gate

Reviewers must verify the exact selected option number, original and proposed
task hashes, source hash and candidate membership. Confirm the complete input
domain, input order, output format and specified algorithm in both languages.
Reject any unresolved ambiguity; a revision requires fresh evidence.
Technical fit is not human semantic approval or rights clearance.
The static canonical overlap check reads TEST without tokenizing/scoring it.
No model loading, optimizer updates, training dataset publication, independent
evaluation or model promotion. Issue #22 remains open.
