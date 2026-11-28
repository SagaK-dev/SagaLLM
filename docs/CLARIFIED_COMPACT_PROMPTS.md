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

## Completing the two blocked Japanese rows

At PR #114 head `e2b830316baad835689644e3dbc0d96f02b3611c`, the
[dedicated audit](https://github.com/SagaK-dev/SagaLLM/actions/runs/46134157593)
succeeded with 14 of 16 selected rows. The shortest measured option for each
blocked row needed 389 positions. Artifact 15258872154 has ZIP SHA-256
`5b7378b1870cbcb3fcebf116b4ca50958bd9af2da91f02373d0fde6252c91b1f`.
Replaying the four pinned inputs reproduced its JSON byte for byte
(`1d029db9660079c8747292e7049466e311e536cbeb7659fad289e587e0614fc7`).

Two appended options bring the selection to 8 Japanese and 8 English records.
The other 14 report rows and all 24 earlier option measurements match that
replay. The dedicated workflow checks the digest of the full 26-option report.

| Development row | Selected option | Positions / 384 | Response tokens incl. EOS | Exact task SHA-256 |
| --- | ---: | ---: | ---: | --- |
| interval_union_length-sort_merge-ja | 3 | 383 | 265 | `b1a79e189c010dc61a8d8db86b726c6d75784798ea4222043d3e4eef97030337` |
| unweighted_graph_distance-floyd-ja | 4 | 384 | 292 | `a88937723b85acfe6e6c8ca235b2928330d408f9ce61408f3ae33e627a933286` |

For bilingual review, compare these options with the original Japanese and
English tasks in `src/sagallm/java_algorithm_curriculum.py`:

- **Interval union.** Input begins with the interval count (0–30), followed by
  that many integer pairs satisfying −100 ≤ l ≤ r ≤ 100. Sort and merge the
  half-open intervals and print their total union length. Empty input after
  the count or exclusively zero-length intervals gives 0.
- **Graph distance.** Read the vertex and edge counts, then the undirected
  edge pairs, then a and b. The notation `V=[0,n)` describes integer vertex
  IDs. The proposed Japanese-track prompt uses English output notation.
  Review whether `min hops(a,b)` clearly asks for the minimum number of edges
  and `no path=-1` clearly specifies the unreachable result. The full
  Floyd-Warshall name appears in the candidate. Equal endpoints give 0.

The complete candidate membership SHA-256 is
`6a4a4d8aa5de5df57a11530a7c610b61ea40877164b368f3c4cb4e77005778c7`.
For local reproduction, use the four artifact IDs and command in
`.github/workflows/audit-clarified-compact-prompts.yml`. The expected report
SHA-256 is `d2ec7ea34b7f245d7bca7ba8782cd33ba6d0f7dd8dabb8bcb37b43448b38c253`.
Human bilingual review and rights review remain open.
