# M2.4-h.7 — remaining historical scope and integration

PR #75 final head `bf36b83a85fcad00f19257d88fef3a3c8bf82c1e` passed CPU CI 46003548871 and dedicated CI 46003548951. Job 140520948395 completed tests, actual report comparison, and artifact upload. Artifact 15199855239 was unexpired at inspection, API ZIP SHA-256 `8f49329e762f9a34960f7748d7d18198054e761aad141686657068ec4a386b44`. This work pins that exact archive and reopens the earlier Parameter CE inventory archive 15186012660, checking its digest chain and exact 15 path/group entries.

## Remaining archived scope

- Contrastive STDIO A/B (45167198398): recount historical 54 case categories across three model labels; all 0/18. These are report records from historical OCI execution, not new runtime results. Do not interpret NONZERO_EXIT_OR_JAVAC_ERROR as compile-only failure or verified isolation. Confirm 160 reported steps per arm, parity declarations, and no adoption. Control checkpoint unavailable in the historical artifact.
- M1 Conditioning (45369765716) and rerun (45369987553): compare 12 prompt declarations and 24 historical generation hash declarations with the A/B records. Both refer to the same six development tasks, not an independent benchmark. The two original reports are not byte-identical and their scores are not recomputed.
- Blocked Alignment v1 (45372372592): 194/320 admitted directions, 126 excluded, 282 aligned positions, one entirely missing pair type; zero optimizer steps and no new checkpoint.

## Historical reconstruction

At exact historical revisions, reconstruct A/B corpus and curriculum bytes, admitted TRAIN rows, schedules and selection/holdback split; check report-declared parity. New schedule and selection digests are derived here, not present in the original report and not proof of executed order. Rebuild six M1 input prefixes and reference lengths with the pinned tokenizer, compare both report records; check unchanged diagnostic/source code between M1 revisions. Rebuild the blocked v1 preflight masks against the original digest. The historical curriculum/corpus builders regenerate synthetic splits, but no model sees them in this audit. The workflow extracts only the tokenizer from an exact pinned historical model ZIP, never loads any checkpoint, starts optimization or runs Java.

## Integrated scope (not a freeze)

The twelve archive manifest reports are SHA-checked and indexed with evidence scope. The map contains 17 distinct declared checkpoint digests (ancestor/best/last, A/B candidate and six arms each of Ranking and Parameter CE) and a separate unresolved A/B control declaration. Source assignments are conservative pool and context links, not exact per-model optimizer exposure; the same source may have multiple roles. A blocked preflight has no new checkpoint. No new inventory sources are added, no prior source rescan or sharding performed. v0.16/v0.17 source links are inherited from prior lineage scope, not new training proof.

The read-only CI emits four metadata JSON files only: integrated-exposure-scope.json and early-{ab,m1,blocked}-reconstruction.json. Unit/mutation tests include role drift, duplicate sources, missing archive report, altered scope, altered historical case or generation counts, false admission and ZIP mismatch. CI pending.

## Remaining gates

Enumerate all historical runs and local/unarchived experiments; recover model-specific actual optimizer traces, source bytes and transformed probe inputs; audit exact checkpoint bytes and lineage, reconcile corpus rights and provenance, obtain genuinely independent reviewed benchmark and separate VM approval. Expired/missing artifacts must block corresponding checks, not be silently replaced. No model promotion or Issue #22 closure.
