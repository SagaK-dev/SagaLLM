# Compact algorithm contract and boundary audit (M2.6-i)

PR #112 head 8e29356d441e6c2d42214bae70f15e493c496746 passed its two CI
runs 46129795701 and 46129795708. The compact archive 15256963954 is pinned
by ZIP and member digest; #107's original reference report is independently
pinned by the inherited verifier. All 16 compact records are bound to exact
original/compact task, reference source and candidate row hashes; no source or
prompt bytes are changed in this stage.

## Author-side contract inspection (not human approval)

| Family | Contract | Review points |
| --- | --- | --- |
| Range sum | Inclusive zero-based ranges; one result per query | Japanese A[l..r] notation and numeric constraints imply inclusion and zero-based indices |
| Interval union | Integer half-open intervals, union length | Japanese short prompt omits the explicit word integer; confirm the intended input domain |
| Edit distance | ASCII lowercase 1..20, unit insertion/deletion/substitution | Regex and rolling-DP shorthand need bilingual review |
| Graph distance | Undirected unit-edge path length, -1 unreachable | Japanese Floyd says distance rather than minimum edge count; compact notation does not explicitly state unit edge costs |

These are author-side observations, not proof of linguistic equivalence or
independence. The original and compact source bytes are identical; executing
the references cannot prove what a learner or model interprets from the prompt.
Keep semantic and rights review at REVIEW_REQUIRED. Do not silently amend
prompts after the pinned report; any revision needs fresh admission evidence.

## Additional boundary evidence

Ten new distinct development inputs per family, all checked against the
original oracle and a separately specified exact expected output:
maximum sizes, negative/positive extremes, repeated range queries,
empty-width/touching/nested/disjoint intervals, maximum edit lengths,
maximum graph diameter/edge count, disconnected and duplicate/self-loop edges.
The last two graph cases are within the literal stated bounds (the original
contract does not exclude them); confirm that interpretation in human review.

The dedicated workflow compiles the eight fixed references with JDK 17 and
compares each exact stdout against the explicit expected bytes (80 comparisons).
No model-generated code is executed. The report stores only hashes, contract
notes and metadata, not Java source or input/output examples. This is not a
sandbox or independent benchmark. Record CI run/job/artifact/ZIP digest after
success. No changes to training, context policy, checkpoint, dataset roles,
heldout scoring or model adoption. Issue #22 remains open.
