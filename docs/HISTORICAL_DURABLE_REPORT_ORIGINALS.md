# M2.4-o — durable report original preservation

The #79 recovery artifact 15240845377 contains 35 screened original JSON files and a recovery manifest. The archive has SHA-256 25a0d7cbacdead46d18c7261287a601bf7a2c114c8ff9a5a6ebf22a0e8ba3040; the exact manifest has SHA-256 8d5181f65d442a3aa283c3de606fc51fe2e73c0eae1a4005e6d63b8662d2a251. The artifact expires February 26, 2027.

Preserve the exact bytes in a separate evidence directory, including all repeated report instances and their original artifact/run identities. Reverify the recovery ZIP and prior triage ZIP before staging. Offline verification pins the original manifest and every member hash/size, rejects unexpected files and reruns the conservative metadata screen. No Action artifact is needed after the Git originals are present. This is not a license/provenance approval, independent benchmark, model checkpoint publication or complete exposure freeze. No model weights or raw corpus rows are stored.

The initial one-time publication workflow uses an owner/same-repository/fixed-branch PR-open guard, with an ordinary non-force push only to the evidence draft branch. After publication the workflow must be replaced by read-only verification. No main write or automatic merge. If publication cannot complete, do not claim permanent archival. Issue #22 remains open.

## Publication result

Run 46102253707, job 140847086641 succeeded. The exact originals and manifest were committed to the fixed draft evidence branch at `04b16fbb3a4901123d2891827b38beb6f492b4f4`. Publication staged 36 additions and checked each Git index blob against the original bytes before committing. The one-time write workflow was replaced by a read-only offline verification workflow in commit `5dc78c910164367153fc90dea7d9463aac1c2ce1`. No merge or main write occurred. The new verification CI result must be checked separately.
