# TritonBench-G Proof Blockers

No current TritonBench-G port exposes an explicit algorithm-layer `hAlg`
correctness blocker.

The mechanical audit still remains a translation-consistency gate, not a
substitute for future human line review against `review_criteria.md`.

The stronger proof-status inventory for #146 lives in
`proof_gap_manifest.tsv` and is checked by
`bench/check_proof_gap_manifest.py`. It classifies every current
`output_summary` declaration and links each remaining non-full proof gap to a
specific follow-up issue plus a blocker family. The manifest is intentionally conservative:
`full_value_candidate` means no local proof-gap marker was found in the summary
context, not that future human review is forbidden from downgrading it.
The broad #147 quantization bucket has been split: real-to-int8 cast semantics
now track under #154, while rows whose local blocker is primarily attention,
matmul, recurrence, reduction, or explicit blocked-summary work track under the
corresponding family issue.
The broad #153 rotary/cache bucket has also been split into narrower value
proof follow-ups: `rope-head-slice-lift` covers RoPE summaries whose Python
surface is faithful but whose value proof is still stated over Q/K head slices,
and `rotary-2d-tile-value-lift` covers rotary summaries with row-level `o0` /
`o1` value proofs that still need the full `[BLOCK_M, BLOCK_HALF]` tile lift.
The explicit blocked-output summaries formerly tracked by the broad #152
`semantic-blocker` bucket are all quantization `llrint` / int8-cast blockers
and now track under the open #154 `fixed-width-int8-cast-semantics` family.

## Translation-Surface Blockers

No current TritonBench-G port has an active documented translation-surface
blocker.

### Required VeriTile surface extensions

The attention helper-call surfaces are currently modeled as opaque outer-kernel
statements for translation coverage. Future proof work may add executable
semantics for projecting those helper calls, but that is no longer a
translation-surface blocker.
