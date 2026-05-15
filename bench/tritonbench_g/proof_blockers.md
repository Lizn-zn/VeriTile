# TritonBench-G Proof Blockers

No current TritonBench-G port exposes an explicit algorithm-layer `hAlg`
correctness blocker.

The mechanical audit still remains a translation-consistency gate, not a
substitute for future human line review against `review_criteria.md`.

## Translation-Surface Blockers

No current TritonBench-G port has an active documented translation-surface
blocker.

### Required VeriTile surface extensions

The attention helper-call surfaces are currently modeled as opaque outer-kernel
statements for translation coverage. Future proof work may add executable
semantics for projecting those helper calls, but that is no longer a
translation-surface blocker.
