# bmm_chunk_bwd

- Source: `bmm_chunk_bwd.py`, TritonBench-G v1; `_bmm_chunk_bwd_kernel` at line 21.
- Public source: `Kernels.lean`.
- Real correctness: `Correct.lean`.
- Admitted FP equivalence: `FPEquiv.lean`.
- Status: complete real/FP contracts, regressions and independent replay accepted.

The original computes chunked `Doutᵀ · A`, optionally adds the initial
residual, converts to the output dtype and stores a masked gradient tile.
The candidate changes only `acc + res` to `res + acc` in the residual arm.

`Correct.lean` gives both versions the same independent initial-array sum,
using the actual in-body length `min(chunk_size, seqlen - pid_c * chunk_size)`.
The contract proves successful flat execution, active-address bounds and
preservation of every tagged cell outside the active output window. Partial
contraction/output tiles, all three program IDs, independent strides and both
residual arms remain symbolic. Only positive contraction block size and
injectivity of active output addresses are required by the value proof.
Natural index arithmetic models valid host chunks; host-grid composition,
fixed-width overflow and launch-wide races are separate obligations.

A and Dout independently use fp32/fp16/bf16; all nine pairs retain the host's
bf16-then-fp16-then-fp32 dot-dtype priority. Residual/output dtypes and casts
remain explicit. Pointer storage is preserved across nested load/cast
expressions, with the original load kwargs unchanged. Real correctness erases
these precision annotations; FP equivalence retains them.

`#print_fp_assumptions bmm_residual_equiv` prints only `add_commute`.
The condition checks the finite actual fp32 accumulator and converted residual
at the reached addition after the complete loop and residual load. The
no-residual path is identical. No dot/reduction reassociation or additional
GPU experiment is used, and the claim is not a concrete IEEE identity.

The shared IO core now supports a formula that depends on the third program
ID, needed for the final partial chunk. The existing two-coordinate formula
interface reuses that bridge. `KernelIOGrid` checks an actual kernel that
stores its third program ID.

The integrated `VeriTile`, `TritonBenchGVerified` and
`TritonBenchSpecExamples` build passes (4,002 jobs). Source gates, all 370
coverage reviews across 195 public files, the proof-gap manifest and layout
checks pass. `BmmChunkBwdComplete` runs both variants with all three program
IDs nonzero, a length-three final chunk, both residual choices and partial
outputs. The full flat executions produce 112/339 without the residual and
129/358 with it; inactive fp16 cells retain 9. Earlier load/loop tests use
nonzero padding and undefined values to distinguish the source's load masks
from its narrower store mask. Separate consistent abstract FP executions
retain typed storage and witness the reached addition domain.

`Proofs/` contains the source connections, dynamic-loop induction, residual
and store execution, independent formula, safety and shared flat-IO assembly.
`Proofs/Legacy.lean` retains the older no-residual whole-contraction
specialization; the public contracts use the complete source proof.
Independent comparator/trust replay accepts all 37 files: every source/proof
file in this directory, the shared IO core and six focused regressions. The
verified inventory contains 207 named declarations and 1,179 kernel theorem
targets; all 990 snapshot inputs matched at acceptance. Receipt:
`Logs/comparator-check-l1uxmrt9`.
