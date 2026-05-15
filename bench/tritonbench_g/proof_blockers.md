# TritonBench-G Proof Blockers

No current TritonBench-G port exposes an explicit algorithm-layer `hAlg`
correctness blocker.

The mechanical audit still remains a translation-consistency gate, not a
substitute for future human line review against `review_criteria.md`.

## Translation-Surface Blockers

### Remaining non-faithful or split surface categories

The strict `bench/audit_tritonbench_g.sh` marker set currently exempts the
following translation surfaces because the Lean first surface is intentionally
not a one-to-one Python transcription yet:

- Attention/context/flash attention kernels:
  `attention_forward_triton`, `attention_fwd_triton2`,
  `attention_fwd_triton3`, `attn_fwd_causal`, and `attn_fwd_triton`.
  These now expose outer-kernel surfaces with opaque helper-call statements
  where the Python `_attn_fwd` delegates to `_attn_fwd_inner`, but they still
  keep proof-oriented output/accumulator slices as the correctness target.
  Closing this category needs either executable semantics for helper-call
  projection or a target-JIT-aware audit path that can mechanically compare the
  outer `_attn_fwd` surface without falling back to the first helper JIT.

### Required VeriTile surface extensions

The remaining blockers are not currently local proof-script failures:

- The attention/context/flash kernels need full streaming-softmax loop
  semantics/proofs behind the helper-call surface where the Python kernel
  delegates to a separate `@triton.jit` helper. The current proof slices
  intentionally start from precomputed accumulator/output tiles.
