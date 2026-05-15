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
  `attention_fwd_triton3`, `attn_fwd_causal`, `attn_fwd_triton`,
  `context_attn_fwd`, `context_attn_llama`,
  `context_attn_mistral`, and `mixed_sparse_attention`.
  These use proof-oriented output/accumulator slices or helper-inlined,
  fixed-stage surfaces while the Python kernels contain larger streaming
  softmax loops, helper JIT calls, or precomputed accumulator inputs. Closing
  this needs full helper-call surface modeling or full-kernel attention
  transcriptions rather than output-slice fronts.
  In particular, `attention_forward_triton` currently inlines the helper JIT
  `_attn_fwd_inner` into the `_attn_fwd` surface; removing its marker makes the
  mechanical audit compare helper parameters and call sequences against the
  inlined outer surface. `mixed_sparse_attention` is a true final-store slice
  from a precomputed normalized accumulator; the block-sparse and column-sparse
  softmax loops are not represented in the first surface.

- `rotary_transform` and `rotary_transform_ops`: the Python kernels combine
  varlen/non-varlen, tensor/scalar seqlen offsets, interleaved/non-interleaved,
  and conjugate/non-conjugate paths. Current Lean surfaces cover the
  non-varlen, non-interleaved, non-conjugate path. Closing this requires
  path-sensitive parameter refinement for `SEQLEN_OFFSETS`, `seqlen`, and
  `rotary_dim`-derived half ranges.

### Required VeriTile surface extensions

The remaining blockers are not currently local proof-script failures:

- The attention/context/flash kernels need full streaming-softmax loop
  surfaces, including helper-call modeling where the Python kernel delegates
  to a separate `@triton.jit` helper. The current proof slices intentionally
  start from precomputed accumulator/output tiles.
- `rotary_transform` and `rotary_transform_ops` need path-sensitive scalar
  parameter refinement across the varlen/non-varlen and scalar/tensor
  `SEQLEN_OFFSETS` paths. The full Python kernel rebinds `seqlen`, `X`, and
  `OUT` differently by path before the shared rotary body; the current first
  Lean surface covers only the non-varlen scalar-offset path.
