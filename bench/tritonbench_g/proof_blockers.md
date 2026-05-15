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
  `attention_forward_triton`, `attention_fwd_triton1`,
  `attention_fwd_triton2`, `attention_fwd_triton3`, `attention_kernel`,
  `attention_kernel_aligned`, `attn_fwd_causal`, `attn_fwd_triton`,
  `context_attn_bloom`, `context_attn_fwd`, `context_attn_llama`,
  `context_attn_mistral`, `context_attn_nopad`, `flash_attn`,
  `mixed_sparse_attention`, and `triton_attention`.
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

- `attention_score`: the Python test surface fixes `_BLOCK_M == _BLOCK_N`,
  while the general kernel shape uses separate M/N parameters. The Lean surface
  currently specializes this equality to keep the `tl.zeros([_BLOCK_M])` /
  `tl.sum(..., axis=0)` shape flow typable. Closing this requires constraint
  propagation for constexpr shape equalities.

- `rotary_transform` and `rotary_transform_ops`: the Python kernels combine
  varlen/non-varlen, tensor/scalar seqlen offsets, interleaved/non-interleaved,
  and conjugate/non-conjugate paths. Current Lean surfaces cover the
  non-varlen, non-interleaved, non-conjugate path. Closing this requires
  path-sensitive parameter refinement for `SEQLEN_OFFSETS`, `seqlen`, and
  `rotary_dim`-derived half ranges. It also requires statement-level early
  return support: the Python kernel uses `if pid_m * BLOCK_M >= seqlen:
  return`, while the current Lean surface represents that as an active-block
  guard because `Stmt` / `ComputeStmt` and `stepStmts` have no early-exit
  channel.

### Required VeriTile surface extensions

The remaining blockers are not currently local proof-script failures:

- `attention_score` needs macro-time constexpr shape-equality propagation.
  The Python wrapper fixes `BLOCK_M == BLOCK_N`, but the kernel signature
  carries both names. Faithfully restoring `o = tl.zeros([BLOCK_M],
  dtype=tl.float32)` makes the later `o += tl.sum(p, axis=0)` combine
  `[BLOCK_M]` with `[BLOCK_N]`; today's DSL shape checker compares the syntax
  of the dimension terms and therefore rejects the body.
- `rotary_transform` / `rotary_transform_ops` need an early-return statement
  in the core AST and operational semantics before the Python control flow can
  be transcribed literally.
- The attention/context/flash kernels need full streaming-softmax loop
  surfaces, including helper-call modeling where the Python kernel delegates
  to a separate `@triton.jit` helper. The current proof slices intentionally
  start from precomputed accumulator/output tiles.
