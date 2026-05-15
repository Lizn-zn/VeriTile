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

- `attention_score`: the Python test surface fixes `_BLOCK_M == _BLOCK_N`,
  while the general kernel shape uses separate M/N parameters. The Lean surface
  currently specializes this equality to keep the `tl.zeros([_BLOCK_M])` /
  `tl.sum(..., axis=0)` shape flow typable. Closing this requires constraint
  propagation for constexpr shape equalities.

- `bmm_chunk_fwd`: Python uses an early `return` under the causal guard; the
  Lean surface represents the active body with an explicit guard. Closing this
  requires a first-class early-return/terminated-block surface, or an audit rule
  that treats the guard-normal form as a proven-preserving translation.

- `iv_dependent_matmul`: Python dispatches five scheduling modes through a
  string constexpr `type`; Lean currently exposes the modes as separate
  surfaces. Closing this requires string-valued constexpr branching in the DSL
  surface, or a review rule that accepts a complete family of mode-specific
  surfaces as the faithful representation.

- `kcache_copy_triton`: the full Python path includes signed negative
  `cur_token_shift` arithmetic before selecting the cache slot. The proof
  surface starts after that arithmetic. Closing this requires signed pointer
  offset flow for the full path.

- `layer_norm_ops`: the Python constexpr branches make `mean` live only in the
  non-RMS path, while the current Lean surface needs extra initialization to
  keep later conditional expressions in scope. Closing this requires
  path-sensitive constexpr environment tracking.

- `rotary_transform` and `rotary_transform_ops`: the Python kernels combine
  varlen/non-varlen, tensor/scalar seqlen offsets, interleaved/non-interleaved,
  and conjugate/non-conjugate paths. Current Lean surfaces cover the
  non-varlen, non-interleaved, non-conjugate path. Closing this requires
  path-sensitive parameter refinement for `SEQLEN_OFFSETS`, `seqlen`, and
  `rotary_dim`-derived half ranges.
