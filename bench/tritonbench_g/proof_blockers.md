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
The broad #147 quantization bucket has been split. Real-to-int8 cast semantics
track under #154 and now have an executable DSL/AST semantics path. The
now-discharged #158 quantization scale/value coupling rows connect the checked
int8, quantize-copy-kv, grouped quantize-kv-copy, and quantize-kv-transform
outputs directly to their full Python-shape surfaces. Rows whose local blocker
is primarily attention, matmul,
recurrence, reduction, or explicit blocked-summary work track under the
corresponding family issue.
The broad #150 recurrent/cumsum bucket has been split by mechanism:
the now-discharged `chunk-cumsum-carry-fold` (#185) tracks chunk cumsum
summaries now connected to full scalar, vector, and chunked forward surfaces,
the now-discharged
`decay-cumsum-scan-fold` (#186) tracks `decay_cumsum.py` summaries now
connected to full prepare, forward cumsum, and backward global-cumsum surfaces,
the now-discharged `recurrent-state-loop-fold` (#187) connects chunk-gate,
HGRN, and RWKV recurrent state summaries to their full producer surfaces,
the now-discharged `gla-output-tile-producer` (#188) tracks
`chunk_gla_simple.py` summaries whose proof used to start from a precomputed
output tile, and
the now-discharged `reverse-cumsum-directional-scan` (#94) tracks reverse
cumsum direction semantics. All reversed-cumsum case rows now connect the
checked surfaces to full-surface output readbacks.
The broad #149 attention/softmax bucket has been split as well:
`attention-final-store-lift` tracks summaries that still connect a faithful
surface to a final-store/proof-oriented writeback from precomputed Acc/Score/Prob
tiles under #161. The #162 attention recurrence rows are now discharged. The `triton_attention.py` forward
row now connects `Out`, `L`, and `M` readbacks to the full forward surface.
Flash-attention cases 1 and 2 now connect `O` and `L` readbacks to their full
causal and non-causal forward surfaces.
All mixed-sparse attention cases now connect `Out` readbacks to the full
mixed-sparse forward surface.
The lightning-attention row now connects `Out`/`DQ`/`DK`/`DV` readbacks to
launched full surfaces.
The token-attention-reduction rows
are now discharged: the LLaMA and Bloom token-softmax case-1 rows, the
softmax-reduceV row, and the reduce-V, Mistral, and LLaMA2 token-attention
case-1 rows all connect the checked probability/output directly to their full
Python-shape surfaces.
The broad #151 reduction/layernorm aggregation bucket has been split into
narrower follow-ups: the now-discharged
`chunk-delta-forward-recurrence-store` (#190) connects the two chunk-delta
forward summaries to the full producer surface for `h`, `v_new`, and optional
`final_state`, while the now-discharged
`layernorm-backward-residual-recompute-aggregation` (#191) covers the LayerNorm
backward residual/recompute summaries that used to compose row-level C1/C2
reductions, DX/Y writebacks, and partial DW/DB slices.
The #161 final-store bucket has been split again into kernel-specific producer
obligations: the now-discharged
`attention-fwd-triton1-bo-bhpre-producers` (#165),
the now-discharged `dense-attention-acc-store` (#166), whose remaining
Q/K/V streaming-softmax `Acc`/`L` producer proof is now discharged by
`dense-attention-online-softmax-recurrence` (#199),
the now-discharged `context-attention-mistral-sliding-window-acc-store` (#167),
`context-attention-nopad-varlen-acc-store` (#167), and
`flash-decode-normalized-vector-store` (#168).
The #168 flash-decode normalized-vector bucket has been split into
kernel-specific stage2 recurrence obligations:
the now-discharged `flash-decode-llama-stage2-normalization` (#171) and
`flash-decode-phi-stage2-normalization` (#172).
The #171 LLaMA stage2 normalization bucket now connects loop-produced `Acc` and
`SumExp` values to the final `O` writeback as a full-value candidate, and the
#181 LLaMA running-max recurrence step over `Mid_O_LogExpSum` is now also a
full-value candidate.
The #172 Phi stage2 recurrence bucket has been split into narrower
obligations: the now-discharged `flash-decode-phi-running-max-recurrence`
(#175), the now-discharged `flash-decode-phi-masked-accumulator-recurrence`
(#176), and the now-discharged `flash-decode-phi-normalization-store` (#177).
The #175 path states and proves the Python test-shape running `max_logic`
recurrence, #176 carries the Python test-shape masked `Mid_O` load through the
`AccOut` and `SumExpOut` recurrence step, and #177 connects the Python
test-shape `Acc` and `SumExp` outputs to the final masked `Out` writeback; all
three are full-value candidates in `proof_gap_manifest.tsv`.
The broad #148 matmul/dot bucket is now discharged: GEMV, BMM, dequantization,
IV-dependent matmul, plain matmul, activation-tail, and TMA summaries connect
their checked outputs directly to full Python-shape surfaces and are
full-value candidates in `proof_gap_manifest.tsv`.
The #191 layer-norm backward residual/recompute paths now connect the checked
Python test-shape outputs to the full backward surface for DX, recomputed Y,
and partial DW/DB, so the affected `layer_norm_ops.py` summaries are
full-value candidates.
The broad #153 rotary/cache bucket is now discharged. The rotary 2D tile rows,
forward rope-transform row, and backward rope-transform row connect their
checked surfaces to full-surface output readbacks and are full-value candidates
in `proof_gap_manifest.tsv`.
The explicit blocked-output summaries formerly tracked by the broad #152
`semantic-blocker` bucket are all quantization `llrint` / int8-cast blockers
and now track under the open #154 `fixed-width-int8-cast-semantics` family.

## Translation-Surface Blockers

A port whose Lean `triton { }` surface deliberately deviates from a literal
transcription of the upstream Python kernel body declares that with an
explicit `Translation-surface blocker:` line in its module preamble.
`bench/audit_tritonbench_g.sh` keys the textual py↔lean surface-scan
exemptions on that marker only, and requires every marker to be registered
here and in `completion_audit.md`. The current registered set — each entry is
a documented, deliberate surface deviation, not an unproven correctness gap
unless stated:

- `attn_fwd_triton` — the `_attn_fwd_inner` helper JIT (both call sites) is
  inlined into the single streaming loop; the hard-coded `tl.arange(0, 128)` /
  `< 96` head constants are generalized to the `BLOCK_DMODEL` / `HEAD_ACTIVE`
  binders (the Python literals are the `128`/`96` instantiation of the
  dimension-general top theorem).
- `attn_fwd_causal` — same helper inlining and `128`/`96` → `BLOCK_DMODEL` /
  `HEAD_ACTIVE` generalization as `attn_fwd_triton`.
- `attention_fwd_triton2` — same helper inlining and `128`/`96` →
  `BLOCK_DMODEL` / `HEAD_ACTIVE` generalization.
- `attention_fwd_triton3` — the `_attn_fwd_inner` helper JIT is inlined into
  the single streaming loop (its loads/`tl.where`/`tl.advance` appear in the
  Lean surface but not in the Python `_attn_fwd` body).
- `sgmv_expand_slice` — proven path is `EVEN_K` loads with
  `ADD_INPUTS = false`, `CAST_TYPE = false`; those constexpr parameters and
  branches are dropped; the `pid → (pid_m, pid_n)` CTA linearization is
  supplied via program-id axes (trusted host boundary);
  `tl.max_contiguous`/`tl.multiple_of` hints erased.
- `matmul_kernel` — in-body constants (`M = N = K = 4096`, stride literals)
  and the `tl.cdiv(K, BLOCK_SIZE_K)` trip count are supplied as antiquoted
  binders (`K = BLOCK_SIZE_K · numKBlocks`).
- `matmul_triton_autotune` — the `tl.cdiv(K, BLOCK_SIZE_K)` trip count is the
  antiquoted `numKBlocks` binder; the K-loop counter is spelled `kk`.
- `bmm_chunk_bwd` — the in-body `chunk_size_limit = min(chunk_size, seqlen -
  pid_c·chunk_size)` is supplied as the precomputed `chunk_size_limit`
  parameter (`seqlen` is not a surface binder).
- `iv_dependent_matmul` — only the canonical `type == "pre_load"` scheduling
  mode is mechanized; the string constexpr `type` and the four other mode
  branches are described informally, not transcribed.
- `matmul_tma` — the `OUTPUT_F16` constexpr branch is split into two Lean
  surfaces (f32 and f16), so that parameter/branch/cast is absent from the
  first surface.
- `softmax_flaggems` — value correctness targets the `ONE_TILE_PER_CTA = true`
  single-tile specializations; the multi-tile fallback branches (pointer `+=`
  advances, online recurrences) are not transcribed. This one is a genuine
  partial-coverage scope restriction, not just a notational rewrite.
- `relu_strided_buffer` — the `one_tile_per_cta` constexpr branch is split
  into two Lean surfaces (one-tile and grid-stride); the `relu_forward`
  helper JIT (`tl.where(x > 0, x, 0)`) is inlined at its call site; the
  rank-1 stride-order constexprs (`in0_stride_order0 = out0_stride_order0
  = 0`, their only rank-1 value) are instantiated in the
  `boundary_check=(...)` / `order=(...)` slots. Both branches are fully
  value-verified (including the multi-iteration grid-stride loop).
- `pow_scalar_tensor` — the `one_tile_per_cta` constexpr branch is split
  into two Lean surfaces (one-tile and grid-stride); the
  `pow_func_scalar_tensor` helper JIT (`_pow(x.to(tl.float32), exponent)`)
  is inlined at its single call site per branch as
  `tl.extra.cuda.libdevice.pow($((val0 : ℝ)).to(tl.float32), in0)`; the
  rank-1 stride-order constexprs (`= 0`) are instantiated in the
  `boundary_check=(...)` / `order=(...)` slots. Both branches are fully
  value-verified.
- `fused_recurrent_delta` — the Python conditional-expression pointer
  increments (`p_beta += V if IS_HEADWISE_BETA else 1`, `p_dbeta` analogue)
  and conditional definitions are spelled as `if IS_HEADWISE_BETA { … } else
  { … }` constexpr gates, one statement per arm (adds one `if` to the
  control-flow count and repeats the `p_beta`/`p_dbeta` lhs per arm);
  `do` → `do_` (Lean keyword), `_` loop vars → `_i`; scalar-beta `… + T - 1`
  pointer inits parenthesized into the ℕ domain.
- `fused_recurrent_retention` — the four dead stride parameters in the
  Python kernel signatures (`s_qk_t`, `s_qk_d`, `s_vo_t`, `s_vo_d` — never
  used in either kernel body) are omitted from the Lean surface binders
  (`fused_rwkv6_kernel`/`fused_recurrent_hgrn` precedent); both bodies are
  otherwise transcribed verbatim.
- `triton_linear_activation` — the `K_LOAD_MASK_NEEDED` constexpr branch is
  specialized to its `@triton.heuristics` value `True` (exact-multiple `K`;
  `SPLIT_K = 1` is the only launched value), so the descending
  `for k in range(K, 0, -BLOCK_K)` is the ascending trip-count loop with
  `K = BLOCK_K · numKBlocks` as the antiquoted binder (loop variable dead in
  the unmasked arm). The `tanh`/`relu`/`gelu`/`fast_gelu` helper JITs are
  inlined at call sites; module constants `sqrt2`/`sqrt2pi` are antiquoted
  reals; unused `CACHE_KEY_M/N/K` and `SPLIT_K` parameters dropped; the
  string constexpr `ACTIVATION` is a Lean `String` parameter with all four
  `==` gates transcribed verbatim.
- `rbe_triton_transform` — the helper `get_freq_multi_tokens` JIT is inlined
  at its single call site; the helper-local `DIM: tl.constexpr = 128` is
  generalized to a `DIM : Nat` binder (`NB_TOKENS` instantiated to
  `BLOCK_SIZE_M` as at the call site); Python's dtype/shape-changing `freqs`
  rebindings get fresh names (`freqs`/`freqs_f`/`freqs_p`/`freqs_mn`);
  `.to(tl.float32)` on the int tile is spelled `tl.toReal`; the implicit
  int→float promotion of `tl.arange(0, NB_TOKENS) + starting_idx` is the
  explicit `tks`/`tks_f` binding pair. Fully value-verified on the full 2-D
  tile.
- `chunk_gla_fwd` — the file's four `A`-builder kernels (including its first,
  `chunk_gla_fwd_A_kernel_intra_sub_inter`) each open with bare early
  `return`s (`if i_t * BT + i_i * BC >= T: return`, `if i_i <= i_j: return`);
  `Stmt` has no early exit, and the `bgmv_shrink_kernel` guard-wrap
  workaround is a per-kernel port of its own, one per builder. The port
  covers the fifth kernel, `chunk_gla_fwd_kernel_o`, with a full multi-block
  K-loop headline; the four builders that fill the `A` region are the
  trusted boundary.
- `bgmv_shrink_kernel` — the Python early `return` on the signed sentinel
  (`lora_index == -1`) is an `if lora_index != -1 { ... }` guard in the
  faithful surface (DSL has no early exit); the guard-false path is proven
  write-free, and the proof surfaces verify the active body with a
  `Region .nat`-typed `lora_indices` (sentinel skip = trusted host boundary,
  `lora_expand_gemv`/`sgmv_expand_slice` precedent). The constexpr tail
  `if SPLIT_K == 1: tl.store else: tl.atomic_add` is split into two proof
  surfaces (`matmul_tma` precedent), gated by `SPLIT_K_ONE : Bool` in the
  faithful surface. `tl.max_contiguous` erased to its value argument; the
  upstream unused `xk_stride` parameter dropped.
- `layer_norm_welfold` — mechanical spellings only (no helper inlining, no
  scope restriction): `tl.full` positional dtype spelled `dtype=tl.float32`;
  `tl.broadcast_to` via the tuple `tl.broadcast` form; the fused
  `tmpN = tl.sum(_tmpN, 1)[:, None]` split into two statements;
  `tl.store(..., None)` as the unmasked `tl.store`; `libdevice.rsqrt`
  spelled `tl.rsqrt`; register casts parenthesized `(tmpM).to(tl.float32)`.
- `fused_layernorm_triton` — the inductor `triton_helpers.welford_reduce` /
  `triton_helpers.welford(…, 1)` helper calls are inlined as their exact-ℝ
  closed forms (general-branch Welford update; combine as the
  `(Σw, Σ(w·m)/Σw, Σ(m2 + w·m²) − Σw·mean²)` moment identity);
  `tl.broadcast_to` is spelled via the tuple `tl.broadcast` form;
  `tl.store(..., None)` is the unmasked `tl.store`; `libdevice.rsqrt` is
  spelled `tl.rsqrt` (rmsnorm_triton / layer_norm_liger precedent); register
  casts are parenthesized `(tmpN).to(tl.float32)`.

### Required VeriTile surface extensions

The attention helper-call surfaces are inlined into the outer-kernel streaming
loops for translation coverage (see the `attn_fwd_*` / `attention_fwd_*`
entries above). Future work may add executable semantics for projecting
helper calls as calls, but that is a surface-notation concern, not a proof
gap.
