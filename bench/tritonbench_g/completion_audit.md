# TritonBench-G Completion Audit

This document separates the port inventory, the scope of each headline theorem,
and the checks required for a successful audit. Passing the mechanical gates
is not evidence that every upstream kernel has a full correctness proof.
Faithful translation still requires review against
[`review_criteria.md`](./review_criteria.md).

## Current Source Inventory

The 2026-10-11 statement-scope inventory records 173 Python/Lean port pairs and
503 headline (`specification`) declarations in 233 specification files. Another 11 of the 184 kernel work
directories are README-only scaffolds and are not counted as implemented ports.
The `_spec-sheets` metadata directory is not a kernel work directory.

[`coverage_review.json`](./coverage_review.json) records the scope, assumptions,
numeric model, and Python function links for each headline. Its Lean and Python
source fingerprints are checked by
[`bench/coverage_review.py`](../coverage_review.py); the reviewed categories
are reflected in [`proof_gap_manifest.tsv`](./proof_gap_manifest.tsv).

| Coverage level | Headlines | What the label records |
|---|---:|---|
| `full_value_candidate` | 254 | A candidate full-value contract within its stated model and assumptions. |
| `specialization` | 218 | A contract restricted to the shapes, branches, parameters, or execution scope stated in the review. |
| `precomputed_input_slice` | 26 | A contract whose inputs include prepared intermediates; it does not prove the omitted producer. |
| `pre_rounding_slice` | 3 | A real-valued result before the faithful fixed-width rounding/cast result. |
| `blocked_summary` | 2 | An explicitly blocked summary. |
| **Total** | **503** | **Across 233 specification files in 173 ports.** |

These categories describe the actual headline statements. A full producer
surface or projection elsewhere in a file does not automatically strengthen a
headline that consumes prepared values. A `full_value_candidate` label is not
an independent certification of the Python translation or GPU execution.
Axiom cleanliness and successful comparator replay are separate checks.
Historical issue links identify blocker families; they do not assert that an
issue is still open or that closing it completed the proof.

See [`proof_blockers.md`](./proof_blockers.md) for the remaining scope and frame
limitations, and the
[browsable coverage table](https://lizn-zn.github.io/VeriTile/proofs/coverage/)
for the individual theorem statements and source evidence.

## Audit Gates and CI Evidence

[`bench/audit_tritonbench_g.sh`](../audit_tritonbench_g.sh) enforces the gates
below. This table describes what is checked; it does not certify that an
in-progress run or a later revision has passed.

| Gate | Check and interpretation |
|---|---|
| Port inventory | Match Python/Lean file counts; reject compiled-port READMEs that still advertise TODO status. Scaffolds remain outside the implemented port set. |
| Port elaboration and proof replay | The final `bench/audit_trust.sh` gate elaborates the standalone corpus with the Lean trust/statement checks and requires official comparator export/replay. The aggregate audit also comparator-checks the proven library manifest targets. A separate duplicate compilation pass is not run. |
| Faithful-translation patterns | Check dtype-load additions, `keep_dims`, missing `+=` updates and normalized pointer lhs, `rsqrt`, Lean-only `tl.where`, `tl.*(...)` call set/order, control-flow counts, statement lhs order, and cast/scope notes. These scans do not establish arbitrary arithmetic equivalence. |
| Translation exceptions | Require explicit `Translation-surface blocker:` preamble markers and matching documentation. The 47 registered ports are listed below; a documented exception is not a discharged proof obligation. |
| Correctness surfaces | Require a named correctness surface or an explicitly registered `Correctness-surface blocker:`. The 10 registered ports and their 16 frame-free headlines are documented in `proof_blockers.md`; passing this gate does not put every headline on a named surface. |
| Statement-scope inventory | Check every headline against the manifest and the explicit review, including Lean/Python fingerprints and Python function links. |
| Placeholder proofs | Scan comment-stripped Lean for `sorry`, `admit`, `True := by`, and whole-proof `trivial`; elaborated trust checks and comparator replay provide additional proof checks. |
| Explicit algorithm blockers | Reject `hAlg` blockers. Their absence does not discharge specialization, prepared-input, rounding, frame, or translation limitations. |

For a standalone port check, [`bench/check_ports.sh`](../check_ports.sh) also
requires compilation and official comparator replay. `lake build` alone does
not cover the standalone corpus or all of these gates.

The website's
[recorded completed corpus audit](../../site/src/lib/coverage-ci.json), when
present, pins the successful run URL, commit, and completion time. A `null`
record means no completed public-repository audit has been recorded yet.
Check that commit and its
workflow when using the record as release evidence: an earlier success does
not establish that later changes passed newly added gates. Only a completed,
successful full run can be recorded with
[`scripts/record_coverage_ci.py`](../../scripts/record_coverage_ci.py).
Queued, running, cancelled, or skipped runs are not successful audit evidence.
The [Bench audit workflow](https://github.com/Lizn-zn/VeriTile/actions/workflows/bench-audit.yml)
provides the current run status and logs.

## Remaining Blockers

No explicit TritonBench-G `hAlg` blocker remains. Forty-seven ports carry an
explicit `Translation-surface blocker:` preamble marker — a documented,
deliberate deviation of the Lean `triton { }` surface from a literal
transcription of the upstream Python body (helper-JIT inlining, constexpr-path
specialization, antiquoted in-body constants); see `proof_blockers.md` for the
per-port descriptions. `bench/audit_tritonbench_g.sh` keys the textual
py↔lean surface-scan exemptions on that marker only and requires this list to
match the active marker set exactly.

The current documented blocker set is:

- `bgmv_expand_slice` — dtype and cast specialization is staged by a source
  macro, the implicit output conversion is explicit, and early return becomes
  a sentinel guard. Full-source correspondence and precision/configuration
  regressions accompany the real IO and admitted-atom equivalence proofs.
- `attention_score` — fp16 Q/K/Out and fp32 M storage are explicit; the
  source's output element-type alias resolves to `tl.float16` for the supplied
  dtype configuration. Explicit region lists and the terminal cast are
  checked with dot/load precision and signed-window regressions.
- `attention_kernel` — fp16/bf16 storage specializations, region types and
  implicit bias-scalar conversions are explicit. The source regression checks
  storage bias precision, default dot outputs, casts and both B0 windows.
- `attention_fwd_triton1` — the six fp32 dtype-alias casts are spelled
  `tl.float32` to retain FP operations; `p_h.dtype.element_ty`, `b_q.dtype` and
  `p_o.dtype.element_ty` all select fp32 in the supplied tests. The README
  records the reviewed mapping; the lexical gate cannot establish it.
- `attention_forward_triton` — `_attn_fwd_inner` is fully inlined; fp16 Q/K/V/Out
  and fp32 scale storage are explicit, 128/96 head constants are symbolic, and
  the output element-type alias resolves to `tl.float16`. The source loop,
  typed dots, masks and store are reviewed and checked in Lean; the generic
  outer-call text scan does not align the inlined representation.
- `parallel_attention` — `parallel_rebased_fwd_kernel` + the
  `_parallel_rebased_bwd_dkv` helper ported; the helper's scalar arguments
  become universally-quantified binders (no cross-JIT call surface; marker
  registered in `proof_blockers.md`); its trailing bare `return` dropped;
  its descending `-BTS` loop as the ascending `hi − j·BTS` change of
  variable with a `cdiv` trip count; the backward shell and
  `_parallel_rebased_bwd_dq` are the trusted boundary.
- `bmm_optimized` — `bmm_kernel` now retains all eight constexpr mask arms,
  both CTA-reorder branches, the shortened final group, actual runtime grid
  dimensions and the three batch-pointer updates. The macro specializes
  optional masks, splits tuple assignment and makes range start/step explicit.
  Common fp32/fp16/bf16 storage and `allow_tf32=false` remain; implicit store
  conversion is explicit. Source masked loads still omit `other`, and the real
  product contract explicitly needs zero unspecified values. See the checked
  counterexample in `Proofs/UndefinedTail.lean`. Final replay status is tracked
  in `verification_round.md`; autotuning and grid composition are separate.
- `parallel_retention_attention` — `parallel_retention_fwd_kernel` + the
  `_parallel_retention_bwd_dkv` helper ported; the helper's scalar arguments
  become universally-quantified binders (no cross-JIT call surface; marker
  registered in `proof_blockers.md`); its trailing bare `return` dropped;
  its descending `-BTS` loop as the ascending `hi − j·BTS` change of
  variable with a `cdiv` trip count; the diagonal unary-minus decay index
  respelled as a subtraction under the causal `tl.where` mask; int→float
  promotions spelled `tl.toReal(...)`; the backward shell and
  `_parallel_retention_bwd_dq` are the trusted boundary.
- `cache_transform_triton` — both complete gathers specialize the nine independent
  cosine/sine fp32/fp16/bf16 storage pairs because region dtype annotations are
  syntax-level. Explicit signatures retain both inputs, signed metadata and
  both outputs. Original masks, omitted `other`, the full decoding tile and
  all strides remain; the only candidate edit commutes an exact index product.
- `llama_ff_triton` — `ff_llama` ported at the `USE_FP8 = False` arm (the
  fp8 path is a `bitcast=True` bit-reinterpretation, an ℝ-model limit;
  marker registered in `proof_blockers.md`); RMSNorm-fused SwiGLU dual GEMM
  proven as one exec closed form (3-accumulator/4-pointer loop invariant);
  `tl.cdiv` trip count as the antiquoted `numKBlocks`; implicit store fp16
  cast spelled explicitly; host launch and per-dtype dispatch are the
  trusted boundary.
- `rms_matmul_rbe` — both kernels ported: the GEMM as a twin mirror of
  the `rms_rbe_matmul` port, and `rms_matmul_rbe_qkv` with its three
  cross-JIT calls inlined (markers registered in `proof_blockers.md`);
  the QKV headline is one bundled specification (a conjunction of three
  `Realizes` faces per MAIN_THEOREM_CONVENTIONS §4) proven via one
  region-parameterized pass bundle applied thrice with frame transport
  (10 region-distinctness hypotheses).
- `rms_rbe_matmul` — target JIT `rms_matmul_rbe` (the file's second
  kernel) ported at the `USE_FP8 = False` arm; `rbe_triton` is dead
  in-file (never launched; calls an undefined helper) and unmodeled;
  unused params dropped; batched RMSNorm-fused GEMM proven as one exec
  closed form (2-accumulator/3-pointer invariant + batch-axis register
  clause); markers registered in `proof_blockers.md`.
- `triton_matmul` — `matmul_kernel` ported with both constexpr epilogue
  arms as twin surfaces (fp16 + fp8e4nv; marker registered in
  `proof_blockers.md`), one od-generic proof stack instantiated per arm
  (exec closed form + `⊨[R]` streaming face each); `tl.cdiv` trip count as
  the antiquoted `numKBlocks`; K-loop counter spelled `kk`;
  `launch_metadata` hook and per-dtype host config table are the trusted
  boundary.
- `attention_llama` — full `_fwd_kernel` sources for fp16 and int8/E5M2
  K/V, retaining both causal branches and source masks. The textual-scan
  exception covers constexpr storage specialization, explicit dtypes/casts
  and positional zero-fill written as `other=`. No FP8 arm is omitted.
  Correct.lean proves both versions' independent natural-exp formulas and
  complete flat IO, including original byte-cell inputs/frame, under positive
  block/count, whole streamed span, nonempty causal visibility, output
  injectivity and FP8 buffer-separation conditions. FPEquiv.lean proves the
  complete carry-first rewrite with add_commute and finite actual operands in
  every reached iteration. Both keep arbitrary source strides. Partial-tail
  conventional-attention formulas, empty visible-row formulas, host launch
  composition and hardware execution remain outside the real contract.
- `int4_matmul` — `matmul_kernel` ported at the `SPLIT_K = 1` store arm
  (markers registered in `proof_blockers.md`); GPTQ int4-dequantize GEMM
  proven as one exec closed form mirroring the `matmul_dequantize_int4`
  twin (packed `Region .nat` channels, per-lane group rows); the
  sub-first dequant runs on the `.int` channel via explicit `tl.cast`
  and promotes through `Op.intToReal` (first consumer of the
  signed-promotion lever); `tl.cdiv` trip count as the antiquoted
  `numKBlocks`; autotune sweep and host launch are the trusted boundary.
- `int8_dequant_matmul` — `_int8_matmul_rowwise_dequantize` ported at the
  `SPLIT_K = 1` store arm and the `EVEN_K = True` load arm (markers
  registered in `proof_blockers.md`); int8×int8→int32 GEMM with rowwise
  dequant epilogue, the first `Op.dotInt` consumer — the accumulator
  stays on the ℤ channel through the whole K loop and promotes to ℝ once
  via `Op.intToReal` in the epilogue; `has_bias` a genuine `Bool` with
  both arms under one headline; the faithful `.to(tl.float16)` casts make
  the masked store `.fp16`-typed, read back at the MemCell level; the
  autotune sweep and host launch are the trusted boundary.
- `int_scaled_matmul` — both JIT kernels launched and modeled in py
  order (markers registered in `proof_blockers.md`): the block-pointer
  int8 GEMM (first `.int`-channel block-ptr port — element dtype
  inherited from the typed base region via the port's inference rider,
  private `.int` boundary-check load lemmas, `tl.advance` walk,
  `MemCell.of .int` readback; step-form K loop spelled directly under
  `0 < BK` only) and
  the pointer GEMM with the inductor store suffix (descending loop as
  ascending substitution, `EVEN_K` a genuine `Bool` with both arms,
  `tl.broadcast_to` — whose syntax rider landed with this port — flat
  `col + N·row` store of the `Op.intToReal`-promoted `acc · s1(row)`
  read back as exact-`.real` cells); the host launches are the trusted
  boundary.
- `int8_matmul_kernel` — `matmul_kernel` ported in full (markers
  registered in `proof_blockers.md`); a pure-integer 2-bit-packed-weight
  GEMM proven as one fully dimension-general exec closed form — the
  first **nested-loop** `Op.dotInt` consumer (static field loop ×
  dynamic K loop; the A pointer advances continuously across both, the
  B pointer rebinds per field); `tl.static_assert` spelled faithfully
  (second consumer) and its divisibility is the headline's only K
  hypothesis; `.int`-typed store cells read back over launch-state
  accessors; the autotune sweep and host launch are the trusted
  boundary.
- `int8_matmul_quantization` — both JIT kernels launched and modeled in
  py order (markers registered in `proof_blockers.md`): the per-row
  float→int8 quantizer (two sequential K loops — a 0-seeded streaming
  abs-max pass, then the `Op.castRealToInt8` quantize-and-store pass;
  one exec headline with two MemCell readback conjuncts, int8 cells +
  fp16 scale cells) and the int8×int8→int32 GEMM (second `Op.dotInt`
  consumer, remainder-masked K loads at the honestly-ragged ceil-form
  `numKBlocks`, `SPLIT_K = 1` store arm, per-row × per-col scale
  epilogue promoted through `Op.intToReal`, fp16 MemCell readback);
  the autotune sweeps and host launches are the trusted boundary.
- `matmul_dequant_int4` — the target JIT is the file's second kernel,
  `dequantize_kernel`, the only one launched (the first, `matmul4_kernel`,
  is dead code byte-identical to the `matmul_dequantize_int4` port's
  kernel — not modeled; markers registered in `proof_blockers.md`);
  straight-line masked int4 dequantization proven as one fully
  dimension-general exec closed form whose only hypothesis is store-map
  injectivity; the signed nibble difference runs on the `.int` channel via
  explicit `tl.cast` and promotes through `Op.intToReal` (second consumer
  of the signed-promotion lever); the masked loads keep `other=0.0`
  faithfully; the autotune sweep and host launch are the trusted boundary.
- `matmul_dequantize` — all three JIT kernels launched and modeled in py
  order (markers registered in `proof_blockers.md`); three disjoint proof
  stacks mirroring the `matmul_dequantize_int4` / `int4_matmul` /
  `matmul_dequant_int4` twins; `matmul4_kernel` proven with `NO_GROUPS`
  as a genuine `Bool` parameter (both arms), `matmul_kernel` at the
  `SPLIT_K = 1` store arm with a MemCell-level `.fp16` readback (its
  source keeps explicit `.to(tl.float16)` casts), `dequantize_kernel` a
  1:1 mirror of the byte-identical 168th port; the three host launches
  are the trusted boundary.
- `attn_fwd_triton` — both helper JIT calls are inlined with independent
  pointer copies; the supplied bf16 Q/K/V/Out and fp32 scales are explicit.
  The value dot preserves its V-to-fp16 cast and fp16 result. Both defined
  outer stages and the finite additive penalty are retained. Head constants
  `128`/`96` become `BLOCK_DMODEL`/`HEAD_ACTIVE`; undefined helper stages are
  excluded. The changed historical model lives in `Proofs/LegacyExactMask`.
- `attn_fwd_causal` — both JIT helper calls are inlined with separate local
  pointers; fp16 Q/K/V/Out and fp32 scales are explicit. The `128`/`96`
  constants are generalized to `BLOCK_DMODEL`/`HEAD_ACTIVE`. Both defined
  outer stages and the finite additive penalty are retained; stages leaving
  the Python helper bounds undefined are excluded. The changed historical
  exact-mask model is isolated under `Proofs/LegacyExactMask.lean`.
- `attention_fwd_triton2` — `_attn_fwd_inner` inlined; `128`/`96` generalized.
  fp16 Q/K/V and fp32 scale storage are explicit; the output element-type alias
  resolves to `tl.bfloat16`. The explicit V-to-fp16 cast, fp16 dot result and
  bf16 terminal cast/store are retained and checked in the source regression.
- `attention_fwd_triton3` — the helper JIT is inlined and fp16 Q/K/V/Out,
  fp32 M/L and the source-dependent probability/output casts are explicit.
  The current shared source retains signed windows and bounds, KV tail checks,
  initial pointer advance, NaN alpha repair and all stores. Both real IO
  contracts cover independent strides, INIT/END, grouped heads and checked
  partial KV tails. The FP candidate changes only END normalization using
  div_mul_rcp under actual finite/nonzero intermediate-value guards.
  `Proofs/LegacySource.lean` preserves the altered historical transcription;
  its claims are excluded from the public headline inventory. The checked
  source-mask counterexample remains in `Proofs/SourceReview.lean`.
- `sgmv_expand_slice` — `EVEN_K`/`ADD_INPUTS=false`/`CAST_TYPE=false` path;
  CTA linearization via program-id axes; layout hints erased.
- `matmul_kernel` — in-body `4096` constants and `tl.cdiv` trip count as
  antiquoted binders.
- `matmul_triton_autotune` — `tl.cdiv` trip count as antiquoted `numKBlocks`;
  loop counter spelled `kk`.
- `bmm_chunk_bwd` — host-selected A/Dout storage and dot dtypes are
  specialized explicitly. The complete source now computes its own chunk
  length and retains all residual arms, partial tails and output casts;
  the former precomputed-length specialization lives in `Proofs/Legacy.lean`.
- `bmm_chunk_fwd` — host-selected A/B storage and dot dtypes are specialized
  explicitly. The nested causal early return is written as its complementary
  active-tile guard; skipped tiles perform no loads/stores. The actual
  in-body length, partial K blocks, signed sequence filter, original load
  storage and output conversion remain in the complete source.
- `iv_dependent_matmul` — only the `type == "pre_load"` mode mechanized.
- `matmul_tma` — `OUTPUT_F16` branch split into two surfaces.
- `softmax_flaggems` — `ONE_TILE_PER_CTA = true` single-tile specializations
  only (genuine partial-coverage scope restriction).
- `relu_strided_buffer` — `one_tile_per_cta` branch split into two surfaces;
  `relu_forward` helper inlined; rank-1 stride-order constexprs (= 0)
  instantiated in `boundary_check`/`order`.
- `pow_scalar_tensor` — `one_tile_per_cta` branch split into two surfaces;
  `pow_func_scalar_tensor` helper inlined; rank-1 stride-order constexprs
  (= 0) instantiated in `boundary_check`/`order`.
- `fused_recurrent_delta` — ternary pointer increments spelled as
  `IS_HEADWISE_BETA` constexpr if/else gates (one statement per arm);
  `do` → `do_`; `_` loop vars → `_i`; scalar-beta `… + T - 1` pointer inits
  ℕ-parenthesized.
- `fused_recurrent_retention` — four dead Python signature stride params
  (`s_qk_t`/`s_qk_d`/`s_vo_t`/`s_vo_d`) omitted from the surface binders.
- `triton_linear_activation` — `K_LOAD_MASK_NEEDED=True` heuristics arm
  specialized (descending K-loop as antiquoted ascending trip count);
  activation helper JITs inlined; module constants antiquoted; unused
  `CACHE_KEY_*`/`SPLIT_K` dropped; `ACTIVATION` as a `String` parameter.
- `rbe_triton_transform` — `get_freq_multi_tokens` helper inlined; helper
  `DIM = 128` → `DIM : Nat` binder; dtype/shape-changing `freqs` rebinds
  renamed; `.to(tl.float32)` spelled `tl.toReal`; explicit int→float
  promotion bindings.
- `chunked_cumsum_fwd` — bias storage and finite/positive-infinite upper
  bounds select explicit typed source templates. Quoted clamp/multiply
  statements retain the complete preprocessing, masks, stores and scan.
  The default infinite minimum remains opaque in FP semantics. The public
  real contract states the host-valid remaining-length condition; both full
  real and FP paths are checked. Independent acceptance is tracked in
  `verification_round.md`.
- `chunk_retention_ops` — all four typed sources retain signed `tl.arange`
  through an explicit `.to(tl.int32)` identity cast. The backward-state loop
  uses NT-1-j, preserving per-chunk stores, scaled queries, weighted incoming
  gradients and full-BT carry decay. All tile dimensions remain independent;
  the empty loop is valid. Complete typed equalities connect the other three
  source pairs to the shared proofs in `chunk_retention`.
- `chunk_retention` — all four complete typed kernels and their candidates
  retain decay intrinsics, casts, masks and stores. Signed `tl.arange` is
  explicit as `.to(tl.int32)`, adding an identity cast to the call sequence.
  Backward-state's descending loop is an ascending change of variable;
  `NT > 0` is required for its original post-loop-index use. Its square dot
  shapes, unused inputs, dead history load, final-only decay and sole final
  store are preserved. The real contracts expose zero unspecified values,
  access bounds and a full tagged-cell frame; see the case README.
- `block_sparse_attn` — complete typed source specialized for fp32/fp16/bf16
  storage, preserving signed CSR metadata and all mask/channel branches.
  Q-storage probability casts and implicit output conversions are explicit;
  `static_print` has a fixed debug payload. The candidate commutes only score
  scaling. Correct.lean exposes independent-array real flat IO; FPEquiv.lean
  exposes the admitted rewrite. The legacy natural-metadata summary is kept
  in Proofs/LegacySummary.lean and is not a current public contract.
- `bgmv_shrink_kernel` — `-1` sentinel early return as a guard (write-free
  path proven); `SPLIT_K == 1` store-vs-atomic tail split into two surfaces;
  `tl.max_contiguous` hint erased; unused `xk_stride` dropped.
- `layer_norm_welfold` — `tl.full` positional dtype spelled `dtype=`;
  `tl.broadcast_to` via tuple `tl.broadcast`; fused `tl.sum(…,1)[:, None]`
  split into two statements; unmasked `tl.store`; `libdevice.rsqrt` spelled
  `tl.rsqrt`; register casts parenthesized.
- `fused_layernorm_triton` — inductor `welford_reduce`/`welford` helpers
  inlined as exact-ℝ closed forms; `tl.broadcast_to` via tuple
  `tl.broadcast`; unmasked `tl.store`; `libdevice.rsqrt` spelled
  `tl.rsqrt`; register casts parenthesized.
- `fast_ce_loss` — DO_SOFTCAPPING and DO_LOGIT_SCALING are staged through
  cap/scale before dtype inference. All four combinations retain complete
  source templates; full statement-tree comparison and FPFastCeSource check
  branch/cast correspondence. The old pointer-load/cast exception is removed:
  the signed label pointer bump and direct load cast now remain literal.

The translation-marker list above is separate from the per-headline proof-scope
inventory. The manifest has 302 rows outside `full_value_candidate`: 268
specializations, 29 prepared-input slices, 3 pre-rounding slices, and 2 blocked
summaries. The blocked summaries are in `quant_transpose_kernel` and
`bgmv_expand_slice`. The former's companion scaled-store theorem and the
`quantize_global` and `rowwise_quantization_triton` headlines are pre-rounding
slices. None of those slices proves the faithful CUDA `llrint` / final int8 result. The per-statement review and
`proof_blockers.md` retain the remaining limitations.


## attention_kernel migration

The public directory now has Kernels/Correct/FPEquiv entries for both fp16 and
bf16 storage. Both originals and reciprocal candidates have independent real
IO proofs: scaled Q·K and both B0 bias windows feed the base-2 weighted sum,
with successful flat execution, active-address safety and a memory frame.
The real scope is contiguous head rows, unit inner stride, positive block/head
sizes, P_SEQ=0 and one or more whole key blocks; head and bias strides are
symbolic. Arbitrary-layout and partial-key-block real correctness remain outside
this contract. No identity-fp16-rounding premise is used by the current entries.

The FP proof uses only div_mul_rcp at actual post-loop fp32 operands, under
finite accumulator and finite/nonzero denominator checks. It retains every
source stride, both bias windows, storage-dtype bias arithmetic/scalar conversion,
default fp32 dots, Q/P casts, exp2, loads and store. Nonempty 2×2 full-source
executions for both dtypes inhabit the domain in a model of the admitted rules.
The textual translation exception now concerns explicit storage specializations
and implicit scalar/dtype conversions, checked by the source regression.
Historical cast-bearing claims are retained in Proofs/Legacy.lean; the new real
proofs execute a checked cast-free source and do not import that legacy model.
Final replay status is recorded in verification_round.md.


## attention_score: signed-source replacement

The current Kernels/Correct/FPEquiv entries retain signed offsets and sizes,
all four flags, fp16 Q/K/Out, fp32 M/accumulation, both checked-load arms and
masked output stores. Both real projections implement an independent finite
Q/K/M column sum, with complete trace safety and flat IO. Positive common
query/key block size is the only dimension premise; partial query blocks and
empty loops are included. The IO windows require unmasked M padding, K reads
before empty loops, and output allocation only for active keys.

The scalar-left FP candidate uses only mul_commute at finite actual fp32
operands in every reached iteration. Typed-source regressions retain the
precision boundaries and IS_EVEN_N-controlled query-tail selection.
Independent comparator replay and trust checks accept all 33 case/regression
files; all 198 named proofs are in the actual inventories, and snapshot inputs
match the final sources. This is the fifteenth completed directory.

The historical natural-subtraction window is preserved only in
Proofs/LegacyNaturalWindow.lean. Query 0/key 1 is included by that model but
excluded by the source's ordinary positive signed window. Its old statements
are absent from the public scope inventory. attention_kernel_aligned remains
user-deferred and is not a prerequisite for this work.


## attn_fwd_causal: complete two-helper source contracts

Kernels.lean retains both defined outer stages, independent helper pointer
copies, source strides, masks and finite additive penalty. Correct.lean now
proves both complete erased kernels against an independent normalized base-2
weighted sum. The formula retains repeated visits and masked-load padding;
it does not silently replace the finite penalty with exact causal masking.

The real contracts include successful flat-memory execution, all actual
address bounds and preservation of every cell outside active output writes.
Only actual input reads are pinned. Scale loads remain unmasked even for
fully padded key tiles. The partial, unaligned-range regression checks this
allocation requirement. No loop invariant or output equation is assumed by
the public contracts. Shapes and independent source strides are symbolic.

FPEquiv.lean proves complete-program division-to-reciprocal equivalence using
only div_mul_rcp. Guards inspect actual fp32 accumulators and denominators
after both loops; all source casts, dot precisions, masks, stores and failures
are retained. TypedExecution supplies a consistent-model execution/domain
witness. Experimental dimensions do not restrict the symbolic derivation.

RealFlatCounterexample supplies a concrete disjoint flat allocation and
successful execution of both versions: Q/K=0, scales=1 and V=[0,1] yield a
strictly positive first output. Thus the real source is not exact causal
masking. This is not a claim about GPU fp16 underflow. The earlier changed
exact-mask model remains in Proofs/LegacyExactMask and has no current public
coverage rows. Host launches and fixed-width address overflow are outside
these per-program contracts.

Final validation: `TritonBenchGVerified` builds (3,723 jobs); both focused
source/IO regressions, all 29 metadata/layout/shard tests and all source gates
pass. Independent comparator/trust replay accepts all 48 case/regression
files, including 307 named proofs and 1231 kernel theorem targets;
all 824 snapshot input hashes match. Evidence: `Logs/comparator-check-sve03cl2`.

## attn_fwd_triton: bf16 two-helper source contracts

Kernels.lean contains both complete sources with bf16 Q/K/V/Out, fp32
scales/accumulators, a bf16-input fp32-output score dot and explicit fp16
probability/V casts before the fp16-output value dot. The candidate changes
only final division to reciprocal multiplication. Both legal outer stages,
finite additive penalty, repeated visits, pointer reset and masks remain.

Correct.lean displays an independent weighted-sum formula and both complete
real flat IO contracts, including actual input windows, successful execution
and the memory frame. RealProjection checks equality of the complete erased
sources with the verified two-helper causal source before transferring these
contracts. It does not assert FP equivalence between the two benchmark cases.
The real exact-mask refutation also applies to actual flat source execution.

FPEquiv.lean retains the bf16 source and uses only div_mul_rcp, checking actual
finite fp32 accumulators and finite/nonzero row denominators after both loops.
Its consistent-model execution witness covers both defined outer stages.
Host launch composition and fixed-width overflow remain outside the per-program
scope. The historical changed model has no current public coverage rows.

Final validation: `TritonBenchGVerified` builds (3,733 jobs). The bf16 source
regression, refreshed causal source regression, 29 metadata/layout/shard tests
and all source gates pass. Independent comparator/trust replay accepts all
12 case/regression files, including 176 named proofs and
567 kernel theorem targets; all 834 snapshot inputs match.
Evidence: `Logs/comparator-check-_zp921x6`.

## batched_vecmat_mult: complete streamed GEMV contracts

Kernels.lean contains both complete fp32 sources. The candidate swaps only the
two operands of each vecmat addition; the dynamic K loop, input broadcasting,
axis-2 reduction, transpose and unmasked store are unchanged.

Correct.lean exposes both independent real double-sum formulas and complete
flat-memory IO contracts. Dimensions remain symbolic, with K=BK*T, BK>0 and
BN≤N plus bounds for the displayed unmasked input/output windows and disjoint
allocations. The source does not use M; actual access windows determine
validity. Both contracts include successful execution and every non-output
cell's preservation, including T=0. RealSource proves whole-source erasure
correspondence and the candidate's complete flat execution equality.
StreamReal adds the exact real two-stream/two-program-ID IO face; identity
rounding is confined to its internal bridge to the existing real foundation.

FPEquiv.lean uses only add_commute, checking finite actual carry and block-sum
operands at each reached update after the shared input prefix. Complete state
and failure are preserved. Source regression checks fp32 loads, the leading
unit broadcast axis, reduction/transpose expression, loop bounds and store.
RealExecution witnesses disjoint flat memory at nonzero program IDs, a 2×3
output tile and two K blocks: all six output values are 16. TypedExecution
witnesses both complete executions and satisfiable domains in a consistent
abstract model. Neither witness is a GPU experiment. Host launch composition,
fixed-width address overflow and concrete IEEE identities remain outside the
per-program contracts.

Final validation: `TritonBenchGVerified` builds (3,743 jobs); the precision/geometry
regression, 29 inventory/layout/shard tests and every source gate pass.
Independent comparator/trust replay accepts all 11 case/shared/regression
files, including 80 named proofs and 472 kernel theorem targets;
all 844 final snapshot input hashes match.
Evidence: `Logs/comparator-check-m5wnbzx6`.


## FP8 conversion source mapping

`f8_conversion_utils` now uses explicit typed region declarations and the
source auditor's checked implicit-destination-cast correspondence. It no
longer needs a translation-surface exception. Both complete JITs retain all
input/output precisions, load defaults, masks and the duplicate forward store.
The current real contracts erase precision for the copy formula, while typed
FP equivalence preserves every conversion. Host reinterpret/launches and
physical rounding remain outside these per-program statements. See its
case README and verification round for the current proof acceptance status.


## fast_ce_loss source migration

The three current typed sources preserve the pointer bump, direct signed load
cast, libdevice tanh, constexpr flags, precision transitions and all original
stores. Their complete statement trees match the three Python JITs after only
Lean notation and explicit implicit-destination cast normalization. The old pointer-load/cast exemption is removed; the remaining constexpr
specialization is documented and all four flag combinations are checked. Six real specifications expose the actual
padded LSE/partial-loss/backward formulas and finite flat IO; three full FP
index commutations have an empty domain and no numerical atoms. The nonzero
chunk write and vocabulary-only capped-tail claims have source counterexamples.
Integrated builds, affected regressions and independent replay are tracked in
verification_round.md; migration itself is not acceptance.


## Fast normalization and rotary source migration

fast_layernorm, fast_rms_layernorm and fast_rope_embedding now each expose
complete Kernels/Correct/FPEquiv entries, independent full finite-flat real
contracts and empty-domain structural FP equivalences. These retain the
padded LayerNorm variance, RMSNorm's source-specific weight strides and cast,
and the complete rotary head loop with initial coefficient snapshots.
Storage/cast checks and public finite/typed execution witnesses are included.
The source auditor accepts whitespace before tl.store parentheses through its
existing argument parser, with negative pointer/value/type regressions.
Builds and independent replay are tracked in verification_round.md; source
migration and scope classification alone do not mark the cases accepted.
