# TritonBench-G Completion Audit

This document separates the port inventory, the scope of each headline theorem,
and the checks required for a successful audit. Passing the mechanical gates
is not evidence that every upstream kernel has a full correctness proof.
Faithful translation still requires review against
[`review_criteria.md`](./review_criteria.md).

## Current Source Inventory

The 2026-09-24 statement-scope review records 173 Python/Lean port pairs and
345 headline (`specification`) declarations. Another 11 of the 184 kernel work
directories are README-only scaffolds and are not counted as implemented ports.
The `_spec-sheets` metadata directory is not a kernel work directory.

[`coverage_review.json`](./coverage_review.json) records the scope, assumptions,
numeric model, and Python function links for each headline. Its Lean and Python
source fingerprints are checked by
[`bench/coverage_review.py`](../coverage_review.py); the reviewed categories
are reflected in [`proof_gap_manifest.tsv`](./proof_gap_manifest.tsv).

| Coverage level | Headlines | What the label records |
|---|---:|---|
| `full_value_candidate` | 8 | A candidate full-value contract within its stated model and assumptions. |
| `specialization` | 296 | A contract restricted to the shapes, branches, parameters, or execution scope stated in the review. |
| `precomputed_input_slice` | 37 | A contract whose inputs include prepared intermediates; it does not prove the omitted producer. |
| `pre_rounding_slice` | 3 | A real-valued result before the faithful fixed-width rounding/cast result. |
| `blocked_summary` | 1 | An explicitly blocked summary. |
| **Total** | **345** | **Across 173 port files.** |

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
| Translation exceptions | Require explicit `Translation-surface blocker:` preamble markers and matching documentation. The 40 registered ports are listed below; a documented exception is not a discharged proof obligation. |
| Correctness surfaces | Require a named correctness surface or an explicitly registered `Correctness-surface blocker:`. The 16 registered ports and their 25 frame-free headlines are documented in `proof_blockers.md`; passing this gate does not put every headline on a named surface. |
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

No explicit TritonBench-G `hAlg` blocker remains. Forty ports carry an
explicit `Translation-surface blocker:` preamble marker — a documented,
deliberate deviation of the Lean `triton { }` surface from a literal
transcription of the upstream Python body (helper-JIT inlining, constexpr-path
specialization, antiquoted in-body constants); see `proof_blockers.md` for the
per-port descriptions. `bench/audit_tritonbench_g.sh` keys the textual
py↔lean surface-scan exemptions on that marker only and requires this list to
match the active marker set exactly.

The current documented blocker set is:

- `parallel_attention` — `parallel_rebased_fwd_kernel` + the
  `_parallel_rebased_bwd_dkv` helper ported; the helper's scalar arguments
  become universally-quantified binders (no cross-JIT call surface; marker
  registered in `proof_blockers.md`); its trailing bare `return` dropped;
  its descending `-BTS` loop as the ascending `hi − j·BTS` change of
  variable with a `cdiv` trip count; the backward shell and
  `_parallel_rebased_bwd_dq` are the trusted boundary.
- `bmm_optimized` — `bmm_kernel` ported at the fully-masked
  `DIVISIBLE_M = DIVISIBLE_N = DIVISIBLE_K = False` constexpr arm (marker
  registered in `proof_blockers.md`); both `GROUP_M` CTA-reorder arms
  transcribed (runtime `GROUP_SIZE` gate as a nested `ifThenElse`);
  batch-offset parameter reassignments folded into the pointer tiles;
  tuple assignment split; `range(num_iters)` spelled with explicit
  start/step; the autotune sweep and host launch are the trusted boundary.
- `parallel_retention_attention` — `parallel_retention_fwd_kernel` + the
  `_parallel_retention_bwd_dkv` helper ported; the helper's scalar arguments
  become universally-quantified binders (no cross-JIT call surface; marker
  registered in `proof_blockers.md`); its trailing bare `return` dropped;
  its descending `-BTS` loop as the ascending `hi − j·BTS` change of
  variable with a `cdiv` trip count; the diagonal unary-minus decay index
  respelled as a subtraction under the causal `tl.where` mask; int→float
  promotions spelled `tl.toReal(...)`; the backward shell and
  `_parallel_retention_bwd_dq` are the trusted boundary.
- `f8_conversion_utils` — both jit kernels ported (first consumer of the
  fp8 dtype channel); the stores' implicit destination-element-type casts
  spelled explicitly (`(x).to(tl.float16)` / `(x).to(tl.float8e5)`; marker
  registered in `proof_blockers.md`); `BLOCK_SIZE: tl.constexpr` as a Lean
  `Nat` parameter; the upstream duplicate store in `kernel_f8_to_f16`
  transcribed verbatim; host launch and `reinterpret` calls are the
  trusted boundary.
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
- `attention_llama` — `_fwd_kernel` ported at the `USE_FP8 = False` arm
  (the fp8 path is a `bitcast=True` bit-reinterpretation, an ℝ-model
  limit; marker registered in `proof_blockers.md`); `IS_CAUSAL` split into
  twin faithful surfaces (runtime `forRangeDyn` loop bound in both); two
  dimension-general exec closed forms — full natural-exp softmax attention
  (non-causal, under `N_CTX = BLOCK_N · numKVBlocks`) and shifted-causal
  softmax (under `start_position ≤ pid₀ · BLOCK_M` and span `≤ N_CTX`);
  the K/V positional-`other` loads spelled as plain masked loads whose
  dead lanes are pinned by `hundef`; host launch is the trusted boundary.
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
- `attn_fwd_triton` — `_attn_fwd_inner` inlined; `128`/`96` head constants
  generalized to `BLOCK_DMODEL`/`HEAD_ACTIVE` binders.
- `attn_fwd_causal` — `_attn_fwd_inner` inlined; `128`/`96` generalized.
- `attention_fwd_triton2` — `_attn_fwd_inner` inlined; `128`/`96` generalized.
- `attention_fwd_triton3` — `_attn_fwd_inner` inlined.
- `sgmv_expand_slice` — `EVEN_K`/`ADD_INPUTS=false`/`CAST_TYPE=false` path;
  CTA linearization via program-id axes; layout hints erased.
- `matmul_kernel` — in-body `4096` constants and `tl.cdiv` trip count as
  antiquoted binders.
- `matmul_triton_autotune` — `tl.cdiv` trip count as antiquoted `numKBlocks`;
  loop counter spelled `kk`.
- `bmm_chunk_bwd` — in-body `chunk_size_limit = min(...)` supplied as a
  precomputed parameter (no `seqlen` binder).
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
- `chunk_retention_ops` — the state-recurrence pair ported; `fwd_kernel_h`
  is statement-identical to `chunk_retention`'s (with `initial_state`/
  `final_state` spelled `h0`/`ht`); its `bwd_kernel_dh` is the clean
  dimension-general store-history sibling (in-loop stores, fixed decay).
  Same `tl.toReal(...)` promotion casts (marker registered in
  `proof_blockers.md`); tuple assignments split; descending loop as the
  ascending change of variable; `do` → `do_`.
- `chunk_retention` — the state-recurrence pair (`fwd_kernel_h`, the file's
  first kernel, + `bwd_kernel_dh`) ported; `fwd_kernel_o`/`bwd_kernel_dqkv`
  are the trusted boundary. The decay prologue's implicit int→float
  promotions are spelled `tl.toReal(...)` (marker registered in
  `proof_blockers.md`); tuple assignments split; `bwd_kernel_dh`'s
  descending `range(NT-1, -1, -1)` spelled as the ascending
  `for j in range(0, NT)` with `i_t = NT - 1 - j` (the `chunk_linear_attn`
  respelling), its post-loop `i_t` use pre-initialized (DSL scopes loop-body
  names), its single shared block size bound as `BT` (its `tl.dot` shapes
  force `BK = BT = BV`), and its dead signature params (`q`, `s_qk_*`,
  `scale`) omitted (`fused_recurrent_retention` precedent); `do` → `do_`.
- `chunk_gla_fwd` — output kernel `chunk_gla_fwd_kernel_o` ported with a
  full multi-block K-loop headline; the four `A`-builder kernels open with
  bare early `return`s (no `Stmt` early exit) and are the trusted boundary.
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
- `fast_ce_loss` — `_cross_entropy_forward`'s `labels_ptr += row_idx`
  pointer bump folded into the load offset (`tl.load(labels_ptr + row_idx)` —
  same address, same single load) and the following `.to(tl.int32)` (identity
  on the already-`.int` channel) dropped (marker registered in
  `proof_blockers.md`); forced by the `MetaGatherMasked2DKernelIO₂ₓ₂` skin,
  whose `mwinL := fun pid₀ _ => pid₀` field is the label's address, so the
  metadata load must be written in that shape. The sibling
  `_chunked_cross_entropy_forward` surface — stated on the plain
  `Realizes_without_Rounding` face, which constrains no load shape — keeps the
  literal spelling. Host launch and `@triton.heuristics` are the trusted
  boundary.

The translation-marker list above is separate from the per-headline proof-scope
inventory. The manifest has 337 rows outside `full_value_candidate`: 296
specializations, 37 prepared-input slices, 3 pre-rounding slices, and 1 blocked
summary. The blocked summary is in `quant_transpose_kernel`; its companion
scaled-store theorem and the `quantize_global` and `rowwise_quantization_triton`
headlines are pre-rounding slices. None of those slices proves the faithful
CUDA `llrint` / final int8 result. The per-statement review and
`proof_blockers.md` retain the remaining limitations.
