# TritonBench-G Proof Blockers

The current statement-scope inventory is `coverage_review.json`, checked against
both Lean and Python source fingerprints by `bench/coverage_review.py` and
rendered into `proof_gap_manifest.tsv` by `bench/check_proof_gap_manifest.py`.
The [browsable coverage table](https://lizn-zn.github.io/VeriTile/proofs/coverage/)
shows exact theorem statements, IO definitions, Python functions, numeric
models, and limitations for all 431 headlines across 173 ports.

The current review records 141 `full_value_candidate`, 258 `specialization`,
27 `precomputed_input_slice`, 3 `pre_rounding_slice`, and 2 `blocked_summary` rows. A candidate is subject to its model and explicit assumptions; a scope
review is not an independent certification of the Python translation or GPU
execution. Source changes require a new review rather than automatically
retaining a favorable label. Explicit source annotations cannot be overridden
by the review metadata. Axiom cleanliness is a separate property.

Several headline statements consume prepared intermediates even when the file
also contains a full producer surface or projection proof. For example,
`flash_decode2_llama`, `flash_decode2_phi`, and
`layer_norm_ops` include writeback contracts whose inputs are already computed.
The inventory records the actual headline's scope, without transferring claims
from neighboring declarations. `rotary_emb` proves separate store contracts;
its full-surface projection does not itself compose them into one full IO proof.

The quantization gaps remain explicit: `int8_quantization` takes `ScalePre`;
`quantize_global` and `rowwise_quantization_triton` have pre-rounding claims;
faithful fixed-width cast/rounding behavior is not inferred from these slices.
Existing issue links identify the historical blocker family, not an assertion
that the linked issue is currently open or that closing it completed a proof.

The older completion narrative is preserved in
[the pre-review revision](https://github.com/Lizn-zn/VeriTile/blob/c29712db635d72fec66c6c5f6ee1fb22b3605e2e/bench/tritonbench_g/proof_blockers.md).
Its broad “discharged” and “full-value” labels must not replace the current
per-statement inventory. The sections below track additional named-surface,
frame, and translation limitations. The mechanical translation scan remains
a consistency gate, not a substitute for review against `review_criteria.md`.

## Block-sparse public contract

`block_sparse_attn/Correct.lean` now states complete source-to-initial-array
correctness and flat IO for both source variants and all three storage types.
The signed CSR loop, independent column stride, actual loads/dots/normalization
and final masked stores are composed without a whole-body `hstep` premise.
Arithmetic allocation windows imply access safety; the flat postcondition
includes tagged outputs and every other memory cell. `FPEquiv.lean` exposes
whole-source score-scale commutation using only admitted `mul_commute`.

The translation-surface marker for `block_sparse_attn` covers explicit
fp32/fp16/bf16 storage specialization, probability/output casts and a fixed
`static_print` payload. Signed metadata, all masks, branches and `tl.exp`
remain in the complete source; integer overflow and host launch are outside
the per-program model.

The old conditional summary and detached store slices remain explicitly marked
in `Proofs/LegacySummary.lean`; neither public entry imports them. The manifest
now describes the new public contracts. Final acceptance independently
replayed all 59 case/test files, checked 480 named declarations and 2,709 kernel
theorem targets, and verified all 953 snapshot input hashes. The case is
complete for the displayed contracts; see [the case notes](block_sparse_attn/README.md).

## Correctness-Surface Blockers

Every completed port must expose its correctness claim through a **named**
surface — the `KernelIO` `⊨` / `⊨[R]` faces, `ComputeCorrect.Realizes`,
`ComputeRefine.Realizes`, `ComputeCorrect.General`, or a named
`correct_target`. `bench/audit_tritonbench_g.sh`'s correctness-surface scan
enforces that.

The ports below do **not** yet meet it: their `*_exec_genuine` headlines state
correctness as an inline exec-existential

```lean
∃ sF, exec <surface>.toAlgKernel s = some sF ∧ ∀ idx, <guard> → sF.mem out <addr> = <spec>
```

which proves termination and every stored output cell but **omits the frame** —
"no cell outside the write set changed" is not part of the statement. **None**
of the 16 headlines across these files carries a frame conjunct. Lifting one to `⊨`
is therefore a *proof* obligation (fit an IO skin, discharge `FlattenOk` /
`TraceSafeR`, prove the frame), not a rename — which is why they are registered
here rather than silently accepted.

Each file carries the matching machine-readable preamble marker

```
Correctness-surface blocker: <why this headline is not on a named surface>
```

and the audit rejects both an unregistered offender and a stale entry here.

| Port | Headlines | Frame in statement |
|---|---|---|
| `int4_matmul` | `int4_matmul_exec_genuine` | no |
| `int8_dequant_matmul` | `int8_dequant_matmul_exec_genuine` | no |
| `int8_matmul_kernel` | `int8_matmul_kernel_exec_genuine` | no |
| `int8_matmul_quantization` | `int8_matmul_quantization_quantize_exec_genuine`, `int8_matmul_quantization_matmul_exec_genuine` | no |
| `int_scaled_matmul` | `int_scaled_matmul_matmul_exec_genuine`, `int_scaled_matmul_scaled_exec_genuine` | no |
| `matmul_dequant_int4` | `matmul_dequant_int4_exec_genuine` | no |
| `matmul_dequantize` | `matmul_dequantize_matmul4_exec_genuine`, `matmul_dequantize_matmul_exec_genuine`, `matmul_dequantize_dequantize_exec_genuine` | no |
| `matmul_dequantize_int4` | `matmul_dequantize_int4_exec_genuine` | no |
| `parallel_attention` | `pa_fwd_o_exec_genuine`, `pa_bwd_dkv_exec_genuine` | no |
| `parallel_retention_attention` | `pra_fwd_o_exec_genuine`, `pra_bwd_dkv_exec_genuine` | no |

These are the 10 remaining ports added between 2026-08-10 and 2026-08-25 (ports 156–173:
the descending-lever, fp8 and integer families). Every other port states its
headline on a named surface.

### What each of these is actually blocked on

The remaining 10 ports fall into two signature families. All contain loops,
so completing them requires actual prologue/loop/store execution, active
access bounds and a full memory frame. The shared `UKernelIO.ImplementsAt`
bridge now supports exact real execution with state-dependent launch
conditions, as used by `bmm_optimized`. Existing streaming `ImplementsR`
contracts remain available where appropriate; converting every proof to
`execR` is no longer a prerequisite for a named correctness surface.

**Family 1 — three program axes, all-float channels (2 ports).** Both already have an arity-matching skin, so at least no new library surface
is needed for the signature.

| Port | channels in/out | arity-matching skin |
|---|---|---|
| `parallel_attention` | `q`,`k`,`v` / `o`,`z` (3/2) | `StreamMasked3DKernelIO₃ₓ₂` |
| `parallel_retention_attention` | `q`,`k`,`v` / `o` (3/1) | `StreamMasked3DKernelIO₃` |

These are arity matches read off the surface signatures; confirm input and
output roles against the actual stores before selecting a contract.

`bmm_optimized` has left this blocker list. Its public contract now uses the
shared pointwise `UKernelIO.ImplementsAt` bridge, with both runtime grid
dimensions pinned, full active-address safety and preservation of all other
memory cells. The source retains all eight mask arms and the final short CTA
group. Its real product formula explicitly requires zero unspecified loads;
`Proofs/UndefinedTail.lean` refutes removing that condition. Final acceptance
is tracked separately in `verification_round.md`.

**Family 2 — typed `.int` / `.nat` data channels (8 ports).** `int4_matmul`,
`int8_dequant_matmul`, `int8_matmul_kernel`, `int8_matmul_quantization`,
`int_scaled_matmul`, `matmul_dequant_int4`, `matmul_dequantize`,
`matmul_dequantize_int4`.

These are blocked on missing library surface, not on proof effort: **every one
of the 61 skins declares its streamed inputs and its terminal output as a plain
`RegionName`** — the `.real` channel. The only typed field anywhere in the skin
family is `sty : Fin nMeta → ChanTy`, and that types the *metadata scalar
slots*, never the data. A port whose GEMM operands or output ride a
`Region .int` / `Region .nat` channel therefore has no skin to fit, at any
arity. Unlocking this family means a typed-channel streaming skin — one lever
that would move all eight, and the highest-leverage item on this list.


## Translation-Surface Blockers

`attention_fwd_triton1` resolves the Python test's fp32 dtype aliases explicitly:
its six casts target `p_h.dtype.element_ty`, four occurrences of `b_q.dtype`,
and `p_o.dtype.element_ty`, all `tl.float32` for fp32 Q/K/V and the wrapper's
same-dtype H/O allocations. Region metadata supplies load types. The explicit
casts retain FP operations that the legacy alias shorthand would erase.
The source mapping is documented in the directory README; Lean checks the
resulting complete typed sources, not Python-to-Lean alias resolution.


`attention_forward_triton` uses a fully inlined helper in `Kernels.lean`.
`_attn_fwd_inner`'s entire loop replaces its call and return; its unused stage
argument disappears. The source's 128/96 head constants become symbolic
`BLOCK_DMODEL`/`HEAD_ACTIVE`; the supplied Python test selects fp16 Q/K/V and
fp32 scale buffers, and the wrapper fixes fp16 output. These storage annotations
and the explicit `tl.float16` terminal cast preserve that configuration.
`tl.math.exp2`, the two dots (including `out_dtype=tl.float16`), the load masks,
pointer updates and output store remain in the code. The real source/loop
identities and named `FPAttentionSource.lean` cast/store/mask checks connect the
proofs to the displayed source. This is a reviewed transcription exception;
those Lean checks do not independently verify Python-to-Lean translation or GPU
execution. The real mathematical claim requires whole KV blocks.


A port whose Lean `triton { }` surface deliberately deviates from a literal
transcription of the upstream Python kernel body declares that with an
explicit `Translation-surface blocker:` line in its module preamble.
`bench/audit_tritonbench_g.sh` keys the textual py↔lean surface-scan
exemptions on that marker only, and requires every marker to be registered
here and in `completion_audit.md`. The current registered set — each entry is
a documented, deliberate surface deviation, not an unproven correctness gap
unless stated:

- `bgmv_expand_slice` — the source macro specializes input/weight dtype and
  stages CAST_TYPE before register inference; the store spells out Triton's
  implicit output conversion. The early return becomes a non-sentinel active
  branch. Complete-source and precision regressions check all specializations;
  real source correspondence, full flat IO and typed execution witnesses retain
  the metadata, split calculation, masks, exact pointers and all numeric flags.
- `attention_score` — Q/K/Out fp16 and M fp32 storage are explicit, with
  `Out.type.element_ty` resolved to `tl.float16` for the supplied Python test
  configuration. Explicit region lists retain all inputs and the output.
  `FPAttentionScoreSource.lean` checks dot/load/store precision, signed window
  arithmetic and the source's IS_EVEN_N-controlled query-tail selection.
  Lean checks the transcription; the lexical gate does not verify this alias
  resolution or GPU behavior.
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
- `attention_fwd_triton2` — same helper inlining and `128`/`96` →
  `BLOCK_DMODEL` / `HEAD_ACTIVE` generalization, explicit fp16 Q/K/V and fp32
  scale storage, and the wrapper's bf16 output alias resolved to `tl.bfloat16`.
  The V-to-fp16 cast inside the fp16-output dot and final bf16 cast/store are
  preserved in both sources and checked by `FPAttentionFwdTriton2Source.lean`.
  The real projection/IO proofs transfer the independent attention formula;
  partial KV tails have checked counterexamples.
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
- `parallel_attention` — the DSL has no cross-`@triton.jit` function-call
  surface, so the backward dk/dv helper `_parallel_rebased_bwd_dkv` is
  ported as a standalone kernel over universally-quantified scalar arguments
  `i_bh, i_c, i_k, i_v, i_h` (the values the shell
  `parallel_rebased_bwd_kernel` would pass), with its trailing bare `return`
  dropped and its descending
  `for i in range(cdiv(T,BTS)·BTS − BTS, (i_c+1)·BTL − BTS, −BTS)` respelled
  as the ascending change of variable
  `for j in range(0, cdiv(cdiv(T,BTS)·BTS − (i_c+1)·BTL, BTS))` with
  body-first `i = cdiv(T,BTS)·BTS − BTS − j·BTS` (the `±BTS` in Python's
  `start − stop` cancels exactly in ℤ; ℕ-truncated subtraction reproduces
  Python's empty range, so the trip counts agree for all parameters). The
  port covers `parallel_rebased_fwd_kernel` (the file's first kernel) and
  the dk/dv helper; the backward shell and `_parallel_rebased_bwd_dq` are
  the trusted boundary.
- `parallel_retention_attention` — the same helper-as-standalone shape as
  `parallel_attention`: the backward dk/dv helper
  `_parallel_retention_bwd_dkv` is ported as a standalone kernel over
  universally-quantified scalar arguments `i_bh, i_c, i_k, i_v, i_h` (the
  values the shell `parallel_retention_bwd_kernel` would pass), with its
  trailing bare `return` dropped and its descending
  `for i in range(cdiv(T,BTS)·BTS − BTS, (i_c+1)·BTL − BTS, −BTS)` respelled
  as the ascending change of variable
  `for j in range(0, cdiv(cdiv(T,BTS)·BTS − (i_c+1)·BTL, BTS))` with
  body-first `i = cdiv(T,BTS)·BTS − BTS − j·BTS` (the `parallel_attention`
  respelling verbatim). Additionally its diagonal decay's unary-minus index
  spelling `-o_k[:, None] + o_q[None, :]` is respelled as the subtraction
  `o_q[None, :] - o_k[:, None]` (no unary tile negation in the DSL; the two
  agree on every lane the causal `tl.where` keep mask retains), and the
  decay prologue's implicit int→float promotions are spelled with the
  explicit `tl.toReal(...)` cast (the older recurrent-state translation). The
  port covers `parallel_retention_fwd_kernel` (the file's first kernel) and
  the dk/dv helper; the backward shell and `_parallel_retention_bwd_dq` are
  the trusted boundary.
- `attention_llama` — `Kernels.lean` contains both USE_FP8 storage
  specializations and preserves both IS_CAUSAL branches. Textual differences:
  positional load zero-fill is written as `other=`, Q.dtype.element_ty resolves
  to fp16, pointer-region dtypes are explicit, and the implicit fp16 output
  conversion is spelled as a cast. Both int8-to-E5M2 reinterpretations and
  subsequent fp16 casts are present; the FP8 arm is no longer omitted.
  Source tests inspect dot precision, casts, masks and stores. The textual
  scanner's specialization exception is not a Python-to-Lean proof.
  Correct.lean exposes independent formulas and full flat IO for both storage
  modes and versions, with its whole-block/nonempty-visible-row conditions.
  FPEquiv.lean uses only add_commute and retains symbolic shapes/strides.
  Host launch configuration and concrete hardware execution remain outside
  these per-program specifications.
- `bmm_optimized` — `Kernels.lean` specializes all eight `DIVISIBLE_M/N/K`
  mask choices because `None` is syntax-level. Both CTA-reorder branches and
  the final shortened group remain, as do all three pointer-parameter updates
  and actual `tl.num_programs` calls. Tuple assignment is split and range
  start/step are explicit. Common fp32/fp16/bf16 storage, fp32 dot accumulation
  and `allow_tf32=false` are preserved; implicit store conversion is explicit.
  Masked loads still omit `other`. The autotune sweep, fixed-width address
  overflow and grid-wide execution are outside the per-program proof.
- `cache_transform_triton` — both complete gathers specialize the nine independent
  cosine/sine fp32/fp16/bf16 storage pairs because region dtype annotations are
  syntax-level. Explicit signatures retain both inputs, signed metadata and
  both outputs. Original masks, omitted `other`, the full decoding tile and
  all strides remain; the only candidate edit commutes an exact index product.
- `int4_matmul` — three disclosed surface deviations, none semantic.
  (1) The `SPLIT_K` constexpr is fixed to `1` (the `tl.store` epilogue arm;
  the autotune table sweeps `SPLIT_K ∈ {1,2}` and the atomic-add arm
  accumulates into a host-zeroed `C` via `reset_to_zero` — the
  `bmm_optimized` fixed-arm precedent), so the headline carries the launch
  fact `s.pids 1 = 0` (grid axis 1 has extent `SPLIT_K = 1`). (2) The
  K-loop bound `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` is the antiquoted
  `numKBlocks` with the honest side condition
  `K = BLOCK_SIZE_K · numKBlocks` (the kernel docstring's own assert; the
  loads are unmasked so the exact presentation is required —
  `llama_ff_triton` precedent). (3) The dequant subtraction is spelled
  `(tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs`: the
  packed nibbles live on the `.nat` channel (bitwise ops are nat-only) but
  the nibble difference ranges over `[-15, 15]`, so the subtraction must
  run on the `.int` channel — `tl.cast(x, tl.int32)` is the explicit
  `Op.castNatToInt` spelling (`kcache_copy_triton` precedent), and the
  `int × real` product promotes through `Op.intToReal` (this port is its
  first consumer). The statement's trailing `.to(a.dtype)` — an
  erased-identity cast on the ℝ carrier — is dropped from this one
  assignment (the DSL's `.to(<ident>.dtype)` wrapper raw-text-matches
  `tl.int*` and rejects the ℝ `bs` operand); the following
  `tl.dot(a, (b).to(a.dtype))` keeps its cast faithfully. The packed
  weight/zero-point regions are `Region .nat`
  (`matmul_dequantize_int4` precedent). The `triton.autotune` sweep and
  the host launch are the trusted boundary.
- `llama_ff_triton` — the `USE_FP8` constexpr arm is dropped entirely
  (parameter + branches): that path bit-reinterprets int8 weight bytes as
  `tl.float8e5` (`.to(tl.float8e5, bitcast=True)`) — a bit-level decode with
  no value-level ℝ transcription (the fixed-width-cast ℝ-model limit family),
  so only the `USE_FP8 = False` (fp16-weights) arm is transcribed
  (`sgmv_expand_slice` dropped-arm precedent). The loop trip count
  `tl.cdiv(K, BLOCK_SIZE_K)` is the antiquoted `numKBlocks`
  (`K = BLOCK_SIZE_K · numKBlocks`; the loads are unmasked, so the exact
  presentation is required). The store's implicit fp16 cast (the host
  allocates `out` at `x.dtype = float16`) is spelled
  `(accumulator).to(tl.float16)` (`f8_conversion_utils` precedent). The
  K-loop counter is spelled `_i` for Python's `_`.
- `matmul_dequant_int4` — two disclosed surface deviations, neither
  semantic, plus a dead-kernel disclosure. The target JIT is the file's
  **second** kernel, `dequantize_kernel` — the only one any host launches
  (`matmul_dequantize_int4_s1` → `dequantize_int4` →
  `dequantize_kernel[grid]`, followed by a host-side `torch.mm`); the
  file's first kernel, `matmul4_kernel`, is dead code — no host references
  it, and its body is byte-identical to the kernel the
  `matmul_dequantize_int4` port proves — so it is not modeled
  (`rms_rbe_matmul` dead-kernel precedent). (1) The dequant statement is
  spelled `(tl.cast((int32_b >> b_shift) & $(15), tl.int32) -
  tl.cast((zp_b >> bzp_shift) & $(15), tl.int32)) * scale_b`: the packed
  containers live on the `.nat` channel (bitwise ops are nat-only), the
  nibble difference goes negative and ℕ subtraction truncates, so the
  subtraction runs on the `.int` channel via `tl.cast`
  (`Op.castNatToInt`, `kcache_copy_triton` precedent) and the
  `int × real` product promotes through `Op.intToReal` — the
  `int4_matmul` disclosure-(3) respell, applied inline. (2) Index
  literals are antiquoted (`$(8)` / `$(4)`; `0xF` as the decimal `$(15)`;
  the runtime `group_size` as `$(group_size)`). The three masked loads
  keep their `other=0.0` kwargs faithfully — on the two packed `.nat`
  channels the expander reads the literal `0.0` as the nat constant `0`,
  value-identical. The packed weight/zero-point regions are `Region .nat`
  (`matmul_dequantize_int4` precedent); the autotune sweep and the host
  launch (grid, strides, the downstream `torch.mm`) are the trusted
  boundary.
- `int8_dequant_matmul` — five disclosed surface deviations, none semantic.
  (1) The `SPLIT_K` constexpr is fixed to `1` (the `tl.store` arm; the
  autotune table sweeps `SPLIT_K ∈ {1,2,4,8,16}` with
  `pre_hook=init_to_zero("C")`, and the `tl.atomic_add` arm drops with the
  constexpr — the `int4_matmul` fixed-arm precedent); every `* SPLIT_K`
  factor folds to 1, `pid_z` and `rk = pid_z*BLOCK_K + arange` stay
  faithfully spelled, and the headline carries `s.pids 1 = 0`. (2) The
  `EVEN_K` `triton.heuristics` constexpr (`K % (BLOCK_K*SPLIT_K) == 0`)
  is fixed to `True` (the unmasked-load arm; the masked `else` arm drops —
  the `bmm_optimized` heuristics + `sgmv_expand_slice` `EVEN_K`
  precedents), with the `tl.cdiv` trip count as the antiquoted
  `numKBlocks` and the honest side condition `K = BLOCK_K · numKBlocks`.
  (3) `has_bias` stays a genuine `Bool` parameter with BOTH arms modeled
  (the `matmul_dequantize` `NO_GROUPS` precedent); one headline covers
  both via a guarded bias term. (4) The two `.to(C.dtype.element_ty)`
  casts are spelled `.to(tl.float16)` (the host allocates `C` as
  `float16`; `llama_ff_triton` precedent) — the accumulator cast is a
  genuine `Op.castFloat real → fp16` making the masked store
  `.fp16`-typed (MemCell-level readback, the `matmul_dequantize`
  precedent), and `tl.load(bias + rn).to(tl.float16)` compiles to a
  direct `.fp16`-typed load by a dedicated syntax rule (not
  load-then-cast). (5) `$(n)` literal antiquoting. The int8 `A`/`B`
  regions are `Region .int` (signed channel); `acc += tl.dot(a, b)` is
  the first `Op.dotInt` consumer, and the epilogue's `acc * divfactor`
  promotes through `Op.intToReal`. The autotune sweep, the heuristics
  lambda, and the host launch are the trusted boundary.
- `int_scaled_matmul` — two JIT kernels, both launched, both modeled in
  py order (audit anchor = the first, `matmul_kernel_with_block_pointers`).
  **`matmul_kernel_with_block_pointers`** (block-pointer int8 GEMM):
  (1) integer widths erased — the block-ptr loads are spelled faithfully
  (`tl.load(bp, boundary_check=(0, 1))`, no added kwargs; their `.int`
  channel is inherited from the typed base region of `tl.make_block_ptr`
  via this port's inference rider — untyped bases keep the historical
  `.real` default, so every existing block-ptr port lowers unchanged),
  and the int8/int32 widths collapse onto the `.int` channel (#154
  fixed-width family);
  (2) tuple→bracket respells (`tl.advance` deltas, `tl.zeros` shape);
  (3) the `GROUP_M = min(…, GROUP_M)` parameter-name rebind is spelled
  as a register assignment whose RHS reads the antiquoted binder —
  shadowing works, no rename; (4) `order=(1, 0)` erased (`matmul_tma`
  precedent). The step-form loop `range(0, K, BLOCK_K)` is spelled
  directly (no trip-count binder; boundary checks keep it ragged-safe
  under the sole hypothesis `0 < BK`).
  **`scaled_matmul_kernel_with_block_pointers`** (pointer GEMM +
  inductor store suffix): (1) the descending loop
  `range(K, 0, -BLOCK_K)` is the ascending change of variable with the
  in-body `k` rematerialization (`parallel_retention_attention`
  precedent); (2) `ACC_TYPE : tl.constexpr = tl.int32` fixed at its only
  instantiation (`bmm_optimized` fixed-arm family); (3) the
  `tl.broadcast_to(…, mask.shape)` shape attribute is respelled to the
  literal dims `[BLOCK_M, BLOCK_N]` (the `tl.broadcast_to` syntax itself
  landed with this port); (4) `eviction_policy="evict_last"` is kept and
  macro-erased; (5) integer widths erased, and the suffix's
  `acc * tmp0` store — an `.int × .real` product promoted through
  `Op.intToReal` — writes `.real`-typed cells into the int32-allocated
  `C`: the cell carries the exact real product and the hardware's
  float→int32 store truncation is outside the model (#154 family).
  `EVEN_K` stays a genuine `Bool` with both arms modeled
  (`hEven : EVEN_K = true → K = numKBlocks·BK` guards only the unmasked
  arm); `stride_s1m`/`stride_s1n` are dead parameters (the `s1` load is
  strideless — `fused_recurrent_retention` dead-stride precedent). The
  host launches are the trusted boundary.
- `int8_matmul_kernel` — four disclosed surface deviations, none
  semantic, in a **pure-integer** kernel (int32 A × 2-bit-packed uint8 B
  → int32 C; no floats anywhere). (1) The inner loop bound
  `tl.cdiv(K // 4, BLOCK_SIZE_K)` is the antiquoted `numKBlocks` with
  the honest side condition `K = 4 · (BK · numKBlocks)` — exactly the
  kernel's own `tl.static_assert(K % (4*BLOCK_SIZE_K) == 0)` (spelled
  faithfully; the assert makes both load masks degenerately all-true,
  which the port proves). (2) `tl.full((1,), 1, dtype=tl.int8)` spells
  its shape `[1]` (tuple → bracket; the rank-1 `[1]` operand needs no
  further respell — the expander's broadcast does rank promotion).
  (3) Integer width erasure at three sites: the A-load's `.to(tl.int8)`
  is a width-erasure no-op on the `.int` channel (the host feeds
  0..255, so hardware int8 would wrap values ≥ 128 — the #154
  fixed-width caveat); the extract's `.to(tl.int8)` is the genuine
  `Op.castNatToInt` signed hop (values {0..3}, no honesty gap); the
  `tl.full` dtype hint's int8 width is erased to `.int`. (4) `$(n)`
  literal antiquoting. The `out_dtype=tl.int32` dot kwarg is kept and
  macro-erased (redundant on the exact-ℤ `Op.dotInt`). The autotune
  sweep and host launch are the trusted boundary.
- `int8_matmul_quantization` — two JIT kernels, both launched
  (`matmul_quantize_int8` chains the per-row quantizer into the int8
  GEMM), both modeled in py order (audit anchor = the first,
  `quantize_int8_perrow_kernel`). Per-kernel disclosures, none semantic.
  **`quantize_int8_perrow_kernel`**: (1) both `tl.cdiv` K-loop bounds are
  the antiquoted `numKBlocks` in the honestly-ragged **ceil form**
  `K ≤ numKBlocks · BK` (the loads and the int8 store carry genuine
  remainder masks, so extra all-masked blocks contribute nothing — weaker
  than the exact-divisibility `matmul_triton_autotune` form); (2) the
  final scale store is Triton's implicit fp16 store cast (the host
  allocates `a_scale` as `float16`), spelled `(a_scale).to(tl.float16)`
  (`f8_conversion_utils` precedent) — an `.fp16`-typed store read back at
  the MemCell level; (3) `127.` is written `127.0` and index literals are
  antiquoted `$(n)`. The `.to(tl.int8)` quantization is
  `Op.castRealToInt8` (truncate-toward-zero; the ±127 hardware-saturation
  caveat is the #154-family disclosure, `quantize_copy_kv` wording).
  **`matmul_kernel`**: (1) `SPLIT_K` fixed to `1` (the `tl.store` arm;
  autotune sweeps {1,2} with `reset_to_zero=['c_ptr']`, the
  `tl.atomic_add` arm drops; headline carries `s.pids 1 = 0`); (2) the
  epilogue's `accumulator.to(tl.float32)` has no explicit int→float
  spelling in the DSL — the promotion is carried by the auto-inserted
  `Op.intToReal` at exactly the Python cast's site in
  `(accumulator * a_scale[:, None] * b_scale[None, :]).to(tl.float16)`;
  (3) the same ceil-form `numKBlocks`; (4) `$(n)` literals. The masked
  `.int` K loads' `other=0.0` kwargs are kept faithfully — the expander's
  `.int` channel gained the literal-`0.0` arm (a two-line mechanical
  clone of the `.nat` arm) with this port. The autotune sweeps and both
  host launches are the trusted boundary.
- `matmul_dequantize` — three JIT kernels, all launched by hosts the test
  exercises, all three modeled in py order (the audit anchor is the first,
  `matmul4_kernel`). Per-kernel disclosures, none semantic.
  **`matmul4_kernel`** (light textual variant of the kernel the
  `matmul_dequantize_int4` port proves — the source drops the dot's
  `.to(a.dtype)` cast and spells the dead `c = accumulator.to(…)` binding
  directly as `tl.float16`): `$(n)` literal antiquoting only (`0xF` as the
  decimal `$(15)`); `NO_GROUPS` stays a genuine `Bool` parameter with both
  arms modeled. **`matmul_kernel`** (light textual variant of the kernel
  the `int4_matmul` port proves): the int4_matmul disclosures (1)–(3)
  verbatim — `SPLIT_K` fixed to `1` (the `tl.store` arm; the
  `tl.atomic_add` arm and its `reset_to_zero` host contract are dropped
  with the constexpr; headline carries `s.pids 1 = 0`), the
  `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` trip count as the antiquoted
  `numKBlocks` with `K = BLOCK_SIZE_K · numKBlocks` (unmasked loads), and
  the signed dequant respell through `tl.cast(·, tl.int32)` +
  `Op.intToReal`. Unlike the twin, this source spells its value casts
  directly (`.to(tl.float16)` on the dequantized tile, both `tl.dot`
  operands, and the stored `c`) — all kept faithfully; the genuinely
  `.fp16`-typed masked store makes the honest headline readback
  MemCell-level (the `matmul_tma` fp16 precedent, masked).
  **`dequantize_kernel`** (byte-identical to the kernel the
  `matmul_dequant_int4` port proves): that port's disclosures verbatim —
  the inline `.int`-hop dequant respell and `$(n)` antiquoting; the
  masked loads keep `other=0.0` faithfully. The autotune sweeps and the
  three host launches are the trusted boundary.
- `rms_matmul_rbe` — the second kernel `rms_matmul_rbe_qkv` (the only one
  any host function launches) is a **cross-JIT caller**: its body invokes
  `rms_matmul_rbe(...)` three times (Q/K/V). The DSL has no jit-to-jit
  call surface, so the callee body is inlined three times (`attn_fwd`
  inline precedent); the callee's unused parameters
  (`start_token_position` / `RBE_EPILOGUE` / `THETA`) vanish under
  inlining. The `USE_FP8` constexpr arm is dropped (`bitcast=True`
  bit-reinterpretation, the ℝ-model-limit family). The loop trip count
  `tl.cdiv(K, BLOCK_SIZE_K)` is the antiquoted `numKBlocks` (unmasked
  loads); the stores' implicit fp16 casts are spelled
  `(accumulator).to(tl.float16)`; loop counters `_i` for Python's `_`.
- `rms_rbe_matmul` — the target JIT is the file's **second** kernel,
  `rms_matmul_rbe`; the first, `rbe_triton`, is dead code in this file (no
  host function launches it, and it calls `get_freq_multi_tokens`, which
  the file never defines — the upstream file itself cannot run it) and is
  not modeled. The `USE_FP8` constexpr arm is dropped (parameter +
  branches; `.to(tl.float8e5, bitcast=True)` bit-reinterpretation, the
  ℝ-model-limit family; `llama_ff_triton` precedent). The unused
  parameters `start_token_position` / `RBE_EPILOGUE` / `THETA` are dropped
  (`sgmv_expand_slice` unused-param precedent). The loop trip count
  `tl.cdiv(K, BLOCK_SIZE_K)` is the antiquoted `numKBlocks` (unmasked
  loads). The store's implicit fp16 cast is spelled
  `(accumulator).to(tl.float16)` (`f8_conversion_utils` precedent). The
  K-loop counter is spelled `_i` for Python's `_`.
- `triton_matmul` — the constexpr epilogue branch
  `if (c_ptr.dtype.element_ty == tl.float8e4nv):` (a compile-time dispatch
  on the output buffer's element dtype) is split into two Lean surfaces
  (`triton_matmul_f16_surface` / `triton_matmul_f8_surface`, one per arm —
  the `matmul_tma` twin-surface precedent), so the branch statement itself
  appears in neither surface; the loop trip count `tl.cdiv(K, BLOCK_SIZE_K)`
  is supplied as the antiquoted `numKBlocks` binder
  (`K = BLOCK_SIZE_K · numKBlocks` presentation, `matmul_triton_autotune`
  precedent); the K-loop counter is spelled `kk` where Python spells it `k`
  (clash with the antiquoted dimension binder). The
  `triton.jit(launch_metadata=...)` hook and the host's per-dtype config
  table are the trusted boundary; `tl.max_contiguous`/`tl.multiple_of` are
  kept in the surface text and erased by the DSL to their value argument.
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
- `bmm_chunk_bwd` — host-selected A/Dout storage and dot dtypes are
  specialized explicitly. The complete source now computes its own chunk
  length and retains all residual arms, partial tails and output casts;
  the former precomputed-length specialization lives in `Proofs/Legacy.lean`.
- `bmm_chunk_fwd` — host-selected A/B storage and dot dtypes are specialized
  explicitly. The nested causal early return is written as its complementary
  active-tile guard; skipped tiles perform no loads/stores. The actual
  in-body length, partial K blocks, signed sequence filter, original load
  storage and output conversion remain in the complete source.
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
- `chunk_retention` — the complete typed source spells the source's signed
  `tl.arange` index type with an explicit `.to(tl.int32)` identity cast. This
  extra cast is required to distinguish signed negative tail/score offsets
  from natural-number subtraction in the DSL. All four full real and FP
  source pairs retain the intrinsic decay operations, casts and stores;
  implicit integer-to-float promotion is handled by the DSL.
- `chunked_cumsum_fwd` — bias storage and finite/positive-infinite upper
  bounds select explicit typed source templates. The selected clamp and
  multiply statements are quoted into one complete body. All preprocessing,
  casts, masks, both stores and the scan remain. The public real contract
  requires `pid_c * chunk_size <= seqlen`, as supplied by the host launch, so
  natural subtraction matches the source's remaining length. The default
  `tl.minimum(x, float("inf"))` remains an opaque FP operation and projects
  to identity only in the real semantics.
- `chunk_retention_ops` — all four complete typed sources retain signed
  indices via an explicit `.to(tl.int32)` identity cast, with automatic float
  promotion. The reverse loop visits NT-1-j with an ascending counter. All
  backward-state history stores and full-BT decay updates are retained, with
  independent tile dimensions. Three shared source pairs are checked equal
  to their `chunk_retention` counterparts; backward state has its own full
  real execution, bounds and tagged-cell frame proof.
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
- `fast_ce_loss` — DO_SOFTCAPPING and DO_LOGIT_SCALING are staged through
  cap/scale before dtype inference. All four combinations retain complete
  source templates; full statement-tree comparison and FPFastCeSource check
  branch/cast correspondence. The old pointer-load/cast exception is removed:
  the signed label pointer bump and direct load cast now remain literal.

### Required VeriTile surface extensions

The attention helper-call surfaces are inlined into the outer-kernel streaming
loops for translation coverage (see the `attn_fwd_*` / `attention_fwd_*`
entries above). Future work may add executable semantics for projecting
helper calls as calls, but that is a surface-notation concern, not a proof
gap.


## `attention_kernel` migration

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


## Deferred attention_kernel_aligned

The user deferred this directory on 2026-10-08. Python's bf16 arm passes
out_dtype=bfloat16 to the score dot; current Triton explicitly rejects that
output type. The Lean frontend retains the same rejection. The historical
port's generic OUT_DTYPE also fails current elaboration. This is not a numeric
refutation and is not a missing two-gates rule. See the source link and planned
scope in attention_kernel_aligned/README.md. Original Python is unchanged;
new real/FP entries and independent replay remain pending.

### attention_llama: retained formula scope

The three public entries cover fp16 and int8/E5M2 sources and their carry-first
candidates. Both causal branches keep the upstream predicate
`key + start_position ≤ query`. The local `offs_n` score mask is unchanged.

The independent real formulas require whole streamed key blocks and a nonempty
visible key set for each causal row. Non-divisible tail formulas and empty
visible-row formulas remain outside this contract; active-address safety also
covers partial tails. FP8 real correctness requires finite signed-byte inputs
only on active loads; the input-array IO pins the original integer cells.
Successful flat execution and the complete raw-cell frame are proved for both
versions. No decoded-memory replacement or output equation is a premise.

FP equivalence requires finite computed sum(p) and rescaled l_prev at each
reached l_curr addition. Only add_commute is used. Nonempty execution/domain
witnesses exist for both storage modes and both causal flags. No new atomic
experiment is missing for this candidate. See verification_round.md for the
separate build/replay acceptance record.


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
