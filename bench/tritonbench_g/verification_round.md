# Correctness and FP equivalence review

This round applies the `veritile-example-review` workflow to the whole
TritonBench-G directory: 173 imported ports and 11 source-import scaffolds.
A completed example must have shared original/candidate code, independent
real formulas for both versions, a kernel-level FP equivalence using admitted
atoms, visible numerical domains, actual assumption output, compilation and
independent proof replay. A source-level audit alone does not complete a row.
Existing specialized/precomputed-input proofs retain their actual scope until
stronger statements have been proved. Missing atoms and missing structural
proofs are recorded separately; no GPU experiment is rerun by default.

The migrated layout is `Kernels.lean`, `Correct.lean`, `FPEquiv.lean`, with
supporting proofs in `Proofs/`. The public correctness and FP proofs do not
import each other. Corpus discovery, scope fingerprints and trust replay must
include all these files. No port is considered complete simply because another
member of its operator family has passed.

## Reopened precision issue

The typed-sum review reopened `bgmv_expand_slice`: its previous expansion
implicitly converted half/bfloat16 reductions to fp32. The corresponding FP
addition proof therefore covered the wrong lowering. The public source is
corrected to retain the sum dtype and specialize output-dependent promotion
before expansion; both real specifications pass their updated checks. The FP
proof remains pending. The old receipt does not certify this correction.
The padding review below reopened three cases; scalar and vector cumsum have
since passed their repairs and replay; dqkg has now passed its conditional
source repair, full-source padding refutation and independent replay as well.
Current accepted count is **59 complete, 125 pending**; earlier counts below
are historical. `attention_kernel_aligned` remains the only user-deferred case.
The fp16 `add_commute` candidate has no current admission; the choice between
new atomic experiments and retaining the original order is awaiting the user.

## Block-padding repair

The DSL previously equated omitted block-pointer padding with explicit zero
padding. It now preserves Triton 3.5.1's unspecified values. All three reopened
cases have passed their repairs and independent replay. Scalar/vector cumsum
outputs are independent of arbitrary padding; dqkg still needs its explicit
zero-oracle premise. Its complete original source stores 15 instead of the
mathematical zero in the checked V=1, 16×16-tile witness below.
See [the source inventory and correction record](block_padding_review.md).
`chunk_delta_fwd` has now passed its source repair, complete three-output real IO,
padding refutation and independent replay below. It retains the explicit
zero-unspecified-value premise.

## Additional padding source-alignment recheck

Rebuilding the completed set against current shared semantics reopened
`attention_fwd_triton1`, `attention_fwd_triton3`, `attention_kernel` and
`attention_score`. Their source-equality helpers still used zero-padding AST
nodes for unmasked block loads with no boundary-check axes. The current source
correctly retains unspecified padding. The attention_fwd_triton3 failure was
also reproduced in the snapshot preceding the signed-inference repair.

The repairs preserve the source tag and use only the fact that an unchecked,
unmasked load never selects padding. All four public real/FP entries, the
72-entry integrated rebuild, and independent replay now pass. These four cases
have returned to the completed set. No new numerical assumptions were added.
The final receipt and source-fingerprint checks are recorded below.

## Completed cases

| Directory | Real correctness | FP equivalence |
|---|---|---|
| `softmax_triton1` | Original and reciprocal candidate: independent masked softmax formula, arbitrary positive block size and row strides | `div_mul_rcp`; finite computed numerator and finite/nonzero row sum, preserving opaque `tl.exp` and reductions |
| `vector_addition` | Original and commuted candidate: `xs i + ys i`, arbitrary element count and block size | `add_commute`; finite actual loaded operands at the addition site, including evaluated masked lanes |
| `add_example` | Original and commuted candidate: `xs i + ys i`, arbitrary element count and block size | `add_commute`; finite actual loaded operands at the addition site, including evaluated masked lanes |
| `add_value` | Original and scalar-left candidate: `xs i + value`, arbitrary element count, block size and value | `add_commute` through scalar broadcasting; finite actual tile and scalar, including evaluated masked lanes |
| `triton_mul2` | Both ordinary and in-place originals and candidates: `2 * xs i`, arbitrary element count and block size; in-place output refers to the originally loaded values | `mul_commute` through scalar broadcasting, preserving each store and its aliasing; finite actual tile and interpreted 2 |
| `vector_addition_custom` | Original and candidate: `as i + bs i`, with the addition kept inside the masked store and arbitrary symbolic dimensions | `add_commute` inside the store value; finite actual operands with identical store address, mask and dtype |
| `adam_update_triton` | Original and candidate: both independent Lion parameter/momentum formulas, arbitrary sizes and scalar parameters; two distinct output regions updated in place | `add_commute` at the final momentum assignment; finite computed `diff * beta2` and loaded gradient, preserving both stores and comparisons |
| `apply_penalty` | Original and reciprocal candidate: independent repetition/frequency/presence penalty formula, symbolic gather/scatter windows; per-lane values require distinct active token addresses | `div_mul_rcp` inside `tl.where`; finite actual gathered logits and finite/nonzero loaded repetition scalar, preserving count promotion, comparison, eager arms, store order and mask |
| `attention_forward_triton` | Both typed sources erase to the independent base-2 attention weighted sum, with termination, six active-address windows and flat-memory frame; whole KV blocks and shared contiguous head layout | `div_mul_rcp` at actual fp32 row normalization, preserving fp16 storage/casts, typed dot options, exp2, reductions and store; finite accumulators and finite/nonzero denominators |
| `attention_fwd_triton2` | Both typed sources erase to the independent base-2 attention formula with full flat IO, active-address safety and frame; whole KV blocks and shared contiguous layout; partial-tail refutations retained | `div_mul_rcp` at the actual fp32 normalization; explicit V-to-fp16 cast, fp16 dot output, bf16 final cast/store, masks and failures preserved |
| `attention_llama` | Both fp16 and int8/E5M2 versions and causal flags: independent natural-exp attention, successful flat execution and complete cell frame; symbolic strides, whole streamed spans and nonempty causal visibility | `add_commute` on finite actual sum(p) and rescaled l_prev in each reached iteration, preserving fp16/fp32 precision, byte bitcasts, tl.exp, dots, masks, stores and failure |
| `attn_fwd_causal` | Both complete two-helper sources: independent finite-penalty weighted sum, symbolic shapes/strides and both defined stages; successful flat IO, active-address safety and complete frame, including repeated visits and masked padding | `div_mul_rcp` at actual fp32 final normalization, preserving fp16 Q/K/V/Out, fp32 scales/accumulation, both dot precisions, casts, helper pointer reset, masks, stores and failure |
| `attn_fwd_triton` | Both complete two-helper sources: independent finite-penalty weighted sum and full flat IO transferred through checked equality of the complete erased kernels; symbolic shapes/strides, both stages and full memory frame | `div_mul_rcp` at actual fp32 normalization, retaining bf16 Q/K/V/Out, fp32 scales/accumulation, explicit probability/V-to-fp16 casts, fp16 value-dot output, both helpers, masks, stores and failure |
| `batched_vecmat_mult` | Both complete sources: independent real double sum, symbolic whole K blocks, successful flat IO and complete memory frame | `add_commute` at each reached vecmat update; finite actual carry and block-sum operands, preserving fp32 loads, broadcast, reduction axis, transpose, stores and failure |
| `bgmv_shrink_kernel` | Both complete sources: independent scaled rank-slice GEMV, plus initial output for atomic addition; both input dtypes, explicit output storage, signed sentinel, symbolic strides and partial K tails, successful flat IO, actual bounds and full-cell frame | `mul_commute` at final fp32 scaling; finite actual post-loop accumulator and scalar, preserving products, reductions, casts, masks and both store/atomic tails |
| `block_sparse_attn` | Both complete sources: independent initial-array sparse attention formula, signed CSR visits, all three storage types, symbolic strides, all mask/channel branches, successful flat execution, actual bounds and complete frame | `mul_commute` at reached fp32 score scaling; finite actual score tile and scale, retaining typed dots, casts, intrinsic exp, normalization and stores |
| `bmm_chunk_bwd` | Both complete sources: independent initial-array masked contraction with optional residual, partial tails, symbolic strides and three program IDs; successful flat IO, actual bounds and complete frame | `add_commute` at the reached residual addition, preserving all nine input storage pairs, casts, typed dots, masks and failure |
| `bmm_chunk_fwd` | Both complete sources: independent initial-array A·Bᵀ with signed sequence filtering, whole-tile causal skipping, partial K/chunk tails, symbolic strides and three program IDs; successful flat IO, actual bounds and complete frame | `add_commute` at every reached K-loop addition, preserving all nine input storage pairs, casts, typed dots, masks, stores and failure |
| `bmm_optimized` | Both complete sources: independent batched matrix product, all eight masks and CTA groups, symbolic sizes, valid runtime grid, successful flat IO, bounds and full frame; explicit zero unspecified loads, with a checked counterexample if omitted | `add_commute` at each reached fp32 carry/dot update; common fp32/fp16/bf16 storage, TF32 disabled, output cast, masks and unspecified values retained |
| `cache_transform_triton` | Both complete gathers and index-commuted candidates: independent initial arrays, symbolic shapes/strides, successful flat IO, actual bounds, both outputs and typed-cell frame; unpadded metadata safety refuted | Exact integer-index commutation, no numerical atoms; all nine independent cosine/sine fp32/fp16/bf16 storage pairs, signed metadata, masks, full decoding tile and unspecified values retained |
| `chunk_cumsum_kernel` | Both complete sources: independent global prefix sums, all chunks and partial tails; arbitrary unspecified real values for region outputs, with flat IO safety/frame under the shared zero-oracle adapter condition | `add_commute` on the reached fp32 carry and block total; preserves storage types, explicit casts, opaque scan/sum, masks, unspecified loads, stores and failure |
| `chunk_cumsum_vector` | Both complete sources: independent per-column prefix sums, all chunks and both tails; arbitrary unspecified real values for region execution/outputs/frame, with flat IO safety under the shared zero-oracle adapter condition; symbolic strides and nonoverlapping active writes | `add_commute` at the reached post-store fp32 carry update; preserves storage types, explicit casts, TF32-disabled opaque dot, reduction, both masks, unspecified loads, stores and failure |
| `chunk_bwd_dqkg` | Both complete sources: independent whole-V dq/dk/dg formulas, arbitrary value-loop count and partial tails, active-address bounds, successful flat IO and complete tagged-cell frame; explicit zero unspecified loads (necessary by a checked full-source counterexample) and sufficient history allocation | `add_commute` on the reached fp32 dq carry and typed dot result, preserving all storage combinations, casts, dots, reductions, masks, stores and failure |
| `chunk_delta_fwd` | Both complete sources: independent delta-rule recurrence with actual rounded-up inner windows, all state flags and storage choices, partial tails, successful flat IO, active bounds and complete frame; explicit zero unspecified values, necessary by a checked full-source counterexample | `add_commute` at the reached fp32 outer carry after every inner update; casts, TF32-disabled dots, masks, stores, unspecified padding and failure are preserved |
| `chunk_gate_recurrence` | Both complete forward/backward sources: four independent initial-array real contracts, full flat execution, actual-access bounds and tagged-cell frame; symbolic fitting tile sizes, independent storage, optional seed and permitted input/output aliasing | `add_commute` at every reached fp32 update, including backward's separate final DL update; products, full DG contraction, casts, pointers, stores and failure preserved |
| `chunk_gated_attention` | Both complete cumulative and recurrent sources/candidates: independent initial-array prefix and seed/product/sum formulas, symbolic dimensions/strides, all gate/seed/final branches, successful flat IO, actual bounds and full tagged-cell frame; explicit zero unspecified values for H, with complete-source refutation if omitted | Cumulative pointer arithmetic is exact and uses no FP atom; H uses add_commute at each reached fp32 post-gate carry/dot update, retaining storage, casts, intrinsic exp, masks, stores and failure |
| `chunk_gla_fwd` | All five complete sources/candidates: ten independent real contracts, full flat execution, active bounds and tagged-cell frame; explicit zero unspecified values and stated aliasing conditions, with a complete-source padding refutation | Five complete typed FP equivalences using reached add_commute or mul_commute sites, preserving gates, dots, casts, loops, masks, stores and failure |
| `chunk_gla_simple` | Both complete sources: independent gated Q·H and causal Q·K·V formula over every K block, symbolic strides and tails, full flat execution, active bounds and tagged-cell frame; explicit zero unspecified values | mul_commute at the actual final fp32 total and scale; retains all compatible dot types, independently selected V/G storage, score cast, intrinsic exp, full K loop, store and failure |
| `chunk_linear_attn` | All four complete sources/candidates: eight independent initial-array contracts, symbolic dimensions/strides, full flat execution, active bounds and tagged-cell frame; explicit zero unspecified values and stated aliasing conditions | Four complete typed equivalences using add_commute for state and gradients, mul_commute for forward output, at actual reached operands; dots, casts, loops, masks, stores and failure retained |
| `chunk_retention` | Four complete sources/candidates with independent signed-decay formulas, eight real flat IO contracts, actual-access bounds and whole tagged-cell frame; backward-state square shape and NT>0 restriction retained | Four complete typed equivalences using reached add_commute or mul_commute; all intrinsic decay, casts, loops, stores and failures retained |
| `chunk_retention_ops` | Four complete source pairs and eight initial-array real flat contracts; descending weighted state recurrence with every history store, independent tile sizes and empty-loop coverage; three shared typed bodies checked equal to retention counterparts | Four complete typed FP equivalences using actual reached add_commute/mul_commute operands; storage, decay, scan/dot order, casts, stores and failure preserved |
| `attention_fwd_triton1` | Both sources implement the independent chunked linear-attention O formula and optional pre-update H snapshots for all flags; symbolic whole chunks and contiguous head layout; successful flat execution, address safety and frame | `mul_commute` at query scaling in every reachable iteration; finite actual loaded query and scalar, retaining fp32 casts, dot options, block pointers and both stores |
| `attention_fwd_triton3` | Both complete sources: independent six-input maximum/weighted-sum and END formulas; full flat IO, safety and frame; symbolic independent strides, grouped heads, INIT/END, signed windows and checked KV tails | `div_mul_rcp` only in END normalization; actual finite accumulator and finite/nonzero denominator after log2, preserving fp16/fp32 precision, NaN repair, casts, masks, stores and failures |
| `attention_kernel` | Both storage dtypes and versions: independent base-2 attention with scaled Q·K and both bias windows, successful flat IO and frame; whole key blocks and contiguous head layout | `div_mul_rcp` at actual fp32 normalization; both storage dtypes, storage-precision bias/scalar conversion, default fp32 dots, casts and all memory effects preserved |
| `attention_score` | Both complete sources: independent column sum of base-2 weights over finite Q/K/M inputs; symbolic positive common block size, all four flags, signed windows and independent strides; successful flat IO, active-address safety and frame | `mul_commute` at each reached fp32 qk scaling; finite actual tile/scalar, preserving fp16 Q/K/Out, fp32 M, dot, exp2, casts, masked memory effects and failure |
| `chunked_cumsum_fwd` | Both complete typed sources: independent prepared-dt and scaled-prefix-sum formulas, finite/default +infinity upper bounds, all preprocessing flags, partial tails, symbolic strides, successful finite flat execution, actual bounds and full tagged-cell frame; no unspecified-value premise | mul_commute at the reached fp32 broadcast product after the first store; independent dt/A storage, bias variants, casts, clamp, scan, masks and both stores retained |
| `context_attn_bloom` | Both complete sources: independent finite-sentinel guarded normalized-block formula, symbolic signed metadata/strides, grouped heads, gathered pages and partial tails; successful finite flat IO, actual bounds and full tagged-cell frame, without a zero-unspecified-value premise | `mul_commute` at every reached fp32 qk/scale site; preserves all three storage types, probability/output casts, tl.exp, reductions, guards, masks, stores and failure |
| `context_attn_fwd` | Both complete typed sources: independent base-2 finite-sentinel weighted sum, signed metadata/strides, grouped heads and partial tails; successful finite flat IO, actual address bounds and full tagged-cell frame | `mul_commute` on every reached eager-where fp32 qk/scale operand; preserves storage/casts, exp2, binary dot, opaque accumulator-dot, masks, stores and failure |
| `context_attn_llama` | Both complete typed sources: independent base-2 initial-array weighted sum, signed paged K/V gathers, finite sentinels, full finite flat IO and tagged-cell frame | `mul_commute` on actual reached fp32 qk/scale operands in eager where; retains casts, exp2 and opaque accumulator-dot |
| `context_attn_mistral` | Both complete typed sources: independent natural-exp causal/window weighted sum, finite -1e9 sentinels and both guards, full finite flat IO and tagged-cell frame | `mul_commute` on actual reached fp32 qk/scale operands before masks; retains casts, natural exp, both guards and dot-plus-add order |
| `context_attn_nopad` | Both full typed sources: independent causal natural-exp weighted sum, negative-infinity masking in real semantics, symbolic layouts, successful finite flat IO and full tagged-cell frame | `mul_commute` at every actual reached fp32 qk/scale site; normalized updates, casts, exp, dots, masks, stores and failure retained |
| `cosine_compute` | Both full typed sources: independent elementwise Real.cos, masked flat IO, active bounds and full frame; the shared flat adapter's zero-oracle condition is explicit | Exact integer index-product commutation, no numerical atoms; all storage branches, fp32 conversion, opaque tl.cos and destination conversion retained |
| `cross_entropy1` | Both complete forward/backward JITs and their candidates: independent loss/LSE and gradient formulas, symbolic sizes/strides, full finite flat IO and tagged-cell frame | Exact integer column-index multiplication commutation, no numerical atoms; all storage, signed labels, branches, computations, casts and accesses retained |
| `cross_entropy2` | Both complete forward/backward JITs and their candidates: independent scaled loss/LSE, gated z-loss and gradient formulas, symbolic sizes/strides, full finite flat IO and tagged-cell frame | Exact integer column-index multiplication commutation, no numerical atoms; all storage, signed labels, branches, computations, casts and accesses retained |
| `cross_entropy_ops` | Both complete forward/backward JITs and their candidates: independent scaled loss/LSE, gated z-loss and gradient formulas, symbolic sizes/strides, full finite flat IO and tagged-cell frame | Exact integer column-index multiplication commutation, no numerical atoms; all storage, signed labels, branches, computations, casts and accesses retained |
| `decay_cumsum` | Three complete JITs and candidates: six independent real scan/preparation/gradient formulas, symbolic sizes/strides, full finite flat IO and tagged-cell frame | Exact natural-number lane-mask product commutation, no numerical atoms; independent storage, unmasked final gate load and reverse pointer tails retained |
| `dequantize_matmul` | Complete signed-integer dequantization JIT and candidate: two independent integer-weight/scale product specs, symbolic strides, finite flat IO and tagged-cell frame | Integer mask product commutation, no numerical atoms; signed loads/conversion, wider scale mask and three output storage types retained; host torch.mm outside scope |
| `dequantize_rowwise` | Complete signed-integer rowwise source/candidate: two initial scale*integer-weight*factor specs, finite flat IO and tagged-cell frame | Integer block-start product commutation, no numerical atoms; masked load without other, unmasked row scale and fp16 output retained |
| `destindex_copy` | Complete signed-metadata dual scatter/candidate: two independent copy formulas, finite flat IO and tagged-cell frame | Exact integer source-address commutation, no numerical atoms; both payload storage choices, unmasked padding and first-head-only source behavior retained |
| `destindex_copy_kv1` | Complete signed-metadata masked scatter/candidate: two independent copy formulas, finite flat IO and tagged-cell frame | Exact integer source-address commutation, no numerical atoms; independent input/output casts, head and column masks and unmasked metadata retained |
| `destindex_copy_kv2` | Complete signed-metadata head-masked scatter/candidate: two independent copy formulas, finite flat IO and tagged-cell frame | Exact integer source-address commutation, no numerical atoms; independent input/output casts, head-only mask, all active columns and unmasked metadata retained |
| `diag_ssm_triton` | All four complete real/complex forward/backward sources/candidates: eight independent real recurrence/gradient contracts, all lengths including zero, successful finite flat IO and tagged-cell frames | Four exact integer-index commutations, no numerical atoms; matching real fp16/bf16/fp32 or interleaved complex64 fp32, all masks/branches/stores and failure retained |
| `dropout_triton` | Complete int32/bool keep source and candidate; independent dropout formulas, finite flat success, masked output and tagged-cell frame | Exact integer-index commutation across matching fp16/bf16/fp32 storage; implicit store casts, eager divide and failure retained; no numerical atoms |
| `embedding_triton_kernel` | Complete signed-ID lookup for arbitrary chunk sizes and the full loop; independent initial-input formula, finite flat success and tagged-cell frame | Exact integer-index commutation across matching fp16/bf16/fp32 storage, retaining all masks, casts and failure; no numerical atoms |
| `f8_conversion_utils` | Both complete typed conversion kernels and candidates; four precision-erased copy specifications, finite flat success and tagged-cell frame | Two full typed index-commutation equivalences, preserving fp8/fp16/fp32 conversions, duplicate stores, masks and failure; no numerical atoms |
| `fast_ce_loss` | Three complete original/candidate JITs, six independent finite-flat real contracts, all outputs and tagged-cell frame; capped-padding and unwritten-chunk source behaviors explicitly verified | Three whole typed equivalences, signed labels and libdevice backend preserved, integer index commutation with no numerical atoms |

| `fast_layernorm` | Both complete forward/backward sources and candidates: four independent finite-flat real contracts, all outputs and tagged-cell frame; source padded-variance counterexample retained | Two whole typed-state/failure equivalences for all 27 input/weight/bias combinations; integer row-index commutation with no numerical atoms |
| `fast_rms_layernorm` | Three complete sources and candidates: six independent finite-flat real contracts, caller-statistic backward, all outputs and frame; regular weight stride and Gemma/backward contiguous weight behavior retained | Three whole typed equivalences for nine input/weight combinations, both backward GEMMA flags, explicit normalized-input conversion to W dtype; no numerical atoms |
| `fast_rope_embedding` | Complete dynamic grouped-head loop and candidate: two independent finite-flat real contracts for both directions, odd tails and empty groups; coefficient snapshots and full frame even with Q/table aliasing | Whole typed-state/failure equivalence for 27 Q/cos/sin storage combinations, both directions, preserved sine-dtype casts and stores; no numerical atoms |

All fifty-nine retain the original masks, accesses and stores. Real contracts
include termination, active-address bounds and the memory frame; FP contracts
preserve complete state and execution failure. These are per-program results;
host launch composition, concrete IEEE identities and performance claims are
outside their scope. Experimental sizes select assumptions and do not replace
symbolic kernel dimensions.

## Validation

The earlier twenty-five-directory benchmark library built through
`TritonBenchGVerified` (2026-10-09), together with `TritonBenchSpecExamples`.
The typed-sum correction below invalidates the BGMV precision result and its
current build; that directory is reopened. Other historical validation below
predates this correction unless explicitly stated.
Each completed directory has independently replayed real/FP entries and nested
helpers. Scope classification is separate from proof checking and GPU execution.
The current public FP printers list only the atoms shown above.

The earlier attention_forward_triton batch replayed all 14 files in the port, the exact five-stream
IO adapter, the identity-rounding reciprocal safety lemma and the typed-source
regression: 17 files pass comparator replay and trust checks against a final
source snapshot for that completed case. Named target checks include both public real specifications,
the public FP equivalence, complete execution and frame, loop invariant, both
safety walks, flat IO assembly, source connections, nonempty model/domain/run
witnesses and the two tail refutations. The source-header transcription note
was included in that snapshot. The completed linear-attention case below has its own independent replay
batch, including the exact three-input/two-output IO adapter.

The earlier preparation also checked dtype erasure, typed dot options,
FP integer address arithmetic, broadcasting and affected memory/typing proofs.
Those checks established the supporting semantics; equality of two real
executions alone was not counted as independent formula correctness. The new
entry now supplies that missing formula and complete IO proof.

Nineteen tactic, matrix and guarded-execution regression tests pass. One guarded
fixture initially exceeded its 180-second limit while builds and independent
replays were running concurrently; its isolated rerun passes. Twenty-nine
inventory/layout/shard tests pass. Source gates cover all 173 imported ports;
the helper-inlining and dtype transcription exception for attention is explicit
in Kernels.lean, proof_blockers.md and completion_audit.md. The textual gate
does not certify Python-to-Lean translation for that exception.

The reviewed inventory has 383 declarations across 203 public specification
files. Source/support fingerprints, Python function links, proof-gap manifest
and generated specification sheets agree. None of these checks implies a
full-corpus proof replay in this round: 155 directories remain pending,
including the reopened BGMV case and user-deferred attention_kernel_aligned.

Reproduction entry points (finish builds before comparator snapshots):

```bash
lake build TritonBenchGVerified TritonBenchSpecExamples
python3 -m unittest scripts.test_equiv_tactics scripts.test_fp_matrix_examples scripts.test_fp_guarded_examples
python3 -m unittest scripts.test_coverage_review scripts.test_port_layout scripts.test_audit_shards
bench/audit_trust.sh attention_forward_triton attention_fwd_triton1 attention_fwd_triton2
python3 scripts/check_comparator.py --file VeriTile/Triton/Math/BroadcastRewriteR.lean --trust
python3 scripts/check_comparator.py --file VeriTile/Triton/Memory/KernelSpec/StreamAttentionReal.lean --trust
python3 scripts/check_comparator.py --file bench/tests/FPAttentionSource.lean --trust
bench/audit_tritonbench_g.sh --source-only
python3 bench/coverage_review.py
python3 bench/check_proof_gap_manifest.py
```

## Attention scope and preserved counterexample

`attention_forward_triton` is complete for its displayed contracts. Correct.lean
shows the five input streams, output window, masks, dimensions and independent
weighted-sum formula. FPEquiv.lean exposes the actual post-loop numerical domain
and compares the complete typed programs. Kernels.lean contains both sources.
Historical rounding statements remain in Proofs/LegacyRounding.lean.

The real claim requires `N_CTX = BLOCK_N * numKVBlocks`, positive key-block size
and count, shared contiguous head strides, and active head lanes fitting the
block and output row. The real IO contract fixes undefined masked-load values
to zero. The source's partially filled KV blocks mask K/V loads but
not their scores. Proofs/TailCounterexample.lean checks successful execution of
both complete real sources with `N_CTX=1`, `BLOCK_N=2`, zero Q/K and unit V:
both store `1/2` while standard length-one attention is `1`. This retains the
128/96 head layout and uses a small symbolic Lean block, not a GPU launch.

The FP candidate changes final division to reciprocal multiplication. Its only
numerical atom is `div_mul_rcp`; neither dot nor reduction needs a new identity.
Proofs/TypedExecution.lean supplies a shared model of the admitted rules and
successful nonempty executions with a satisfied domain. It is an abstract Lean
witness, not another admission experiment.

General frontend precision work remains for future ports: three-argument
`tl.dot(a, b, acc)` is still a dot-plus-add desugaring and needs fused-dot FP
semantics; typed fp64 dot and general low-precision division promotion are not
covered by this batch. This attention source uses two-argument dot followed by
an explicit fp32 accumulator addition and fp32 normalization.

## Linear-attention case

`attention_fwd_triton1` now has the shared `Kernels.lean`, `Correct.lean` and
`FPEquiv.lean` entries. It is chunked linear attention. Both flags, all five
block pointers, six casts and `allow_tf32=False` dots remain visible; the
candidate only swaps the operands of query scaling.

Correct.lean displays the three Q/K/V streams, aggregate O/H output windows,
all masks, buffer separation and independent formulas. For each chunk c,
`H_c = sum_{j<c} K_j^T V_j` and
`O_c = ((scale Q_c) K_c^T) V_c + (scale Q_c) H_c`. STORE saves H before its
update; when disabled all H memory stays unchanged. IFCOND skips only the
zero carry contribution in the first chunk. Both complete sources satisfy
these formulas, successful flat-memory execution, active-address safety and
the frame outside both active output windows.

The real scope is `s_qt=BD`, `s_qd=1`, `T=NT*BT`, `BT>0` and `BD≤s_ht`, with
O and H distinct from each other and Q/K/V. All dimensions remain symbolic.
Unmasked block pointers do not justify partially allocated tails. H bounds
are required only when STORE is enabled. These are per-program contracts.

The proof helpers cover source erasure, the first-chunk branch, matrix stores,
complete iteration and loop invariants, candidate execution, access safety
and exact three-input/two-output IO transport. The invariant also records the
actual `.real` output cells, including zero-valued H snapshots. The exact IO
contract observes `readMem`; it does not assume a typed readback bridge.
Historical default-branch and rounding proofs remain in Proofs/Legacy.lean.

FPEquiv.lean compares the complete typed sources through `equiv_decompose`
and `fp_prove` in its helper. Its only numerical atom is `mul_commute`.
The domain checks the actual loaded Q lanes and interpreted scalar at every
reachable iteration after the local prefix, including any preceding H store.
Both flags, every source dimension/stride, full state and failure are retained.
No dot, carry-update, reassociation or cast-identity assumption is used.

The admitted-rule model and two-iteration execution witnesses establish a
nonempty domain for both sources and every flag combination. Their opaque
zero dots establish consistency only. A separate nonzero real witness gives
O=[1,2] and optional pre-update H=[0,1], preserving untouched cells. These
small Lean witnesses do not constrain general theorem dimensions or claim
GPU launchability.

The six Python cast targets (H pointer element type, four Q register types,
O pointer element type) are all fp32 in the supplied tests. They are written
explicitly as `tl.float32` because the legacy alias shorthand erases floating
casts. This mapping is reviewed and registered as a transcription exception;
the generic lexical gate does not verify Python dtype resolution.

Validation at completion: both benchmark/example libraries built (3632 jobs),
29 inventory/layout/shard tests pass, and all 173 source gates pass with the
documented transcription exceptions. All 16 files in this directory and the
exact streamed IO adapter pass independent comparator replay and trust checks.
The batch verified 55 named targets and 614 source inputs, including both
public real specifications, public FP equivalence, all-flag executions,
access safety, flat IO, raw-cell frames and the nonempty model/run witnesses.
The six cast targets and all load/store expressions also match after applying
only the three documented fp32 dtype-alias resolutions.

## Completed attention_fwd_triton2 batch

`attention_fwd_triton2` now has both complete typed sources and separate
Correct.lean/FPEquiv.lean entries. Q/K/V are fp16, both scales are fp32 and
the wrapper allocates bf16 output. The inner source differs from
attention_forward_triton only in its explicit V-to-fp16 cast; the outer kernel
AST is identical. Both the nested cast and bf16 terminal cast/store are
retained, with fp16 dot output promoted into the fp32 accumulator.

The real proof checks equality of both complete erased kernels before reusing
the previous attention IO helpers. Correct.lean still displays its own five
input streams, output window, masks and independent base-2 attention formula.
The transferred contract includes successful flat execution, active-address
safety and frame. It covers positive whole KV blocks with shared batch/head
strides and contiguous head lanes; source dimensions remain symbolic.

The FP proof compares this directory's own original and reciprocal candidate
with `equiv_decompose` and `fp_prove`. Its sole atom is `div_mul_rcp`, applied
to actual finite accumulator lanes and finite/nonzero row denominators after
the complete loop. Casts, typed dots, exp2, reductions, masks, full state and
failed executions are preserved. An admitted-rule model, satisfied domain and
successful nonempty executions rule out an empty-premise argument.

The partial-tail limitation is carried through to both sources: at N_CTX=1,
BLOCK_N=2, zero Q/K and unit V, successful real execution stores 1/2 rather than
standard attention's 1. Both unqualified tail specifications are refuted.
This uses the 128/96 head layout and establishes mathematical source behavior,
not GPU launchability. Historical rounding theorems stay in Proofs/LegacyRounding.lean.

Both libraries build (3641 jobs), 29 inventory/layout/shard tests and the
new source-precision regression pass, and all 173 source gates pass with their
documented transcription exceptions. Coverage and proof-gap inventories agree
on 365 declarations across 184 public files. All nine files in this directory
and the precision regression pass independent comparator replay and trust
checks: 27 named targets were verified, and all 623 snapshot inputs match the
final source. This includes both public real specifications, the public FP
equivalence, source-erasure connections, nonempty model/domain/execution
witnesses, both tail refutations and the cast/store precision checks.
The corpus count is **11 complete, 173 pending**.

## attention_fwd_triton3 repair history

The repair began with the following source mismatches. The historical Lean program
uses natural-truncated distances, omits window loop bounds/initial pointer
advance, KV boundary checks and tail masks, the NaN alpha correction and the
probability cast, and adds a scale multiplication. Its existing proofs cannot
be transferred as correctness of the Python source.

`Proofs/SourceReview.lean` finds the actual historical distance assignment and
checks the row-0/key-1 counterexample: natural distance 0 retains that key,
whereas signed distance -1 rejects it. A Lean `refutation` rejects universal
agreement of these masks. The coverage scopes, README and transcription
exception now state this difference explicitly.

Supporting work adds exact signed comparisons and runtime signed block-pointer
advancement to the FP interpreter, preserving missing-coordinate failure, real
and rounding semantics, dtype erasure, typing, memory locality and flattening.
No numerical atom is added. `Proofs/WindowBounds.lean` establishes agreement
of the clipped block-aligned bounds for truncation-to-zero and Euclidean
division when BLOCK_N is positive, together with loop-test and step conversion.
The real-correctness work still requires connecting this arithmetic to the
symbolic signed loop, with INIT/END, continuation, tail and O/M/L proofs.
The completion count remains **11 complete, 173 pending**.

Foundation validation: the existing example/verified-port libraries and the
in-progress source review build successfully (3644 jobs), and WindowBounds
builds separately. All 23 tactic/matrix/domain regressions, 29 inventory/layout/
shard tests and the 173-port source gate pass with the registered transcription
exceptions. The latter is not a Python/Lean correspondence proof.

Nineteen changed/supporting files pass independent comparator and trust checks,
including 25 named targets for dynamic advancement, signed comparisons, memory
flattening/locality, the source-mask refutation and window bounds. A pre-existing
local name `full` in the frontend conflicted with the audit parser; its four
local occurrences were renamed without changing the generated expression.
The frontend and its two dependent checks were replayed against a fresh snapshot.
All 626 inputs of that final snapshot match the workspace; the previous snapshot
differs only by that checked local rename. No numerical atom or GPU experiment
was added in this step.

### Signed range support

The core and compute ASTs now have `forRangeInt`: signed bounds and index, with
a nonnegative stride evaluated once on entry. Real, rounding-parametric and
opaque-FP interpreters preserve negative indices, empty intervals and failures.
The frontend selects this node for signed bounds instead of clipping them to
natural numbers. Dtype erasure, region typing, trace safety, locality and the
real/rounded flat-memory bridges retain the signed index.

`bench/tests/SignedRange.lean` exercises the actual frontend and both execution
paths. A loop over -3, -1, 1 changes its own lo/hi/stride registers inside the
body; its sum is still -3. Empty intervals, missing bounds, body failure,
signed constexpr bounds, typing and erasure have separate regression checks.
This is exact control semantics; no numerical atom is added. The later sections
record the runtime signed-division correspondence, repaired source and FP
proofs. Independent real correctness for all INIT/END/output behavior remains
pending.

Signed-range validation: the example/verified-port/in-progress build passes
3645 jobs and the complete core library passes 3505 jobs. All 24 tactic,
matrix and guarded-domain regressions pass, as do the 29 inventory/layout/shard
tests and the 173-port source gates with their documented exceptions. Eighteen
changed/supporting files pass independent comparator replay and trust checks;
37 named targets and all 627 snapshot input hashes were checked. The replay
includes both public attention_fwd_triton2 specifications and the historical
attention_fwd_triton3 mask refutation, not a new correctness claim for that port.

### Runtime integer division and conditional normalization

Runtime signed `//` and `%` now use `Op.truncDiv` and `Op.truncMod`, constructed
from exact integer operations. Their real, rounding-parametric and opaque-FP
evaluation is proved equal to `Int.tdiv`/`Int.tmod`, including operand failures
and arbitrary broadcast shapes. Natural-number lowering and the raw Euclidean
integer primitives retain their previous meaning. Signed Python constexpr
arithmetic is explicitly computed in Lean antiquotations with `Int.fdiv` and
`Int.fmod`; the frontend does not infer Python constexpr-ness. Fixed-width
overflow and invalid GPU division remain outside this mathematical-Int claim.

`attention_fwd_triton3/Proofs/TypedSource.lean` now contains both complete typed
programs. `Proofs/FP.lean` proves their conditional full-state equivalence,
using only `div_mul_rcp` at final normalization. The common context includes
grouped heads, INIT/END arms, signed sliding windows, tails, NaN alpha repair,
casts and all M/L/Out stores. A new branch site checks the actual fp32 operands
only when the END arm is selected. `equiv_decompose` descends into a changed
then or else arm when its domain is supplied; `fp_prove` composes the numeric
step. Conditions, the other arm and failures are preserved.

The scalar model in `Proofs/FPModel.lean` demonstrates consistency of the
admitted arithmetic assumptions. `Proofs/TypedExecution.lean` now checks
successful full-source and candidate executions in that same model, together
with the actual normalization domain. Independent real correctness with IO and
frame conditions remains pending. The public trio is therefore not published and
the completion count remains **11 / 184**, with **173 pending**.

New regressions check runtime versus constexpr division, negative operands,
broadcast and missing-register behavior; branch guards and their failure
boundaries; and the actual signed distance and range in the replacement
attention source. All 22 tactic and 5 matrix/domain tests pass. The updated
source's row-0/key-1 distance evaluates to -1, addressing the historical
natural-subtraction discrepancy.

The final runtime-integer/branch validation independently replays 18 files and
checks 57 named targets, with all 637 source-snapshot input hashes matching.
This includes the full-source FP theorem, its scalar-model consistency proof,
integer semantics, branch rewriting and affected completed attention entries.
`VeriTile`, `TritonBenchGVerified`, `TritonBenchSpecExamples` and
`TritonBenchGInProgress` build together (3689 jobs). The 29 inventory/layout/shard
tests, source gates, coverage review and proof-gap manifest also pass. These
checks validate the stated groundwork; they predate the execution witnesses
described below.

### Broadcast masks and nonempty executions

The full-source execution check found that the KV-tail condition lowers to
`Op.remap` for singleton-dimension broadcasting. The opaque-FP interpreter
previously failed on this node. It now evaluates the operand at the unchanged
precision and reindexes its values, preserving dtype and missing-operand
failure. Regressions cover fp16 values, nested FP operations, absent masks and
empty output shapes, plus the actual source mask across all rows.

`Proofs/TypedExecution.lean` constructs complete original and candidate runs
in the consistent `FPModel` backend. The fixture has two query heads sharing
one KV head, 15 valid Q/KV rows in 16-lane blocks, INIT/END and a signed sliding
window with one nonempty iteration. The END-arm M/Out stores execute, and the
actual row denominator after the log2 update is 2. The same model and initial
state satisfy the admitted scalar assumptions and every normalization guard.
These concrete dimensions witness satisfiability; they do not narrow the
symbolic FP theorem. Independent real formulas, symbolic source execution,
address safety and the full memory frame remain pending.

The final broadcast/execution batch builds all four libraries (3690 jobs).
All 28 tactic/matrix/domain tests and 29 inventory/layout/shard tests pass, as
do the source gates, coverage review and proof-gap manifest. Independent
comparator replay and trust checks accept 11 files, including the new complete
execution witness and affected attention_fwd_triton2 entries. All 72 named
targets are present in the checked theorem inventory, and all 639 input hashes
match the final source snapshot. The FP atom printer still lists only
`div_mul_rcp` for the repaired attention source. No GPU experiment or admission
entry changed.

## Per-directory progress

| Directory | Status |
|---|---|
| `adam_update_triton` | verified per-program: real + admitted FP |
| `add_example` | verified per-program: real + admitted FP |
| `add_value` | verified per-program: real + admitted FP; legacy rounding preserved |
| `apply_penalty` | verified per-program: real + admitted FP |
| `attention_forward_triton` | verified per-program: both real specs on whole KV blocks and shared contiguous layout; admitted FP reciprocal equivalence; partial-tail standard-attention refutations retained |
| `attention_fwd_triton1` | verified per-program: both real specs for all flags on whole contiguous chunks, optional pre-update H snapshots, admitted FP query-scaling equivalence |
| `attention_fwd_triton2` | verified per-program: both real specs on whole KV blocks, admitted FP reciprocal equivalence retaining the V cast and bf16 output; partial-tail refutations |
| `attention_fwd_triton3` | verified per-program: both real specs for symbolic independent strides, grouped heads, INIT/END and signed windows; admitted FP normalization equivalence and nonempty witnesses |
| `attention_kernel` | verified per-program: both storage dtypes and both real specs on whole key blocks and contiguous head layout; admitted FP reciprocal equivalence and nonempty witnesses |
| `attention_kernel_aligned` | deferred by user: original bf16 score dot requests unsupported out_dtype; revisit source repair and real/FP migration later |
| `attention_llama` | verified per-program: fp16 and int8/E5M2 originals/candidates, both causal flags, independent real formula and full flat IO/frame; admitted add_commute with per-iteration domains and nonempty witnesses |
| `attention_score` | complete: signed source, independent real flat IO and admitted FP equivalence |
| `attn_fwd_causal` | complete: both source-correct real flat IO contracts and admitted FP equivalence independently replayed; finite-penalty exact-mask refutation preserved |
| `attn_fwd_triton` | complete: bf16 full source, both real flat IO contracts, admitted FP equivalence and exact-mask real refutation independently replayed |
| `batched_vecmat_mult` | complete: both full real flat IO contracts, admitted add_commute equivalence, nonempty execution witnesses and source regression independently replayed |
| `bgmv_expand_slice` | reopened: correct half-sum dtype and output-dependent promotion; refresh real/FP proofs and admission scope |
| `bgmv_shrink_kernel` | completed: full real flat IO and admitted FP equivalence |
| `bmm_chunk_bwd` | complete: both independent real flat IO contracts, all partial tails and residual arms, nine input storage pairs, admitted add_commute and nonempty executions independently replayed |
| `bmm_chunk_fwd` | complete: full source, independent real flat-IO contracts, reached-loop admitted FP equivalence, regressions and 34-file replay accepted |
| `bmm_optimized` | complete: all mask/source variants, independent real flat IO with explicit zero-undefined premise and checked counterexample, admitted FP equivalence, regressions and 34-file replay accepted |
| `cache_transform_triton` | complete: both full gathers, independent real flat IO, both outputs and typed frame, all nine storage pairs, FP equivalence without atoms, unpadded metadata safety refutation and 38-file independent replay accepted |
| `chunk_bwd_dqkg` | complete: whole-V real flat IO, all outputs and full frame, admitted add_commute, bounds regressions and 22-file independent replay accepted |
| `chunk_cumsum_kernel` | complete: full global real prefix sum and flat IO, admitted carry commutation, source/scan regressions and 15-file independent replay accepted |
| `chunk_cumsum_vector` | complete: full per-column real prefix sums and flat IO, both tails, admitted carry commutation, source/layout regressions and 18-file independent replay accepted |
| `chunk_gate_recurrence` | complete: four whole-source real flat IO contracts, two admitted FP specifications and verified whole-case replay |
| `chunk_gated_attention` | complete: four whole-source real flat IO contracts, two FP equivalences, explicit padding premise/refutation and 39-file independent replay accepted |
| `chunk_gla_fwd` | complete: all five source/candidate pairs, ten full real flat IO specifications, five admitted FP equivalences, padding refutation and 82-file independent replay accepted |
| `chunk_gla_simple` | complete: full source/candidate, two real flat IO specifications over all K blocks, mul_commute FP equivalence, mixed-storage execution and 13-file independent replay accepted |
| `chunk_linear_attn` | complete: four full source/candidate pairs, eight independent real flat IO specifications, four admitted FP equivalences, concrete regressions and 19-file independent replay accepted |
| `chunk_retention` | complete: four full typed source/candidate pairs, eight real flat IO contracts and four admitted FP equivalences; 20-file independent replay and all concrete regressions accepted |
| `chunk_retention_ops` | complete: four full typed source pairs, eight real flat IO contracts, four admitted FP equivalences, three-chunk and empty-loop regressions, and 10-file independent replay accepted |
| `context_attn_bloom` | complete: both full real flat IO specifications, admitted FP equivalence, typed/signed source regression and independent replay |
| `context_attn_fwd` | complete: full typed source, both real flat IO specifications, admitted FP equivalence, source regressions and independent replay |
| `context_attn_llama` | complete: both real flat IO specifications, admitted FP equivalence, inhabited execution/domain witnesses and independent replay |
| `context_attn_mistral` | complete: both real flat IO specifications, admitted FP equivalence, inhabited execution/domain witnesses and independent replay |
| `context_attn_nopad` | complete: full real/FP contracts, source/type checks, execution witnesses and independent replay |
| `cosine_compute` | complete: full real/FP contracts, source/type checks, execution witnesses and independent replay |
| `cross_entropy1` | complete: both JITs, four real specs, two FP equivalences, inhabited executions, source checks and independent replay |
| `cross_entropy2` | complete: both JITs, four real specs, two FP equivalences, inhabited executions, source checks and independent replay |
| `cross_entropy_ops` | complete: both JITs, four real specs, two FP equivalences, inhabited executions, source checks and independent replay |
| `decay_cumsum` | complete: three JITs, six real specs, three FP equivalences, inhabited executions, source checks and independent replay |
| `dequantize_matmul` | complete: signed-int JIT/candidate, two real specs, FP equivalence, finite/model executions, source checks and independent replay |
| `dequantize_rowwise` | complete: signed-int source/candidate, two real specs, FP equivalence, finite/model executions, source checks and independent replay |
| `destindex_copy` | complete: signed-metadata dual scatter/candidate, two real specs, FP equivalence, finite/model executions, source checks and independent replay |
| `destindex_copy_kv1` | complete: signed-metadata masked scatter/candidate, two real specs, FP equivalence, finite/model executions, source checks and independent replay |
| `destindex_copy_kv2` | complete: signed-metadata head-masked scatter/candidate, two real specs, FP equivalence, finite/model executions, source checks and independent replay |
| `diag_ssm_triton` | complete: all four typed JITs/candidates, eight real specifications, four FP equivalences, finite/model witnesses, source/type checks and independent replay |
| `dropout_triton` | complete: full typed source, original/candidate real contracts, FP equivalence, finite/model witnesses, source/type checks and independent replay |
| `embedding_triton_kernel` | complete: full signed-ID source, arbitrary chunk loop, real finite-IO contracts, FP equivalence, finite/model witnesses and independent replay |
| `f8_conversion_utils` | complete: both typed conversions, four real specifications, two FP equivalences, exact cast/mask checks, finite/model witnesses and independent replay |
| `fast_ce_loss` | complete review: three full sources, six real finite-IO specifications and three atom-free FP equivalences; source-behavior refutations, finite/typed witnesses, 55-case regressions and 163-file replay verified |
| `fast_layernorm` | complete review: original/candidate source, full finite real IO, typed FP equivalence, precision and execution witnesses, repository gates and independent trust replay verified |
| `fast_rms_layernorm` | complete review: original/candidate source, full finite real IO, typed FP equivalence, precision and execution witnesses, repository gates and independent trust replay verified |
| `fast_rope_embedding` | complete review: original/candidate source, full finite real IO, typed FP equivalence, precision and execution witnesses, repository gates and independent trust replay verified |
| `fifth_order_sph_harmonics` | pending review |
| `flash_attn` | pending review |
| `flash_decode2_llama` | pending review |
| `flash_decode2_phi` | pending review |
| `fp4_to_bf16` | pending source import |
| `fp4_to_bf16_conversion` | pending source import |
| `fused_activation` | pending review |
| `fused_layernorm_triton` | pending review |
| `fused_recurrent_delta` | pending review |
| `fused_recurrent_hgrn` | pending review |
| `fused_recurrent_retention` | pending review |
| `fused_rotary_embedding` | pending review |
| `fused_rwkv6_kernel` | pending review |
| `geglu_tanh_triton` | pending review |
| `index_select_bwd` | pending review |
| `index_select_cat` | pending review |
| `int4_matmul` | pending review |
| `int8_dequant_matmul` | pending review |
| `int8_matmul_kernel` | pending review |
| `int8_matmul_quantization` | pending review |
| `int8_quantization` | pending review |
| `int_scaled_matmul` | pending review |
| `isfinite_kernel` | pending source import |
| `iv_dependent_matmul` | pending review |
| `kcache_copy_triton` | pending review |
| `kldiv_compute` | pending review |
| `kldiv_ops` | pending review |
| `kldiv_triton` | pending review |
| `ksoftmax_triton` | pending review |
| `kv_cache_copy` | pending review |
| `kv_cache_filling` | pending review |
| `l2_norm_bwd` | pending review |
| `l2_norm_triton1` | pending review |
| `l2_norm_triton2` | pending review |
| `layer_norm_fwd` | pending source import |
| `layer_norm_liger` | pending review |
| `layer_norm_ops` | pending review |
| `layer_norm_triton` | pending source import |
| `layer_norm_welfold` | pending review |
| `layernorm_fwd_triton` | pending review |
| `lightning_attention` | pending review |
| `llama_ff_triton` | pending review |
| `log_softmax` | pending review |
| `logsumexp_fwd` | pending review |
| `lora_expand_gemv` | pending review |
| `masked_add_cuda` | pending review |
| `masked_select` | pending review |
| `matmul_dequant_int4` | pending review |
| `matmul_dequantize` | pending review |
| `matmul_dequantize_int4` | pending review |
| `matmul_kernel` | pending review |
| `matmul_leakyrelu` | pending review |
| `matmul_leakyrelu_fp8` | pending review |
| `matmul_persistent_triton` | pending source import |
| `matmul_tma` | pending review |
| `matmul_triton1` | pending review |
| `matmul_triton2` | pending review |
| `matmul_triton_autotune` | pending review |
| `matrix_reduction` | pending review |
| `matrix_transpose` | pending review |
| `matrix_vector_multip` | pending review |
| `max_reduction` | pending review |
| `mean_reduction` | pending review |
| `mixed_sparse_attention` | pending review |
| `mul_exponent_compensator` | pending review |
| `multinomial_sampling` | pending source import |
| `nested_loops_processing` | pending review |
| `parallel_attention` | pending review |
| `parallel_retention_attention` | pending review |
| `pow_scalar_tensor` | pending review |
| `quant_transpose_kernel` | pending review |
| `quantize_copy_kv` | pending review |
| `quantize_global` | pending review |
| `quantize_kv_copy` | pending review |
| `quantize_kv_transform` | pending review |
| `rbe_triton_transform` | pending review |
| `relu_strided_buffer` | pending review |
| `relu_triton_kernel` | pending review |
| `reversed_cumsum` | pending review |
| `reversed_cumsum_scalar` | pending review |
| `rms_matmul_rbe` | pending review |
| `rms_norm_triton` | pending review |
| `rms_rbe_matmul` | pending review |
| `rmsnorm_fused` | pending review |
| `rmsnorm_fused_llama` | pending review |
| `rmsnorm_implementation` | pending review |
| `rmsnorm_triton` | pending review |
| `rope_backward_transform` | pending review |
| `rope_embedding` | pending review |
| `rope_transform` | pending review |
| `rotary_emb` | pending review |
| `rotary_emb_nopad` | pending review |
| `rotary_transform` | pending review |
| `rotary_transform_ops` | pending review |
| `rowwise_quantization_triton` | pending review |
| `seeded_dropout` | pending source import |
| `sgmv_expand_slice` | pending review |
| `sin_computation` | pending review |
| `sin_kernel` | pending review |
| `softmax_flaggems` | pending review |
| `softmax_optimize` | pending review |
| `softmax_reducev` | pending review |
| `softmax_triton1` | verified: both real specs and conditional FP equivalence (per-program scope) |
| `softmax_triton2` | pending review |
| `softmax_triton3` | pending review |
| `spinning_lock_reduction` | pending source import |
| `square_matrix` | pending review |
| `streamk_matmul` | pending source import |
| `swiglu_backward` | pending review |
| `swiglu_fwd` | pending review |
| `swiglu_triton` | pending review |
| `token_attn_llama2` | pending review |
| `token_attn_mistral` | pending review |
| `token_attn_reduceV` | pending review |
| `token_softmax_bloom` | pending review |
| `token_softmax_llama` | pending review |
| `triton_argmax` | pending review |
| `triton_attention` | pending review |
| `triton_conv2d_fwd` | pending review |
| `triton_linear_activation` | pending review |
| `triton_matmul` | pending review |
| `triton_mul2` | verified per-program: real + admitted FP |
| `triton_softmax` | pending review |
| `uniform_sampling` | pending source import |
| `var_len_copy` | pending review |
| `vector_addition` | verified per-program: real + admitted FP |
| `vector_addition_custom` | verified per-program: real + admitted FP |

### Independent real formulas and signed-window key coverage

The 2026-10-08 attention_fwd_triton3 groundwork adds three real-side helpers
and extends WindowBounds. RealFormula defines independent maximum/weighted-sum
formulas with initial or resumed M/L/Out carry and END-dependent writeback.
Its block algebra equals the closed formula; normalized outputs reduce to an
unshifted weighted sum even at a zero denominator in the total real semantics.
No loop iteration and an executed all-masked block remain distinct for M/L.

RealKeys proves that clipping the window and splitting it into BLOCK_N lanes,
then applying the row and KV-tail masks, gives exactly the full-context kept
key list in the same order, without omissions or repetitions. It covers all
window flags, arbitrary signed offsets/sizes and partial final blocks. The
proof reduces blocked fresh/resumed outputs to the full-context row formulas.
It is a mathematical key partition, not a claim of successful kernel execution.

RealExpressions locates the distance assignment and both mask arms in each
complete TypedSource AST. It proves evaluation and successful execution of
the distance/mask fragment and connects its result to RealKeys.keep. The
complement arm is dist >= size, rather than the Boolean negation of the
ordinary two-sided window. No FP assumption or numerical domain was changed.

Whole-kernel Q/K dot evaluation, the online state invariant, complete M/L/Out
execution, active-address safety and the flat-memory frame remain outstanding.
The new helpers therefore do not change the corpus count: **11 complete,
173 pending**. They stay under Proofs until the full port is verified.

Validation for this real-foundation batch: `lake build TritonBenchGInProgress`
passes (3446 jobs), all 29 inventory/layout/shard tests pass, and the 173-port
source gates, coverage review and proof-gap manifest agree. All four real
helper files pass independent comparator replay and trust checks; their 57
named theorem targets are present in the actual replay inventory and all 642
snapshot input hashes match the final sources. The replay logs are under
`Logs/comparator-check-subzu00_`. No shared numerical semantics, FP rules or
GPU admission results were changed in this batch.

### Source loop and real numerical execution fragments

The next attention_fwd_triton3 step connects the complete erased loop in both
TypedSource kernels to named parts in RealBody. The connection retains all
flags, both qk/p masks, NaN-repair syntax, loads and pointer advances. Selecting
the loop before erasure avoids expanding unrelated signed address arithmetic;
a generic theorem checks that this selection commutes with erasure.

RealTiles connects finite Q/K and P/V dots, masked max and sum, exponentiation
and rescaling to RealFormula. The kept key set may be empty; only a max
reduction's physical dimension must be positive. The real self-inequality is
false even at bottom, while the separate FP model keeps that comparison opaque.

RealNumeric proves actual execution of the Q/K sum-product/scale prefix,
KV-tail mask evaluation and all seven rescaling assignments. The rescaling
proof preserves memory and every register outside its write set. Its
rescale_denominator_formula theorem connects successful execution to the
independent denominator; the accumulator at that point is still the scaled
carry before the V load/dot.

These are symbolic fragment proofs with explicit initial-register values.
Preceding loads must establish those values, and the mask/probability paths,
V load/accumulation, pointer advances and all iterations still need composition.
Initialization, END-dependent writeback, active-address safety and the full
flat-memory frame remain outstanding. No port is newly counted complete:
**11 complete, 173 pending**. No FP atom or admission result changed.

Validation: `TritonBenchGInProgress` builds successfully (3449 jobs), the 29
inventory/layout/shard tests pass, and source gates, coverage review, proof-gap
manifest and whitespace checks pass. All three new helper files pass independent
comparator replay and trust checks. Their 27 named theorem targets are present
in the actual replay inventory; all 645 snapshot input hashes match the final
workspace sources. Replay logs are under `Logs/comparator-check-s3_mfqc0`.

### Complete real loop iteration

The next attention_fwd_triton3 step composes the entire checked source body.
RealScores executes the Q/K sum-product, scale, signed regular/complement
window and KV-tail masks. Its mask-to-key theorem states the necessary valid
lane condition when the source disables boundary checks. It does not impose
the experimental shape on this derivation.

RealProbabilities executes the row maximum, shift, exp2 and both subsequent
probability masks. The implications needed by the second masks follow from
the actual score predicates and are discharged in the composition theorem.
The physical reduction dimension is positive, but an entire row can be masked
and the previous maximum can be bottom.

RealAccumulation executes the actual K/V load choices, P/V dot, maximum carry
and pointer advances. Values come from the block-pointer memory view, with
checked out-of-bounds lanes zero-filled in the real interpreter. RealIteration
then proves successful execution of the complete body, giving independent
center, denominator and numerator formulas, updated pointers, unchanged memory
and a register frame. Q and the initial carry are entry-register premises;
loaded K/V values, scores and probability outputs are derived by execution.

The remaining real obligations are signed-loop induction, connection to the
full-context key formula, Q loading, INIT/END initialization and writeback,
active-address safety and the full flat-memory contract. The new body theorem
does not claim these obligations. Corpus status stays **11 complete, 173
pending**. No FP rule, numerical semantics or admission result changed.

Validation: `lake build TritonBenchGInProgress` passes (3453 jobs), and all
29 inventory/layout/shard tests pass (8.839 seconds). Source gates, the 365
coverage reviews across 184 source pairs, proof-gap manifest and whitespace
checks pass. All four new files pass independent comparator replay and trust
checks; their 23 named theorem targets occur in the actual replay inventory,
and all 649 snapshot input hashes match the final sources. The batch logs are
under `Logs/comparator-check-x62b3x0a`.

### Whole signed loop and full-context real formula

The attention_fwd_triton3 loop proof now starts from its entry registers and
executes every iteration of the checked source body. RealLoopFormula defines
independent maximum/weighted-sum formulas after a prefix of blocks, with one
shared center/denominator per row and a numerator for each output lane. It
proves the update law without evaluating statements or folding machine states.

RealLoop instantiates the complete body theorem with actual K/V pointer tiles,
including independent strides and base offsets, then proves the signed-range
invariant and successful execution to the closed formulas. It preserves Q,
scale and start_m, advances both pointers, and proves the register/memory frame.
Negative upper bounds, empty loops and executed all-masked blocks retain their
distinct semantics. Only BLOCK_N must be positive in this loop proof.

RealLoopContext connects the pointer views to ordinary global K/V indexing,
then applies the ordered key partition to remove blocking from the final
center, denominator and numerator formulas. The source's aligned lower bound,
capped upper bound and launch condition `NKV_CTX % BLOCK_N = 0` when IS_EVEN_N
is enabled imply that every unchecked KV block is complete. Checked tails
remain allowed. The final formula is over the kept keys of the entire context.

RealLoopSource checks the complete for-range statement in both typed kernels,
including lo/hi register reads and the BLOCK_N stride, and proves execution
of that statement to the full-context formulas. Its register, pointer and bound
premises are loop-entry obligations for the actual prologue. They do not assume
successful loop execution or any kernel-output equation.

Remaining work is source prologue/initialization (Q load, INIT/continuation
M/L/Out, exact scale, grouped-head offsets, signed bounds and pointer setup),
END-dependent stores, active-address safety and the complete flat-memory IO
contract. This is still an in-progress port: **11 complete, 173 pending**.
No FP atom, experimental admission or shared numerical semantics changed.

Validation: `TritonBenchGInProgress` builds (3457 jobs), and all 29 inventory/
layout/shard tests pass (8.819 seconds). Source gates, coverage review (365
reviews, 184 source pairs), proof-gap manifest and whitespace checks pass.
All four new files pass independent comparator replay and trust checks: 33
named targets are present in the actual replay inventory and all 653 input
hashes match the final workspace. Logs: `Logs/comparator-check-k3eh8j70`.


### Carry initialization and complete window selection

The next attention_fwd_triton3 batch proves the actual INIT/continuation carry,
exact decimal scale and both Q load modes. RealInit obtains continued M/L/Out
values from memory at the supplied pointers; it does not assume a resulting
carry equation. RealInitSource checks these complete fragments against both
full typed implementations. Pointer construction remains an earlier obligation.

RealWindow executes the entire source window-selection branch. Signed runtime
division, lower clamping, upper capping, the alignment hint and both dynamic
K/V advances are retained. The complement mode uses the full context, and the
nested complement arm excluded by the outer condition remains in the checked
AST. Offset and size remain arbitrary signed integers; a negative upper bound
is preserved for the actual loop test. The enters_loop theorem establishes the
exact lo/hi registers and K/V metadata needed by RealLoop from the unadvanced
pointers. RealWindowSource checks the whole branch in both implementations.

Remaining work is grouped-head address and pointer construction, composition
of the full source prefix with these fragments and the loop, END-dependent
M/L/Out writeback, active-address safety and the complete flat-memory IO
contract. The port remains in progress: **11 complete, 173 pending**. No FP
atom, experimental admission or shared numerical semantics changed.

Validation: TritonBenchGInProgress builds (3461 jobs), all 29 inventory/layout/
shard tests pass (8.584 seconds), and source gates, coverage (365 reviews,
184 source pairs), proof-gap manifest and whitespace checks pass. RealInit,
RealInitSource, RealWindow and RealWindowSource pass independent comparator
replay and trust checks. Their 27 named theorem targets occur in the actual
replay inventories, and all 657 snapshot input hashes match the final sources.
Logs: `Logs/comparator-check-k8hk2_pk`.


### Complete prologue and both source computations through the loop

RealHeader and RealPointers now execute the actual program-ID coordinates,
grouped-head offsets and all Q/K/V/Out/M/L pointer construction. Query and KV
heads, independent strides and the three context bounds remain distinct. The
proofs follow the source AST's Nat address registers and real phantom pointer
tags; fp32 M/L precision belongs to the later memory operations. No floating
rule is used for integer addresses.

RealHeaderSource checks the entire leading assignment sequence in both typed
sources. RealPrologue then composes those assignments with carry initialization,
exact decimal scale, Q loading and signed-window selection. It establishes the
whole loop entry from the initial state, including Q and continued M/L/Out
values read from initial memory. The loop theorem is applied with all its entry
premises discharged, yielding independent full-context row formulas and an
unchanged-memory/register frame.

RealPrologueSource checks contiguity and order through the actual signed loop,
in addition to each fragment's correspondence. Its original_computation and
candidate_computation execute both typed source prefixes from their first
statement to loop completion, with no register-value or kernel-output premise.
They retain symbolic dimensions, all INIT/window flags, arbitrary signed window
parameters, partial checked tails and the empty-loop case. BLOCK_N is positive;
unchecking KV loads requires NKV_CTX to consist of full blocks.

Remaining obligations are the END normalization / L-store branch, final M/Out
stores, active-address safety and the full memory/IO contract. Both public real
and FP entries still need final port integration and validation. Corpus status
remains **11 complete, 173 pending**. No admitted FP atom, distribution, dtype
policy or shared numerical semantics changed.

Validation: TritonBenchGInProgress builds (3466 jobs), all 29 inventory/layout/
shard tests pass (9.441 seconds), and source gates, coverage (365 reviews across
184 source pairs), proof-gap manifest and whitespace checks pass. RealHeader,
RealPointers, RealHeaderSource, RealPrologue and RealPrologueSource pass
independent comparator replay and trust checks. All 32 named targets are in the
actual replay inventories, and all 662 snapshot input hashes match the final
sources. Logs: `Logs/comparator-check-s0y92pi2`.


### Complete END writeback and whole-source region-memory execution

RealPrologue and RealPrologueSource now also establish that the computed
M/L/Out pointer registers survive the full loop. RealStores proves successful
execution of the actual pointer and unchecked block-pointer stores, exact
readback for injective addresses and preservation of every other memory cell.

RealEpilogue executes both END branches and the final M/Out stores. END adds
log2 of the denominator to M, normalizes Out and preserves L; continuation
stores all three carry views. The division and reciprocal candidates agree
with the independent RealFormula.writeback formula under total real arithmetic,
including zero denominators and the real store's bottom-to-zero conversion.
No FP guards or new numerical assumptions enter this real proof.
RealEpilogueSource checks the entire suffix of both typed source kernels and
proves that splitting at the signed loop preserves all source statements.

RealExecution and RealExecutionSource compose these results with the actual
prefixes: both complete programs execute successfully from the initial state,
with mathematical readback for all output lanes, the required L branch behavior
and exact region-memory preservation outside written addresses. The displayed
layout conditions are injective output addresses and distinct M/L/Out regions;
row-address injectivity is derived. No entry-register, successful-execution or
kernel-output equation is a premise of the final source execution theorems.

Remaining obligations are active-address/allocation safety, flat-memory IO
transport and public Kernels/Correct/FPEquiv integration. This batch proves
region-memory execution; it does not yet complete the whole public IO contract.
Corpus status remains **11 complete, 173 pending**. No admitted FP atom,
experimental configuration or shared numerical semantics changed.

Validation: TritonBenchGInProgress builds (3471 jobs), all 29 inventory/layout/
shard tests pass (8.801 seconds), and source gates, coverage (365 reviews across
184 source pairs), proof-gap manifest and whitespace checks pass. The five new
modules and strengthened RealPrologue/RealPrologueSource pass independent
comparator replay and trust checks. All 46 named theorem targets occur in the
actual replay inventories, and all 667 snapshot input hashes match the final
sources. Logs: `Logs/comparator-check-omv2mas3`.


### Complete source access safety from strided array windows

RealAccess provides compositional trace-safety and active pointer/block-pointer
bounds lemmas. RealRegisterSafety constructs certificates from the actual
register-only AST fragments and proves them trace-safe at every state. Loads
and stores have no certificate constructor; RealMemorySafety handles their
actual address bounds, including the selected Q/K/V boundary-check branches,
INIT carry reads and all END-dependent stores.

RealSafetyPrologue follows the executed coordinate and pointer construction.
RealSafetyLoop reuses the real execution invariant at every reached signed
iteration, deriving the K/V pointer registers needed by both active-load checks.
The loop has the original step and upper bound; empty loops, checked tails and
arbitrary signed window parameters are retained. RealSafety composes the full
trace, and RealSafetySource checks both complete source bodies against that
composition before proving original_safe and candidate_safe.

RealAddressBounds derives the internal trace-bound contract from ordinary
strided array windows. Q bounds follow its selected load mode. Global K/V
windows cover each active key load; the existing source-bound theorem discharges
unchecked-block completeness from BLOCK_N positivity and the IS_EVEN_N launch
condition. M and Out windows cover all stored lanes, including padded rows.
Their store bounds also cover continuation reads. L requires bounds only when
INIT=0 or END=0. No premise assumes a resulting register, execution or output
value, and no FP atom or shared numerical semantics changed.

The remaining obligations are full flat-memory IO transport, output-layout
conditions and public Kernels/Correct/FPEquiv integration. The port remains in
progress; corpus status is still **11 complete, 173 pending**.

Validation: TritonBenchGInProgress builds (3480 jobs), all 29 inventory/layout/
shard tests pass (9.034 seconds), and source gates, coverage (365 reviews across
184 source pairs), proof-gap manifest and whitespace checks pass. All eight new
safety modules pass independent comparator replay and trust checks. Every one
of the 49 named theorem targets occurs in the actual replay inventories, and
all 675 snapshot input hashes match the final sources. Logs:
`Logs/comparator-check-k85q0pba`.


### Output layout and six-input mathematical formula

RealOutputLayout derives output address injectivity from a positive inner stride
and `BLOCK_DMODEL * innerStride ≤ rowStride`. Independent Q/K/V/Out strides,
non-unit inner strides and row padding remain supported. The output allocations
M/L/Out must be distinct; this is a per-program statement, not grid race freedom.

RealInputs defines six finite mathematical input arrays and their initial-memory
pinning contract. Only active Q rows are pinned, and old M/L/Out values are
required only for INIT=0. Every initial state supplies such views, including
in-place carry. The query, key, value, initial carry and scaled score connections
are proved without assuming any final output equation.

RealInputFormula gives a closed maximum and weighted-sum formula over the finite
global KV context and relates the executed carry and final M/Out readback to it.
The formula contains no memory reads or statement execution. It retains signed
window filtering, checked Q-tail zeros, the exact source scale decimal, resumed
carry and the distinct no-iteration branch. L's denominator is the carry value
when END=0; END≠0 preserves its memory as already proved by RealExecution.

RealFlatten checks the prologue, complete signed loop and both epilogues against
the fragment supported by the existing region-to-flat-memory bridge. Dynamic
block pointers and all stores are included. The remaining work is assembly of
the full six-input/three-output IO contract with UKernelIO and integration of the
public Kernels/Correct/FPEquiv entries. Corpus status remains **11 complete,
173 pending**; no FP admission or shared numerical semantics changed.

Validation: TritonBenchGInProgress builds (3484 jobs), all 29 inventory/layout/
shard tests pass (8.618 seconds), and source gates, coverage (365 reviews across
184 source pairs), proof-gap manifest and whitespace checks pass. The four new
modules pass independent comparator replay and trust checks. All 23 named
theorem targets occur in the actual replay inventories, and all 679 snapshot
input hashes match the final sources. Logs: `Logs/comparator-check-qj1acvso`.


### Full six-input IO and public attention_fwd_triton3 entries

RealIO gives six named Q/K/V/M/L/Out windows and a three-output contract with
in-place carry. Its core adapter has no scratch and no value-dependent windows.
RealIOBridge translates pins, active-address bounds, readback and the frame into
the existing UKernelIO.Implements.intro theorem, which supplies flat-memory
transport. No duplicate interpreter or flat-memory proof was introduced.

RealSource bundles the original source parameters and proves supported
projection, full-body correspondence, flattenability, trace safety and successful
execution for both complete programs. RealContract connects their strided IO
windows to the already verified initial input arrays, independent formula,
output-layout conditions and exact region-memory frame. Its implements theorem
proves full flat IO for both sources without a loop-register, successful-run or
final-output premise.

Kernels.lean now holds both complete typed implementations. Correct.lean exposes
the six IO windows, masks, independent maximum/weighted-sum formula, END behavior
and two real specifications. FPEquiv.lean exposes the two typed Programs, the
selected END normalization domain and full-state equivalence, with the public
printer listing only div_mul_rcp. The old altered source moved to
Proofs/LegacySource.lean; its checked source-mask counterexample remains in
Proofs/SourceReview.lean. The historical eight headlines no longer populate the
current public inventory. The three current specifications replace them.

Source/support reviews now describe the new contracts, and all 185 generated
specification sheets match the 360 public headlines. The current scope counts
are 25 full-value candidates, 294 specializations, 37 prepared-input slices,
three pre-rounding slices and one blocked summary. This inventory does not
certify the remaining benchmark directories.

The final sources build through TritonBenchGVerified (3568 jobs). All 29
inventory/layout/shard tests pass (8.740 seconds); the three relevant FP source,
branch and broadcast regressions pass (16.055 seconds). All 173 source gates,
coverage/source hashes, Python links, proof-gap and whitespace checks pass.
Final independent comparator replay and trust checks pass for all 14 public/
integration/migration and source/witness files. Every one of the 200 named
theorem targets occurs in the actual replay inventories, and all 685 snapshot
input hashes match the final sources. Logs: `Logs/comparator-check-ffxftyod`.
The case is now complete for its displayed contracts: **12 complete, 172 pending**.


## Current case: attention_kernel

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

Final validation passes: TritonBenchGVerified builds (3579 jobs), both focused
source/erasure regressions pass, all 29 inventory/layout/shard tests pass, and
all 173 source gates, coverage fingerprints, the 361-headline/186-file manifest
and whitespace checks pass. Independent comparator replay and trust checks
accept all 13 public/helper/adapter/regression files. Every one of the
204 named theorem targets appears in the actual replay inventories, and
all 696 snapshot input hashes match the final sources.
Logs: `Logs/comparator-check-hfg0l07c`.
The directory is complete for its displayed contracts: **13 complete, 171 pending**.
The next directory is attention_kernel_aligned; its explicit storage-output
score dot and unscaled bias must be preserved in its own proofs.


## Deferred case: attention_kernel_aligned

Deferred at the user's request on 2026-10-08. The host accepts bf16 while the
score dot requests out_dtype=bfloat16, which the current Triton and Lean
frontends reject. This is a compilation boundary, not an unadmitted FP atom.
The historical generic-OUT_DTYPE Lean port also fails current elaboration.
No new entries are counted as complete; experimental migration files were
removed from the source tree. The original Python remains unchanged.
Next active case: attention_llama. Progress remains 13 complete, 171 pending;
one pending directory is explicitly deferred.

## attention_llama: typed fp16 equivalence in progress

The complete fp16 source and carry-first candidate now live in
`attention_llama/Proofs/TypedSource.lean`; `Proofs/FP.lean` applies
`equiv_decompose; fp_prove` through the dynamic loop for both causal flags.
The only numerical assumption is `add_commute`, guarded on the actual fp32
reduction result and rescaled `l_prev` in each reachable iteration. Dimensions,
strides and nonnegative start_position remain parameters. The source's
block-local key mask, natural exp, two typed dots, probability cast, zero
padding and masked fp16 writeback remain explicit.

The execution check found that positional-mask load assignments bypassed
pointer dtype hints. The frontend now uses the same hint path as keyword-mask
loads and converts floating `other` to the load's element dtype. The typed
source writes the store's implicit fp16 conversion explicitly, as the existing
frontend does not infer store dtype through pointer registers. The two kernels
also declare `[Q, K, V]` / `[Out]` explicitly because the metadata scanner does
not resolve these pointer aliases at their load/store sites.

`Proofs/TypedExecution.lean` establishes successful original/candidate execution
for both causal flags with one nonempty iteration and masked query/key lanes,
together with the actual domain conditions and a consistent admitted-rule
model. These are abstract execution witnesses, not GPU or real-math results.

This does **not** complete the directory: the new typed source still needs
independent real IO, safety and frame proofs, and the int8-to-float8e5 payload
branch still needs representation semantics. The historical real theorems are
kept separate. Completion remains **13 complete, 171 pending**, including the
user-deferred attention_kernel_aligned case.

Validation for this FP checkpoint: both `TritonBenchGVerified` (3,579 jobs)
and `TritonBenchSpecExamples` (3,567 jobs) built successfully, as did the typed
fp16 execution witness. Nine focused Lean test files passed, including the
positional-mask dtype, explicit fp32 commutation, missing/wrong-site guard,
fp64 rejection and source-precision regressions. The 29 layout/coverage/shard
unit tests, all 173 source gates, coverage review and whitespace checks passed.
Fresh independent comparator + trust replay accepted all nine changed/new Lean
files: 33 explicitly named theorem targets were located in the kernel inventory
(398 targets including generated/anonymous declarations), and all 702 snapshot
input hashes matched the current tree. Logs: `Logs/comparator-check-5b79cs7g`.
This validates the stated FP checkpoint, not the pending whole-directory work.

## attention_llama: typed real source, output and frame

The typed fp16 original and carry-first candidate now connect to the real
execution proof through `Proofs/RealSource.lean`. The mathematical twin in
`Proofs/RealKernel.lean` retains explicit K/V zero padding. Constant causal
branches are inlined through complete-state execution congruence, and the
candidate's sum/carry exchange is established by real arithmetic. The new
`ProgramRewrite` lemmas preserve failure as well as successful execution;
none of these real proofs uses an admitted FP relation.

`Proofs/RealOutput.lean` proves both typed sources succeed and compute the
independent natural-exp attention formula, or the source's shifted causal
formula. The same execution preserves every complete memory cell outside the
active output window. Shapes and all strides remain parameters. Non-causal
results require whole key blocks and at least one block; causal results require
a whole-block streamed span inside N_CTX and start_position no larger than the
first query row. These remain explicit hypotheses. Non-divisible key tails and
empty visible rows are not silently classified as standard attention.

The real regression uses different V values: non-causal output is 9, causal
output is 7, the masked output retains 99 and the input memory is preserved.
It checks both source variants. Structural regressions check selected/unselected
failing branches, multiple range iterations, empty/zero-stride loops and a
missing dynamic bound. Both new test files pass. The affected benchmark and
example libraries build together (3,743 jobs); all 29 inventory/layout/shard
tests, 173 source gates, coverage review and proof-gap checks pass.

This is still a regional-memory checkpoint. Active-address safety, flat-memory
transport/public IO and the FP8 payload path remain pending. The new exact
three-input IO adapter prepares that transport; it does not itself discharge
this kernel's bounds. Progress remains **13 complete, 171 pending**, including
the user-deferred attention_kernel_aligned case.

Independent comparator + trust replay accepted all 7 changed/new Lean files
in this real checkpoint. All 91 explicitly named theorems occur in the actual
replay inventories (431 kernel targets including generated declarations), and
all 708 snapshot input hashes match the final sources. Logs:
`Logs/comparator-check-i11k792n`. This validates the regional-memory result above;
it does not discharge flat-allocation safety or the FP8 branch.

## attention_llama: active-address safety and flat execution

`Proofs/RealSafety.lean` now proves the Q/K/V loads and masked output store stay
inside their allocated windows for both causal flags. The proof constructs the
streaming invariant from execution; it does not ask the caller to supply a
numeric output equation or a whole-program safety assumption. Shapes and all
strides remain symbolic, with the same whole-block and causal-position
conditions as the mathematical result.

`Proofs/RealSource.lean` connects both complete typed sources to those safety
proofs. The generic `TraceRewrite` lemmas require preservation of both execution
and access safety when lifting a rewrite through branches and dynamic ranges.
Equal final states or equal failures alone cannot establish safety.

`Proofs/RealMemory.lean` consequently proves successful execution of each actual
source erasure inside one disjoint flat allocation, its independent attention
formula, and preservation of every complete cell outside the active output
window. The causal span premises concern the current program's IDs; no
impossible premise over every program ID is introduced.

The flat regression distinguishes non-causal output 9 from causal output 7,
preserves a masked output cell containing 99 and an input cell containing 7,
and allocates Q space for only the active row. Both source variants pass. The
new safety regressions distinguish selected/unselected unsafe branches,
empty/zero-stride/nonempty ranges, and an unreachable tail after failure.
Existing memory-safety, memory-frame and program-rewrite tests also pass.

The affected benchmark and example libraries build together (3,749 jobs).
All 29 inventory/layout/shard tests, source gates over 173 ports, coverage
review and proof-gap manifest checks pass. Public independent IO contracts and
the FP8 payload-reinterpretation path remain pending; this directory is not
counted complete. Progress remains **13 complete, 171 pending**, including the
user-deferred attention_kernel_aligned case.

Independent comparator + trust replay accepted all 10 changed/new Lean files
in this safety checkpoint. All 69 explicitly named theorems occur in the
actual inventories (404 kernel targets including generated declarations), and
all 715 snapshot input hashes match the final sources. Logs:
`Logs/comparator-check-6hlcrm5b`. This validates the flat-memory result above;
it does not discharge the public IO migration or FP8 branch.

## attention_llama: independent input-array IO contract

`Proofs/RealIO.lean` connects both complete fp16 sources, for both causal flags,
to an independent formula over pinned input arrays. Q comes from the first
static stream slice with the source row mask; K and V are reassembled from
streamed [BLOCK_N, BLOCK_DMODEL] tiles. The specification uses natural-exp
attention and the source's shifted causal visibility, without referring to a
kernel execution or its recurrence. Successful flat execution and complete-cell
preservation come from the checked RealMemory result.

The three-input IO interface now has the same default-True `pre` convention as
its related interfaces. Its rounded and exact contracts thread this launch
condition separately from the masks. attention_llama states its whole-block and
causal-span conditions there, locally to each program ID. Output-stride
injectivity is independent of program ID, and all source strides remain
parameters. The existing attention_score caller keeps the default True
condition; its only proof changes consume that trivial premise in two helpers.
Its scope and numerical model are unchanged, and its reviewed source fingerprint
has been refreshed after inspecting those changes.

`bench/tests/RealAttentionLlamaIO.lean` checks an actual nonempty execution at
program ID 1 with start_position=2, three key blocks, a masked query lane and
distinct V values. Both source variants output 7 for query row 2; the masked
output retains 99 and V[0] retains 7. The same fixed span correctly excludes
program ID 0 from this contract. The previous regional/flat output and safety
rewrite regressions also pass, as does direct compilation of attention_score.

The affected benchmark and example libraries build together (3,750 jobs).
All 29 inventory/layout/shard tests, source gates over 173 ports, coverage
review and proof-gap checks pass. The independent fp16 IO result is ready for
the public entry migration. FP8 byte reinterpretation and the final three-entry
migration remain pending; progress stays **13 complete, 171 pending**, including
the user-deferred attention_kernel_aligned case.

Independent comparator + trust replay accepted all five changed/new Lean files
in this IO checkpoint, including the shared interfaces and existing caller.
All 107 explicitly named theorems occur in the actual inventories (338 kernel
targets including generated declarations), and all 717 snapshot input hashes
match the final sources. Logs: `Logs/comparator-check-calpgaqo`. This validates
the stated fp16 IO result and compatibility; the FP8 path remains outstanding.

## attention_llama: E5M2 byte semantics

The shared AST now represents int8 payload reinterpretation as a separate
`decodeFloat8E5` node. The E5M2 format retains all 256 byte patterns, including
the sign of zero and NaN payloads. FP execution passes these bits to an opaque
backend and leaves the subsequent fp16 cast explicit. Real execution projects
only finite encodings, including subnormals; invalid signed-byte values and
nonfinite lanes cause projection failure. The success and rejection conditions
are proved independently of any attention formula or admitted FP rule.

Precision erasure keeps this decoding operation. Memory typing, access safety,
flattening for both evaluators and locality now recurse through its byte
operand. Tests exhaust the five encoding classes, check minimum/maximum finite
values and negative encodings, and execute an actual int8 load followed by
reinterpretation and fp16 conversion. In particular, byte 60 projects to real 1;
FP execution preserves negative-zero and NaN payloads all the way to the cast.
Out-of-range integer 128 cannot masquerade as a signed byte.

The affected libraries and attention_llama real IO build together (3,752 jobs).
The new byte and execution tests, existing program/safety rewrite and integer
execution tests, and both regional/flat and IO attention fixtures pass. The
29 inventory/layout/shard tests, source gates over 173 ports, coverage reviews
and proof-gap manifest also pass.

This is shared semantic support, not completion of the FP8 attention branch.
Frontend int8-width tracking, the actual USE_FP8=True source and its two proof
layers, and the final public-entry migration remain outstanding. No new
numerical rule or GPU experiment was introduced. Progress remains
**13 complete, 171 pending**, including the user-deferred
attention_kernel_aligned case.

Independent comparator + trust replay accepted all 17 selected files: the 15
changed/new foundation and test files plus attention_llama's existing FP and
real IO proofs. All 332 explicitly named theorems occur in the actual target
inventories (4,998 kernel targets including generated declarations). All 721
snapshot input hashes match the final sources. Logs:
`Logs/comparator-check-tlzqfjuv`. The separate FP execution-witness build also
passes (3,378 jobs), and its source regression retains only `add_commute`.

## attention_llama: full FP8 source and conditional equivalence

`Proofs/TypedSource.lean` now also contains the complete USE_FP8=True original
and carry-first programs. K/V load signed int8 payloads, reinterpret them as
E5M2 and then convert to fp16. Both causal conditionals, all symbolic dimensions
and strides, integer zero padding, source masks, natural exp, dot operations,
probability conversion and masked fp16 output remain present. Source comparison
confirms that each candidate changes only the l_curr addition order.

`Proofs/FP8.lean` proves full-program conditional equivalence using only
`add_commute`, with finiteness checked on the actual operands at every reached
addition site. `Proofs/FP8Model.lean` and `Proofs/FP8TypedExecution.lean` prove
that the admitted rules and these domains admit successful nonempty executions
for both variants and causal flags. The witness uses an abstract backend, not
a GPU emulator. No new numerical atom or experiment was introduced.

The DSL tracks int8 storage width separately from its mathematical Int carrier.
Only a known byte source permits the E5M2 bitcast. Numerical casts, arithmetic,
dot products and RMW return values drop storage provenance; ordinary integer
stores retain their prior behavior. A dotted floating cast no longer pins its
operand to Nat. Regression tests cover the distinct conversion operations,
pointer aliases, masks and integer zero padding, reject unknown widths, and
execute a masked load through bitcast, fp16 conversion and store. A masked NaN
byte does not replace the specified zero padding.

`Proofs/FP8RealLoads.lean` proves complete-state and failure-preserving fusion
of the erased conversion fragment, plus its exact finite E5M2 output. It has no
FP admission dependency. The complete FP8 real attention formula, raw-byte input
IO and memory frame remain pending, followed by the three public entry files.
The fp16 real result cannot be substituted for these obligations. Progress
remains **13 complete, 171 pending**; attention_kernel_aligned remains deferred.

The final integrated build passes (3,490 jobs), including the fp16 real IO proof
and both storage modes' FP execution witnesses. FP8 frontend, source fidelity,
Triton smoke, integer execution and regional/flat/IO attention regressions pass.
The 29 inventory/layout/shard tests, source gates, coverage review and proof-gap
manifest pass. The additional audit-gate regressions pass after making the
installed comparator and its required local systemd sandbox available.
The separate legacy `int8_matmul_kernel` port still fails its source/body `rfl`
at line 616; the same failure was reproduced in the previous verified snapshot.
That pending case was not modified or counted as a passing regression.

Final independent comparator + trust replay accepts all 17 selected files,
covering the changed frontend, source and regression modules, both FP proofs
and execution witnesses, the conversion-fragment real proof and existing real
IO proof. All 44 explicitly named theorems occur in the actual inventories
(511 kernel targets including generated declarations). All 726 snapshot input
hashes match the final sources. Logs: `Logs/comparator-check-7qtc6fnr`.


## attention_llama: real FP8 source and input-view bridges

`Proofs/FP8RealSource.lean` connects both complete FP8 sources after real erasure
to their fused load/decode bodies. Both causal flags, all dimensions and strides,
source masks, output stores and execution failures are retained. Fusion also
preserves intermediate access safety, and the full erased sources are checked
for flat-memory transport support. This uses no admitted floating relation.

`Proofs/FP8RealMemory.lean` defines the decoded K/V input view and proves its
masked-load correspondence. Valid finite bytes are required only on active
lanes. Pure register fragments and separated masked output writes commute with
the view. `Proofs/FP8RealSimulation.lean` applies this to the complete eighteen-
statement prologue with symbolic shapes and strides; its Q bound is derived
from the actual address expression rather than assumed as a trace property.

`bench/tests/RealAttentionLlamaFP8.lean` checks that an active byte 60 decodes to
one while inactive NaN byte 127 and invalid signed integer 128 are skipped and
padded with zero. The combined bridge build passes (3,450 jobs), the regression
passes, and all new modules include axiom-footprint checks.

The complete FP8 loop induction, independent attention formula, raw-byte IO and
full memory frame remain pending, followed by the three public entry files.
No fixed-shape fixture replaces that obligation, and no new GPU atom is needed.
Progress remains **13 complete, 171 pending**, with attention_kernel_aligned
deferred at the user's request.

Independent comparator + trust replay accepts all four new bridge/regression
files. All 31 explicitly named theorems appear in the actual target inventories
(340 kernel targets including generated declarations), and all 730 snapshot
input hashes match the current sources. Logs: `Logs/comparator-check-xoefzzc2`.


## attention_llama: complete real FP8 range correspondence

`Proofs/FP8RealIteration.lean` proves that one full fused FP8 source iteration
has the same real execution on the decoded input view. Both candidates and
causal flags remain supported with symbolic shapes and strides; the statement
preserves failure as well as success. The arithmetic registers are unrestricted.
Only actual active K/V load lanes require finite signed bytes.

`Proofs/FP8RealLoopFrame.lean` checks the source's final two pointer updates,
retains offs_n and advances both K/V tiles by BLOCK_N using their own strides.
`Proofs/FP8RealLoop.lean` uses those facts in an input-address invariant to lift
the relation through the complete dynamic range. Its BytesFrom condition follows
the scheduled block starts and source masks. It also proves exact preservation
of every raw memory cell; this is independent of decoded-memory equality.
Empty ranges, zero strides and intermediate failures are included.

The range build passes (3,450 jobs). The extended real FP8 regressions pass:
masked NaN/invalid-int8 lanes, a masked partial-tail NaN, NaNs in unvisited
blocks, and rejection of an active NaN. No extra GPU atom or FP axiom is used.
The source's partial-tail behavior is preserved by this correspondence; it is
not thereby proved to equal conventional attention on non-divisible key tails.

The full-kernel composition with the prologue and output store, transfer to the
independent mathematical formula, raw-byte IO and final output frame remain
pending, followed by the three public entry files. Progress remains
**13 complete, 171 pending**; attention_kernel_aligned remains user-deferred.

Independent comparator + trust replay accepts the three loop modules and the
updated regression file. All 21 explicitly named theorems occur in the actual
inventories (55 kernel targets including generated declarations); all 733
snapshot input hashes match the final sources. Logs:
`Logs/comparator-check-ummumlgk`.


## attention_llama: whole-source real FP8 attention and raw frame

`Proofs/FP8RealSetup.lean` establishes the exact initial byte-loop cursor and
causal bound from the actual prologue. `FP8RealPostamble.lean` relates the output
store and preserves original cells outside Out without assuming decoding is
injective. `FP8RealReference.lean` identifies the real reference with the typed
source already proved against the independent attention formula.
`FP8RealExecution.lean` composes setup, the full dynamic range and postamble for
both source variants and causal flags, retaining success and failure.

`FP8RealOutput.lean` proves successful execution of the actual byte source,
noncausal/causal attention on the exact decoded E5M2 input values, and the full
raw-memory frame, including masked output lanes. Shape, strides and source
flags remain symbolic. The existing whole-block and causal-position conditions
are retained. Finite-byte conditions concern reached, unmasked K/V loads only.
No FP admission axiom is used by these real results.

The shared pure-assignment correspondence helpers moved from FP8RealIteration
to FP8RealMemory for reuse by setup and postamble; their statements and proofs
are unchanged. The integrated output and existing real IO build passes (3,465
jobs). RealAttentionLlamaFP8, FPAttentionLlamaSource and RealAttentionLlamaIO
regressions pass. The extended FP8 source regression executes two noncausal
iterations (output 3/2) and one causal iteration (output 1), for either candidate;
it checks the masked output and exact preservation of a negative-zero K byte
and the distinct V byte. Existing inactive/invalid-byte tests still pass.

Remaining work: derive FP8 active-address safety and flat-memory transport,
connect independent raw-byte input arrays (pinning actual integer memory cells),
and expose the three public entry files. Regional execution does not itself
prove flat-address safety. Progress remains **13 complete, 171 pending**;
attention_kernel_aligned remains user-deferred.

Independent comparator + trust replay accepts all eight changed/new proof and
regression files. All 55 explicitly named targets occur in the actual theorem
inventories (107 kernel theorem targets including generated declarations).
All 738 snapshot input hashes match the final sources. Logs:
`Logs/comparator-check-724363_g`.


## attention_llama: FP8 address safety, flat execution and raw-byte IO

The loop now preserves its program IDs and batch/head registers as well as
its byte-pointer cursor. `FP8RealSetup.runs_with_launch` establishes those facts
from the actual prologue. `FP8RealSafetyLoop.lean` derives active K/V address
bounds at every reached iteration, while `FP8RealSafety.lean` composes the full
source's initialization, range and masked output store. Numerical accumulators
are unrestricted: no output formula or finite-byte premise is used for safety.
Partial key tails, zero strides and decode failures are included.

`FP8RealFlat.lean` combines source safety, supported flattening and the regional
real formula/frame to execute each byte source inside a disjoint allocation.
The existing generic real-output transport lemma is shared with the fp16 path;
its input-cell representations are unrestricted. Both independent attention
formulas and the original raw-cell frame survive this transport.

`FP8RealIO.lean` connects independent Q real arrays and K/V finite signed-byte
arrays. Its contract pins actual real/integer memory cells. Only the expected
mathematical formula decodes the input bytes; no decoded replacement buffer is
assumed. The shared `StreamCellsReal.lean` relation keeps the established stream
addresses, masks, launch conditions, flat execution and memory frame, while
parameterizing the encodings of the three input cell types.

The integrated build passes (3,470 jobs). Updated FP8 regressions verify flat
branch outputs 3/2 and 1, a masked output cell, raw negative-zero/other bytes,
and partial-tail source safety with no allocation for the masked fourth key.
A separate IO regression runs at program ID 1 with start_position 2, proves
nonempty shifted-causal execution and preserves the original V byte. Existing
fp16 regional/flat and IO regressions also pass. No new GPU atom or FP law is
introduced; the FP equivalence assumptions remain add_commute only.

Remaining work is public Kernels/Correct/FPEquiv migration, metadata updates
and final directory acceptance. Non-divisible key tails and empty visible
causal rows remain outside the independent conventional-attention formulas,
even though the new address-safety result includes such partial tails.
Progress remains **13 complete, 171 pending**; attention_kernel_aligned stays
user-deferred.

Independent comparator + trust replay accepts all fourteen selected files,
including the new encoded-cell contract, safety/flat/IO proofs, affected
execution/output proofs and both FP8 regression files. All 67 explicitly named
targets occur in the actual inventories (182 kernel theorem targets including
generated declarations). All 744 snapshot input hashes match the final sources.
Logs: `Logs/comparator-check-fzseqw16`.


## attention_llama: public entries and complete case acceptance

The directory now has exactly `Kernels.lean`, `Correct.lean` and `FPEquiv.lean`
at its top level. Kernels contains both complete fp16 and int8/E5M2 source
pairs; it replaces the development `Proofs/TypedSource.lean` referenced in
older checkpoints. The old single-file port is superseded by the connected
mathematical derivations in `Proofs/RealKernel.lean`.

Correct exposes the full input/output addresses, masks and local launch
precondition, an independent natural-exp weighted-sum formula, and two public
specifications. Both quantify over original/carry-first and causal/noncausal.
The FP8 contract pins exact integer K/V cells and decodes their finite E5M2
values without replacing execution memory. Successful flat execution, every
active output and all other cells are covered. The formula requires positive
BM/BN/T, whole streamed spans, nonempty causal visibility and injective output
offsets; FP8 also separates Q/Out from K/V. Partial-tail conventional-attention
and empty-visible-row formulas remain outside these displayed contracts.

FPEquiv exposes both complete Programs and each reached addition's domain.
Both public specifications print only add_commute. The domain checks finite
actual sum(p) and rescaled l_prev after the score/exp prefix. Dtypes, byte
reinterpretations, casts, tl.exp, dots, masks, causal branches and store stay
unchanged. The real and FP import stacks are independent. No new atomic
experiment is needed.

Validation: TritonBenchGVerified builds successfully (3,646 jobs); all five
focused Lean fixtures pass, including the two IO fixtures now invoking the
public specifications. Forty-one distinct source/inventory/layout/shard tests
pass across the two test runs. The full source-only scan passes for all 173
imported ports. Two scan false positives were corrected and covered by tests:
a proved P=True no longer matches the bare-True placeholder check, and fp32
casts are checked in the explicitly named JIT rather than unrelated helpers.
No Lean statement or proof was weakened to satisfy those lexical scans.

Independent comparator replay and trust checks accept 39 files, 255 named theorem targets and 1394 kernel theorem targets.
All 745 snapshot input hashes match the final workspace sources. The two real
and two FP public specifications are explicitly present in the elaborated
target inventories. Logs: `Logs/comparator-check-pnw4wkeh`. This is a fresh replay of every Lean
file in this directory, the five fixtures and the mixed-cell IO contract.

Coverage reviews, source/support fingerprints, Python links, proof-gap manifest
and generated sheets agree: 363 public declarations across 187 specification
files (28 full-value candidates, 294 specializations, 37 prepared-input slices,
3 pre-rounding slices and 1 blocked summary). These labels preserve each
statement's actual scope; they do not certify hardware execution.

Current progress: **14 complete, 170 pending**. The user-deferred
`attention_kernel_aligned` remains pending. Next active case: `attention_score`.
No commit or push was made.

## attention_score: typed source and arithmetic foundation

This and the following region-memory checkpoint record earlier snapshots.
The final signed-source contract section below records current acceptance.

The next active case remains in progress. Its historical case-1 port used
natural-subtraction window distances, which turn query 0/key 1 into distance
0 and wrongly include that future key in a regular positive window. The
coverage review and manifest now explicitly limit those old headlines to
the historical Lean program. They are not accepted as proofs of the signed
Python source. This correction does not change the fourteen-completed count.

The new Proofs/TypedSource.lean keeps both window flags, both even-size flags,
grouped heads, independent source strides, checked/unchecked Q/K loads,
fp16 Q/K/Out and fp32 M/accumulation, signed distances/offsets/sizes, both casts
and the final masked store. BLOCK_M=BLOCK_N is the source wrapper's typing
invariant, separate from experimental admission shape. The unused stride_on
and the IS_EVEN_N-controlled query-tail mask are retained deliberately.

The complete scalar-left candidate changes only qk*qk_scale to qk_scale*qk.
Proofs/FP.lean uses equiv_decompose/fp_prove and prints only mul_commute.
A new Site.mulScalarFP32 bridge checks both operands in the actual explicit
fp32 computation context, including arbitrary computed scalar expressions.
Tests preserve the exact output precision, mask and signed arithmetic, and
reject missing guards and use of an fp64 multiplication with fp32 guards.
No numerical admission or GPU experiment was added.

The independent RealFormula sums base-2 weights over query blocks from Q/K/M
input functions. M is supplied by the caller, not assumed to be the output of
an omitted normalization producer. RealExpressions connects the signed
source distance to global indices. RealTiles relates dot, normalization,
window/tail selection and column reduction to the independent weight formula.
RealArithmetic identifies the nine actual arithmetic statements in both
real-erased loops, proves they execute from loaded register values, preserves
memory/program IDs/undefined values, and proves real candidate commutation
without FP admission. RealArithmeticOutput then connects that actual execution
to the independent column formula for both variants, including successful
execution and the arithmetic memory/program-ID/undefined-value frame. These
intermediate loaded-register premises are not a public whole-kernel input contract.

Still required: connect actual memory loads to those tiles, prove the complete range invariant,
address safety, final store/frame and flat IO for both sources. The public
Kernels/Correct/FPEquiv migration waits for these obligations. The experimental
model witnesses concern satisfiable FP execution/domains, not the independent
real IO proof or concrete GPU behavior.

Validation: the benchmark/example libraries and the initial seven modules
built successfully (3,783 jobs); the final real-commutation helper and the
additional arithmetic-output module were rebuilt afterward. Five focused Lean
regressions and 41 source/inventory/layout/shard tests pass. All 173 source
gates, coverage records, manifest and generated sheets agree.

The first independent comparator/trust batch accepts all seven initial proof/
source modules, shared ProfiledRewrite, two changed regressions, the range
rewrite fixture and the existing linear-attention FP entry: 12 files, 61 named
targets and 878 kernel theorem targets. All 753 snapshot input hashes match.
Logs: `Logs/comparator-check-nylev19c`. The arithmetic-output addition also
passes its separate fresh snapshot: both named theorems and all seven kernel
targets, with all 754 snapshot inputs matching. Its logs are
`Logs/comparator-check-hxvga6d1`. Combined coverage is 13 files, 63 named targets
and 885 kernel targets; no accepted input from the first batch changed.
This is a foundation checkpoint, not whole-case acceptance.

## attention_score: complete real region-memory execution

The complete original and scalar-left sources now execute to the independent
initial-Q/K/M column formula in `Proofs/RealExecution.source_correct`. The
connection includes grouped-head initialization, actual checked/unchecked
loads, every query iteration, pointer advancement and the final masked store.
All four flags and signed window parameters remain unrestricted. The theorem
also preserves every memory cell outside the active output window.

`RealLoop` proves the trip count `ceil(ROUND_CTX / B)` and the partial-column
invariant for arbitrary stops, including zero and a partial final block.
`RealLoads`, `RealBody`, `RealFrame` and `RealIteration` connect the source
fragments to the carried state. `RealProgram` checks the entire source
decomposition, with `RealPrelude` and `RealOutput` proving its endpoints.
These intermediate register invariants are proved from the actual input
memory; they are not user-supplied output assumptions.

The end-to-end regression uses N=1, ROUND_CTX=3 and B=2. With zero inputs,
IS_EVEN_N=true retains all four visited query lanes and writes 4; when false,
the source query mask retains only row 0 and writes 1. The latter case also
checks that the inactive output lane and the next cell are unchanged. Both
complete source variants are checked. This deliberately retains the source's
key-even-controlled query mask instead of silently repairing its condition.

The new proof modules build successfully (3,339 jobs), and the end-to-end Lean
regression passes. The 29 port-layout, coverage-review and audit-shard tests
pass. Coverage/manifest validation still reports 363 declarations across 187
public specification files; generated sheets and whitespace checks agree.

Independent comparator/trust replay accepts all nine new proof modules and
the end-to-end regression: **10 files, 42 named targets and 128 kernel theorem
targets**, with all **764 snapshot input hashes** matching. Logs:
`Logs/comparator-check-61w7qv0g`. Both prior foundation snapshots still match
their accepted input hashes; this phase did not change their proof sources.

Physical allocation safety and the flat-memory IO bridge remain. In particular,
M is loaded without a mask, so all cells in the padded last query block must
be allocated even when query rows are masked. Public Kernels/Correct/FPEquiv
migration and whole-case acceptance wait for these obligations. The total
remains **14 complete / 170 pending**, with attention_kernel_aligned deferred
by the user. No numerical admission, GPU experiment, commit or push was added.

## attention_score: complete signed-source real and FP contracts

The public Kernels/Correct/FPEquiv layout now replaces the old natural-window
headlines. Both complete sources implement the independent finite Q/K/M column
formula. The public real IO displays every stride, load mask and output mask;
source initialization, loop invariants, trace safety and flat-memory simulation
are proved internally. There is no precomputed score/output or loaded-register
premise. The positive common query/key block size matches the source wrapper.
All four flags, signed offset/size, empty loops and partial query blocks remain
symbolic. Unmasked M padding and K loads before an empty loop are explicit.

The scalar-left candidate uses only admitted mul_commute, with finite actual
fp32 qk and scale operands at each reached site. FP execution witnesses cover
all flag combinations in a consistent abstract model. The independent real
proof does not assume that M is produced correctly by an omitted kernel.
The historical Nat-subtraction model is retained only in
Proofs/LegacyNaturalWindow.lean, with its source mismatch explicitly stated.

The fp16/fp32 region annotations, explicit input/output lists and resolved
Out.type.element_ty cast are registered as a reviewed transcription exception.
The source regressions check dot/load/store precision, signed distances and
the IS_EVEN_N-controlled query-tail selection. These checks do not certify
Python alias resolution or concrete GPU arithmetic.

The final benchmark library builds successfully (3,677 jobs). All three
source/execution/IO regressions and 29 layout/coverage/shard unit tests pass.
Source gates over 173 ports, all 364 scope reviews across 188 public files,
manifest and generated sheets agree. The manifest has 31 full-value candidates,
292 specializations, 37 prepared-input slices, 3 pre-rounding slices and one
blocked summary. These scope labels remain separate from proof acceptance.

Independent comparator + trust replay accepts all **33 files** in the final
case/regression batch. All **198 named proofs** appear in the inventories
(1100 kernel theorem targets including generated declarations), and all
777 snapshot input hashes match the final sources. Evidence:
`Logs/comparator-check-3r3pe16j`.

Current progress: **15 complete, 169 pending**, including user-deferred
attention_kernel_aligned. The next case is attn_fwd_causal: its historical
probability-zeroing transcription differs from the finite score penalty in
Python and must be corrected before its old proof can be assessed for reuse.
No commit or push was made.


## attn_fwd_causal: verified source and FP foundation

The replacement typed source retains both helper invocations and both defined
outer stages. STAGE=3 visits prefix then diagonal; STAGE=2 visits the diagonal
twice. Each helper has local pointer copies. The diagonal uses an additive
finite -1e6 score penalty and never zeros probabilities. fp16 Q/K/V/Out,
fp32 scales/accumulators, both dot output types, casts and load/store masks
remain explicit in both complete source variants.

The full conditional FP proof uses only div_mul_rcp at the final actual row
normalization, with finite accumulators and finite/nonzero denominators.
One consistent abstract backend satisfies the admitted rules, executes both
versions and inhabits the actual guards for both stages, with a nonempty
iteration in each helper. This is not a GPU experiment or real IO theorem.

The independent real formula sums base-2 weights over all visited lanes,
retaining repeated visits and padded zero loads. The actual diagonal mask and
score expressions have checked real interpretations. A two-key mathematical
counterexample distinguishes finite penalty from exact masking: the first
row is 2^(-1000000)/(1+2^(-1000000)) > 0. This refutes the formula-level exact
masking claim; it does not yet refute a full source IO specification.

The historical single-loop module is explicitly scoped to its altered model.
Its former implicit widening before a real-output dot is now explicit to
preserve that old AST under the current DSL. Existing proofs and their two
coverage rows do not certify the Python source. The historical module is
included in this independent replay batch.

Validation: TritonBenchGInProgress builds (3,450 jobs); the final typed-source
regression and 29 layout/coverage/shard tests pass. Comparator/trust replay
accepts all 9 files, including every one of the
168 named proof targets and all
632 kernel theorem targets; all
785 snapshot source inputs match. Evidence:
`Logs/comparator-check-iq44vh9u`. The reviewed inventory remains 364 declarations across 188 public
files, with unchanged coverage counts.

Still pending: general execution through both real-erased helpers,
active-address safety, finite input-memory pinning, output/frame and flat IO
for both complete versions. Public migration waits for those proofs and their
validation. Overall progress remains **15 complete, 169 pending**;
attention_kernel_aligned remains deferred at the user's request.

## attn_fwd_causal: verified real helper ranges

Both erased source helper bodies now have complete iteration and range
execution proofs. The proof chain covers actual K/V and scale memory loads,
finite score penalties, center/probability computation, online carry updates
and all three local pointer advances. It preserves symbolic dimensions,
independent source strides, partial tails, an unaligned range origin and empty
ranges. Masked reads use the explicit zero-undef contract. Active-address
safety is not part of this intermediate result.

The loop-entry contract pins query/scale registers to the original input
memory, row/key indices, the carried center/denominator/numerator and current
helper pointers. Register-only execution preserves the original memory.
`RealBody` checks both helper fragments against both complete typed sources;
`RealHelperSource` checks the first constant branch and the second diagonal
body. `RealLoop.loop_executes` executes either actual helper range from this
entry contract. The prologue must still establish the entry contract.

`RealLoopFormula.after_row` identifies each final carry with the independent
visited input pairs. `both_normalized` proves that composing the two range
formulas yields `RealFormula.expected` for both legal stages, retaining the
duplicate diagonal visit in stage 2 and weights from finite-penalty/padded
lanes. This mathematical composition does not yet execute the inter-helper
pointer reset or final store.

Validation: `lake build TritonBenchGInProgress` passes (3,469 jobs), as do 29
coverage/layout/shard tests. Coverage/manifest checks retain 364 declarations
across 188 public files. Comparator/trust replay accepts all 10 new real
modules, all 47 named proofs and 282 kernel theorem targets; all 795 snapshot
input hashes match. Evidence: `Logs/comparator-check-xkrqsb68`. This batch adds
no FP assumptions; the existing FP proof still uses only `div_mul_rcp`.

Remaining: source prologue and input-cell pinning, first-helper setup, the
pointer reset between helpers, final normalization/store, active-address
bounds, output/frame and flat IO for both versions. The finite-penalty
refutation remains formula-level until the full source IO bridge exists.
Public migration and case completion wait for these obligations. Progress
remains **15 complete, 169 pending**, with attention_kernel_aligned deferred.

## attn_fwd_causal: complete real source execution

`RealSource.body` checks every statement of both complete erased sources:
coordinate/stride setup, pointer initialization, query/scale loads, initial
carry, first helper setup/range, second helper pointer reset/range, final
normalization and masked output store. `RealExecution.executes` composes these
steps from initial program IDs and actual input memory, with the explicit
zero-undef convention. No intermediate carry, loaded-value register or
successful-execution premise is left for the caller to assume.

For both legal stages and both source variants, each active output equals
`RealFormula.expected` when active output addresses are injective. Inactive
lanes may alias; they do not create extra injectivity requirements. Every
memory cell outside the active output writes is preserved. Symbolic shapes,
independent source strides, empty prefix ranges, partial tails, unaligned
diagonal origins and repeated stage-2 visits remain supported.

`RealInputPins` supplies the connection to caller-provided input values.
Q/K/V pins apply only at active lanes of actual visits; query scale and each
visited key scale remain unmasked reads. `expected_congr` proves that changes
to unused inputs cannot change the formula. `pins_from_memory` shows that
every initial memory supplies a matching input context. `RealAccessSafety`
uses the same finite visited windows for its local load/store bounds; it does
not yet compose them into whole-program trace safety.

`RealCounterexample.source_output` executes both complete erased sources on
the two-key Q/K=0, scales=1, V=[0,1] fixture and obtains the strictly positive
first output `2^(-1000000)/(1+2^(-1000000))`. The declaration-level refutation
uses `disprove` to reject the corresponding zero-output exact-causal-mask
claim. This establishes a real region-memory source counterexample, without
claiming hardware execution or rounded FP inequivalence.

Remaining: compose trace safety for the complete source, then transport the
input pins, formula readback and memory frame to flat-memory IO for both
versions. The public three-file migration and completed-case count wait for
those obligations and their validation. Progress stays **15 complete, 169
pending**; attention_kernel_aligned remains user-deferred.

Validation: `lake build TritonBenchGInProgress` passes (3,506 jobs), all 29
metadata/layout/shard tests pass, and coverage/manifest checks retain 364
declarations across 188 public files. Independent comparator/trust replay
accepts all 16 new modules, all 53 named proofs and all 238 kernel theorem
targets; all 811 snapshot input hashes match. Evidence:
`Logs/comparator-check-g18nmvw5`. The FP proof still uses only `div_mul_rcp`.

## attn_fwd_causal: complete real/FP source verification

The case now has the three public entries Kernels, Correct and FPEquiv.
Kernels preserves both complete helper calls and their independent pointer
copies; STAGE=3 visits prefix then diagonal, STAGE=2 visits diagonal twice.
All source-unused strides remain parameters. The finite -1e6 score penalty,
fp16 Q/K/V/Out, fp32 scales/accumulators, dots and casts remain explicit.

Correct displays exact source addresses and the independent normalized
base-2 weighted sum over actual visits, including repeated visits and padded
K/V loads. Both versions successfully execute in disjoint flat allocations,
satisfy the formula at active outputs and preserve every other cell. The
contract assumes only legal stage, positive key block, bounded actual reads/
writes, actual-read input pins, zero masked undef and distinct active output
addresses. Loop registers and output equations are proved, not preconditions.
The unaligned-tail regression retains the unmasked key-scale read for a fully
masked K/V tile; an allocation missing that scale fails the bound predicate.

FPEquiv exposes the actual post-loop fp32 division operands and proves
complete typed-source equivalence using only div_mul_rcp. Finite accumulators
and finite/nonzero row denominators are checked for every evaluated lane.
The consistent-model execution/domain witness covers both legal stages.

RealFlatCounterexample executes both complete erased sources in a concrete
disjoint flat allocation with Q/K=0, unit scales and V=[0,1]. The first output
is positive, refuting exact causal masking in real semantics. This does not
claim a valid physical GPU tile or an fp16-underflow counterexample. The old
single full-context exact-mask model is preserved under Proofs/LegacyExactMask;
its two historical public coverage rows have been replaced by three current
source specifications. Host launches and fixed-width overflow remain outside
the per-program scope. The transcription exception describes helper inlining,
explicit dtypes and generalized 128/96 head constants, not altered masking.

Final validation: `TritonBenchGVerified` builds (3,723 jobs); both focused
source/IO regressions, all 29 metadata/layout/shard tests and all source gates
pass. Independent comparator/trust replay accepts all 48 case/regression
files, including 307 named proofs and 1231 kernel theorem targets;
all 824 snapshot input hashes match. Evidence: `Logs/comparator-check-sve03cl2`.

Coverage and generated sheets now agree on 365 declarations in 189 public
files: 34 full-value candidates, 290 specializations, 37 prepared-input slices,
3 pre-rounding slices and one blocked summary. This count describes scope;
it is separate from the independent proof checks above.

Current progress: **16 complete, 168 pending**, including the user-deferred
attention_kernel_aligned. The next case is attn_fwd_triton, whose bf16 storage
and explicit V-to-fp16 conversion must be preserved in its own FP proof.

## attn_fwd_triton: complete bf16 real/FP source verification

Both complete sources now live in Kernels.lean. The supplied Q/K/V and output
are bf16; the score dot has bf16 inputs and fp32 result. The value dot retains
explicit fp16 casts of probabilities and V, plus its fp16 result. Scales and
accumulators remain fp32. The candidate changes only final division to a
reciprocal multiplication. Both helper calls, separate pointer bases, both
defined stages, repeated diagonal visits, finite additive penalty and source
strides are retained. The wrapper's test-local STAGE assignments do not change
its fixed stage=3; the kernel contract nevertheless supports both defined
outer-stage choices, 2 and 3.

Correct displays the independent base-2 weighted-sum formula, exact source
addresses and both flat IO contracts. RealProjection proves equality of the
complete precision-erased original/candidate with the verified two-helper
causal sources. This transfers their independent formula, successful flat
execution, actual input windows and full memory frame with identical legal
stage, positive block, bounded-allocation, input-pin, zero-undef and output
separation conditions. It is not FP equivalence between benchmark cases.

The FP derivation retains bf16 storage and all casts. It uses div_mul_rcp alone
on actual finite fp32 accumulators and finite/nonzero row denominators after
both helpers. A separate source regression checks the bf16 loads/dots/store,
V conversion, two loops, finite penalty and unchanged common prefix. Both
legal stages have a successful execution/domain witness in one consistent
abstract model. No new GPU experiment or output equation is assumed.

RealCounterexample connects the complete erased source to a concrete disjoint
flat execution with positive first output on Q/K=0, unit scales and V=[0,1],
refuting exact causal masking in the real semantics. The old changed
single-loop model remains under Proofs/LegacyExactMask and has no current
public scope rows. Host launch composition, physical tile validity and
fixed-width address overflow remain outside the per-program statements.

Final validation: `TritonBenchGVerified` builds (3,733 jobs). The bf16 source
regression, refreshed causal source regression, 29 metadata/layout/shard tests
and all source gates pass. Independent comparator/trust replay accepts all
12 case/regression files, including 176 named proofs and
567 kernel theorem targets; all 834 snapshot inputs match.
Evidence: `Logs/comparator-check-_zp921x6`.

The causal source regression header was also corrected to describe its now
complete real IO proof; the assertions are unchanged and its final file is
included in this fresh replay snapshot. The earlier causal milestone remains
valid for its recorded snapshot and all its semantic sources are unchanged.

Coverage/manifest/sheets agree on 366 declarations across 190 public files:
37 full-value candidates, 288 specializations, 37 prepared-input slices,
3 pre-rounding slices and one blocked summary. This is a scope inventory,
separate from the proof checks above.

Current progress: **17 complete, 167 pending**, including the user-deferred
attention_kernel_aligned. Next: batched_vecmat_mult, beginning from its existing
full-loop real proof and adding explicit fp32 source/candidate equivalence.


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

Coverage/manifest/sheets agree on 367 declarations across 191 public files:
40 full-value candidates, 286 specializations, 37 prepared-input slices,
3 pre-rounding slices and 1 blocked summary. These categories record the
statement's scope; they do not replace independent proof replay.

Current progress: **18 complete, 166 pending**, including the user-deferred
attention_kernel_aligned. Next: bgmv_expand_slice. Its old real theorem covers
only the non-accumulating, no-cast masked configuration; remaining work includes
all source flags, precision on offsetted pointer loads and one-arm branch
support in the FP decomposition path. It remains pending.

## bgmv_expand_slice: typed-source and FP preparation

This case remains pending. The original public real statement in
`BgmvExpandSlice.lean` still covers only the masked, no-cast,
non-accumulating configuration. Its general source-lowering statement does not
establish the other numeric branches.

`Proofs/TypedSource.lean` now supplies the complete source matrix: fp16/bf16/fp32
input, fp16/bf16 weights, explicit output dtype, both CAST_TYPE choices,
EVEN_K and ADD_INPUTS. Compile-time cast specialization precedes register
typing, retaining the dtype-changing assignment. The candidate commutes only
`sum(tiled_a * tiled_b, 1) + tiled_out`. Both source versions preserve the
signed sentinel, calculated split length, dynamic n loop, exact pointer
arithmetic, masks, typed products/reduction and output conversion.

This required two shared corrections. Offsetted pointer aliases now retain
their storage dtype on load, including symbolic DTYPE/OUT_DTYPE declarations;
rebinding and nested control flow have dedicated precision regressions.
Source location and `equiv_decompose` now support one-arm conditionals using
`contextual_if_then`, preserving the unselected path and failures. Negative
tactic regressions reject a missing guard or the wrong selected arm.

`Proofs/FP.lean` proves whole-program conditional equivalence for all supported
storage/cast configurations using only `add_commute`. The domain follows the
actual non-sentinel branch, reached n-loop iterations and ADD_INPUTS arm,
checking the addition's actual operands. A symbolic source check ensures that
its domain cannot silently disappear after a failed location lookup.
`Proofs/TypedExecution.lean` establishes successful executions of both complete
sources, a nonempty domain and consistency with the admitted scalar rules in
one abstract model. Its fixture has fp32 input cast to bf16, fp16 output,
nonzero program IDs, K padding and two n iterations with an output tail; the
signed -1 skip is checked separately. This is a semantic witness, not a GPU
experiment or an independent real GEMV formula.

`Proofs/RealSource.lean` proves exact source projection and dtype erasure to
one canonical real source per cast/candidate choice. It preserves the mask and
accumulation flags. Full real correctness remains to be connected: the original
output must be a real input when accumulating, with successful flat execution,
actual load/store bounds and the frame. The absence of a global N mask and the
unscaled split term in c_ptr must remain visible in the contract. Active signed
metadata also needs a nonnegative condition to match natural-address semantics;
-1 remains a valid skip case.

Build validation passes for the shared library, all 18 previously completed
cases and bench/examples (3,881 jobs), and for the current case development
target (3,445 jobs). All 26 tactic/source/precision tests pass, as do 14 source-
audit/layout tests, every source gate and the coverage/manifest checks.
Coverage remains 367 declarations across 191 public files and 173 ports;
this preparation does not upgrade the legacy statement's coverage.

Independent comparator/trust replay accepts all 13 changed shared, case and
regression files, including the legacy real entry: **96 named targets and 615
kernel theorem targets**. All **850** snapshot input hashes match the final
sources. Evidence: `Logs/comparator-check-wf3745g1`.

Current progress remains **18 complete, 166 pending**. The next task is the
full real IO/accumulation proof for bgmv_expand_slice. The user-deferred
attention_kernel_aligned remains skipped.

## bgmv_expand_slice: full real output loop

The complete `n` loop now has an independent real formula for both candidates
and both ADD_INPUTS branches. `RealLoopSource.source_loop` checks that its
statement, dynamic bounds and body are exactly those in the dtype-erased full
source, for every input/weight/output dtype and both CAST_TYPE/EVEN_K choices.
The new proof does not substitute the legacy non-accumulating kernel.

`RealBlock` and `RealBlockExecution` prove every statement of an iteration:
current indices, both masks, the weight load, masked rank reduction, optional
old-output load/addition and output store. The reduction is an independent
finite sum over BK lanes with the source's K mask. Both EVEN_K input choices
have the same masked rank reduction, even when unmasked padding is nonzero.
Masked old-output lanes retain the unspecified-load convention, without an
extra zero-undef premise. The step proof preserves full memory cells outside
its actual stores and retains the setup registers for subsequent iterations.

`RealLoop.Geometry.executes` lifts this to the entire dynamic loop. Its
invariant keeps future output cells equal to initial memory and already
visited cells equal to the dot product plus their initial value when adding.
The result includes successful execution, every active output, and the full-
cell frame. The symbolic length may be zero; positive block size and output
stride are explicit, and the weight region must differ from the output region.
No kernel-output equation is an initial premise. The proof uses real arithmetic
only and is independent of the FP atom admission layer.

`RealLoopExecution.executes` instantiates both candidates and both accumulation
modes with K=3, BK=4, three output lanes in two BN=2 blocks, nonunit strides,
a negative input and a nonzero padded input lane. The loop outputs are
141/172/203 when accumulating and 30/58/86 otherwise. Only Y[11], Y[14], Y[17]
may change; Y[20], the masked tail cell, and all other memory are preserved.
This is a real semantic execution witness, not a GPU experiment or a complete
flat-kernel IO proof.

The case remains pending. Still required: connect the full metadata/sentinel,
input-load, split, optional-cast and pointer prefix to the loop's setup
registers; express the original output as an input in the final IO contract;
and discharge active-address bounds and the flat-memory bridge for the whole
kernel. Active signed metadata must match the natural-address model. If the
public formula is a sum over K rather than the source's masked BK tile, expose
and prove the required K ≤ BK bridge. Public migration and coverage promotion
remain after those obligations.

Validation passes for the current case development target (3,451 build jobs),
all source gates, coverage and the proof-gap manifest. Independent comparator/
trust replay accepts all five new real modules: **32 named targets, 108 kernel
theorem targets**, with all **855** snapshot hashes matching the final sources.
Evidence: `Logs/comparator-check-2d7p7ti3`. No FP admission or shared semantics
was changed in this loop-proof step.

Progress remains **18 complete, 166 pending**. Continue with bgmv_expand_slice's
full kernel prefix and flat IO; attention_kernel_aligned remains user-deferred.

## bgmv_expand_slice: complete real block-state execution

`RealSetupSource.source_body` now checks the entire erased source against its
internal proof representation. It includes program IDs, signed metadata,
the sentinel branch, both input-load paths, N/SPLIT_N, the staged cast's
identity register assignment, exact pointers and the already-proved loop.
`RealSetup` executes that prefix from the initial state and establishes the
loop registers without changing memory.

`RealProgram.source_executes` proves successful execution of the complete
actual source, the independent full rank-K GEMV formula, optional addition
of initial output, and preservation of every memory cell outside active stores.
It covers all source dtypes, cast choices, input-mask choices and both versions.
The K-sum bridge explicitly requires K ≤ BK; positive BN, output stride and
split count are stated, with nonnegative active metadata and separate
weight/output regions. No kernel-output equation is a premise. The -1 sentinel
has a separate full memory-preservation theorem.

`RealExecution` starts with empty registers and exercises the complete source
with N=5, SPLIT_N=2, K=3, BK=4, BN=2, nonzero program IDs and slot, nonunit
strides, negative input, nonzero input padding and a masked output tail.
Both versions and all precision/cast/input-mask choices produce 141/172/203
when accumulating and 30/58/86 otherwise. Only Y[11], Y[14], Y[17] may change;
all other memory, including Y[20], is preserved. The signed sentinel is also
checked. This is an exact real-model witness, not a GPU experiment.

The remaining work is the complete flat-memory IO: metadata and active-address
bounds, the safety proof, and transport of execution/readback/frame. Initial
output must be an explicit read input, while its allocation appears only once.
Input vector bounds must be independent of output-loop length: at length zero,
the active source still loads the input vector before entering the loop.
The public three-entry migration and coverage promotion await those obligations.

Validation: `TritonBenchGInProgress` passes (3,455 jobs); source gates, coverage
and the proof-gap manifest pass. Independent comparator/trust replay accepts
all four new modules: **23 named targets, 59 kernel theorem targets**, with all
**859** snapshot hashes matching current sources. Evidence:
`Logs/comparator-check-25gk31qk`. No FP admission or shared semantics changed.
Progress remains **18 complete, 166 pending**; attention_kernel_aligned remains
user-deferred.

## bgmv_expand_slice: complete public contracts and flat IO

This case now has all three public entries: `Kernels.lean`, `Correct.lean`
and `FPEquiv.lean`. Reusable legacy proofs live in `Proofs/RealFoundation.lean`.
Both correctness specifications cover the complete precision-erased source,
with an independent full rank-K GEMV formula, optional addition of initial
output, successful flat-memory execution, active-address bounds and complete
cell preservation. They cover every represented storage dtype, both cast/load
choices and both ADD_INPUTS branches; K≤BK and the positive block/stride/split
conditions are explicit. The signed -1 sentinel preserves all memory.

The independent IO inputs are the vector, weight matrix and initial output;
the read/write output allocation is listed once. Input bounds remain necessary
even at zero output-loop length because the active source loads the vector
before the loop. Weight/output bounds use the actual split-row mask, without
inventing a global N mask. The exact source output base is retained, including
the split term that is not multiplied by output stride.

The complete flat IO witness uses nonzero program IDs and slot, N=5/SPLIT_N=2,
K=3/BK=4, BN=2, nonunit strides, negative input, nonzero input padding and a
masked output tail. All configurations and both sources produce 141/172/203
when accumulating and 30/58/86 otherwise. Only flat cells 91/94/97 may change;
masked weight padding may lie outside its allocation. A separate complete
signed-sentinel execution preserves every cell. These witnesses instantiate
the displayed contracts, rather than a detached loop or store fragment.

The FP specification preserves the entire typed execution and changes only
the final addition's operand order. Its printer lists `add_commute`; actual
finite-operand conditions follow the reached sentinel/loop/accumulation path.
A consistent abstract model has successful full-source executions and
satisfiable guards. No extra numerical rule or GPU experiment was introduced.

Validation: `TritonBenchGVerified` passes (3,766 build jobs), the source dtype
regression passes, and all 28 layout/source-audit/coverage tests pass. Source
gates, current source/support hashes, scope inventory, proof-gap manifest and
192 generated public specification sheets agree. The reviewed inventory now
contains 368 declarations: 43 full-value candidates, 284 specializations,
37 precomputed-input results, 3 pre-rounding results and 1 blocked declaration.
These classifications do not themselves certify a case as complete.

Independent comparator/trust replay accepts all **24 files**: the 23 Lean
modules in this case plus its typed-source regression. Its explicit inventory
covers **170 named targets** and **856 kernel theorem targets**. All
**868** snapshot input hashes match the completed sources. Evidence:
`Logs/comparator-check-m44fabao`. No shared semantics or admission changed in
this completion step.

Progress is **19 complete, 165 pending**. Continue with `bgmv_shrink_kernel`;
`attention_kernel_aligned` remains the sole user-deferred case.

## bgmv_shrink_kernel: next review obligations

The source includes both store (`SPLIT_K == 1`) and atomic-add branches, and
its `xk_stride` argument is unused. Preserve both facts, the signed sentinel,
masked K tails, low-precision products, fp32 reduction/accumulation/scaling and
output-pointer conversion. The existing detached store/atomic proof surfaces
are not yet complete typed-source contracts.

The initial review found that structural FP execution did not support
`atomic_add`. The milestone below adds an opaque atomic backend selected by
storage dtype and a masked read-modify-write fold that reads each lane's
updated memory. Preserve the per-program execution scope; do not introduce
atomic commutation or launch-wide interleaving assumptions. Required checks
include repeated addresses, masked cells, dtype/operand conversion, unsupported
backends, missing operands and successful full-source runs of both branches.
The candidate can commute the final fp32 scaling with `mul_commute` while
leaving every reduction and memory effect in place. Complete real IO, source
correspondence and independent replay remain required before promotion.

## bgmv_shrink_kernel: typed equivalence and atomic execution

The complete typed source now has original/scalar-left variants for fp16 and
bf16 input/weights and explicit output storage. It retains the signed -1
sentinel, every masked K-loop iteration, the exact SPLIT_K == 1 comparison
and both output branches. Its unused xk_stride parameter stays unused. Both
store and atomic operands explicitly retain Triton's output-pointer cast.
The source regressions check these details and the low-precision product
followed by fp32 reduction/accumulation.

`Proofs/FP.scaling_equiv` proves whole-program equivalence using only
`mul_commute`. Its domain checks the actual accumulator and scale after the
complete reached non-sentinel prefix and K loop. No reduction or atomic
addition is reassociated, commuted or otherwise rewritten.

Structural FP execution now supports atomic addition. Floating atomics use
a separate optional backend selected by storage dtype, so ordinary addition
assumptions impose no atomic laws. Each lane reads the memory updated by
preceding lanes; masks, block-pointer boundaries, repeated addresses and typed
cells are preserved. Missing backend/value/pointer/mask support fails. Tests
cover both algorithm and compute statements, explicit operand conversion,
independent storage selection and unchanged default-precision profiling.
This is one program's execution order, without cross-program interleaving.

`Proofs/TypedExecution` starts from empty registers and exercises both full
sources and both tails in the same consistent model. Its bf16-input/fp16-output
fixture has K=7/BK=2, N=3/BN=4, nonzero batch/slot and a nonzero split ID on
the atomic path. Four store-loop iterations yield 12/12/12; two atomic-loop
iterations update initial output 2 to 8/8/8. Actual final-memory readback proves
these values and preservation of the masked fourth cell. The signed sentinel
preserves all memory; the model satisfies the admitted rules and actual guards.
The opaque reduction in this witness returns one: it demonstrates successful
FP execution and consistency, not GPU numerical behavior or real GEMV values.

`Proofs/RealSource` checks lowering and storage erasure of the complete source
for every represented dtype and both versions. The subsequent milestone below
connects that source to real prefix/loop/tail correctness and an independent
input formula. Both branches' active-address bounds and flat IO/frame, plus
public three-file migration, remain required. The legacy .nat-metadata active
surfaces do not discharge these obligations.

Progress remains **19 complete, 165 pending**. This FP milestone does not
promote the case; attention_kernel_aligned remains user-deferred.

Validation (2026-10-09): the completed benchmark library, all specification
examples and the current development target build together (**3,902 jobs**).
All **33** tactic/matrix/guarded-execution tests and **28** layout/source-audit/
coverage tests pass. Source gates, coverage fingerprints and the proof-gap
manifest pass; scope counts remain 368 declarations across 192 public files.

Independent comparator/trust replay accepts **27 files**: the changed
structural interpreter and model, all four new case modules, both new regression
files and all 19 completed public FP entries. The explicit target inventory
covers **76 named targets** and **734 kernel theorem targets**. All
**874** snapshot input hashes match current sources. Evidence:
`Logs/comparator-check-j0w4lka1`. This checks the shared FP extension without
promoting the still-incomplete real correctness contract of bgmv_shrink_kernel.

## bgmv_shrink_kernel: complete real block execution

`Proofs/RealProgram.source_executes` now establishes successful execution of
the complete precision-erased source against an independent input-memory
sum, for both source versions and every represented dtype. It covers the
actual signed metadata load, arbitrary nonnegative slots, the sentinel guard,
the zero accumulator, all split-K iterations, both scaling orders and the
actual SPLIT_K == 1 store/atomic choice. The atomic result includes the output
cell's initial value. The -1 case separately executes and preserves all memory.

The numeric prerequisites are positive BLOCK_K*SPLIT_K and output stride,
and N ≤ BLOCK_N for the complete N-lane output statement. K need not divide
the loop step; the mathematical sum includes the source's K mask. The proof
does not require input/weight/output disjointness at block-state level: all
data reads occur before the final writes. Every typed cell outside active
output writes is preserved, including masked lanes and metadata. No equation
about output is assumed, and the real proof imports no FP equivalence proof.

The source-to-AST connection is checked in `RealSetupSource.source_body`.
`RealSetup` executes the signed prefix and establishes an explicit-slot loop
invariant. `RealLoop` advances that invariant without substituting natural
metadata, and `RealTail` proves both exact output branches. Its internal
scaled-sum function is named `scaledContribution` to distinguish a pure
mathematical helper from a public specification declaration.

`RealExecution.executes` checks full-source real values from empty registers
with N=3/BN=4, K=7/BK=2, batch/slot 1 and nonunit output stride. Four store
iterations produce [420,1176,1932]; two atomic iterations with split ID 1
produce [335,715,1095]. Both variants preserve all cells except Y[13,15,17],
including masked Y[19]. The sentinel witness also checks full-cell preservation.

`RealAccessSafety.loop_safe` proves the entire loop's active-address bounds:
input loads require their K mask; weight loads require both N and K masks.
In particular, N=0 does not remove the input-read obligation. Remaining work
is composition of signed-prefix and output-tail bounds, transport to flat
memory and the complete IO contract, then public three-file migration.
Progress remains **19 complete, 165 pending**; this is not a completed flat-IO
review, and attention_kernel_aligned remains user-deferred.

Validation (2026-10-09): `TritonBenchGVerified`, `TritonBenchSpecExamples` and
`TritonBenchGInProgress` build together (**3,909 jobs**). The **28** existing
layout/source-audit/coverage tests pass. Independent comparator/trust replay
accepts all **7** new real modules, including **34 named targets** and **80
kernel theorem targets**; all **881** snapshot input hashes match the current
sources. Evidence: `Logs/comparator-check-wddam1xg`. The initial audit caught
the misleading internal `shrinkSpec` name; the final replay checks the renamed
helper and all dependent modules. No audit rule was disabled.
Final source gates, coverage fingerprints and proof-gap manifest checks also
pass; the inventory remains 368 declarations across 192 public files.

## bgmv_shrink_kernel: complete public contracts and flat IO

`Kernels.lean`, `Correct.lean` and `FPEquiv.lean` now expose the complete
original/scalar-left sources, two independent real specifications and the FP
equivalence. The previous natural-metadata surfaces moved to
`Proofs/RealFoundation.lean` solely for reusable mathematical lemmas. Public
real correctness uses the full signed source and imports no FP proof.

`RealSetupSafety`, `RealTailSafety` and `RealSafety` compose bounds for the
metadata prefix, each reached K-loop iteration and the actual output branch.
`RealFlat` transports successful source execution to four disjoint flat
allocations and proves the full typed-cell frame. `RealIO` pins independent
initial X/W values on only the visited valid ranks, initial output only for
atomic addition, and connects them to the mathematical rank-slice sum. Its
contract assumes neither a computed output equation nor prior execution.
The signed -1 branch needs only metadata and preserves all flat memory.

Dimensions and strides stay symbolic. Required numerical conditions are
positive BLOCK_K*SPLIT_K, positive output stride and N≤BLOCK_N. K may have a
partial final block. X is contiguous within its batch, matching the unused
xk_stride argument; its loads remain active independently of N. Both storage
types and explicit output dtypes erase to the same real computation. Atomic
results include the initial output value, within one program's execution.
No cross-program atomic interleaving or launch-wide reduction is asserted.

`RealIOExecution` checks the entire flat IO contract from empty registers,
with K=7/BK=2, N=3/BN=4, batch/slot 1 and nonunit output stride. Four store
iterations produce [420,1176,1932]; two atomic iterations with split ID 1
produce [335,715,1095]. Only flat[141,143,145] may change; masked padding
and every other typed cell are preserved. The sentinel and both source
versions are included. The separate FP witness retains its opaque model and
nonempty guards; its numbers are not presented as real GEMV results.

Validation (2026-10-09): `TritonBenchGVerified` and
`TritonBenchSpecExamples` build together (**3,916 jobs**). All **28** existing
layout/source-audit/coverage tests and the typed-source regression pass.
Source gates, source/support fingerprints, Python links and proof-gap manifest
checks pass. Generated specification sheets match **369 declarations across
193 public files**: 46 full-value candidates, 282 specializations, 37
precomputed slices, 3 pre-rounding slices and 1 blocked summary.

Independent comparator/trust replay accepts all **21 files**: the complete
20-module case and its typed-source regression. The actual inventory includes
**145 named targets** and **661 kernel theorem targets**; all **889** snapshot
input hashes match the final source. Evidence: `Logs/comparator-check-se2cakdp`.
The public FP assumption output is only `mul_commute`. The case is now included
in the verified benchmark library; its temporary development target is gone.

Progress is **20 complete, 164 pending**. Next: `block_sparse_attn`.
`attention_kernel_aligned` remains the sole user-deferred case. Initial review
of block_sparse_attn found an unproved whole-loop `hstep` premise, a CSR column
stride mismatch in the generalized closed form, and missing signed-range
support in contextual FP decomposition. These are recorded in that directory's
README and have not been counted as completed work.


## Block-sparse attention preparation (in progress)

The generalized legacy summary is now classified as a
`public_summary_with_proof_gap`: its `hstep` assumes execution and preservation
of the invariant for the complete CSR loop body. The new real proof must also
keep the independent CSR column stride, which the old generalized closed form
replaced with `num_heads`. Detached copy-slice proofs do not discharge either
obligation. The current inventory is 46 full-value candidates, 281
specializations, 37 precomputed slices, 3 pre-rounding slices, 1 blocked
summary and 1 summary with a proof gap, across 369 declarations/193 files.

The in-progress helper source preserves signed CSR loads, derived offsets and
range bounds; grouped heads, all strides, both mask flags, both channel
branches and explicit fp32/fp16/bf16 storage casts are retained. The full-program
FP rewrite compiles using only `mul_commute` on the actual fp32 score/scalar at
every reached signed-loop iteration. The original `tl.exp` and online
recurrence remain unchanged. Precision erasure also compiles, but is not the
independent attention formula.

The frontend now preserves signed casted loads and their derived pointer
arithmetic. Compile-time `static_print` lowers to an executable no-op. Source
location, reachable guards and `equiv_decompose` support signed ranges, with
regressions for negative visits, missing/incorrect guards, casts and unchanged
intrinsics. Forty-one inventory/layout/source/shard tests and the source-only
corpus gate pass. The generated manifest and 193 specification sheets agree.

Complete typed executions now compose the prefix, two signed CSR iterations
and stores for all eight mask/channel combinations. Readbacks check the actual
visited indices, two normalizer updates, both output channel branches and
masked-cell preservation at nonzero offsets. A generic profiled soundness lemma
transports the complete execution to the candidate in the same model of the
admitted scalar atoms. This is an abstract consistency/execution witness,
not GPU execution or the independent real attention formula.

The shared-semantics rebuild passes for all twenty completed ports and all
examples (**3,920 jobs**). The in-progress block-sparse library also builds
(**3,384 jobs**); the final execution/readback module takes 5.7 seconds.
All **37** tactic/matrix/guarded regression tests pass, in addition to the
41 inventory/layout/source/shard checks above. Independent replay passes for
**57 files**, including the shared changes, regressions, block-sparse source/FP
and execution helpers, and all twenty completed Correct/FPEquiv pairs.
Evidence: `Logs/comparator-check-zi461bho`. The exact 182 named declarations,
1,453 kernel theorem targets and all 900 snapshot input hashes were verified.

The subsequent `StreamingMath` and `CSRMath` helpers prove the normalized
row recurrence equals an independent initial-array weighted sum. Both shifts
and both normalization factors are accounted for. Empty/reversed signed CSR
intervals, all-masked prefixes, arbitrary order and duplicate columns are
covered without a first-visible-key premise. The formula uses independent row
and column strides and the source's one/two-channel condition. This is real
mathematics; it does not assert IEEE behavior or full source execution.

The in-progress library builds with these additions (**3,386 jobs**); six
dedicated mathematical boundary regressions and the source-only corpus gate
pass. Independent replay of the three new files passes in
`Logs/comparator-check-7kg6bl7t`: 34 named declarations, 76 kernel theorem
targets and all 903 snapshot input hashes were verified. The next obligation
is the real source-to-recurrence/input correspondence, followed by
active-address safety and full flat IO/frame. The legacy `hstep` gap remains.
Progress remains **20 complete, 164 pending**; `attention_kernel_aligned` is
still user-deferred.


The actual normalization segment now has a symbolic execution proof. Named
source decompositions check the entire erased source and its loop segments;
the explicit normalization/dot/carry statements are proved equal to the source
slices for both candidate directions. Normalization establishes exact row
outputs, preserves all non-register state fields and untouched registers, and
requires `acc2` only when the source's second-channel branch is enabled.
The subsequent matrix-dot and carry-assignment helpers connect these outputs
to the independent row recurrence. Their score/carry/V-register premises remain
intermediate obligations for the actual surrounding source execution.

The new mixed-row execution regression checks a fully masked row beside a
visible row, the actual first-dot update `(5 + 9) / 2 = 7`, complete state-field
preservation and the absent second-channel register. The in-progress library
build passes (**3,392 jobs**), as do the regression and source-only corpus gate.
Independent replay of the six new proof modules and regression passes in
`Logs/comparator-check-imh1kqo4`: 43 named declarations, 163 kernel theorem
targets and all 910 snapshot input hashes were verified. At that milestone, both earlier replay
snapshots matched the then-current source inputs.
Q/K/CSR/V producers, signed-loop induction, final stores, initial-input pinning
and flat-memory safety/frame still remain; this case is not promoted.


The next execution layer discharges the preloaded-V premise. Both source V
loads now read actual memory through the signed start, row stride and optional
BLOCK_D channel offset, retaining the unspecified values of masked lanes.
`RealAccumulation` composes the loads with both dots and carry writes;
`RealLoopTail` composes the full normalization/accumulation suffix from either
source, with all memory/non-register state and unrelated registers preserved.
Its observed accumulator cells are linked to the independent initial-array
weighted sum. Causality proves inactive V probabilities are zero on stored
query rows, so partial query blocks need no extra all-rows-valid premise.
Local access-safety proofs cover both V-load branches and their actual masks.

The new source-suffix regression uses nonzero pointers/start, stride seven and
an undefined masked V value of 1069. The visible row computes [7,9] and [4,6]
for the first/second channels; both mask choices and the missing acc2 binding
in the single-channel branch are covered. Masked addresses may lie outside
the allocation; a first-channel bound is proved insufficient for the second
channel. The final in-progress build passes (**3,439 jobs**), as do the new
regression and source-only corpus gate. Independent replay of the six new
helper modules and regression passes in `Logs/comparator-check-1e12vigc`:
50 named declarations, 293 kernel theorem targets and all 917 snapshot input
hashes were verified. At that milestone, all three earlier snapshots matched the source inputs.

The Q/K/CSR source producers, initial input/pointer pinning, signed-loop
induction, final stores and whole-kernel flat safety/frame still remain.
Progress is still **20 complete, 164 pending**. Only `attention_kernel_aligned`
is user-deferred; this partial source proof does not promote block_sparse_attn.


The CSR column-read proof exposed an early signed-offset conversion in the
frontend: layout 1, column stride 13 and visit -2 addressed slot 13 instead of
11. Static region addition now preserves the complete signed sum before the
natural-address boundary. Scalar/vector execution regressions check negative
addends, parentheses and later recovery to a nonnegative address. The actual
column read and `start_n` assignment now have a symbolic execution proof and
initial-array formula connection, keeping the column stride independent of
heads. Layout/visit production and complete input validity remain outer-kernel
obligations. The shared frontend change invalidates the current-input hash
match of the historical replay receipts above; their replacement validation is recorded below. The case is still not complete.


Both actual K-load branches also have source correspondence, symbolic memory
execution and local active-address safety proofs. A nonsquare 2×3 regression
covers transpose orientation, both channel offsets, masks, nonzero undefined
masked values, complete state preservation and insufficient second-channel
allocation. Score dots/scaling/causal masking and Q/prefix production remain.

The shared rebuild found one consumer of the old clipping behavior:
`bgmv_expand_slice/Proofs/RealFoundation.lean` also contained unused historical
headlines for arbitrary signed slots. Its old lowered AST and proofs are now
explicitly isolated in `LegacyClippedOffset.lean`, with a checked address
counterexample. The current proof imports only the generic helpers and its
setup now uses the corrected complete signed sum. The public contract still
requires the existing sentinel-or-nonnegative metadata condition; its scope is
unchanged. The adjusted complete public correctness and flat-IO proofs pass the shared rebuild.


Validation of the signed-address correction passes: **3,941** shared build
jobs, the CSR-column and nonsquare K execution regressions, and all **38**
tactic/matrix/guarded tests. Independent comparator/trust replay accepts all
**100 files** in `Logs/comparator-check-b880f0sd`. Its inventory includes
**524 named declarations** and
**3225 kernel theorem targets**; every one of
**923** snapshot input hashes matches current code.
This replaces the earlier shared-source snapshots' current-input claims.
Coverage categories and public specifications are unchanged; progress remains
**20 complete, 164 pending**, with only `attention_kernel_aligned` deferred.
At that validation point, Q/prefix producers, score composition, signed-loop
induction, final stores, initial-array pinning and whole-kernel safety/frame
remained for block-sparse. The next milestone below closes score composition.


The next layer composes the signed CSR column read, actual K loads, one/two
score dots, scaling and causal masking, with score-to-initial-array correspondence
on stored query rows only. The complete CSR visit then composes this producer
with normalization, actual V loads and the carry update. Its assumptions are
incoming registers and carries; no whole-body execution or preloaded qk/K/V
is assumed. Partial query blocks remain allowed. The nonsquare regression
checks scores 94/320, all source/mask/channel variants, the complete visit,
state frames and local score-prefix access bounds.

The score/complete-visit integration passes the in-progress build (**3,443
jobs**), the imported execution regression and the corpus source gate.
Independent replay accepts both new proof modules and the regression in
`Logs/comparator-check-ij4d3nzm`: **41 named declarations**,
**101 kernel theorem targets** and all
**926 snapshot input hashes** were checked. The prior
100-file shared replay still matches all its inputs. This closes local score
production and single-visit execution, not the prologue, loop induction or
full kernel IO proof.

Q loads, CSR row-bound reads, input/address pinning, signed-range induction,
final stores and whole-kernel safety/frame remain. Coverage stays at **20
complete, 164 pending**, with only `attention_kernel_aligned` deferred.


The entire real pre-loop source now executes from the initial memory/program
ids and produces all fixed registers and zero carries. `RealSetup` preserves
batch/query/KV-head mapping and distinct Q/K/V strides; `RealQuery` loads both
masked/unmasked Q branches and the optional second channel; `RealRows` loads
the signed CSR endpoints using the independent row stride. `RealPrologue`
composes these actual source segments, their readbacks, state/register frames
and active-Q/unmasked-CSR-cell access safety, without a prepared-register or
prefix-execution premise.

The regression starts with empty registers, nonzero batch/head and distinct
strides. A partial three-row tile produces Q values 181/183, masked 1188 versus
unmasked 188, the expected grouped K/V pointers and CSR endpoints -2 and 1.
It covers both source/mask/channel variants, absent single-channel registers,
complete state preservation and an insufficient second-channel allocation.

The full prologue passes the in-progress build (**3,447 jobs**), its imported
execution regression and the corpus source gate. Independent replay accepts
all **5 new files** in `Logs/comparator-check-cntv6qvg`. The exact **53 named
declarations**, **576 kernel theorem targets** and all **931 snapshot input
hashes** were verified. Both preceding shared and score/visit snapshots still
match their inputs. These checks close the pre-loop producer, not the signed
range induction, full output-store/flat-IO proof or independent-array pinning.

The case remains **20 complete, 164 pending** overall. Signed-loop induction,
initial-array/address correspondence, loop access safety, final stores and
the full flat-memory frame remain; only `attention_kernel_aligned` is deferred.


The signed-loop invariant now executes every actual source visit from the
prologue, preserving fixed inputs and deriving the exact normalization carry.
It handles signed endpoints, empty and reversed intervals without assuming
body execution or a visible first key.

The complete signed-loop induction passes the in-progress build (**3,448
jobs**), the imported loop regression, source gate and manifest check.
Independent replay accepts both new files in `Logs/comparator-check-bks7qye2`:
**24 named declarations**, **84 kernel theorem targets** and all **933 snapshot
input hashes** were verified. The regression covers a range crossing zero,
repeated CSR columns, both source/mask/channel variants and empty/reversed
ranges. The preceding shared, score/visit and prologue snapshots still match
their inputs. This establishes execution through the entire loop; the history
still needs correspondence to the independent initial-array attention formula.

Progress remains **20 complete, 164 pending**. Final stores, independent
initial-array/address correspondence, loop access safety and the full
flat-memory frame remain. Only `attention_kernel_aligned` is user-deferred.


The actual output-address setup and final masked stores now compose after the
complete signed loop. `RealStores.kernel_executes` executes the full real
source; `RealStoreOutputs` proves tagged cell readback, channel separation
under row-stride bounds and the logical-region memory frame. Local output
access safety requires only active rows, with second-channel bounds needed
only on that branch. The output mask remains present for both EVEN_M choices.

The partial 3×2 store regression checks values 71/72 and 91/92 at addresses
601–604, unchanged natural-number cell tags on masked rows, absent-channel
preservation and insufficient second-channel allocation. An empty-CSR test
runs the complete source through zero output stores and mask preservation.

The final-store integration passes the in-progress build (**3,451 jobs**),
the imported store regression, corpus source gate and manifest check.
Independent comparator/trust replay accepts all **4 new files** in
`Logs/comparator-check-hfwnzag2`: **33 named declarations**, **83 kernel theorem
targets** and all **937 snapshot input hashes** were verified. The earlier
shared, score/visit, prologue and loop snapshots still match their inputs.
This closes execution of the complete real source and its logical-region
output readback/frame. Initial-array formula correspondence, loop access
safety and the full flat-memory IO contract remain pending.

Progress remains **20 complete, 164 pending**. `attention_kernel_aligned` is
still the sole user-deferred case.

The complete real source is now connected to independent initial Q/K/V and CSR
arrays. `RealInputs.Memory` pins input cells and `RealInputs.Valid` expresses
signed metadata-slot and selected-key validity; neither includes execution,
an intermediate score nor an output equation. `RealFormula.source_correct`
proves the independent CSR attention formula for every stored output channel
and the logical-region memory frame. The regression uses signed visits
[-1,2), distinct strides, repeated keys and partial query/key tiles, yielding
32/5 and 64/5 while preserving the masked row's natural-number cell tag.

This formula integration passes the in-progress build (**3,455 jobs**), its
imported regression, source gate and manifest check. Independent replay accepts
all **5 new files** in `Logs/comparator-check-hke_pb47`: **27 named declarations**,
**72 kernel theorem targets** and all **942 snapshot input hashes** were verified.

`RealVisitSafety` and `RealLoopSafety` now prove the complete source's access
safety through every signed CSR visit and final masked store. `RealFlatten`
establishes support for the shared flat-memory bridge. The safety regression
provides small allocations for checked lanes and extra padding for unchecked
Q/K/V, and rejects dropping the K mask with the smaller allocation. Independent
comparator/trust replay accepts all **4 new files** in
`Logs/comparator-check-upzvsh_w`: **26 named declarations**, **68 kernel theorem
targets** and all **946 snapshot input hashes** were verified.

`RealWindows` derives the trace bounds from initial-array address windows;
`RealFlat.source_correct` transports execution, the initial-array formula and
the complete frame to flat memory. No trace-safety or whole-body premise is
left to the caller. The flat proof uses disjoint allocations, zero undefined
loads as required by the shared bridge, positive BN and nonoverlapping output
rows. The in-progress build passes (**3,460 jobs**) and the complete flat
regression passes for every source/mask/channel choice: output at 80/81 and
an untouched natural-number padding cell at 83. Independent comparator/trust
replay accepts all **3 new files** in `Logs/comparator-check-b7z0bfmz`:
**15 named declarations**, **55 kernel theorem targets** and all **949 snapshot
input hashes** were verified. The source gate and manifest pass. All seven
preceding current-snapshot receipts still match their inputs.

Progress remains **20 complete, 164 pending** until the new full-source
contracts replace the legacy public specification and pass final acceptance.
Only `attention_kernel_aligned` is user-deferred.


The public block-sparse source and contracts now use the shared
`Kernels/Correct/FPEquiv` layout. `RealIO` displays initial-array pinning,
value-dependent CSR read windows, active/padded data allocation and the full
output frame. `RealContract` proves that this IO is the checked complete
source for all fp32/fp16/bf16 storage choices. Both real specifications display
the independent weighted-sum formula and require only input/index/layout
conditions; no `hstep`, trace or computed-output premise remains. FP prints
only `mul_commute` and keeps `tl.exp`, casts, masks and the complete signed loop.

The former natural-metadata entry is retained as `Proofs/LegacySummary.lean`
and is imported by neither current proof. The coverage manifest and spec
sheets now describe the three current public specifications. The corpus source
gate, coverage fingerprints, manifest and layout tests pass. `lake build
TritonBenchGVerified` passes (**3,834 jobs**), as do all **13 case regressions**.
`BlockSparseFlat` now exercises the actual public specifications for every
storage/source/mask/channel choice with concrete initial arrays and disjoint
allocations. Independent comparator/trust replay accepts all **59 files** in
`Logs/comparator-check-roalqlu7`: **480 named declarations**, **2,709 kernel
theorem targets**, and all **953 snapshot input hashes** were verified against
the actual target inventory before acceptance. This supersedes the affected
component receipts from before the public-file migration.

Progress is **21 complete, 163 pending**. Only `attention_kernel_aligned` is
user-deferred. Continue with `bmm_chunk_bwd`, retaining its runtime chunk
limit, partial contraction/output tiles and optional residual in both channels.


### Chunked backward matmul preparation

The complete typed source now retains the in-body minimum, all partial-tile
masks and optional residual, with all nine A/Dout storage pairs and explicit
residual/output dtypes. A nonzero mixed-storage execution regression exposed
a nested-load inference issue. The DSL correction retains the declared pointer storage through a nested
load/cast assignment. Both source transcriptions keep their original loads
and conversions, with no added load kwargs; interpreter semantics and
admission rules are unchanged. AST regressions verify
that A is read as fp16 before its bf16 dot conversion and that residuals are
read as bf16 before fp32 conversion.

The full FP proof uses only `add_commute` at the reached residual addition.
A consistent abstract model supplies complete original/candidate executions
for a last chunk of length three with two dot iterations, nonzero batch/chunk/
group offsets and partial output tiles. It obtains 2 without the residual,
5 with it, and preserves the masked fp16 cells containing 9. These witnesses
are not GPU measurements or the real matrix-product proof.

The real preparation connects every represented dtype to the actual source,
proves the symbolic pre-loop setup and derives masked block summation without
requiring whole blocks. A separate arithmetic regression with nonzero padded
inputs obtains 112, or 129 with the residual. The registered in-progress
library builds (**3,445 jobs**); both regressions, source gates, coverage
fingerprints, manifest and layout checks pass. Independent comparator/trust
replay accepts all **10 preparation/test files** in
`Logs/comparator-check-4a5890yv`: **42 named declarations**, **505 kernel
theorem targets** and all **963 snapshot input hashes** were verified.

Progress remains **21 complete, 163 pending**. The complete contraction loop,
residual/output producer and flat-memory IO/frame are still required for
`bmm_chunk_bwd`. Only `attention_kernel_aligned` is user-deferred.


The next BMM real layer now derives both actual masked load tiles from memory
and evaluates the dot/add as the masked contraction sum. `RealLoads` proves
that its operations are the complete five-statement source loop, including
both pointer advances. A regression fills inactive memory and undefined
values with large nonzero numbers and still obtains 13×5=65 in the live lane
and zero in the masked column. The library builds (**3,447 jobs**), and the
regression passes. Independent replay accepts all **3 files** in
`Logs/comparator-check-wvamemyp`: **12 named declarations**, **32 kernel theorem
targets**, and all **966 snapshot hashes** verified. Those snapshot hashes matched at acceptance. Subsequent DSL/source
corrections require updated replay. Full loop induction and the
residual/store/flat-IO assembly remain; progress stays **21 complete, 163 pending**.

The BMM `RealStep` module now executes the complete five-statement loop body
and proves its state/register frame. Independent replay accepts its **3 named
declarations** and **7 kernel theorem targets** in
`Logs/comparator-check-l8f6ws2u`; all **967 snapshot hashes** matched at
acceptance. The registered in-progress library builds (**3,448 jobs**).
This establishes one iteration; it does not promote the case to complete.

### Load/cast storage correction — accepted

A nonzero BMM witness exposed loss of pointer storage dtype in a load/cast
assignment. Review found the same issue in the `attention_fwd_triton3` resume
arm. The DSL now preserves declared fp16/bf16/float8/parametric storage through
these assignments before applying the requested floating conversion. Both
transcriptions retain the original load kwargs; no interpreter or admitted
atom changed. Generic AST regressions cover pointer rebinding, positional
masks, checked block pointers and a parameterized dtype. The attention
regression also executes its actual selected assignment with a nonzero value
and an observable conversion.

The integrated build of `VeriTile`, `TritonBenchGVerified`,
`TritonBenchSpecExamples` and `TritonBenchGInProgress` passes (3,985 jobs).
All ten focused regressions pass, as do the source gate, coverage fingerprints,
proof-gap manifest and layout checks. Independent comparator/trust replay
accepts 30 files in `Logs/comparator-check-qgooy8if`: 130 named declarations,
895 kernel theorem targets, and all 973 snapshot hashes matched at acceptance.
The remaining dtype regression initially failed parsing because the audit
import reserves its local variable name `full`. Renaming that variable to
`fullExpr` fixes the collision; its tests are unchanged. Follow-up replay in
`Logs/comparator-check-cwzs4h63` accepts that regression and the new BMM
`RealFinish` module: 2 files, 10 named declarations, 16 kernel theorem targets,
and all 974 snapshot hashes matched at acceptance. Both receipts were checked
against the complete named declaration inventories and snapshot inputs before
further source changes. Earlier receipts refer to their acceptance snapshots.

BMM now has a complete real dynamic-loop induction (`RealLoop`), optional
residual execution (`RealResidual`), final masked store/readback/frame
(`RealStore`), and complete source execution from the initial state
(`RealFinish`). `BmmChunkBwdLoop` executes a nonzero partial-tail example,
checking accumulators 112 and 339, the masked column, and the source's wider
Dout row-load mask. These modules and that regression passed the replays above;
the registered in-progress build passes (3,452 jobs). The independent-array
formula and flat-memory IO/safety assembly remain. BMM is still pending;
progress remains 21 complete and 163 pending, with only
`attention_kernel_aligned` user-deferred.


### BMM full-contract assembly — independent replay in progress

`bmm_chunk_bwd` now has the shared `Kernels` source, both independent real
specifications in `Correct`, and admitted residual-add equivalence in
`FPEquiv`. The complete real execution, memory readback/frame, independent
initial-array formula and actual access safety are connected to the flat IO
contract. Partial contraction/output tails, all independent strides, three
program IDs, all nine input storage pairs and both residual arms are retained.
Only `add_commute` is used by the FP proof. The older specialization is kept
under `Proofs/Legacy` and does not supply the new public contract.

The shared `UKernelIO.Implements3D` passes all three program IDs to the formula;
the previous `Implements` is its compatible specialization. This is necessary
because the final chunk length depends on the third coordinate. The existing
flat-memory bridge remains the single shared assembly proof.

The integrated libraries build (4,002 jobs). Source gates, the proof-gap
manifest, 370 coverage reviews across 195 public files and both layout tests
pass. The new full-source regression uses nonzero tile/batch/chunk/head
coordinates, a length-three final chunk and partial output masks. Its flat
executions store 112/339 without residual and 129/358 with residual, preserving
inactive fp16 cells containing 9. Six focused regressions and a 37-file
independent comparator/trust replay are being finalized. Progress remains
21 complete and 163 pending until that receipt's target inventory and snapshot
inputs have been verified; `attention_kernel_aligned` remains user-deferred.


### BMM backward accepted; forward migration continues

The complete 37-file BMM backward comparator/trust batch passes at
`Logs/comparator-check-l1uxmrt9`. Before further source edits, the verifier
matched all **207 explicitly named declarations**, all **1,179 kernel theorem
targets** and all **990 snapshot input hashes**. The receipt includes both
public specifications, all source/proof files, the shared three-coordinate IO
bridge and six focused regressions. The integrated 4,002-job build and source,
coverage, manifest and layout checks also pass.

Progress is **22 complete, 162 pending**. Only `attention_kernel_aligned` is
user-deferred. The next case, `bmm_chunk_fwd`, retains whole-tile causal skipping
without introducing an elementwise diagonal mask, signed sequence-ID filtering,
partial K blocks and the distinction between shortened load masks and the full
chunk store mask. Its complete-source FP draft uses only `add_commute`; the
remaining real execution, independent formula, flat IO and verification work
will be accepted separately.


## BMM forward complete-source acceptance

`bmm_chunk_fwd` now exposes `Kernels.lean`, `Correct.lean` and `FPEquiv.lean`.
Both real specifications use independent initial arrays and signed sequence
IDs. The full source computes the shortened chunk length, visits partial K
blocks, filters unequal IDs and writes padded final-chunk positions as zero.
Causal mode skips whole tiles; it does not apply an elementwise triangular
mask to a retained tile. Successful flat execution, actual masked-address
safety and every-cell frame are proved for both sources.

The FP candidate reverses only each fp32 carry-plus-dot addition. Its sole
atom is `add_commute`, with finite actual operands checked after the loads in
every reached iteration. All nine A/B storage pairs, dot priority, output
conversion, metadata and branches remain explicit. A consistent abstract
model provides complete original/candidate runs through two K blocks and both
metadata arms, so the domain is not vacuous.

The integrated build passes 4,033 jobs. Three focused regressions cover
storage, dot options, masks, signed IDs, causal skipping, independent products
336/1025, zero-filled padding and tagged-memory preservation in both region
and flat execution. Sixteen layout/coverage tests and source gates pass; the
inventory is 371 declarations across 196 public files.

The final comparator/trust batch accepts all 34 selected files. Exact named
target checking covers 189 declarations and 1,299 kernel theorem targets;
all 1,023 snapshot input hashes match. Receipt: `Logs/comparator-check-gs9xcq3i`.

Progress is **23 complete, 161 pending**. `attention_kernel_aligned` remains
the only user-deferred case. The next case is `bmm_optimized`: retain all eight
mask combinations, both CTA-ordering arms and the implicit output-store cast;
its real IO contract must account for both `tl.num_programs` dimensions.

## Complete optimized BMM contract and pointwise IO bridge

`bmm_optimized` now exposes complete source, real correctness and FP equivalence
through `Kernels.lean`, `Correct.lean` and `FPEquiv.lean`. Both addition orders
have independent initial-array product formulas with successful flat execution,
actual active-address safety and preservation of every cell outside the output
tile. All eight mask-elision arms, both runtime grid dimensions, pointer updates,
CTA swizzling and the final short group remain. Sizes are symbolic; positive
tile/group sizes, divisibility for elided masks and a valid host grid are explicit.
Common fp32/fp16/bf16 storage, fp32 accumulation and allow_tf32=false are retained.

The source omits `other` on masked loads. Real matrix-product correctness
therefore explicitly uses zero unspecified load values. `UndefinedTail.lean`
refutes removing that premise: a supported 32×32 tile, K=33 and all-one inputs
writes 808 instead of 33 when unspecified values are five. This is complete
real-interpreter execution, not a GPU measurement. FP equivalence does not need
zero unspecified values: only actual finite carry/dot operands and admitted
`add_commute` are used, with all source precision and memory behavior retained.

The shared `UKernelIO.ImplementsAt` bridge allows launch-grid conditions to be
stated before flat execution. The original three-axis IO statement is unchanged
and proved equivalent to a family of pointwise statements; its introduction
lemma reuses the same bridge. Regression checks include both runtime grid axes
and a 65×48 output with K=33, M/N/K tails, a shortened CTA group, nonzero outputs
and exact preservation of neighboring fp16-tagged cells in region and flat memory.

The three benchmark/example libraries build successfully (4055 jobs).
All five focused regressions pass, as do the 28 layout/coverage/source tests
and all source gates. The reviewed inventory has 373 declarations across 197
public files. Independent comparator/trust replay accepts all 34 selected
files: all 28 case modules, the shared pointwise IO core and five regressions.
Every one of the 165 named declarations is in the replay inventory;
1275 kernel theorem targets and all 1055 snapshot input hashes are checked.
Receipt: `Logs/comparator-check-_3eonbb6`. These checks certify the displayed Lean
contracts and their dependencies, not concrete GPU execution.

### Complete cache gather acceptance

`cache_transform_triton` now has shared complete Triton sources, four public
real correctness specifications and two public FP equivalences. Prefill and
full BLOCK_SIZE × HIDDEN_DIM decoding use independent initial arrays and
actual strides. Both original and integer-index-commuted versions successfully
execute in one flat address space, return both active output values and retain
every other typed cell. Active decode indices are nonnegative; prefill's signed
maximum has no artificial zero initializer. The current flat bridge's zero
unspecified-value premise remains explicit. Address arithmetic is mathematical.

Typed execution covers all nine independent cosine/sine fp32/fp16/bf16 pairs,
symbolic shapes and arbitrary opaque payloads. Index commutation is exact, so
both FP assumption printers report `none`. The source keeps its original masks,
omitted `other` and both stores. The host's unpadded metadata is not repaired:
`HostSafety.prefill_unpadded_metadata_safe` refutes access safety when its
allocation is shorter than N_ELEMENTS. Output bounds retain the cache strides
that the source also uses for fresh output buffers.

The complete dependent library build passes, and the verified benchmark target
builds with the new public entries. Three semantic regressions check signed
lowering/reductions, typed execution and mixed storage, full flat decoding IO,
inactive tagged cells, and the three-entry/four-load metadata failure.
All 28 layout/coverage/source unit tests and the corpus source gates pass.
The reviewed inventory is 377 declarations across 198 public files.

Independent comparator/trust replay accepts all **38 files**: 20 case modules,
15 shared semantics/DSL/memory modules and three regressions. The actual replay
inventory contains all **508 named proof targets** and **6,188 kernel theorem
targets**; all **1,077 snapshot input hashes** match the reviewed sources.
Receipt: `Logs/comparator-check-dvgpy8_e`. These are Lean proof checks, not GPU
measurements or certification of every host call.

Progress is **25 complete, 159 pending**. `attention_kernel_aligned` remains the
only user-deferred case. The next case is `chunk_bwd_dqkg`: its current proof
collapses the value loop under `0 < V ≤ BV`; the host runs multiple iterations
when V > 64. Its README now states that limitation explicitly. The full loop,
independent-array real flat IO and FP equivalence still need to be completed.


## Chunk backward value-loop milestone (2026-10-09)

The case remains pending. `Proofs/RealMath.lean` now states independent
input-array sums for the four value-axis carries and formulas for all three
final outputs. `RealStep.lean` checks the entire twelve-statement real loop
body with arbitrary incoming carries and loop index. `RealLoop.lean` reaches
those carries through the actual prologue and the complete ceiling-divided
loop. Its invariant preserves every memory cell, unspecified values, launch
coordinates and untouched registers; `full_sums` removes the value-block
partition without assuming V ≤ BV or divisibility. This is real arithmetic;
no floating-point reassociation is inferred from it.

`HostSafety.lean` checks an active H/DH address in the first original Python
fixture: the last head's second time chunk loads offset 32768, while its
`[2,4,64,64]` allocation contains 32768 elements. The access-window safety
claim is formally refuted. A separate checked lane shows why the source's
NT*K column mask must not be replaced by a logical-K mask.

The internal target `TritonBenchGInProgress` builds. There are still no new
public correctness or FP specifications: the typed source/reduction bridge,
post-loop execution, all three output stores, flat address safety, full IO and
admitted FP rewrite remain to be connected. The corpus count remains
**25 complete, 159 pending**, with only `attention_kernel_aligned` deferred by
the user.

Independent comparator/trust replay accepts all five files in this milestone
(the legacy real source/proof and the four new internal modules). All **78
named proof targets**, **269 kernel theorem targets**, and **1,081 snapshot
input hashes** were checked. Receipt: `Logs/comparator-check-5rvj3aqt`.
This validates the internal milestone only; it does not change the completion
count or certify the pending FP lowering.

## Typed floating reduction progress (2026-10-09)

The shared source lowering retains floating sum dtypes and explicit dtype
conversions. Whole-tile sums flatten before one reduction. The FP structural
interpreter preserves this as a distinct opaque operation, supports equal-size
row-major reshape, and forwards the execution precision to the typed backend.
Real execution, memory safety, region typing, flattening and erasure have cases
for the new constructor. A real-only theorem connects the flattened matrix
sum to the double sum used in independent formulas.

`chunk_bwd_dqkg/Kernels.lean` now holds both full typed programs; its original
history masks and all value-loop iterations remain. The legacy real AST and
new arbitrary-length real loop compile with the flat sum. TypedSum regressions
include a complete successful execution and a backend distinguishing flat from
nested reductions. The new real-source bridge is still being checked. The
epilogue, stores, flat IO and admitted FP candidate remain.

Independent comparator/trust replay accepts all **24 files** in this semantic
milestone: 17 shared modules, the typed-sum regression, both chunk source
modules and the four real-loop/safety modules. The verified inventory contains
**513 named proof targets**, **5,282 kernel theorem targets**, and **1,086
matching snapshot input hashes**. Receipt: `Logs/comparator-check-g4j1q471`.
This receipt precedes the BGMV execution-witness backend update; that update
requires its own checks. It does not cover the unfinished typed-source bridge
or restore the reopened BGMV FP proof. Progress remains **24 complete, 160
pending**.

The affected-library build detected the reopened BGMV issue above; it must not
be reported as passing. No additional completed case is claimed.

## Typed source and FP proof milestone (2026-10-09)

The full `chunk_bwd_dqkg` source now preserves the Python dtype-attribute casts
by spelling their resolved targets explicitly. A direct FP-AST regression
checks the history and score conversions, final-add operands and output cast.
`RealSource.lean` connects every supported storage combination to the full real
body. `FP.lean` proves the complete candidate using only `add_commute`, with
finite checks on the actual carry and typed dot after the whole value loop.

`TypedExecution.lean` checks a consistent abstract model with two value
iterations, mixed storage and partial time/key/value tiles. Both complete
programs execute successfully. Actual dq/dk/dg readback and preservation of the
masked lanes are checked; this is an execution witness, not a GPU experiment.
The shared witness backend now implements typed sums, restoring the existing
BGMV shrink execution and output tests. Both BGMV expand real specifications
also compile; its FP proof is still reopened pending the typed-atom decision.

The combined in-progress chunk, BGMV shrink witness and BGMV expand correctness
build passes (3,475 targets). Independent replay and trust checking accept
**10 files**, covering **28 named proof targets**, **905 kernel theorem
targets** and **1,089 matching source hashes**. Receipt:
`Logs/comparator-check-fli84kwp`.

The subsequent `RealPost.lean` connects the full-loop totals to the old tail
execution proof and all three independent output formulas. This uses V only
as a formula summation extent; the source retains its original BV and all
iterations. Its separate independent check passes: eight named proof targets,
22 kernel theorem targets and 1,090 matching source hashes, recorded in
`Logs/comparator-check-zo3rrlyx`. Full flat IO, access bounds, memory frame and
candidate real correctness still remain, so
the completion count is unchanged at **24 complete, 160 pending**.

The `chunk_bwd_dqkg` real candidate transfer is checked separately in
`Logs/comparator-check-pz2i0zgm`: three named targets, 22 kernel targets and
1,091 matching source hashes. Both real versions reach the same final state;
flat IO, active bounds and the full memory frame still remain before acceptance.
The round remains at 24 accepted and 160 pending.

`chunk_bwd_dqkg/Proofs/RealFrame.lean` now proves preservation of all tagged
cells outside the three actual active output windows, including masked lanes.
The full loop is included. Independent check: `Logs/comparator-check-jzyvgs96`,
nine named targets, 25 kernel targets and 1,092 matching snapshot input hashes.
Access bounds and flat IO are still pending; this is not a new accepted case.

The next chunk milestone completes all three address-safety components and
successful flat execution of both sources. Public Correct/FPEquiv entries now
compile in isolation, with whole-V independent formulas and only add_commute
on the FP side. The old global tile injectivity premise was stronger than
needed for masked tails: the reused scatter readback now requires collisions
be absent only between active writes. K≤row stride discharges that condition,
including K<BK, with a dedicated K=3/BK=4 regression. Legacy helpers moved into
Proofs/LegacyReal.lean. A complete build and current 22-file independent audit
are pending; the count remains 24 accepted and 160 pending until they pass.

The complete chunk milestone is now accepted. `lake build
TritonBenchGInProgress` passes (3,467 jobs), and the typed-source and masked-tail
bounds regressions pass. Current independent replay and trust checking accept
all 20 case files and both regressions: **22 files**, **141 named proof
targets**, **1,119 kernel theorem targets** and **1,101 matching snapshot
inputs**. Receipt: `Logs/comparator-check-xeoveh_w`. The two real entries
cover the entire value loop; the FP printer reports only `add_commute`.
Progress is **25 complete, 159 pending**; the reopened BGMV FP proof remains
pending, and `attention_kernel_aligned` remains user-deferred.

## Chunked scalar cumulative sum (2026-10-09)

`chunk_cumsum_kernel` now has three reader-facing entries and complete typed
original/sum-left sources. Both real specifications use the independent global
prefix-sum formula, positive symbolic block width, all loop iterations and
partial tails. Full flat execution, actual masked-address bounds and complete
cell frame are transferred from the checked real loop. The candidate real
rewrite is independent of FP admissions. The FP proof retains all source
casts, scan, sum, masks and stores, using only `add_commute` at each reached
carry update.

The shared structural interpreter now delegates scan to an opaque backend,
retaining kind, axis, direction and computation precision. Missing support
fails; no scan decomposition or reduction law is introduced. Meaningful
regressions distinguish backend parameters and check the typed source AST.
A consistent abstract fp16-to-bf16 execution covers three chunks, nonzero
program ID, active output values and untouched neighboring cells.

`TritonBenchGInProgress` and `TritonBenchSpecExamples` build successfully
(3,612 targets), and both scan/source regressions pass. Current independent
replay and the affected earlier-directory build are pending. Progress remains
**25 complete, 159 pending** until this case passes final acceptance.

The scalar cumsum independent audit now passes: **15 files**, **117 named
proof targets**, **1,131 kernel theorem targets**, **1,113 matching snapshot
inputs**, receipt `Logs/comparator-check-mhghduz5`. All public real/FP entries,
full-loop and IO helpers, source conversion, abstract execution/model and both
regressions are covered. The earlier-directory affected build has one target
remaining; acceptance count is unchanged pending its result.

The affected 24 earlier accepted directories now build successfully: **439
explicit modules**, **3,920 build targets**. Only the previously reopened
BGMV expand directory was excluded; the unfiltered verified-library target
is not claimed to pass. Together with the completed 15-file receipt above,
this accepts `chunk_cumsum_kernel` and advances the round to **26 complete,
158 pending**. `attention_kernel_aligned` remains user-deferred.

The next vector-cumsum case has temporary compiled full typed sources, FP
composition using only `add_commute`, a real-source erasure bridge and generic
per-column prefix partition lemmas. Its existing repository proof still covers
a single-block program and a separate precomputed store slice. Complete loop
correctness, cross-chunk output uniqueness, flat IO and final acceptance remain;
it is not included in the completed count.


## Chunked vector cumulative sum (2026-10-09)

`chunk_cumsum_vector` now has the three public entries and complete typed
original/sum-left sources. Both real specifications cover the entire time
loop against independent per-column prefix sums over initial inputs, with
symbolic time/channel sizes and strides, positive BT, both partial tails,
actual active-address bounds and successful flat execution. The frame preserves
all other tagged memory cells. Active output addresses must be distinct across
all chunks; masked lanes need not be injective. BS need not be positive.

The FP proof checks the actual carry and axis-0 block total after the reached
store and source branch. It uses only `add_commute`, preserving the triangular
dot, TF32 setting, input promotion, independent output dtype and both boundary
axes. A consistent abstract bf16-to-fp16 run covers three chunks, nonzero
feature/head IDs, both tails, output values and untouched neighboring cells.
This witnesses satisfiable FP premises, not GPU numerical behavior.

`TritonBenchGInProgress` builds successfully (3,495 targets), and both source
and layout regressions pass. Independent comparator replay and trust checking
accept **18 files**, **89 named proof targets**, **948 kernel theorem targets**
and **1,130 matching snapshot inputs**. Receipt:
`Logs/comparator-check-bv8x3udw`. The reviewed inventory has 380 declarations
across 201 public specification files; coverage fingerprints, proof-gap
manifest and generated specification sheets agree.

The 173-port source gate and 16 scanner regressions pass. The scanner now
recognizes the explicit source headers and quoted kernel parameters, expands
literal original statement branches, resolves unambiguous dtype attributes
from source bindings, and excludes multiline region declarations from runtime
statement lists. Unknown or ambiguous dtype bindings remain unresolved. These
lexical checks complement source review and Lean proofs.

Progress is **27 complete, 157 pending**. `attention_kernel_aligned` remains
user-deferred, and the reopened BGMV FP admission decision remains pending.
The next case is `chunk_delta_fwd`.


## Chunk delta preparation and static-range automation (2026-10-09)

`chunk_delta_fwd/Proofs/TypedSource.lean` preserves both complete typed sources,
including both nested loops, optional initial/final states, all stores and
casts, and both TF32-disabled dots. K/D share storage while V and the two
optional states have independent fp32/fp16/bf16 choices. `Proofs/FP.lean`
proves full source equivalence by commuting only the outer fp32 carry update;
its only printed atom is `add_commute`. The finite domain follows the actual
history store and inner loop in every reached outer iteration.

The source locator now also represents static natural bounds by constant
operations. `equiv_decompose` retains the source's static range constructor,
and a new contextual theorem lifts the body proof using the original reached
states. Static and dynamic nested-loop regressions pass, including refusals
for missing or misplaced guards, changed bounds and absent admissions.
Typed-source checks preserve residual promotion, both dot-input casts, TF32
settings, history-store order and corrected-value output masks.

The historical real proof needed compatibility repairs for the dot options
parameter and the source's `allow_tf32=False` setting. It compiles again;
its single-inner-block, no-time/key-tail scope is unchanged. `RealSource.lean`
checks erasure of all typed storage combinations against that full historical
surface. The abstract execution witness checks two outer and two inner
iterations, all three tails, nonzero head/value IDs, both state switches
enabled, both full programs and all three outputs with neighboring cells.
It witnesses satisfiable FP premises and does not model GPU matrix products.

`TritonBenchSpecExamples` builds successfully (3,571 targets). The source gate,
coverage review, proof-gap manifest, specification sheets and four targeted
Lean regressions pass. `TritonBenchGInProgress` also builds successfully (3,503
targets). Independent comparator/trust replay accepts **12 files**, **108 named
proof targets**, **1,224 kernel theorem targets**, and **1,136 matching snapshot
inputs**; receipt `Logs/comparator-check-_evnind7`. This case remains pending: general real
recurrence, multi-inner-block/tail reasoning, full flat IO and tagged-cell frame
are not yet complete. Progress stays **27 complete, 157 pending**.


## Chunk delta general inner-loop real proof (2026-10-09)

The new `chunk_delta_fwd/Proofs/RealStep.lean` executes the complete inner body
with arbitrary inner/outer indices, frozen state and incoming accumulator.
`RealInnerLoop.lean` composes every `ceil(BT/BC)` iteration, derives the corrected
V rows and accumulated real sum from input arrays, preserves earlier outputs
and inputs, and frames complete memory cells outside the current chunk's
active write span. `RealInnerSafety.lean` checks the actual two-axis load/store
bounds; no output-correctness claim is an input premise.

`RealValues.lean` relates the padded source correction to the independent
K-term matrix formula under `i_k=0` and `K≤BK`. With `BC>0` and `BC∣BT`,
the completed sum is one BT-row update, including a partial trailing time
chunk. The unrestricted inner-loop execution keeps the original rounded-up
span; it does not claim BC-divisibility follows from every host invocation.

The new `ChunkDeltaInnerReal` regression executes two inner blocks at outer
index 1 with K/V/time tails, derives the concrete final accumulator 71 from
incoming 5 and three contributions of 22, and checks that an untouched output
cell preserves its non-real memory tag.

This completes the inner-loop proof, not the full case. Outer-state induction,
optional seed/final paths and the complete three-output flat IO remain. The
case stays pending, with **27 complete and 157 pending** overall.

Validation passes: `TritonBenchGInProgress` (3,509 targets), the concrete
real-execution regression, coverage review (380 declarations / 201 public
files) and the proof-gap manifest. Independent comparator/trust replay accepts
all **7 new files**, **31 named proof targets**, **125 kernel theorem targets**
and **1,143 matching snapshot inputs**, receipt
`Logs/comparator-check-b216an0s`. No FP admission or shared tactic changed in
this batch.


## Chunk delta complete original real execution (2026-10-09)

`Proofs/RealExecution.lean` connects the actual typed source's real projection
to independent matrix formulas for all three outputs. It composes optional
initialization, all outer and inner iterations, and optional final-state
writeback. Both switches remain symbolic, all storage choices remain covered,
and the conclusion includes complete tagged-cell frame.

The recurrence in `RealWindow.lean` separates stride BT from window width
`ceil(BT/BC)*BC`. `RealOuterLoop.lean` preserves rows whose final writer has
run, while permitting later windows to overwrite rows reached early. Thus the
proof covers nondivisible BC as well as the usual disjoint geometry; at width
BT it reduces to `RealMath.state`. K/V/time tails and padded K-to-K-term-sum
correspondence remain covered. Positive BT/BC, full key ownership, sufficient
outer iterations and active-address noncollision are explicit hypotheses.

The complete-source regression exercises all four optional-state switch
pairs, overlapping update windows and all three tails. It checks history,
corrected V, final state and untouched non-real memory cells. A `refutation`
also proves that a disjoint three-row formula is wrong for that source
geometry: its H=103 differs from the source's H=137. This conclusion is about
real source semantics, not GPU compilation or FP admission.

Complete source bounds, the final flat IO contract, the candidate real proof
and the three-entry public layout remain. This case is still pending; overall
progress was **27 complete and 157 pending** before the padding review above
reopened three cases; the current count is **24 complete and 160 pending**.


The outer execution batch independently replayed all eight selected files,
38 named proof targets and 128 kernel theorem targets, with 1,151 matching
snapshot inputs (`Logs/comparator-check-z40y5zb0`). The in-progress library
build (3,516 targets), complete execution regression, coverage review and
proof-gap manifest pass. This receipt validates the existing zero-padding
model; the subsequently discovered [padding mismatch](block_padding_review.md)
prevents interpreting it as acceptance of the source's unspecified tails.
The source-snapshot freeze for this batch has ended.

## Block-padding semantic foundation (2026-10-09)

The AST, precision projection, memory relocation and real/FP evaluators now
retain zero versus unspecified block-load padding. A shared load-level lemma
uses an explicit zero-oracle premise; locality preserves arbitrary agreeing
oracles. The DSL switch and source-proof reconnection are the next step, so
the current acceptance count remains **24 complete and 160 pending**.

The joint in-progress/example build passes 3,650 targets; both block-pointer
regression files and 17 Python source-audit tests pass. Independent replay
accepts 18 files, 544 named proof targets and 6,045 kernel theorem targets,
with all 1,153 current snapshot inputs matching
(`Logs/comparator-check-_a0onijy`). Coverage metadata explicitly qualifies the
three reopened cases as zero-padding-model results. See
[the padding review](block_padding_review.md) for the source inventory and
the distinction between a conditional model result and a source guarantee.

## Padding frontend and vector-cumsum source connection (2026-10-09)

Omitted/empty block-load padding now selects unspecified values in both the
algorithm and fp32 compute AST. Explicit zero remains zero; ordinary pointer
loads retain their previous behavior. Unsupported padding is rejected.
The new source regression checks the lowered precision and padding policy,
actual tail execution and unchecked memory reads.

The complete `chunk_cumsum_vector` source is reconnected. Its real execution
and safety proofs now receive the existing zero-oracle IO premise explicitly;
its FP proof preserves arbitrary unspecified values and still uses only
`add_commute`. The public real contract is unchanged. Removing its zero-oracle
premise is not yet proved, so acceptance remains **24 complete / 160 pending**.

The targeted vector correct/equivalence/execution-witness build plus
`TritonBenchSpecExamples` passes 3,593 targets. Four source/memory regression
files, coverage review (380 declarations / 201 files), and the proof-gap
manifest pass. The in-progress library as a whole remains incomplete:
`chunk_cumsum_kernel`, `chunk_bwd_dqkg`, and `chunk_delta_fwd` still have old
zero-padding handwritten ASTs whose source bridges must be migrated.
Independent comparator/trust replay accepts all 18 selected files, 85 named
proof targets and 930 kernel theorem targets. All 1,155 current snapshot
inputs match (`Logs/comparator-check-t6_m7rbs`). The source-snapshot freeze
for this batch has ended.

## Scalar-cumsum padding repair and acceptance (2026-10-09)

The complete scalar original and sum-left candidate now preserve unspecified
block loads. Their loop carry sums actual loaded values, including the tail;
every valid output prefix is proved independent of arbitrary real tail values.
`RealPadding.source_outputs` states this for both full typed sources after
precision erasure. The flat IO specifications retain their shared adapter's
zero-oracle premise, active-address safety and full tagged-cell frame. The FP
proof still uses only `add_commute` on the actual reached operands.

The targeted build passes 3,452 targets. Both source/padding regressions pass,
including full three-chunk executions with arbitrary padding and the concrete
value seven. Coverage reviews (380 declarations / 201 files) and the proof-gap
manifest pass. Independent comparator/trust replay accepts 13 files, 93 named proof targets and 668 kernel theorem targets;
all 1,157 current snapshot inputs match (`Logs/comparator-check-xj59vm4q`).

Scalar cumsum is reinstated: **25 complete / 159 pending**. The dqkg and chunk
delta source bridges remain unrepaired; vector cumsum still has a conditional
real proof pending its oracle-independence extension. No whole-corpus build
success is claimed. `attention_kernel_aligned` remains user-deferred.

## Vector-cumsum padding independence and acceptance (2026-10-09)

The complete original and sum-left candidate now have region proofs of
successful execution, independent column prefixes and full tagged-cell frame
for arbitrary unspecified real values on both axes. The loop invariant tracks
the actual loaded carry; valid prefixes only contain valid earlier rows in
the observed column. Symbolic dimensions and strides, all chunks and both
tails remain covered. The shared flat IO adapter retains its zero-oracle
condition, and both public real specifications are unchanged. FP equivalence
still uses only `add_commute` on the actual reached operands.

The targeted build passes 3,458 targets; source, layout and complete padding
regressions pass. The latter checks both full source executions with arbitrary
padding including seven, nonzero feature/head IDs and untouched neighboring
cells. Coverage reviews (380 declarations / 201 files) and the proof-gap
manifest pass.
Independent comparator/trust replay accepts 20 files, 96 named proof targets and 927 kernel theorem targets;
all 1,159 snapshot inputs match the checked worktree (`Logs/comparator-check-gcqf5awz`).

Vector cumsum is reinstated: **26 complete / 158 pending**. The dqkg and chunk
delta source bridges remain open; no whole-corpus build success is claimed.
`attention_kernel_aligned` remains the only user-deferred case.

## Dqkg padding repair and conditional acceptance (2026-10-09)

All seven handwritten block loads now select `.unspecified`, matching the
full typed source. Zero-oracle premises are propagated through setup, every
value-loop iteration, postlude, bounds and flat IO; the public real contracts
keep their existing shape/stride/allocation conditions and zero-value premise.
Both independent whole-V formulas remain unchanged. The full FP equivalence
retains unspecified values and still uses only admitted `add_commute`.

`UndefinedTailLoop` executes the actual V=1 value block with 16×16 dot tiles
and unspecified V/H/DO/DH lanes equal to one. `UndefinedTail.source_stores_fifteen`
connects setup, the complete loop and all three stores to the original typed
source after real erasure. Actual inputs are all zero, yet dq[0,0]=15;
`mathematical_value` is zero. The `refutation` declaration
`dq_without_zero_padding` checks the resulting contradiction. A synthetic
16-value array only describes the loaded tiles in the proof; it never changes
the source V=1 or the actual initial input arrays. This is a real-model result,
not a GPU/compiler observation. The separate history-allocation counterexample
is retained.

The final targeted build passes 3,469 jobs, including real correctness, FP
equivalence, typed execution, the host-bounds refutation and the padding
counterexample. Both source/precision and active-address regressions pass.
Coverage review (380 declarations / 201 files) and the proof-gap manifest pass.
Independent comparator/trust replay accepts **24 files**, **150 named
proof targets** and **1,124 kernel theorem targets**; all **1,161 snapshot
input hashes** match the workspace. Receipt: `Logs/comparator-check-hrl4w00r`.

Dqkg is reinstated within this conditional scope: **27 complete / 157 pending**.
`attention_kernel_aligned` remains user-deferred; the BGMV fp16 admission choice
and the remaining corpus proofs are still pending. No full-corpus success is
claimed. The next source-padding repair is `chunk_delta_fwd`.


## Chunk delta: complete source contracts and padding refutation (2026-10-09)

`chunk_delta_fwd` now has the standard Kernels/Correct/FPEquiv entries. Both
complete typed sources implement the independent real recurrence, including
all inner and outer iterations, all four optional-state flag pairs, history,
corrected V and final state. Successful flat execution, actual active-address
bounds and preservation of every other full tagged cell are proved. The window
width is ceil(BT/BC)*BC; BC need not divide BT, and corrected row t is finalized
by outer chunk t/BT. The full key axis remains owned by pid0=0 with K≤BK.

All four omitted-padding loads now retain `.unspecified`; zero-oracle premises
are explicit through real execution and the public IO contract. A complete
original-source witness at T=1, K=V=BT=BC=BK=BV=16, NT=1, no seed and enabled
final storage starts from all-zero actual memory. One-valued K/V tail lanes
produce final H[0,0]=15, whereas the independent formula is zero. The
`refutation final_state_without_zero_padding` command rejects removing the
premise. This is a real-model witness, not a GPU measurement. FP equivalence
still uses only add_commute on the actual finite outer carry operands.

The final targeted build passes 3,474 jobs. Source/precision, inner-loop and
complete real-execution regressions pass. The complete test checks both
versions, every flag pair, overlapping writes, typed-cell frame, sufficient
bounds and rejection of a history extent missing its last active address.
The 173-port source gates, coverage review (380 declarations / 202 files),
proof-gap manifest and generated sheets pass; the source-checker suite passes
18 tests, including a quotation-category normalization that preserves dtype
mismatches.

Independent comparator/trust replay accepts **31 files**, **179 named proof
targets** and **1,271 kernel theorem targets**. All **1,171 snapshot input
hashes** match the current workspace. Receipt: `Logs/comparator-check-4rbqaiz0`.

Current acceptance is 28 complete / 156 pending. Only
`attention_kernel_aligned` is user-deferred. BGMV's missing fp16 add_commute
decision is still pending; no unfiltered whole-corpus build is claimed.
The next case is `chunk_gate_recurrence`.


## Chunk gate recurrence: FP proofs and executable paths (2026-10-09)

Both complete typed originals and sum-left candidates are staged under
`chunk_gate_recurrence/Proofs/`. Forward and backward independently preserve
fp32/fp16/bf16 source, gate and seed/incoming storage; element-type output casts
are resolved explicitly. Both full-program equivalence proofs use only
`add_commute` on the actual fp32 product and promoted source/incoming operand.
Backward covers its loop and final DL update, with guards on original prefixes.
The optional `equiv_decompose (reverse := true)` order composes the final
rewrite first; existing default traversal remains unchanged.

The shared FP interpreter now executes natural pointer subtraction instead of
returning failure for every decrement. Regression checks cover reverse strides,
truncation and missing operands. Mixed-storage complete-source witnesses run
three chunks with 2×2 tiles and all program IDs nonzero, both forward seed
flags, both candidates and untouched neighboring cells. A one-chunk backward
witness writes DL from DS[0] and leaves DI/DG untouched. Their abstract reduction
returns one; these check execution and memory flow, not the real DG formula or
GPU behavior. The last DI/DG slices remain unchanged by the JIT; Python's
subsequent zeroing and split-tile DG reduction stay outside the kernel claim.

Independent real forward/product/sum, reverse suffix and whole-tile DG formulas
are defined, with step and arbitrary-count fold identities. Both typed
originals project to the historical complete real source. Historical store/step
proofs still assume materialized carries: general whole-source execution,
address safety and flat IO/frame proofs for both versions remain open. No
public completion entries have been manufactured from those assumptions.

The execution build passes 3,382 jobs. Source/cast and reverse-pointer checks,
six related regression files, coverage review (380 declarations / 202 source
pairs), the proof-gap manifest and 173-port source gates pass. The new typed
sources have explicit Lean precision regressions; they are not yet substituted
for the legacy sources selected by those 173-port gates. The shared dependency
build passes 3,703 jobs, including VeriTile, all bench/examples and the FP entries
of the 28 currently accepted corpus cases.

Independent comparator/trust replay accepts **12 files**, **113 named proof
targets** and **1,444 kernel theorem targets**. All **1,179 snapshot input
hashes** match the checked workspace. Receipt: `Logs/comparator-check-3ajzvpvc`.

This is a checked progress milestone, not case acceptance. The count remains
**28 complete / 156 pending**. Next is the complete real loop/IO proof for
`chunk_gate_recurrence`; `attention_kernel_aligned` remains user-deferred and
the BGMV fp16 admission decision remains pending.


## Chunk gate recurrence: complete forward real execution (2026-10-09)

The complete original forward source and its sum-left candidate now have a
real execution proof for symbolic dimensions and every supported storage
selection. The proof derives the initial state and loop carry from ordinary
input/optional seed memory facts, connects every output chunk to the independent
`RealMath.forward` formula and preserves all tagged memory cells outside the
write footprint. It does not assume a materialized carry or an FP atom.

The flat-memory theorem in `chunk_gate_recurrence/Proofs/RealForwardFlat.lean`
covers both versions. It requires disjoint allocation, valid actual accesses,
N > 0, BK <= K, BV <= V and source/gate regions separate from the output.
The optional seed can alias the output because it is read before the first
store. The internal flattening bridge adds no zero-padding-oracle premise.
`RealForwardBounds.sufficient` derives the access conditions from whole-block
fit and allocation lengths; the last source/gate chunk is not loaded, and
N = 1 needs neither source nor gate allocation.

The targeted forward build passes 3,428 jobs. The regression in
`bench/tests/ChunkGateForwardReal.lean` checks both versions and seed flags,
three chunks, nonzero key/value program IDs, mixed storage, actual seed/output
aliasing and an untouched tagged integer cell in both region and flat memory.
It also checks the one-chunk allocation boundary and rejects an output missing
its last active address. Coverage review (380 declarations / 202 source pairs)
and the proof-gap manifest pass.

Independent comparator/trust replay accepts **11 files**, **62 named proof
targets** and **190 kernel theorem targets**. All **1,190 snapshot input
hashes** match the current workspace. Receipt: `Logs/comparator-check-ev_okbco`.

This is a checked forward milestone. The complete backward real execution,
address safety and flat IO proof for `DI`, `DG` and `DL` are still open; the
historical materialized-carry lemmas do not discharge them. Acceptance remains
**28 complete / 156 pending**. `attention_kernel_aligned` remains user-deferred;
the BGMV fp16 admission decision is unchanged. No unfiltered whole-corpus build
or new GPU experiment is claimed.


## Chunk gate recurrence: complete backward region execution (2026-10-09)

The original backward source now executes from ordinary input-memory facts
and its actual zero initialization. `Proofs/RealBackward.lean` composes the
prologue, all N-1 reverse iterations and the final DL postlude. DI stores the
independent reverse suffix beginning at t+1, DG stores its full BK×BV
contraction with S, and DL stores suffix zero. Every tagged cell outside the
write footprint is preserved. `RealBackwardCandidate.region_exec_equal`
transfers this result to the complete sum-left candidate using real algebra,
without FP atoms or materialized-carry premises. The proof covers symbolic
positive N, positive block sizes fitting K/V and all storage selections.

DI/DG must be separate from one another and from S/D/DS. DL must be separate
from DI/DG but may alias an input because its store follows the last load.
The source proof needs S values only for t < N-1. The original source's
actual-access safety is proved in `RealBackwardSafety`; `RealBackwardBounds`
derives it from whole-block fit and allocation lengths. N=1 requires no
S/DI/DG allocation, while DS/D reads and the DL store remain necessary.
Absolute initial addresses minus executed strides model all reverse pointer
updates, including the final unused decrements. They are not replaced by
chunk-zero addresses.

The targeted build passes 3,436 jobs. `bench/tests/ChunkGateBackwardReal.lean`
passes both complete versions on three chunks with mixed storage and a
nonuniform 2×2 source tile. It checks DI values 8/3, full DG contractions 64/36,
DL=17 while aliasing S, untouched last DI/DG chunks and a neighboring tagged
integer cell. It also checks complete one-chunk execution, missing live output
cells, minimal live allocations and the distinction between dead pointer
values and chunk-zero pointers. Coverage review (380 declarations / 202 source
pairs), the proof-gap manifest and scoped whitespace checks pass.

Independent comparator/trust replay accepts **11 files**, **75 named proof
targets** and **192 kernel theorem targets**. All **1,201 snapshot input
hashes** match the current workspace. Receipt: `Logs/comparator-check-2b67mhu_`.

Backward flat IO remains open: the bridge must preserve the registers used by
the DL postlude while tolerating unused S/DI/DG pointer differences after the
last iteration. `flat_exec_equal` currently compares the two flat source
versions; it does not establish their independent output formula. Public
entry migration and final case acceptance follow that proof. Acceptance stays
**28 complete / 156 pending**; the user-deferred attention case and pending
BGMV fp16 admission decision are unchanged.


## Chunk gate recurrence: complete backward flat execution (2026-10-09)

`Proofs/RealBackwardFlat.source_executes` now proves the complete original and
sum-left backward sources in flat memory against the independent DI/DG/DL
formulas. It reuses the established region result only after pairing the
actual flat prologue, every reverse iteration and the actual DL postlude with
the region execution. All tagged memory cells outside the translated output
footprint are preserved. Dimensions and storage selections remain symbolic;
the theorem adds disjoint allocation and bounds for actual accesses.

The reverse loop's load/update/store core commutes strictly with relocation.
Before the last iteration its pointer tail does too. At loop exit the bridge
ignores only the now-unused S/DI/DG register values, retaining the full memory,
DS/D, accumulator and position registers required by the DL postlude. No
head-index restriction, dead-pointer no-underflow premise or external zero
padding-oracle condition is imposed. DL may still alias input memory.

The targeted build passes 3,441 jobs. `bench/tests/ChunkGateBackwardFlat.lean`
checks both complete versions in concrete disjoint allocations with nonzero
bases, three chunks, head zero, mixed storage and a nonuniform 2×2 tile.
It checks DI=8/3, DG=64/36, DL=17 with DL aliasing S, unchanged last DI/DG
slices and a neighboring tagged integer cell. A one-chunk fixture allocates
no DI/DG memory, writes DL=1 and preserves every other flat cell. A separate
proved inequality witnesses final dead-pointer divergence after relocation.
Coverage review (380 declarations / 202 source pairs), the proof-gap manifest
and scoped whitespace checks pass.

Independent comparator/trust replay accepts **7 files**, **54 named proof
targets** and **105 kernel theorem targets**. All **1,208 snapshot input
hashes** match the current workspace. Receipt: `Logs/comparator-check-izcc8f8s`.

This completes the checked backward flat-IO milestone. Public contract/entry
migration and final case acceptance remain; the count stays **28 complete /
156 pending**. The user-deferred attention case and pending BGMV fp16
admission are unchanged. No unfiltered whole-corpus build or new GPU experiment
is claimed.


## Chunk gate recurrence: public contracts and case acceptance (2026-10-09)

The directory now exposes shared `Kernels.lean`, four real specifications in
`Correct.lean` and two FP equivalences in `FPEquiv.lean`. The historical
materialized-carry slices live in `Proofs/LegacyReal.lean`; current correctness
uses the complete initialization, actual register loops and final postlude.
The helper layer never imports the public entries, and real/FP proof layers
remain independent. Public IO definitions show initial-array reads, output
windows, bounds and aliasing conditions; the mathematical formulas are stated
independently at the entry.

Forward covers both the seeded and zero-seeded recurrence. Backward derives
DI from the reverse suffix, DG from the complete BK×BV contraction, and DL
from suffix zero. Both candidates meet the same formulas. The four contracts
conclude successful execution in arbitrary disjoint flat allocations and
preservation of every other tagged cell. Dimensions remain symbolic under
necessary tile-fit and positive-size conditions. Seed/output aliasing is
permitted; backward DL may alias inputs after the last load. Neither a supplied
carry, an FP atom nor an unspecified-value premise is required by real
correctness. Final unused reverse-pointer values do not impose head-index or
underflow conditions. Host last-slice clearing and DG aggregation are outside
these per-program kernel contracts.

Both FP entries print only `add_commute`. Domain checks refer to actual fp32
operands after storage conversion at every reached addition, including the
backward DL update after the complete original loop. Source, gate and
seed/incoming fp32/fp16/bf16 storage, products, reductions, casts and all memory
effects are preserved.

All 36 case modules build (3,481 jobs). Five focused Lean regressions pass:
`ChunkGateForwardReal`, `ChunkGateBackwardReal`, `ChunkGateBackwardFlat`,
`FPChunkGateSource` and `ChunkGatePublic`. The new public regression applies all
four specifications directly at N=1, including optional seed/output aliasing,
absent DI/DG allocations and a nonzero initial padding oracle. Existing tests
cover multiple chunks, nonuniform tiles, mixed storage, DG contraction,
unchanged final slices, tagged cells and dead reverse-pointer divergence.
All 16 layout/coverage unit tests and corpus source-only audit gates pass.
Coverage review and generated specification sheets agree on 383 declarations
across 203 public files: 81 full-value candidates, 268 specializations, 29
prepared-input slices, three pre-rounding slices and two blocked summaries.
The six current public statements are registered in the kernel manifest.

Independent comparator/trust replay accepts **41 files**, **284 named proof targets** and **1521 kernel theorem targets**.
All **1,212 snapshot input hashes** match the current checked source; every named
target in the 36 case modules and five regressions is present in the exported
inventory. Receipt: `Logs/comparator-check-tou10lkr`. The old generated sheet was removed
when the two current specification sheets replaced it.

Current acceptance is **29 complete / 155 pending**. The next case is
`chunk_gated_attention`. `attention_kernel_aligned` remains user-deferred and
the BGMV fp16 admission decision remains pending. This targeted acceptance does
not claim an unfiltered whole-corpus build or new GPU experiments.

## `chunk_gated_attention`: cumulative and FP milestone (2026-10-10)

This case remains pending; the accepted count is still **29 complete, 155
pending**. `attention_kernel_aligned` remains explicitly deferred. The existing
public inventory is unchanged: 383 specifications in 203 files; the newly
staged helpers do not upgrade the historical specialized/prepared H claims.

Complete typed cumulative and recurrent sources now live in
`chunk_gated_attention/Proofs/TypedSource.lean`. The cumulative candidate changes
only natural head/stride multiplication; the recurrent candidate commutes only
the reached fp32 carry/dot addition. Full-source FP equivalences print `none`
and `add_commute`, respectively. The latter retains actual post-gate finite
operand checks, all storage conversions, both gate branches, the optional seed
addition, pre-update history and optional final store. Intrinsic `tl.exp` is
unchanged and needs no exponential atom for this rewrite.

Both cumulative sources now have complete real region/flat execution proofs,
independent chunk-local prefix formulas, actual-access bounds and tagged-cell
frame. Source/output aliasing is allowed. Valid prefixes are independent of
arbitrary unspecified padding, with no zero-padding premise. A mixed-type,
partial-tile regression exercises nonzero program IDs, shared input/output and
a neighboring natural-valued cell. The H source has a complete precision-erasure
bridge and an independent closed form, but its full real loop/IO proof and the
zero-extension premise analysis are still open.

Typed execution witnesses cover both candidates, including a two-chunk H run
with fp16 dot operands, bf16 gate/history/seed and fp32 final storage, every
combination of gate/seed/final flags, and partial time/K/V tiles. They establish
successful execution and inhabited domains in explicit abstract algebras; they
are not hardware experiments or real-math correctness results.

The source regression exposed a DSL bug: in-place arithmetic bypassed the
ordinary assignment's load/cast storage inference. The expansion now resolves
the operand's actual pointer storage before its cast for `+=`, `-=`, `*=` and
`/=`. `LoadCastStorage.lean` checks all four forms, pointer rebinding and generic
storage. The source-level bf16 seed-addition assertion now passes. Exact integer
base congruence was also added for dynamic block pointers; a negative regression
rejects a changed pointer offset.

The targeted build completed 3467 jobs, including the unchanged historical
entry. Six relevant Lean regressions and 16 layout/coverage tests passed;
source-only corpus gates and proof-gap/coverage validation passed. No
unfiltered whole-corpus build or additional GPU experiment was performed.
Independent comparator/trust replay accepts **23 files**, **130 named proof
targets** and **1,553 kernel theorem targets**. All **1,226 snapshot input
hashes** match the current source. Named theorems were checked against each
accepted target inventory, including the historical entry and the shared DSL
and expression-congruence changes. Receipt: `Logs/comparator-check-4hctu04g`.
This certifies the staged milestone, not completion of the pending H real IO
proof or acceptance of the whole case.


## `chunk_gated_attention`: complete H real execution (2026-10-10)

The full H source now computes the independent closed form, for both candidate
orders, all gate/seed/final flags, every iteration, arbitrary symbolic dimensions,
strides and program IDs. `RealHProgram.source_body` checks the exact connection
to every typed-storage combination. `RealHExecution.executes` proves successful
region execution, pre-update history, optional final output and the full tagged
memory frame. It assumes explicit input/history separation and noncolliding
active output addresses; seed/output aliasing is permitted. Finite-allocation
bounds and flat-memory transport are still pending, so this is **not** an
acceptance of the entire case. The count stays **29 complete / 155 pending**.

The pointer, load, gate and complete-step lemmas preserve arbitrary unspecified
values. The independent zero-extended formula uses an explicit zero-oracle
premise. `UndefinedTail.final_state_without_zero_padding` refutes the claim
without that premise using both complete sources: `T=1`, `K=V=BT=BK=BV=16`,
zero valid inputs, zero gate and unspecified K/V tails of one produce final
output 15 rather than zero. The complete counterexample runs both gate branches.
It is a permitted real-model execution, not an observed GPU padding value.

`bench/tests/GatedAttentionReal.lean` passes for both sources and all gate/seed/
final switches, with `T=17`, `K=V=19`, `BT=BK=BV=16`, `NT=2` and every program ID
set to one. Nonzero gate inputs distinguish full first-chunk endpoints from
the checked-off last-chunk endpoint: the first update uses `exp(2)`, while the
last valid contribution uses `2*exp(-2)`. It checks both pre-update history
outputs, optional final output and preservation of an adjacent natural-valued
cell. The targeted module build completes 3428 jobs; the new regression and
16 layout/coverage tests pass. Source-only corpus gates, coverage validation
and proof-gap validation pass; the public inventory remains 383 specifications
across 203 source pairs.

Independent comparator/trust replay accepts all **14 new files**, covering
**60 named proof targets** and **223 kernel theorem targets**. Each named theorem
was found in its accepted target inventory; all **1,240 snapshot input hashes**
match the current source. Receipt: `Logs/comparator-check-iuzdhuh1`. This receipt
certifies this execution milestone, including the complete-source regression
and padding refutation, without upgrading the historical public claims.

Remaining acceptance work is H finite-allocation safety/flat transport, public
`Kernels`/`Correct`/`FPEquiv` migration, historical-entry/import metadata updates,
and final coverage/source/proof validation. The separate FP results continue
to use no atom for the cumulative candidate and only `add_commute` for H.
`attention_kernel_aligned` remains deferred at the user's request.

## `chunk_gated_attention`: public contracts and final acceptance (2026-10-10)

The case now exposes the standard three entry files. `Kernels.lean` contains
both complete source/candidate pairs; `Correct.lean` has four `Spec.Real`
contracts; `FPEquiv.lean` has the two whole-program equivalences and prints
`none` / `add_commute`. The old slice and prepared-input declarations moved
to `Proofs/LegacyReal.lean` and are excluded from the public inventory.

`RealHAccessSafety`, `RealHGateSafety`, `RealHStateSafety`, `RealHLoopSafety`
and `RealHSafety` prove bounds for the complete source. Only checked-on and
enabled accesses require allocation. The gate endpoint remains the exact
one-dimensional T*K or T*V parent, including full chunk-end addressing and
source strides. `RealHFlat` and `RealHIO` prove successful execution in finite
disjoint allocations, the independent history/final formula and preservation
of every other tagged cell. Neither success nor expected outputs are premises.

The real H contract retains its necessary zero-unspecified-value premise;
`UndefinedTail` retains the complete-source counterexample for both gate
branches and both candidate orders. The cumulative public IO uses the shared
adapter's zero-oracle premise, while its lower-level valid region/flat prefix
results admit arbitrary padding and source/output aliasing. H's seed may alias
outputs, but history is separated from K/V/gate and enabled final storage;
active output addresses cannot collide. Dimensions and strides remain symbolic.

The finite-allocation regression instantiates both public H specifications at
T=17, K=V=19, BT=BK=BV=16, NT=2 and all program IDs equal to one, across all
gate/seed/final switches and candidate orders. Six nonoverlapping allocations
cover the exact accesses; explicit outputs distinguish the exp(2) first-chunk
gate from the checked-off last endpoint. Disabled final stores and a natural
cell beside the active history window are preserved. The three Lean regressions
(`GatedAttentionReal`, `GatedCumulativeReal`, `FPChunkGatedSource`) pass.

All 36 case modules build successfully (3491 jobs). Source-only corpus gates,
16 layout/coverage tests, coverage fingerprints and the proof-gap manifest pass.
The generated public inventory is now **386 specifications in 204 files**:
87 full-value candidates, 267 specializations, 27 prepared-input slices,
3 pre-rounding slices and 2 blocked summaries. The two current sheets replace
the historical single-entry sheet. The pointer macro keeps its region explicit,
so dtype-attribute source comparison resolves the actual output storage without
adding an audit exemption.

Independent comparator/trust replay accepts **39 files**,
**185 named proof targets** and
**1963 kernel theorem targets**. Every named target
appears in its accepted inventory, and all **1250
snapshot input hashes** match the current source. Receipt: `Logs/comparator-check-9f4lhb14`.
These are per-program conditional real/FP results, not a new GPU experiment,
whole-host launch proof or unfiltered corpus acceptance.

Current acceptance is **30 complete / 154 pending**. `attention_kernel_aligned`
remains user-deferred; the separate BGMV fp16 admission decision is unchanged.
The next case is `chunk_gla_fwd`: its old output-only source/body bridge fails
under current padding semantics, and its headline lacks an allocation/frame
contract and FP equivalence. The four A-producer kernels are outside that old
proof; their coverage must not be inferred from the consumer theorem.


## chunk_gla_fwd output consumer — pending case

The new `Proofs/TypedOutput.lean` retains the complete output JIT source and a
final-addition-commuted candidate, including Q's scale/gate casts, history and
A conversions, default inter dot versus TF32-disabled final dot, `tl.exp`,
unspecified padding and all masks. Q/V/G independently select fp32/fp16/bf16;
H/O remain explicit dtype parameters and A matches the host's fp32 allocation.

Both erased versions now have an independent initial-memory gated contraction,
full K-loop execution, actual active-address bounds, flat-memory transport and
a complete tagged-cell frame. Zero unspecified values, distinct output
addresses and disjoint finite allocation windows are explicit premises.
`FPOutput.output_equiv` uses only `add_commute` at actual reached fp32 operands.
The typed execution witness has two K iterations, mixed storage and partial
tails; the real finite-allocation regression computes 4 and preserves an
adjacent non-real cell. Both original and candidate execute in these tests.

All eight new proof modules and both targeted regressions compile. Independent
comparator/trust replay accepts all **10 files**, **76 named proof targets**
and **1295 kernel theorem targets**. Every named target appears in the accepted
inventory, and all **1260 snapshot input hashes** match the current source.
Receipt: `Logs/comparator-check-pvn1mzv8`. This certifies the consumer milestone;
no whole-case acceptance is claimed. The old single-file proof remains historical and fails
its source/body bridge with current omitted-padding semantics. Four A-producing
JIT kernels, public entry migration, padding counterexample analysis and final
case-level source/coverage checks remain. The host's BT-sized split temporary
and H allocation without a chunk axis also require separate bounds analysis.
Current acceptance stays **30 complete / 154 pending**.


## chunk_gla_fwd merge producer — pending case

The complete typed merge source now preserves its early return, runtime head
count, NK loop, fp32 temporary loads, unspecified padding and explicit output
cast. Its candidate commutes only the carry/load addition. Both erased
versions have complete execution proofs, an independent initial-array sum,
actual active-access bounds and full tagged-cell frame preservation. The
region proof supports arbitrary padding and input/output aliasing; finite
flat transport keeps the shared adapter's explicit zero-oracle premise.

The FP proof uses only `add_commute` at actual reached operands inside the
branch and loop. An executable model witness and the real finite-allocation
regression exercise three K slices, head count 3, nonzero IDs, a partial row,
outputs 144/147, neighboring preservation, skipped branches and NK=0. The
regressions also reject a missing rewrite domain and establish that the
host's BT-sized temporary can fail the merge's T-strided input bound
(`4096` allocated cells versus an active read at `8192`). No host launch or
GPU behavior is inferred from this allocation-bound counterexample.

The consumer FP execution witness was corrected to tag its fp32 A input with
the `.real` carrier used by algorithm loads. This correction and the ten merge
modules plus two merge regressions pass current-source independent
comparator/trust replay: **13 files, 58 named proof targets and 1011 kernel theorem targets**. Every named target occurs in its
accepted inventory, and all **1272 snapshot input hashes** match the source.
Receipt: `Logs/comparator-check-9itwli90`.

The case remains pending: inter-subblock, intra-subblock and split producers,
source-padding counterexample analysis, public entry migration and final
case-level source/coverage checks remain. Acceptance stays **30 complete /
154 pending**. The user-deferred `attention_kernel_aligned` is unchanged.


## chunk_gla_fwd inter-subblock FP and pointer inference — pending case

The complete inter-subblock source and its carry/dot-commuted candidate now
have a kernel FP equivalence using only `add_commute`. Both early returns,
all K blocks, the separate K-strided gate reference, matrix strides, omitted
padding, intrinsic exp, default dot and output dtype are retained. The 27
query/key/gate storage combinations remain symbolic in shape and launch.
An executable abstract-model witness checks mixed storage, partial tails,
two iterations, both skipped branches, zero K, outputs/neighbor preservation
and distinct reference/matrix gate addresses 58/60/70.

This exposed a frontend gap: pointer element dtype was lost through layout
hints. `rootedPtrDType` now descends through `tl.multiple_of` and
`tl.max_contiguous`. Dedicated parser regressions check half/bfloat storage,
masked load plus widening and unchanged untyped fallback. No extra dtype
annotation or cast was added to the kernel source. Relevant existing kernel
entries using hints, the consumer/merge modules and dtype regressions rebuild
successfully.

Real storage projection and an independent gated contraction with a real
reference-cancellation identity are in place. The inter producer's complete
real execution, access bounds and frame are still missing. Intra-subblock,
split, padding counterexample analysis, public entry migration and final
case-level checks also remain.

A fresh independent audit covers the updated inference dependency, all 23
current case proof modules, five case regressions and three dtype regressions:
**32 files, 162 named targets and 2748 kernel theorem targets**, with every named target present in the accepted inventory
and **1279 current snapshot hashes** verified. Receipt:
`Logs/comparator-check-d0t25koo`. This rechecks the earlier consumer/merge milestones after
the inference change; it does not certify the unfinished real inter execution.
Acceptance remains **30 complete / 154 pending**.


## chunk_gla_fwd inter-subblock real execution — pending case

The inter-subblock producer now has a complete real execution proof for both
addition orders, connected by `RealInterProgram.source_body` to every typed
query/key/gate combination and the output storage cast. The independent formula
is the initial-array gated Q/K contraction in `RealInterMath.expected`. The flat
gate reference retains its actual K stride; its cancellation is established by
a real exponential identity. No such identity is used in FP equivalence.

`RealInterExecution.source_executes` covers initialization, both skipped
branches, every K block and the final store. The zero unspecified-value oracle
is explicit: omitted block-pointer padding has not been changed to zero in the
source. `RealInterSafety.Bounds` lists every checked matrix load, each masked
flat reference load, and the final active stores. The result preserves all
other tagged cells, permits input/output region aliasing, and needs no extra
output-address injectivity premise: that property is proved from the store's
boundary checks. Sizes, strides, launch coordinates and NC remain symbolic.
`RealInterFlat.source_executes` transports the complete result to finite,
disjoint flat allocations using those same access bounds.

The `ChunkGLAInterReal` regression checks both addition orders with a nonzero
chunk and head, two K blocks, partial rows/features, and timeStride=5 versus
K=3. The real outputs are 6*exp(10) and 6*exp(5); the tagged adjacent cell is
preserved. It also covers both skipped branches and K=0. In the finite flat
allocation, the outputs are at 292/293 and cell 294 retains its nat tag/value.
The dependency build and this regression pass. Independent current-source
replay accepts **13 files, 61 named targets and 396 kernel theorem targets**;
the target inventory and **1292 source hashes** were checked against the working
tree. Receipt: `Logs/comparator-check-8no9u3eb`. The prior consumer/merge/inter-FP
receipt `Logs/comparator-check-d0t25koo` also still matches all of its inputs.

The whole case remains pending. Intra-subblock and split producers, padding
counterexample analysis, the three public entry files and case-level coverage
checks still remain. The existing inter FP theorem continues to use only
`add_commute`; no numerical admission or GPU experiment was changed. Acceptance
stays **30 complete / 154 pending**. `attention_kernel_aligned` remains deferred.


## chunk_gla_fwd intra-subblock execution — pending case

The intra-subblock producer now has complete original and scale-first typed
sources for all 27 query/key/gate fp32/fp16/bf16 combinations. Output A remains
fp32, matching the host intermediate. The source preserves its early return,
`min(...)` column limit, unspecified matrix loads, zero-masked flat loads,
explicit fp32 casts, intrinsic exp, reduction, causal where, masked stores and
both K-strided pointer increments.

`FPIntra.intra_equiv` proves the complete candidate with only `mul_commute`.
`WhereRewrite.contextual_where_mul_scalar` preserves the condition, other arm,
failed evaluations and surrounding statements. The domain checks the actual
sum and scale at every reached column iteration, on every lane because where
evaluates both numeric arms. No new GPU admission is introduced. A consistent
abstract model witnesses three complete iterations, mixed storage, both causal
arms, masked tails, early return and the same candidate execution. Its exp/sum
return one; these witnesses are not hardware observations.

`RealIntraExecution.source_executes` connects both complete sources to the
independent K-feature gated contraction. It covers the full column loop and
preserves every other tagged memory cell. Matrix query/gate inputs use the
actual time stride; streamed key/gate inputs use K. The feature coverage
condition is K<=BK and columns<=BT prevents output address collisions. Key/gate
regions must differ from output to protect later streamed reads. Cached query
input may alias output. The region proof permits arbitrary unspecified matrix
padding: masked-out key lanes are explicitly zero, so their real product
vanishes. This is a real derivation, not a new FP atomic assumption.

`RealIntraSafety` checks all active matrix/flat loads and every in-loop store.
`RealIntraFlat` transports the result to finite disjoint allocations; the
shared flat adapter still requires a zero unspecified-value oracle. These
conditions are separate from the stronger arbitrary-padding region result.
`ChunkGLAIntraReal` checks nonzero chunk/block/head, three streamed columns,
timeStride=5 versus K=3, partial row/K tiles, both candidate orders, arbitrary
padding, early return and zero K. Real outputs at 220/221/229/238 are
6*exp(24), 0, 6*exp(26), 6*exp(28); cells223/244 retain their nat99 values.
The finite flat witness checks outputs580/581/598 and unchanged cell583.
The FP regression checks casts, store dtype, masks, pointer advances, source
return/min and that an empty numerical domain cannot admit the rewrite.

The dependency build and both regressions pass. Independent comparator/trust
replay accepts **19 files, 76 named targets and 806 kernel theorem targets**.
Every named target is present and **1311 source hashes** match the working tree.
Receipt: `Logs/comparator-check-c8_r1nih`. The earlier inter-real receipt
`Logs/comparator-check-8no9u3eb` and combined consumer/merge/inter-FP receipt
`Logs/comparator-check-d0t25koo` were also checked and still match their inputs.

The README now reflects the completed inter-real and intra milestones. The
whole case is still pending: split producer, padding counterexample analysis,
public entry migration and case-level coverage/source checks remain. Temporary
split FP source, executable witness and parser checks compile but are not yet
integrated or independently audited. Acceptance remains **30 complete / 154
pending**. The user-deferred `attention_kernel_aligned` is unchanged.

### chunk_gla_fwd split producer — complete real and FP milestone

The fifth source kernel is now integrated with its complete typed original
and scale-first candidate. It retains program-ID decoding with NC, runtime
head count, the early return and min-bounded column loop, unspecified cached
matrix loads, zero-masked flat loads, fp32 casts, per-column zero initialization,
explicit addition of the sum, intrinsic exp, causal where, masked stores and
both K-stride pointer advances. All 27 query/key/gate storage combinations
are covered, with fp32 temporary output. The FP proof uses only `mul_commute`
and checks the reached operands after the source zero/add steps; an executable
mixed-storage domain witness covers three columns and early return.

Thirteen real proof modules connect the complete source to an independent
partial-K contraction, prove every column and preserve all other tagged memory
cells. The region proof permits arbitrary unspecified padding and does not
assume K fits one tile. Column-range injectivity follows from min(BC,...).
Streamed key/gate arrays remain separate from output; cached Q may alias.
Actual input/output bounds and a finite flat-allocation adapter are proved;
the adapter retains its explicit zero-oracle condition. The source's T*BC
slice stride is preserved, so these results do not repair the host allocation.

The real regression uses nonzero feature/chunk/subblock/head IDs, distinct
matrix and flat strides, a partial row and K tile, both scaling orders,
arbitrary padding, early return, zero K and finite flat memory. Expected
values 6*exp(24), 6*exp(26), 6*exp(28), a zero causal arm and neighboring tagged
cells are checked. The FP regression checks precision, source structure and
refusal when the rewrite domain is omitted.

Module builds and both regressions pass. Independent comparator/trust replay
accepts **18 files, 83 named targets and 821 kernel theorem targets**. Every
named target is present and **1329 snapshot input hashes** match the current
source. Receipt: `Logs/comparator-check-of362_jt`. The previous intra,
inter-real and combined consumer/merge/inter-FP receipts were rechecked and
still match their source snapshots.

Acceptance remains **30 complete / 154 pending**. All five per-kernel proof
components are now present, but padding analysis, public entry migration and
case-level source/coverage checks remain. A temporary output-padding refutation
has compiled: with T=1, K=17, V=16 and tiles BT=16, BK=32, BV=16, the complete
source can store 33 while the independent zero-extended formula gives 18.
This draft includes actual access bounds and is not yet integrated or
independently audited; it is a real-semantics witness, not a GPU observation.
The user-deferred `attention_kernel_aligned` remains unchanged.

### chunk_gla_fwd output padding — checked complete-source refutation

`Proofs/UndefinedOutput.lean` now contains the integrated counterexample. It
executes the complete prefix, K loop, second dot and output store, checks the
actual active-access bounds, and independently evaluates the mathematical
formula. The stored value is 33 while the formula gives 18. Precision erasure
connects the witness to all supported query/value/gate storage selections,
arbitrary history/output dtypes and both original/candidate orders. The
positive declaration `refutation output_without_zero_fill` retains bounds and
zero launch IDs. It refutes an unconditional real formula, not FP admission;
no GPU measurement is claimed.

The module builds and its axiom audit passes. Independent comparator/trust
replay accepts the file's **13 named targets and 38 kernel theorem targets**;
all **1330 snapshot input hashes** match the current source. Receipt:
`Logs/comparator-check-3wsdswpe`.

Case acceptance remains pending (**30 complete / 154 pending**). Besides the
public entry migration and case-level coverage checks, the next source review
must restore literal early returns in `TypedInter` and `TypedMerge`. They
currently use complementary enabled branches: mathematically equivalent
control flow, but not the current line-for-line source transcription standard.
This was identified while preparing the public source consolidation. Current
proof receipts certify the current Lean implementations, and must be refreshed
for affected files after the source-form repair. No frozen snapshot is active.

### chunk_gla_fwd public entries and literal source control flow

All five sources and their arithmetic candidates now live in `Kernels.lean`,
in Python kernel order. Inter/merge use the original early returns; the merge
uses the original two-argument range and its real proof follows the resulting
dynamic-range AST. The output-only obsolete file is removed. `Correct.lean`
exposes ten real specifications, and `FPEquiv.lean` exposes five conditional
program equivalences. Each entry shows its IO/formula or reached-operand
domain and a short checked composition of the case-local proofs. Actual bounds,
zero unspecified-value conditions, aliasing restrictions and tagged-memory
frames remain explicit. Public FP output is only `add_commute` or `mul_commute`.

The real entry and all ten FP/real regressions compile after the migration.
The source-only audit passes, including the regenerated 400-headline coverage
manifest. Obsolete correctness/translation blocker records have been removed.
A separate source-pair inspection checks the call sequence and branch/loop/
return counts for all five macros against their corresponding Python JITs.

Final module rebuilding and independent comparator/trust replay of the current
snapshot remain. Earlier receipts above are historical after this consolidation
and must not be used to accept the new snapshot. Acceptance stays **30 complete /
154 pending**; `attention_kernel_aligned` remains deferred by the user.

The merge FP domain witness's monolithic simplifier reached its step limit
after restoring literal source control flow. It is now replaced by checked
initialization, three individual loop steps, loop composition and store.
Full execution, values 144/147, untouched neighbor 99, early return, empty loop
and candidate execution all pass. The complete public entries and every case
module apart from that separately checked witness also finish their rebuild.
The final incremental build and 82-file current-source replay are the remaining
acceptance checks. Source inputs are frozen for that replay.


### chunk_gla_fwd accepted after complete current-source replay

All 72 case modules and ten real/FP regressions pass the final build. The
82-file independent comparator/trust replay passed in
`Logs/comparator-check-2_3bwsqe`: all 414 named declarations were present among
4,870 kernel theorem targets, and all 1,333 recorded source/configuration
fingerprints matched at verification. The ten real contracts and five FP
equivalences refer to the complete shared sources. Actual bounds, explicit
zero unspecified values, aliasing conditions and full tagged-memory frames
remain in their stated scope; host allocation/composition is not certified.

Current acceptance is **31 complete / 153 pending**. The next case is
`chunk_gla_simple`; its full-source real and admitted-FP replacement, including
all K blocks, is ready for integrated build and independent replay.
`attention_kernel_aligned` remains deferred at the user's request.


### chunk_gla_simple accepted after integrated source and proof checks

The complete original and scale-first candidate now share `Kernels.lean`.
`Correct.original_correct` and `Correct.scaleFirst_correct` establish the
independent initial-array contraction across all K blocks; there is no `K=BK`
restriction. Both conclusions include successful finite flat-memory execution,
actual active-access bounds and preservation of every other tagged cell.
The contract explicitly chooses zero unspecified boundary values. The supplied
history is an input; surrounding host allocation and launch composition remain
outside the per-JIT claim.

`FPEquiv.scale_equiv` uses only `mul_commute` on the actual final fp32 total and
scale after the full common prefix. Compatible Q/K/H storage, independent V/G
storage, score-to-V cast, intrinsic `tl.exp`, TF32-disabled dots, checked tails,
output cast and store remain. A mixed-storage abstract execution with two K
iterations witnesses a nonempty domain. The separate real regression checks
both full flat executions produce 12 and preserve a neighboring integer-tagged
99 with partial time/key/value tiles and nonzero program IDs.

The integrated build passes (3,459 jobs), as does the real regression. Source
translation gates, 401 coverage reviews across 206 specification files and
`git diff --check` pass. Independent comparator/trust replay accepts all
13 files in `Logs/comparator-check-usy6q186`: 107 named proof declarations,
1,234 kernel theorem targets and all 1,344 source/configuration fingerprints
match. This batch includes the twelve new case/regression files and the
comment-only update to the preceding case's RealOutput helper.

Current acceptance is **32 complete / 152 pending**. The next case is
`chunk_linear_attn`, whose four complete typed drafts and four admitted FP
proofs pass; full real execution and memory contracts are still in progress.
`attention_kernel_aligned` remains deferred at the user's request.


### chunk_linear_attn accepted after complete four-kernel checks

All four original/candidate pairs share `Kernels.lean`. The eight public real
specifications cover forward state, forward output, backward state and all
three backward-gradient outputs. Formulas use initial arrays, every K/V/time
iteration and the stated seed/final branches. Each contract includes successful
finite flat-memory execution, bounds on actual active accesses and preservation
of every other tagged cell. Dimensions, strides, scales and program IDs remain
symbolic. The zero unspecified-value premise and necessary aliasing conditions
remain explicit; source padding was not changed to requested zero padding.

The four public FP equivalences use only `add_commute` (state and gradients)
or `mul_commute` (forward output), checking actual reached operands. Common dot
storage supports fp32/fp16/bf16 with the documented independently typed inputs
and outputs. TF32-disabled dots, casts, runtime head count, masks and all stores
are preserved. Complete abstract typed executions establish nonempty domains;
they are not GPU measurements.

The integrated build passes (3,469 jobs), as do all four concrete real
regressions. The tests include nonzero program IDs, partial 16-wide tiles,
multiple K/V/time iterations, all forward state flags and empty reverse time.
The gradient fixture produces dQ=1900, dK=912 and both dV blocks=96; adjacent
tagged cells are preserved. Source gates and the 411-headline scope inventory
across 207 specification files pass.

Independent comparator/trust replay accepts all 19 case/regression files in
`Logs/comparator-check-n641rovk`: 257 named proof declarations and
1,684 kernel theorem targets, with all 1,362 recorded source/configuration
fingerprints matching the checked snapshot. Host composition, allocation/grid
construction, backend tile legality, races between programs and performance
remain outside these per-JIT contracts.

Current acceptance is **33 complete / 151 pending**. Work continues with
`chunk_retention`; `attention_kernel_aligned` remains deferred at the user's
request.


### chunk_retention accepted after complete four-kernel checks

All four complete typed sources and candidates share `Kernels.lean`. Eight
real specifications expose independent initial-array decay formulas and full
finite flat-memory execution, actual-access bounds and preservation of every
other tagged cell. Dimensions and strides remain symbolic. Zero unspecified
values and necessary aliasing conditions are explicit. Signed tail and score
offsets, intrinsic exp2/log2 and TF32-disabled dots are retained.

The supplied backward-state source is intentionally represented as written:
square BT tiles, untransposed dO/V contraction, unused query/scale, a dead
history load, full-chunk decay only after the reverse loop and one final store
at its unusual address. The real contract requires NT>0 for the Python
post-loop-index use. It does not assert the intended mathematical gradient of
a different retention algorithm.

The four FP proofs use only add_commute (state and gradients) or mul_commute
(forward output), at the actual reached finite operands. Complete typed
executions under the named abstract model establish nonempty domains. The
3,470-job integrated build, all four finite-memory real regressions, source
gates and 421-headline coverage check pass. Regressions cover nonzero program
IDs, 16-wide tiles, partial time/key/value tails, multiple loop iterations,
seed/final branches and all gradient outputs.

Independent comparator/trust replay accepts all 20 case/regression files in
`Logs/comparator-check-v3__2clu`: 299 named declarations and 1,759 kernel theorem
targets, with all 1,381 recorded source/configuration fingerprints matching
the checked snapshot. Host composition, allocation/grid construction, backend
tile legality, cross-program races, concrete IEEE identities and performance
remain outside these per-JIT contracts.

Current acceptance is **34 complete / 150 pending**. `chunk_retention_ops` is
next; `attention_kernel_aligned` remains deferred at the user's request.


### chunk_retention_ops accepted after complete four-kernel checks

All four typed original/candidate pairs appear in `Kernels.lean`. The public
entries expose eight independent real formulas and four conditional FP
specifications. Three complete source-function pairs are checked equal to the
corresponding retention bodies, including signatures and both variants; their
real execution results are reused against this case's own source. The backward
state has a separate complete proof: reverse-time pre-update history stores,
scaled query and row-weighted incoming-gradient contraction, full-BT decay at
every update, independent BT/BK/BV and the empty NT=0 loop.

The real contracts include successful execution in finite flat allocations,
actual-access bounds, active-store injectivity and the full tagged-cell frame.
Zero unspecified values and necessary aliasing constraints remain explicit.
The backward state retains full-BT decay on a partial last chunk, as written
in Python. Forward state retains its different, shortened last-chunk decay.
The FP proofs use only add_commute or mul_commute and check actual reached
finite operands, with complete abstract typed executions establishing nonempty
domains. Intrinsic decay operations, all casts and TF32-disabled dots remain.

The 3,472-job integrated build, source gates and 431-headline scope inventory
pass. The new full-source flat regression checks three reverse time chunks,
16-wide tiles, nonzero program IDs, a partial tail and repeated carry decay.
It checks both candidates and adjacent tagged-memory preservation; NT=0 is
also checked as write-free. The shared real regressions are connected through
complete typed-source equalities. The spec-sheet index now follows both
retention directories' public Correct/FPEquiv entries instead of deleted files.

Independent comparator/trust replay accepts all 10 case/regression files in
`Logs/comparator-check-w_2hgm4d`: 90 named declarations and 485 kernel theorem
targets; all 1,390 snapshot input fingerprints match the checked source.
Host composition, grid/allocation construction, backend legality, cross-program
races, IEEE identities and performance remain outside these per-JIT statements.

Current acceptance is **35 complete / 149 pending**. `chunked_cumsum_fwd` is
next; `attention_kernel_aligned` remains deferred at the user's request.

### Broadcast multiplication support for chunked cumulative sum

The profiled rewrite layer now supports commuting two broadcast operands,
including a matrix and a per-row vector. Separate default-fp32 and explicit-fp32
paths retain the precision context. Both require finite actual operands after
the reached common prefix and use only the admitted `mul_commute` atom.
The regression preserves mixed-storage loads, casts, cumulative scan and store
around the changed multiplication; both source pairs print only `mul_commute`.

The three focused Lean tests and both retention FP entries compile. The full
Python tactic suite cannot finish its setup because the previously reopened
`bgmv_expand_slice` FP proof is still pending; this is not counted as a passing
suite. Independent comparator/trust replay accepts all three changed Lean
files in `Logs/comparator-check-n3r5xup8`: 26 named declarations, 289 kernel
theorem targets and 1,391 matching snapshot input fingerprints.

`chunked_cumsum_fwd` remains pending. Its complete FP source-pair draft compiles.
Both full region-memory real executions now connect the original inputs,
bias, softplus, finite clamp and both stores to independent formulas, with a
whole-cell memory frame. The concrete two-version regression gives prepared
values `[0,2,3,0]` and cumulative values `[0,4,10,10]`, checking the partial
sequence tail, masked head and adjacent tagged cells. A complete mixed-storage
abstract execution with both preprocessing flags witnesses inhabited FP
premises; its chosen scan and comparisons are not a GPU numerical model.

Active-address safety and the finite flat-memory IO adapter still need to be
connected. The Python wrapper's default upper bound `+infinity` also needs a
supported representation; current drafts explicitly model finite bounds only.
Public entry migration and independent replay of this case remain pending.
No completed-case count is increased by this support change.

### chunked_cumsum_fwd accepted with finite and default infinite limits

The three public entries now cover both complete typed source versions. The
candidate commutes only the broadcast dt/A product. Two independent initial-array
real formulas describe the prepared dt output and its scaled inclusive prefix
sum, including optional bias, cutoff softplus, lower/upper clamping, remasking,
both stores and the scan. Input dt/A storage is independent; the bias has
fp32/fp16/bf16 variants and both outputs are fp32. Sizes and strides are symbolic.

The real contracts include successful finite flat-memory execution, actual
active-access bounds, active-store injectivity and preservation of every other
tagged cell. They retain the valid host chunk-start condition and necessary
aliasing constraints: A differs from the first output, and the outputs differ.
No zero-unspecified-value premise is needed because every load uses other=0.
The default +infinity upper clamp stays a distinct opaque FP operation. The DSL
requires explicit fp32/fp64 conversion of lower-precision clamp operands; it
does not silently widen them or assume an FP identity.

The FP proof uses only mul_commute on actual reached finite operands after the
first store. Both finite and infinite branches have full abstract typed
executions under a consistent model; these are nonempty-domain witnesses,
not measured GPU values. Concrete real regressions use nonzero program IDs,
partial head/time tails and finite allocations. Prepared values are [0,2,3,0]
with a finite upper bound and [0,2,5,0] with +infinity; the corresponding sums
are [0,4,10,10] and [0,4,14,14]. Adjacent tagged cells are preserved.

The 3,461-job integrated build and subsequent 3,451-job dtype-guard rebuild
pass, together with focused execution, precision/broadcast, source and coverage
checks. The scope inventory has 432 headlines across 210 specification files.
The obsolete single-file spec sheet was removed.

Independent comparator/trust replay accepted all 35 case/test/shared-semantic
files in `Logs/comparator-check-pizbuz2z`: 398 named declarations,
6,012 kernel theorem targets and 1,413 matching input fingerprints.
After tightening the clamp dtype check, a separate final replay in
`Logs/comparator-check-hh4qulhn` accepted the 6 affected source/test/entry files:
16 named declarations, 238 kernel theorem targets and
1,413 matching final-snapshot fingerprints. The first receipt describes
the earlier snapshot; the second certifies the added type check and its users.

Current acceptance is **36 complete / 148 pending**. `context_attn_bloom` is
next. Its temporary full typed source pair and FP proof compile using a pending
signed-inference repair; the general real execution/flat IO proof is not yet
complete. `attention_kernel_aligned` remains user-deferred. The reopened
`bgmv_expand_slice` proof still lacks admitted fp16 add_commute, so the aggregate
Python tactic suite is not reported as passing.

### Recheck after signed min/max and view inference

The new SignedMinMaxViews regression and the existing SignedRange regression
pass in both interpreter paths. A full rebuild of the 36 previously completed
cases found the four padding-source alignment failures listed above. They are
temporarily excluded while their repairs are checked: **32 complete / 152
pending**. This supersedes the preceding historical count. The user-deferred
attention_kernel_aligned case remains separate. chunked_cumsum_fwd retains its
completed two-snapshot acceptance; context_attn_bloom is still in progress.

### Signed inference and unchecked-load repair accepted

The signed-expression inference now retains sign through minimum/maximum,
value-preserving layout hints, and singleton-axis views. The negative-value
regression passes in both real and arbitrary FP interpreters. Unchecked block
loads retain their actual unspecified-padding source tag; the shared semantic
lemma shows padding is never selected when no axes are checked and no mask is
present. No assumption identifies arbitrary padding with zero.

All 72 public Correct/FPEquiv entries across the 36 accepted cases rebuild
successfully. Source-alignment gates and the 432-headline / 210-file scope
inventory pass. Independent comparator/trust replay accepted 28 semantic, proof, entry and
regression files in `Logs/comparator-check-r9l7pjvx`. The remaining
source test still expected zero-padding tags in two Q loads; after correcting
both to unspecified, it passed a separate final replay in
`Logs/comparator-check-stl16obo`. The two receipts cover
29 files, 273 named declarations and
2054 kernel theorem targets. All 1415
source/config fingerprints match the final snapshot, and the test file is the
only source difference between these two snapshots.

Current acceptance is **36 complete / 148 pending**. This restores the four
attention cases reopened above. context_attn_bloom remains in progress;
attention_kernel_aligned remains user-deferred. The missing fp16 add_commute
admission still blocks bgmv_expand_slice, so the aggregate Python tactic suite
is not reported as passing.


## Context Bloom attention acceptance

`context_attn_bloom` now has shared complete typed sources and independent
real/FP entries. The candidate commutes only qk and sm_scale. Its real formula
uses initial arrays, the finite -1e8 sentinel and the source's separate
acc_scale guard; it is a guarded normalized-block recurrence rather than ideal
causal softmax. The formula-level finite-mask witness proves a difference
between those formulas; it is not an executed-source or GPU counterexample.

Both real contracts cover successful finite flat execution, actual signed
addresses, injective active stores and every untouched tagged cell. Metadata
and gathered page indices stay int32; Q/K/V/Out support fp16, bf16 and fp32.
Grouped KV heads, cached prefix, channel/time masks and both numerical
conversions are retained. Inputs may alias output because every load precedes
the only store. No zero-unspecified-value assumption is required.

The real execution witness has a finite allocation, grouped head, cached
prefix, nonzero program IDs and three visited key blocks including a tail.
The FP witness starts without registers and reaches the admitted multiplication
and actual masked store under a consistent abstract model. FP assumptions
print only mul_commute, on finite actual qk/scale operands. tl.exp is unchanged
and requires no exponential rewrite assumption.

The repository closure builds in 3507 jobs. The typed/signed source regression,
source-only gates, 19 focused source-audit tests and coverage/manifest checks
pass. Independent comparator/trust replay accepts all **68 files**:
**239 named declarations**, **1315 kernel theorem targets**,
and **1482 matching source/configuration fingerprints**.
Receipt: `Logs/comparator-check-u0a7p72i`. All named specifications, helpers and execution witnesses
are included in the verified target inventories. Explicit destination casts
are source-audited only when they match the known storage dtype and unchanged
pointer/value/mask; unknown or reassigned pointer aliases remain mismatches.

Current acceptance is **37 complete / 147 pending**. The next case is
context_attn_fwd. Its three-argument dot needs a distinct FP accumulator-dot
operation; that support and the next case remain in temporary drafts and are
not included in this receipt. attention_kernel_aligned remains user-deferred.


## Context forward attention integration

`context_attn_fwd` now has separate complete sources, real correctness and FP
equivalence entries. The mathematical formula is an initial-array base-2
weighted sum with finite sentinel and padded weights. Full execution, loop
invariants, actual address bounds, finite flat IO, full tagged-cell frame and
a nonempty three-block real witness are proved. The FP proof prints only
mul_commute and checks reached eager-where operands; its consistent-model
witness starts without registers and reaches the actual store.

The shared model now retains the explicit accumulator in three-argument
floating dot as a separate opaque dotAcc operation. Real interpretation adds
the matrix product to the accumulator. FP interpretation does not assert a
dot-plus-add rewrite. Default/output-accumulator dtype checks follow Triton
3.7.1, the compiler version recorded in the admission reports. A generic
where-arm multiplication rule composes the existing admitted scalar atom
while retaining all finite guards, both eager arms and failures. Focused
regressions distinguish fused and separate backends and reject missing guards
and inconsistent output/accumulator precision.

Five pending single-file consumers receive compatibility changes to retain
the new node in their real AST/proofs; their existing coverage classifications
are unchanged. Repository checks and independent replay are in progress.
Current accepted count remains **37 complete / 147 pending**.
attention_kernel_aligned remains user-deferred.


## Context forward validation follow-up

The broad shared-semantics rebuild exposed stale omitted-padding annotations
in three historical helpers: attention_score/Proofs/LegacyNaturalWindow,
attention_kernel/Proofs/Legacy, and attention_fwd_triton3/Proofs/LegacySource.
Their ASTs now retain unspecified padding, as their unchanged DSL source does.
The real execution proofs use the existing unchecked-load theorem: no boundary
check and no mask means no lane selects padding. All three repaired files pass
standalone Lean checks and their scope fingerprints have been refreshed.
They are included in the forthcoming 71-file independent replay.

All five final focused Lean regressions pass, including explicit dot
accumulators and eager-where guard preservation. Source-only gates, scope
inventory, proof manifest and spec-sheet generation pass. The full aggregate
build also confirms the known bgmv_expand_slice FP failure from missing fp16
add_commute admission; that failure is not treated as accepted. Final build
validation will select every file in the 37 accepted cases plus context_attn_fwd,
the affected libraries/examples, and the reopened case's real layer. Acceptance
remains 37 complete / 147 pending until that build and independent replay finish.


## Context forward attention acceptance

The complete original and scale-first candidate now pass review for fp16,
bf16 and fp32 storage. Both real contracts equal the independent initial-array
base-2 weighted sum, including the finite causal sentinel and padded scores.
The proofs establish successful execution, active output values, actual signed
address safety and a frame for all untouched tagged cells in finite flat memory.
Dimensions, strides, grouped heads and metadata remain symbolic; BN is positive
and active stores are injective. Inputs may alias output because all loads
precede the sole store. No zero-unspecified-value premise is needed.

The FP proof uses only mul_commute on finite actual qk/scale operands at every
reached iteration, including all lanes of the eager where arm. Dot, exp2,
reductions, casts, masks and stores are preserved. Three-argument dot remains
an opaque accumulator-dot operation. Neither a dot-plus-add identity nor an
exponential rewrite is assumed. Real and FP execution witnesses show inhabited
contracts; they do not claim hardware simulation or GPU validation.

The library, example and 38-case build plus five focused Lean regressions pass: source precision,
dtype resolution, dot options, accumulator-dot semantics, and rewrite tactics.
The 38-case build selects all files in the 37 accepted cases plus this case;
the reopened bgmv real layer is also checked. Its unadmitted FP path is excluded
and remains a known aggregate-build failure. Source-only corpus gates,
21 audit/layout unit tests and the scope/manifest
checks also pass. The broader rebuild exposed three historical attention
helpers with stale unchecked-padding AST annotations. Those annotations now
match their retained source, using the existing theorem that unchecked,
unmasked loads never select padding. Public source/contract scope is unchanged.
Five pending dot consumers retain their existing coverage classifications.

Independent comparator/trust replay accepts **71 files**,
**1173 named declarations**, and **8369 kernel theorem targets**.
All **1528 source/configuration fingerprints** match the final snapshot.
Receipt: `Logs/comparator-check-xynmtro4`. The verified inventories include the new source contracts,
execution witnesses, shared dot/where rules and all eight compatibility files.

Current acceptance is **38 complete / 146 pending**. context_attn_llama is next;
attention_kernel_aligned remains user-deferred. The missing fp16 add_commute
admission still blocks bgmv_expand_slice.


## Llama and Mistral context attention integration

Both cases now have complete original/candidate sources, two independent real
correctness specifications and full-program admitted FP equivalence. Llama
retains signed page gathers, exp2 and opaque accumulator-dot. Mistral retains
signed sliding-window subtraction, finite -1e9 masks, both numerical guards,
natural exp, and dot-plus-add order. Each candidate swaps only qk and sm_scale;
FP assumptions print mul_commute, with guards on actual reached operands.

Both real layers prove the independent initial-array weighted sum, successful
finite flat execution, actual address bounds and a full tagged-cell frame.
Their source finite sentinels and padded denominator weights are retained.
Dimensions/strides/launch IDs stay symbolic. Mistral requires nonnegative source
starts to preserve signed K/V pointer advances before conversion; both require
valid active addresses, positive BN and injective active stores. No result
register, output equality or zero-undefined-memory premise is required.

Complete real and abstract FP execution witnesses make the contracts inhabited.
The draft proof closures, source type regressions and source expression/order
checks pass. Repository checks and independent comparator/trust replay are next;
the accepted count remains 38 complete / 146 pending. The aligned case remains
user-deferred, and the reopened bgmv fp16 add_commute gap is unchanged.


The source audit now resolves unique stored-pointer aliases before interpreting
loaded tensor dtype attributes, and keeps macro-selected cast types visible in
the cast-sequence check. Reassigned/ambiguous pointers remain unresolved;
literal or wrong-precision casts remain mismatches. Explicit destination casts
are normalized only for the known storage type and unchanged pointer/value.
The source-only corpus gates and 23 focused audit/layout regressions pass after
these checks; no Lean numerical semantics or admission data changed.


The two-case repository closure now builds in 3570 jobs and both final typed
source regressions pass. The new source alias/cast audit passes the full
source-only corpus gates and all 23 focused tests. The independent 97-file
proof replay is now the remaining acceptance check; neither case is counted
as complete until the named inventories and final source fingerprints match.


## Llama and Mistral context attention acceptance

Both complete source/candidate pairs now pass real correctness, admitted FP
equivalence, source/type checks and independent proof replay. Each covers
fp16/bf16/fp32 storage and symbolic dimensions, strides and launch IDs.
The real outputs equal independent initial-array weighted sums, including
finite source sentinels and padded denominator weights. Successful finite
flat execution and preservation of every untouched tagged cell are proved.

Llama preserves signed page-table gathers, grouped heads, cached-prefix
handling, exp2 and opaque accumulator-dot. Its bounds cover actual request,
page, metadata, Q/K/V and output accesses. Mistral preserves signed sliding-
window subtraction, both -1e9 masks, the all-sentinel center replacement,
zero-denominator guard, natural exp and normalized dot-plus-add updates.
Its nonnegative start condition preserves signed K/V pointer advances before
natural-offset conversion. Both require positive BN, valid active addresses
and injective active output stores; inputs may alias output because all loads
precede the sole store. Their full execution witnesses make the contracts
inhabited without precomputed results or zero-undefined-memory assumptions.

Both FP proofs print only mul_commute, with guards on every actual reached
operand lane; Llama also checks unselected lanes of the eager where arm.
Neither proof adds an exponential identity, associativity or dot-fusion rule.
Consistent abstract FP execution witnesses establish nonempty premises, not
physical GPU execution or IEEE equality. Fixed-width index overflow and
concurrent launch safety are outside these per-program contracts.

The repository closure builds in 3570 jobs; both final source precision
regressions, source-only corpus gates, 23 source-audit/layout tests and the
coverage/manifest checks pass. Source audits resolve only unique pointer
aliases and retain symbolic cast targets so missing/wrong casts remain visible.
Independent comparator/trust replay accepts **97 files**,
**342 named declarations**, and
**2733 kernel theorem targets**; all
**1623 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-b23tm5xw`.

Current acceptance is **40 complete / 144 pending**. context_attn_nopad is next.
attention_kernel_aligned remains user-deferred. The missing fp16 add_commute
admission still leaves bgmv_expand_slice's FP layer reopened; no aggregate
all-pass claim is made.


## Nopad context attention and cosine integration

Both cases now have complete typed original/candidate pairs, two independent
real specifications, full-program FP equivalence and inhabited real/abstract FP
execution witnesses. Nopad retains causal negative-infinity masks and normalized
online updates, using only admitted mul_commute. Cosine preserves intrinsic
tl.cos and all conversions while commuting only an exact integer index product;
its FP proof uses no numerical atoms.

Nopad proves full finite flat IO with actual bounds and tagged-cell preservation,
for symbolic dimensions/strides and all storage types, under BN>0 and the stated
start/address/store conditions. Its real model uses WithBot propagation for
masked arithmetic. Cosine proves masked real flat IO and the full frame; its
shared flat adapter retains the zero unspecified-value oracle premise, while
its region proof permits arbitrary unspecified values. Both cover all
fp16/bf16/fp32 storage branches. No GPU, IEEE, index-overflow or concurrent-launch
claim is made.

Source normalization now handles explicit destination casts through b+offset
without accepting wrong casts, changed values or changed addresses. Both draft
closures and witnesses compile. Repository tests and independent replay are
pending; acceptance remains 40 complete / 144 pending.


## Nopad context attention and cosine acceptance

Both complete typed source/candidate pairs now pass their real specifications,
whole-program FP equivalence, source/type checks and independent proof replay.
Nopad's real formula has zero causal-masked weights in the WithBot real model;
its normalized online update, every visited key block and source storage/casts
remain. Full flat execution and the tagged-cell frame follow under positive BN,
nonnegative source start, actual active-access bounds and injective stores.
Its FP derivation uses only mul_commute at actual reached qk/scale operands.

Cosine proves lane-wise Real.cos and masked flat IO for every storage branch.
Its region result permits arbitrary unspecified values; the shared flat IO
adapter's zero-oracle launch premise remains explicit. Exact integer index
commutation preserves opaque tl.cos and all conversions without numerical atoms.
Both cases have successful real and abstract FP execution witnesses. These
are per-program model results, not IEEE or physical/concurrent GPU guarantees.

Repository proof closures, both source-type regressions, source-only corpus
gates, 25 audit/layout tests and coverage/manifest checks pass. Independent
comparator/trust replay accepts **59 files**,
**192 named declarations**, and
**1834 kernel theorem targets**; all
**1680 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-vqdq8v6p`.

Current acceptance is **42 complete / 142 pending**. cross_entropy1 is next;
its full backward layer is being implemented, replacing the old store-only
slice. attention_kernel_aligned remains user-deferred and the bgmv fp16
add_commute gap is unchanged. No aggregate all-pass claim is made.


## Cross entropy integration

Both complete forward/backward JITs now have typed original/candidate sources,
independent real formulas, full flat IO and whole-program FP equivalence.
Backward no longer accepts precomputed probabilities: it derives the gradient
from the initial logits and legitimate LSE/dloss inputs. All smoothing/split,
ignored-label and signed-label paths are retained. Only exact integer column
index multiplication is commuted, so neither FP proof uses numerical atoms.
The three storage branches and the explicit fp32 loss/LSE conversions remain.

Forward's positive-block, separation, active-bound and shared zero-oracle
conditions are visible. Backward bounds follow actual accesses; ignored labels
skip dloss only, and all inputs may alias the final output. Successful finite
flat execution and full tagged-cell frames are proved. Finite real witnesses
cover all storage and compile-time branches, and abstract typed FP executions
establish inhabited premises without claiming GPU behavior. Per-program model
scope excludes host split aggregation, fixed-width overflow and concurrent
launch safety.

Draft builds and both-JIT source expression/type checks pass. Repository
validation and independent replay are pending; acceptance remains 42/184.


## Cross entropy acceptance

Both full JITs and index-commuted candidates now pass all four real and both
FP specifications. Backward derives its probabilities from the initial logits
and legitimate LSE input, replacing the old precomputed-probability store slice.
The independent forward block-LSE/loss and backward gradient formulas include
all signed-label, ignored-label, smoothing and split paths. Both real layers
prove successful finite flat execution and the full tagged-cell frame for
symbolic sizes/strides and all three storage branches. Forward retains its
explicit positive-block, separation and zero-oracle conditions; backward
requires actual-access bounds and permits input/output aliases.

Both FP proofs print none: only an exact integer index product is commuted.
Numerical operations, casts, masks, stores and failure are preserved. Real
finite witnesses cover the branches, including an empty dloss allocation for
ignored labels. Abstract FP executions cover nonzero row/block and partial
lanes; they do not certify physical GPU execution. Host aggregation, IEEE
rounding, fixed-width index overflow and concurrent launches remain outside
these per-program model contracts.

The repository closure, dtype regression, both-JIT expression correspondence,
source-only corpus gates, 28 audit/layout tests and coverage/manifest checks
pass. Independent comparator/trust replay accepts **19 files**,
**76 named declarations**, and
**1313 kernel theorem targets**; all
**1698 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-jxlilcow`.

Current acceptance is **43 complete / 141 pending**. cross_entropy2 is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Scaled cross entropy integration

Both complete scaled forward/backward sources and candidates are migrated.
Four real specifications include the scaled initial-array formulas and gated
z-loss output; two whole-program FP equivalences commute only integer column
index multiplication. Shared DSL repairs preserve nested load storage and
floating branch joins. Targeted regressions, affected example builds and
independent replay must finish before acceptance. Acceptance remains 43/184.


## Scaled cross entropy acceptance

Both full JITs and index-commuted candidates now pass all four real and both
FP specifications. Backward derives its probabilities from the initial scaled logits
and legitimate LSE input, adding the previously missing complete backward proof.
The independent forward scaled block-LSE/loss/z-loss and backward gradient formulas include
all signed-label, ignored-label, smoothing and split paths, including the
gated z-loss store. Both real layers
prove successful finite flat execution and the full tagged-cell frame for
symbolic sizes/strides and all three storage branches. Forward retains its
explicit positive-block, pairwise output separation, LSE/logit separation and zero-oracle conditions; backward
requires actual-access bounds and permits input/output aliases.

Nested low-precision loads now retain storage type and floating branch joins
retain their fp32 binding. Both shared DSL repairs pass targeted regressions
and rebuilds of the migrated cases and worked examples.

Both FP proofs print none: only an exact integer index product is commuted.
Numerical operations, casts, masks, stores and failure are preserved. Real
finite witnesses cover the branches, including an empty dloss allocation for
ignored labels. Abstract FP executions cover nonzero row/block and partial
lanes; they do not certify physical GPU execution. Host aggregation, IEEE
rounding, fixed-width index overflow and concurrent launches remain outside
these per-program model contracts.

The repository closure, dtype regression, both-JIT expression correspondence,
source-only corpus gates, 28 audit/layout tests and coverage/manifest checks
pass. Independent comparator/trust replay accepts **22 files**,
**85 named declarations**, and
**1484 kernel theorem targets**; all
**1718 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-5gyzxque`.

Current acceptance is **44 complete / 140 pending**. cross_entropy_ops is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Cross entropy ops integration

Both complete scaled forward/backward sources and candidates are migrated.
Four real specifications include the scaled initial-array formulas and gated
z-loss output; two whole-program FP equivalences commute only integer column
index multiplication. The extra backward smooth_positive assignment is retained. Source checks,
repository builds and independent replay must finish before acceptance. Acceptance remains 44/184.


## Cross entropy ops acceptance

Both full JITs and index-commuted candidates now pass all four real and both
FP specifications. Backward derives its probabilities from the initial scaled logits
and legitimate LSE input, replacing the precomputed-probability store slice. The unused smooth_positive
assignment remains in the complete backward source.
The independent forward scaled block-LSE/loss/z-loss and backward gradient formulas include
all signed-label, ignored-label, smoothing and split paths, including the
gated z-loss store. Both real layers
prove successful finite flat execution and the full tagged-cell frame for
symbolic sizes/strides and all three storage branches. Forward retains its
explicit positive-block, pairwise output separation, LSE/logit separation and zero-oracle conditions; backward
requires actual-access bounds and permits input/output aliases.

The source uses the repaired nested-load and branch-type propagation.

Both FP proofs print none: only an exact integer index product is commuted.
Numerical operations, casts, masks, stores and failure are preserved. Real
finite witnesses cover the branches, including an empty dloss allocation for
ignored labels. Abstract FP executions cover nonzero row/block and partial
lanes; they do not certify physical GPU execution. Host aggregation, IEEE
rounding, fixed-width index overflow and concurrent launches remain outside
these per-program model contracts.

The repository closure, dtype regression, both-JIT expression correspondence,
source-only corpus gates, 28 audit/layout tests and coverage/manifest checks
pass. Independent comparator/trust replay accepts **19 files**,
**79 named declarations**, and
**1409 kernel theorem targets**; all
**1736 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-xi8elffs`.

Current acceptance is **45 complete / 139 pending**. decay_cumsum is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Decay cumsum integration

All three complete typed JITs and mask-index-commuted candidates are migrated.
Six real specifications include independent prefix, preparation and reverse
gradient formulas, successful finite flat execution and the full tagged-cell
frame. Preparation retains the unmasked final gate load, and backward retains
its in-place gradient stores and final pointer decrements. Three whole-program
FP equivalences commute only the natural-number lane-mask product and print
none. Source storage and gate storage remain independent where applicable.
Repository/source checks and independent replay remain pending. Acceptance
stays 45/184 until those checks pass.


## Decay cumsum acceptance

All three complete JITs and index-commuted candidates pass six real and three
FP specifications. The forward formula is the scaled initial gate prefix sum;
preparation yields q*exp2(g)*scale and k*exp2(last_g-g); backward yields the
corrected dq/dk streams and the reverse sum of dq*q-dk*k.
The complete real contracts prove successful finite flat execution and the
full tagged-cell frame with symbolic sizes/strides, BK<=DK, positive BT for
preparation/backward, explicit region separations, actual-access bounds and
zero unspecified values. Preparation includes every last-row gate lane in
its unmasked read; backward retains in-place dq/dk stores and final pointer
decrements. All three storage types are covered, with independent data/gate
storage where applicable.

FP proofs print none: only the exact natural-number lane-mask product is
commuted. Numerical operations, casts, masks, stores, complete states and
failure are preserved. Finite real witnesses cover all storage pairs and
source variants; real-valued abstract FP executions cover two iterations,
nonzero program IDs and a partial lane from empty registers. These witnesses
are model executions, not physical GPU or IEEE certification. Host launches,
concurrent programs and fixed-width index overflow remain outside scope.

The source auditor now preserves declared pointer element types through
+=/-= advances; regressions retain ambiguity and detect changed value,
address, mask or storage cast. Repository builds, all-three-JIT source and
precision checks, source-only corpus gates, 31 audit/layout tests and
coverage/manifest checks pass. Independent comparator/trust replay accepts
**27 files**, **197 named declarations**
and **1768 kernel theorem targets**. All
**1762 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-d9lunxqu`.

Current acceptance is **46 complete / 138 pending**. dequantize_matmul is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Dequantize matmul integration

The complete typed JIT now retains signed integer weight loads and conversion,
fp32 column scales and all three output storage types. Two real specifications
prove the initial integer-weight/scale product, successful finite flat IO and
the full tagged-cell frame. The wider scale read remains required even with
no active weight/output rows. FP equivalence commutes only an integer mask
product and prints none. The host torch.mm call remains outside the JIT scope.
Repository/source checks and independent replay remain pending. Acceptance
stays 46/184 until those checks pass.


## Dequantize matmul acceptance

The full dequantization JIT and index-commuted candidate pass two real
specifications and whole-program FP equivalence. Signed int8 weights now
remain integer loads with an explicit conversion in the expression, replacing
the earlier real-only input model. Column scales use fp32 storage and output
supports fp16/bf16/fp32. The mathematical payload uses Int; the real formula
is its real conversion times the initial column scale.

Real contracts prove successful finite flat execution and the full tagged-cell
frame for symbolic sizes/strides, positive K block, injective tile output
addresses, actual-access bounds and a zero unspecified-value oracle. The
column-scale mask is wider than the weight/output mask, including when K is
empty. Inputs may alias output because all reads precede the sole store.
Finite witnesses use negative integer cells and partial tiles at nonzero
program IDs, and empty weight/output allocations with a nonempty scale read.
FP equivalence prints none: only the natural-number column-mask product is
commuted. Full typed abstract executions are inhabited; they are not physical
GPU/IEEE measurements. The host torch.mm call, concurrent launches and
fixed-width index overflow remain outside the JIT contract.

Repository builds, source/precision checks, source-only corpus gates,
31 audit/layout tests and coverage/manifest checks pass. Independent
comparator/trust replay accepts **13 files**,
**30 named declarations** and
**467 kernel theorem targets**. All
**1774 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-6pu188lk`.

Current acceptance is **47 complete / 137 pending**. dequantize_rowwise is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Rowwise dequantization integration

The complete source and candidate retain signed integer loads/conversion,
fp32 scale/product, fp16 output and the omitted weight-load default. Two real
specifications prove initial row-scale*integer-weight*factor, successful finite
flat execution and the full tagged-cell frame. Positive block/lane counts
retain the bound for the unmasked row-scale load. FP equivalence commutes only
the exact integer block-start product and prints none. Repository/source
checks and independent replay remain pending. Acceptance stays 47/184.


## Rowwise dequantization acceptance

The complete typed source and candidate pass both real specifications and
whole-program FP equivalence. Signed int8 weights use the actual integer
load and conversion, fp32 row scales/products and fp16 output. The independent
real formula is initial row scale*(integer weight as real)*inv_127. The source
still omits other on its masked weight load and reads its row scale unmasked.

Both real specifications prove successful finite flat execution and the
complete tagged-cell frame with symbolic sizes, program IDs and factor.
Positive block/lane counts provide an active lane for the unmasked scalar
scale bound. Finite allocations, actual-access bounds and zero unspecified
values remain explicit. Input/output aliasing is allowed because reads
precede the sole store. Finite real and abstract FP witnesses use empty
registers, negative integer inputs, a nonzero row and a masked tail.
FP equivalence prints none: only an exact integer block-start product is
commuted, preserving the numerical operations, conversions, memory and
failure. These are model executions; host grid/power-of-two selection,
physical GPU/IEEE rounding, concurrency and fixed-width overflow remain
outside scope.

Repository builds, source/type/load-policy checks, source-only corpus gates,
31 audit/layout tests and coverage/manifest checks pass. Independent
comparator/trust replay accepts **10 files**,
**23 named declarations** and
**407 kernel theorem targets**. All
**1783 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-nwk3uoa7`.

Current acceptance is **48 complete / 136 pending**. destindex_copy is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Destination-indexed dual-copy integration

The complete source and candidate now retain signed destination metadata,
two independent payload storage types and both unmasked stores. Real
specifications prove independent initial copy formulas, successful finite
flat execution and tagged-cell framing with nonnegative destination rows,
actual-access bounds and collision-free outputs. Source head parameters
remain unused; only the first head is copied and padding remains unmasked.
FP equivalence composes exact integer/address congruences and prints none.
Repository/source checks and independent replay remain pending.
Acceptance stays 48/184.


## Destination-indexed dual-copy acceptance

The complete typed source and candidate pass two independent real copy
specifications and full-program FP equivalence. Destination metadata is
signed; nope/rope payload storage independently supports fp16/bf16/fp32,
with each output matching its corresponding input. The source ignores head
counts/strides, copies the first head and leaves all padding lanes unmasked.
No host padding repair or all-head claim is introduced.

The real contracts prove successful finite flat execution, both output
readbacks and the complete tagged-cell frame. They retain nonnegative loaded
rows, actual-access bounds, injective output lane addresses, distinct output
regions and zero unspecified values. Inputs may alias outputs because every
read precedes both stores. Finite real and abstract typed executions cover
all supported dtype pairs and source variants from empty registers, with
unequal power-of-two tile widths and different nonzero source/destination rows.

FP assumptions print none: exact natural-number multiplication commutation
and pointer congruence preserve all numerical values, metadata arithmetic,
precisions, memory effects and failure. The result does not certify physical
GPU/IEEE rounding, host allocation, concurrent launch races or fixed-width
overflow.

Repository closure, source/type checks, source-only corpus gates,
31 audit/layout tests and coverage/manifest checks pass. Independent
comparator/trust replay accepts **14 files**,
**29 named declarations** and
**715 kernel theorem targets**. All
**1796 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-ixadhmi3`.

Current acceptance is **49 complete / 135 pending**. destindex_copy_kv1 is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Masked multi-head destination-copy integration

The complete source and candidate retain signed destination metadata,
independent input/output precision, the store conversion and both head/column
masks. Real specifications prove independent initial copy values, successful
finite flat execution and complete tagged-cell framing with nonnegative
rows, actual-access bounds and an injective output tile layout. Metadata
remains unmasked even for an inactive payload tile. FP equivalence composes
exact integer/address congruences and prints none. Repository/source checks
and independent replay remain pending. Acceptance stays 49/184.


## Masked multi-head destination-copy acceptance

The complete typed source and candidate pass two independent real copy
specifications and whole-program FP equivalence. Destination metadata is
signed, input/output independently support fp16/bf16/fp32, and the store's
precision conversion is retained. Both head and column masks and other=0.0
remain unchanged. The scalar metadata load is unmasked even for an entirely
inactive payload tile.

The real contracts prove successful finite flat execution, active output
readback and the complete tagged-cell frame. They retain nonnegative loaded
rows, actual-access bounds, an injective output tile layout and zero
unspecified values. Input/output aliases are allowed because the entire tile
is read before the sole store. Finite real and abstract typed executions
cover all nine storage pairs and both source variants from empty registers,
with both tails masked in a 4x4 tile and distinct nonzero source/destination
rows. An additional finite witness has empty payload allocations with a
nonempty metadata read.

FP assumptions print none: exact natural-number multiplication commutation
and pointer congruence preserve values, metadata arithmetic, storage casts,
masks, memory effects and failure. Host allocation/grid composition, races,
fixed-width overflow and physical GPU/IEEE behavior remain outside scope.

Repository closure, source/type checks, source-only corpus gates,
31 audit/layout tests and coverage/manifest checks pass. Independent
comparator/trust replay accepts **13 files**,
**30 named declarations** and
**722 kernel theorem targets**. All
**1808 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-nbhwhlk1`.

Current acceptance is **50 complete / 134 pending**. destindex_copy_kv2 is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Head-masked multi-head destination-copy integration

The complete source and candidate retain signed destination metadata,
independent input/output precision, the store conversion and the head-only
mask. Real specifications prove independent initial copy values, successful
finite flat execution and complete tagged-cell framing with nonnegative
rows, actual-access bounds and an injective output tile layout. Metadata
remains unmasked even for an inactive payload tile. FP equivalence composes
exact integer/address congruences and prints none. Repository/source checks
and independent replay remain pending. Acceptance stays 50/184.


## Head-masked multi-head destination-copy acceptance

The complete typed source and candidate pass two independent real copy
specifications and whole-program FP equivalence. Destination metadata is
signed, input/output independently support fp16/bf16/fp32, and the store's
precision conversion is retained. The head-only mask and other=0.0 remain unchanged. Every column of an
active head is accessed, with no added width mask or host padding repair. The scalar metadata load is unmasked even for an entirely
inactive payload tile.

The real contracts prove successful finite flat execution, active output
readback and the complete tagged-cell frame. They retain nonnegative loaded
rows, actual-access bounds, an injective output tile layout and zero
unspecified values. Input/output aliases are allowed because the entire tile
is read before the sole store. Finite real and abstract typed executions
cover all nine storage pairs and both source variants from empty registers,
with a head tail and all active columns in a 4x4 tile and distinct nonzero source/destination
rows. An additional finite witness has empty payload allocations with a
nonempty metadata read.

FP assumptions print none: exact natural-number multiplication commutation
and pointer congruence preserve values, metadata arithmetic, storage casts,
masks, memory effects and failure. Host allocation/grid composition, races,
fixed-width overflow and physical GPU/IEEE behavior remain outside scope.

Repository closure, source/type checks, source-only corpus gates,
31 audit/layout tests and coverage/manifest checks pass. Independent
comparator/trust replay accepts **13 files**,
**31 named declarations** and
**723 kernel theorem targets**. All
**1820 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-qwtmvj4y`.

Current acceptance is **51 complete / 133 pending**. diag_ssm_triton is next.
attention_kernel_aligned remains user-deferred and the bgmv fp16 add_commute
gap remains open. No aggregate all-pass claim is made.


## Expression pointer subtraction support

The DSL now preserves static pointer element types and accumulates signed
subtraction offsets before converting the complete address to a natural
number. Dynamic pointer-register subtraction uses the existing natural-offset
semantics. This admits the original previous-row loads in both diagonal-SSM
backward JITs. The numerical subtraction path remains unchanged.

A new execution regression checks positive/negative offsets, an intermediate
negative static offset recovered by a later addition, vector broadcasting,
explicitly typed pointer-register loads, and ordinary natural/real subtraction.
The base library, three destination-copy typed sources, PointerSignedOffset,
FPReversePointer, TritonSmoke, 31 source/layout tests and source-only corpus
gates all pass. Independent replay accepts 2 files,
3 named declarations and 113
kernel targets; all 1821 input fingerprints match.
Receipt: `Logs/comparator-check-f1u7eri0`.

This is dependency validation only. Case acceptance remains **51 complete /
133 pending**; diagonal-SSM source/FP and real-IO work is still in progress.


## Diagonal SSM integration

All four complete real/complex forward/backward sources now have typed
index-commuted candidates. Eight public real contracts cover independent
forward/reverse recurrences, any sequence length including zero, masked tails,
finite flat execution and tagged-cell frames. Sufficient alias conditions are
visible; complex forward permits in-place input/output. Empty backward scans
allow aliased terminal outputs. Four whole-program FP equivalences use no
numerical atoms. Repository/source checks and independent replay remain
pending; acceptance stays **51 complete / 133 pending**.


## Diagonal SSM acceptance

All four complete real/complex forward/backward JITs and their integer-index
commuted candidates pass eight independent real specifications and four
whole-program FP equivalences. Real paths retain matching fp16/bf16/fp32
storage; complex64 retains interleaved fp32 components and source gradient
conjugation signs. Precision is erased only for mathematical correctness.

Public contracts display the initial-array recurrence/gradient formulas,
finite allocation and actual bounds, successful full flat execution, all
outputs and complete tagged-cell frames. Sequence length is arbitrary,
including zero, and partial tiles do not require BLOCK_SIZE<=batch*dim.
Complex forward permits in-place input/output and odd allocation bases.
The real positive-length adapter still requires distinct input/output names;
the sufficient nonempty backward separation conditions are explicit. Empty
backward paths permit aliased terminal outputs because both gradients are
zero; transition loads and terminal stores remain bounded. Zero unspecified
values remain a flat-memory adapter premise.

Finite witnesses cover positive and empty sequences, all real storage choices,
both previous-state branches, negative values, complex conjugation, odd bases,
masked tails and retained integer-tagged cells. Abstract typed executions from
empty registers inhabit all original/candidate variants. FP assumptions print
none: integer multiplication commutation preserves floating arithmetic,
precision, masks, stores, final states and failure. These are per-program
results, excluding host autograd/batch reduction, concurrent launch races,
fixed-width index overflow and physical GPU rounding certification.

Repository closure, source correspondence for all four JITs, precision/mask
regressions, source-only corpus gates, 31 audit/layout tests and coverage/
manifest checks pass. Independent comparator/trust replay accepts
**50 files**, **325 named declarations**
and **1532 kernel theorem targets**. All
**1870 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-kmtknxce`.

Current acceptance is **52 complete / 132 pending**. dropout_triton is next.
attention_kernel_aligned remains user-deferred; the bgmv fp16 add_commute gap
remains open. No aggregate all-pass claim is made.


## Dropout integration

Full typed dropout now retains the signed int32 keep buffer in the Python
examples; integer conditions use nonzero truthiness. Bool keep buffers and
matching fp16/bf16/fp32 data/output are also covered. Original and integer-index
commuted candidates have independent real finite-IO contracts and a complete
FP equivalence with no numerical atoms. Destination casts, eager division and
masked omitted-padding loads are retained. Finite partial/empty witnesses and
abstract typed executions pass in staging. Repository and independent replay
checks remain; acceptance stays **52 complete / 132 pending**.


## Dropout acceptance

The complete source and integer-index commuted candidate now preserve int32
keep storage from the Python examples, bool keep storage, matching
fp16/bf16/fp32 data/output, omitted load defaults, eager selection arithmetic
and implicit destination conversion. Integer keep values use nonzero
truthiness, including negative/even values; generic signed/natural condition
regressions check the frontend conversion.

Both real specifications prove the independent initial-input dropout formula,
finite flat execution, actual bounds and the complete tagged-cell frame.
Shapes/program indices remain symbolic and region names may alias. Zero
unspecified values remain a visible flat-adapter premise. The real division
is Lean's total division (including p=1), not an IEEE claim. Partial-tile
witnesses yield 0/-6/-6 from keep 0/2/-1 and retain an integer-tagged tail;
an all-masked p=1 case executes with empty allocations.

Whole-program FP equivalence uses no numerical atoms and preserves failure.
All storage/keep choices have abstract typed executions from empty registers
for arbitrary p. These results exclude random-mask generation, host launches,
concurrent races, fixed-width index overflow and physical rounding certification.

Repository closure, precision/mask/execution and existing frontend regressions,
full Python-source correspondence, source-only corpus gates, 31 audit/layout
tests and coverage/manifest checks pass. Independent comparator/trust replay
accepts **19 files**, **63 named declarations**
and **773 kernel theorem targets**, including the
shared integer-condition frontend and regressions. All
**1881 source/configuration fingerprints** match.
Receipt: `Logs/comparator-check-sce_37j5`.

Current acceptance is **53 complete / 131 pending**. embedding_triton_kernel
is next. attention_kernel_aligned remains user-deferred; the bgmv fp16
add_commute gap remains open. No aggregate all-pass claim is made.


## Embedding integration

The full source now retains signed int32 token IDs instead of the previous
Nat surrogate. Original and integer-index commuted candidate have independent
real finite-IO contracts for arbitrary chunk sizes and full loops, including
final chunks extending past BLOCK_N. The output stride constraint covers only
active columns, preserving padded widths larger than physical row stride.
Typed FP equivalence needs no numerical atoms. Staged public finite witnesses
cover negative valid/invalid IDs, two-row chunks, overshoot and tagged tail;
typed execution covers masked tails across all three storage choices.
Repository and independent replay checks remain; acceptance stays
**53 complete / 131 pending**.


## Embedding acceptance

The complete original/candidate now retain signed int32 token IDs and matching
fp16/bf16/fp32 payload/output storage. Both real specifications prove the
independent initial-weight lookup, successful finite flat execution, every
active output and the full tagged-cell frame. Shapes and chunk sizes are
symbolic: multi-row chunks and a final chunk extending past BLOCK_N are
included. Only the source's n_ctx/hidden masks suppress writes. Output stride
covers min(padded width,hidden width), allowing inactive padded columns to
overlap physical rows.

The two input regions must differ from output. Actual-address bounds, finite
allocation/coverage and zero unspecified values for the flat adapter remain
explicit. Zero chunk stride follows formal range semantics, without asserting
that such a launch compiles in Triton. Scope excludes host/grid races,
fixed-width index overflow and physical rounding certification.

FP equivalence commutes only the integer start_n multiplication and retains
all floating conversion, masks, stores, final states and failure. No numerical
atom or guard is required. Typed model executions from empty registers cover
all storage choices and both candidates across three iterations with a masked
row/column. The public finite witness uses NN=2,N=3,D=4,hidden=stride=2,
negative valid/invalid IDs and the overshooting last row, while preserving an
integer-tagged tail. All-masked runs use empty allocations.

Repository closure, precision/mask/signed-operation regressions, public
execution witnesses, full Python-source correspondence, corpus source gates,
32 audit/layout tests and coverage/manifest checks pass. The source auditor
now trims whitespace left by typed default-value splices; its regression also
checks that changed parameter names and arithmetic still differ.
Independent comparator/trust replay accepts **16 files**,
**40 named declarations** and
**682 kernel theorem targets**, with all
**1896 source/configuration fingerprints** current.
Receipt: `Logs/comparator-check-ziggvwz7`.

Current acceptance is **54 complete / 130 pending**. f8_conversion_utils is
next. attention_kernel_aligned remains user-deferred; the bgmv fp16
add_commute gap remains open. No aggregate all-pass claim is made.


## FP8 conversion integration

Both complete conversion JITs now retain fp8 e4nv/e5 input/output and fp16/fp32
source channels. The duplicate forward store, actual destination casts,
masks and omitted padding remain. Four real contracts prove the initial-input
identity only after precision erasure; two typed FP equivalences commute the
integer index while preserving all floating conversions. No numerical atom,
cast-idempotence or lossless-conversion assumption is used. All six precision
combinations have abstract typed executions; public finite partial/empty
witnesses and exact precision/cast/mask checks pass in staging. Repository and
independent replay checks remain; acceptance stays **54 complete / 130 pending**.


## FP8 conversion acceptance

Both original/candidate conversions now retain their actual precision channels:
fp8 e4nv/e5 to fp16 (both original stores), and fp16/fp32 to fp8 e4nv/e5.
Implicit destination casts are explicit in the typed source; load defaults,
masks and all stores are unchanged. Four real contracts prove the independent
initial-input copy formula after precision erasure, successful finite flat
execution, all active outputs and the complete tagged-cell frame for symbolic
N, block size and program index. Logical regions may alias; actual bounds,
finite allocation/coverage and zero unspecified values for the flat adapter
remain visible.

The two typed FP equivalences commute only the integer index product. They
preserve all conversion nodes, full states and failure, without numerical atoms,
cast-idempotence assumptions, lossless fp8 conversion or round-trip claims.
Every precision-direction combination has original/candidate executions from
empty registers in an abstract model. Public finite witnesses copy -3 and 2,
retain an integer-tagged masked tail and execute all-masked cases with empty
allocations. Physical IEEE quantization, host reinterpret/launch correctness,
grid races and fixed-width index overflow remain outside scope.

Repository closure, public execution witnesses, typed model executions, actual
precision/cast/mask checks, full correspondence for both Python JITs, source
gates, 32 audit/layout tests and coverage/manifest checks pass. The old source
exception was removed because typed destination conversion is checked directly.
The first independent replay ran out of disk space and supplied no accepted
result. After removing only temporary replay caches while retaining sources
and receipts, a fresh independent comparator/trust replay accepts
**11 files**, **39 named declarations** and
**570 kernel theorem targets**, with all
**1906 source/configuration fingerprints** current.
Receipt: `Logs/comparator-check-eauavdpw`.

Current acceptance is **55 complete / 129 pending**. fast_ce_loss is next.
attention_kernel_aligned remains user-deferred; the bgmv fp16 add_commute gap
remains open. No aggregate all-pass claim is made.


## Fast CE source and frontend review

The next case retains three complete JITs and all storage/constexpr variants
in the isolated development workspace `/tmp/fast-ce-backend-work`. The staged
three whole-program FP index-commutation proofs compile with no numerical
atoms. Source-level precision checks identified unwanted widening of half
constexpr multiplication, and direct `tl.load(...).to(tl.int32)` was missing
from the signed-load assignment cases. The staged fixes have focused
regressions, including a loaded fp32 scalar that must still promote. A distinct
opaque libdevice-tanh operation prevents substituting the intrinsic backend.
The source branches are staged before register typing, as Triton constexpr
specialization requires.

The original chunked source puts both stores inside `chunk_idx == 0`. A checked
complete-source real counterexample at chunk one leaves its LSE cell at the
initial value 17, while the one-element chunk has mathematical LSE 2. This is
not a floating-point experiment and does not reject any FP atom. Softcapping
also brings padded lanes into the forward reduction; the corresponding full
source refutation is being completed. The new specifications must preserve
these behaviors and may not claim the host's aggregation is correct.

The 55 previously accepted cases are being rebuilt against the staged shared
changes before any integration. Full real input/output contracts, all typed
execution witnesses, source correspondence and independent replay remain.
The public repository's Lean sources are unchanged by this development step,
and acceptance stays **55 complete / 129 pending**. Run state and detailed
logs are recorded in `/tmp/fast-ce-affected-plan.json`,
`/tmp/fast-ce-affected-results.json`, `/tmp/fast-ce-backend-changes.json` and
`/tmp/fast-ce-*-*.log`.

## Fast CE proof development checkpoint

The isolated development snapshot now compiles the full Forward and Backward
real input/output contracts in finite flat memory, for both original and
index-commuted sources and all three input storage variants. These prove
successful whole-program execution, independently defined values, bounds,
and the complete tagged-cell frame. Public real domains require signed typed
labels equal to -100 or in vocabulary range; enabled softcap and logit scale
parameters are positive. This keeps the negative-infinity padding interpretation
within the stated real semantics. Host aggregation and arbitrary nonpositive
parameter behavior are not claimed.

Both complete-source counterexamples compile: chunk one retains its LSE cell
at 17 although its mathematical one-element LSE is 2; capped padding at V=1,
B=2, logit=0 and SC=1 produces log(1+exp(-1)), which is strictly positive.
The independent finite exponential-sum bridge includes this padding term.
The full typed originals have 120 successful model executions across storage,
flags, ignored labels and chunk/block positions; all corresponding candidate
executions follow from the three atom-free FP equivalences. The real candidate
proof separately equates the complete flattened executions by exact integer
index arithmetic.

All three Python JIT statement trees match the prepared source template,
including cast locations, assignment order, reduction arguments, and nested
store branches, after explicit destination-store casts and Lean syntax are
normalized. The checked layout separates the constexpr-staging helper into
its own module; it is prepared but not yet applied to the development imports.
Chunked real contracts, the concrete finite public witnesses, final layout,
source gates and independent replay remain pending. Shared frontend regression
is still running with per-case results saved; no new corpus acceptance is made.
The durable development index is `/tmp/fast-ce-progress.json`; source checks
are in `/tmp/fast-ce-source-check.json`. Acceptance remains **55 / 184**.


## Fast CE public specification checkpoint

The isolated snapshot now passes all six original/candidate real specifications
and all three complete typed FP equivalences. Forward LSE retains capped tail
contributions; Chunked output is conditional on chunk zero, with full memory
preservation on every later chunk. Backward exposes its caller-supplied LSE
and upstream values. All formulas, typed valid labels, positive enabled
parameters, finite access bounds, symbolic dimensions, successful execution
and tagged-cell frames are visible in the public real entry. FP has an empty
domain and no numerical atoms, preserving whole states and failure.

Both source refutations, concrete finite flat witnesses (including an ignored
backward label with no upstream allocation), and 120 typed original executions
with their candidate counterparts pass. Precision checks cover all storage and
flag combinations and the distinct forward/chunked gather cast positions.
Prepared migration separates the pure constexpr staging helper into Source/,
proofs into Proofs/, and regressions into bench/tests/. It has not yet replaced
the repository entry. Shared regressions, integrated checks and independent
comparator/trust replay remain pending. Acceptance stays 55/184.


## Fast CE repository integration

The reviewed 27 case/test files and 15 shared semantic/frontend changes are now
in the repository. The source templates and public contracts replace the old
single-file partial proof; pure constexpr staging is under Source/, support
proofs under Proofs/, and precision/execution regressions under bench/tests/.
The old pointer-load/cast exception is removed. A documented constexpr-staging
exception retains both flags through all four specializations; the full source
comparison and dtype tests check their correspondence. The nine public specifications
have explicit source-fingerprinted coverage reviews. The source inventory is
490 specifications in 230 files, with 236 full-value candidates and 223
specializations; this is a scope inventory, not a new acceptance count.

The official-path build and the isolated 55-case shared-dependency regression
are running independently. No later source changes are made in the isolated
regression snapshot. Repository integration and staged compilation do not
replace independent comparator/trust replay. Acceptance stays 55 complete /
129 pending until all gates pass.


## Fast CE source, real contracts and typed equivalence accepted

All three full JIT bodies now have original/candidate sources, six real
specifications and three whole typed-state/failure equivalences. Real contracts
include successful finite flat execution, all outputs, explicit bounds and
preservation of all other tagged cells. Signed labels retain the live -100
sentinel and int32 cast. Forward requires positive enabled scale/softcap
parameters for its negative-infinity padding; backward allows arbitrary
scaling and any nonzero enabled softcap. Backward uses caller-supplied LSE and
upstream values without a host-composition or calculus claim.

The specifications preserve two source defects. Capped padded logits contribute
exp(-SOFTCAP) to LSE; a complete-source counterexample gives positive loss for a
one-class input. Chunked forward writes both outputs only for chunk zero, and
other chunks preserve all memory; the host's aggregation of unwritten LSE cells
is not certified. These are explicit audit results, not ordinary-cross-entropy
or host-correctness claims.

The libdevice tanh backend is distinct from intrinsic tanh. Half/bfloat scaling
and cast placement, scalar precision inference and signed direct-load casts
are preserved. Candidate equivalence only commutes the integer row/stride
product, with empty domains and no numerical FP atoms. There are 120 original
typed executions with candidate counterparts, plus finite IO witnesses for
ignored labels, untouched chunks, tagged neighbors and negative backward
parameters. Their abstract algebra witnesses consistency, not physical IEEE
rounding. Host scheduling, races and fixed-width overflow remain outside scope.

The final repository closure and all 18 case/frontend regressions pass, as do
all 55 previously accepted cases' 110 public entries. Full correspondence for
all three Python JITs, source/layout gates, 44 configured audit tests and
coverage/manifest checks pass. An extra audit-suite invocation lacked the
comparator PATH and failed before fixture checks; the configured successful
44-test run and final build reconciliation supersede that environment failure.
Independent comparator/trust replay accepts **163 files**,
**761 named declarations** and
**7251 kernel theorem targets**. Every named declaration
is inventoried and all **1932 source/configuration
fingerprints** match the current repository. Receipt: `Logs/comparator-check-gqnownvb`.

Current acceptance is **56 complete / 128 pending**. fast_layernorm is staged
for integration and independent review. attention_kernel_aligned remains
user-deferred, and the bgmv fp16 add_commute admission gap remains open.
No aggregate all-pass claim is made.


## Fast normalization and rotary cases accepted

fast_layernorm, fast_rms_layernorm and fast_rope_embedding pass the complete
source/contract/equivalence review. Their six JITs expose twelve independent real specifications and six typed
FP equivalences: LayerNorm 4+2, RMSNorm
6+3, and RoPE 2+1. Each real contract includes successful finite flat execution,
all outputs, actual bounds and the complete tagged-cell memory frame. All
candidate rewrites commute only integer address products, with empty FP domains
and no numerical FP atoms. Precision and cast placement are retained for every
supported storage combination, as are execution failures.

LayerNorm's variance includes centered padded lanes. A complete N=3/B=4
constant-row witness gives reciprocal standard deviation 1/2 rather than the
ordinary LayerNorm value 1. Backward consumes caller-supplied statistics and
retains the unused bias load in its bounds. RMSNorm regular forward uses the
weight stride and converts normalized X to W storage before multiplication;
Gemma forward/backward use contiguous weights regardless of WR. Backward
retains both GEMMA flags and consumes caller statistics; dW, DWR and epsilon
are unused. Neither backward contract asserts host composition or calculus.

RoPE covers the whole dynamic head loop and both rotation directions. Its
sine/cosine loads precede the loop, including empty groups. Three-iteration
finite witnesses check an odd head size, a partial group and unallocated
masked coefficient tails. A second complete execution aliases both tables
with written Q cells and verifies the initial coefficient snapshot is reused.
Odd tails, prior heads and integer-tagged neighbors remain unchanged.

All three full statement-tree comparisons, the repository proof closure,
six precision/finite execution regression files, source/layout gates and
46 configured audit/tool/layout tests pass. Those tests include the source
auditor's whitespace-before-store repair and pointer/value/dtype negative
regressions. Independent comparator/trust replay accepts **52
files**, **188 named declarations** and
**2017 kernel theorem targets**, with all
**1981 source/configuration fingerprints** matching.
Receipt: `Logs/comparator-check-pq1sfkt8`. Typed execution witnesses use an abstract algebra to
establish satisfiable nonempty executions, without claiming physical IEEE
rounding. Host scheduling, races and fixed-width overflow remain outside scope.

Current acceptance is **59 complete / 125 pending**. fifth_order_sph_harmonics
is in development; attention_kernel_aligned remains user-deferred and the
bgmv fp16 add_commute admission gap remains open. No aggregate all-pass claim
is made.
