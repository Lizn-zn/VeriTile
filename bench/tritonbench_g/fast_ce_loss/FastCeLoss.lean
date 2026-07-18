import VeriTile.Triton

/-!
# `fast_ce_loss` — strict per-kernel correctness

`_cross_entropy_forward` computes, per row program, the row log-sum-exp of the
logits (with optional logit scaling and softcapping), and the cross-entropy loss
`logsumexp − x_label` (or `0` for the ignored `-100` label), storing both. The
`_chunked_cross_entropy_forward` variant computes a per-chunk log-sum-exp and the
partial `-x_label` loss in chunk 0. `_cross_entropy_backward` builds the softmax
gradient `exp(x − logsumexp)`, subtracts one at the label, applies the scaling /
softcap derivative factors, and writes `dloss · y` back in place, masked by
`col_offsets < VOCAB_SIZE`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (the grids `(n_rows,)` /
`(n_rows, cdiv(VOCAB_SIZE, BLOCK_SIZE))`, scheduling, and how the runtime
composes per-program writes / reduces the per-chunk log-sum-exp side outputs) is
the *trusted boundary*. Because the program ids are universally quantified, the
per-program statements cover every program of the grid.

## Proof architecture

```
cross_entropy_forward_correctness                           ← TOP (fwd, ⊨ headline)
  ├─ fastCeForwardIO                                        MetaMasked2DKernelIO₂ₓ₂ signature (.int label slot + gather)
  ├─ cross_entropy_forward_flattenOk                        flat-memory bridge coverage
  ├─ cross_entropy_forward_traceSafe                        per-execution safety walk
  └─ cross_entropy_forward_region_run                       termination + both cell values + frame
       ├─ fceLse_withBot_id / fceLse_withBot_scale          WithBot LSE carrier = some fastCeLseSpec
       └─ fastCeLseSpec_eq_log_sum                          stable form ↦ pure fceLseLocal
fceLseLocal_full_eq_stableLSE / fceLossLocal_eq_crossEntropyLoss  ← canonical-math bridges
chunked_cross_entropy_forward_output_summary                ← TOP (chunked fwd, genuine, chunk 0)
  ├─ (toAlgorithm? = Except.ok _)                           chunked fwd surface lowers
  ├─ chunked_cross_entropy_forward_lse_correct              ← genuine per-chunk LSE store (chunk 0)
  └─ chunked_cross_entropy_forward_loss_correct             ← genuine chunk-0 partial loss = −transform(label)
cross_entropy_backward_surface_toAlgorithm_supported        ← bwd surface lowers
cross_entropy_backward_store_slice_compute_correct          ← masked dloss·y writeback
  └─ cross_entropy_backward_store_slice_correct
```

The forward kernel's headline is a metadata-genre **`⊨` masked Hoare triple**
(`MetaMasked2DKernelIO₂ₓ₂`) under `DO_SOFTCAPPING = false`: the loaded label
is a named `Int` ghost binder pinned to the `.int` slot cell, and every
program writes to `logsumexp_ptr[row]` the pure `fceLseLocal` of the pinned
transformed row values (transform = optional `LOGIT_SCALE * ·`) and to
`loss_ptr[row]` the pure `fceLossLocal` — `0` on the **genuine** `-100`
sentinel branch, else `LSE − transform(gathered label logit)`. The chunked
forward keeps its genuine input-memory value summary (chunk 0); the backward
kernel is proved to *lower* plus the proof-oriented masked `dloss·y` store
slice.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.heuristics` is
not modeled (`DO_SOFTCAPPING`/`DO_LOGIT_SCALING` are plain `Bool` parameters).

**Softcapping is out of scope for the genuine value specs.** A masked
out-of-vocab lane loads as `-inf` (carrier `⊥`); `mul`/`div` propagate `⊥`, but
`tanh(⊥) = -1` in the IEEE-faithful semantics, so under softcapping the masked
lanes become the finite value `SOFTCAP·tanh(-inf) = -SOFTCAP` and are *included*
by the kernel's `tl.max`/`tl.sum` — the masked-reduction lemmas (which need the
transform to send `⊥ ↦ ⊥`) do not apply. The genuine forward/chunked theorems
are therefore stated for `DO_SOFTCAPPING = false`. See `fastCeTransform`.

**Label-channel faithfulness.** The `.py` loads the label with
`tl.load(labels_ptr).to(tl.int32)` and compares it against the live `-100`
ignored-index sentinel — signed semantics. In the **forward** surface the
label load rides the typed `.int` region channel
(`label_idx = tl.load(labels_ptr + row_idx)`; see the surface docstring's
transcription notes), so the sentinel comparison is genuine and the ignore
branch is really modeled. In the **chunked forward** surface the label still
goes through a dynamic pointer register, and in the **backward** surface the
`.to(tl.int32)` dynamic-cast path likewise erases the load to the `Nat`
channel, so in those two the `label_idx != -100` guard is
**dead in the algorithm-layer model** (a `Nat` image can never equal `-100`;
the model always takes the active branch) — a documented modeling boundary of
those two kernels, not of the forward headline. The
`.to(tl.float32)`/`.to(tl.int32)`/`.to(tl.int64)` casts erase to identity at
the algorithm layer. The forward logits load uses
`other = -float("inf")` for out-of-vocab lanes. The masked backward store leaves
inactive lanes (`col_offsets ≥ VOCAB_SIZE`) untouched and assumes the per-tile
output offset is injective. The spec is built inline; it does not reference a
`VeriTile.Triton.Math.*` oracle.
-/

namespace VeriTile.Bench.TritonBenchG.FastCeLoss

open VeriTile.Triton
open VeriTile.Triton.TiledLogSumExp

set_option linter.unusedSimpArgs false
set_option maxHeartbeats 800000

/-- Faithful transcription of `fast_ce_loss.py`'s `_cross_entropy_forward`.

Python's hard-coded `label_idx != -100` sentinel is preserved as the literal
`-100`, and the label load rides the typed `.int` region channel so the
sentinel comparison is a genuine signed comparison. Two label-channel
transcription notes: the `.py`'s `labels_ptr += row_idx` pointer bump is
folded into the load offset (`tl.load(labels_ptr + row_idx)` — same address,
same single load) because a dynamic pointer register erases the region's
element dtype, and the `.to(tl.int32)` cast (identity on the already-int32
channel) is dropped for the same reason. -/
def cross_entropy_forward_surface
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
  loss_ptr += row_idx
  logsumexp_ptr += row_idx
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr + row_idx)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $(LOGIT_SCALE) * logits
  }
  if DO_SOFTCAPPING {
    logits = $(SOFTCAP) * triton_tanh(logits / $(SOFTCAP))
  }
  logits = (logits).to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if label_idx != $((-100 : Int)) {
    x = tl.load(logits_ptr + label_idx)
    if DO_LOGIT_SCALING {
      x = $(LOGIT_SCALE) * x
    }
    if DO_SOFTCAPPING {
      x = $(SOFTCAP) * triton_tanh(x / $(SOFTCAP))
    }
    loss = logsumexp - (x).to(tl.float32)
  } else {
    loss = 0.0
  }
  tl.store(logsumexp_ptr, logsumexp)
  tl.store(loss_ptr, loss)
}

/-- The faithful full forward surface lowers to the algorithm layer, including
optional logit scaling and softcapping branches. -/
theorem cross_entropy_forward_surface_toAlgorithm_supported
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ∃ alg,
      (cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride BLOCK_SIZE SOFTCAP LOGIT_SCALE
        DO_SOFTCAPPING DO_LOGIT_SCALING).toAlgorithm? = Except.ok alg := by
  simp [cross_entropy_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Shared per-lane transform and masked-LSE carrier lemmas

(Consumed by both the forward `⊨` headline's region run and the chunked
summary's value lemmas.)

The forward kernel applies a per-lane MAP to the loaded logits before the
max/sum reduction: optionally `LOGIT_SCALE * x` (logit scaling), then optionally
`SOFTCAP * tanh(x / SOFTCAP)` (softcapping). We capture this as
`fastCeTransform`.

**Softcapping caveat (modeling fact, not a proof gap).** A masked out-of-vocab
lane is loaded as `-inf` (carrier `⊥`). Scaling and division propagate `⊥`
(`Option.map₂`), but `tanh(⊥) = some (-1)` in the IEEE-faithful semantics
(`WithBot.realTanh ⊥ = -1`). Hence under `DO_SOFTCAPPING = true` the masked
lanes are *not* `⊥` after the transform — they become `SOFTCAP * tanh(-inf) =
-SOFTCAP`, a finite value that the kernel's `tl.max`/`tl.sum` then *include*.
The masked-reduction lemmas (`sup'_masked_map_eq` / `sum_exp_masked_map_eq`)
require the transform to send `⊥ ↦ ⊥`, which the softcap branch violates. The
genuine value-correctness theorems below are therefore stated for
`DO_SOFTCAPPING = false`, where the transform is `mul`/`id` only and *does*
propagate `⊥`. -/

/-- The per-lane logit transform `fast_ce_loss.py` applies before the reduction:
optional `LOGIT_SCALE * x`, then optional `SOFTCAP * tanh(x / SOFTCAP)`. -/
noncomputable def fastCeTransform
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) (x : ℝ) : ℝ :=
  let scaled := if DO_LOGIT_SCALING then LOGIT_SCALE * x else x
  if DO_SOFTCAPPING then SOFTCAP * Real.tanh (scaled / SOFTCAP) else scaled

/-- Row-logits function for the single-program forward kernel: lane `i` reads
INPUT memory `logits_ptr` at `row_idx * logits_row_stride + i`. -/
noncomputable def fastCeRowLogits
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride VOCAB_SIZE : Nat) (j : Fin VOCAB_SIZE) : ℝ :=
  s.readMem logits_ptr (s.pids 0 * logits_row_stride + j.val)

/-- Genuine stable log-sum-exp of the transformed masked logits for the tail
block `i_d` (block size `n+1`): `m + log(∑ exp(transform(raw) - m))` over the
valid lanes, where `m` is the max over valid lanes. Mirrors `partialLSE_full`
but with an arbitrary per-lane transform `g` (here the `mul`/`id` part of
`fastCeTransform`). -/
noncomputable def fastCeLseSpec
    {VOCAB_SIZE : Nat} (xs : Fin VOCAB_SIZE → ℝ) {n : Nat} (i_d : Nat)
    (h_tail : i_d * (n+1) < VOCAB_SIZE)
    (g : ℝ → ℝ) : ℝ :=
  let vl := validLanes n VOCAB_SIZE i_d
  let lane : Fin (n+1) → ℝ := fun i =>
    if h : i_d * (n+1) + i.val < VOCAB_SIZE then g (xs ⟨i_d * (n+1) + i.val, h⟩) else 0
  let m := vl.sup' (validLanes_nonempty h_tail) lane
  m + Real.log (∑ i ∈ vl, Real.exp (lane i - m))

/-- Output offset for the single-program forward kernel: `logsumexp_ptr` and
`loss_ptr` are both indexed at `row_idx = pid`. -/
def fceOutOffset (s : BlockState) : Nat := s.pids 0

/-- The label value loaded by the **chunked** forward kernel (whose label
load still goes through a dynamic pointer register and is therefore carried
as a `Nat` offset under the algorithm-layer cast erasure — see the modeling
boundary note; the non-chunked forward now rides the `.int` channel). -/
noncomputable def fceLabelNat (s : BlockState) (labels_ptr : Region .int) : Nat :=
  s.readMemValue .nat (Region.cast labels_ptr) (s.pids 0)

/-- The transformed label logit `transform(tl.load(logits_ptr + label_idx))`
read from INPUT memory at the data-dependent (Nat) label position — the
chunked kernel's gather value. -/
noncomputable def fceLabelLogit
    (s : BlockState) (logits_ptr : RegionName) (labels_ptr : Region .int)
    (logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) : ℝ :=
  fastCeTransform SOFTCAP LOGIT_SCALE DO_SOFTCAPPING DO_LOGIT_SCALING
    (s.readMem logits_ptr (s.pids 0 * logits_row_stride + fceLabelNat s labels_ptr))

/-- The raw `WithBot ℝ` stable-LSE carrier produced by the forward surface for
the *no-scaling* lane carrier (`if i<V then some(rm i) else none`) equals
`some (fastCeLseSpec … id)`. Shared by the LSE and loss stores. -/
theorem fceLse_withBot_id
    (n : Nat) (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride VOCAB_SIZE : Nat)
    (rm : Fin (n+1) → ℝ) (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty)
    (h_filter : (validLanes n VOCAB_SIZE 0).Nonempty)
    (hrm : rm = fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val)) :
    Option.map₂ (fun a b => a + b)
        (@Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
          (fun x => if (↑x : Nat) < VOCAB_SIZE then ((some (rm x)) : WithBot ℝ) else none))
        (WithBot.realLog
          (∑ x : Fin (n+1), WithBot.realExp
            (Option.map₂ (fun a b => a - b)
              (if (↑x : Nat) < VOCAB_SIZE then ((some (rm x)) : WithBot ℝ) else none)
              (@Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
                (fun x => if (↑x : Nat) < VOCAB_SIZE then ((some (rm x)) : WithBot ℝ) else none)))))
      = some (fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => x)) := by
  have hcond : ∀ i : Fin (n+1),
      (↑i < VOCAB_SIZE) = (0 * (n+1) + ↑i < VOCAB_SIZE) := by
    intro i; rw [Nat.zero_mul, Nat.zero_add]
  simp only [hcond]
  erw [sup'_masked_eq h_ne h_filter rm, sum_exp_masked_eq rm]
  simp only [WithBot.realLog_coe]
  erw [Option.map₂_coe_coe]
  unfold fastCeLseSpec
  simp only [Bool.false_eq_true, reduceIte]
  have h_lane : ∀ i ∈ validLanes n VOCAB_SIZE 0, rm i =
      (if h : 0 * (n+1) + i.val < VOCAB_SIZE then
        fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE ⟨0 * (n+1) + i.val, h⟩ else 0) := by
    intro i hi
    simp only [validLanes, Finset.mem_filter, Finset.mem_univ, true_and,
      Nat.zero_mul, Nat.zero_add] at hi
    rw [dif_pos (by rw [Nat.zero_mul, Nat.zero_add]; exact hi)]
    simp only [hrm, fastCeRowLogits, Nat.zero_mul, Nat.zero_add]
  rw [Finset.sup'_congr h_filter rfl h_lane]
  congr 2
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  rw [h_lane i hi]

/-- The raw `WithBot ℝ` stable-LSE carrier for the *scaled* lane carrier
(`Option.map (LOGIT_SCALE * ·) (if i<V then some(rm i) else none)`) equals
`some (fastCeLseSpec … (LOGIT_SCALE * ·))`. -/
theorem fceLse_withBot_scale
    (n : Nat) (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride VOCAB_SIZE : Nat) (LOGIT_SCALE : ℝ)
    (rm : Fin (n+1) → ℝ) (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty)
    (h_filter : (validLanes n VOCAB_SIZE 0).Nonempty)
    (hrm : rm = fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val)) :
    Option.map₂ (fun a b => a + b)
        (@Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
          (fun x => Option.map (fun b => LOGIT_SCALE * b)
            (if (↑x : Nat) < VOCAB_SIZE then ((some (rm x)) : WithBot ℝ) else none)))
        (WithBot.realLog
          (∑ x : Fin (n+1), WithBot.realExp
            (Option.map₂ (fun a b => a - b)
              (Option.map (fun b => LOGIT_SCALE * b)
                (if (↑x : Nat) < VOCAB_SIZE then ((some (rm x)) : WithBot ℝ) else none))
              (@Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
                (fun x => Option.map (fun b => LOGIT_SCALE * b)
                  (if (↑x : Nat) < VOCAB_SIZE then ((some (rm x)) : WithBot ℝ) else none))))))
      = some (fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => LOGIT_SCALE * x)) := by
  have hcond : ∀ i : Fin (n+1),
      (↑i < VOCAB_SIZE) = (0 * (n+1) + ↑i < VOCAB_SIZE) := by
    intro i; rw [Nat.zero_mul, Nat.zero_add]
  simp only [hcond]
  erw [sup'_masked_map_eq h_ne h_filter rm (LOGIT_SCALE * ·),
       sum_exp_masked_map_eq rm (LOGIT_SCALE * ·)]
  simp only [WithBot.realLog_coe]
  erw [Option.map₂_coe_coe]
  unfold fastCeLseSpec
  simp only [reduceIte]
  have h_lane : ∀ i ∈ validLanes n VOCAB_SIZE 0, LOGIT_SCALE * rm i =
      (if h : 0 * (n+1) + i.val < VOCAB_SIZE then
        LOGIT_SCALE * fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE ⟨0 * (n+1) + i.val, h⟩ else 0) := by
    intro i hi
    simp only [validLanes, Finset.mem_filter, Finset.mem_univ, true_and,
      Nat.zero_mul, Nat.zero_add] at hi
    rw [dif_pos (by rw [Nat.zero_mul, Nat.zero_add]; exact hi)]
    simp only [hrm, fastCeRowLogits, Nat.zero_mul, Nat.zero_add]
  rw [Finset.sup'_congr h_filter rfl h_lane]
  congr 2
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  rw [h_lane i hi]

/-! ## The `⊨` specification (forward)

The forward headline states `_cross_entropy_forward` on the metadata-genre IO
skin `MetaMasked2DKernelIO₂ₓ₂`: the loaded label is a named `Int` ghost binder
pinned to the `labels_ptr` slot cell on the `.int` channel, the masked logits
row and the sentinel-gated label gather cell are the two data inputs, and the
per-row `logsumexp`/`loss` cells are the two 1-lane outputs. The pure specs
below are built only from the pinned values — no `BlockState` reads. -/

/-- Pure per-program value of the `logsumexp` output cell: the plain
log-sum-exp `log (∑ exp)` over the valid lanes (`j < VOCAB_SIZE`) of the
per-lane transformed block values (transform = optional `LOGIT_SCALE * ·`,
the `DO_SOFTCAPPING = false` regime). The kernel's max-shifted stable form
collapses to it via `fastCeLseSpec_eq_log_sum`. Built only from the pinned
block values `xs`. -/
noncomputable def fceLseLocal (VOCAB_SIZE B : Nat) (LOGIT_SCALE : ℝ)
    (DO_LOGIT_SCALING : Bool) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j ∈ Finset.univ.filter (fun j : Fin B => j.val < VOCAB_SIZE),
    Real.exp (if DO_LOGIT_SCALING then LOGIT_SCALE * xs j else xs j))

/-- Pure per-program value of the `loss` output cell: `0` for the ignored
`-100` sentinel label, otherwise `LSE − transform(g)` with `g` the gathered
label-logit cell. Built only from the pinned label, block values, and gather
cell. -/
noncomputable def fceLossLocal (VOCAB_SIZE B : Nat) (LOGIT_SCALE : ℝ)
    (DO_LOGIT_SCALING : Bool) (lab : Int) (xs : Fin B → ℝ) (g : ℝ) : ℝ :=
  if lab = -100 then 0
  else fceLseLocal VOCAB_SIZE B LOGIT_SCALE DO_LOGIT_SCALING xs
    - (if DO_LOGIT_SCALING then LOGIT_SCALE * g else g)

/-- The kernel's stable (max-shifted) masked LSE of any row function agreeing
with the pinned block values `xs` on the valid lanes equals the plain pure
`log (∑ exp)` form, for an arbitrary per-lane transform `t` (instantiated at
`id` and `LOGIT_SCALE * ·`). Routes through the library's
`partialLSE_full_eq_blockLSE`. -/
private theorem fastCeLseSpec_eq_log_sum
    {VOCAB_SIZE n : Nat} (xsRow : Fin VOCAB_SIZE → ℝ) (t : ℝ → ℝ)
    (h_tail : 0 * (n+1) < VOCAB_SIZE) (xs : Fin (n+1) → ℝ)
    (hx : ∀ (j : Fin (n+1)) (hj : j.val < VOCAB_SIZE), xsRow ⟨j.val, hj⟩ = xs j) :
    fastCeLseSpec xsRow 0 h_tail t
      = Real.log (∑ j ∈ Finset.univ.filter (fun j : Fin (n+1) => j.val < VOCAB_SIZE),
          Real.exp (t (xs j))) := by
  have h1 : fastCeLseSpec xsRow 0 h_tail t
      = partialLSE_full (n := n) (fun v => t (xsRow v)) 0 h_tail Bool.false 0 := by
    rfl
  rw [h1, partialLSE_full_eq_blockLSE]
  unfold blockLSE
  congr 1
  refine Finset.sum_congr ?_ ?_
  · unfold validLanes
    ext i
    simp
  · intro i hi
    have hii : i.val < VOCAB_SIZE := by
      have h := Finset.mem_filter.mp hi
      simpa using h.2
    unfold scaledLane_full
    simp only [Bool.false_eq_true, reduceIte, Nat.zero_mul, Nat.zero_add]
    rw [dif_pos hii]
    exact congrArg Real.exp (congrArg t (hx i hii))

/-- The forward kernel sits inside the flat-memory bridge's covered fragment
(typed `.int` slot load, pointer arithmetic, masked load with `other`, casts,
reductions, the sentinel branch with its gather load, two scalar stores). -/
theorem cross_entropy_forward_flattenOk
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ((cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
      VOCAB_SIZE logits_row_stride BLOCK_SIZE SOFTCAP LOGIT_SCALE
      DO_SOFTCAPPING DO_LOGIT_SCALING).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [cross_entropy_forward_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 3200000 in
/-- **The region-model masked Hoare triple** — termination, both output cells'
values (the pure `fceLseLocal`/`fceLossLocal` of the pinned label `lab`, block
values `xs`, and gather cell `g`), and the cell frame off the two written
cells. This is the `hrun` obligation of the `⊨` headline. One symbolic
execution walk per `DO_LOGIT_SCALING` × sentinel case; the raw `WithBot` LSE
carriers are folded by `fceLse_withBot_id`/`fceLse_withBot_scale` and
transported to the pure spec by `fastCeLseSpec_eq_log_sum`. `0 < VOCAB_SIZE`
feeds the max-reduce's nonempty valid-lane set; `logsumexp_ptr ≠ loss_ptr`
lets the LSE readback see through the trailing loss store (the two stores are
the kernel's last statements, so no other distinctness is needed). -/
theorem cross_entropy_forward_region_run
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride : Nat) (n : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (hV : 0 < VOCAB_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr)
    (s₀ : BlockState) (lab : Int) (xs : Fin (n+1) → ℝ) (g : ℝ)
    (hlab : s₀.readMemValue .int (Region.cast labels_ptr) (s₀.pids 0) = lab)
    (hx : ∀ j : Fin (n+1), j.val < VOCAB_SIZE →
      s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + j.val) = xs j)
    (hg : lab ≠ -100 →
      s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + lab.toNat) = g) :
    ∃ s1,
      exec ((cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false
        DO_LOGIT_SCALING).toAlgKernel) s₀ = some s1
      ∧ s1.readMem logsumexp_ptr (s₀.pids 0)
          = fceLseLocal VOCAB_SIZE (n+1) LOGIT_SCALE DO_LOGIT_SCALING xs
      ∧ s1.readMem loss_ptr (s₀.pids 0)
          = fceLossLocal VOCAB_SIZE (n+1) LOGIT_SCALE DO_LOGIT_SCALING lab xs g
      ∧ (∀ r o, (r ≠ logsumexp_ptr ∨ o ≠ s₀.pids 0) →
          (r ≠ loss_ptr ∨ o ≠ s₀.pids 0) →
          s1.mem r o = s₀.mem r o) := by
  have h_tail : 0 * (n+1) < VOCAB_SIZE := by simpa using hV
  have h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have h_filter : (validLanes n VOCAB_SIZE 0).Nonempty := validLanes_nonempty h_tail
  cases hDLS : DO_LOGIT_SCALING
  · by_cases hIgn : lab = -100
    · cases hstep : exec ((cross_entropy_forward_surface logits_ptr loss_ptr
          logsumexp_ptr labels_ptr VOCAB_SIZE logits_row_stride (n+1) SOFTCAP
          LOGIT_SCALE Bool.false Bool.false).toAlgKernel) s₀ with
      | none =>
          exfalso
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
      | some s1 =>
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
          subst hstep
          refine ⟨_, rfl, ?_, ?_, ?_⟩
          · -- logsumexp readback
            rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            erw [fceLse_withBot_id n s₀ logits_ptr logits_row_stride VOCAB_SIZE
              (fun i => s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + i.val))
              h_tail h_ne h_filter rfl, WithBot.unbotD_coe]
            rw [fastCeLseSpec_eq_log_sum
              (fastCeRowLogits s₀ logits_ptr logits_row_stride VOCAB_SIZE)
              (fun x => x) h_tail xs (fun j hj => hx j hj)]
            simp [fceLseLocal]
          · -- loss readback
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            simp [fceLossLocal, hIgn]
            norm_num
          · -- frame
            intro r o hr1 hr2
            rw [BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr2 with h | h
                · exact h hc.1
                · exact h hc.2),
              BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr1 with h | h
                · exact h hc.1
                · exact h hc.2)]
            rfl
    · cases hstep : exec ((cross_entropy_forward_surface logits_ptr loss_ptr
          logsumexp_ptr labels_ptr VOCAB_SIZE logits_row_stride (n+1) SOFTCAP
          LOGIT_SCALE Bool.false Bool.false).toAlgKernel) s₀ with
      | none =>
          exfalso
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
      | some s1 =>
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
          subst hstep
          refine ⟨_, rfl, ?_, ?_, ?_⟩
          · -- logsumexp readback
            rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            erw [fceLse_withBot_id n s₀ logits_ptr logits_row_stride VOCAB_SIZE
              (fun i => s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + i.val))
              h_tail h_ne h_filter rfl, WithBot.unbotD_coe]
            rw [fastCeLseSpec_eq_log_sum
              (fastCeRowLogits s₀ logits_ptr logits_row_stride VOCAB_SIZE)
              (fun x => x) h_tail xs (fun j hj => hx j hj)]
            simp [fceLseLocal]
          · -- loss readback
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            rw [hg hIgn]
            erw [fceLse_withBot_id n s₀ logits_ptr logits_row_stride VOCAB_SIZE
              (fun i => s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + i.val))
              h_tail h_ne h_filter rfl]
            rw [fastCeLseSpec_eq_log_sum
              (fastCeRowLogits s₀ logits_ptr logits_row_stride VOCAB_SIZE)
              (fun x => x) h_tail xs (fun j hj => hx j hj)]
            simp [fceLossLocal, fceLseLocal, hIgn]
          · -- frame
            intro r o hr1 hr2
            rw [BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr2 with h | h
                · exact h hc.1
                · exact h hc.2),
              BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr1 with h | h
                · exact h hc.1
                · exact h hc.2)]
            rfl
  · by_cases hIgn : lab = -100
    · cases hstep : exec ((cross_entropy_forward_surface logits_ptr loss_ptr
          logsumexp_ptr labels_ptr VOCAB_SIZE logits_row_stride (n+1) SOFTCAP
          LOGIT_SCALE Bool.false Bool.true).toAlgKernel) s₀ with
      | none =>
          exfalso
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
      | some s1 =>
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
          subst hstep
          refine ⟨_, rfl, ?_, ?_, ?_⟩
          · -- logsumexp readback
            rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            erw [fceLse_withBot_scale n s₀ logits_ptr logits_row_stride VOCAB_SIZE
              LOGIT_SCALE
              (fun i => s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + i.val))
              h_tail h_ne h_filter rfl, WithBot.unbotD_coe]
            rw [fastCeLseSpec_eq_log_sum
              (fastCeRowLogits s₀ logits_ptr logits_row_stride VOCAB_SIZE)
              (fun x => LOGIT_SCALE * x) h_tail xs (fun j hj => hx j hj)]
            simp [fceLseLocal]
          · -- loss readback
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            simp [fceLossLocal, hIgn]
            norm_num
          · -- frame
            intro r o hr1 hr2
            rw [BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr2 with h | h
                · exact h hc.1
                · exact h hc.2),
              BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr1 with h | h
                · exact h hc.1
                · exact h hc.2)]
            rfl
    · cases hstep : exec ((cross_entropy_forward_surface logits_ptr loss_ptr
          logsumexp_ptr labels_ptr VOCAB_SIZE logits_row_stride (n+1) SOFTCAP
          LOGIT_SCALE Bool.false Bool.true).toAlgKernel) s₀ with
      | none =>
          exfalso
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
      | some s1 =>
          simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
            evalOp.eq_def,
            Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast, Tile.ptrAdd,
            Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComparableDType.ne, hlab, hIgn] at hstep
          subst hstep
          refine ⟨_, rfl, ?_, ?_, ?_⟩
          · -- logsumexp readback
            rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            erw [fceLse_withBot_scale n s₀ logits_ptr logits_row_stride VOCAB_SIZE
              LOGIT_SCALE
              (fun i => s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + i.val))
              h_tail h_ne h_filter rfl, WithBot.unbotD_coe]
            rw [fastCeLseSpec_eq_log_sum
              (fastCeRowLogits s₀ logits_ptr logits_row_stride VOCAB_SIZE)
              (fun x => LOGIT_SCALE * x) h_tail xs (fun j hj => hx j hj)]
            simp [fceLseLocal]
          · -- loss readback
            rw [BlockState.writeMem_readMem, if_pos ⟨rfl, rfl⟩]
            rw [hg hIgn]
            erw [fceLse_withBot_scale n s₀ logits_ptr logits_row_stride VOCAB_SIZE
              LOGIT_SCALE
              (fun i => s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + i.val))
              h_tail h_ne h_filter rfl]
            rw [fastCeLseSpec_eq_log_sum
              (fastCeRowLogits s₀ logits_ptr logits_row_stride VOCAB_SIZE)
              (fun x => LOGIT_SCALE * x) h_tail xs (fun j hj => hx j hj)]
            simp [fceLossLocal, fceLseLocal, hIgn]
          · -- frame
            intro r o hr1 hr2
            rw [BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr2 with h | h
                · exact h hc.1
                · exact h hc.2),
              BlockState.writeMem_mem, if_neg (fun hc => by
                rcases hr1 with h | h
                · exact h hc.1
                · exact h hc.2)]
            rfl

set_option maxHeartbeats 3200000 in
/-- Per-execution safety walk: one computational unfold per
`DO_LOGIT_SCALING` × sentinel case reduces the kernel's memory accesses to the
skin-shaped bounds hypotheses — the `.int` label slot cell (`labels_ptr[pid₀]`),
the active row lanes (`j < VOCAB_SIZE`), the sentinel-gated gather cell
(`lab ≠ -100`), and the two unconditional per-row output cells. -/
theorem cross_entropy_forward_traceSafe
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride : Nat) (n : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (bounds : RegionBounds) (s : BlockState) (lab : Int)
    (hlab : s.readMemValue .int (Region.cast labels_ptr) (s.pids 0) = lab)
    (hbL : s.pids 0 < bounds (Region.cast labels_ptr))
    (hbr : ∀ j : Fin (n+1), j.val < VOCAB_SIZE →
      s.pids 0 * logits_row_stride + j.val < bounds logits_ptr)
    (hbg : lab ≠ -100 →
      s.pids 0 * logits_row_stride + lab.toNat < bounds logits_ptr)
    (hbw1 : s.pids 0 < bounds logsumexp_ptr)
    (hbw2 : s.pids 0 < bounds loss_ptr) :
    Kernel.TraceSafe bounds
      ((cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false
        DO_LOGIT_SCALING).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  cases hDLS : DO_LOGIT_SCALING <;> by_cases hIgn : lab = -100
  · simp (maxSteps := 16000000) [cross_entropy_forward_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
        MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
        MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
        tile_elementwise, Bool.and_eq_true,
        Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, ComparableDType.ne,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        FloatDType.cast, TileShape.allIndices,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hlab, hIgn, hbL, hbw1, hbw2]
    exact fun a ha => hbr a ha
  · have hbg' := hbg hIgn
    simp (maxSteps := 16000000) [cross_entropy_forward_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
        MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
        MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
        tile_elementwise, Bool.and_eq_true,
        Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, ComparableDType.ne,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        FloatDType.cast, TileShape.allIndices,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hlab, hIgn, hbL, hbw1, hbw2, hbg']
    exact fun a ha => hbr a ha
  · simp (maxSteps := 16000000) [cross_entropy_forward_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
        MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
        MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
        tile_elementwise, Bool.and_eq_true,
        Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, ComparableDType.ne,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        FloatDType.cast, TileShape.allIndices,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hlab, hIgn, hbL, hbw1, hbw2]
    exact fun a ha => hbr a ha
  · have hbg' := hbg hIgn
    simp (maxSteps := 16000000) [cross_entropy_forward_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
        MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
        MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
        tile_elementwise, Bool.and_eq_true,
        Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, ComparableDType.ne,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        FloatDType.cast, TileShape.allIndices,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hlab, hIgn, hbL, hbw1, hbw2, hbg']
    exact fun a ha => hbr a ha

/-- `cross_entropy_forward_surface`'s metadata-genre **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`MetaMasked2DKernelIO₂ₓ₂`, the cross-entropy metadata shape):

* `mbufL` — the `.int` label slot: program `row_idx = pid₀` loads
  `labels_ptr[pid₀]` (`mwinL`), yielding the named `Int` scalar `lab`;
* `inp = logits_ptr` — read twice: the masked row block (`read`/`mask`: lane
  `j` at `pid₀·stride + j`, active while `j < VOCAB_SIZE`) and the
  sentinel-gated single-cell gather (`gwin`/`gmask`: cell
  `pid₀·stride + lab.toNat`, read exactly when `lab ≠ -100`);
* `out1 = logsumexp_ptr`, `out2 = loss_ptr` — the two per-row cells, both at
  offset `pid₀`, written unconditionally (`writeMask` defaults).

`DO_SOFTCAPPING` is pinned `false` (see the softcap modeling note above);
`DO_LOGIT_SCALING` stays a spec parameter. The slot cell, windows, gates, and
masks are declared, not parsed from the kernel; the headline **proves** the
kernel's actual slot load, addressing, gating, and masking match them. -/
def fastCeForwardIO
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool) :
    MetaMasked2DKernelIO₂ₓ₂ where
  kernel := cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
    labels_ptr VOCAB_SIZE logits_row_stride BLOCK_SIZE SOFTCAP LOGIT_SCALE
    Bool.false DO_LOGIT_SCALING
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := logsumexp_ptr
  out2 := loss_ptr
  B := BLOCK_SIZE
  mwinL := fun pid₀ _ => pid₀
  read := fun pid₀ _ _ j => pid₀ * logits_row_stride + j.val
  mask := fun _ _ _ j => j.val < VOCAB_SIZE
  gwin := fun pid₀ _ lab => pid₀ * logits_row_stride + lab.toNat
  gmask := fun _ _ lab => lab ≠ -100
  write1 := fun pid₀ _ _ => pid₀
  write2 := fun pid₀ _ _ => pid₀

open scoped VeriTile.Triton.MetaMasked2DKernelIO₂ₓ₂ in
/-- **The headline**: `_cross_entropy_forward` implements the pure per-row
cross-entropy pair on its metadata-genre IO signature — for every disjoint
flat placement of the four buffers, every program `row_idx = pid₀` whose
declared cells/lanes are in bounds, and every launch state pinning the label
`lab` at the `.int` slot cell, the block logits `xs` on the active lanes, and
the gather cell `g` under its `lab ≠ -100` gate, the translated pointer kernel
terminates and writes

* `logsumexp_ptr[pid₀] = fceLseLocal … xs` — the log-sum-exp of the valid
  transformed lanes, and
* `loss_ptr[pid₀] = fceLossLocal … lab xs g` — `0` for the ignored `-100`
  label, else `LSE − transform(g)`,

and every other memory cell is unchanged. The `-100` sentinel branch is
**genuine**: the label rides the `.int` channel, so an ignored row really
takes the `loss = 0` path. `0 < VOCAB_SIZE` (at least one valid lane) and
`0 < BLOCK_SIZE` feed the `max` reduce; `logsumexp_ptr ≠ loss_ptr` is the one
output-distinctness side condition (matching the old summary's `hne`).
`DO_SOFTCAPPING = false` is pinned (softcap breaks `⊥`-propagation on masked
lanes; see `fastCeTransform`); `DO_LOGIT_SCALING` parametrizes the spec.
Proof: `MetaMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
specification cross_entropy_forward_correctness
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (hV : 0 < VOCAB_SIZE) (hB : 0 < BLOCK_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr) :
    fastCeForwardIO logits_ptr loss_ptr logsumexp_ptr labels_ptr VOCAB_SIZE
        logits_row_stride BLOCK_SIZE SOFTCAP LOGIT_SCALE DO_LOGIT_SCALING ⊨
      fun _ _ lab xs g =>
        (fceLseLocal VOCAB_SIZE BLOCK_SIZE LOGIT_SCALE DO_LOGIT_SCALING xs,
         fceLossLocal VOCAB_SIZE BLOCK_SIZE LOGIT_SCALE DO_LOGIT_SCALING
           lab xs g) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  refine MetaMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact cross_entropy_forward_flattenOk logits_ptr loss_ptr logsumexp_ptr
      labels_ptr VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
      Bool.false DO_LOGIT_SCALING
  · intro bounds s lab hlab hbL hbr hbg hbw1 hbw2
    have hlab' : s.readMemValue .int (Region.cast labels_ptr) (s.pids 0)
        = lab := hlab
    have hbL' : s.pids 0 < bounds (Region.cast labels_ptr) := hbL
    have hbr' : ∀ j : Fin (n+1), j.val < VOCAB_SIZE →
        s.pids 0 * logits_row_stride + j.val < bounds logits_ptr := hbr
    have hbg' : lab ≠ -100 →
        s.pids 0 * logits_row_stride + lab.toNat < bounds logits_ptr := hbg
    exact cross_entropy_forward_traceSafe logits_ptr loss_ptr logsumexp_ptr
      labels_ptr VOCAB_SIZE logits_row_stride n SOFTCAP LOGIT_SCALE
      DO_LOGIT_SCALING bounds s lab hlab' hbL' hbr' hbg'
      (hbw1 trivial) (hbw2 trivial)
  · intro s₀ lab xs g hlab hx hg
    have hlab' : s₀.readMemValue .int (Region.cast labels_ptr) (s₀.pids 0)
        = lab := hlab
    have hx' : ∀ j : Fin (n+1), j.val < VOCAB_SIZE →
        s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + j.val)
          = xs j := hx
    have hg' : lab ≠ -100 →
        s₀.readMem logits_ptr (s₀.pids 0 * logits_row_stride + lab.toNat)
          = g := hg
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
      cross_entropy_forward_region_run logits_ptr loss_ptr logsumexp_ptr
        labels_ptr VOCAB_SIZE logits_row_stride n SOFTCAP LOGIT_SCALE
        DO_LOGIT_SCALING hV hne s₀ lab xs g hlab' hx' hg'
    refine ⟨s1, hexec, fun _ => hval1, fun _ => hval2,
      fun r o h1 h2 => ?_⟩
    refine hframe r o ?_ ?_
    · rcases h1 with hner | hno
      · exact Or.inl hner
      · exact Or.inr (hno trivial)
    · rcases h2 with hner | hno
      · exact Or.inl hner
      · exact Or.inr (hno trivial)

open VeriTile.Triton.TiledLoss in
/-- Full-vocab, no-transform regime (`BLOCK_SIZE = VOCAB_SIZE = n+1`, no
scaling): the headline's pure LSE spec is exactly the canonical stable row
log-sum-exp. -/
theorem fceLseLocal_full_eq_stableLSE
    (n : Nat) (xs : Fin (n+1) → ℝ) (LOGIT_SCALE : ℝ) :
    fceLseLocal (n+1) (n+1) LOGIT_SCALE Bool.false xs
      = stableLSE xs (Nat.succ_pos n) Bool.false 0 := by
  rw [stableLSE_eq_LSE]
  unfold fceLseLocal LSE
  rw [Finset.filter_true_of_mem (fun i _ => i.isLt)]
  simp

open VeriTile.Triton.TiledLoss in
/-- **Bridge: textbook cross-entropy.** In the full-vocab, no-transform regime,
for an in-range label `lbl : Fin (n+1)` (whose `Int` image can never be the
`-100` sentinel) and the gather cell holding the label logit `xs lbl`, the
headline's pure loss spec is the canonical pure `crossEntropyLoss` of the row
logits at the target class. -/
theorem fceLossLocal_eq_crossEntropyLoss
    (n : Nat) (xs : Fin (n+1) → ℝ) (LOGIT_SCALE : ℝ) (lbl : Fin (n+1)) :
    fceLossLocal (n+1) (n+1) LOGIT_SCALE Bool.false (lbl.val : Int) xs (xs lbl)
      = crossEntropyLoss xs lbl (Nat.succ_pos n) := by
  unfold fceLossLocal crossEntropyLoss
  rw [if_neg (by omega), fceLseLocal_full_eq_stableLSE]
  simp

/-- Surface transcription of `fast_ce_loss.py`'s
`_chunked_cross_entropy_forward`.

Python's hard-coded `label_idx != -100` sentinel is preserved as the literal
`-100`. -/
def chunked_cross_entropy_forward_surface
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  chunk_idx = tl.program_id(1)
  logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
  loss_ptr += row_idx
  logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
  labels_ptr += row_idx
  col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = (tl.load(labels_ptr)).to(tl.int32)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $(LOGIT_SCALE) * logits
  }
  if DO_SOFTCAPPING {
    logits = $(SOFTCAP) * triton_tanh(logits / $(SOFTCAP))
  }
  logits = (logits).to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if chunk_idx == 0 {
    if label_idx != $((-100 : Int)) {
      x = (tl.load(logits_ptr + label_idx)).to(tl.float32)
      if DO_LOGIT_SCALING {
        x = $(LOGIT_SCALE) * x
      }
      if DO_SOFTCAPPING {
        x = $(SOFTCAP) * triton_tanh(x / $(SOFTCAP))
      }
      loss = -1.0 * (x).to(tl.float32)
    } else {
      loss = 0.0
    }
    tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)
  }
}

/-- The faithful chunked forward surface lowers to the algorithm layer,
including optional logit scaling and softcapping branches. -/
theorem chunked_cross_entropy_forward_surface_toAlgorithm_supported
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ∃ alg,
      (chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
        labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE SOFTCAP
        LOGIT_SCALE DO_SOFTCAPPING DO_LOGIT_SCALING).toAlgorithm? =
        Except.ok alg := by
  simp [chunked_cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Chunked logsumexp output offset `row_idx * N_CHUNKS + chunk_idx`. -/
def fceChunkLseOffset (s : BlockState) (N_CHUNKS : Nat) : Nat :=
  s.pids 0 * N_CHUNKS + s.pids 1

/-- **Genuine chunked forward LSE correctness (chunk 0, no softcapping).** The
chunked surface stores both side outputs only under `chunk_idx == 0`. For chunk
`0` with `DO_SOFTCAPPING = false`, executing the surface writes to
`logsumexp_ptr[row_idx * N_CHUNKS + 0]` exactly the masked-lane stable
log-sum-exp `fastCeLseSpec` of the per-lane transformed chunk-0 logits, read from
INPUT memory. The trailing loss store targets `loss_ptr ≠ logsumexp_ptr`. -/
theorem chunked_cross_entropy_forward_lse_correct
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s s' : BlockState)
    (hchunk : s.pids 1 = 0)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (_hne : logsumexp_ptr ≠ loss_ptr)
    (hExec : exec (chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
      labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
      Bool.false DO_LOGIT_SCALING) s = some s') :
    s'.readMem logsumexp_ptr (fceChunkLseOffset s N_CHUNKS) =
      fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
        0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x) := by
  have h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have h_filter : (validLanes n VOCAB_SIZE 0).Nonempty := validLanes_nonempty h_tail
  cases hDLS : DO_LOGIT_SCALING
  · simp [exec, chunked_cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ne, hchunk, hDLS] at hExec
    rw [← hExec]
    simp only [fceChunkLseOffset]
    rw [BlockState.writeMem_readMem, if_pos (⟨rfl, by rw [hchunk, Nat.add_zero]⟩)]
    erw [fceLse_withBot_id n s logits_ptr logits_row_stride VOCAB_SIZE
      (fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val))
      h_tail h_ne h_filter rfl, WithBot.unbotD_coe]
    simp only [Bool.false_eq_true, reduceIte]
  · simp [exec, chunked_cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ne, hchunk, hDLS] at hExec
    rw [← hExec]
    simp only [fceChunkLseOffset]
    rw [BlockState.writeMem_readMem, if_pos (⟨rfl, by rw [hchunk, Nat.add_zero]⟩)]
    erw [fceLse_withBot_scale n s logits_ptr logits_row_stride VOCAB_SIZE LOGIT_SCALE
      (fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val))
      h_tail h_ne h_filter rfl, WithBot.unbotD_coe]
    simp only [if_true, reduceIte]

/-- **Genuine chunked forward loss correctness (chunk 0, no softcapping).** For
chunk `0` with `DO_SOFTCAPPING = false`, the surface writes to `loss_ptr[row_idx]`
exactly `-1 * transform(label logit)`.

Chunked-kernel modeling note (see the header's label-channel paragraph): this
surface's label load still goes through a dynamic pointer register, so the
`label_idx != -100` guard is always true under the algorithm-layer
cast-to-`Nat` erasure and the ignore case is dead here (unlike the non-chunked
forward headline, where the sentinel branch is genuine). -/
theorem chunked_cross_entropy_forward_loss_correct
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s s' : BlockState)
    (hchunk : s.pids 1 = 0)
    (_h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hne : loss_ptr ≠ logsumexp_ptr)
    (hExec : exec (chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
      labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
      Bool.false DO_LOGIT_SCALING) s = some s') :
    s'.readMem loss_ptr (fceOutOffset s) =
      (-1 : ℝ) * fceLabelLogit s logits_ptr labels_ptr logits_row_stride
        SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING := by
  cases hDLS : DO_LOGIT_SCALING
  · simp [exec, chunked_cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ne, hchunk, hDLS] at hExec
    rw [← hExec]
    rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
    simp only [fceOutOffset]
    rw [BlockState.writeMem_readMem, if_pos (⟨rfl, rfl⟩)]
    rw [show fceLabelLogit s logits_ptr labels_ptr logits_row_stride
        SOFTCAP LOGIT_SCALE Bool.false Bool.false =
        s.readMem logits_ptr (s.pids 0 * logits_row_stride + fceLabelNat s labels_ptr) from by
      unfold fceLabelLogit fastCeTransform; simp only [hDLS, Bool.false_eq_true, reduceIte]]
    unfold fceLabelNat
    norm_num
  · simp [exec, chunked_cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ne, hchunk, hDLS] at hExec
    rw [← hExec]
    rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
    simp only [fceOutOffset]
    rw [BlockState.writeMem_readMem, if_pos (⟨rfl, rfl⟩)]
    have hrhs : fceLabelLogit s logits_ptr labels_ptr logits_row_stride
        SOFTCAP LOGIT_SCALE Bool.false Bool.true =
        LOGIT_SCALE * s.readMem logits_ptr (s.pids 0 * logits_row_stride + fceLabelNat s labels_ptr) := by
      unfold fceLabelLogit fastCeTransform
      simp only [hDLS, Bool.false_eq_true, if_true, reduceIte]
    rw [hrhs]
    unfold fceLabelNat
    norm_num

/-- Surface transcription of `fast_ce_loss.py`'s `_cross_entropy_backward`.

This preserves the block logits load, optional logit scaling, optional softcap
transform and derivative factor, softmax-minus-one update at the label, and
final masked in-place gradient writeback. Python's hard-coded
`label_idx != -100` sentinel is preserved as the literal `-100`. Python's
local name `partial` is written `partial_` because `partial` is a Lean keyword. -/
def cross_entropy_backward_surface
    (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  block_idx = tl.program_id(1)
  logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
  dloss_ptr += row_idx * $(dloss_row_stride)
  col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = (tl.load(labels_ptr + row_idx)).to(tl.int32)
  if label_idx != $((-100 : Int)) {
    dloss = tl.load(dloss_ptr)
  } else {
    dloss = 0.0
  }
  x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    x = x * $(LOGIT_SCALE)
  }
  if DO_SOFTCAPPING {
    partial_ = triton_tanh(x / $(SOFTCAP))
    x = $(SOFTCAP) * partial_
  }
  logsumexp = tl.load(logsumexp_ptr + row_idx)
  y = tl.exp((x).to(tl.float32) - logsumexp)
  y = tl.where(col_offsets == label_idx, y - 1.0, y)
  if DO_LOGIT_SCALING {
    y = y * $(LOGIT_SCALE)
  }
  if DO_SOFTCAPPING {
    y = y * (1.0 - partial_ * partial_)
  }
  tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)
}

/-- The faithful full backward surface lowers to the algorithm layer, including
the ignored-label, logit scaling, softcapping derivative, and masked writeback
branches. -/
theorem cross_entropy_backward_surface_toAlgorithm_supported
    (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ∃ alg,
      (cross_entropy_backward_surface logits_ptr dloss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE SOFTCAP
        LOGIT_SCALE DO_SOFTCAPPING DO_LOGIT_SCALING).toAlgorithm? =
        Except.ok alg := by
  simp [cross_entropy_backward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented backward final-store slice of `fast_ce_loss.py`'s
`_cross_entropy_backward`.

The full kernel builds `y` from logits, logsumexp, labels, optional logit
scaling, and optional softcapping. This slice starts from a precomputed gradient
tile `Grad` and proves the final masked in-place writeback
`logits[col_offsets] = dloss * Grad[col_offsets]`. -/
def cross_entropy_backward_store_slice
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  block_idx = tl.program_id(1)
  col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  y = tl.load(Grad + row_idx * $(grad_row_stride) + col_offsets,
    mask=mask, other=0.0)
  tl.store(logits_ptr + row_idx * $(logits_row_stride) + col_offsets,
    dloss * y, mask=mask)
}

def colOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * BLOCK_SIZE + i.val

def active (s : BlockState) (VOCAB_SIZE BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Prop :=
  colOffset s BLOCK_SIZE i < VOCAB_SIZE

instance activeDecidable (s : BlockState) (VOCAB_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s VOCAB_SIZE BLOCK_SIZE i) := by
  unfold active
  infer_instance

def logitsOffset
    (s : BlockState) (logits_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * logits_row_stride + colOffset s BLOCK_SIZE i

def gradOffset
    (s : BlockState) (grad_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * grad_row_stride + colOffset s BLOCK_SIZE i

noncomputable def expectedBackward
    (s : BlockState) (dloss_ptr Grad : RegionName)
    (dloss_row_stride grad_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem dloss_ptr (s.pids 0 * dloss_row_stride) *
    s.readMem Grad (gradOffset s grad_row_stride BLOCK_SIZE i)

/-- Algorithm-layer correctness for the masked backward writeback. -/
theorem cross_entropy_backward_store_slice_correct
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (cross_entropy_backward_store_slice logits_ptr dloss_ptr Grad
        VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride BLOCK_SIZE)
        s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem logits_ptr (logitsOffset s logits_row_stride BLOCK_SIZE i) =
        if active s VOCAB_SIZE BLOCK_SIZE i then
          expectedBackward s dloss_ptr Grad dloss_row_stride grad_row_stride
            BLOCK_SIZE i
        else
          s.readMem logits_ptr (logitsOffset s logits_row_stride BLOCK_SIZE i) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * logits_row_stride + (s.pids 1 * BLOCK_SIZE + idx.1.val)) := by
    intro a b h
    have hInner :
        s.pids 1 * BLOCK_SIZE + a.1.val =
          s.pids 1 * BLOCK_SIZE + b.1.val := by
      exact Nat.add_left_cancel h
    have hab : a.1 = b.1 := Fin.ext (Nat.add_left_cancel hInner)
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, cross_entropy_backward_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [logitsOffset, colOffset, active, expectedBackward, gradOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hActive : s.pids 1 * BLOCK_SIZE + i.val < VOCAB_SIZE
  · simp [hActive]
  · simp [hActive]

/-- Compute-facing correctness for the masked backward writeback. -/
theorem cross_entropy_backward_store_slice_compute_correct
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := cross_entropy_backward_store_slice logits_ptr dloss_ptr Grad
        VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s VOCAB_SIZE BLOCK_SIZE i)
        (fun i => (logits_ptr, logitsOffset s logits_row_stride BLOCK_SIZE i)))
      (expected := fun i =>
        expectedBackward s dloss_ptr Grad dloss_row_stride grad_row_stride
          BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [cross_entropy_backward_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := cross_entropy_backward_store_slice_correct logits_ptr dloss_ptr Grad
    VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride BLOCK_SIZE
    s s' hExec i
  simpa [hActive] using h

/-- **Per-kernel chunked-forward output summary for
`chunked_cross_entropy_forward_surface` (genuine, end-to-end, chunk 0, no
softcapping).**

The chunked surface stores both side outputs only under `chunk_idx == 0`. Stated
as a conjunction of `ComputeCorrect.Realizes_without_Rounding` claims for chunk `0` with
`DO_SOFTCAPPING = false`, at least one valid lane, and `logsumexp_ptr ≠ loss_ptr`,
bundling:
1. **genuine per-chunk LSE output**: `logsumexp_ptr[row * N_CHUNKS + 0]` holds
   exactly `fastCeLseSpec` of the per-lane transformed INPUT chunk-0 logits;
2. **genuine chunk-0 partial loss output**: `loss_ptr[row]` holds exactly
   `-1 * transform(label logit)`, read from INPUT memory.

Each `ComputeCorrect.Realizes_without_Rounding` internalizes the execution (`exec ... = some s'`)
and the lowering to the algorithm layer. All value specs read INPUT memory;
non-self-referential. The softcapping branch is out of scope; the `-100` ignore
label is dead under cast-to-`Nat` erasure. -/
specification chunked_cross_entropy_forward_output_summary
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s : BlockState)
    (hchunk : s.pids 1 = 0)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
        labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
        Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (logsumexp_ptr, fceChunkLseOffset s N_CHUNKS))
      (expected := fun _ =>
        fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
        labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
        Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, fceOutOffset s))
      (expected := fun _ =>
        (-1 : ℝ) * fceLabelLogit s logits_ptr labels_ptr logits_row_stride
          SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)) := by
  refine ⟨?_, ?_⟩
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [chunked_cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact chunked_cross_entropy_forward_lse_correct logits_ptr loss_ptr logsumexp_ptr labels_ptr
      VOCAB_SIZE N_CHUNKS logits_row_stride SOFTCAP LOGIT_SCALE DO_LOGIT_SCALING n s s'
      hchunk h_tail hne hExec
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [chunked_cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact chunked_cross_entropy_forward_loss_correct logits_ptr loss_ptr logsumexp_ptr labels_ptr
      VOCAB_SIZE N_CHUNKS logits_row_stride SOFTCAP LOGIT_SCALE DO_LOGIT_SCALING n s s'
      hchunk h_tail hne.symm hExec

end VeriTile.Bench.TritonBenchG.FastCeLoss
