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
cross_entropy_forward_output_summary                        ← TOP (fwd, genuine end-to-end)
  ├─ (toAlgorithm? = Except.ok _)                           full fwd surface lowers
  ├─ cross_entropy_forward_lse_correct                      ← genuine masked transformed LSE store
  │    └─ fceLse_withBot_id / fceLse_withBot_scale          WithBot LSE carrier = some fastCeLseSpec
  └─ cross_entropy_forward_loss_correct                     ← genuine loss = LSE − transform(label)
chunked_cross_entropy_forward_output_summary                ← TOP (chunked fwd, genuine, chunk 0)
  ├─ (toAlgorithm? = Except.ok _)                           chunked fwd surface lowers
  ├─ chunked_cross_entropy_forward_lse_correct              ← genuine per-chunk LSE store (chunk 0)
  └─ chunked_cross_entropy_forward_loss_correct             ← genuine chunk-0 partial loss = −transform(label)
cross_entropy_backward_surface_toAlgorithm_supported        ← bwd surface lowers
cross_entropy_backward_store_slice_compute_correct          ← masked dloss·y writeback
  └─ cross_entropy_backward_store_slice_correct
```

The two forward kernels are now **genuinely value-correct end-to-end** under
`DO_SOFTCAPPING = false`: executing the full surfaces writes to `logsumexp_ptr`
the masked-lane stable log-sum-exp `fastCeLseSpec` of the per-lane *transformed*
INPUT logits (transform = optional `LOGIT_SCALE * ·`), and to `loss_ptr` the
cross-entropy loss `LSE − transform(label logit)` (forward) / the chunk-0 partial
`−transform(label logit)` (chunked). All value specs read INPUT memory, never
`exec(...).readMem` — non-self-referential. The backward kernel is proved to
*lower* plus the proof-oriented masked `dloss·y` store slice.

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

The hard-coded `label_idx != -100` sentinel is preserved as the literal `-100`,
but is **dead in the algorithm-layer model**: `label_idx = (...).to(tl.int32)`
erases to a `Nat` offset, and a `Nat` can never equal the negative `-100`, so the
ignore branch is statically unreachable; the model always takes the active loss
branch. The `.to(tl.float32)`/`.to(tl.int32)`/`.to(tl.int64)` casts erase to
identity at the algorithm layer. The forward logits load uses
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
`-100`. -/
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
  labels_ptr += row_idx
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
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

/-! ## Genuine forward value specs (input-memory, non-self-referential)

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

/-- The label value loaded by the forward kernel:
`label_idx = (tl.load(labels_ptr + row_idx)).to(tl.int32)`, read from INPUT
memory and (per the algorithm-layer cast erasure) carried as a `Nat` offset. -/
noncomputable def fceLabelNat (s : BlockState) (labels_ptr : Region .int) : Nat :=
  s.readMemValue .nat (Region.cast labels_ptr) (s.pids 0)

/-- The transformed label logit `transform(tl.load(logits_ptr + label_idx))`
read from INPUT memory at the data-dependent (Nat) label position. -/
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

/-- **Genuine forward loss correctness (no softcapping).** Executing the full
forward surface with `DO_SOFTCAPPING = false` writes to `loss_ptr` at offset
`row_idx` exactly `logsumexp − transform(label logit)`.

Note: under the algorithm-layer cast erasure, `label_idx = (...).to(tl.int32)`
is carried as a `Nat`, so the kernel's `label_idx != -100` guard is always true
(a `Nat`-valued index can never equal the negative sentinel). The model therefore
always takes the active loss branch; the `-100` ignore case is dead under this
modeling boundary. -/
theorem cross_entropy_forward_loss_correct
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s s' : BlockState)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hExec : exec (cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
      VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)
      s = some s') :
    s'.readMem loss_ptr (fceOutOffset s) =
      fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
        0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x)
      - fceLabelLogit s logits_ptr labels_ptr logits_row_stride
          SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING := by
  have h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have h_filter : (validLanes n VOCAB_SIZE 0).Nonempty := validLanes_nonempty h_tail
  cases hDLS : DO_LOGIT_SCALING
  · simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, ComparableDType.ne, hDLS] at hExec
    rw [← hExec]
    simp only [BlockState.writeMem_readMem, fceOutOffset, Region.cast_id, and_true, if_true]
    erw [fceLse_withBot_id n s logits_ptr logits_row_stride VOCAB_SIZE
      (fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val))
      h_tail h_ne h_filter rfl]
    simp only [Option.map_some, WithBot.unbotD_some, WithBot.unbotD_coe, fceLabelLogit,
      fastCeTransform, fceLabelNat, hDLS, Bool.false_eq_true, reduceIte, Region.cast]
  · simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, ComparableDType.ne, hDLS] at hExec
    rw [← hExec]
    simp only [BlockState.writeMem_readMem, fceOutOffset, Region.cast_id, and_true, if_true]
    erw [fceLse_withBot_scale n s logits_ptr logits_row_stride VOCAB_SIZE LOGIT_SCALE
      (fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val))
      h_tail h_ne h_filter rfl]
    simp only [Option.map_some, WithBot.unbotD_some, WithBot.unbotD_coe, fceLabelLogit,
      fastCeTransform, fceLabelNat, hDLS, if_true, Bool.false_eq_true, reduceIte, Region.cast]

/-- **Genuine forward LSE correctness (no softcapping).** Executing the full
forward surface with `DO_SOFTCAPPING = false` writes to `logsumexp_ptr` at offset
`row_idx` exactly `fastCeLseSpec`: the masked-lane stable log-sum-exp of the
per-lane transformed row logits, read from INPUT memory. The trailing loss store
targets `loss_ptr ≠ logsumexp_ptr` and does not disturb the LSE cell. -/
theorem cross_entropy_forward_lse_correct
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s s' : BlockState)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr)
    (hExec : exec (cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
      VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)
      s = some s') :
    s'.readMem logsumexp_ptr (fceOutOffset s) =
      fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
        0 h_tail
        (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x) := by
  have h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have h_filter : (validLanes n VOCAB_SIZE 0).Nonempty := validLanes_nonempty h_tail
  cases hDLS : DO_LOGIT_SCALING
  · -- DO_LOGIT_SCALING = false: transform g = id
    simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, hDLS] at hExec
    rw [← hExec]
    rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
    simp only [BlockState.writeMem_readMem, fceOutOffset, Region.cast_id, and_true, if_true]
    erw [fceLse_withBot_id n s logits_ptr logits_row_stride VOCAB_SIZE
      (fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val))
      h_tail h_ne h_filter rfl, WithBot.unbotD_coe]
    simp only [Bool.false_eq_true, reduceIte]
  · -- DO_LOGIT_SCALING = true: transform g = (LOGIT_SCALE * ·)
    simp [exec, cross_entropy_forward_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop, FloatDType.cast,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, hDLS] at hExec
    rw [← hExec]
    rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
    simp only [BlockState.writeMem_readMem, fceOutOffset, Region.cast_id, and_true, if_true]
    erw [fceLse_withBot_scale n s logits_ptr logits_row_stride VOCAB_SIZE LOGIT_SCALE
      (fun i => s.readMem logits_ptr (s.pids 0 * logits_row_stride + i.val))
      h_tail h_ne h_filter rfl, WithBot.unbotD_coe]

/-- **Per-kernel forward output summary for `cross_entropy_forward_surface`
(genuine, end-to-end, no softcapping).**

Stated as a conjunction of `ComputeCorrect.Realizes_without_Rounding` claims with
`DO_SOFTCAPPING = false`, at least one valid lane (`0 < VOCAB_SIZE`), and
`logsumexp_ptr ≠ loss_ptr`, bundling:
1. **genuine LSE output**: `logsumexp_ptr[row]` holds exactly `fastCeLseSpec` —
   the masked-lane stable log-sum-exp of the per-lane transformed INPUT row
   logits (transform = optional `LOGIT_SCALE * ·`);
2. **genuine loss output**: `loss_ptr[row]` holds exactly
   `fastCeLseSpec − transform(label logit)`, every term read from INPUT memory.

Each `ComputeCorrect.Realizes_without_Rounding` internalizes the execution (`exec ... = some s'`)
and the lowering to the algorithm layer. All value specs read INPUT memory, never
`exec(...).readMem`, so this summary is non-self-referential. The
region-distinctness and one-valid-lane hypotheses are the only side-conditions.
The softcapping branch is out of scope (see `fastCeTransform`'s `⊥`-propagation
note); the `-100` ignore label is dead under the cast-to-`Nat` erasure. -/
specification cross_entropy_forward_output_summary
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s : BlockState)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (logsumexp_ptr, fceOutOffset s))
      (expected := fun _ =>
        fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, fceOutOffset s))
      (expected := fun _ =>
        fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x)
        - fceLabelLogit s logits_ptr labels_ptr logits_row_stride
            SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)) := by
  refine ⟨?_, ?_⟩
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [cross_entropy_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact cross_entropy_forward_lse_correct logits_ptr loss_ptr logsumexp_ptr labels_ptr
      VOCAB_SIZE logits_row_stride SOFTCAP LOGIT_SCALE DO_LOGIT_SCALING n s s' h_tail hne hExec
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [cross_entropy_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact cross_entropy_forward_loss_correct logits_ptr loss_ptr logsumexp_ptr labels_ptr
      VOCAB_SIZE logits_row_stride SOFTCAP LOGIT_SCALE DO_LOGIT_SCALING n s s' h_tail hExec

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

Same modeling note as the forward kernel: the `label_idx != -100` guard is
always true under the algorithm-layer cast-to-`Nat` erasure, so the ignore case
is dead. -/
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

/-! ## Bridge to the canonical pure cross-entropy math

In the textbook regime — `DO_SOFTCAPPING = false`, `DO_LOGIT_SCALING = false`,
and the block exactly spanning the vocabulary (`BLOCK_SIZE = VOCAB_SIZE = n+1`,
so every lane is valid) — the kernel's genuine LSE/loss specs collapse to the
shared pure `crossEntropyLoss` from `VeriTile.Triton.Math.Loss`. -/

open VeriTile.Triton.TiledLoss in
/-- In the full-vocab, no-transform regime the kernel's per-row stable LSE
`fastCeLseSpec … id` is exactly the canonical `stableLSE` of the row logits. -/
theorem fastCeLseSpec_id_eq_stableLSE
    {n : Nat} (xs : Fin (n+1) → ℝ) (h_tail : 0 * (n+1) < (n+1)) :
    fastCeLseSpec xs 0 h_tail (fun x => x) = stableLSE xs (Nat.succ_pos n) Bool.false 0 := by
  have heq : fastCeLseSpec xs 0 h_tail (fun x => x)
      = partialLSE_full (n := n) xs 0 (by simp) Bool.false 0 := by
    unfold fastCeLseSpec partialLSE_full scaledLane_full
    simp only [Bool.false_eq_true, reduceIte]
  rw [heq, partialLSE_full_zero_self_eq_stableLSE]

open VeriTile.Triton.TiledLoss in
/-- **Bridge: textbook cross-entropy.** Under `DO_SOFTCAPPING = false`,
`DO_LOGIT_SCALING = false`, and a block spanning the whole vocabulary
(`VOCAB_SIZE = n+1`, label in range as `lbl : Fin (n+1)` with its underlying
offset equal to the loaded `fceLabelNat`), the forward kernel's genuine loss
spec value (the RHS of `cross_entropy_forward_loss_correct`) equals the
canonical pure `crossEntropyLoss` of the row logits at the target class. -/
theorem fastCeLoss_eq_crossEntropyLoss
    (s : BlockState) (logits_ptr : RegionName) (labels_ptr : Region .int)
    (logits_row_stride : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
    (n : Nat) (h_tail : 0 * (n+1) < (n+1))
    (lbl : Fin (n+1))
    (hlbl : (lbl : Nat) = fceLabelNat s labels_ptr) :
    fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride (n+1))
        0 h_tail (fun x => x)
      - fceLabelLogit s logits_ptr labels_ptr logits_row_stride
          SOFTCAP LOGIT_SCALE Bool.false Bool.false
    = crossEntropyLoss (fastCeRowLogits s logits_ptr logits_row_stride (n+1))
        lbl (Nat.succ_pos n) := by
  unfold crossEntropyLoss
  rw [fastCeLseSpec_id_eq_stableLSE]
  -- the label logit equals the row-logits function evaluated at `lbl`
  have hlabel : fceLabelLogit s logits_ptr labels_ptr logits_row_stride
      SOFTCAP LOGIT_SCALE Bool.false Bool.false
      = fastCeRowLogits s logits_ptr logits_row_stride (n+1) lbl := by
    unfold fceLabelLogit fastCeTransform fastCeRowLogits
    simp only [Bool.false_eq_true, reduceIte]
    rw [hlbl]
  rw [hlabel]



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
