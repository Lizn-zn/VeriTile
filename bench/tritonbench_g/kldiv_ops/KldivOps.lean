import VeriTile.Triton

/-!
# `kldiv_ops` — strict per-kernel correctness

`kldiv_ops.py` is the Liger-style KL-divergence kernel pair: forward
`_kldiv_kernel_forward` (per row `pid`, streams `n_cols` in `BLOCK_SIZE` tiles,
computing `y_true * (log(max(y_true, eps)) - y)` or, for `log_target`,
`exp(y_true) * (y_true - y)`; stores per element for `reduction=none` or
accumulates a sum otherwise) and backward `_kldiv_kernel_backward` (`-target`,
or `-exp(target)` for `log_target`).

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (grid `(BT,)`, the `BLOCK_SIZE`/`num_warps`
choices, the `reduction`-mode mapping and host-side post-reduction
`sum/mean/batchmean`, and how the runtime composes per-program writes) is the
*trusted boundary*, not a proof obligation here. The row id `pid` is universally
quantified, so each per-program statement covers every row of the grid.

## Proof architecture

This is a multi-branch port (`log_target` × `reduction` × forward/backward);
there is no single top theorem. Each verified constexpr branch has its own
compute-facing correctness theorem:

```
kldiv_backward_default_compute_correct          (-target)        ← defaultSpec
  └─ kldiv_backward_default_correct
kldiv_backward_log_target_compute_correct       (-exp target)    ← logTargetSpec
  └─ kldiv_backward_log_target_correct
kldiv_forward_default_none_compute_correct      (reduction=none) ← forwardDefaultSpec
  └─ kldiv_forward_default_none_correct
       └─ ComparableDType.real_gt_some_some_eq_true_iff  (eps-clamp bridge)
kldiv_forward_log_target_none_compute_correct   (reduction=none) ← forwardLogTargetSpec
  └─ kldiv_forward_log_target_none_correct

kldiv_forward_surface_toAlgorithm_supported     full surface lowers (lowering only)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. **This port is partial.** The full
reduction-mode forward surface `kldiv_forward_surface` (with eps clamp and the
accumulated `loss_sum`) is only proved to *lower* to the algorithm layer; value
correctness targets the `reduction=0` (none) elementwise branches plus both
backward branches. The KL loss spec is the per-element expression
`y_true * (log(max(y_true, eps)) - y)` / `exp(y_true) * (y_true - y)`; the
`tl.maximum(y_true, eps)` clamp is bridged through
`ComparableDType.real_gt_some_some_eq_true_iff` and a `max_eq_left`/`max_eq_right`
case split. The forward-none and backward branches are modeled as one-block
slices (one `BLOCK_SIZE` iteration after the row-pointer advance); masked lanes
use `other=0.0` and out-of-bounds lanes (`i ≥ n_cols`) are preserved. Output
offset injectivity is an explicit hypothesis. The specs reference `Real.log` /
`Real.exp` directly, not `VeriTile.Triton.Math.*`.
-/

namespace VeriTile.Bench.TritonBenchG.KldivOps

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `kldiv_ops.py`'s `_kldiv_kernel_forward`.

This keeps the row pointer advances, dynamic `n_cols` loop, `eps`-clamped
default target log, `log_target` selection, accumulated reduction value, and
the final reduced store. -/
def kldiv_forward_surface
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (log_target : Bool) (reduction : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)

  base_offsets = tl.arange(0, $(BLOCK_SIZE))

  loss_sum = 0.0
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

    if not log_target {
      loss = y_true * (tl.log(tl.maximum(y_true, $(eps))) - y)
    } else {
      loss = tl.exp(y_true) * (y_true - y)
    }

    if $(reduction) == $(0) {
      tl.store(loss_ptr + offsets, loss, mask=mask)
    } else {
      loss_sum += tl.sum(loss, axis=0)
    }
  }

  if $(reduction) != $(0) {
    tl.store(loss_ptr, loss_sum)
  }
}

/-- The full KL-divergence ops forward surface lowers to the algorithm layer,
including eps clamping, the `log_target` branch, dynamic column loop, and
reduction modes. -/
theorem kldiv_forward_surface_toAlgorithm_supported
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (log_target : Bool) (reduction : Nat) :
    ∃ alg,
      (kldiv_forward_surface y_ptr gt_ptr loss_ptr y_stride gt_stride
        loss_stride n_cols BLOCK_SIZE eps log_target reduction).toAlgorithm? =
        Except.ok alg := by
  simp [kldiv_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Documented one-block slice of `kldiv_ops.py`'s `_kldiv_kernel_forward`
for the `log_target = False`, `reduction = 0` (None) constexpr branch.

This models one `BLOCK_SIZE` iteration of Python's
`for i in range(0, n_cols, BLOCK_SIZE)` loop after the row pointers have been
advanced, taking the elementwise-store path of `reduction == 0`. -/
def kldiv_forward_default_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
  y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
  loss = y_true * (tl.log(tl.maximum(y_true, $(eps))) - y)
  tl.store(loss_ptr + offsets, loss, mask=mask)
}

/-- Documented one-block slice of `_kldiv_kernel_forward` for the
`log_target = True`, `reduction = 0` (None) constexpr branch. -/
def kldiv_forward_log_target_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
  y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
  loss = tl.exp(y_true) * (y_true - y)
  tl.store(loss_ptr + offsets, loss, mask=mask)
}

/-- Documented one-block slice of `kldiv_ops.py`'s `_kldiv_kernel_backward`
for the `log_target = False` constexpr branch.

This models one `BLOCK_SIZE` iteration of Python's `for i in range(0, n_cols,
BLOCK_SIZE)` loop after the row pointers have been advanced.

Allowed mechanical Lean-syntax-only changes:
- Python `log_target: tl.constexpr` → separate kernel defs per branch. -/
def kldiv_backward_default
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  target_ptr += pid * $(target_stride)
  new_grads_ptr += pid * $(new_grads_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
  res = target * -1
  tl.store(new_grads_ptr + offsets, res, mask=mask)
}

/-- Documented one-block slice of `_kldiv_kernel_backward` for the
`log_target = True` constexpr branch. -/
def kldiv_backward_log_target
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  target_ptr += pid * $(target_stride)
  new_grads_ptr += pid * $(new_grads_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
  res = -tl.exp(target)
  tl.store(new_grads_ptr + offsets, res, mask=mask)
}


def inOffset (s : BlockState) (target_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * target_stride + i.val

noncomputable def defaultSpec
    (s : BlockState) (target_ptr : RegionName) (target_stride : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem target_ptr (inOffset s target_stride i) * (0.0 - 1)

noncomputable def logTargetSpec
    (s : BlockState) (target_ptr : RegionName) (target_stride : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  0.0 - Real.exp (s.readMem target_ptr (inOffset s target_stride i))

/-- Forward KL-divergence per-element value (`log_target = True`)
for the `reduction = 0` (None) elementwise-store path. -/
noncomputable def forwardLogTargetSpec
    (s : BlockState) (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let y := s.readMem y_ptr (inOffset s y_stride i)
  let y_true := s.readMem gt_ptr (inOffset s gt_stride i)
  Real.exp y_true * (y_true - y)

/-- Forward KL-divergence per-element value (default, `log_target = False`)
for the `reduction = 0` (None) elementwise-store path. -/
noncomputable def forwardDefaultSpec
    (s : BlockState) (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  let y := s.readMem y_ptr (inOffset s y_stride i)
  let y_true := s.readMem gt_ptr (inOffset s gt_stride i)
  y_true * (Real.log (max y_true eps) - y)

/-- Bridge: For real `a b`, the kernel's Bool comparison
`ComparableDType.real.gt (some a) (some b) = true` is equivalent to `b < a`. -/
lemma _root_.VeriTile.Triton.ComparableDType.real_gt_some_some_eq_true_iff
    (a b : ℝ) :
    ComparableDType.real.gt (some a) (some b) = Bool.true ↔ b < a := by
  unfold ComparableDType.gt
  rw [decide_eq_true_iff]
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- `some a > some b` (WithBot) → `b < a` (ℝ)
    have h' : (b : WithBot ℝ) < (a : WithBot ℝ) := h
    exact_mod_cast h'
  · -- `b < a` (ℝ) → `some a > some b` (WithBot)
    show (b : WithBot ℝ) < (a : WithBot ℝ)
    exact_mod_cast h

/-- Algorithm-layer correctness for the default backward one-block slice. -/
theorem kldiv_backward_default_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i))
    (hExec : exec (kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem new_grads_ptr (linearOffset s new_grads_stride i) =
        if i.val < n_cols then
          defaultSpec s target_ptr target_stride i
        else s.readMem new_grads_ptr (linearOffset s new_grads_stride i) := by
  intro i
  simp [exec, kldiv_backward_default, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  by_cases h : i.val < n_cols
  · simp [defaultSpec, inOffset, h]
  · simp [h]
/-- Algorithm-layer correctness for the log-target backward one-block slice. -/
theorem kldiv_backward_log_target_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i))
    (hExec : exec (kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem new_grads_ptr (linearOffset s new_grads_stride i) =
        if i.val < n_cols then
          logTargetSpec s target_ptr target_stride i
        else s.readMem new_grads_ptr (linearOffset s new_grads_stride i) := by
  intro i
  simp [exec, kldiv_backward_log_target, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  by_cases h : i.val < n_cols
  · simp [logTargetSpec, inOffset, h]
  · simp [h]
/-- Compute-facing correctness for the default backward one-block slice. -/
specification kldiv_backward_default_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_default]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_default_correct target_ptr new_grads_ptr
    target_stride new_grads_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h
/-- Compute-facing correctness for the log-target backward one-block slice. -/
specification kldiv_backward_log_target_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_log_target]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_log_target_correct target_ptr new_grads_ptr
    target_stride new_grads_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h
/-- Algorithm-layer correctness for the forward `log_target=False`,
`reduction=0` one-block slice.

Uses the `ComparableDType.real_gt_some_some_eq_true_iff` bridge to lift the
kernel's Bool comparison `tl.maximum(y_true, eps)` to `eps < y_true`, then
case-splits to discharge the clamp via `max_eq_left`/`max_eq_right`. -/
theorem kldiv_forward_default_none_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i))
    (hExec : exec (kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem loss_ptr (linearOffset s loss_stride i) =
        if i.val < n_cols then
          forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride eps i
        else s.readMem loss_ptr (linearOffset s loss_stride i) := by
  intro i
  -- Disable the `@[simp]` Prop-form bridge so `simp` doesn't push `decide` into
  -- a classical Decidable. Keep the Bool comparison form intact.
  simp [exec, kldiv_forward_default_none, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, -ComparableDType.real_gt_eq_true] at hExec
  rw [← hExec]
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  -- Reduce the Prod projection at the top.
  show (if i.val < n_cols then _ else _) =
    if i.val < n_cols then _ else s.readMem loss_ptr (s.pid * loss_stride + i.val)
  -- Reduce `Prod.fst` projections; both LHS and RHS use the same `i.val`.
  have hfst : ((i, PUnit.unit) : TileIndex [BLOCK_SIZE]).1 = i := rfl
  rw [hfst]
  by_cases h : i.val < n_cols
  · rw [if_pos h, if_pos h]
    by_cases hgt : eps < s.readMem gt_ptr (s.pids 0 * gt_stride + i.val)
    · have hbool : ComparableDType.gt ComparableDType.real
          (some (s.readMem gt_ptr (s.pids 0 * gt_stride + i.val))) (some eps) =
          Bool.true :=
        (ComparableDType.real_gt_some_some_eq_true_iff _ _).mpr hgt
      simp only [hbool, if_true]
      simp only [forwardDefaultSpec, inOffset, BlockState.pid_eq,
        max_eq_left (le_of_lt hgt)]
      rw [← Int.natCast_mul, Int.toNat_natCast]
      simp [h]
    · push Not at hgt
      have hbool : ComparableDType.gt ComparableDType.real
          (some (s.readMem gt_ptr (s.pids 0 * gt_stride + i.val))) (some eps) =
          Bool.false := by
        rw [Bool.eq_false_iff, Ne, ComparableDType.real_gt_some_some_eq_true_iff]
        exact not_lt.mpr hgt
      simp only [hbool, if_false]
      simp only [forwardDefaultSpec, inOffset, BlockState.pid_eq,
        max_eq_right hgt]
      rw [← Int.natCast_mul, Int.toNat_natCast]
      simp [h]
  · rw [if_neg h, if_neg h]
    simp [BlockState.pid_eq]

/-- Compute-facing correctness for the forward `log_target=False`,
`reduction=0` one-block slice. -/
specification kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_forward_default_none]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_forward_default_none_correct y_ptr gt_ptr loss_ptr
    y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps s s' hOutInj hExec i
  simpa [hActive] using h

/-- Algorithm-layer correctness for the forward `log_target=True`,
`reduction=0` one-block slice. -/
theorem kldiv_forward_log_target_none_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i))
    (hExec : exec (kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem loss_ptr (linearOffset s loss_stride i) =
        if i.val < n_cols then
          forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i
        else s.readMem loss_ptr (linearOffset s loss_stride i) := by
  intro i
  simp [exec, kldiv_forward_log_target_none, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * loss_stride + idx.1.val) := by
    intro a b h
    exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) rfl
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  by_cases h : i.val < n_cols
  · rw [← Int.natCast_mul, Int.toNat_natCast]
    simp [forwardLogTargetSpec, inOffset, h]
  · simp [h]

/-- Compute-facing correctness for the forward log-target one-block slice. -/
specification kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_forward_log_target_none]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_forward_log_target_none_correct y_ptr gt_ptr loss_ptr
    y_stride gt_stride loss_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## ════════ `⊨` IO face for the forward log-target slice ════════

The `Realizes_without_Rounding` faces above are stated per *declared write map*.
This section restates the `log_target = True`, `reduction = None` forward slice
on the audit-once IO surface `MaskedKernelIO₂.Implements` (`⊨`), which additionally
pins the **flat-memory** placement: for every disjoint placement of the three
buffers, every program id whose active lanes are in bounds, and every launch
state whose input windows hold `ys`/`gts` at the active lanes, the translated
pointer kernel terminates, every active output lane holds the spec value, and
every other memory cell is unchanged.

The three per-buffer bases differ (`y_stride` / `gt_stride` / `loss_stride`),
which is exactly what `MaskedKernelIO₂`'s separate `read1` / `read2` / `write`
express. The mask is pid-independent here (`offsets < n_cols`). -/

/-- Masked-scatter frame off the written lanes. `bench` files are standalone, so
this six-line induction is a private copy rather than an import. -/
theorem foldl_writeMem_masked_unhit {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (l : List α) (off : Nat) (ho : ∀ k ∈ l, P k → offsetFn k ≠ off) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
          s).mem region off) = s.mem region off := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih (fun k hk => ho k (List.mem_cons_of_mem hd hk))]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem,
          if_neg (fun hc => ho hd (List.mem_cons_self) hP hc.2.symm)]
      · rw [if_neg hP]

/-- The slice sits inside the flat-memory bridge's covered fragment. -/
theorem kldiv_forward_log_target_none_flattenOk
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ((kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [kldiv_forward_log_target_none, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
their own buffer's window at `pid * stride + j`, active only when `j < n_cols`,
so the bounds contract is lane-wise. -/
theorem kldiv_forward_log_target_none_traceSafe
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hy : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * y_stride + j.val < bounds y_ptr)
    (hgt : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * gt_stride + j.val < bounds gt_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * loss_stride + j.val < bounds loss_ptr) :
    Kernel.TraceSafe bounds
      ((kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [kldiv_forward_log_target_none, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg, Op.PointerAddressesSafeOn,
    Op.MemorySafe]
  refine ⟨fun a ha => ?_, fun a ha => ?_, fun a ha => ?_⟩ <;>
    rw [← Int.natCast_mul, Int.toNat_natCast]
  · simpa [BlockState.pid] using hy a ha
  · simpa [BlockState.pid] using hgt a ha
  · simpa [BlockState.pid] using hout a ha

set_option maxHeartbeats 1600000 in
/-- The slice touches nothing but its own active output lanes. -/
theorem kldiv_forward_log_target_none_frame
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ r o, (r ≠ loss_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
        o ≠ s.pid * loss_stride + j.val) →
      s'.mem r o = s.mem r o := by
  simp [exec, kldiv_forward_log_target_none, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  intro r o hcond
  rw [← hExec]
  by_cases hr : r = loss_ptr
  · subst hr
    rcases hcond with hne | hno
    · exact absurd rfl hne
    · refine foldl_writeMem_masked_unhit
        (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
        _ _ _ o (fun k _ hk => ?_) _
      rw [← Int.natCast_mul, Int.toNat_natCast]
      simpa [BlockState.pid] using (hno k.1 hk).symm
  · exact BlockState.foldl_writeMem_prop_masked_mem_preserve_other_region
      _ _ _ _ r hr o _

set_option maxHeartbeats 1600000 in
/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose input
windows are loaded at the active lanes only. This is the `hrun` obligation of
`MaskedKernelIO₂.Implements.intro`. -/
theorem kldiv_forward_log_target_none_region_run
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s₀ : BlockState) (ys gts : Fin BLOCK_SIZE → ℝ)
    (hy : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem y_ptr (s₀.pid * y_stride + j.val) = ys j)
    (hgt : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem gt_ptr (s₀.pid * gt_stride + j.val) = gts j) :
    ∃ s1, exec ((kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem loss_ptr (s₀.pid * loss_stride + j.val)
            = Real.exp (gts j) * (gts j - ys j))
      ∧ (∀ r o,
          (r ≠ loss_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * loss_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s₀ loss_stride i) := by
    intro a b h
    exact Fin.ext (Nat.add_left_cancel h)
  cases hsrc : exec ((kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, kldiv_forward_log_target_none, ComputeKernel.toAlgKernel,
          stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, ?_⟩
      · have h := kldiv_forward_log_target_none_correct y_ptr gt_ptr loss_ptr
          y_stride gt_stride loss_stride n_cols BLOCK_SIZE s₀ s1 hInj
          (by simpa using hsrc) j
        rw [if_pos hj] at h
        simpa [linearOffset, forwardLogTargetSpec, inOffset, hy j hj, hgt j hj]
          using h
      · exact kldiv_forward_log_target_none_frame y_ptr gt_ptr loss_ptr
          y_stride gt_stride loss_stride n_cols BLOCK_SIZE s₀ s1
          (by simpa using hsrc)

/-- `kldiv_forward_log_target_none`'s masked **IO signature** — the whole
kernel-specific audit surface of the headline: which buffer is which argument,
where program `pid` reads each input tile and writes its output tile (each at
its **own** stride), and the active-lane predicate `j < n_cols`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`ptr += pid * stride; offsets = arange; mask = offsets <
n_cols`), and the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the active lanes. -/
def kldivForwardLogTargetIO (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
    y_stride gt_stride loss_stride n_cols BLOCK_SIZE
  in1 := y_ptr
  in2 := gt_ptr
  out := loss_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * y_stride
  read2 := fun pid => pid * gt_stride
  write := fun pid => pid * loss_stride
  mask := fun _pid j => j.val < n_cols

/-- **The headline on the IO surface**: the `log_target = True`,
`reduction = None` forward slice implements `exp(y_true) · (y_true − y)` on its
masked IO signature — for every disjoint flat placement of the three buffers,
every program id whose active lanes are in bounds, and every launch state whose
input windows hold `ys`/`gts` at the active lanes, the translated pointer kernel
terminates, every active output lane holds the spec value, and every other
memory cell is unchanged.

This is the audit-once surface; the `Realizes_without_Rounding` face above
states the same computation against a declared write map but does not pin the
flat placement. -/
specification kldiv_forward_log_target_none_correctness
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    kldivForwardLogTargetIO y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE
      ⊨ fun ys gts i => Real.exp (gts i) * (gts i - ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact kldiv_forward_log_target_none_flattenOk y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact kldiv_forward_log_target_none_traceSafe y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE bounds s h1 h2 h3
  · intro s₀ ys gts hy hgt
    obtain ⟨s1, hexec, hval, hframe⟩ := kldiv_forward_log_target_none_region_run
      y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols BLOCK_SIZE
      s₀ ys gts hy hgt
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.KldivOps
