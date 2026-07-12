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

end VeriTile.Bench.TritonBenchG.KldivOps
