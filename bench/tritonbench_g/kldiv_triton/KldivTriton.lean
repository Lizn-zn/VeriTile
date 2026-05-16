import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KldivTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`.

This keeps the row pointer advances, dynamic `n_cols` loop, `log_target`
selection, and reduction-mode store behavior. -/
def kldiv_forward_surface
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (log_target : Bool) (reduction : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)

  base_offsets = tl.arange(0, $(BLOCK_SIZE))

  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

    if not log_target {
      loss = y_true * (tl.log(y_true) - y)
    } else {
      loss = tl.exp(y_true) * (y_true - y)
    }

    if $(reduction) == $(0) {
      tl.store(loss_ptr + offsets, loss, mask=mask)
    } else {
      loss = tl.sum(loss, axis=0)
      tl.store(loss_ptr, loss)
      loss_ptr += $(1)
    }
  }
}

/-- Documented one-block slice of `kldiv_triton.py`'s `_kldiv_kernel_backward`
for the `log_target = False` constexpr branch.

This models one `BLOCK_SIZE` iteration of Python's `for i in range(0, n_cols,
BLOCK_SIZE)` loop after the row pointers have been advanced.

Allowed mechanical Lean-syntax-only changes:
- Python `log_target: tl.constexpr` → separate kernel defs per branch. -/
def kldiv_backward_default
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
  res = target * -1
  tl.store(input_ptr + offsets, res, mask=mask)
}

/-- Documented one-block slice of `kldiv_triton.py`'s `_kldiv_kernel_forward`
for the `log_target = True`, `reduction = 0` (None) constexpr branch.

Mirrors the kldiv_ops port; covers a single `BLOCK_SIZE` iteration of the
Python loop with the elementwise-store path. -/
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

/-- Documented one-block slice of `_kldiv_kernel_backward` for the
`log_target = True` constexpr branch. -/
def kldiv_backward_log_target
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
  res = -tl.exp(target)
  tl.store(input_ptr + offsets, res, mask=mask)
}

def outOffset (s : BlockState) (input_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * input_stride + i.val

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

/-- Algorithm-layer correctness for the default backward one-block slice. -/
theorem kldiv_backward_default_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i))
    (hExec : exec (kldiv_backward_default input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem input_ptr (outOffset s input_stride i) =
        if i.val < n_cols then
          defaultSpec s target_ptr target_stride i
        else s.readMem input_ptr (outOffset s input_stride i) := by
  intro i
  simp [exec, kldiv_backward_default, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * input_stride + idx.1.val) := by
    intro a b h
    exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) rfl
  simp only [outOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases h : i.val < n_cols
  · simp [defaultSpec, inOffset, h]
  · simp [h]
/-- Algorithm-layer correctness for the log-target backward one-block slice. -/
theorem kldiv_backward_log_target_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i))
    (hExec : exec (kldiv_backward_log_target input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem input_ptr (outOffset s input_stride i) =
        if i.val < n_cols then
          logTargetSpec s target_ptr target_stride i
        else s.readMem input_ptr (outOffset s input_stride i) := by
  intro i
  simp [exec, kldiv_backward_log_target, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * input_stride + idx.1.val) := by
    intro a b h
    exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) rfl
  simp only [outOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases h : i.val < n_cols
  · simp [logTargetSpec, inOffset, h]
  · simp [h]
/-- Compute-facing correctness for the default backward one-block slice. -/
theorem kldiv_backward_default_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_default input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, outOffset s input_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_default]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_default_correct input_ptr target_ptr
    input_stride target_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h
/-- Compute-facing correctness for the log-target backward one-block slice. -/
theorem kldiv_backward_log_target_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_log_target input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, outOffset s input_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_log_target]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_log_target_correct input_ptr target_ptr
    input_stride target_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h
/-- Algorithm-layer correctness for the forward `log_target=True`,
`reduction=0` one-block slice. -/
theorem kldiv_forward_log_target_none_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s loss_stride i))
    (hExec : exec (kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem loss_ptr (outOffset s loss_stride i) =
        if i.val < n_cols then
          forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i
        else s.readMem loss_ptr (outOffset s loss_stride i) := by
  intro i
  simp [exec, kldiv_forward_log_target_none, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * loss_stride + idx.1.val) := by
    intro a b h
    exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) rfl
  simp only [outOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases h : i.val < n_cols
  · simp [forwardLogTargetSpec, inOffset, h]
  · simp [h]

/-- Compute-facing correctness for the forward log-target one-block slice. -/
theorem kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s loss_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, outOffset s loss_stride i)))
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

end VeriTile.Bench.TritonBenchG.KldivTriton
