import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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

def outOffset (s : BlockState) (new_grads_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * new_grads_stride + i.val

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

/-- Algorithm-layer correctness for the default backward one-block slice. -/
theorem kldiv_backward_default_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s new_grads_stride i))
    (hExec : exec (kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem new_grads_ptr (outOffset s new_grads_stride i) =
        if i.val < n_cols then
          defaultSpec s target_ptr target_stride i
        else s.readMem new_grads_ptr (outOffset s new_grads_stride i) := by
  intro i
  simp [exec, kldiv_backward_default, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * new_grads_stride + idx.1.val) := by
    intro a b h
    exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) rfl
  simp only [outOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases h : i.val < n_cols
  · simp [defaultSpec, inOffset, h]
  · simp [h]
/-- Algorithm-layer correctness for the log-target backward one-block slice. -/
theorem kldiv_backward_log_target_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s new_grads_stride i))
    (hExec : exec (kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem new_grads_ptr (outOffset s new_grads_stride i) =
        if i.val < n_cols then
          logTargetSpec s target_ptr target_stride i
        else s.readMem new_grads_ptr (outOffset s new_grads_stride i) := by
  intro i
  simp [exec, kldiv_backward_log_target, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  rw [← hExec]
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * new_grads_stride + idx.1.val) := by
    intro a b h
    exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) rfl
  simp only [outOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases h : i.val < n_cols
  · simp [logTargetSpec, inOffset, h]
  · simp [h]
/-- Compute-facing correctness for the default backward one-block slice. -/
theorem kldiv_backward_default_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, outOffset s new_grads_stride i)))
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
theorem kldiv_backward_log_target_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, outOffset s new_grads_stride i)))
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
end VeriTile.Bench.TritonBenchG.KldivOps
