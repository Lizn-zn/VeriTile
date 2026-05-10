import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KldivTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented one-block slice of `kldiv_triton.py`'s
`_kldiv_kernel_backward` for the default `log_target = False` branch.

The upstream kernel loops over `range(0, n_cols, BLOCK_SIZE)`. This slice
captures the per-block masked load, pointer arithmetic, negation, and masked
store contract without introducing a loop invariant. -/
def kldiv_backward_default_one_block
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  input_base = input_ptr + pid * $(input_stride)
  target_base = target_ptr + pid * $(target_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_base + offsets, mask=mask, other=0.0)
  res = 0.0 - target
  tl.store(input_base + offsets, res, mask=mask)
}

/-- Proof-oriented one-block slice of `_kldiv_kernel_backward` for
`log_target = True`. -/
def kldiv_backward_log_target_one_block
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  input_base = input_ptr + pid * $(input_stride)
  target_base = target_ptr + pid * $(target_stride)
  offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_cols)
  target = tl.load(target_base + offsets, mask=mask, other=0.0)
  res = 0.0 - tl.exp(target)
  tl.store(input_base + offsets, res, mask=mask)
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
  0.0 - s.readMem target_ptr (inOffset s target_stride i)

noncomputable def logTargetSpec
    (s : BlockState) (target_ptr : RegionName) (target_stride : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  0.0 - Real.exp (s.readMem target_ptr (inOffset s target_stride i))

/-- Algorithm-layer correctness for the default backward one-block slice. -/
theorem kldiv_backward_default_one_block_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i))
    (hExec : exec (kldiv_backward_default_one_block input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem input_ptr (outOffset s input_stride i) =
        if i.val < n_cols then
          defaultSpec s target_ptr target_stride i
        else s.readMem input_ptr (outOffset s input_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * input_stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, kldiv_backward_default_one_block, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
          NumericDType.sub, ComparableDType.lt] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, defaultSpec, inOffset, NumericDType.sub]
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Algorithm-layer correctness for the log-target backward one-block slice. -/
theorem kldiv_backward_log_target_one_block_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i))
    (hExec : exec (kldiv_backward_log_target_one_block input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem input_ptr (outOffset s input_stride i) =
        if i.val < n_cols then
          logTargetSpec s target_ptr target_stride i
        else s.readMem input_ptr (outOffset s input_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * input_stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, kldiv_backward_log_target_one_block, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add,
          NumericDType.mul, NumericDType.sub, ComparableDType.lt] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, logTargetSpec, inOffset, WithBot.realExp]
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the default backward one-block slice. -/
theorem kldiv_backward_default_one_block_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_default_one_block input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, outOffset s input_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_default_one_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_default_one_block_correct input_ptr target_ptr
    input_stride target_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- Compute-facing correctness for the log-target backward one-block slice. -/
theorem kldiv_backward_log_target_one_block_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => outOffset s input_stride i)) :
    ComputeCorrect.Realizes
      (kernel := kldiv_backward_log_target_one_block input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, outOffset s input_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_log_target_one_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_log_target_one_block_correct input_ptr target_ptr
    input_stride target_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.KldivTriton
