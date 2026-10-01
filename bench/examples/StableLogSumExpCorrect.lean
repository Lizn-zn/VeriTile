/- Real correctness of direct and max-shifted logsumexp. Both source kernels
retain their bf16 store; only the mathematical interpretation erases dtype. -/
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.StableLogSumExpCorrect

open VeriTile Triton Triton.TiledLogSumExp
open scoped VeriTile.Triton.KernelIO₁

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

/-- Numerically-stable log-sum-exp: subtract the row max first,
`y = m + log Σ exp(x − m)`, stored rounded to bf16. -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

/-- Executable real projections used by the correctness proofs. -/
def directMathKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Real projection of the shifted logsumexp, with the output cast erased. -/
def stableMathKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, y)
}


private theorem erase_output_cast :
    (Op.castFloat .real .bf16 (Op.ref .real [] "y")).eraseDType =
      Op.ref .real [] "y" := by
  rw [Op.eraseDType_castFloat .real .bf16 (Op.ref .real [] "y")]
  change (Op.ref .real [] "y").eraseDType = Op.ref .real [] "y"
  rw [Op.eraseDType_ref]
  rfl

theorem direct_projection (xReg yReg : RegionName) (B : Nat) :
    (directLSEKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (directMathKernel xReg yReg B).toAlgKernel := by
  simp [directLSEKernel, directMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals first
    | exact erase_output_cast
    | (rw [Op.eraseDType.eq_def]; simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType])

theorem direct_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (directMathKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, directMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem direct_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid + 1 ≤ bounds yReg) :
    Kernel.TraceSafe bounds (directMathKernel xReg yReg B).toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hx hy
  simp [directMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop, Tile.vec,
    NumericDType.add, NumericDType.mul,
    Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx, hy⟩

theorem direct_region_run (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (directMathKernel "x" "y" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin 1, s1.readMem "y" (s.pid + i.val) = Real.log (∑ i, Real.exp (xs i))) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin 1, o ≠ s.pid + i.val) →
        s1.mem r o = s.mem r o) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  simp only [BlockState.pid_eq] at hx ⊢
  simp [directMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
    Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨?_, ?_⟩
  · simp [hx]
    try rfl
  · intro r o hmiss
    rw [BlockState.writeMem_mem]
    apply if_neg
    rintro ⟨rfl, rfl⟩
    rcases hmiss with h | h
    · exact h rfl
    · exact h rfl

def directIO (B : Nat) : KernelIO₁ where
  kernel := (directLSEKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, direct_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

specification direct_logsumexp_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (directIO B ⊨ fun xs _ => Real.log (∑ i, Real.exp (xs i))) := by
  have halg : (directIO B).kernel.toAlgKernel = (directMathKernel "x" "y" B).toAlgKernel := by
    simp [directIO, ComputeKernel.toAlgKernel, direct_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact direct_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact direct_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := direct_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

theorem stable_projection (xReg yReg : RegionName) (B : Nat) :
    (stableLSEKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (stableMathKernel xReg yReg B).toAlgKernel := by
  simp [stableLSEKernel, stableMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals first
    | exact erase_output_cast
    | (rw [Op.eraseDType.eq_def]; simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType])

theorem stable_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (stableMathKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, stableMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem stable_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid + 1 ≤ bounds yReg) :
    Kernel.TraceSafe bounds (stableMathKernel xReg yReg B).toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hx hy
  simp [stableMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop, Tile.vec,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx, hy⟩

theorem stable_region_run (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (stableMathKernel "x" "y" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin 1, s1.readMem "y" (s.pid + i.val) = stableLSEWithShift xs (blockMax hB xs)) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin 1, o ≠ s.pid + i.val) →
        s1.mem r o = s.mem r o) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  simp only [BlockState.pid_eq] at hx ⊢
  simp [stableMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨?_, ?_⟩
  · simp [hx, stableLSEWithShift, TiledLogSumExp.blockMax]
    try rfl
  · intro r o hmiss
    rw [BlockState.writeMem_mem]
    apply if_neg
    rintro ⟨rfl, rfl⟩
    rcases hmiss with h | h
    · exact h rfl
    · exact h rfl

def stableIO (B : Nat) : KernelIO₁ where
  kernel := (stableLSEKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, stable_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

specification stable_logsumexp_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (stableIO B ⊨ fun xs _ => Real.log (∑ i, Real.exp (xs i))) := by
  have halg : (stableIO B).kernel.toAlgKernel = (stableMathKernel "x" "y" B).toAlgKernel := by
    simp [stableIO, ComputeKernel.toAlgKernel, stable_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact stable_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact stable_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := stable_region_run B hB s xs hx
    refine ⟨s1, he, fun i => ?_, fun r o hmiss _ => hf r o hmiss⟩
    change s1.readMem "y" (s.pid + i.val) = _
    rw [hv i]
    exact (log_sum_exp_shift_invariant hB xs (blockMax hB xs)).symm

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.StableLogSumExpCorrect
