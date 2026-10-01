/- Real correctness of stable softmax with per-lane division and with a
shared reciprocal. The original bf16 output cast is retained in both sources. -/
import bench.examples.SoftmaxStableCorrect
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.SoftmaxReciprocalCorrect

open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Optimized stable softmax: precompute `1 / S` once, then multiply per lane
(`y = e · S⁻¹`), stored bf16. -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Executable real projection of reciprocal multiplication. -/
def recipMathKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, y)
}


theorem div_projection (xReg yReg : RegionName) (B : Nat) :
    (stableSoftmaxKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (OnlineSoftmax.batchSoftmaxKernel xReg yReg B).toAlgKernel :=
  SoftmaxStableCorrect.stable_projection xReg yReg B

theorem recip_projection (xReg yReg : RegionName) (B : Nat) :
    (softmaxRecipKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (recipMathKernel xReg yReg B).toAlgKernel := by
  have hcast : (Op.castFloat .real .bf16 (Op.ref .real [B] "y")).eraseDType =
      Op.ref .real [B] "y" := by
    rw [Op.eraseDType_castFloat .real .bf16 (Op.ref .real [B] "y")]
    change (Op.ref .real [B] "y").eraseDType = Op.ref .real [B] "y"
    rw [Op.eraseDType_ref]
    rfl
  simp [softmaxRecipKernel, recipMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals first
    | exact hcast
    | (rw [Op.eraseDType.eq_def]; simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType])

theorem recip_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (recipMathKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, recipMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem recip_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (recipMathKernel xReg yReg B).toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hx hy
  simp [recipMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

theorem recip_region_run (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (recipMathKernel "x" "y" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "y" (s.pid * B + i.val) = naiveSpec xs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  have hinj : Function.Injective (fun i : TileIndex [n+1] => s.pids 0 * (n+1) + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hx ⊢
  simp [recipMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp only [hx, ← div_eq_mul_inv]
    change stableSoftmaxMath xs (tileMax hB xs) i = naiveSoftmaxMath xs i
    exact (congrFun (naive_eq_stable xs (tileMax hB xs)) i).symm
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

def divIO (B : Nat) : KernelIO₁ where
  kernel := (stableSoftmaxKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, div_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def recipIO (B : Nat) : KernelIO₁ where
  kernel := (softmaxRecipKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, recip_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification softmax_div_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (divIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) :=
  SoftmaxStableCorrect.stable_softmax_correct B hB

specification softmax_reciprocal_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (recipIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  have halg : (recipIO B).kernel.toAlgKernel = (recipMathKernel "x" "y" B).toAlgKernel := by
    simp [recipIO, ComputeKernel.toAlgKernel, recip_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact recip_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact recip_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := recip_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.SoftmaxReciprocalCorrect
