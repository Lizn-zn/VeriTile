import bench.examples.SoftmaxStable.Kernels
import bench.examples.OnlineSoftmax.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for SoftmaxStable.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.SoftmaxStableCorrect
open VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-- Executable mathematical projection, used only by the real proof. -/
def naiveMathKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  e := libdevice.exp(x)
  s := tl.sum(e, axis=0)
  y := e / s
  tl.store($(yReg) + offs, y)
}

private theorem erase_output_cast (B : Nat) :
    (Op.castFloat .real .bf16 (Op.ref .real [B] "y")).eraseDType =
      Op.ref .real [B] "y" := by
  rw [Op.eraseDType_castFloat .real .bf16 (Op.ref .real [B] "y")]
  change (Op.ref .real [B] "y").eraseDType = Op.ref .real [B] "y"
  rw [Op.eraseDType_ref]
  rfl

theorem naive_projection (xReg yReg : RegionName) (B : Nat) :
    (naiveSoftmaxKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (naiveMathKernel xReg yReg B).toAlgKernel := by
  simp [naiveSoftmaxKernel, naiveMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals first
    | exact erase_output_cast B
    | (rw [Op.eraseDType.eq_def]; simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType])

theorem stable_projection (xReg yReg : RegionName) (B : Nat) :
    (stableSoftmaxKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (OnlineSoftmax.Kernels.batchSoftmaxKernel xReg yReg B).toAlgKernel := by
  simp [stableSoftmaxKernel, OnlineSoftmax.Kernels.batchSoftmaxKernel,
    OnlineSoftmax.Kernels.stableSoftmaxKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals first
    | exact erase_output_cast B
    | (rw [Op.eraseDType.eq_def]; simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType])

/-- Value, termination, and untouched-memory obligations for the naive kernel. -/
theorem naive_region_run (B : Nat) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (naiveMathKernel "x" "y" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "y" (s.pid * B + i.val) = naiveSpec xs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hx ⊢
  simp [naiveMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.div,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hx, naiveSpec]
    rfl
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem naive_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (naiveMathKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, naiveMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem naive_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (naiveMathKernel xReg yReg B).toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hx hy
  simp [naiveMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.div, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

end VeriTile.Bench.Examples.SoftmaxStableCorrect
