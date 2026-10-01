/- Real correctness of both original softmax implementations. The public
formula is exp(x[i]) / sum(exp(x)); dtype erasure belongs only to this real
interpretation. The numerical shift rewrite needs a separate FP proof. -/
import bench.examples.OnlineSoftmaxCorrect
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.SoftmaxStableCorrect

open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

def naiveSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  e := tl.exp(x)
  s := tl.sum(e, axis=0)
  y := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

def stableSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  m := tl.max(x, axis=0)
  e := tl.exp(x - m)
  s := tl.sum(e, axis=0)
  y := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Executable mathematical projection, used only by the real proof. -/
def naiveMathKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  e := tl.exp(x)
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
      Except.ok (OnlineSoftmax.batchSoftmaxKernel xReg yReg B).toAlgKernel := by
  simp [stableSoftmaxKernel, OnlineSoftmax.batchSoftmaxKernel,
    OnlineSoftmax.stableSoftmaxKernel, ComputeKernel.eraseDType,
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

def naiveIO (B : Nat) : KernelIO₁ where
  kernel := (naiveSoftmaxKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, naive_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def stableIO (B : Nat) : KernelIO₁ where
  kernel := (stableSoftmaxKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, stable_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification naive_softmax_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (naiveIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  have halg : (naiveIO B).kernel.toAlgKernel = (naiveMathKernel "x" "y" B).toAlgKernel := by
    simp [naiveIO, ComputeKernel.toAlgKernel, naive_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact naive_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact naive_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := naive_region_run B s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

specification stable_softmax_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (stableIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  have halg : (stableIO B).kernel.toAlgKernel =
      (OnlineSoftmax.batchSoftmaxKernel "x" "y" B).toAlgKernel := by
    simp [stableIO, ComputeKernel.toAlgKernel, stable_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact OnlineSoftmax.batchSoftmax_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact OnlineSoftmax.batchSoftmax_traceSafe "x" "y" B bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := OnlineSoftmax.batchSoftmax_region_run B hB s xs hx
    refine ⟨s1, he, fun i => ?_, fun r o hmiss _ => hf r o hmiss⟩
    change s1.readMem "y" (s.pid * B + i.val) = _
    rw [hv i, (OnlineSoftmax.online_softmax_recurrence_eq_batch hB xs).1,
      (OnlineSoftmax.online_softmax_recurrence_eq_batch hB xs).2]
    change stableSoftmaxMath xs (tileMax hB xs) i = naiveSoftmaxMath xs i
    exact (congrFun (naive_eq_stable xs (tileMax hB xs)) i).symm

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.SoftmaxStableCorrect
