import bench.examples.FloatDTypeAdd.Kernels
/- Real correctness of the float-annotated addition kernel. Dtype erasure
removes both working precision and the explicit output quantization before
interpreting the implementation as a mathematical function. -/
import bench.examples.VectorAdd.Correct
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FloatDTypeAddCorrect
open VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.KernelIO₂

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl


theorem real_projection (xReg yReg outReg : RegionName) (B : Nat) :
    (floatAddKernel xReg yReg outReg B).eraseDType.toAlgorithm? =
      Except.ok (VectorAdd.Kernels.addKernel xReg yReg outReg B).toAlgKernel := by
  have hcast : (Op.castFloat .real .fp32 (Op.ref .real [B] "out")).eraseDType =
      Op.ref .real [B] "out" := by
    rw [Op.eraseDType_castFloat .real .fp32 (Op.ref .real [B] "out")]
    change (Op.ref .real [B] "out").eraseDType = Op.ref .real [B] "out"
    rw [Op.eraseDType_ref]
    rfl
  simp [floatAddKernel, VectorAdd.Kernels.addKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  exact hcast

/-- Correctness uses the mathematical projection of the original kernel. -/
def floatAddIO (B : Nat) : KernelIO₂ where
  kernel := (floatAddKernel "x" "y" "out" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, real_projection]
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B

specification float_add_correctness (B : Nat) :
    Spec.Real (floatAddIO B ⊨ fun xs ys i => xs i + ys i) := by
  have halg : (floatAddIO B).kernel.toAlgKernel =
      (VectorAdd.Kernels.addKernel "x" "y" "out" B).toAlgKernel := by
    simp [floatAddIO, ComputeKernel.toAlgKernel, real_projection]
  refine KernelIO₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]
    exact VectorAdd.addKernel_flattenOk "x" "y" "out" B
  · intro bounds s hx hy hout
    rw [halg]
    exact VectorAdd.addKernel_traceSafe "x" "y" "out" B bounds s hx hy hout
  · intro s xs ys hx hy
    rw [halg]
    rcases Nat.eq_zero_or_pos B with rfl | hB
    · simp [floatAddIO, VectorAdd.Kernels.addKernel, exec, stepStmts, stepStmt,
        evalOp.eq_def, Tile.bop, NumericDType.add, NumericDType.mul,
        TileShape.allIndices]
    · exact VectorAdd.addKernel_region_run B hB s xs ys hx hy

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeAddCorrect
