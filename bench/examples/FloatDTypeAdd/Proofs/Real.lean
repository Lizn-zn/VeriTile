import bench.examples.FloatDTypeAdd.Kernels
import bench.examples.VectorAdd.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for FloatDTypeAdd.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

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

theorem optimized_projection (B : Nat) :
    (optimizedKernel B).eraseDType.toAlgorithm? =
      Except.ok (VectorAdd.Kernels.optimizedKernel B).toAlgKernel := by
  have hcast : (Op.castFloat .real .fp32 (Op.ref .real [B] "out")).eraseDType =
      Op.ref .real [B] "out" := by
    rw [Op.eraseDType_castFloat .real .fp32 (Op.ref .real [B] "out")]
    change (Op.ref .real [B] "out").eraseDType = Op.ref .real [B] "out"
    rw [Op.eraseDType_ref]
    rfl
  simp [optimizedKernel, VectorAdd.Kernels.optimizedKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  exact hcast

end VeriTile.Bench.Examples.FloatDTypeAddCorrect
