import bench.tritonbench_g.rms_norm_triton.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def baseStmt (commuted : Bool) (r : RegionName) (reg : RegName) (stride : Nat) : Stmt :=
  .assign .ptr [] reg (.ptrAdd .nil (.ptrBase r)
    (if commuted then .mul .nat .nil (.constNat stride) (.ref .nat [] "pid")
      else .mul .nat .nil (.ref .nat [] "pid") (.constNat stride)))

def mathKernel (commuted : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) : Kernel :=
  .mk [X, W] [Y] (.assign .nat [] "pid" (.programId 0) ::
    baseStmt commuted Y "Y" ysr :: baseStmt commuted X "X" xsr ::
      (rms_norm_kernel Y X W ysr ysc xsr xsc N eps B).toAlgKernel.body.drop 3)

theorem typed_projects (storage weight : Storage) (v : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    (typed storage weight v Y X W ysr ysc xsr xsc N eps B).toAlgorithm? =
      .ok (typed storage weight v Y X W ysr ysc xsr xsc N eps B).toAlgKernel := by
  cases storage <;> cases weight <;> cases v <;> rfl

theorem typed_supported (storage weight : Storage) (v : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    (typed storage weight v Y X W ysr ysc xsr xsc N eps B).eraseDType.toAlgorithm? =
      .ok (typed storage weight v Y X W ysr ysc xsr xsc N eps B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]

theorem typed_projection (storage weight : Storage) (v : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    (typed storage weight v Y X W ysr ysc xsr xsc N eps B).eraseDType.toAlgKernel =
      mathKernel v Y X W ysr ysc xsr xsc N eps B := by
  simp only [ComputeKernel.eraseDType,typed_projects,ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases weight <;> cases v
  all_goals simp only [typed,mathKernel,baseStmt,Bool.false_eq_true,↓reduceIte,
    rms_norm_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,FloatDType.toTileDType,ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    FloatDType.toTileDType,ComputeDType.eraseDType]
  all_goals rfl

/-- Rebound pointer assignments retain the two source IO ports. -/
theorem io_signature (storage weight : Storage) (v : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    (typed storage weight v Y X W ysr ysc xsr xsc N eps B).inputs = [X, W] ∧
      (typed storage weight v Y X W ysr ysc xsr xsc N eps B).outputs = [Y] := by
  cases storage <;> cases weight <;> cases v <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton
