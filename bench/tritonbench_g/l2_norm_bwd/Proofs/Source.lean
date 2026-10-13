import bench.tritonbench_g.l2_norm_bwd.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.L2NormBwd
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def baseStmt (commuted : Bool) (r : RegionName) (reg : RegName) (stride : Nat) : Stmt :=
  .assign .ptr [] reg (.ptrAdd .nil (.ptrBase r)
    (if commuted then .mul .nat .nil (.constNat stride) (.ref .nat [] "row")
      else .mul .nat .nil (.ref .nat [] "row") (.constNat stride)))

def mathBwd (commuted : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Kernel :=
  .mk [X, DY] [DX] (.assign .nat [] "row" (.programId 0) ::
    baseStmt commuted X "X" stride :: baseStmt commuted DX "DX" stride :: baseStmt commuted DY "DY" stride ::
      (l2_norm_bwd_kernel X DY DX stride N eps B).toAlgKernel.body.drop 4)

theorem bwd_projects (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).toAlgorithm? =
      .ok (typedBwd storage gradient v X DY DX stride N eps B).toAlgKernel := by
  cases storage <;> cases gradient <;> cases v <;> rfl

theorem bwd_supported (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).eraseDType.toAlgorithm? =
      .ok (typedBwd storage gradient v X DY DX stride N eps B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,bwd_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]

theorem bwd_projection (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).eraseDType.toAlgKernel =
      mathBwd v X DY DX stride N eps B := by
  simp only [ComputeKernel.eraseDType,bwd_projects,ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases gradient <;> cases v
  all_goals simp only [typedBwd,mathBwd,baseStmt,Bool.false_eq_true,↓reduceIte,
    l2_norm_bwd_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,FloatDType.toTileDType,ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    FloatDType.toTileDType,ComputeDType.eraseDType]
  all_goals rfl

theorem bwd_io (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).inputs = [X, DY] ∧
      (typedBwd storage gradient v X DY DX stride N eps B).outputs = [DX] := by
  cases storage <;> cases gradient <;> cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormBwd
