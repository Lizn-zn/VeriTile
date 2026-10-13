import bench.tritonbench_g.l2_norm_triton1.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.L2NormTriton1
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def baseStmt (commuted : Bool) (r : RegionName) (reg : RegName) (stride : Nat) : Stmt :=
  .assign .ptr [] reg (.ptrAdd .nil (.ptrBase r)
    (if commuted then .mul .nat .nil (.constNat stride) (.ref .nat [] "row")
      else .mul .nat .nil (.ref .nat [] "row") (.constNat stride)))

def mathKernel (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Kernel :=
  .mk [X] [Y] (.assign .nat [] "row" (.programId 0) ::
    baseStmt commuted X "X" stride :: baseStmt commuted Y "Y" stride ::
      (l2_norm_fwd_1pass_kernel X Y stride N eps B).toAlgKernel.body.drop 3)

theorem typed_projects (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedKernel storage v X Y stride N eps B).toAlgorithm? =
      .ok (typedKernel storage v X Y stride N eps B).toAlgKernel := by
  cases storage <;> cases v <;> rfl

theorem typed_supported (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedKernel storage v X Y stride N eps B).eraseDType.toAlgorithm? =
      .ok (typedKernel storage v X Y stride N eps B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]

theorem typed_projection (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedKernel storage v X Y stride N eps B).eraseDType.toAlgKernel =
      mathKernel v X Y stride N eps B := by
  simp only [ComputeKernel.eraseDType,typed_projects,ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases v
  all_goals simp only [typedKernel,mathKernel,baseStmt,Bool.false_eq_true,↓reduceIte,
    l2_norm_fwd_1pass_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,FloatDType.toTileDType,ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    FloatDType.toTileDType,ComputeDType.eraseDType]
  all_goals rfl

/-- Rebound pointer assignments retain the two source IO ports. -/
theorem io_signature (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedKernel storage v X Y stride N eps B).inputs = [X] ∧
      (typedKernel storage v X Y stride N eps B).outputs = [Y] := by
  cases storage <;> cases v <;> exact ⟨rfl,rfl⟩

/-- Runtime epsilon is explicitly converted to fp32 before sqrt and division;
the complete expression retains its explicit fp32 compute precision. -/
theorem epsilon_precision (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedKernel storage v X Y stride N eps B).surfaceBody[7]? =
      some (.assign .real [] "rstd" (.compute (.alg .fp32 (.div .real .nil (.const 1)
        (.sqrt (.add .real .nil (.ref .real [] "var")
          (.castFloat .real .real (.const eps)))))))) := by
  cases storage <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton1
