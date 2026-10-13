import bench.tritonbench_g.swiglu_backward.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SwigluBackward
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def loadAlgorithm (pointer name : RegName) (storage : Storage) (B N : Nat) : Stmt :=
  match (typedLoad pointer name storage B N).toAlgorithm? with
  | .ok st => st
  | .error _ => .ifThen (.constBool Bool.false) []

def storeAlgorithm (pointer name : RegName) (storage : Storage) (B N : Nat) : Stmt :=
  match (typedStore pointer name storage B N).toAlgorithm? with
  | .ok st => st
  | .error _ => .ifThen (.constBool Bool.false) []

theorem load_projects (pointer name : RegName) (storage : Storage) (B N : Nat) :
    (typedLoad pointer name storage B N).toAlgorithm? =
      .ok (loadAlgorithm pointer name storage B N) := by cases storage <;> rfl

theorem store_projects (pointer name : RegName) (storage : Storage) (B N : Nat) :
    (typedStore pointer name storage B N).toAlgorithm? =
      .ok (storeAlgorithm pointer name storage B N) := by cases storage <;> rfl

theorem load_erasure (pointer name : RegName) (storage : Storage) (B N : Nat) :
    (loadAlgorithm pointer name storage B N).eraseDType =
      (loadAlgorithm pointer name .fp32 B N).eraseDType := by
  cases storage <;> simp [loadAlgorithm, typedLoad, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, loadAddress, VeriTile.Triton.eraseDType]

theorem store_erasure (pointer name : RegName) (storage : Storage) (B N : Nat) :
    (storeAlgorithm pointer name storage B N).eraseDType =
      (storeAlgorithm pointer name .fp32 B N).eraseDType := by
  cases storage <;> simp [storeAlgorithm, typedStore, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, storeAddress, VeriTile.Triton.eraseDType]

def mathKernel (v : Bool) (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy N B : Nat) (R : Bool) : Kernel :=
  (template v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).eraseDType.toAlgKernel

theorem typed_projects (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).toAlgorithm? =
      .ok (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).toAlgKernel := by
  cases v <;> simp [typedKernel, template, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?,
    load_projects, store_projects, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, recomputedValue, Except.bind, bind]

theorem typed_projection (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).eraseDType.toAlgKernel =
      mathKernel v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R := by
  cases v <;> simp only [typedKernel, template, mathKernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    load_projects, store_projects, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, recomputedValue, Except.bind, bind, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, load_erasure, store_erasure,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, Bool.false_eq_true,
    ↓reduceIte, List.take, List.drop, List.nil_append, List.cons_append]
  all_goals simp [loadAlgorithm, storeAlgorithm, typedLoad, typedStore,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Except.bind, bind, Stmt.eraseDType, loadAddress, storeAddress,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType]

theorem typed_supported (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).eraseDType.toAlgorithm? =
      .ok (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward
