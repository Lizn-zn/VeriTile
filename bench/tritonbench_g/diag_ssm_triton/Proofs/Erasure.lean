import bench.tritonbench_g.diag_ssm_triton.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- The same full sources interpreted over real arithmetic, solely for
mathematical correctness. Floating equivalence never uses these erasures. -/
def forward (swapped : Bool) : forwardSource :=
  if swapped then diagSsmForwardSource%[tl.float64, indexCommuted]
  else diagSsmForwardSource%[tl.float64, original]

def backward (swapped : Bool) : backwardSource :=
  if swapped then diagSsmBackwardSource%[tl.float64, indexCommuted]
  else diagSsmBackwardSource%[tl.float64, original]

def complexForward (swapped : Bool) : complexForwardSource :=
  if swapped then diagSsmComplexForwardSource%[tl.float64, indexCommuted]
  else diagSsmComplexForwardSource%[tl.float64, original]

def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float64, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float64, original]

theorem forward_body (storage : Storage) (swapped : Bool) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.forward storage swapped s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (forward swapped s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases storage <;> cases swapped
  all_goals simp only [TypedSource.forward,forward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

theorem backward_body (storage : Storage) (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.backward storage swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (backward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases storage <;> cases swapped
  all_goals simp only [TypedSource.backward,backward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

theorem complexForward_body (swapped : Bool) (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.complexForward swapped s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (complexForward swapped s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases swapped
  all_goals simp only [TypedSource.complexForward,complexForward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [
    VeriTile.Triton.eraseDType]
  all_goals rfl

theorem complexBackward_body (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.complexBackward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (complexBackward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases swapped
  all_goals simp only [TypedSource.complexBackward,complexBackward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [
    VeriTile.Triton.eraseDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.Erasure
