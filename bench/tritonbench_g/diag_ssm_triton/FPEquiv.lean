import bench.tritonbench_g.diag_ssm_triton.Proofs.FP
/-! Equivalence of all four complete typed JITs and their index-commuted
candidates in Kernels.lean. The only change is the natural-number product
forming col_idx. Precision, masks, complex gradient signs, branches, loops,
complete final states and failure are preserved. No numerical atom or
operand-domain premise is needed for this structural transformation. -/
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardOriginal (storage : Storage) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.forward storage Bool.false s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE
  domain := []

def forwardIndexCommuted (storage : Storage) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.forward storage Bool.true s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE
  domain := []

specification forward_equiv (storage : Storage) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    forwardOriginal storage s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      forwardIndexCommuted storage s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE := by
  exact FPProof.forward_equivalent storage s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE

#print_fp_assumptions forward_equiv

def backwardOriginal (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.backward storage Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  domain := []

def backwardIndexCommuted (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.backward storage Bool.true s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  domain := []

specification backward_equiv (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    backwardOriginal storage s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      backwardIndexCommuted storage s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE := by
  exact FPProof.backward_equivalent storage s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE

#print_fp_assumptions backward_equiv

def complexForwardOriginal (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.complexForward Bool.false s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE
  domain := []

def complexForwardIndexCommuted (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.complexForward Bool.true s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE
  domain := []

specification complexForward_equiv (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    complexForwardOriginal s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      complexForwardIndexCommuted s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE := by
  exact FPProof.complexForward_equivalent s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE

#print_fp_assumptions complexForward_equiv

def complexBackwardOriginal (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.complexBackward Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  domain := []

def complexBackwardIndexCommuted (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.complexBackward Bool.true s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  domain := []

specification complexBackward_equiv (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    complexBackwardOriginal s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      complexBackwardIndexCommuted s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE := by
  exact FPProof.complexBackward_equivalent s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE

#print_fp_assumptions complexBackward_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.Equivalence
