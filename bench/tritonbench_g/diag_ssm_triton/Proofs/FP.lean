import bench.tritonbench_g.diag_ssm_triton.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (storage : Storage) (swapped : Bool) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.forward storage swapped s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE
  domain := []

theorem forward_equivalent (storage : Storage) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    forwardProgram storage Bool.false s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      forwardProgram storage Bool.true s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions forward_equivalent

def backwardProgram (storage : Storage) (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.backward storage swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  domain := []

theorem backward_equivalent (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    backwardProgram storage Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      backwardProgram storage Bool.true s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions backward_equivalent

def complexForwardProgram (swapped : Bool) (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.complexForward swapped s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE
  domain := []

theorem complexForward_equivalent (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    complexForwardProgram Bool.false s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      complexForwardProgram Bool.true s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE := by
  equiv_decompose; fp_prove

#print_fp_assumptions complexForward_equivalent

def complexBackwardProgram (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) : Program where
  kernel := TypedSource.complexBackward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  domain := []

theorem complexBackward_equivalent (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    complexBackwardProgram Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ≡[assumptions]
      complexBackwardProgram Bool.true s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE := by
  equiv_decompose; fp_prove

#print_fp_assumptions complexBackward_equivalent

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.FPProof
