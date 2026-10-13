import bench.tritonbench_g.softmax_flaggems.Kernels
import bench.tritonbench_g.softmax_flaggems.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Typed complete-source equivalence for a discrete address transformation.
Only the two Nat factors in the program-base address are commuted. Floating
operations, reductions, casts, unspecified padding, loops and stores are retained.
No numerical report is required. The relation preserves the entire state and
execution failure for both branches; successful mathematical IO is separate. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Equivalence
open VeriTile Triton FP.ProfiledRewrite FullSource
open scoped VeriTile.Spec
set_option maxHeartbeats 2500000
set_option maxRecDepth 10000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def program (kernel : ComputeKernel) : Program := ⟨kernel, []⟩

specification forward_non_inner_equiv (inputType outputType : Storage)
    (output_ptr input_ptr : RegionName) (M N K TN TK : Nat) (oneTile : Bool) :
    program (softmax_kernel_non_inner inputType outputType
      output_ptr input_ptr M N K TN TK oneTile) ≡[assumptions]
    program (softmax_kernel_non_inner_candidate inputType outputType
      output_ptr input_ptr M N K TN TK oneTile) := by
  cases inputType <;> cases outputType <;> equiv_decompose <;> fp_prove

specification forward_inner_equiv (inputType outputType : Storage)
    (output_ptr input_ptr : RegionName) (M N TN : Nat) (oneTile : Bool) :
    program (softmax_kernel_inner inputType outputType
      output_ptr input_ptr M N TN oneTile) ≡[assumptions]
    program (softmax_kernel_inner_candidate inputType outputType
      output_ptr input_ptr M N TN oneTile) := by
  cases inputType <;> cases outputType <;> equiv_decompose <;> fp_prove

specification backward_non_inner_equiv (storage : Storage)
    (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N K TN TK : Nat) (oneTile : Bool) :
    program (softmax_backward_kernel_non_inner storage
      out_ptr out_grad_ptr in_grad_ptr M N K TN TK oneTile) ≡[assumptions]
    program (softmax_backward_kernel_non_inner_candidate storage
      out_ptr out_grad_ptr in_grad_ptr M N K TN TK oneTile) := by
  cases storage <;> equiv_decompose <;> fp_prove

specification backward_inner_equiv (storage : Storage)
    (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N TM TN : Nat) (oneTile : Bool) :
    program (softmax_backward_kernel_inner storage
      out_ptr out_grad_ptr in_grad_ptr M N TM TN oneTile) ≡[assumptions]
    program (softmax_backward_kernel_inner_candidate storage
      out_ptr out_grad_ptr in_grad_ptr M N TM TN oneTile) := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions forward_non_inner_equiv
#print_fp_assumptions forward_inner_equiv
#print_fp_assumptions backward_non_inner_equiv
#print_fp_assumptions backward_inner_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Equivalence
