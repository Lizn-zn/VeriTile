import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerAll
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerAll
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerAll
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerAll
import bench.tritonbench_g.softmax_flaggems.Proofs.SourceSupport

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Correctness
open VeriTile Triton

theorem forward_non_inner_domain_nonempty (one : Bool) :
    RealForwardNonInner.ExecutionDomain one "X" "Y" 1 1 := by
  cases one <;> simp [RealForwardNonInner.ExecutionDomain]

theorem forward_inner_domain_nonempty (one : Bool) :
    RealForwardInner.ExecutionDomain one "X" "Y" 1 1 := by
  cases one <;> simp [RealForwardInner.ExecutionDomain]

theorem backward_non_inner_domain_nonempty (one : Bool) :
    RealBackwardNonInner.ExecutionDomain one "P" "G" "Y" 1 1 := by
  cases one <;> simp [RealBackwardNonInner.ExecutionDomain]

theorem backward_inner_domain_nonempty (one : Bool) :
    RealBackwardInner.ExecutionDomain one "P" "G" "Y" 1 1 := by
  cases one <;> simp [RealBackwardInner.ExecutionDomain]

/-- Actual source access safety and successful region IO for arbitrary padding.
The flat IO clause exposes the zero-undef requirement of the current flattening
bridge; it is not an antecedent of the region IO clause. Both IO predicates
include a full raw-cell frame and an independent mathematical output formula. -/
def SourceCorrect (A : FlatAlloc) (s : BlockState) (kernel : ComputeKernel)
    (regionResult flatResult : BlockState → Prop) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafe A.extent s ∧
    (∃ final, exec kernel.eraseDType.toAlgKernel s = some final ∧ regionResult final) ∧
    (s.undef = (fun _ _ ↦ 0) →
      ∃ final, exec (A.flattenKernel kernel.eraseDType.toAlgKernel) (A.flattenState s) = some final ∧ flatResult final)

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Correctness
