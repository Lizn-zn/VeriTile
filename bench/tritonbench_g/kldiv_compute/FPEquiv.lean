import bench.tritonbench_g.kldiv_compute.Proofs.FP
import bench.tritonbench_g.kldiv_compute.Proofs.Precision
import bench.tritonbench_g.kldiv_compute.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.KldivCompute.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) : Program where
  kernel := kernel x y output v X Y O N B
  domain := []

/-- Complete typed state, memory and failure equivalence for every admitted
source floating triple and every initial state, without a numerical rewrite assumption. -/
specification source_equivalence (x y output : Storage)
    (_hSource : sourceSupported x y = Bool.true)
    (X Y O : RegionName) (N B : Nat) :
    program x y output Bool.false X Y O N B ≡[assumptions]
      program x y output Bool.true X Y O N B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases x <;> cases y <;> cases output <;> rfl)
  intro α _ M D _ s _
  exact FPProof.source_equivalence M x y output X Y O N B s

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute.FPEquiv
