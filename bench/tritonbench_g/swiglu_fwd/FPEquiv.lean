import bench.tritonbench_g.swiglu_fwd.Proofs.Execution
import bench.tritonbench_g.swiglu_fwd.Kernels
import bench.tritonbench_g.swiglu_fwd.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open FPProof
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : Program where
  kernel := typedKernel input partner output v X Y OUT sx sy so N B
  domain := []

theorem decomposition (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).surfaceBody =
      sourcePrefix v X Y OUT sx sy so B ++
        (typedKernel input partner output Bool.false X Y OUT sx sy so N B).surfaceBody.drop 5 := by
  cases input <;> cases partner <;> cases output <;> cases v <;> rfl

/-- Both complete floating executions have equal state, memory and failure behavior;
only three natural row-address products commute, with all floating operations retained. -/
specification source_equivalence (input partner output : Storage)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    program input partner output Bool.false X Y OUT sx sy so N B ≡[assumptions]
      program input partner output Bool.true X Y OUT sx sy so N B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases input <;> cases partner <;> cases output <;> rfl)
  intro α _ M D _ s _
  change run (engine M)
      (typedKernel input partner output Bool.false X Y OUT sx sy so N B).surfaceBody s =
    run (engine M)
      (typedKernel input partner output Bool.true X Y OUT sx sy so N B).surfaceBody s
  rw [decomposition input partner output Bool.false X Y OUT sx sy so N B,
    decomposition input partner output Bool.true X Y OUT sx sy so N B]
  exact prefix_run M s X Y OUT sx sy so B _

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd.FPEquiv
