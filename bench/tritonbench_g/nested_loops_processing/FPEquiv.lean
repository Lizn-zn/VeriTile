import bench.tritonbench_g.nested_loops_processing.Proofs.FP
import bench.tritonbench_g.nested_loops_processing.Proofs.SourceChecks
import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionWitness

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FPEquiv
open VeriTile Triton TypedSource FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

/-- The default fp32 backend profile retains explicit source-local fp64 loads.
No payload arithmetic or conversion is introduced for homogeneous storage. -/
def program (storage : Storage) (candidate : Bool)
    (src dst : RegionName) (sm sn : Int) : Program where
  kernel := kernel storage candidate src dst sm sn
  domain := []

/-- Full nested-loop state equivalence, including aliases and evaluator failures.
Source-facing address safety and nonnegative strides belong to the real IO contract. -/
specification source_equivalence (storage : Storage)
    (src dst : RegionName) (sm sn : Int) :
    program storage Bool.false src dst sm sn ≡[assumptions]
      program storage Bool.true src dst sm sn := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ ↦ False) rfl
    (by cases storage <;> rfl)
  intro α _ M D _ s _
  have h := FullFP.source_equivalence (engine M) storage src dst sm sn s
  have e (k : ComputeKernel) : FP.Structural.exec (engine M) k s =
      FP.Structural.run (engine M) k.surfaceBody s := by cases k; rfl
  rw [e, e] at h
  exact h

#print_fp_assumptions source_equivalence
end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FPEquiv

#guard_msgs (drop info) in
#auditModuleAxioms
