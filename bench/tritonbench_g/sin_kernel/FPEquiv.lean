import bench.tritonbench_g.sin_kernel.Proofs.Execution
import bench.tritonbench_g.sin_kernel.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.SinKernel.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open FPProof
open scoped VeriTile.Spec
set_option maxHeartbeats 2000000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : Program where
  kernel := typedKernel input v X Y N B
  domain := []

theorem decomposition (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input v X Y N B).surfaceBody = sourcePrefix v B ++
      (typedKernel input Bool.false X Y N B).surfaceBody.drop 2 := by
  cases input <;> cases v <;> rfl

/-- Full floating execution equivalence from arbitrary states: same signature,
registers, memory and failure behavior; no numerical atom or domain restriction. -/
specification source_equivalence (input : InputStorage)
    (X Y : RegionName) (N B : Nat) :
    program input Bool.false X Y N B ≡[assumptions]
      program input Bool.true X Y N B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases input <;> rfl)
  intro α _ M D _ s _
  change run (engine M) (typedKernel input Bool.false X Y N B).surfaceBody s =
    run (engine M) (typedKernel input Bool.true X Y N B).surfaceBody s
  rw [decomposition input Bool.false X Y N B,
    decomposition input Bool.true X Y N B]
  exact prefix_run M s B _

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinKernel.FPEquiv
