import bench.tritonbench_g.var_len_copy.Proofs.DiscreteFP
import bench.tritonbench_g.var_len_copy.Proofs.DiscreteExecution
import bench.tritonbench_g.var_len_copy.Proofs.FP
import bench.tritonbench_g.var_len_copy.Proofs.SourceChecks
import bench.tritonbench_g.var_len_copy.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FPEquiv
open VeriTile Triton TypedSource FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- No floating algebraic law is used by signed address commutation. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []

/-- The engine uses its default fp32 profile; source fp64 and storage conversion
nodes retain their explicit local precision. Metadata widths share Int semantics. -/
def program (input output : Storage) (oldWidth lengthWidth newWidth : Metadata) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Program where
  kernel := typedKernel input output oldWidth lengthWidth newWidth v os ol ns src dst B
  domain := []

/-- Complete state and failure equality covers every signed loop iteration,
all aliases and every floating storage pair, including the source's overcopy. -/
specification source_equivalence (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    program input output oldWidth lengthWidth newWidth Bool.false os ol ns src dst B ≡[assumptions]
      program input output oldWidth lengthWidth newWidth Bool.true os ol ns src dst B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases input <;> cases output <;> rfl)
  intro α _ M D _ s _
  have h := FPProof.source_equivalence (engine M) input output os ol ns src dst B s
  have e (k : ComputeKernel) : FP.Structural.exec (engine M) k s =
      FP.Structural.run (engine M) k.surfaceBody s := by cases k; rfl
  rw [e, e] at h
  exact h

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VarLenCopy.FPEquiv

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

/-- Homogeneous discrete profiles perform no payload conversion. -/
def discrete_program (storage : DiscreteSource.Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Program where
  kernel := DiscreteSource.kernel storage v os ol ns src dst B
  domain := []

/-- Complete typed state and failure equality for all nine discrete profiles. -/
specification discrete_source_equivalence (storage : DiscreteSource.Storage)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    discrete_program storage Bool.false os ol ns src dst B ≡[assumptions]
      discrete_program storage Bool.true os ol ns src dst B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases storage <;> rfl)
  intro α _ M D _ s _
  have h := DiscreteFP.source_equivalence (engine M) storage os ol ns src dst B s
  have e (k : ComputeKernel) : FP.Structural.exec (engine M) k s =
      FP.Structural.run (engine M) k.surfaceBody s := by cases k; rfl
  rw [e, e] at h
  exact h

#print_fp_assumptions discrete_source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VarLenCopy.FPEquiv
