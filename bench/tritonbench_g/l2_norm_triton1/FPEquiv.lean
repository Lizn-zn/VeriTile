import bench.tritonbench_g.l2_norm_triton1.Kernels
import bench.tritonbench_g.l2_norm_triton1.Proofs.FP
import bench.tritonbench_g.l2_norm_triton1.Proofs.SourceChecks
import bench.tritonbench_g.l2_norm_triton1.Proofs.NumericWitness

/-! Complete typed-state/failure equivalence. X and Y share storage dtype;
loads convert to fp32, runtime epsilon is fp32, and stores convert back to the
pointer dtype. The default fp32 profile retains sqrt, sum order, division and
all masks. Exact natural address commutation needs no numerical domain or atom. -/
namespace VeriTile.Bench.TritonBenchG.L2NormTriton1.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 2000000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (storage : Storage) (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typedKernel storage commuted X Y stride N eps B
  domain := []

specification source_equivalence (storage : Storage) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    program storage Bool.false X Y stride N eps B ≡[assumptions]
      program storage Bool.true X Y stride N eps B := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton1.FPEquiv
