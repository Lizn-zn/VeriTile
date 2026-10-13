import bench.tritonbench_g.l2_norm_triton2.Kernels
import bench.tritonbench_g.l2_norm_triton2.Proofs.FP
import bench.tritonbench_g.l2_norm_triton2.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.L2NormTriton2.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forward (storage : Storage) (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typedFwd storage commuted X Y stride N eps B
  domain := []

def backward (storage gradient : Storage) (commuted : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typedBwd storage gradient commuted X DY DX stride N eps B
  domain := []

/-- Both complete forward programs preserve every execution outcome. The changed
row/stride products are exact natural-number address arithmetic. -/
specification forward_index_equiv (storage : Storage) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (_hB : 0 < B) :
    forward storage Bool.false X Y stride N eps B ≡[assumptions]
      forward storage Bool.true X Y stride N eps B := by
  cases storage <;> equiv_decompose <;> fp_prove

/-- Both complete backward programs preserve every execution outcome, including
all registers, memory and failure. Neither reduction nor arithmetic is reordered. -/
specification backward_index_equiv (storage gradient : Storage) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (_hB : 0 < B) :
    backward storage gradient Bool.false X DY DX stride N eps B ≡[assumptions]
      backward storage gradient Bool.true X DY DX stride N eps B := by
  cases storage <;> cases gradient <;> equiv_decompose <;> fp_prove

#print_fp_assumptions forward_index_equiv
#print_fp_assumptions backward_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton2.FPEquiv
