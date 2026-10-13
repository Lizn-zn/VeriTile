import bench.tritonbench_g.max_reduction.Kernels
import bench.tritonbench_g.max_reduction.Proofs.IndexRewrite
import bench.tritonbench_g.max_reduction.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.MaxReduction.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def stage1Program (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) : Program where
  kernel := stage1 storage v inp out M B
  domain := []

def stage2Program (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) : Program where
  kernel := stage2 storage v inp out M B
  domain := []

/-- Complete first-stage typed executions agree after commuting the natural
index product. Every floating load, reduction, narrowing cast and store stays
in the same order and uses the same backend. No floating atom is admitted. -/
specification stage1_index_equiv (storage : Storage) (inp out : RegionName) (M B : Nat) :
    stage1Program storage Bool.false inp out M B ≡[assumptions]
      stage1Program storage Bool.true inp out M B := by
  cases storage <;> equiv_decompose <;> fp_prove

/-- Complete second-stage typed executions agree after complementing the
reversed natural bound. The full input mask, not a precomputed maximum, is
rewritten. No floating atom is admitted. -/
specification stage2_mask_equiv (storage : Storage) (inp out : RegionName) (M B : Nat) :
    stage2Program storage Bool.false inp out M B ≡[assumptions]
      stage2Program storage Bool.true inp out M B := by
  cases storage <;> equiv_decompose <;> fp_prove

def dimensionProgram (storage : Storage) (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) : Program where
  kernel := dimensionSource storage v inp ov oi M N K BM BN
  domain := []

/-- The complete dimension kernel preserves both floating values and signed
integer output indices while commuting only the natural row-index product.
Both selected value and index retain the paired indexed-max backend. -/
specification dimension_index_equiv (storage : Storage) (inp ov oi : RegionName)
    (M N K BM BN : Nat) :
    dimensionProgram storage Bool.false inp ov oi M N K BM BN ≡[assumptions]
      dimensionProgram storage Bool.true inp ov oi M N K BM BN := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions dimension_index_equiv
#print_fp_assumptions stage1_index_equiv
#print_fp_assumptions stage2_mask_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction.FPEquiv
