import bench.tritonbench_g.fast_layernorm.Proofs.FP
/-! Whole typed sources, all 27 input/weight/bias storage combinations, empty
numerical assumptions and empty input domain. The candidate commutes only the
natural-number row-stride multiplication. Casts, complete final states and
failure are preserved. Numerical work follows explicit fp32 load conversions;
Y/dY stores convert to input storage, and r/mu use fp32. -/
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (input weight bias : Storage) (swapped : Bool)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Program where
  kernel := forward input weight bias swapped Y X W b r mu YR XR N B eps
  domain := []

specification forward_equiv (input weight bias : Storage)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    forwardProgram input weight bias Bool.false Y X W b r mu YR XR N B eps ≡[assumptions]
    forwardProgram input weight bias Bool.true Y X W b r mu YR XR N B eps := by
  exact FPProof.forward_equivalent input weight bias Y X W b r mu YR XR N B eps

#print_fp_assumptions forward_equiv
def backwardProgram (input weight bias : Storage) (swapped : Bool)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Program where
  kernel := backward input weight bias swapped Y X W b r mu YR XR N B eps
  domain := []

specification backward_equiv (input weight bias : Storage)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    backwardProgram input weight bias Bool.false Y X W b r mu YR XR N B eps ≡[assumptions]
    backwardProgram input weight bias Bool.true Y X W b r mu YR XR N B eps := by
  exact FPProof.backward_equivalent input weight bias Y X W b r mu YR XR N B eps

#print_fp_assumptions backward_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastLayernorm.Equivalence
