import bench.tritonbench_g.f8_conversion_utils.Proofs.FP
/-! Complete typed equivalence for each conversion direction and its integer
index commutation. Both forward stores, precision casts, masks, unspecified
padding, final states and failure are retained. No numerical atom is used. -/
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def originalToHalf (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toHalf format Bool.false Y X N B
  domain := []

def indexCommutedToHalf (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toHalf format Bool.true Y X N B
  domain := []

def originalToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toFP8 input format Bool.false Y X N B
  domain := []

def indexCommutedToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toFP8 input format Bool.true Y X N B
  domain := []

specification f8_to_f16_equiv (format : Format) (Y X : RegionName) (N B : Nat) :
    originalToHalf format Y X N B ≡[assumptions] indexCommutedToHalf format Y X N B := by
  exact FPProof.to_half_equivalent format Y X N B

specification to_fp8_equiv (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    originalToFP8 input format Y X N B ≡[assumptions] indexCommutedToFP8 input format Y X N B := by
  exact FPProof.to_fp8_equivalent input format Y X N B

#print_fp_assumptions f8_to_f16_equiv
#print_fp_assumptions to_fp8_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.Equivalence
