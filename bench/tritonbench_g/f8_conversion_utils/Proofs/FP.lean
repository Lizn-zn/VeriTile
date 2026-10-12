import bench.tritonbench_g.f8_conversion_utils.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def toHalfProgram (format : Format) (swapped : Bool) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toHalf format swapped Y X N B
  domain := []

def toFP8Program (input : Input) (format : Format) (swapped : Bool) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toFP8 input format swapped Y X N B
  domain := []

theorem to_half_equivalent (format : Format) (Y X : RegionName) (N B : Nat) :
    toHalfProgram format Bool.false Y X N B ≡[assumptions] toHalfProgram format Bool.true Y X N B := by
  cases format <;> equiv_decompose <;> fp_prove

theorem to_fp8_equivalent (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    toFP8Program input format Bool.false Y X N B ≡[assumptions] toFP8Program input format Bool.true Y X N B := by
  cases input <;> cases format <;> equiv_decompose <;> fp_prove

#print_fp_assumptions to_half_equivalent
#print_fp_assumptions to_fp8_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof
