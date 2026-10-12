import bench.tritonbench_g.fast_rms_layernorm.Proofs.FP
/-! Complete typed-state/failure equivalence for integer row-index commutation.
All nine storage pairs and both backward constexpr branches are preserved.
In particular, regular forward quantizes normed to weight precision before
multiplying; its statically resolved dtype is explicit. No numerical atom is
assumed, and the program input domains are empty. -/
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (input weight : Storage) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ) (swapped : Bool) : Program where
  kernel := forward input weight swapped Y X W r YR XR WR RR N B eps
  domain := []

specification forward_equiv (input weight : Storage) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ) :
    forwardProgram input weight Y X W r YR XR WR RR N B eps Bool.false ≡[assumptions]
    forwardProgram input weight Y X W r YR XR WR RR N B eps Bool.true := by
  exact FPProof.forward_equivalent input weight Y X W r YR XR WR RR N B eps

#print_fp_assumptions forward_equiv
def gemmaProgram (input weight : Storage) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ) (swapped : Bool) : Program where
  kernel := gemma input weight swapped Y X W r YR XR WR RR N B eps
  domain := []

specification gemma_equiv (input weight : Storage) (Y X W r : RegionName) (YR XR WR RR N B : Nat) (eps : ℝ) :
    gemmaProgram input weight Y X W r YR XR WR RR N B eps Bool.false ≡[assumptions]
    gemmaProgram input weight Y X W r YR XR WR RR N B eps Bool.true := by
  exact FPProof.gemma_equivalent input weight Y X W r YR XR WR RR N B eps

#print_fp_assumptions gemma_equiv
def backwardProgram (input weight : Storage) (Y X W r dW : RegionName) (YR XR WR RR DWR N B : Nat) (eps : ℝ) (isGemma : Bool) (swapped : Bool) : Program where
  kernel := backward input weight swapped Y X W r dW YR XR WR RR DWR N B eps isGemma
  domain := []

specification backward_equiv (input weight : Storage) (Y X W r dW : RegionName) (YR XR WR RR DWR N B : Nat) (eps : ℝ) (isGemma : Bool) :
    backwardProgram input weight Y X W r dW YR XR WR RR DWR N B eps isGemma Bool.false ≡[assumptions]
    backwardProgram input weight Y X W r dW YR XR WR RR DWR N B eps isGemma Bool.true := by
  exact FPProof.backward_equivalent input weight Y X W r dW YR XR WR RR DWR N B eps isGemma

#print_fp_assumptions backward_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.Equivalence
