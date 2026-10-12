import bench.tritonbench_g.fast_rms_layernorm.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
section
variable (input weight : Storage) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) (isGemma : Bool)

def forwardProgram (swapped : Bool) : Program where
  kernel := forward input weight swapped Y X W r YR XR WR RR N B eps
  domain := []

theorem forward_equivalent :
    forwardProgram input weight Y X W r YR XR WR RR N B eps Bool.false ≡[assumptions]
    forwardProgram input weight Y X W r YR XR WR RR N B eps Bool.true := by
  cases input <;> cases weight <;> equiv_decompose <;> fp_prove

def gemmaProgram (swapped : Bool) : Program where
  kernel := gemma input weight swapped Y X W r YR XR WR RR N B eps
  domain := []

theorem gemma_equivalent :
    gemmaProgram input weight Y X W r YR XR WR RR N B eps Bool.false ≡[assumptions]
    gemmaProgram input weight Y X W r YR XR WR RR N B eps Bool.true := by
  cases input <;> cases weight <;> equiv_decompose <;> fp_prove

def backwardProgram (swapped : Bool) : Program where
  kernel := backward input weight swapped Y X W r dW YR XR WR RR DWR N B eps isGemma
  domain := []

theorem backward_equivalent :
    backwardProgram input weight Y X W r dW YR XR WR RR DWR N B eps isGemma Bool.false ≡[assumptions]
    backwardProgram input weight Y X W r dW YR XR WR RR DWR N B eps isGemma Bool.true := by
  cases input <;> cases weight <;> cases isGemma <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions forward_equivalent
#print_fp_assumptions gemma_equivalent
#print_fp_assumptions backward_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.FPProof
