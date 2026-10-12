import bench.tritonbench_g.fast_layernorm.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 9000000
set_option maxRecDepth 25000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
section
variable (input weight bias : Storage) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ)

def forwardProgram (swapped : Bool) : Program where
  kernel := forward input weight bias swapped Y X W b r mu YR XR N B eps
  domain := []

theorem forward_equivalent :
    forwardProgram input weight bias Y X W b r mu YR XR N B eps Bool.false ≡[assumptions]
    forwardProgram input weight bias Y X W b r mu YR XR N B eps Bool.true := by
  cases input <;> cases weight <;> cases bias <;> equiv_decompose <;> fp_prove

def backwardProgram (swapped : Bool) : Program where
  kernel := backward input weight bias swapped Y X W b r mu YR XR N B eps
  domain := []

theorem backward_equivalent :
    backwardProgram input weight bias Y X W b r mu YR XR N B eps Bool.false ≡[assumptions]
    backwardProgram input weight bias Y X W b r mu YR XR N B eps Bool.true := by
  cases input <;> cases weight <;> cases bias <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions forward_equivalent
#print_fp_assumptions backward_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastLayernorm.FPProof
