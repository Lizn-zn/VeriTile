import bench.tritonbench_g.attention_score.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Full-source fp32 score-scaling commutation, retaining fp16 storage and all
window/load/store branches. Only finite actual qk lanes and qk_scale are
required at the multiplication reached in each iteration; no condition is
imposed on the sign of inputs, distances or window parameters. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)

/-- Locate the third qk assignment, after zero initialization and the score
dot. Each reached iteration checks the actual fp32 score tile and scale. -/
def requirements : List Site :=
  match Source.dynamicRange? (TypedSource.original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock).surfaceBody "start_m" with
  | none => []
  | some loop => [.range loop.before "start_m" loop.start loop.stop loop.stride loop.body
      (.mulScalarFP32 (Source.prefixBefore loop.body "qk" 2) BM [BN]
        (.ref .real [BM, BN] "qk") (.ref .real [] "qk_scale"))]

def original : Program where
  kernel := TypedSource.original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock
  domain := requirements Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock

def scalarLeft : Program :=
  { original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock with
    kernel := TypedSource.scalarLeft Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock }

theorem score_scaling_equiv (R : Rules) :
    original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock ≡[R]
    scalarLeft Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock := by
  subst BM
  equiv_decompose
  all_goals fp_prove

end
#print_fp_assumptions score_scaling_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionScore.FPEquivProofs
