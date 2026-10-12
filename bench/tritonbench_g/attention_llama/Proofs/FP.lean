import bench.tritonbench_g.attention_llama.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Conditional equivalence of both complete fp16 K/V programs. The candidate
commutes only the l_curr addition inside each reachable iteration. Finiteness
checks concern the actual reduction result and rescaled l_prev at that site.
The original exp intrinsic, causal branch, key-mask quirk, dot operations,
probability cast, normalization order, zero padding and output store remain. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool)

/-- Check each row's computed sum(p) and updated carry, after the complete
score/exp prefix in every reachable iteration. No nonzero, positivity, output
formula or reduction identity is assumed by this commutation. -/
def requirements : List Site :=
  match Source.dynamicRange? (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).surfaceBody "start_n" with
  | none => []
  | some loop => [.range loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "l_curr") [BM]
        (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p"))
        (.ref .real [BM] "l_prev"))]

def original : Program where
  kernel := TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal
  domain := requirements Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal

def carryFirst : Program :=
  { original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal with kernel := TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal }

theorem attention_carry_first_equiv (R : Rules) :
    original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal ≡[R] carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal := by
  equiv_decompose
  all_goals fp_prove

end
#print_fp_assumptions attention_carry_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FPEquivProofs
