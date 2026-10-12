import bench.tritonbench_g.attention_llama.Kernels
import bench.tritonbench_g.attention_llama.Proofs.FP
import bench.tritonbench_g.attention_llama.Proofs.FP8
import VeriTile.Meta.StatementAudit

/-! Original and carry-first programs differ only at l_curr:
sum(p,1)+l_prev becomes l_prev+sum(p,1). FP16 uses fp16 Q/K/V/Out;
FP8 keeps int8 K/V loads, E5M2 bitcasts and fp16 conversions. Both retain fp32
accumulation, tl.exp, dot operations, the probability cast, causal branches,
source score-mask indexing and the final masked fp16 store.

The domain checks finite actual operands at this addition in every reached
iteration, after executing its score/exp prefix. Shapes, strides, start_position
and the causal flag remain symbolic. No exp or reduction identity is assumed.
These equivalences preserve complete state and failures under the admitted
add_commute assumption; they do not import the real correctness proof. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FPEquivalence
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
  match Source.dynamicRange? (TypedSource.original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).surfaceBody "start_n" with
  | none => []
  | some loop => [.range loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "l_curr") [BM]
        (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p"))
        (.ref .real [BM] "l_prev"))]

def original : Program where
  kernel := TypedSource.original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
  domain := requirements Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal

def carryFirst : Program :=
  { original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal with
    kernel := TypedSource.carryFirst Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal }

specification attention_carry_first_equiv (R : Rules) :
    original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal ≡[R]
    carryFirst Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal := by
  exact FPEquivProofs.attention_carry_first_equiv Q K V Out sm_scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    NH H N SP BM BN BD causal R

#print_fp_assumptions attention_carry_first_equiv
end

section

variable (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool)

/-- Check each row's computed sum(p) and updated carry, after the complete
score/exp prefix in every reachable iteration. No nonzero, positivity, output
formula or reduction identity is assumed by this commutation. -/
def requirementsFP8 : List Site :=
  match Source.dynamicRange? (TypedSource.originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).surfaceBody "start_n" with
  | none => []
  | some loop => [.range loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "l_curr") [BM]
        (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p"))
        (.ref .real [BM] "l_prev"))]

def originalFP8 : Program where
  kernel := TypedSource.originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
  domain := requirementsFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal

def carryFirstFP8 : Program :=
  { originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal with
    kernel := TypedSource.carryFirstFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal }

specification attention_fp8_carry_first_equiv (R : Rules) :
    originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal ≡[R]
    carryFirstFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal := by
  exact FP8EquivProofs.attention_carry_first_equiv Q K V Out sm_scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    NH H N SP BM BN BD causal R

#print_fp_assumptions attention_fp8_carry_first_equiv
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FPEquivalence
