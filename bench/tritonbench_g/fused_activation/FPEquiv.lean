import bench.tritonbench_g.fused_activation.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete fp32 execution equivalence, including all loads, both activation
branches, the masked in-place store, registers and execution failure. -/
namespace VeriTile.Bench.TritonBenchG.FusedActivation.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
abbrev Rules := FP.ScalarArithmetic.Rules

/-- The actual combined input (multiplier * input + x) and bias must be finite.
This includes inactive lanes since arithmetic executes before the masked store.
No input distribution, positivity or nonzero restriction is required. -/
def additionSite (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) : Site :=
  .add (Source.prefixBefore
    (fused_add_mul_activation_kernel x b i W N B m a).surfaceBody "activ_input")
    [B] (.add .real (.consSame .nil)
      (.mul .real .scalarL (.const m) (.ref .real [B] "tmp3"))
      (.ref .real [B] "tmp0")) (.ref .real [B] "tmp1")

def original (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) : Program where
  kernel := fused_add_mul_activation_kernel x b i W N B m a
  domain := [additionSite x b i W N B m a]

def commuted (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) : Program :=
  { original x b i W N B m a with
    kernel := fused_add_mul_activation_commuted_kernel x b i W N B m a }

specification fused_activation_equiv (x b i : RegionName) (W N B : Nat)
    (m : ℝ) (a : Bool) (_hW : 0 < W) (R : Rules) :
    original x b i W N B m a ≡[R] commuted x b i W N B m a := by
  cases a <;> equiv_decompose
  all_goals fp_prove

#print_fp_assumptions fused_activation_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FusedActivation.FPEquiv
