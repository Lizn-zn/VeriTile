import bench.tritonbench_g.mul_exponent_compensator.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete fp32 program equivalence under the admitted scalar multiplication
commutation rule. The finite operands are checked after the actual constant
assignment and input load. All registers, memory and failures are preserved. -/
namespace VeriTile.Bench.TritonBenchG.MulExponentCompensator.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.MulExponentCompensator
open scoped VeriTile.Spec

noncomputable def multiplicationSite (src dst : RegionName) (B : Nat) : Site :=
  .mulScalar (Source.prefixBefore (mul_kernel src dst B).surfaceBody "y") B []
    (.ref .real [B] "x") (.ref .real [] "exponent_compensator")

noncomputable def original (src dst : RegionName) (B : Nat) : Program where
  kernel := mul_kernel src dst B
  domain := [multiplicationSite src dst B]

noncomputable def commuted (src dst : RegionName) (B : Nat) : Program :=
  { original src dst B with kernel := mul_commuted_kernel src dst B }

specification mul_compensator_commuted_equiv (src dst : RegionName) (B : Nat)
    (_hB : 0 < B) (R : FP.ScalarArithmetic.Rules) :
    original src dst B ≡[R] commuted src dst B := by
  equiv_decompose
  -- The shared rule is oriented scalar-left to scalar-right.
  exact (contextual_mul_scalar_commute R _ _ ‹_› _ _ _ _ _ _ site_1).symm

#print_fp_assumptions mul_compensator_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MulExponentCompensator.FPEquiv
