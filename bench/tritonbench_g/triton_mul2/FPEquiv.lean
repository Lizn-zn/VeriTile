import bench.tritonbench_g.triton_mul2.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! fp32 program equivalence of each original/candidate pair in Kernels.lean.
Only the multiplication operands change. Ordinary output and in-place output
keep their respective region signatures, loads, masks and stores. -/

namespace VeriTile.Bench.TritonBenchG.TritonMul2.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.TritonMul2
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Check all actually loaded tile lanes and the fp32 literal 2 at the
multiplication site. Masked lanes still execute arithmetic and are included.
No positivity or nonzero condition is required of the input. -/
def multiplicationSite (in_ptr0 out_ptr : RegionName) (n B : Nat) : Site :=
  .mulScalar (Source.prefixBefore (mul2_kernel in_ptr0 out_ptr n B).surfaceBody
    "output") B [] (.ref .real [B] "x") (.const 2)

def original (in_ptr0 out_ptr : RegionName) (n B : Nat) : Program where
  kernel := mul2_kernel in_ptr0 out_ptr n B
  domain := [multiplicationSite in_ptr0 out_ptr n B]

def commuted (in_ptr0 out_ptr : RegionName) (n B : Nat) : Program :=
  { original in_ptr0 out_ptr n B with kernel := mul2_commuted_kernel in_ptr0 out_ptr n B }

specification mul2_commuted_equiv (in_ptr0 out_ptr : RegionName) (n B : Nat) (R : Rules) :
    original in_ptr0 out_ptr n B ≡[R] commuted in_ptr0 out_ptr n B := by
  equiv_decompose
  all_goals fp_prove

/-- The in-place source checks its own load prefix and writes back to ptr. -/
def inplaceMultiplicationSite (ptr : RegionName) (n B : Nat) : Site :=
  .mulScalar (Source.prefixBefore (mul2_inplace_kernel ptr n B).surfaceBody
    "output") B [] (.ref .real [B] "x") (.const 2)

def inplaceOriginal (ptr : RegionName) (n B : Nat) : Program where
  kernel := mul2_inplace_kernel ptr n B
  domain := [inplaceMultiplicationSite ptr n B]

def inplaceCommuted (ptr : RegionName) (n B : Nat) : Program :=
  { inplaceOriginal ptr n B with kernel := mul2_inplace_commuted_kernel ptr n B }

/-- Includes input/output aliasing: the loaded register is rewritten, then
stored back through the original mask. All states and execution failures agree. -/
specification mul2_inplace_commuted_equiv (ptr : RegionName) (n B : Nat) (R : Rules) :
    inplaceOriginal ptr n B ≡[R] inplaceCommuted ptr n B := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions mul2_commuted_equiv
#print_fp_assumptions mul2_inplace_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.TritonMul2.FPEquiv
