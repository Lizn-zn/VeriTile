import bench.tritonbench_g.vector_addition.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Contextual fp32 equivalence of the complete sources in Kernels.lean.
The profile interprets the source's real carrier through opaque FP operations.
Only addition changes; masks, loads, stores and failure behavior are preserved. -/

namespace VeriTile.Bench.TritonBenchG.VectorAddition.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.VectorAddition
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Both loaded operands must be finite at the actual addition site, including
masked lanes whose arithmetic is evaluated. No experimental dimensions or
input distribution are restrictions on this symbolic derivation. -/
def additionSite (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) : Site :=
  .add (Source.prefixBefore (add_kernel x_ptr y_ptr output_ptr n B).surfaceBody "output") [B]
    (.ref .real [B] "x") (.ref .real [B] "y")

def original (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) : Program where
  kernel := add_kernel x_ptr y_ptr output_ptr n B
  domain := [additionSite x_ptr y_ptr output_ptr n B]

def commuted (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) : Program :=
  { original x_ptr y_ptr output_ptr n B with kernel := add_commuted_kernel x_ptr y_ptr output_ptr n B }

/-- The admitted add_commute atom preserves complete state and execution failure. -/
specification add_commuted_equiv (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) (R : Rules) :
    original x_ptr y_ptr output_ptr n B ≡[R] commuted x_ptr y_ptr output_ptr n B := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.VectorAddition.FPEquiv
