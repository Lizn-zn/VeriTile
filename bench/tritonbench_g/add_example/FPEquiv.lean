import bench.tritonbench_g.add_example.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Contextual fp32 equivalence of the complete sources in Kernels.lean.
The profile interprets the source's real carrier through opaque FP operations.
Only addition changes; masks, loads, stores and failure behavior are preserved. -/

namespace VeriTile.Bench.TritonBenchG.AddExample.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.AddExample
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Both loaded operands must be finite at the actual addition site, including
masked lanes whose arithmetic is evaluated. No experimental dimensions or
input distribution are restrictions on this symbolic derivation. -/
def additionSite (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : Site :=
  .add (Source.prefixBefore (add_kernel in_ptr0 in_ptr1 out_ptr n B).surfaceBody "output") [B]
    (.ref .real [B] "x") (.ref .real [B] "y")

def original (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : Program where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n B
  domain := [additionSite in_ptr0 in_ptr1 out_ptr n B]

def commuted (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : Program :=
  { original in_ptr0 in_ptr1 out_ptr n B with kernel := add_commuted_kernel in_ptr0 in_ptr1 out_ptr n B }

/-- The admitted add_commute atom preserves complete state and execution failure. -/
specification add_commuted_equiv (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) (R : Rules) :
    original in_ptr0 in_ptr1 out_ptr n B ≡[R] commuted in_ptr0 in_ptr1 out_ptr n B := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AddExample.FPEquiv
