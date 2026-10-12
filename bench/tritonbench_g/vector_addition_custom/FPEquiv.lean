import bench.tritonbench_g.vector_addition_custom.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete fp32 equivalence of the two sources in Kernels.lean. Addition
remains inside the masked store. The numerical rule changes only its value;
addresses, masks, dtype and failure behavior are preserved. -/

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.VectorAdditionCustom
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Both loaded operands must be finite before the first (and only) store,
including masked lanes whose arithmetic is evaluated. Locating the store by
kind keeps the site stable when unrelated register assignments are inserted. -/
def additionSite (A B C : RegionName) (size BLOCK : Nat) : Site :=
  .add (Source.prefixBeforeStore (_add_kernel A B C size BLOCK).surfaceBody) [BLOCK]
    (.ref .real [BLOCK] "a") (.ref .real [BLOCK] "b")

def original (A B C : RegionName) (size BLOCK : Nat) : Program where
  kernel := _add_kernel A B C size BLOCK
  domain := [additionSite A B C size BLOCK]

def commuted (A B C : RegionName) (size BLOCK : Nat) : Program :=
  { original A B C size BLOCK with kernel := _add_commuted_kernel A B C size BLOCK }

/-- Admitted add_commute preserves the complete masked-store program. -/
specification add_commuted_equiv (A B C : RegionName) (size BLOCK : Nat) (R : Rules) :
    original A B C size BLOCK ≡[R] commuted A B C size BLOCK := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom.FPEquiv
