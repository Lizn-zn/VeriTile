import bench.examples.FloatDTypeAdd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! FloatDTypeAdd: contextual FP equivalence from the admitted scalar add_commute.
The fp32 values loaded into x and y must be finite at the addition.
The explicit output cast remains in the common suffix.
The domain is evaluated at the actual rewrite site; experimental dimensions
are not restrictions on these symbolic kernels. -/

namespace VeriTile.Bench.Examples.FloatDTypeAddFPEquiv
open VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
open VeriTile Triton FP.Structural FP.GuardedRewrite
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Finite operands after the original common prefix. -/
def additionSite (blockSize : Nat) : Site where
  before := Source.prefixBefore (originalKernel blockSize).surfaceBody "out"
  shape := [blockSize]
  left := .ref .real [blockSize] "x"
  right := .ref .real [blockSize] "y"

def original (blockSize : Nat) : Program where
  kernel := originalKernel blockSize
  domain := [additionSite blockSize]

def optimized (blockSize : Nat) : Program :=
  { original blockSize with kernel := optimizedKernel blockSize }

/-- Under the finite-operand contract, every successful run is preserved,
including all registers and memory; failed surrounding executions also agree. -/
specification float_add_equiv (blockSize : Nat) (R : Rules) :
    original blockSize ≡[R] optimized blockSize := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions float_add_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeAddFPEquiv
