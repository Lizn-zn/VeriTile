import bench.examples.StableLogSumExp.Contract

/-!
FP equivalence target for the direct and shifted sources in Kernels.lean.
This target is not yet proved under the admitted primitive set.

The missing candidates are FP.LogExp.Atom.log_mul, for tl.log(a * b),
and FP.LogExp.Atom.log_exp_libdevice, for tl.log(libdevice.exp(a)).
Both must be admitted with the domains needed by Contract.lean before the
conditional execution result there can supply an FP specification.
The admitted masked log1p/expm1 relation describes a different expression.

Correct.lean already proves real correctness of both sources independently.
-/

namespace VeriTile.Bench.Examples.StableLogSumExpFPEquiv
open VeriTile Triton
open StableLogSumExpFPContract
open scoped VeriTile.Spec

/-- Pending goal only; no equivalence certificate is asserted here. -/
def equivalenceGoal (B : Nat) (R : FP.Exponential.Rules) : Prop :=
  0 < B → stable "x" "y" B ≡[R] direct "x" "y" B

end VeriTile.Bench.Examples.StableLogSumExpFPEquiv
