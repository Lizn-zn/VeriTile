import bench.examples.StableLogSumExp.Contract

/-!
FP equivalence target for the direct and shifted sources in Kernels.lean.
This target is not yet proved under the admitted primitive set.

The missing candidates are FP.LogExp.Atom.log_mul, for tl.log(a * b),
and FP.LogExp.Atom.log_exp_libdevice, for tl.log(libdevice.exp(a)).
Both must be admitted with the domains needed by Contract.lean before the
conditional execution result there can supply an FP specification.
The admitted log_exp_elim and log_mul_split relations retain branches;
neither supplies either missing unconditional identity.

Kernels.candidateLSEKernel now implements the two admitted conditional rewrites
using libdevice.log. Float/LogSumExpCandidate.finish_eq proves their scalar
composition. Correct.lean proves its real correctness independently, alongside
the two older sources. This candidate does not resolve the intrinsic mismatch:
the direct reference still calls tl.log, and no admitted atom equates it with
libdevice.log. The goal below continues to refer to the original source pair.
-/

namespace VeriTile.Bench.Examples.StableLogSumExpFPEquiv
open VeriTile Triton
open StableLogSumExpFPContract
open scoped VeriTile.Spec

/-- Pending goal only; no equivalence certificate is asserted here. -/
def equivalenceGoal (B : Nat) (R : FP.Exponential.Rules) : Prop :=
  0 < B → stable "x" "y" B ≡[R] direct "x" "y" B

end VeriTile.Bench.Examples.StableLogSumExpFPEquiv
