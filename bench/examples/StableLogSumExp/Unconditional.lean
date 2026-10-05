import bench.examples.StableLogSumExp.Contract

/-! Historical source without fallbacks. Its real formula is correct, but its
FP transformation needs unconditional log-product and log-exp cancellation.
Those relations are not admitted by the selected reports. This is a goal,
not a theorem or a counterexample; missing admission alone is no refutation.
The supported optimized source is optimizedLSEKernel in Kernels.lean, proved
in Correct.lean and FPEquiv.lean with its fallback branches intact. -/
namespace VeriTile.Bench.Examples.StableLogSumExpUnconditional
open VeriTile Triton StableLogSumExpFPContract
open scoped VeriTile.Spec

def equivalenceGoal (B : Nat) (R : FP.Exponential.Rules) : Prop :=
  0 < B → stable "x" "y" B ≡[R] direct "x" "y" B

end VeriTile.Bench.Examples.StableLogSumExpUnconditional
