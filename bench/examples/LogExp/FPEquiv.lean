import VeriTile.Triton.Source
import bench.examples.LogExp.Kernels
import bench.examples.LogExp.Proofs.FP
import VeriTile.Triton.Float.LogExp
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

noncomputable section

namespace VeriTile.Bench.Examples.LogExp.FPEquiv
open Triton FP FP.Structural FP.Guarded FP.LogExp
open scoped VeriTile.Spec

set_option maxHeartbeats 1600000

/-- The candidate prefix reads `a` and evaluates the branch selector. Successful
execution requires comparison support, for either branch outcome. The only
numerical input restriction is that each loaded `a` is finite. -/
def domain (xReg : RegionName) (B : Nat) : Precondition where
  code := Source.prefixBefore (optimizedKernel xReg "y" B).surfaceBody "fallback_a"
  guards := [⟨"a", [B], .finite⟩]

/-- The original kernel's input/output windows, domain and fp32 profile. -/
def originalIO (xReg yReg : RegionName) (B : Nat) : FP.PrefixKernel₁ where
  io := {
    kernel := originalKernel xReg yReg B
    inp := xReg, out := yReg, Bin := B, Bout := B
    read := fun pid => pid * B
    write := fun pid => pid * B }
  domain := domain xReg B
  defaultPrecision := some .fp32

/-- Same interface and domain; the implementation is `optimizedKernel`. -/
def optimizedIO (xReg yReg : RegionName) (B : Nat) : FP.PrefixKernel₁ :=
  { originalIO xReg yReg B with
    io := (originalIO xReg yReg B).io.withKernel (optimizedKernel xReg yReg B) }

/-- Under the admitted guarded relation, both kernels succeed, write equivalent
fp32 outputs and preserve all other memory. Dimensions remain symbolic. -/
specification log_exp_equiv (R : Rules)
    (xReg yReg : RegionName) (blockSize : Nat) :
    originalIO xReg yReg blockSize ≡[R] optimizedIO xReg yReg blockSize := by
  apply Spec.FloatingPoint.ofNumerical (lhs := originalIO xReg yReg blockSize)
    (rhs := optimizedIO xReg yReg blockSize) (structural := fun _ _ => False) rfl rfl
  exact Execution.equivalent R xReg yReg blockSize

#print_fp_assumptions log_exp_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.LogExp.FPEquiv
