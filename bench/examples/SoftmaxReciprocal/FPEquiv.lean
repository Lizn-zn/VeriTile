import bench.examples.SoftmaxReciprocal.Kernels
import bench.examples.SoftmaxReciprocal.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Reciprocal
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv
open VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile Triton
open FP.Structural FP.Guarded

set_option maxHeartbeats 1600000

/-- Domain requirements for the two operands at each scalar rewrite site.
No distribution or experimental shape is imposed on the symbolic kernel. -/
def domain (B : Nat) : Precondition where
  code :=  (commonPrefix "x" B).surfaceBody
  guards := [⟨"e", [B], .finite⟩, ⟨"s", [], .finite⟩, ⟨"s", [], .nonzero⟩]

def originalIO (B : Nat) : FP.PrefixKernel₁ where
  io := {
    kernel := originalKernel "x" "y" B
    inp := "x", out := "y", Bin := B, Bout := B
    read := fun pid => pid * B
    write := fun pid => pid * B }
  domain := domain B
  defaultPrecision := none -- The source explicitly tags its floating operations.

def reciprocalIO (B : Nat) : FP.PrefixKernel₁ :=
  { originalIO B with io := (originalIO B).io.withKernel (reciprocalKernel "x" "y" B) }

abbrev Rules := FP.Reciprocal.Rules .fp32

open scoped VeriTile.Spec

specification softmax_reciprocal_equiv (R : Rules) (B : Nat) (hB : 0 < B) :
    originalIO B ≡[R] reciprocalIO B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := originalIO B) (rhs := reciprocalIO B)
    (structural := fun _ _ => False) rfl rfl
  exact Execution.equivalent R B hB

#print_fp_assumptions softmax_reciprocal_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv
