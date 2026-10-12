import bench.examples.FusedSiLU.Kernels
import bench.examples.FusedSiLU.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.FusedSiLUFPEquiv
open VeriTile.Bench.Examples.FusedSiLU.Kernels
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

def unfusedIO (B : Nat) : KernelIO₃ where
  kernel := unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B
  scratch := [{ buf := "z", win := fun pid => pid * B, len := B },
    { buf := "silu", win := fun pid => pid * B, len := B }]

def fusedIO (B : Nat) : KernelIO₃ :=
  { (unfusedIO B).withKernel (fusedSiLUKernel "x" "gate" "residual" "out" B) with
    scratch := [] }

/-- Fusion preserves the same opaque numerical call tree, so no numerical
identity is assumed in this example. -/
def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

specification silu_equiv (B : Nat) : fusedIO B ≡[R] unfusedIO B := by
  apply Spec.FloatingPoint.ofStructural (lhs := fusedIO B) (rhs := unfusedIO B)
    (structural := IO₃Equiv) rfl rfl
  exact Execution.equivalent B

#print_fp_assumptions silu_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSiLUFPEquiv
