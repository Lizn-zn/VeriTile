import bench.examples.RowWiseMax.Kernels
import bench.examples.RowWiseMax.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.RowWiseMaxFPEquiv
open VeriTile.Bench.Examples.RowWiseMax.Kernels
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

def originalIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseMaxKernel "x" "y" nCol B
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := id

def inlinedIO (nCol B : Nat) : KernelIO₁ :=
  (originalIO nCol B).withKernel (inlinedKernel "x" "y" nCol B)

def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

/-- As in the original correctness theorem, max requires a nonempty row.
The row stride and positive block length remain symbolic. -/
specification rowwise_max_equiv (nCol B : Nat) (hB : 0 < B) :
    originalIO nCol B ≡[R] inlinedIO nCol B := by
  apply Spec.FloatingPoint.ofStructural (lhs := originalIO nCol B) (rhs := inlinedIO nCol B)
    (structural := IO₁Equiv) rfl rfl
  exact Execution.equivalent nCol B hB

#print_fp_assumptions rowwise_max_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.RowWiseMaxFPEquiv
