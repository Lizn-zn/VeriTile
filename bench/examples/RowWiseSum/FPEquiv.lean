import bench.examples.RowWiseSum.Kernels
import bench.examples.RowWiseSum.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ReductionSchedule
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Fin.Rev

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.RowWiseSumFPEquiv
open VeriTile.Bench.Examples.RowWiseSum.Kernels
open VeriTile Triton
open FP.Structural FP.Equational
open FP.Guarded FP.ReductionSchedule FP.ScalarReduction

set_option maxHeartbeats 1600000

def originalIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseSumKernel "x" "y" nCol B
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := id

def reversedIO (nCol B : Nat) : KernelIO₁ :=
  (originalIO nCol B).withKernel (reversedKernel "x" "y" nCol B)

/-- Explicit paths check the intermediate operands used in both tree
normalizations, including padding. They impose no experiment-size restriction. -/
def domain (nCol B : Nat) (plans : Schedules) : FP.GuardExpression.Condition :=
  let p := plan B plans
  let q := reindex p (Fin.revPerm : Equiv.Perm (Fin B))
  FP.GuardExpression.Requirements.all
    (((normalize p.tree).operands ++ (normalize q.tree).operands).map fun t =>
      .guard .finite (value (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)
        (inputExpression nCol B) (FP.ScalarArithmetic.zero
          (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)) t))

def original (nCol B : Nat) : FP.Kernel₁ where
  io := originalIO nCol B
  profile := FP.fp32
  domain := domain nCol B
  comparisons := []

def reversed (nCol B : Nat) : FP.Kernel₁ :=
  { original nCol B with io := reversedIO nCol B }

abbrev Rules := FP.ScalarArithmetic.Rules

open scoped VeriTile.Spec

/-- Reversal is derived from scalar commutation, association and the zero
identity used by tree normalization. Every intermediate guard is retained.
The statement includes successful execution and both memory frames. -/
specification rowwise_sum_equiv (nCol B : Nat) (R : Rules) :
    original nCol B ≡[R] reversed nCol B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := original nCol B) (rhs := reversed nCol B)
    (structural := fun _ _ => False) rfl rfl
  apply FP.Kernel₁.ofExecution
  exact Execution.equivalent nCol B R

#print_fp_assumptions rowwise_sum_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.RowWiseSumFPEquiv
