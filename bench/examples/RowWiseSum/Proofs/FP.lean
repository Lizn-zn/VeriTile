import VeriTile.Triton.Float.Kernel
import bench.examples.RowWiseSum.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ReductionSchedule
import VeriTile.Triton.Float.ScheduledIO
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Fin.Rev

/-! FP execution and scalar composition lemmas for RowWiseSum.
Public IO, numerical domains and the equivalence specification are in ../FPEquiv.lean. -/

namespace VeriTile.Bench.Examples.RowWiseSumFPEquiv
open VeriTile.Bench.Examples.RowWiseSum.Kernels
open VeriTile Triton
open FP.Structural FP.Equational
open FP.Guarded FP.ReductionSchedule FP.ScalarReduction

set_option maxHeartbeats 1600000

def reducedValue {α : Type} (M : Algebra α) (B : Nat) (xs : Fin B → α) : α :=
  M.reduceSum (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false
    (fun i => M.fp32Load (xs i.1)) PUnit.unit

theorem original_run {α : Type} [Inhabited α] (M : Algebra α)
    (nCol B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * nCol + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (rowWiseSumKernel "x" "y" nCol B) s = some t ∧
      t.mem "y" (s.pids 0) = Cell.mk .real (reducedValue M B xs) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [rowWiseSumKernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, FP.Structural.bop, store, TileShape.allIndices,
    TileShape.eraseAxis, Region.cast, ComputeDType.eraseDType, hx, reducedValue]
  intro r o hmiss
  exact (State.write_other _ "y" r (s.pids 0) o _ hmiss).trans rfl

theorem reversed_run {α : Type} [Inhabited α] (M : Algebra α)
    (nCol B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * nCol + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (reversedKernel "x" "y" nCol B) s = some t ∧
      t.mem "y" (s.pids 0) = Cell.mk .real (reducedValue M B (fun i => xs i.rev)) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  have hrev (i : Fin B) : (s.mem "x" (s.pids 0 * nCol + (B - 1 - i.val))).read .real = xs i.rev := by
    simpa only [Fin.val_rev, Nat.sub_sub, Nat.add_comm] using hx i.rev
  simp [reversedKernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, FP.Structural.bop, store, TileShape.allIndices,
    TileShape.eraseAxis, Region.cast, ComputeDType.eraseDType, hrev, reducedValue]
  intro r o hmiss
  exact (State.write_other _ "y" r (s.pids 0) o _ hmiss).trans rfl

/-- Read exactly the fp32-loaded values consumed by the reduction. -/
def inputExpression (nCol B : Nat) (i : Fin B) : FP.GuardExpression.Expr FP.GuardExpression.MemoryInput :=
  .fp32Load (.input ⟨"x", fun pid => pid * nCol + i.val, .real⟩)

def plan (B : Nat) (plans : Schedules) : ReductionPlan B :=
  plans (some .fp32) [B] ⟨0, by simp⟩ Bool.false PUnit.unit

theorem value_expression {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (nCol B : Nat) (t : ReductionTree B) :
    (value (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)
      (inputExpression nCol B) (FP.ScalarArithmetic.zero
        (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)) t).eval M
      (FP.GuardExpression.MemoryInput.read s) =
    value M (fun i => M.fp32Load ((s.mem "x" (s.pids 0 * nCol + i.val)).read .real))
      (FP.ScalarArithmetic.zero M) t := by
  induction t with
  | input => rfl
  | zero => rfl
  | add a b ha hb =>
      change FP.ScalarArithmetic.add M _ _ = FP.ScalarArithmetic.add M _ _
      rw [ha, hb]

theorem domain_values {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) (nCol B : Nat) (plans : Schedules)
    (hd : FP.GuardExpression.Condition.Holds M D s <|
      let p := plan B plans
      let q := reindex p (Fin.revPerm : Equiv.Perm (Fin B))
      .all (((normalize p.tree).operands ++ (normalize q.tree).operands).map fun t =>
        .guard .finite (value (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)
          (inputExpression nCol B) (FP.ScalarArithmetic.zero
            (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)) t))) :
    ScheduleDomain M D
      (fun i => M.fp32Load ((s.mem "x" (s.pids 0 * nCol + i.val)).read .real))
      (plan B plans) (reindex (plan B plans) Fin.revPerm) := by
  simp only [FP.GuardExpression.Condition.Holds, FP.GuardExpression.Requirements.holds_all, List.mem_map] at hd
  constructor
  · intro t ht
    have h := hd _ ⟨t, List.mem_append_left _ ht, rfl⟩
    exact (value_expression M s nCol B t) ▸ h
  · intro t ht
    have h := hd _ ⟨t, List.mem_append_right _ ht, rfl⟩
    exact (value_expression M s nCol B t) ▸ h

open scoped VeriTile.Spec

/- Internal execution contracts mirror the public fields in FPEquiv.lean.
The entry's final theorem checks their definitional equality. -/
namespace Execution

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

/-- Compose execution, numerical laws and the memory frames. -/
theorem equivalent (nCol B : Nat) (R : Rules) :
    FP.Scheduled.Equivalent₁ R.assumptions (original nCol B).toIO₁ (reversed nCol B).toIO₁ := by
  refine ⟨by simp [IO₁PrivateScratch, original, originalIO],
    by simp [IO₁PrivateScratch, reversed, reversedIO, originalIO], ?_⟩
  intro α _ M D hM plans s hd
  let xs : Fin B → α := fun i => (s.mem "x" (s.pids 0 * nCol + i.val)).read .real
  let A := FP.fp32.algebra M plans
  obtain ⟨a, ha, hva, hfa⟩ := original_run A nCol B s xs (fun _ => rfl)
  obtain ⟨b, hb, hvb, hfb⟩ := reversed_run A nCol B s xs (fun _ => rfl)
  have hvalue := plans_value R M D hM s (fun i => M.fp32Load (xs i))
    (plan B plans) (reindex (plan B plans) Fin.revPerm)
    (domain_values M D s nCol B plans hd)
  rw [reindex, value_reindex] at hvalue
  refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
  · intro i
    change a.mem "y" (s.pids 0 + i.val) = b.mem "y" (s.pids 0 + i.val)
    rw [Fin.val_eq_zero i, Nat.add_zero, hva, hvb]
    exact congrArg (Cell.mk .real) hvalue
  · intro r o ho _
    exact hfa r o (by simpa [original, originalIO, Fin.forall_fin_one] using ho)
  · intro r o ho _
    exact hfb r o (by simpa [reversed, reversedIO, originalIO, Fin.forall_fin_one] using ho)

end Execution

end VeriTile.Bench.Examples.RowWiseSumFPEquiv
