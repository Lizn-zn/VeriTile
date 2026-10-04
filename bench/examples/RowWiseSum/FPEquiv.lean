import bench.examples.RowWiseSum.Kernels
/- Row-wise sum with forward versus reversed input lanes. The conditional proof
uses the admitted fp32 add_commute and add_assoc instances. Both definitions are
independent of the Correct file; stride and row length remain symbolic. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Fin.Rev

namespace VeriTile.Bench.Examples.RowWiseSumFPEquiv
open VeriTile.Bench.Examples.RowWiseSum.Kernels
open VeriTile Triton
open FP.Structural FP.Equational

set_option maxHeartbeats 1600000


def reducedValue {α : Type} (M : Algebra α) (B : Nat) (xs : Fin B → α) : α :=
  M.reduceSum (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false
    (fun i => M.fp32Load (xs i.1)) PUnit.unit

theorem original_run {α : Type} [Inhabited α] (M : Algebra α)
    (nCol B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * nCol + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (fpRowWiseSumKernel "x" "y" nCol B) s = some t ∧
      t.mem "y" (s.pids 0) = Cell.mk .real (reducedValue M B xs) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [fpRowWiseSumKernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
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

def originalIO (nCol B : Nat) : KernelIO₁ where
  kernel := fpRowWiseSumKernel "x" "y" nCol B
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := id

def reversedIO (nCol B : Nat) : KernelIO₁ :=
  { originalIO nCol B with kernel := reversedKernel "x" "y" nCol B, projection := by rfl }

def addCommute := (FP.ScalarArithmetic.report .addCommute (by decide)).report.bind commuteLHS commuteRHS
def addAssociate := (FP.ScalarArithmetic.report .addAssociate (by decide)).report.bind associateLHS associateRHS

structure Rules where
  add_comm : Spec.EvidenceValidated addCommute.rule addCommute.evidence
  add_assoc : Spec.EvidenceValidated addAssociate.rule addAssociate.evidence

def Rules.assumptions (_R : Rules) : Spec.Assumptions ComputeStmt := [addCommute, addAssociate]

instance : CoeOut Rules (Spec.Assumptions (Spec.ProgramSyntax.Statement KernelIO₁)) :=
  ⟨Rules.assumptions⟩

theorem admitted_commute (R : Rules) : Spec.Derivation R.assumptions commuteLHS commuteRHS :=
  .atom addCommute (by simp [Rules.assumptions])
    ((FP.ScalarArithmetic.report .addCommute (by decide)).report.admit _ _ R.add_comm)

theorem admitted_associate (R : Rules) : Spec.Derivation R.assumptions associateLHS associateRHS :=
  .atom addAssociate (by simp [Rules.assumptions])
    ((FP.ScalarArithmetic.report .addAssociate (by decide)).report.admit _ _ R.add_assoc)

theorem reduced_equiv (R : Rules) (plans : Schedules) (B : Nat) (xs : Fin B → Term α) :
    TermEq R.assumptions (reducedValue (algebra plans) B xs)
      (reducedValue (algebra plans) B (fun i => xs i.rev)) := by
  simp only [reducedValue, algebra, reductionInputs, TileShape.insertAxisIndex,
    SumTree.evalAt_fp32]
  exact ReductionPlan.reorder (admitted_commute R) (admitted_associate R) _
    (fun i => .app .fp32Load [xs i]) _ Fin.revPerm

open scoped VeriTile.Spec

/-- Reversal changes the addition order. The derivation uses the admitted
commutation and association assumptions. Dimensions and schedules are arbitrary;
the numerical table does not supply a whole-reduction IEEE guarantee. -/
specification rowwise_sum_equiv (nCol B : Nat) (R : Rules) :
    originalIO nCol B ≡[R] reversedIO nCol B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := originalIO nCol B) (rhs := reversedIO nCol B)
    (structural := IO₁Equiv) rfl rfl
  refine ⟨?_, ?_, ?_⟩
  · simp [IO₁PrivateScratch, originalIO]
  · simp [IO₁PrivateScratch, reversedIO, originalIO]
  · intro α _ plans s
    let xs : Fin B → Term α := fun i => (s.mem "x" (s.pids 0 * nCol + i.val)).read .real
    obtain ⟨a, ha, hva, hfa⟩ := original_run (algebra plans) nCol B s xs (fun _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := reversed_run (algebra plans) nCol B s xs (fun _ => rfl)
    refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := by have := i.isLt; change i.val < 1 at this; omega
      simp only [originalIO, reversedIO, hi, Nat.add_zero, id_eq]
      rw [hva, hvb]
      exact ⟨.real, _, _, rfl, rfl, reduced_equiv R plans B xs⟩
    · intro r o ho _
      apply hfa r o
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (by simpa [originalIO] using ho ⟨0, by change 0 < 1; decide⟩)
    · intro r o ho _
      apply hfb r o
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (by simpa [reversedIO, originalIO] using ho ⟨0, by change 0 < 1; decide⟩)

#print_fp_assumptions rowwise_sum_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.RowWiseSumFPEquiv
