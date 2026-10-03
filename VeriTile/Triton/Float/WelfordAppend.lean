/- The append step connecting explicit row statistics to the Welford update.
All numerical equalities are derived from the admitted scalar theory. The
floating count remains a tree of ones, not an unmeasured integer conversion. -/
import VeriTile.Triton.Float.WelfordReduction
import Mathlib.Data.Fin.Tuple.Basic

namespace VeriTile.Triton.FP.WelfordAppend
open Structural Guarded ScalarArithmetic ScalarReduction WelfordReduction
open Equational (ReductionTree ReductionPlan)

/-- Preserve an old tree verbatim while embedding its lanes in a longer row. -/
def liftTree : ReductionTree N → ReductionTree (N + 1)
  | .input i => .input i.castSucc
  | .zero => .zero
  | .add a b => .add (liftTree a) (liftTree b)

def appendTree (tree : ReductionTree N) : ReductionTree (N + 1) :=
  .add (liftTree tree) (.input (Fin.last N))

theorem liftTree_leaves (tree : ReductionTree N) :
    (liftTree tree).leaves = tree.leaves.map (Option.map Fin.castSucc) := by
  induction tree with
  | input => rfl
  | zero => rfl
  | add a b ih₁ ih₂ => simp only [liftTree, ReductionTree.leaves, ih₁, ih₂, List.map_append]

/-- Appending one lane preserves validity and every original padding leaf. -/
def appendPlan (plan : ReductionPlan N) : ReductionPlan (N + 1) where
  tree := appendTree plan.tree
  padding := plan.padding
  valid := by
    have h := (plan.valid.map (Option.map Fin.castSucc)).append_right [some (Fin.last N)]
    simp only [List.map_append, List.map_map, Function.comp_def, Option.map_some,
      List.map_replicate, Option.map_none] at h
    change ((liftTree plan.tree).leaves ++ [some (Fin.last N)]).Perm _
    rw [liftTree_leaves]
    refine h.trans ?_
    rw [List.finRange_succ_last, List.map_append, List.map_map]
    simpa only [List.map_cons, List.map_nil, List.append_assoc, Function.comp_def] using
      (List.perm_append_comm (l₁ := List.replicate plan.padding none) (l₂ := [some (Fin.last N)])).append_left
        ((List.finRange N).map (fun i => some i.castSucc))

theorem value_lift {α : Type} (M : Algebra α) (xs : Fin (N + 1) → α)
    (seed : α) (tree : ReductionTree N) :
    value M xs seed (liftTree tree) = value M (fun i => xs i.castSucc) seed tree := by
  induction tree with
  | input => rfl
  | zero => rfl
  | add a b ih₁ ih₂ => simp only [liftTree, value, ih₁, ih₂]

theorem value_append {α : Type} (M : Algebra α) (xs : Fin (N + 1) → α)
    (seed : α) (tree : ReductionTree N) :
    value M xs seed (appendTree tree) =
      add M (value M (fun i => xs i.castSucc) seed tree) (xs (Fin.last N)) := by
  simp only [appendTree, value, value_lift]

theorem count_append {α : Type} (M : Algebra α) (tree : ReductionTree N) :
    count M (appendTree tree) = Welford.nextCount M (count M tree) := by
  simp only [count, value_append, Welford.nextCount]

/-- Arithmetic domains for comparing the next mean with the appended row.
The equality of those means is not a premise. -/
structure MeanAppendDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (tree : ReductionTree N) (x : α) : Prop where
  old : MeanDomain M D xs tree
  step : Welford.MeanStepDomain M D x (mean M xs tree) (count M tree)
  newSum : D .finite (add M (value M xs (zero M) tree) x)
  newMean : D .finite (mean M (Fin.snoc xs x) (appendTree tree))

/-- Welford's mean update equals the appended row's mean under the selected
scalar theory. Padding and the previous reduction tree are unchanged. -/
theorem mean_append {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin N → α) (tree : ReductionTree N) (x : α)
    (hd : MeanAppendDomain M D xs tree x) :
    Welford.nextMean M x (mean M xs tree) (count M tree) =
      mean M (Fin.snoc xs x) (appendTree tree) := by
  apply mul_right_cancel R M D hM s _ _ (Welford.nextCount M (count M tree))
    hd.step.nextMean hd.newMean hd.step.nextCount hd.step.countNonzero hd.step.inverseCount
  have oldRecovery : mul M (mean M xs tree) (count M tree) = value M xs (zero M) tree :=
    div_mul_cancel R M D hM s _ _ (value_finite M D _ _ tree hd.old.inputTree)
      (value_finite M D _ _ tree hd.old.onesTree) hd.old.countNonzero hd.old.inverseCount
  rw [Welford.mean_step R M D hM s x _ _ hd.step, oldRecovery]
  have hnew : mean M (Fin.snoc xs x) (appendTree tree) =
      div M (add M (value M xs (zero M) tree) x) (Welford.nextCount M (count M tree)) := by
    simp only [mean, count_append, value_append, Fin.snoc_castSucc, Fin.snoc_last]
  rw [hnew]
  exact (div_mul_cancel R M D hM s _ _ hd.newSum hd.step.nextCount
    hd.step.countNonzero hd.step.inverseCount).symm

def shift {α : Type} (M : Algebra α) (x m n : α) : α :=
  sub M m (Welford.nextMean M x m n)

/-- Extra domains for comparing the backwards mean shift with the correction.
The relation between their signs or squares is not supplied as a premise. -/
structure ShiftSquareDomain {α : Type} (M : Algebra α) (D : Domain α)
    (x m n : α) : Prop where
  zero : D .finite (ScalarArithmetic.zero M)
  shift : D .finite (WelfordAppend.shift M x m n)
  joined : D .finite (add M (WelfordAppend.shift M x m n) (Welford.correction M x m n))
  negativeMean : D .finite (sub M (ScalarArithmetic.zero M) m)
  shiftSquare : D .finite (Welford.square M (WelfordAppend.shift M x m n))
  correctionSquare : D .finite (Welford.square M (Welford.correction M x m n))
  cross : D .finite (mul M (WelfordAppend.shift M x m n) (Welford.correction M x m n))
  negativeCross : D .finite (sub M (ScalarArithmetic.zero M)
    (mul M (WelfordAppend.shift M x m n) (Welford.correction M x m n)))
  shiftZero : D .finite (mul M (WelfordAppend.shift M x m n) (ScalarArithmetic.zero M))
  negativeShiftZero : D .finite
    (sub M (ScalarArithmetic.zero M) (mul M (WelfordAppend.shift M x m n) (ScalarArithmetic.zero M)))
  correctionZero : D .finite (mul M (Welford.correction M x m n) (ScalarArithmetic.zero M))
  negativeCorrectionZero : D .finite
    (sub M (ScalarArithmetic.zero M) (mul M (Welford.correction M x m n) (ScalarArithmetic.zero M)))

theorem shift_add_correction {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (x m n : α)
    (hm : Welford.MeanStepDomain M D x m n) (hd : ShiftSquareDomain M D x m n) :
    add M (shift M x m n) (Welford.correction M x m n) = zero M := by
  apply add_right_cancel R M D hM s _ _ m hd.joined hd.zero hm.mean hd.zero hd.negativeMean
  rw [add_assoc R M D hM s _ _ m hd.shift hm.correction hm.mean,
    add_comm R M D hM s _ m hm.correction hm.mean]
  exact (sub_add_cancel R M D hM s m (Welford.nextMean M x m n) hm.mean hm.nextMean).trans
    (zero_add R M D hM s m hm.mean hd.zero).symm

theorem shift_square {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (x m n : α)
    (hm : Welford.MeanStepDomain M D x m n) (hd : ShiftSquareDomain M D x m n) :
    Welford.square M (shift M x m n) = Welford.square M (Welford.correction M x m n) :=
  square_eq_of_add_eq_zero R M D hM s _ _ hd.shift hm.correction hd.zero
    hd.shiftSquare hd.correctionSquare hd.cross hd.negativeCross
    hd.shiftZero hd.negativeShiftZero hd.correctionZero hd.negativeCorrectionZero
    (shift_add_correction R M D hM s x m n hm hd)

structure VarianceAppendDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (tree : ReductionTree N) (x : α) : Prop where
  step : Welford.VarianceStepDomain M D x (mean M xs tree) (count M tree)
  recenter : RecenterDomain M D xs tree (Welford.nextMean M x (mean M xs tree) (count M tree))
  shift : ShiftSquareDomain M D x (mean M xs tree) (count M tree)
  newSum : D .finite (add M (value M xs (zero M) tree) x)
  newMean : D .finite (mean M (Fin.snoc xs x) (appendTree tree))
  residualSquare : D .finite (Welford.square M (Welford.residual M x (mean M xs tree) (count M tree)))
  shiftSquareZero : D .finite
    (mul M (Welford.square M (WelfordAppend.shift M x (mean M xs tree) (count M tree))) (zero M))
  negativeShiftSquareZero : D .finite
    (sub M (zero M)
      (mul M (Welford.square M (WelfordAppend.shift M x (mean M xs tree) (count M tree))) (zero M)))

theorem shift_square_sum {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin N → α) (tree : ReductionTree N) (x : α)
    (hd : VarianceAppendDomain M D xs tree x) :
    value M (fun _ => Welford.square M (shift M x (mean M xs tree) (count M tree))) (zero M) tree =
      mul M (count M tree) (Welford.square M (Welford.correction M x (mean M xs tree) (count M tree))) := by
  rw [constant_value R M D hM s _ hd.shift.shiftSquare hd.shift.zero hd.shiftSquareZero
    hd.negativeShiftSquareZero tree hd.recenter.centering.onesTree,
    shift_square R M D hM s x _ _ hd.step.toMeanStepDomain hd.shift]
  exact mul_comm R M D hM s _ _ hd.shift.correctionSquare hd.step.count

theorem squares_append {α : Type} (M : Algebra α) (xs : Fin N → α)
    (tree : ReductionTree N) (x c : α) :
    value M (Welford.deviationSquares M (Fin.snoc xs x) c) (zero M) (appendTree tree) =
      add M (value M (Welford.deviationSquares M xs c) (zero M) tree)
        (Welford.square M (sub M x c)) := by
  simp only [Welford.deviationSquares, value_append, Fin.snoc_castSucc, Fin.snoc_last]
  rfl

/-- The Welford variance update equals the appended row's square sum. The
proof uses scalar-derived centering, both shift-square identities, and the
explicit append tree; there is no admitted variance or reduction identity. -/
theorem variance_append {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin N → α) (tree : ReductionTree N) (x : α)
    (hd : VarianceAppendDomain M D xs tree x) :
    add M (value M (Welford.deviationSquares M xs (mean M xs tree)) (zero M) tree)
        (Welford.increment M x (mean M xs tree) (count M tree)) =
      value M (Welford.deviationSquares M (Fin.snoc xs x)
        (mean M (Fin.snoc xs x) (appendTree tree))) (zero M) (appendTree tree) := by
  have hm := mean_append R M D hM s xs tree x
    ⟨hd.recenter.centering, hd.step.toMeanStepDomain, hd.newSum, hd.newMean⟩
  have hc := centered_square_shift R M D hM s xs tree _ hd.recenter
  have ht : value M (fun _ => Welford.square M
      (sub M (mean M xs tree) (Welford.nextMean M x (mean M xs tree) (count M tree)))) (zero M) tree =
      mul M (count M tree) (Welford.square M (Welford.correction M x (mean M xs tree) (count M tree))) :=
    shift_square_sum R M D hM s xs tree x hd
  have hw : D .finite
      (mul M (count M tree) (Welford.square M (Welford.correction M x (mean M xs tree) (count M tree)))) := by
    rw [← ht]
    exact value_finite M D _ _ tree hd.recenter.squareShift.shiftSquares
  rw [← hm, squares_append, hc, ht, Welford.variance_step R M D hM s x _ _ hd.step]
  exact (congrArg (add M (value M (Welford.deviationSquares M xs (mean M xs tree)) (zero M) tree))
    (add_comm R M D hM s _ _ hd.residualSquare hw)).trans
    (add_assoc R M D hM s _ _ _
      (value_finite M D _ _ tree hd.recenter.squareShift.oldSquares) hw hd.residualSquare).symm

end VeriTile.Triton.FP.WelfordAppend
