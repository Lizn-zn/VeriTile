import VeriTile.Triton.Float.TermModel

/- Shared reduction lemmas for proof search. Every numerical rewrite is still
derived from the two scalar addition atoms; loading and casts stay opaque. -/
namespace VeriTile.Triton.FP.Equational

def rowInputs (xs : Structural.Values (Term α) .real [n]) : Fin n → Term α :=
  fun i => xs (i, PUnit.unit)

def fp32Sum (plans : Schedules) (n : Nat) (xs : Fin n → Term α) : Term α :=
  (algebra plans).reduceSum (some .fp32) (shape := [n]) ⟨0, by simp⟩ Bool.false
    (fun i => xs i.1) PUnit.unit

theorem fp32Sum.reindex {R : Spec.Assumptions ComputeStmt}
    {plans : Schedules} {n : Nat} {xs ys : Fin n → Term α}
    (hc : Spec.Derivation R commuteLHS commuteRHS)
    (ha : Spec.Derivation R associateLHS associateRHS)
    (σ : Equiv.Perm (Fin n)) (inputs : ys = xs ∘ σ) :
    TermEq R (fp32Sum plans n xs) (fp32Sum plans n ys) := by
  rw [inputs]
  simp only [fp32Sum, algebra, reductionInputs, TileShape.insertAxisIndex, SumTree.evalAt_fp32]
  exact ReductionPlan.reorder hc ha _ xs _ σ

/-- Common opaque calls preserve related arguments without changing their
operation, intrinsic, precision, cast or argument multiplicity. -/
theorem TermEq.app_congr {R : Spec.Assumptions ComputeStmt} (op : Symbol)
    {xs ys : List (Term α)} (args : List.Forall₂ (TermEq R) xs ys) :
    TermEq R (.app op xs) (.app op ys) := by
  have go (before : List (Term α)) :
      ∀ {as bs}, List.Forall₂ (TermEq R) as bs →
        TermEq R (.app op (before ++ as)) (.app op (before ++ bs)) := by
    intro as bs h
    induction h generalizing before with
    | nil => exact .refl _
    | @cons a b as bs hab _ ih =>
      exact .trans (.context op before as hab)
        (by simpa only [List.append_assoc, List.cons_append, List.nil_append] using ih (before ++ [b]))
  simpa using go [] args

end VeriTile.Triton.FP.Equational
