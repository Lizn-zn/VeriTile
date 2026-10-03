/- Factor a common fp32 scale through an explicit addition tree.
This is a derivation from scalar atoms, not a numerical reduction assumption.
The tree and its padding remain visible, and all arithmetic domains are kept. -/
import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.TermModel

namespace VeriTile.Triton.FP.ScalarReduction
open Structural Guarded ScalarArithmetic
open Equational (ReductionTree ReductionPlan)

def value {α : Type} (M : Algebra α) (xs : Fin n → α) (seed : α) : ReductionTree n → α
  | .input i => xs i
  | .zero => seed
  | .add a b => ScalarArithmetic.add M (value M xs seed a) (value M xs seed b)

/-- This condition checks actual intermediate additions, not just input leaves.
For an IEEE interpretation it excludes overflowing partial sums at rule sites. -/
def FiniteTree {α : Type} (M : Algebra α) (D : Domain α) (xs : Fin n → α)
    (seed : α) : ReductionTree n → Prop
  | .input i => D .finite (xs i)
  | .zero => D .finite seed
  | .add a b => FiniteTree M D xs seed a ∧ FiniteTree M D xs seed b ∧
      D .finite (ScalarArithmetic.add M (value M xs seed a) (value M xs seed b))

theorem value_finite {α : Type} (M : Algebra α) (D : Domain α) (xs : Fin n → α)
    (seed : α) (tree : ReductionTree n) (h : FiniteTree M D xs seed tree) :
    D .finite (value M xs seed tree) := by
  cases tree with
  | input => exact h
  | zero => exact h
  | add => exact h.2.2

theorem value_congr {α : Type} (M : Algebra α) (xs ys : Fin n → α) (seed seed' : α)
    (h : ∀ i, xs i = ys i) (hs : seed = seed') (tree : ReductionTree n) :
    value M xs seed tree = value M ys seed' tree := by
  induction tree with
  | input i => exact h i
  | zero => exact hs
  | add a b ih₁ ih₂ => simp only [value, ih₁, ih₂]

section Factor
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
include R hM s

/-- Transforming both inputs and padding uses distribution alone. -/
theorem factor_with_seed (c : α) (hc : D .finite c) (xs : Fin n → α) (seed : α)
    (tree : ReductionTree n) (hf : FiniteTree M D xs seed tree) :
    value M (fun i => mul M c (xs i)) (mul M c seed) tree =
      mul M c (value M xs seed tree) := by
  induction tree with
  | input => rfl
  | zero => rfl
  | add a b ih₁ ih₂ =>
    change ScalarArithmetic.add M
      (value M (fun i => mul M c (xs i)) (mul M c seed) a)
      (value M (fun i => mul M c (xs i)) (mul M c seed) b) = _
    rw [ih₁ hf.1, ih₂ hf.2.1]
    exact (mul_distrib R M D hM s c _ _ hc
      (value_finite M D xs seed a hf.1) (value_finite M D xs seed b hf.2.1)).symm

/-- Actual sum padding stays literal zero. Its scaling law is derived from
distribution, cancellation and the accepted addition identities. -/
theorem factor_left (c : α) (hc : D .finite c) (hz : D .finite (zero M))
    (hp : D .finite (mul M c (zero M)))
    (hn : D .finite (sub M (zero M) (mul M c (zero M))))
    (xs : Fin n → α) (tree : ReductionTree n)
    (hf : FiniteTree M D xs (zero M) tree) :
    value M (fun i => mul M c (xs i)) (zero M) tree =
      mul M c (value M xs (zero M) tree) := by
  have h := factor_with_seed R M D hM s c hc xs (zero M) tree hf
  rw [mul_zero R M D hM s c hc hz hp hn] at h
  exact h

theorem factor_right (c : α) (hc : D .finite c) (hz : D .finite (zero M))
    (hp : D .finite (mul M c (zero M)))
    (hn : D .finite (sub M (zero M) (mul M c (zero M))))
    (xs : Fin n → α) (hxs : ∀ i, D .finite (xs i)) (tree : ReductionTree n)
    (hf : FiniteTree M D xs (zero M) tree) :
    value M (fun i => mul M (xs i) c) (zero M) tree =
      mul M (value M xs (zero M) tree) c := by
  calc
    _ = value M (fun i => mul M c (xs i)) (zero M) tree :=
      value_congr M _ _ _ _ (fun i => mul_comm R M D hM s _ c (hxs i) hc) rfl tree
    _ = mul M c (value M xs (zero M) tree) :=
      factor_left R M D hM s c hc hz hp hn xs tree hf
    _ = _ := mul_comm R M D hM s c _ hc (value_finite M D xs (zero M) tree hf)

end Factor

/-- Arithmetic domains needed when normalizing a row before/after a common
scale. This record contains only finite/nonzero predicates, never equations. -/
structure NormalizationDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin n → α) (c : α) (tree : ReductionTree n) : Prop where
  inputs : ∀ i, D .finite (xs i)
  scale : D .finite c
  scaleNonzero : D .nonzero c
  zeroFinite : D .finite (zero M)
  scaledZero : D .finite (mul M c (zero M))
  negScaledZero : D .finite (sub M (zero M) (mul M c (zero M)))
  partialSums : FiniteTree M D xs (zero M) tree
  sumNonzero : D .nonzero (value M xs (zero M) tree)
  scaledInputs : ∀ i, D .finite (mul M (xs i) c)
  scaledSum : D .finite (mul M (value M xs (zero M) tree) c)
  scaledSumNonzero : D .nonzero (mul M (value M xs (zero M) tree) c)
  inverseSum : D .finite (div M (one M) (value M xs (zero M) tree))
  inverseScale : D .finite (div M (one M) c)
  inverseScaledSum : D .finite (div M (one M) (mul M (value M xs (zero M) tree) c))
  quotient : ∀ i, D .finite (div M (xs i) (value M xs (zero M) tree))
  scaledQuotient : ∀ i, D .finite
    (div M (mul M (xs i) c) (mul M (value M xs (zero M) tree) c))
  recoveredNumerator : ∀ i, D .finite
    (mul M (div M (mul M (xs i) c) (mul M (value M xs (zero M) tree) c))
      (value M xs (zero M) tree))

/-- Shared-scale normalization follows from the scalar theory and explicit
addition tree. There is no whole-row atom and no equality premise here. -/
theorem normalize_reduction {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin n → α) (c : α) (tree : ReductionTree n)
    (hd : NormalizationDomain M D xs c tree) (i : Fin n) :
    div M (mul M (xs i) c) (value M (fun j => mul M (xs j) c) (zero M) tree) =
      div M (xs i) (value M xs (zero M) tree) := by
  rw [factor_right R M D hM s c hd.scale hd.zeroFinite hd.scaledZero hd.negScaledZero
    xs hd.inputs tree hd.partialSums]
  exact normalize_scale R M D hM s _ _ c (hd.inputs i)
    (value_finite M D xs (zero M) tree hd.partialSums) hd.scale hd.sumNonzero hd.scaleNonzero
    (hd.scaledInputs i) hd.scaledSum hd.scaledSumNonzero hd.inverseSum hd.inverseScale
    hd.inverseScaledSum (hd.quotient i) (hd.scaledQuotient i) (hd.recoveredNumerator i)

def inputs {α : Type} (shape : TileShape) (axis : Fin shape.length) (keepDims : Bool)
    (xs : Values α .real shape) :
    TileIndex (TileShape.reduceShape shape axis keepDims) →
      Fin (TileShape.axisDim shape axis) → α :=
  match keepDims with
  | false => fun i k => xs (TileShape.insertAxisIndex shape axis i k)
  | true => fun i k => xs (TileShape.replaceAxisIndex shape axis i k)

/-- Expose an fp32 sum's explicit schedule to the execution interpreter.
Other precisions and all non-sum operations retain their original meanings. -/
def algebra {α : Type} (M : Algebra α) (plans : Equational.Schedules) : Algebra α :=
  { M with reduceSum := fun precision {shape} axis keepDims xs i =>
      match precision with
      | some .fp32 =>
        value M (inputs shape axis keepDims xs i) (zero M)
          (plans (some .fp32) shape axis keepDims i).tree
      | _ => M.reduceSum precision axis keepDims xs i }

theorem scalar_values_unchanged {α : Type} (M : Algebra α) (plans : Equational.Schedules)
    (a b c : α) (atom : Atom) :
    leftValue (algebra M plans) a b c atom = leftValue M a b c atom ∧
      rightValue (algebra M plans) a b c atom = rightValue M a b c atom := by
  cases atom <;> exact ⟨rfl, rfl⟩

/-- The same scalar factor theorem now addresses the actual reduceSum field
used by typed execution, including symbolic shape, axis and output position. -/
theorem reduce_scale {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (plans : Equational.Schedules) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α)
    (shape : TileShape) (axis : Fin shape.length) (keepDims : Bool)
    (xs : Values α .real shape) (i : TileIndex (TileShape.reduceShape shape axis keepDims))
    (c : α) (hc : D .finite c) (hz : D .finite (zero M))
    (hp : D .finite (mul M c (zero M)))
    (hn : D .finite (sub M (zero M) (mul M c (zero M))))
    (hxs : ∀ j, D .finite (xs j))
    (hf : FiniteTree M D (inputs shape axis keepDims xs i) (zero M)
      (plans (some .fp32) shape axis keepDims i).tree) :
    (algebra M plans).reduceSum (some .fp32) axis keepDims (fun j => mul M (xs j) c) i =
      mul M ((algebra M plans).reduceSum (some .fp32) axis keepDims xs i) c := by
  cases keepDims <;>
    exact factor_right R M D hM s c hc hz hp hn _ (fun j => hxs _) _ hf

end VeriTile.Triton.FP.ScalarReduction
