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

theorem finiteTree_congr {α : Type} (M : Algebra α) (D : Domain α)
    (xs ys : Fin n → α) (seed seed' : α)
    (h : ∀ i, xs i = ys i) (hs : seed = seed') (tree : ReductionTree n) :
    FiniteTree M D xs seed tree ↔ FiniteTree M D ys seed' tree := by
  induction tree with
  | input i => simp only [FiniteTree, h i]
  | zero => simp only [FiniteTree, hs]
  | add a b ih₁ ih₂ =>
    simp only [FiniteTree, ih₁, ih₂, value_congr M xs ys seed seed' h hs]

section Factor
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
include R hM s

/-- Even an all-zero tree keeps its additions and padding; their identity is
derived from the accepted scalar addition law. -/
theorem value_zero (tree : ReductionTree n) (hz : D .finite (zero M)) :
    value M (fun _ => zero M) (zero M) tree = zero M := by
  induction tree with
  | input => rfl
  | zero => rfl
  | add a b ih₁ ih₂ =>
    simp only [value, ih₁, ih₂]
    exact add_zero R M D hM s _ hz

theorem value_empty (xs : Fin 0 → α) (tree : ReductionTree 0) (hz : D .finite (zero M)) :
    value M xs (zero M) tree = zero M :=
  (value_congr M xs (fun _ => zero M) _ _ (fun i => Fin.elim0 i) rfl tree).trans
    (value_zero R M D hM s tree hz)

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

/-- Pointwise addition distributes through a fixed explicit addition tree.
The sum's actual intermediate values must be finite alongside both parts. -/
theorem value_add_with_seed (xs ys : Fin n → α) (zx zy : α) (tree : ReductionTree n)
    (hx : FiniteTree M D xs zx tree) (hy : FiniteTree M D ys zy tree)
    (hxy : FiniteTree M D (fun i => ScalarArithmetic.add M (xs i) (ys i))
      (ScalarArithmetic.add M zx zy) tree) :
    value M (fun i => ScalarArithmetic.add M (xs i) (ys i))
        (ScalarArithmetic.add M zx zy) tree =
      ScalarArithmetic.add M (value M xs zx tree) (value M ys zy tree) := by
  induction tree with
  | input => rfl
  | zero => rfl
  | add a b ih₁ ih₂ =>
    have hright : D .finite (ScalarArithmetic.add M (value M xs zx b) (value M ys zy b)) := by
      rw [← ih₂ hx.2.1 hy.2.1 hxy.2.1]
      exact value_finite M D _ _ b hxy.2.1
    change ScalarArithmetic.add M _ _ = ScalarArithmetic.add M _ _
    rw [ih₁ hx.1 hy.1 hxy.1, ih₂ hx.2.1 hy.2.1 hxy.2.1]
    exact add_add_swap R M D hM s _ _ _ _
      (value_finite M D xs zx a hx.1) (value_finite M D ys zy a hy.1)
      (value_finite M D xs zx b hx.2.1) (value_finite M D ys zy b hy.2.1)
      hright hy.2.2

/-- Both reductions retain their literal-zero padding. The padding step uses
the accepted addition identity instead of silently deleting zero leaves. -/
theorem value_add (xs ys : Fin n → α) (tree : ReductionTree n)
    (hz : D .finite (zero M))
    (hx : FiniteTree M D xs (zero M) tree) (hy : FiniteTree M D ys (zero M) tree)
    (hxy : FiniteTree M D (fun i => ScalarArithmetic.add M (xs i) (ys i)) (zero M) tree) :
    value M (fun i => ScalarArithmetic.add M (xs i) (ys i)) (zero M) tree =
      ScalarArithmetic.add M (value M xs (zero M) tree) (value M ys (zero M) tree) := by
  have hz' := add_zero R M D hM s (zero M) hz
  have hxy' : FiniteTree M D (fun i => ScalarArithmetic.add M (xs i) (ys i))
      (ScalarArithmetic.add M (zero M) (zero M)) tree := by simpa only [hz'] using hxy
  simpa only [hz'] using value_add_with_seed R M D hM s xs ys (zero M) (zero M) tree hx hy hxy'

/-- A constant row factors into its value times an explicit tree of ones.
This does not identify the floating tree count with conversion of an integer. -/
theorem constant_value (c : α) (hc : D .finite c) (hz : D .finite (zero M))
    (hp : D .finite (mul M c (zero M)))
    (hn : D .finite (sub M (zero M) (mul M c (zero M))))
    (tree : ReductionTree n) (hf : FiniteTree M D (fun _ => one M) (zero M) tree) :
    value M (fun _ => c) (zero M) tree =
      mul M c (value M (fun _ => one M) (zero M) tree) := by
  calc
    _ = value M (fun _ => mul M c (one M)) (zero M) tree :=
      value_congr M _ _ _ _ (fun _ => (mul_one R M D hM s c hc).symm) rfl tree
    _ = _ := factor_left R M D hM s c hc hz hp hn _ tree hf

/-- Sum of deviations plus the same-tree sum of the center recovers the
original sum. Scalar CANCEL is lifted through the tree, with no reduction atom. -/
theorem deviations_add_center (xs : Fin n → α) (c : α) (tree : ReductionTree n)
    (hxs : ∀ i, D .finite (xs i)) (hc : D .finite c) (hz : D .finite (zero M))
    (hx : FiniteTree M D xs (zero M) tree)
    (hd : FiniteTree M D (fun i => sub M (xs i) c) (zero M) tree)
    (hm : FiniteTree M D (fun _ => c) (zero M) tree) :
    ScalarArithmetic.add M (value M (fun i => sub M (xs i) c) (zero M) tree)
      (value M (fun _ => c) (zero M) tree) = value M xs (zero M) tree := by
  have hp (i : Fin n) : ScalarArithmetic.add M (sub M (xs i) c) c = xs i :=
    sub_add_cancel R M D hM s (xs i) c (hxs i) hc
  have hf := (finiteTree_congr M D _ _ (zero M) (zero M) hp rfl tree).mpr hx
  exact (value_add R M D hM s _ _ tree hz hd hm hf).symm.trans
    (value_congr M _ _ (zero M) (zero M) hp rfl tree)

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
