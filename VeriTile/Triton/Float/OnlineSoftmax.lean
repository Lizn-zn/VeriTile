/- A scalar-derived invariant for the original online softmax recurrence.
The running maximum stays opaque. Only the matching intrinsic EXP-SUB law
remains a numerical premise beyond the admitted arithmetic atoms. -/
import VeriTile.Triton.Float.SoftmaxShift

namespace VeriTile.Triton.FP.OnlineSoftmax
open Structural Guarded ScalarArithmetic ScalarReduction
open SoftmaxShift (exp IntrinsicExpSub)

def maximum {α : Type} (M : Algebra α) (a b : α) : α :=
  M.binary (some .fp32) .real .max a b

def update {α : Type} (M : Algebra α) (x : α) (acc : α × α) : α × α :=
  let m := maximum M acc.1 x
  (m, add M (mul M (exp M (sub M acc.1 m)) acc.2) (exp M (sub M x m)))

def state {α : Type} (M : Algebra α) (xs : Nat → α) : Nat → α × α
  | 0 => (M.negInf, zero M)
  | n + 1 => update M (xs n) (state M xs n)

def prefixSum {α : Type} (M : Algebra α) (xs : Nat → α) : Nat → α
  | 0 => zero M
  | n + 1 => add M (prefixSum M xs n) (exp M (xs n))

/-- The online prefix sum is a valid explicit addition schedule. Its single
padding zero is retained; no reduction or count identity is assumed. -/
theorem prefixSum_tree {α : Type} (M : Algebra α) (xs : Nat → α) (n : Nat) :
    prefixSum M xs n = value M (fun i : Fin n => exp M (xs i.val)) (zero M)
      (WelfordInduction.tree .zero n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [prefixSum, WelfordInduction.tree, WelfordAppend.value_append, ih,
      Fin.val_castSucc, Fin.val_last]

/-- Initialization uses the actual exp(-inf - newMax) result. Its finite
domain justifies multiplication by zero; -inf itself is never declared finite. -/
structure InitDomain {α : Type} (M : Algebra α) (D : Domain α) (x : α) : Prop where
  input : D .finite x
  center : D .finite (maximum M M.negInf x)
  inputExp : D .finite (exp M x)
  centerExp : D .finite (exp M (maximum M M.negInf x))
  centerExpNonzero : D .nonzero (exp M (maximum M M.negInf x))
  inverseCenterExp : D .finite (div M (one M) (exp M (maximum M M.negInf x)))
  zero : D .finite (ScalarArithmetic.zero M)
  seedFactor : D .finite (exp M (sub M M.negInf (maximum M M.negInf x)))
  seedProduct : D .finite
    (mul M (exp M (sub M M.negInf (maximum M M.negInf x))) (ScalarArithmetic.zero M))
  negativeSeedProduct : D .finite (sub M (ScalarArithmetic.zero M)
    (mul M (exp M (sub M M.negInf (maximum M M.negInf x))) (ScalarArithmetic.zero M)))
  newTerm : D .finite (exp M (sub M x (maximum M M.negInf x)))

structure StepDomain {α : Type} (M : Algebra α) (D : Domain α) (x m l : α) : Prop where
  input : D .finite x
  oldCenter : D .finite m
  newCenter : D .finite (maximum M m x)
  inputExp : D .finite (exp M x)
  oldCenterExp : D .finite (exp M m)
  newCenterExp : D .finite (exp M (maximum M m x))
  newCenterExpNonzero : D .nonzero (exp M (maximum M m x))
  inverseNewCenterExp : D .finite (div M (one M) (exp M (maximum M m x)))
  oldScale : D .finite (exp M (sub M m (maximum M m x)))
  newTerm : D .finite (exp M (sub M x (maximum M m x)))
  oldTotal : D .finite l
  scaledOldTotal : D .finite (mul M (exp M (sub M m (maximum M m x))) l)
  newTotal : D .finite (update M x (m, l)).2

structure IterationDomain {α : Type} (M : Algebra α) (D : Domain α) (xs : Nat → α) (N : Nat) : Prop where
  initial : InitDomain M D (xs 0)
  steps : ∀ i, 0 < i → i < N → StepDomain M D (xs i) (state M xs i).1 (state M xs i).2

theorem IterationDomain.restrict {α : Type} {M : Algebra α} {D : Domain α} {xs : Nat → α}
    {N k : Nat} (h : IterationDomain M D xs N) (hk : k ≤ N) : IterationDomain M D xs k :=
  ⟨h.initial, fun i hi hn => h.steps i hi (by omega)⟩

section Derivation
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α) (hExp : IntrinsicExpSub M D)
include R hM s hExp

theorem initial_recovery (x : α) (hd : InitDomain M D x) :
    mul M (update M x (M.negInf, zero M)).2 (exp M (update M x (M.negInf, zero M)).1) =
      exp M x := by
  unfold update
  dsimp only
  rw [mul_zero R M D hM s _ hd.seedFactor hd.zero hd.seedProduct hd.negativeSeedProduct,
    zero_add R M D hM s _ hd.newTerm hd.zero,
    hExp.apply x _ hd.input hd.center]
  exact div_mul_cancel R M D hM s _ _ hd.inputExp hd.centerExp
    hd.centerExpNonzero hd.inverseCenterExp

/-- One update recovers the old unshifted mass plus the next exponential.
The previous invariant is not a premise of this local scalar identity. -/
theorem step_recovery (x m l : α) (hd : StepDomain M D x m l) :
    mul M (update M x (m, l)).2 (exp M (update M x (m, l)).1) =
      add M (mul M l (exp M m)) (exp M x) := by
  let n := maximum M m x
  have recover (a : α) (ha : D .finite a) (he : D .finite (exp M a)) :
      mul M (exp M (sub M a n)) (exp M n) = exp M a := by
    rw [hExp.apply a n ha hd.newCenter]
    exact div_mul_cancel R M D hM s _ _ he hd.newCenterExp
      hd.newCenterExpNonzero hd.inverseNewCenterExp
  change mul M (add M (mul M (exp M (sub M m n)) l) (exp M (sub M x n))) (exp M n) = _
  rw [add_mul R M D hM s _ _ _ hd.scaledOldTotal hd.newTerm hd.newCenterExp hd.newTotal,
    recover x hd.input hd.inputExp,
    mul_comm R M D hM s (exp M (sub M m n)) l hd.oldScale hd.oldTotal,
    mul_assoc R M D hM s l _ _ hd.oldTotal hd.oldScale hd.newCenterExp,
    recover m hd.oldCenter hd.oldCenterExp]

/-- Every positive-length prefix recovers its explicit sum. The proof keeps
the actual running maximum and checks every iteration's arithmetic domains. -/
theorem state_recovery (xs : Nat → α) (N : Nat) (hN : 0 < N)
    (hd : IterationDomain M D xs N) :
    mul M (state M xs N).2 (exp M (state M xs N).1) = prefixSum M xs N := by
  induction N with
  | zero => omega
  | succ n ih =>
    cases n with
    | zero =>
      change mul M (update M (xs 0) (M.negInf, zero M)).2
        (exp M (update M (xs 0) (M.negInf, zero M)).1) = add M (zero M) (exp M (xs 0))
      rw [initial_recovery R M D hM s hExp (xs 0) hd.initial,
        zero_add R M D hM s _ hd.initial.inputExp hd.initial.zero]
    | succ n =>
      change mul M (update M (xs (n + 1)) (state M xs (n + 1))).2
        (exp M (update M (xs (n + 1)) (state M xs (n + 1))).1) =
        add M (prefixSum M xs (n + 1)) (exp M (xs (n + 1)))
      rw [step_recovery R M D hM s hExp _ _ _ (hd.steps (n + 1) (by omega) (by omega)),
        ih (by omega) (hd.restrict (by omega))]

/-- The online numerator divided by the actual running total equals the
unshifted prefix normalization. Its center is the computed m, with no max
identity. The additional premises are only normalization-domain predicates. -/
theorem normalized_prefix (xs : Nat → α) (N : Nat) (hN : 0 < N)
    (hi : IterationDomain M D xs N)
    (hd : SoftmaxShift.ShiftDomain M D (fun i : Fin N => xs i.val) (state M xs N).1
      (WelfordInduction.tree .zero N))
    (hl : D .finite (state M xs N).2) (i : Fin N) :
    div M (exp M (sub M (xs i.val) (state M xs N).1)) (state M xs N).2 =
      div M (exp M (xs i.val)) (prefixSum M xs N) := by
  let m := (state M xs N).1
  let l := (state M xs N).2
  let row : Fin N → α := fun i => xs i.val
  let tree := WelfordInduction.tree .zero N
  have hrec := state_recovery R M D hM s hExp xs N hN hi
  have recover : mul M (mul M l (exp M m)) (SoftmaxShift.scale M m) = l := by
    rw [mul_assoc R M D hM s l _ _ hl hd.centerExp hd.normalization.scale,
      show mul M (exp M m) (SoftmaxShift.scale M m) = one M from
        mul_rcp_cancel R M D hM s _ hd.centerExp hd.centerExpNonzero,
      mul_one R M D hM s l hl]
  rw [hrec] at recover
  have hrow := SoftmaxShift.shifted_lane R M D hM s hExp row m tree hd
  have hf : value M (SoftmaxShift.shifted M row m) (zero M) tree =
      mul M (prefixSum M xs N) (SoftmaxShift.scale M m) := by
    rw [value_congr M _ _ (zero M) (zero M) hrow rfl tree,
      factor_right R M D hM s _ hd.normalization.scale hd.normalization.zeroFinite
        hd.normalization.scaledZero hd.normalization.negScaledZero _ hd.normalization.inputs
        tree hd.normalization.partialSums]
    rw [prefixSum_tree]
    rfl
  have htotal : l = value M (SoftmaxShift.shifted M row m) (zero M) tree :=
    recover.symm.trans hf.symm
  change div M (SoftmaxShift.shifted M row m i) l = _
  rw [htotal]
  have hn := SoftmaxShift.normalized R M D hM s hExp row m tree hd i
  rw [prefixSum_tree]
  exact hn

end Derivation
end VeriTile.Triton.FP.OnlineSoftmax
