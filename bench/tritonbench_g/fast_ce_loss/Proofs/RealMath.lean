import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.RealMath
open VeriTile.Triton TiledLogSumExp
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x

/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)

def loss {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V : Nat)
    (lab : Int) (xs : Fin B → ℝ) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else lse cap scale SC SL V 0 xs-transform cap scale SC SL gather

theorem log_sum_shift {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (f : ι → ℝ) (m : ℝ) :
    m + Real.log (∑ j ∈ s,Real.exp (f j-m))=Real.log (∑ j ∈ s,Real.exp (f j)) := by
  have hf : ∀ j, Real.exp (f j)=Real.exp m*Real.exp (f j-m) := by
    intro j
    rw [← Real.exp_add]
    congr 1
    ring
  simp_rw [hf]
  rw [← Finset.mul_sum,Real.log_mul (Real.exp_ne_zero _) (ne_of_gt (Finset.sum_pos (fun _ _ => Real.exp_pos _) hs)),Real.log_exp]

/-- The semantic carrier reduces to a pure finite exponential sum, independently of any kernel. -/
theorem carrier_lse {B : Nat} (p : Fin B → Prop) [DecidablePred p]
    (f : Fin B → ℝ) (hu : (Finset.univ : Finset (Fin B)).Nonempty)
    (hp : (Finset.univ.filter p).Nonempty) :
    let xs : Fin B → WithBot ℝ := fun i => if p i then (some (f i) : WithBot ℝ) else none
    let m := Finset.univ.sup' hu xs
    Option.map₂ (fun x y => x+y) m
      (WithBot.realLog (∑ j, WithBot.realExp (Option.map₂ (fun x y => x-y) (xs j) m))) =
      some (Real.log (∑ j ∈ Finset.univ.filter p, Real.exp (f j))) := by
  dsimp only
  erw [withBot_sup'_partial Finset.univ hu p f hp]
  have he (j : Fin B) (m : ℝ) :
      WithBot.realExp (Option.map₂ (fun x y => x-y)
        (if p j then (some (f j) : WithBot ℝ) else none) ((m : ℝ) : WithBot ℝ)) =
      ((if p j then Real.exp (f j-m) else 0 : ℝ) : WithBot ℝ) := by
    by_cases h : p j
    · simp only [h,↓reduceIte,Option.map₂_some_coe,WithBot.realExp_some]
      rfl
    · simp only [h,↓reduceIte]
      rfl
  have hsum (m : ℝ) :
      (∑ j : Fin B, WithBot.realExp (Option.map₂ (fun x y => x-y)
        (if p j then (some (f j) : WithBot ℝ) else none) ((m : ℝ) : WithBot ℝ))) =
      ((∑ j ∈ Finset.univ.filter p,Real.exp (f j-m) : ℝ) : WithBot ℝ) := by
    rw [Finset.sum_filter, WithBot.coe_sum]
    apply Finset.sum_congr rfl
    intro j _
    exact he j m
  erw [hsum]
  rw [WithBot.realLog_coe]
  change some ((Finset.univ.filter p).sup' hp f +
    Real.log (∑ j ∈ Finset.univ.filter p,Real.exp (f j-(Finset.univ.filter p).sup' hp f))) = _
  congr 1
  exact log_sum_shift _ hp f _

def transformCarrier (cap scale : Bool) (SC SL : ℝ) (x : WithBot ℝ) : WithBot ℝ :=
  let x := if scale then Option.map (SL*·) x else x
  if cap then Option.map (SC*·) (WithBot.realTanh (Option.map (·/SC) x)) else x

theorem carrier_transform (cap scale : Bool) (SC SL : ℝ) (p : Prop) [Decidable p] (x : ℝ) :
    transformCarrier cap scale SC SL (if p then some x else none) =
      if cap=Bool.true ∨ p then some (if p then transform cap scale SC SL x else -SC) else none := by
  cases cap <;> cases scale <;> by_cases h : p
  all_goals simp [transformCarrier,transform,h,WithBot.realTanh]

/-- Whole transformed block LSE, including the capped padding contribution. -/
theorem transformed_lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat)
    (xs : Fin B → ℝ) (hb : 0<B) (hv : start<V) :
    let vals : Fin B → WithBot ℝ := fun j => transformCarrier cap scale SC SL
      (if start+j.val<V then some (xs j) else none)
    let m := Finset.univ.sup' ⟨⟨0,hb⟩,Finset.mem_univ _⟩ vals
    Option.map₂ (fun x y => x+y) m
      (WithBot.realLog (∑ j, WithBot.realExp (Option.map₂ (fun x y => x-y) (vals j) m))) =
      some (lse cap scale SC SL V start xs) := by
  dsimp only
  simp_rw [carrier_transform]
  have hp : (Finset.univ.filter (fun j : Fin B => cap=Bool.true ∨ start+j.val<V)).Nonempty := by
    refine ⟨⟨0,hb⟩,?_⟩
    simp [hv]
  erw [carrier_lse _ _ _ hp]
  congr 1
  unfold lse
  congr 1
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro j _
  cases cap <;> by_cases h : start+j.val<V <;> simp [h]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.RealMath
