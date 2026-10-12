import VeriTile.Triton.Math.Attention
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

/-! Independent real formulas for a row of resumed, masked attention.
Each pair contains a kept key's score and one output lane of its value vector.
The formula is a maximum and two weighted sums over these input pairs, plus
the initial M/L/Out carry. It does not evaluate a kernel or refer to its output.

An empty list of blocks means no loop iteration; an empty block means an
iteration whose row is entirely masked. These differ for the initial
(-infinity, 1, 0) state in the real projection. The lemmas below preserve that
distinction and prove that processing blocks gives the closed formula over
their concatenated keys. Connecting these facts to TypedSource's execution
and memory accesses remains a separate proof obligation. No FP rules are used.
-/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFormula
open VeriTile.Triton
noncomputable section

@[ext] structure RowState where
  center : WithBot ℝ
  denominator : ℝ
  numerator : ℝ

/-- The real projection's base-2 rescaling. A bottom operand propagates through
subtraction to -infinity, whose exponential is zero; IEEE NaN is not modeled. -/
def weight : WithBot ℝ → WithBot ℝ → ℝ
  | .some a, .some b => pow2 (a - b)
  | _, _ => 0

theorem weight_matches_semantics (a b : WithBot ℝ) :
    weight a b = (WithBot.realExp2 (WithBot.realSub a b)).unbotD 0 := by
  cases a <;> cases b
  all_goals try rfl
  simp [weight, pow2, WithBot.realExp2_coe, mul_comm]

@[simp] theorem weight_bot_left (b : WithBot ℝ) : weight ⊥ b = 0 := by
  cases b <;> rfl

@[simp] theorem weight_bot_right (a : WithBot ℝ) : weight a ⊥ = 0 := by
  cases a <;> rfl

@[simp] theorem weight_self (a : ℝ) : weight (a : WithBot ℝ) a = 1 := by
  simp [weight]

/-- Moving a known score through two centers preserves its total weight.
The order premise also excludes a finite score below a bottom intermediate. -/
theorem weight_chain (a b c : WithBot ℝ) (hab : a ≤ b) :
    weight a b * weight b c = weight a c := by
  cases a with
  | bot => simp
  | coe a =>
    cases b with
    | bot => exact False.elim (WithBot.coe_ne_bot (le_bot_iff.mp hab))
    | coe b =>
      cases c with
      | bot => simp
      | coe c =>
        simp only [weight]
        rw [← pow2_add]
        congr 1
        ring

def maxScore : List (ℝ × ℝ) → WithBot ℝ
  | [] => ⊥
  | x :: xs => (x.1 : WithBot ℝ) ⊔ maxScore xs

theorem maxScore_append (xs ys : List (ℝ × ℝ)) :
    maxScore (xs ++ ys) = maxScore xs ⊔ maxScore ys := by
  induction xs with
  | nil => simp [maxScore]
  | cons x xs ih => simp only [List.cons_append, maxScore, ih, sup_assoc]

theorem score_le_maxScore (xs : List (ℝ × ℝ)) (x : ℝ × ℝ) (hx : x ∈ xs) :
    (x.1 : WithBot ℝ) ≤ maxScore xs := by
  induction xs with
  | nil => simp at hx
  | cons y ys ih =>
    rcases List.mem_cons.mp hx with rfl | hx
    · exact le_sup_left
    · exact le_sup_of_le_right (ih hx)

/-- Taking value = 1 gives the denominator; value = Prod.snd gives the
numerator for one output lane. Both sums depend only on the input pairs. -/
def sumAt (xs : List (ℝ × ℝ)) (center : WithBot ℝ) (value : ℝ × ℝ → ℝ) : ℝ :=
  (xs.map (fun x => weight (x.1 : WithBot ℝ) center * value x)).sum

theorem sumAt_append (xs ys : List (ℝ × ℝ)) (center : WithBot ℝ)
    (value : ℝ × ℝ → ℝ) :
    sumAt (xs ++ ys) center value = sumAt xs center value + sumAt ys center value := by
  simp [sumAt]

theorem sumAt_shift (xs : List (ℝ × ℝ)) (oldCenter newCenter : WithBot ℝ)
    (value : ℝ × ℝ → ℝ) (h : ∀ x ∈ xs, (x.1 : WithBot ℝ) ≤ oldCenter) :
    sumAt xs oldCenter value * weight oldCenter newCenter = sumAt xs newCenter value := by
  induction xs with
  | nil => simp [sumAt]
  | cons x xs ih =>
    have hx := h x (by simp)
    have ht := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [sumAt, List.map_cons, List.sum_cons, add_mul] at *
    rw [mul_right_comm, weight_chain _ _ _ hx, ht]

/-- Closed formula after at least one block was processed. The initial carry
is rescaled once to the final maximum; all kept keys contribute directly. -/
def summarize (seed : RowState) (keys : List (ℝ × ℝ)) : RowState :=
  let m := seed.center ⊔ maxScore keys
  { center := m
    denominator := seed.denominator * weight seed.center m + sumAt keys m (fun _ => 1)
    numerator := seed.numerator * weight seed.center m + sumAt keys m Prod.snd }

/-- Splitting the same kept keys into blocks does not change the closed sums,
including empty blocks and bottom initial centers. -/
theorem summarize_append (seed : RowState) (xs ys : List (ℝ × ℝ)) :
    summarize (summarize seed xs) ys = summarize seed (xs ++ ys) := by
  let m := seed.center ⊔ maxScore xs
  let n := m ⊔ maxScore ys
  have hseed : seed.center ≤ m := le_sup_left
  have hscore : ∀ x ∈ xs, (x.1 : WithBot ℝ) ≤ m :=
    fun x hx => le_sup_of_le_right (score_le_maxScore xs x hx)
  have hden := sumAt_shift xs m n (fun _ => 1) hscore
  have hnum := sumAt_shift xs m n Prod.snd hscore
  have hc := weight_chain seed.center m n hseed
  apply RowState.ext
  · simp [summarize, maxScore_append, sup_assoc]
  · change (seed.denominator * weight seed.center m + sumAt xs m (fun _ => 1)) *
        weight m n + sumAt ys n (fun _ => 1) =
      seed.denominator * weight seed.center (seed.center ⊔ maxScore (xs ++ ys)) +
        sumAt (xs ++ ys) (seed.center ⊔ maxScore (xs ++ ys)) (fun _ => 1)
    rw [maxScore_append, ← sup_assoc]
    change _ = seed.denominator * weight seed.center n + sumAt (xs ++ ys) n (fun _ => 1)
    rw [add_mul, mul_assoc, hc, hden, sumAt_append]
    ring
  · change (seed.numerator * weight seed.center m + sumAt xs m Prod.snd) *
        weight m n + sumAt ys n Prod.snd =
      seed.numerator * weight seed.center (seed.center ⊔ maxScore (xs ++ ys)) +
        sumAt (xs ++ ys) (seed.center ⊔ maxScore (xs ++ ys)) Prod.snd
    rw [maxScore_append, ← sup_assoc]
    change _ = seed.numerator * weight seed.center n + sumAt (xs ++ ys) n Prod.snd
    rw [add_mul, mul_assoc, hc, hnum, sumAt_append]
    ring

theorem blocks_after_summary (seed : RowState) (keys : List (ℝ × ℝ))
    (blocks : List (List (ℝ × ℝ))) :
    blocks.foldl summarize (summarize seed keys) = summarize seed (keys ++ blocks.flatten) := by
  induction blocks generalizing keys with
  | nil => simp
  | cons block rest ih =>
    simp only [List.foldl_cons, summarize_append, ih, List.flatten_cons, List.append_assoc]

/-- A formula for the full list of blocks, with the zero-iteration carry
explicit. No list of abstract machine states appears on the right. -/
def result (seed : RowState) (blocks : List (List (ℝ × ℝ))) : RowState :=
  if blocks = [] then seed else summarize seed blocks.flatten

theorem block_updates_eq_closed (seed : RowState) (blocks : List (List (ℝ × ℝ))) :
    blocks.foldl summarize seed = result seed blocks := by
  cases blocks with
  | nil => simp [result]
  | cons block rest =>
    simpa [result] using blocks_after_summary seed block rest

def initial : RowState := ⟨⊥, 1, 0⟩

theorem no_iterations_preserve_initial : result initial [] = initial := by
  simp [result]

/-- Executing an all-masked block resets the real projection's initial
denominator to zero, unlike executing no block. -/
theorem all_masked_after_iteration (blocks : List (List (ℝ × ℝ)))
    (hne : blocks ≠ []) (hkeys : blocks.flatten = []) :
    result initial blocks = ⟨⊥, 0, 0⟩ := by
  simp [result, hne, hkeys, summarize, initial, maxScore, sumAt]

/-- A finite resumed center retains its carry when no new keys are kept. -/
theorem finite_carry_empty (m l a : ℝ) : summarize ⟨m, l, a⟩ [] = ⟨m, l, a⟩ := by
  simp [summarize, maxScore, sumAt]

/-- Unshifted mass of an initial center. Initialization contributes no carry
mass because its center is -infinity. -/
def mass : WithBot ℝ → ℝ
  | .some m => pow2 m
  | .none => 0

def unshiftedSum (keys : List (ℝ × ℝ)) (value : ℝ × ℝ → ℝ) : ℝ :=
  (keys.map (fun x => pow2 x.1 * value x)).sum

theorem weight_to_finite (a : WithBot ℝ) (m : ℝ) :
    weight a m = pow2 (-m) * mass a := by
  cases a with
  | bot => simp [mass]
  | coe a =>
    simp only [weight, mass]
    rw [← pow2_add]
    congr 1
    ring

theorem sumAt_factor (keys : List (ℝ × ℝ)) (m : ℝ) (value : ℝ × ℝ → ℝ) :
    sumAt keys m value = pow2 (-m) * unshiftedSum keys value := by
  simpa [sumAt, weight, unshiftedSum] using sum_map_pow2_sub m keys value

theorem maxScore_eq_bot (keys : List (ℝ × ℝ)) : maxScore keys = ⊥ ↔ keys = [] := by
  cases keys <;> simp [maxScore]

/-- The advertised output formula: the carried unshifted numerator plus the
new weighted values, divided by carried mass plus new softmax mass. No running
maximum or execution recurrence appears in this expression. -/
def normalized (seed : RowState) (keys : List (ℝ × ℝ)) : ℝ :=
  (seed.numerator * mass seed.center + unshiftedSum keys Prod.snd) /
    (seed.denominator * mass seed.center + unshiftedSum keys (fun _ => 1))

/-- Rescaling by the final maximum cancels even if the total denominator is
zero; division here is the existing total real operation, not IEEE division. -/
theorem summarize_normalized (seed : RowState) (keys : List (ℝ × ℝ)) :
    (summarize seed keys).numerator / (summarize seed keys).denominator =
      normalized seed keys := by
  cases hm : seed.center ⊔ maxScore keys with
  | bot =>
    obtain ⟨hs, hk⟩ := sup_eq_bot_iff.mp hm
    have he := (maxScore_eq_bot keys).mp hk
    subst keys
    simp [summarize, normalized, hs, mass, maxScore, sumAt, unshiftedSum]
  | coe m =>
    simp only [summarize, hm, weight_to_finite, sumAt_factor, normalized]
    have hn : seed.numerator * (pow2 (-m) * mass seed.center) +
        pow2 (-m) * unshiftedSum keys Prod.snd =
        pow2 (-m) * (seed.numerator * mass seed.center + unshiftedSum keys Prod.snd) := by ring
    have hd : seed.denominator * (pow2 (-m) * mass seed.center) +
        pow2 (-m) * unshiftedSum keys (fun _ => 1) =
        pow2 (-m) * (seed.denominator * mass seed.center + unshiftedSum keys (fun _ => 1)) := by ring
    rw [hn, hd, mul_div_mul_left _ _ (ne_of_gt (pow2_pos (-m)))]

/-- Fresh attention has the ordinary masked softmax weighted-sum output,
including no-iteration and all-masked rows. M/L still distinguish those cases. -/
theorem initialized_result_normalized (blocks : List (List (ℝ × ℝ))) :
    (result initial blocks).numerator / (result initial blocks).denominator =
      unshiftedSum blocks.flatten Prod.snd / unshiftedSum blocks.flatten (fun _ => 1) := by
  by_cases h : blocks = []
  · simp [h, result, initial, unshiftedSum]
  · rw [result, if_neg h, summarize_normalized]
    simp [normalized, initial, mass]

/-- The same max-free formula applies to arbitrary finite continuation data,
including an empty loop that simply returns the carried ratio. -/
theorem finite_resume_result_normalized (m l a : ℝ) (blocks : List (List (ℝ × ℝ))) :
    (result ⟨m, l, a⟩ blocks).numerator / (result ⟨m, l, a⟩ blocks).denominator =
      normalized ⟨m, l, a⟩ blocks.flatten := by
  by_cases h : blocks = []
  · simp only [h, result, List.flatten_nil, normalized, unshiftedSum,
      List.map_nil, List.sum_nil, add_zero, mass]
    exact (mul_div_mul_right a l (ne_of_gt (pow2_pos m))).symm
  · rw [result, if_neg h, summarize_normalized]

/-- Mathematical values written by the END/continuation paths. A missing L
value means that memory is preserved. The real memory interface represents a
bottom stored center by its finite fallback 0. -/
structure Writeback where
  maxValue : ℝ
  denominator? : Option ℝ
  output : ℝ

def writeback (finish : Bool) (row : RowState) : Writeback where
  maxValue := row.center.elim 0 (fun m =>
    if finish then m + Real.log row.denominator / Real.log 2 else m)
  denominator? := if finish then none else some row.denominator
  output := if finish then row.numerator / row.denominator else row.numerator

/-- The mathematical M value agrees with the real semantics' log2 update and
store conversion, including a bottom center and a zero denominator. -/
theorem writeback_center_matches (finish : Bool) (row : RowState) :
    (writeback finish row).maxValue = FloatDType.real.storeValue
      (if finish then WithBot.realAdd row.center (WithBot.realLog2 row.denominator)
        else row.center) := by
  cases finish <;> cases hm : row.center
  all_goals simp [writeback, hm, FloatDType.storeValue, WithBot.realLog2]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFormula
