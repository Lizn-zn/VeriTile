import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealFormula

/-! Tile algebra for one masked real attention block. The operands are real
input values and the initial row carry. These lemmas connect dot/reduction
semantics to the independent row formula without importing FP proof rules. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles
open VeriTile.Triton
open scoped BigOperators
noncomputable section

def finite {shape : TileShape} (f : TileIndex shape → ℝ) : Tile .real shape :=
  ⟨fun i => some (f i)⟩

theorem dot_finite (BM BN BD : Nat)
    (q : TileIndex [BM, BD] → ℝ) (k : TileIndex [BD, BN] → ℝ) :
    Tile.dot [] (finite q) (finite k) = finite (fun i : TileIndex [BM, BN] =>
      ∑ d : Fin BD, q (i.1, d, PUnit.unit) * k (d, i.2.1, PUnit.unit)) := by
  apply Tile.ext
  rintro ⟨row, key, ⟨⟩⟩
  change (∑ d : Fin BD, ((q (row, d, PUnit.unit) * k (d, key, PUnit.unit) : ℝ) : WithBot ℝ)) =
    ((∑ d : Fin BD, q (row, d, PUnit.unit) * k (d, key, PUnit.unit) : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum _ _).symm

def pairs {BN : Nat} (score value : Fin BN → ℝ) (keep : Fin BN → Bool) : List (ℝ × ℝ) :=
  ((List.finRange BN).filter keep).map (fun j => (score j, value j))

theorem maxScore_le_iff (keys : List (ℝ × ℝ)) (m : WithBot ℝ) :
    RealFormula.maxScore keys ≤ m ↔ ∀ x ∈ keys, (x.1 : WithBot ℝ) ≤ m := by
  induction keys with
  | nil => simp [RealFormula.maxScore]
  | cons x xs ih => simp [RealFormula.maxScore, sup_le_iff, ih]

theorem maxScore_pairs {BN : Nat} (score value : Fin BN → ℝ) (keep : Fin BN → Bool) :
    RealFormula.maxScore (pairs score value keep) =
      Finset.univ.sup (fun j => if keep j then (score j : WithBot ℝ) else ⊥) := by
  apply le_antisymm
  · apply (maxScore_le_iff _ _).mpr
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hx
    have hk := (List.mem_filter.mp hj).2
    simpa [hk] using Finset.le_sup (f := fun j => if keep j then (score j : WithBot ℝ) else ⊥)
      (Finset.mem_univ j)
  · apply Finset.sup_le
    intro j _
    by_cases hk : keep j = Bool.true
    · simp only [hk, if_true]
      apply RealFormula.score_le_maxScore _ (score j, value j)
      apply List.mem_map.mpr
      exact ⟨j, List.mem_filter.mpr ⟨List.mem_finRange j, hk⟩, rfl⟩
    · simp [hk]

theorem sum_pairs {BN : Nat} (score value : Fin BN → ℝ) (keep : Fin BN → Bool)
    (f : ℝ × ℝ → ℝ) :
    ((pairs score value keep).map f).sum = ∑ j : Fin BN, if keep j then f (score j, value j) else 0 := by
  have hlist : ∀ js : List (Fin BN),
      ((js.filter keep).map (fun j => f (score j, value j))).sum =
        (js.map (fun j => if keep j then f (score j, value j) else 0)).sum := by
    intro js
    induction js with
    | nil => simp
    | cons j js ih => cases hk : keep j <;> simp [hk, ih]
  simp only [pairs, List.map_map]
  change (((List.finRange BN).filter keep).map (fun j => f (score j, value j))).sum = _
  rw [hlist, ← List.sum_ofFn]
  congr 1
  rw [List.ofFn_eq_map]

/-- Max reduction, including a block whose every key is masked. Only the
physical reduction dimension must be nonempty; the kept key set may be empty. -/
theorem reduce_max_masked (BM BN : Nat) (hN : 0 < BN)
    (score : Fin BM → Fin BN → ℝ) (keep : Fin BM → Fin BN → Bool) :
    Tile.reduceMaxDrop ⟨1, by simp⟩
      (⟨fun i => if keep i.1 i.2.1 then (score i.1 i.2.1 : WithBot ℝ) else ⊥⟩ : Tile .real [BM, BN]) =
    some (⟨fun i => RealFormula.maxScore (pairs (score i.1) (fun _ => 0) (keep i.1))⟩ : Tile .real [BM]) := by
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [BM, BN] ⟨1, by simp⟩ from hN)]
  congr 1
  apply Tile.ext
  rintro ⟨row, ⟨⟩⟩
  simp only [Finset.sup'_eq_sup, maxScore_pairs]
  rfl

theorem exp2_sub_eq_weight (a b : WithBot ℝ) :
    WithBot.realExp2 (WithBot.realSub a b) = some (RealFormula.weight a b) := by
  rw [RealFormula.weight_matches_semantics]
  cases a <;> cases b <;> rfl

/-- The real carrier has no NaN, so the source's self-inequality repair mask
is false even for a bottom temporary. The FP proof retains its opaque branch. -/
theorem self_inequality_false {shape : TileShape} (tmp : Tile .real shape) (i : TileIndex shape) :
    ComparableDType.real.ne (tmp.data i) (tmp.data i) = Bool.false := by
  simp [ComparableDType.ne]

def center {BM BN : Nat} (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ) (row : Fin BM) : WithBot ℝ :=
  old row ⊔ RealFormula.maxScore (pairs (score row) (fun _ => 0) (keep row))

def probability {BM BN : Nat} (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (row : Fin BM) (key : Fin BN) : ℝ :=
  if keep row key then RealFormula.weight (score row key) (center score keep old row) else 0

theorem alpha_matches (BM BN : Nat) (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ) :
    Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil)
      (⟨fun i => old i.1⟩ : Tile .real [BM]) (⟨fun i => center score keep old i.1⟩ : Tile .real [BM])) =
      finite (fun i : TileIndex [BM] => RealFormula.weight (old i.1) (center score keep old i.1)) := by
  apply Tile.ext
  rintro ⟨row, ⟨⟩⟩
  exact exp2_sub_eq_weight _ _

/-- Subtracting the new row maximum and exponentiating the masked score
produces finite probabilities, with exact zeros on rejected keys. -/
theorem probabilities_match (BM BN : Nat) (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ) :
    Tile.uop WithBot.realExp2
      (Tile.bop NumericDType.real.sub (.consSame (.consR .nil))
        (⟨fun i => if keep i.1 i.2.1 then (score i.1 i.2.1 : WithBot ℝ) else ⊥⟩ : Tile .real [BM, BN])
        (Tile.expandDim ⟨1, by simp⟩ (⟨fun i => center score keep old i.1⟩ : Tile .real [BM]))) =
      finite (fun i : TileIndex [BM, BN] => probability score keep old i.1 i.2.1) := by
  apply Tile.ext
  rintro ⟨row, key, ⟨⟩⟩
  change WithBot.realExp2 (WithBot.realSub
    (if keep row key then (score row key : WithBot ℝ) else ⊥) (center score keep old row)) = _
  rw [exp2_sub_eq_weight]
  cases hk : keep row key <;> simp [finite, probability, hk]

/-- The explicit second mask leaves these probabilities unchanged. -/
theorem probability_remask {BM BN : Nat} (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (row : Fin BM) (key : Fin BN) :
    (if keep row key then probability score keep old row key else 0) =
      probability score keep old row key := by
  cases hk : keep row key <;> simp [probability, hk]

theorem center_matches_summary {BM BN : Nat} (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (row : Fin BM) (l a : ℝ) (value : Fin BN → ℝ) :
    center score keep old row =
      (RealFormula.summarize ⟨old row, l, a⟩ (pairs (score row) value (keep row))).center := by
  simp [center, RealFormula.summarize, maxScore_pairs]

theorem sum_probabilities {BM BN : Nat} (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (row : Fin BM) (value : Fin BN → ℝ) :
    (∑ key : Fin BN, probability score keep old row key) =
      RealFormula.sumAt (pairs (score row) value (keep row)) (center score keep old row) (fun _ => 1) := by
  simp only [RealFormula.sumAt, sum_pairs, mul_one, probability]

theorem sum_weighted_values {BM BN : Nat} (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (row : Fin BM) (value : Fin BN → ℝ) :
    (∑ key : Fin BN, probability score keep old row key * value key) =
      RealFormula.sumAt (pairs (score row) value (keep row)) (center score keep old row) Prod.snd := by
  simp only [RealFormula.sumAt, sum_pairs]
  apply Finset.sum_congr rfl
  intro key _
  cases hk : keep row key <;> simp [probability, hk]

/-- The actual row sum and rescaling arithmetic gives the denominator of
the independent row formula, including entirely masked rows. -/
theorem denominator_matches (BM BN : Nat) (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ) (l : Fin BM → ℝ)
    (row : Fin BM) (value : Fin BN → ℝ) (a : ℝ) :
    (Tile.bop NumericDType.real.add (.consSame .nil)
      (Tile.bop NumericDType.real.mul (.consSame .nil)
        (finite (fun i : TileIndex [BM] => l i.1))
        (finite (fun i : TileIndex [BM] => RealFormula.weight (old i.1) (center score keep old i.1))))
      (Tile.reduceSumDrop ⟨1, by simp⟩
        (finite (fun i : TileIndex [BM, BN] => probability score keep old i.1 i.2.1)))).data
          (row, PUnit.unit) =
      some (RealFormula.summarize ⟨old row, l row, a⟩ (pairs (score row) value (keep row))).denominator := by
  have hsum : (Tile.reduceSumDrop ⟨1, by simp⟩
      (finite (fun i : TileIndex [BM, BN] => probability score keep old i.1 i.2.1))).data
        (row, PUnit.unit) = (↑(∑ key : Fin BN, probability score keep old row key) : WithBot ℝ) := by
    change (∑ key : Fin BN, (↑(probability score keep old row key) : WithBot ℝ)) = _
    exact (WithBot.coe_sum _ _).symm
  change WithBot.realAdd (WithBot.realMul (↑(l row))
    (↑(RealFormula.weight (old row) (center score keep old row)))) _ = _
  simp only [Broadcast.rightIndex]
  erw [hsum]
  rw [WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe, sum_probabilities _ _ _ _ value]
  rw [center_matches_summary score keep old row (l row) a value]
  rfl

/-- The actual P/V dot and rescaled carry gives the numerator of the same
formula for each output lane. No associativity assumption enters the FP layer. -/
theorem numerator_matches (BM BN BD : Nat) (score : Fin BM → Fin BN → ℝ)
    (keep : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (acc : TileIndex [BM, BD] → ℝ) (value : TileIndex [BN, BD] → ℝ)
    (row : Fin BM) (lane : Fin BD) (l : ℝ) :
    (Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) (finite acc)
        (Tile.expandDim ⟨1, by simp⟩
          (finite (fun i : TileIndex [BM] => RealFormula.weight (old i.1) (center score keep old i.1)))))
      (Tile.dot [] (finite (fun i : TileIndex [BM, BN] => probability score keep old i.1 i.2.1))
        (finite value))).data (row, lane, PUnit.unit) =
      some (RealFormula.summarize ⟨old row, l, acc (row, lane, PUnit.unit)⟩
        (pairs (score row) (fun key => value (key, lane, PUnit.unit)) (keep row))).numerator := by
  rw [dot_finite]
  change WithBot.realAdd (WithBot.realMul (↑(acc (row, lane, PUnit.unit)))
    (↑(RealFormula.weight (old row) (center score keep old row))))
    (↑(∑ key : Fin BN, probability score keep old row key * value (key, lane, PUnit.unit))) = _
  rw [WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe,
    sum_weighted_values score keep old row (fun key => value (key, lane, PUnit.unit)),
    center_matches_summary score keep old row l (acc (row, lane, PUnit.unit))
      (fun key => value (key, lane, PUnit.unit))]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles
