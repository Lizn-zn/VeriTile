import bench.tritonbench_g.softmax_flaggems.Proofs.RealMath

/-! The per-column lane model instantiated by the forward source execution proof. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardModel
open VeriTile Triton RealMath
open scoped BigOperators

noncomputable def values (N B lane c : Nat) (x : Nat → ℝ) : List ℝ :=
  (List.range c).filterMap fun t ↦ if t * B + lane < N then some (x (t * B + lane)) else none

noncomputable def state (N B lane c : Nat) (x : Nat → ℝ) : WithBot ℝ × ℝ :=
  (values N B lane c x).foldl onlineStep (⊥, 0)

theorem values_mass (N B lane c : Nat) (x : Nat → ℝ) :
    ((values N B lane c x).map Real.exp).sum =
      ∑ t ∈ Finset.range c, if t * B + lane < N then Real.exp (x (t * B + lane)) else 0 := by
  induction c with
  | zero => simp [values]
  | succ c ih =>
    simp only [values, List.range_succ, List.filterMap_append, List.filterMap_cons,
      List.filterMap_nil, List.map_append, List.sum_append, Finset.sum_range_succ] at *
    rw [ih]
    by_cases h : c * B + lane < N <;> simp [h]

theorem state_consistent (N B lane c : Nat) (x : Nat → ℝ) :
    (state N B lane c x).2 =
      (state N B lane c x).1.elim 0 (fun m ↦ Real.exp (-m)) *
        ∑ t ∈ Finset.range c, if t * B + lane < N then Real.exp (x (t * B + lane)) else 0 := by
  have h := online_denominator_invariant (values N B lane c x) ⊥ 0 0
    (by simp) (fun _ ↦ rfl)
  simpa only [state, zero_add, values_mass] using h.1

theorem state_bot_mass (N B lane c : Nat) (x : Nat → ℝ)
    (hbot : (state N B lane c x).1 = ⊥) :
    (∑ t ∈ Finset.range c, if t * B + lane < N then Real.exp (x (t * B + lane)) else 0) = 0 := by
  have h := online_denominator_invariant (values N B lane c x) ⊥ 0 0
    (by simp) (fun _ ↦ rfl)
  simpa only [zero_add, values_mass] using h.2 hbot

theorem state_succ (N B lane c : Nat) (x : Nat → ℝ) :
    state N B lane (c + 1) x = maskedOnlineStep (state N B lane c x)
      (if c * B + lane < N then some (x (c * B + lane)) else ⊥) := by
  have hs : state N B lane (c + 1) x =
      if c * B + lane < N then onlineStep (state N B lane c x) (x (c * B + lane))
      else state N B lane c x := by
    unfold state values
    rw [List.range_succ, List.filterMap_append, List.foldl_append]
    by_cases h : c * B + lane < N <;> simp [h]
  rw [hs]
  by_cases h : c * B + lane < N
  · simp only [if_pos h]
    exact (maskedOnlineStep_some _ _ _).symm
  · simp only [if_neg h]
    apply Eq.symm
    apply maskedOnlineStep_none
    intro hbot
    change (state N B lane c x).2 = 0
    have hb : (state N B lane c x).1 = ⊥ := hbot
    rw [state_consistent, hb]
    simp [Option.elim]

/-- A visited finite input prevents the online maximum from remaining negative infinity. -/
theorem state_zero_lane_nonbot (N B c : Nat) (x : Nat → ℝ)
    (hN : 0 < N) (hc : 0 < c) : (state N B 0 c x).1 ≠ ⊥ := by
  have hin : x 0 ∈ values N B 0 c x := by
    unfold values
    rw [List.mem_filterMap]
    exact ⟨0, by simp [hc], by simp [hN]⟩
  have hge : ∀ (ys : List ℝ) (a : WithBot ℝ),
      a ≤ ys.foldl (fun acc (y : ℝ) ↦ acc ⊔ (y : WithBot ℝ)) a := by
    intro ys
    induction ys with
    | nil => intro a; exact le_refl a
    | cons z zs ih => intro a; exact le_trans le_sup_left (ih _)
  unfold state
  rw [online_maximum]
  obtain ⟨left, right, heq⟩ := List.append_of_mem hin
  rw [heq, List.foldl_append, List.foldl_cons]
  intro hbot
  have h := le_trans (show (x 0 : WithBot ℝ) ≤
      left.foldl (fun acc (y : ℝ) ↦ acc ⊔ (y : WithBot ℝ)) ⊥ ⊔ (x 0 : WithBot ℝ)
        from le_sup_right) (hge right _)
  rw [hbot] at h
  exact (WithBot.bot_lt_coe (x 0)).not_ge h

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardModel
