import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.MaxReduction.DimensionStore
open VeriTile.Triton BlockState
private theorem foldl_writeMemTyped_int_masked_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → Int) (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
      BlockState.readMemValue ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
          .int region o
      = s.readMemValue .int region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    rw [List.foldl_cons]
    have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
      fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
    by_cases hmaskhd : mask hd = Bool.true
    · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
      simp only [hmaskhd, if_true]
      rw [ih _ htl]
      rw [writeMemTyped_int_readMemValue_int]
      show (if region = region ∧ o = offsetFn hd then valueFn hd
          else s.readMemValue TileDType.int region o) =
        s.readMemValue TileDType.int region o
      rw [if_neg]
      rintro ⟨_, h_eq⟩
      exact hhd h_eq.symm
    · have hmaskhd' : mask hd = Bool.false := by
        rcases hmaskFalse : mask hd
        · rfl
        · exact absurd hmaskFalse hmaskhd
      simp only [hmaskhd', if_false, Bool.false_eq_true]
      exact ih _ htl

theorem scatter_readback_int_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → Int) (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    BlockState.readMemValue ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s)
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change BlockState.readMemValue ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_int_masked_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    rw [writeMemTyped_int_readMemValue_int]
    show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _) = valueFn i
    exact if_pos ⟨rfl, rfl⟩
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_int_masked_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

theorem foldl_writeMemTyped_intAt_prop_masked_readMem_other_region {α : Type}
    (regionFn : α → RegionName) (offsetFn : α → Nat) (valueFn : α → Int)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (R : RegionName) (off : Nat)
    (h_other : ∀ k, k ∈ l → P k → R ≠ regionFn k) :
    BlockState.readMem ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .int (regionFn k) (offsetFn k) (valueFn k) else acc) s))
      R off
      = s.readMem R off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      have htl : ∀ k, k ∈ tl → P k → R ≠ regionFn k :=
        fun k hk hPk => h_other k (List.mem_cons_of_mem hd hk) hPk
      by_cases hhd : P hd
      · simp only [hhd, if_true]
        rw [ih _ htl]
        unfold readMem
        change (match
            (if R = regionFn hd ∧ off = offsetFn hd then
              MemCell.of TileDType.int (valueFn hd)
            else s.mem R off).readAs TileDType.real with
          | some (some value) => value
          | _ => 0) =
          match (s.mem R off).readAs TileDType.real with
          | some (some value) => value
          | _ => 0
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact (h_other hd (List.mem_cons_self) hhd) hR
      · simp only [hhd, if_false]
        exact ih _ htl


/-- A masked signed-index scatter preserves every unhit memory cell. -/
theorem int_frame {α : Type} (region : RegionName) (offset : α → Nat)
    (value : α → Int) (active : α → Prop) [DecidablePred active]
    (l : List α) (s : BlockState) (r : RegionName) (o : Nat)
    (h : ∀ k ∈ l, active k → r ≠ region ∨ o ≠ offset k) :
    (l.foldl (fun acc k => if active k then acc.writeMemTyped .int region (offset k) (value k)
      else acc) s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons k rest ih =>
    rw [List.foldl_cons]
    have ht : ∀ j ∈ rest, active j → r ≠ region ∨ o ≠ offset j :=
      fun j hj ha => h j (List.mem_cons_of_mem k hj) ha
    rw [ih _ ht]
    by_cases hk : active k
    · simp only [hk, if_pos]
      unfold BlockState.writeMemTyped
      simp only
      rw [if_neg]
      rintro ⟨hr, ho⟩
      rcases h k (List.mem_cons_self) hk with hr' | ho'
      · exact hr' hr
      · exact ho' ho
    · simp [hk]

/-- A masked floating-value scatter preserves every unhit memory cell. -/
theorem real_frame {α : Type} (region : RegionName) (offset : α → Nat)
    (value : α → ℝ) (active : α → Prop) [DecidablePred active]
    (l : List α) (s : BlockState) (r : RegionName) (o : Nat)
    (h : ∀ k ∈ l, active k → r ≠ region ∨ o ≠ offset k) :
    (l.foldl (fun acc k => if active k then acc.writeMem region (offset k) (value k)
      else acc) s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons k rest ih =>
    rw [List.foldl_cons]
    have ht : ∀ j ∈ rest, active j → r ≠ region ∨ o ≠ offset j :=
      fun j hj ha => h j (List.mem_cons_of_mem k hj) ha
    rw [ih _ ht]
    by_cases hk : active k
    · simp only [hk, if_pos]
      rw [BlockState.writeMem_mem, if_neg]
      rintro ⟨hr, ho⟩
      rcases h k (List.mem_cons_self) hk with hr' | ho'
      · exact hr' hr
      · exact ho' ho
    · simp [hk]

end VeriTile.Bench.TritonBenchG.MaxReduction.DimensionStore
