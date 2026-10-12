import VeriTile.Triton.Float.Structural

/-! Reachable loop states for contextual FP rewrites. Index arithmetic is
discrete. No arithmetic law, loop invariant or body equivalence is assumed
globally: a caller must supply the local proof at every reachable iteration. -/
namespace VeriTile.Triton.FP.Structural

/-- States before binding the next loop index. An edge exists only after a
successful iteration of the original body. Failed and unexecuted iterations
therefore cannot supply premises for later numerical rewrites. -/
inductive RangeReachable {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (stop stride : Nat) (body : List ComputeStmt)
    (start : Nat) (initial : State α) : Nat → State α → Prop where
  | initial : RangeReachable M idx stop stride body start initial start initial
  | next {cur : Nat} {s t : State α} :
      RangeReachable M idx stop stride body start initial cur s →
      stride ≠ 0 → cur < stop →
      run M body (s.setReg idx .nat [] (fun _ => cur)) = some t →
      RangeReachable M idx stop stride body start initial (cur + stride) t

/-- Lift a local rewrite through a complete range loop, preserving the whole
state and execution failure. The unchanged bounds are evaluated outside this
lemma, once on loop entry. -/
theorem range_congr_on_reachable {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop stride : Nat) (lhs rhs : List ComputeStmt)
    (initial : State α)
    (hbody : ∀ cur s, RangeReachable M idx stop stride lhs start initial cur s →
      stride ≠ 0 → cur < stop →
      run M lhs (s.setReg idx .nat [] (fun _ => cur)) =
        run M rhs (s.setReg idx .nat [] (fun _ => cur))) :
    range M idx start stop stride lhs initial =
      range M idx start stop stride rhs initial := by
  have go : ∀ cur s, RangeReachable M idx stop stride lhs start initial cur s →
      range M idx cur stop stride lhs s = range M idx cur stop stride rhs s := by
    intro cur
    induction cur using (measure (fun i => stop - i)).wf.induction with
    | h cur ih =>
      intro s hr
      rw [range, range]
      by_cases hz : stride = 0
      · simp [hz]
      simp only [hz, if_false]
      by_cases hc : cur < stop
      · simp only [hc, if_true]
        rw [← hbody cur s hr hz hc]
        cases he : run M lhs (s.setReg idx .nat [] (fun _ => cur)) with
        | none => rfl
        | some t =>
          simp only [bind, Option.bind_some]
          apply ih (cur + stride) (by simp_wf; omega) t
          exact .next hr hz hc he
      · simp [hc]
  exact go start initial .initial

/-- A reached signed index, before binding the next loop iteration. -/
inductive IntRangeReachable {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (stop : Int) (stride : Nat) (body : List ComputeStmt)
    (start : Int) (initial : State α) : Int → State α → Prop where
  | initial : IntRangeReachable M idx stop stride body start initial start initial
  | next {cur : Int} {s t : State α} :
      IntRangeReachable M idx stop stride body start initial cur s →
      stride ≠ 0 → cur < stop →
      run M body (s.setReg idx .int [] (fun _ => cur)) = some t →
      IntRangeReachable M idx stop stride body start initial (cur + stride) t

theorem rangeInt_congr_on_reachable {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop : Int) (stride : Nat) (lhs rhs : List ComputeStmt)
    (initial : State α)
    (hbody : ∀ cur s, IntRangeReachable M idx stop stride lhs start initial cur s →
      stride ≠ 0 → cur < stop →
      run M lhs (s.setReg idx .int [] (fun _ => cur)) =
        run M rhs (s.setReg idx .int [] (fun _ => cur))) :
    rangeInt M idx start stop stride lhs initial =
      rangeInt M idx start stop stride rhs initial := by
  have go : ∀ cur s, IntRangeReachable M idx stop stride lhs start initial cur s →
      rangeInt M idx cur stop stride lhs s = rangeInt M idx cur stop stride rhs s := by
    intro cur
    induction cur using (measure (fun i : Int => (stop - i).toNat)).wf.induction with
    | h cur ih =>
      intro s hr
      rw [rangeInt, rangeInt]
      by_cases hz : stride = 0
      · simp [hz]
      simp only [hz, if_false]
      by_cases hc : cur < stop
      · simp only [hc, if_true]
        rw [← hbody cur s hr hz hc]
        cases he : run M lhs (s.setReg idx .int [] (fun _ => cur)) with
        | none => rfl
        | some t =>
          simp only [bind, Option.bind_some]
          apply ih (cur + stride) (by simp_wf; omega) t
          exact .next hr hz hc he
      · simp [hc]
  exact go start initial .initial


end VeriTile.Triton.FP.Structural
