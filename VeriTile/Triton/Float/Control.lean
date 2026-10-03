/- Successful execution and invariant rules for FP control flow. These lemmas
retain opaque numerical operations and require the body to succeed at every
reached iteration. They do not supply any numerical identity. -/
import VeriTile.Triton.Float.Structural

namespace VeriTile.Triton.FP.Structural

variable {α : Type} [Inhabited α] (M : Algebra α)

theorem loop_done (idx : RegName) (start stop : Nat) (body : List ComputeStmt)
    (s : State α) (h : stop ≤ start) : loop M idx start stop body s = some s := by
  rw [loop]
  simp [Nat.not_lt.mpr h]

theorem loop_step (idx : RegName) (start stop : Nat) (body : List ComputeStmt)
    (s : State α) (h : start < stop) :
    loop M idx start stop body s =
      (run M body (s.setReg idx .nat [] (fun _ => start))).bind
        (loop M idx (start + 1) stop body) := by
  rw [loop]
  simp [h]

/-- Success and the final invariant follow from successful steps. No input
distribution or numerical fact is smuggled into the induction principle. -/
theorem loop_invariant {idx : RegName} {n : Nat} {body : List ComputeStmt}
    {P : Nat → State α → Prop}
    (hstep : ∀ i s, i < n → P i s →
      ∃ t, run M body (s.setReg idx .nat [] (fun _ => i)) = some t ∧ P (i + 1) t)
    (start : Nat) (hstart : start ≤ n) (s : State α) (hs : P start s) :
    ∃ t, loop M idx start n body s = some t ∧ P n t := by
  have key : ∀ k i, n - i = k → i ≤ n → ∀ s, P i s →
      ∃ t, loop M idx i n body s = some t ∧ P n t := by
    intro k
    induction k with
    | zero =>
      intro i hk hi s hs
      have heq : i = n := by omega
      subst i
      exact ⟨s, loop_done M idx n n body s (by omega), hs⟩
    | succ k ih =>
      intro i hk hi s hs
      have hlt : i < n := by omega
      obtain ⟨t, ht, hp⟩ := hstep i s hlt hs
      obtain ⟨u, hu, hpu⟩ := ih (i + 1) (by omega) (by omega) t hp
      refine ⟨u, ?_, hpu⟩
      rw [loop_step M idx i n body s hlt, ht]
      exact hu
  exact key (n - start) start rfl hstart s hs

theorem forLoop_invariant {idx : RegName} {n : Nat} {body : List ComputeStmt}
    {P : Nat → State α → Prop} {s : State α} (hs : P 0 s)
    (hstep : ∀ i s, i < n → P i s →
      ∃ t, run M body (s.setReg idx .nat [] (fun _ => i)) = some t ∧ P (i + 1) t) :
    ∃ t, step M (.forLoop idx n body) s = some t ∧ P n t := by
  simpa only [step] using loop_invariant M hstep 0 (Nat.zero_le _) s hs

theorem range_zero (idx : RegName) (cur stop : Nat) (body : List ComputeStmt)
    (s : State α) : range M idx cur stop 0 body s = some s := by
  rw [range]
  simp

theorem range_done (idx : RegName) (cur stop stride : Nat) (body : List ComputeStmt)
    (s : State α) (h : stop ≤ cur) : range M idx cur stop stride body s = some s := by
  rw [range]
  simp [Nat.not_lt.mpr h]

theorem range_step (idx : RegName) (cur stop stride : Nat) (body : List ComputeStmt)
    (s : State α) (hd : stride ≠ 0) (h : cur < stop) :
    range M idx cur stop stride body s =
      (run M body (s.setReg idx .nat [] (fun _ => cur))).bind
        (range M idx (cur + stride) stop stride body) := by
  rw [range]
  simp [hd, h]

/-- A strided range can overshoot its upper bound; expose the reached index
instead of falsely promising that the final invariant is indexed by `stop`. -/
theorem range_invariant {idx : RegName} {stop stride : Nat} {body : List ComputeStmt}
    {P : Nat → State α → Prop} (hd : stride ≠ 0)
    (hstep : ∀ i s, i < stop → P i s →
      ∃ t, run M body (s.setReg idx .nat [] (fun _ => i)) = some t ∧ P (i + stride) t)
    (cur : Nat) (s : State α) (hs : P cur s) :
    ∃ last t, range M idx cur stop stride body s = some t ∧ stop ≤ last ∧ P last t := by
  have key : ∀ k i s, stop - i = k → P i s →
      ∃ last t, range M idx i stop stride body s = some t ∧ stop ≤ last ∧ P last t := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
      intro i s hk hs
      by_cases hdone : stop ≤ i
      · exact ⟨i, s, range_done M idx i stop stride body s hdone, hdone, hs⟩
      · have hlt : i < stop := by omega
        obtain ⟨t, ht, hp⟩ := hstep i s hlt hs
        obtain ⟨last, u, hu, hlast, hpu⟩ := ih (stop - (i + stride)) (by omega)
          (i + stride) t rfl hp
        refine ⟨last, u, ?_, hlast, hpu⟩
        rw [range_step M idx i stop stride body s hd hlt, ht]
        exact hu
  exact key (stop - cur) cur s rfl hs

end VeriTile.Triton.FP.Structural
