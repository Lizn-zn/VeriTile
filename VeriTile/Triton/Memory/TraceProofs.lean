import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.KernelLemmas

/-! Small compositional lemmas for address safety. Numerical register values
may remain abstract while address registers and loop invariants are tracked. -/
namespace VeriTile.Triton.TraceProofs

theorem assign_inv {d sh name} {e : Op d sh} {s t : BlockState}
    (h : stepStmt (.assign d sh name e) s = some t) :
    ∃ v, evalOp e s = some v ∧ t = s.setReg name d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => simp [hv] at h
  | some v => exact ⟨v, rfl, (Option.some.inj (by simpa [hv] using h)).symm⟩

theorem cons_inv {st rest s t} (h : stepStmts (st :: rest) s = some t) :
    ∃ u, stepStmt st s = some u ∧ stepStmts rest u = some t := by
  rw [stepStmts] at h
  cases he : stepStmt st s with
  | none => simp [he] at h
  | some u => exact ⟨u, rfl, by simpa [he] using h⟩

/-- Thread safety through a known step without changing its successor. -/
theorem cons_of_run_iff {bounds : RegionBounds} {st : Stmt} {rest : List Stmt}
    {s t : BlockState} (h : stepStmt st s = some t) :
    Stmt.TraceSafeList bounds (st :: rest) s ↔
      st.TraceSafe bounds s ∧ Stmt.TraceSafeList bounds rest t := by
  rw [Stmt.TraceSafeList, h]

/-- State-independent safety of every statement suffices for its list.
Failure is permitted and does not create an unexecuted successor obligation. -/
theorem all_safe {bounds : RegionBounds} :
    ∀ (body : List Stmt) (s : BlockState),
      (∀ st ∈ body, ∀ t, Stmt.TraceSafe bounds st t) → Stmt.TraceSafeList bounds body s
  | [], _, _ => Stmt.TraceSafeList.nil_intro
  | st :: rest, s, h => by
      apply Stmt.TraceSafeList.cons_intro (h st List.mem_cons_self s)
      intro t _
      exact all_safe rest t (fun st hs => h st (List.mem_cons_of_mem _ hs))

/-- Compose safety along only the successors of the actual prefix execution. -/
theorem append_safe {bounds : RegionBounds} :
    ∀ (first : List Stmt) {rest : List Stmt} (s : BlockState),
      Stmt.TraceSafeList bounds first s →
      (∀ t, stepStmts first s = some t → Stmt.TraceSafeList bounds rest t) →
      Stmt.TraceSafeList bounds (first ++ rest) s
  | [], _, s, _, h => h s stepStmts.nil
  | st :: first, rest, s, h, hn => by
      rw [Stmt.TraceSafeList] at h
      refine Stmt.TraceSafeList.cons_intro h.1 (fun t ht => ?_)
      have hrest := h.2
      rw [ht] at hrest
      exact append_safe first t hrest
        (fun u hu => hn u ((stepStmts.cons_some ht).trans hu))

theorem assign_cons {d sh name} {e : Op d sh} {s : BlockState} {bounds rest}
    (hsafe : e.SafeAt bounds s)
    (htail : ∀ v, Stmt.TraceSafeList bounds rest (s.setReg name d sh v)) :
    Stmt.TraceSafeList bounds (.assign d sh name e :: rest) s := by
  refine Stmt.TraceSafeList.cons_intro (by simpa only [Stmt.TraceSafe] using hsafe) ?_
  intro t ht
  obtain ⟨v, _, rfl⟩ := assign_inv ht
  exact htail v

theorem assign_eval_cons {d sh name} {e : Op d sh} {s : BlockState} {bounds rest}
    (hsafe : e.SafeAt bounds s)
    (htail : ∀ v, evalOp e s = some v → Stmt.TraceSafeList bounds rest (s.setReg name d sh v)) :
    Stmt.TraceSafeList bounds (.assign d sh name e :: rest) s := by
  refine Stmt.TraceSafeList.cons_intro (by simpa only [Stmt.TraceSafe] using hsafe) ?_
  intro t ht
  obtain ⟨v, hv, rfl⟩ := assign_inv ht
  exact htail v hv

/-- An invariant supplies bounds at every reached dynamic-range iteration.
Only successful body runs need to establish its next instance. -/
theorem range_safe {bounds : RegionBounds} {index : RegName} {stop stride : Nat}
    {body : List Stmt} {P : Nat → BlockState → Prop}
    (hsafe : ∀ i s, i < stop → P i s →
      Stmt.TraceSafeList bounds body (s.setReg index .nat [] (Tile.scalar i)))
    (hnext : ∀ i s t, i < stop → P i s →
      stepStmts body (s.setReg index .nat [] (Tile.scalar i)) = some t → P (i + stride) t)
    (cur : Nat) (s : BlockState) (hp : P cur s) :
    Stmt.forRangeTraceSafe bounds index cur stop stride body s := by
  rw [Stmt.forRangeTraceSafe]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hc : cur < stop
    · simp only [hc, ↓reduceIte]
      refine ⟨hsafe cur s hc hp, ?_⟩
      cases hb : stepStmts body (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => trivial
      | some t => exact range_safe hsafe hnext (cur + stride) t (hnext cur s t hc hp hb)
    · simp only [hc, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

/-- Safety follows an invariant at every reached iteration, including loops
whose numerical operations may fail. -/
theorem loop_safe {bounds : RegionBounds} {idx : RegName} {n : Nat} {body : List Stmt} {P : BlockState → Prop}
    (hsafe : ∀ i s, i < n → P s →
      Stmt.TraceSafeList bounds body (s.setReg idx .nat [] (Tile.scalar i)))
    (hstep : ∀ i s t, i < n → P s →
      stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some t → P t) :
    ∀ (fuel start : Nat) (s : BlockState), n - start ≤ fuel → P s →
      Stmt.forLoopTraceSafe bounds idx start n body s
  | 0, start, s, hf, _ => by rw [Stmt.forLoopTraceSafe, if_neg (by omega)]; trivial
  | fuel + 1, start, s, hf, hp => by
    rw [Stmt.forLoopTraceSafe]
    split
    next hlt =>
      refine ⟨hsafe start s hlt hp, ?_⟩
      split
      next t ht =>
        exact loop_safe hsafe hstep fuel (start + 1) t (by omega)
          (hstep start s t hlt hp ht)
      next => trivial
    next => trivial

theorem loop_preserves {idx : RegName} {n : Nat} {body : List Stmt} {P : BlockState → Prop}
    (hstep : ∀ i s t, i < n → P s →
      stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some t → P t) :
    ∀ (fuel start : Nat) (s t : BlockState), n - start ≤ fuel → P s →
      stepForLoopAux idx start n body s = some t → P t
  | 0, start, s, t, hf, hp, ht => by
    rw [stepForLoopAux.step_ge (by omega)] at ht
    cases ht
    exact hp
  | fuel + 1, start, s, t, hf, hp, ht => by
    by_cases hlt : start < n
    · rw [stepForLoopAux.step_lt hlt] at ht
      cases hb : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
      | none => simp [hb] at ht
      | some u =>
        exact loop_preserves hstep fuel (start + 1) u t (by omega)
          (hstep start s u hlt hp hb) (by simpa [hb] using ht)
    · rw [stepForLoopAux.step_ge (by omega)] at ht
      cases ht
      exact hp

theorem matrix_offset_lt (base A B : Nat) (i : Fin A) (j : Fin B) :
    base + i.val * B + j.val < base + A * B := by
  have h := Nat.mul_le_mul_right B i.isLt
  have hj := j.isLt
  rw [Nat.succ_mul] at h
  omega

theorem region_safe {d sh} (r : Region d) (off : Op .nat sh)
    (bounds : RegionBounds) (s : BlockState) (offsets : Tile .nat sh)
    (he : evalOp off s = some offsets)
    (hb : ∀ i, offsets.data i < bounds (Region.cast r)) (active) :
    (MemAccess.region r off).ActiveAddressSafe bounds s active := by
  intro v hv i _
  rw [he] at hv
  cases hv
  exact hb i

theorem store_regs {d sh} (r : Region d) (off : Op .nat sh) (val : Op d sh)
    (s t : BlockState)
    (h : stepStmt (.store d sh (.region r off) val .none) s = some t) : t.regs = s.regs := by
  simp only [stepStmt] at h
  cases hv : evalOp val s with
  | none => simp [hv] at h
  | some values =>
    cases ho : evalOp off s with
    | none => simp [hv, ho] at h
    | some offsets =>
      simp only [hv, ho] at h
      cases h
      funext dtype shape name
      exact BlockState.foldl_writeMemTyped_regs d _ _ _ _ s dtype shape name

end VeriTile.Triton.TraceProofs
