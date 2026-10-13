import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

/-- Related successful states or simultaneous failure. -/
def ResultsRelated (Rel : BlockState → BlockState → Prop)
    (signed natural : Option BlockState) : Prop :=
  match signed, natural with
  | none, none => True
  | some s, some t => Rel s t
  | _, _ => False

/-- Transport a complete nonnegative range through a body simulation. The
signed and natural registers may differ; the supplied relation records exactly
which state components the concrete kernel needs to preserve. -/
theorem loop_related (Rel : BlockState → BlockState → Prop)
    (idx : RegName) (stop step : Nat) (signedBody naturalBody : List Stmt)
    (body_related : ∀ (i : Nat), i < stop → ∀ s t, Rel s t →
      ResultsRelated Rel
        (stepStmts signedBody (s.setReg idx .int [] (Tile.scalar (i : Int))))
        (stepStmts naturalBody (t.setReg idx .nat [] (Tile.scalar i))))
    (cur : Nat) (s t : BlockState) (hrel : Rel s t) :
    ResultsRelated Rel
      (stepForRangeIntAux idx (cur : Int) (stop : Int) step signedBody s)
      (stepForRangeAux idx cur stop step naturalBody t) := by
  generalize hn : stop - cur = n
  induction n using Nat.strong_induction_on generalizing cur s t with
  | h n ih =>
    by_cases hz : step = 0
    · subst step
      simpa only [stepForRangeIntAux.step_zero_step, stepForRangeAux.step_zero_step,
        ResultsRelated] using hrel
    by_cases hlt : cur < stop
    · have hltI : (cur : Int) < (stop : Int) := by omega
      rw [stepForRangeIntAux.step_lt hz hltI, stepForRangeAux.step_lt hz hlt]
      have hb := body_related cur hlt s t hrel
      cases hs : stepStmts signedBody (s.setReg idx .int [] (Tile.scalar (cur : Int))) with
      | none =>
        cases ht : stepStmts naturalBody (t.setReg idx .nat [] (Tile.scalar cur)) <;>
          simp_all [ResultsRelated, Option.bind]
      | some s' =>
        cases ht : stepStmts naturalBody (t.setReg idx .nat [] (Tile.scalar cur)) with
        | none => simp [hs, ht, ResultsRelated] at hb
        | some t' =>
          have hb' : Rel s' t' := by simpa [hs, ht, ResultsRelated] using hb
          simp only [Option.bind_some]
          rw [← Int.natCast_add]
          exact ih (stop - (cur + step)) (by omega) (cur + step) s' t' hb' rfl
    · have hge : stop ≤ cur := Nat.le_of_not_gt hlt
      have hgeI : (stop : Int) ≤ (cur : Int) := by omega
      simpa only [stepForRangeIntAux.step_ge hz hgeI, stepForRangeAux.step_ge hz hge,
        ResultsRelated] using hrel

/-- A successful natural execution yields a successful signed execution and
related final states; success is a consequence of the body simulation. -/
theorem successful_transport (Rel : BlockState → BlockState → Prop)
    (idx : RegName) (stop step : Nat) (signedBody naturalBody : List Stmt)
    (body_related : ∀ (i : Nat), i < stop → ∀ s t, Rel s t →
      ResultsRelated Rel
        (stepStmts signedBody (s.setReg idx .int [] (Tile.scalar (i : Int))))
        (stepStmts naturalBody (t.setReg idx .nat [] (Tile.scalar i))))
    (cur : Nat) (s t t' : BlockState) (hrel : Rel s t)
    (hexec : stepForRangeAux idx cur stop step naturalBody t = some t') :
    ∃ s', stepForRangeIntAux idx (cur : Int) (stop : Int) step signedBody s = some s' ∧
      Rel s' t' := by
  have h := loop_related Rel idx stop step signedBody naturalBody body_related cur s t hrel
  rw [hexec] at h
  cases hs : stepForRangeIntAux idx (cur : Int) (stop : Int) step signedBody s with
  | none => simp [hs, ResultsRelated] at h
  | some s' => exact ⟨s', rfl, by simpa [hs, ResultsRelated] using h⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
