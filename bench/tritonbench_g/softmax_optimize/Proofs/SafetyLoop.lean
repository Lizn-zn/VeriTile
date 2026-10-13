import VeriTile.Triton.Memory.Flatten
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton

/-- An invariant proves access safety for every iteration of the signed source
loop. Failed body execution terminates the trace without an extra assumption. -/
theorem signed_loop_safe (bounds : RegionBounds) (index : RegName)
    (stop step : Nat) (body : List Stmt) (Inv : Nat → BlockState → Prop)
    (hstep : 0 < step)
    (hsafe : ∀ c s, c < stop → Inv c s →
      Stmt.TraceSafeList bounds body (s.setReg index .int [] (Tile.scalar (c : Int))))
    (hnext : ∀ c s t, c < stop → Inv c s →
      stepStmts body (s.setReg index .int [] (Tile.scalar (c : Int))) = some t →
      Inv (c + step) t) :
    ∀ c s, Inv c s →
      Stmt.forRangeIntTraceSafe bounds index (c : Int) (stop : Int) step body s := by
  intro c s
  induction h : stop - c using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hinv
    by_cases hc : c < stop
    · have hci : (c : Int) < (stop : Int) := by exact_mod_cast hc
      rw [Stmt.forRangeIntTraceSafe, if_neg (Nat.ne_of_gt hstep), if_pos hci]
      refine ⟨hsafe c s hc hinv, ?_⟩
      cases he : stepStmts body (s.setReg index .int [] (Tile.scalar (c : Int))) with
      | none => trivial
      | some t =>
        have ht := ih (stop - (c + step)) (by omega) (c + step) t rfl
          (hnext c s t hc hinv he)
        simpa only [Nat.cast_add] using ht
    · have hci : ¬ (c : Int) < (stop : Int) := by exact_mod_cast hc
      rw [Stmt.forRangeIntTraceSafe, if_neg (Nat.ne_of_gt hstep), if_neg hci]
      trivial

/-- State facts used by later loops survive every successful iteration. -/
theorem signed_loop_preserves (index : RegName) (stop step : Nat)
    (body : List Stmt) (P : BlockState → Prop)
    (hbody : ∀ c s t, c < stop → P s →
      stepStmts body (s.setReg index .int [] (Tile.scalar (c : Int))) = some t → P t) :
    ∀ (c : Nat) (s t : BlockState), P s →
      stepForRangeIntAux index (c : Int) (stop : Int) step body s = some t → P t := by
  intro c s t
  induction h : stop - c using Nat.strong_induction_on generalizing c s t with
  | _ remaining ih =>
    intro hs hexec
    by_cases hz : step = 0
    · subst step
      rw [stepForRangeIntAux.step_zero_step] at hexec
      cases hexec
      exact hs
    by_cases hc : c < stop
    · rw [stepForRangeIntAux.step_lt hz (by omega)] at hexec
      cases he : stepStmts body (s.setReg index .int [] (Tile.scalar (c : Int))) with
      | none => simp [he] at hexec
      | some next =>
        have hn := hbody c s next hc hs he
        simp only [he, Option.bind_some, ← Int.natCast_add] at hexec
        exact ih (stop - (c + step)) (by omega) (c + step) next t rfl hn hexec
    · rw [stepForRangeIntAux.step_ge hz (by omega)] at hexec
      cases hexec
      exact hs

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
