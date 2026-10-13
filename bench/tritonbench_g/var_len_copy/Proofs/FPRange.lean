import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof
open VeriTile.Triton FP.Structural

/-- Whole signed-loop transport uses body equality at every reached state,
including evaluation failure and the exact signed index. -/
theorem rangeInt_congr {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (stop : Int) (stride : Nat) (a b : List ComputeStmt)
    (hbody : ∀ s, run M a s = run M b s) :
    ∀ cur s, rangeInt M idx cur stop stride a s = rangeInt M idx cur stop stride b s := by
  intro cur s
  induction h : (stop - cur).toNat using Nat.strong_induction_on generalizing cur s with
  | _ remaining ih =>
    by_cases hz : stride = 0
    · rw [rangeInt, rangeInt, if_pos hz, if_pos hz]
    · by_cases hc : cur < stop
      · rw [rangeInt, rangeInt, if_neg hz, if_neg hz, if_pos hc, if_pos hc, hbody]
        cases he : run M b (s.setReg idx .int [] (fun _ ↦ cur)) with
        | none => rfl
        | some next =>
          exact ih (stop - (cur + stride)).toNat (by omega) (cur + stride) next rfl
      · rw [rangeInt, rangeInt, if_neg hz, if_neg hz, if_neg hc, if_neg hc]

end VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof

#guard_msgs (drop info) in
#auditModuleAxioms
