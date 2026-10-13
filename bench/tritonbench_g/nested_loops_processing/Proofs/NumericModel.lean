import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.NumericExecution
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical

/-- Nonempty exact-real backend for copy/conversion execution. This witness
checks typed cells and actual loops, not target rounding behavior. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r ↦ some r
  negInf := none
  binary := fun _ _ op a b ↦ match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ _ x ↦ x
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

end
end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.NumericExecution

#guard_msgs (drop info) in
#auditModuleAxioms
