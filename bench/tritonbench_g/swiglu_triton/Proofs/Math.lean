import VeriTile.Triton.Math.Activation
import Mathlib.Tactic.Ring
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SwigluTriton.Math
open VeriTile.Triton

/-- Independent scalar sigmoid formula. -/
noncomputable def gate (a : ℝ) : ℝ := (1 + Real.exp (-a))⁻¹

/-- Independent forward formula. -/
noncomputable def forward (a b : ℝ) : ℝ := a * b * gate a

/-- Independent pair of source backward expressions from the old inputs. -/
noncomputable def backward (dc a b : ℝ) : ℝ × ℝ :=
  (dc * b * gate a * (1 + a * (1 - gate a)), dc * a * gate a)

theorem gate_eq (a : ℝ) : gate a = Real.sigmoid a := rfl

theorem forward_eq (a b : ℝ) : forward a b = TiledActivation.swiglu a b := by
  simp only [forward, gate_eq, TiledActivation.swiglu, TiledActivation.silu]
  ring

theorem backward_eq (dc a b : ℝ) : backward dc a b =
    (TiledActivation.swigluBwdA dc a b, TiledActivation.swigluBwdB dc a) := by
  apply Prod.ext <;> simp only [backward, gate_eq, TiledActivation.swigluBwdA,
    TiledActivation.swigluBwdB, TiledActivation.silu] <;> ring

theorem forward_zero (b : ℝ) : forward 0 b = 0 := by simp [forward]

theorem backward_zero (dc b : ℝ) : backward dc 0 b = (dc * b / 2, 0) := by
  apply Prod.ext <;> norm_num [backward, gate] <;> ring

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton.Math
