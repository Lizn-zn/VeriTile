import VeriTile.Triton.Math.Activation
import Mathlib.Tactic.Ring
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Math
open VeriTile.Triton

/-- Independent scalar formula, with no dependence on kernel execution or memory. -/
noncomputable def forward (x y : ℝ) : ℝ := x * y / (1 + Real.exp (-x))

theorem forward_eq (x y : ℝ) : forward x y = TiledActivation.swiglu x y := by
  simp only [forward, TiledActivation.swiglu, TiledActivation.silu, Real.sigmoid_def,
    div_eq_mul_inv]
  ring

theorem forward_zero (y : ℝ) : forward 0 y = 0 := by simp [forward]

theorem denominator_pos (x : ℝ) : 0 < 1 + Real.exp (-x) := by
  positivity

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd.Math
