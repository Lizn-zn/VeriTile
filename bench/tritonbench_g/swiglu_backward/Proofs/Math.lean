import VeriTile.Triton.Math.Activation
import Mathlib.Tactic.Ring
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Math

/-- Independent scalar sigmoid, with a strictly positive real denominator. -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- Gate gradient from the original gate, value and upstream windows. -/
noncomputable def dx (x y d : ℝ) : ℝ :=
  d * y * sigmoid x * (1 + x * (1 - sigmoid x))

/-- Value gradient from the original gate and upstream windows. -/
noncomputable def dy (x d : ℝ) : ℝ := d * x * sigmoid x

/-- Optional recomputation of the forward value, using original inputs. -/
noncomputable def forward (x y : ℝ) : ℝ := x * sigmoid x * y

theorem denominator_pos (x : ℝ) : 0 < 1 + Real.exp (-x) := by positivity

theorem sigmoid_eq (x : ℝ) : sigmoid x = Real.sigmoid x := by
  simp [sigmoid, Real.sigmoid_def]

theorem dx_eq (x y d : ℝ) : dx x y d = VeriTile.Triton.TiledActivation.swigluBwdA d x y := by
  simp only [dx, sigmoid_eq, VeriTile.Triton.TiledActivation.swigluBwdA,
    VeriTile.Triton.TiledActivation.silu]
  ring

theorem dy_eq (x d : ℝ) : dy x d = VeriTile.Triton.TiledActivation.swigluBwdB d x := by
  simp only [dy, sigmoid_eq, VeriTile.Triton.TiledActivation.swigluBwdB,
    VeriTile.Triton.TiledActivation.silu]
  ring

theorem forward_eq (x y : ℝ) : forward x y = VeriTile.Triton.TiledActivation.swiglu x y := by
  simp [forward, sigmoid_eq, VeriTile.Triton.TiledActivation.swiglu,
    VeriTile.Triton.TiledActivation.silu]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward.Math
