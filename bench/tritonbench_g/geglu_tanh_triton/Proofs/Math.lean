import VeriTile.Triton.Math.Activation
import Mathlib.Tactic.Ring
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math

/-- Exact decimal coefficient written in the Python kernel. -/
def scale : ℝ := 0.7978845608028654

/-- The source's cubic approximation argument, independently of execution. -/
noncomputable def argument (a : ℝ) : ℝ := scale * (a + 0.044715 * a ^ 3)

/-- Scalar gate evaluated by the forward kernel and recomputed in backward. -/
noncomputable def gate (a : ℝ) : ℝ := a / 2 * (1 + Real.tanh (argument a))

/-- Independent forward output formula. -/
noncomputable def forward (a b : ℝ) : ℝ := gate a * b

/-- Source backward gate factor, using the exact decimal approximation coefficient. -/
noncomputable def gateFactor (a : ℝ) : ℝ :=
  (1 + Real.tanh (argument a)) / 2 +
    a / 2 * (1 - Real.tanh (argument a) ^ 2) *
      (scale * (1 + 3 * 0.044715 * a ^ 2))

/-- Both output channels refer to the input values before either in-place store. -/
noncomputable def backward (dc a b : ℝ) : ℝ × ℝ :=
  (dc * b * gateFactor a, dc * gate a)

open VeriTile.Triton.TiledActivation

theorem argument_eq (a : ℝ) : argument a = geluTanhArg a := by
  unfold argument scale geluTanhArg
  ring

theorem gate_eq (a : ℝ) : gate a = geluTanhCore a := by
  rw [gate, argument_eq]
  unfold geluTanhCore
  ring

theorem forward_eq (a b : ℝ) : forward a b = geluTanhFwd a b := by
  simp only [forward, gate_eq, geluTanhFwd]

theorem backward_eq (dc a b : ℝ) :
    backward dc a b = (geluTanhBwdA dc a b, geluTanhBwdB dc a) := by
  apply Prod.ext
  · simp only [backward, gateFactor, argument_eq, geluTanhBwdA, scale]
    norm_num only [OfScientific.ofScientific] <;> ring
  · simp only [backward, geluTanhBwdB, gate_eq]

@[simp] theorem gate_zero : gate 0 = 0 := by
  simp [gate, argument]

@[simp] theorem forward_zero (b : ℝ) : forward 0 b = 0 := by
  simp [forward]

/-- At zero gate input the first gradient is nonzero for nonzero upstream and
partner values, while the second gradient vanishes. -/
theorem backward_zero (dc b : ℝ) : backward dc 0 b = (dc * b / 2, 0) := by
  simp [backward, gateFactor, argument]
  ring

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math
