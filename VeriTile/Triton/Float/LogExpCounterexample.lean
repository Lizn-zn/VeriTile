/- Exact-value counterexamples, separate from statistical FP equivalence.
PR #13's boundaries.json records log(exp(a)) = 0 at a = 2^-25 and at a = 0.
Transcendental evaluations are explicit premises: this module does not
implement libdevice or turn a GPU observation into a Lean axiom. All remaining
arithmetic and casts are kernel-checked in the concrete software profile. -/
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Triton.FP.LogExpCounterexample

def tiny : Value .fp32 := .ofBits .fp32 0x33000000
def zero : Value .fp32 := .zero .fp32

set_option maxRecDepth 4096 in
theorem tiny_value : tiny.toRat? = some (1 / 33554432) ∧ zero.toRat? = some 0 := by
  decide +kernel

set_option maxRecDepth 4096 in
theorem tiny_ne_zero : tiny ≠ zero := by decide +kernel

/-- Refute the universal exact identity using the reported local evaluation.
This says nothing about whether a two-gates rule should be admitted. -/
theorem plain_log_exp_not_identity (logExp : Value .fp32 → Value .fp32)
    (reported : logExp tiny = zero) : ¬ ∀ a, logExp a = a := by
  intro h
  exact tiny_ne_zero ((h tiny).symm.trans reported)

/-- For a singleton row, direct LSE is log(exp(a)); stable LSE is
a + log(exp(a-a)). The original examples store both results as bf16.
These definitions assume the singleton sum/max return their one operand. -/
def directSingleton (logExp : Value .fp32 → Value .fp32) (a : Value .fp32) : Value .bf16 :=
  cast .bf16 Config.ieeeValue (logExp a)

def stableSingleton (logExp : Value .fp32 → Value .fp32) (a : Value .fp32) : Value .bf16 :=
  cast .bf16 Config.ieeeValue (add Config.ieeeValue a (logExp (sub Config.ieeeValue a a)))

set_option maxRecDepth 4096 in
theorem tiny_sub_self : sub Config.ieeeValue tiny tiny = zero := by decide +kernel

set_option maxRecDepth 4096 in
theorem stored_values :
    (cast .bf16 Config.ieeeValue zero).toRat? = some 0 ∧
    (cast .bf16 Config.ieeeValue (add Config.ieeeValue tiny zero)).toRat? =
      some (1 / 33554432) := by decide +kernel

set_option maxRecDepth 4096 in
theorem stored_values_differ :
    cast .bf16 Config.ieeeValue zero ≠
      cast .bf16 Config.ieeeValue (add Config.ieeeValue tiny zero) := by decide +kernel

/-- A conditional counterexample to exact equality of the singleton formulas.
The two explicit premises are composed log-exp observations, not separate
claims about exp/log and not a measurement of the complete LSE kernels. -/
theorem singleton_lse_not_exact (logExp : Value .fp32 → Value .fp32)
    (atTiny : logExp tiny = zero) (atZero : logExp zero = zero) :
    directSingleton logExp tiny ≠ stableSingleton logExp tiny := by
  simpa only [directSingleton, stableSingleton, tiny_sub_self, atTiny, atZero]
    using stored_values_differ

end VeriTile.Triton.FP.LogExpCounterexample
