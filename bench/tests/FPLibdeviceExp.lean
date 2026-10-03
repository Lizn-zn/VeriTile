/- Implementation identity and admitted libdevice exp-sub binding.
These rational fixtures are logical checks, not additional GPU evidence. -/
import bench.examples.SoftmaxStableFPEquiv
import Mathlib.Tactic.NormNum

namespace FPLibdeviceExpTests
open VeriTile Triton FP.Structural

def mixed : ComputeKernel := triton {
  e := libdevice.exp(1) + tl.exp(1)
}

def qualified : ComputeKernel := triton {
  e := tl.extra.cuda.libdevice.exp(1) + tl.exp(1)
}

theorem aliases_match : qualified = mixed := rfl

theorem nested_symbols_preserved : mixed.surfaceBody = [
    .assign .real [] "e" (.alg (.add .real .nil
      (.libdeviceExp (.const 1)) (.exp (.const 1))))] := rfl

def mixedFP32 : ComputeKernel := triton {
  a := tl.load($(("x" : RegionName)) + 0, dtype=tl.float32)
  e := libdevice.exp(a) + tl.exp(a)
}

theorem precision_preserved : mixedFP32.surfaceBody[1]? = some
    (.assign .real [] "e" (.compute (.alg .fp32 (.add .real .nil
      (.libdeviceExp (.ref .real [] "a")) (.exp (.ref .real [] "a")))))) := rfl

theorem distinct_syntax : (Op.libdeviceExp (.const 0) : Op .real []) ≠ .exp (.const 0) := by
  intro h
  cases h

theorem same_real_meaning (x : Op .real shape) (s : BlockState) :
    Triton.evalOp (.libdeviceExp x) s = Triton.evalOp (.exp x) s := by
  rw [Triton.evalOp_libdeviceExp, Triton.evalOp_exp]

theorem distinct_fp_meaning {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (x : Op .real shape) :
    evalOp M (some .fp32) (.libdeviceExp x) s =
      (do return (M.unary (some .fp32) .libdeviceExp) ∘ (← evalOp M (some .fp32) x s)) := by
  rw [evalOp_unfold]

private noncomputable def model : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else 1
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ op _ => if op = .libdeviceExp then 1 else 2
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def domain : FP.Guarded.Domain ℚ
  | .finite => fun _ => True
  | .nonzero => fun a => a ≠ 0
  | .positive => fun a => 0 < a

theorem library_law_holds : FP.SoftmaxShift.LibdeviceExpSub model domain := by
  constructor
  intros
  norm_num [FP.SoftmaxShift.exp, FP.ScalarArithmetic.div, model]

theorem intrinsic_law_does_not_follow :
    model.unary (some .fp32) .exp (FP.ScalarArithmetic.sub model 0 0) ≠
      FP.ScalarArithmetic.div model (model.unary (some .fp32) .exp 0)
        (model.unary (some .fp32) .exp 0) := by
  have h : Unary.exp ≠ .libdeviceExp := by decide
  norm_num [model, FP.ScalarArithmetic.div, h]

#axiomsClean FP.Exponential.exp_sub
#axiomsClean VeriTile.Bench.Examples.SoftmaxStableFPEquiv.softmax_stable_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPLibdeviceExpTests
