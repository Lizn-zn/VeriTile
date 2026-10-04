/- Libdevice softmax execution and the boundary of its scalar law.
The rational fixtures are logical models, not GPU evidence or FP admissions. -/
import bench.examples.SoftmaxStable.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

open VeriTile.Bench.Examples.SoftmaxStable.Kernels

namespace FPSoftmaxStableTests
open VeriTile Triton FP.Structural
open VeriTile.Bench.Examples SoftmaxStableFPExecution
open scoped VeriTile.Spec

-- The original max operation rejects an empty row. Equality of failures
-- must not be mistaken for a successful FP equivalence.
theorem stable_empty_fails {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    FP.Structural.exec M (stableSoftmaxKernel "x" "y" 0) s = none := by
  simp [stableSoftmaxKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, TileShape.axisDim, TileShape.eraseAxis, Region.cast]

-- Source reads precede the store, so output aliasing does not invalidate
-- either the typed output equation or the frame around the row.
example {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (hB : 0 < B)
    (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (stableSoftmaxKernel "x" "x" B) s = some t ∧
      (∀ i : Fin B, t.mem "x" (s.pids 0 * B + i.val) = .mk .bf16
        (normalizedValue M (shifted M xs) i)) ∧
      (∀ (r : RegionName) o, (r ≠ "x" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) :=
  stable_run M "x" "x" B hB xs s hx

private noncomputable def model (exponential : ℚ → ℚ) : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else 1
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ _ a => exponential a
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

private def initial : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

-- The required libdevice equation is satisfiable independently of any
-- experimental claim. It is not a law of an arbitrary unary interpretation.
theorem constant_one_exp_sub : FP.SoftmaxShift.LibdeviceExpSub (model fun _ => 1) domain := by
  constructor
  intros
  norm_num [FP.SoftmaxShift.exp, FP.ScalarArithmetic.div, model]

theorem arbitrary_exp_sub_fails : ¬ FP.SoftmaxShift.LibdeviceExpSub (model fun a => a + 1) domain := by
  intro h
  have bad := h.apply 0 1 trivial trivial
  norm_num [FP.SoftmaxShift.exp, FP.ScalarArithmetic.sub, FP.ScalarArithmetic.div, model] at bad

-- Check the actual reified contract, including the scheduled padded sum,
-- rather than only the finite input guards.
theorem domain_is_satisfiable :
    (SoftmaxStableFPContract.requirements "x" 2 FP.Equational.seededSchedules).Holds
      (model fun _ => 1) domain initial := by
  rw [SoftmaxStableFPContract.requirements_holds]
  change FP.SoftmaxShift.ShiftDomain _ _ _ _ (.add (.input 0) (.add (.input 1) .zero))
  constructor
  · exact fun _ => trivial
  · trivial
  · trivial
  · norm_num [domain, FP.SoftmaxShift.exp, model]
  · constructor <;>
      norm_num [domain, FP.SoftmaxShift.exponentials, FP.SoftmaxShift.exp,
        FP.SoftmaxShift.scale, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.one,
        FP.ScalarArithmetic.div, FP.ScalarArithmetic.mul, FP.ScalarArithmetic.add,
        FP.ScalarReduction.value, FP.ScalarReduction.FiniteTree, model]

-- Even EXP-SUB alone does not justify inverse cancellation: constant zero
-- satisfies that equation over rationals but fails the nonzero domain.
theorem zero_exp_sub : FP.SoftmaxShift.LibdeviceExpSub (model fun _ => 0) domain := by
  constructor
  intros
  norm_num [FP.SoftmaxShift.exp, FP.ScalarArithmetic.div, model]

theorem zero_exp_rejected :
    ¬ (SoftmaxStableFPContract.requirements "x" 2 FP.Equational.seededSchedules).Holds
      (model fun _ => 0) domain initial := by
  intro h
  have bad := (SoftmaxStableFPContract.requirements_holds _ _ _ _ _ _).mp h
  have hn := bad.centerExpNonzero
  exact hn rfl

private def lhs := SoftmaxStableFPContract.stable "x" "y" 2
private def rhs := SoftmaxStableFPContract.naive "x" "y" 2

theorem output_length_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with io := { lhs.io with Bout := 1 } } := by
  intro h
  have bad := congrArg (fun sig => sig.1.Bout) h
  cases bad

theorem precision_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with profile := ⟨.fp64, Bool.true⟩ } := by
  intro h
  have bad := congrArg (fun sig => sig.2.1.defaultPrecision) h
  cases bad

theorem domain_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with domain := fun _ => .top } := by
  intro h
  have bad := congrArg (fun sig => sig.2.2 FP.Equational.seededSchedules) h
  cases bad

-- This self comparison exercises the new scheduled IO instance, including
-- successful empty-row execution. It is not the target softmax rewrite.
private theorem naive_self (B : Nat) : IO₁Equiv (naiveIO "x" "y" B) (naiveIO "x" "y" B) := by
  refine ⟨by simp [IO₁PrivateScratch, naiveIO], by simp [IO₁PrivateScratch, naiveIO], ?_⟩
  intro α _ M s
  obtain ⟨t, ht, _, hf⟩ := naive_run M "x" "y" B
    (SoftmaxStableFPContract.rowValues s "x" B) s (fun _ => rfl)
  exact ⟨t, t, ht, ht, fun _ => rfl, fun r o ho _ => hf r o ho, fun r o ho _ => hf r o ho⟩

specification naive_self_spec (B : Nat) :
    (SoftmaxStableFPContract.naive "x" "y" B) ≡[[]]
      (SoftmaxStableFPContract.naive "x" "y" B) :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl
    (FP.Scheduled.Equivalent₁.ofStructural [] rfl (naive_self B))

specification opaque_softmax (h : FP.Scheduled.Equivalent₁ [] lhs rhs) : lhs ≡[[]] rhs :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl h

#axiomsClean stable_empty_fails
#axiomsClean FP.SoftmaxShift.normalized
#axiomsClean SoftmaxStableFPContract.original_runs_under_exp
#axiomsClean constant_one_exp_sub
#axiomsClean arbitrary_exp_sub_fails
#axiomsClean domain_is_satisfiable
#axiomsClean zero_exp_sub
#axiomsClean zero_exp_rejected
#axiomsClean naive_self_spec
#print_fp_assumptions opaque_softmax
#print_fp_assumptions naive_self_spec

end FPSoftmaxStableTests
