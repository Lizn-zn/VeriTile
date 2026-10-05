/- Piecewise log-exp implementation, branching and exact-counterexample checks.
The rational interpreter below is a test fixture, not numerical evidence. -/
import VeriTile.Triton.DSL
import bench.examples.LogExp.FPEquiv
import VeriTile.Triton.Float.LogExpCounterexample
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

noncomputable section
namespace FPLogExpTests
open VeriTile Triton FP.Structural Bench.Examples

def shortNames : ComputeKernel := triton {
  a := tl.load($(("x" : RegionName)) + 0, dtype=tl.float32)
  b := libdevice.log1p(libdevice.expm1(a))
  c := libdevice.log(libdevice.exp(a))
}

def qualifiedNames : ComputeKernel := triton {
  a := tl.load($(("x" : RegionName)) + 0, dtype=tl.float32)
  b := tl.extra.cuda.libdevice.log1p(tl.extra.cuda.libdevice.expm1(a))
  c := tl.extra.cuda.libdevice.log(tl.extra.cuda.libdevice.exp(a))
}

theorem aliases_match : shortNames = qualifiedNames := rfl

theorem precision_and_intrinsics_preserved : shortNames.surfaceBody[1]? = some
    (.assign .real [] "b" (.compute (.alg .fp32
      (.libdeviceLog1p (.libdeviceExpm1 (.ref .real [] "a")))))) := rfl

theorem log_symbols_distinct : (Op.libdeviceLog (.const 1) : Op .real []) ≠ .log (.const 1) := by
  intro h
  cases h

private noncomputable def model : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else if r = 80 then 80 else 1 / 2
  negInf := 0
  binary := fun _ _ _ a b => a - b
  unary := fun _ op a => match op with
    | .libdeviceExpm1 => a + 10
    | .libdeviceLog1p => a + 100
    | .libdeviceExp => a + 1000
    | .libdeviceLog => a + 10000
    | _ => 0
  compareLt := fun p _ => if p = some .fp32 then some (fun a b => decide (a < b)) else none
  compareLe := fun p _ => if p = some .fp32 then some (fun a b => decide (a ≤ b)) else none
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def initial : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

set_option maxHeartbeats 1600000 in
theorem piecewise_execution (a : ℚ) :
    (evalOp model (some .fp32) (FP.LogExp.expression .libdevice FP.LogExp.input)
      (initial.setReg "a" .real [] (fun _ => a))).map (fun v => v PUnit.unit) =
      some (if 1 / 2 < (if a < 0 then -a else a) ∧
        (if a < 0 then -a else a) ≤ 80 then a else a + 11000) := by
  norm_num [evalOp_unfold, FP.LogExp.expression, FP.LogExp.Backend.log, FP.LogExp.useIdentity, FP.LogExp.absolute,
    FP.LogExp.input, numeric, numericLt, numericLe,
    State.setReg, bop, model]
  split <;> simp_all <;> ring

-- Both signs at 0.5 use fallback, both signs at 80 return a, and the
-- near-zero and extreme negative inputs still execute the original calls.
theorem threshold_branches :
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        (-(1 / 2)) = 21999 / 2 ∧
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        (1 / 2) = 22001 / 2 ∧
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        (-1) = -1 ∧
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        80 = 80 ∧
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        (-80) = -80 ∧
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        (-90) = 10910 ∧
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        0 = 11000 ∧
    FP.LogExp.value .libdevice model (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
        90 = 11090 := by
  norm_num [FP.LogExp.value, FP.LogExp.Backend.logOp, model]

theorem missing_comparison_fails :
    evalOp model none (FP.LogExp.useIdentity (.const 0)) initial = none := by
  simp [evalOp_unfold, FP.LogExp.useIdentity, FP.LogExp.absolute,
    numericLt, numericLe, model]

theorem comparison_default_precision :
    (model.withDefaultPrecision .fp32).compareLe none .real =
      some (fun a b => decide (a ≤ b)) ∧
    (model.withDefaultPrecision .fp32).compareLe (some .fp64) .real = none := by
  simp [Algebra.withDefaultPrecision, resolvePrecision, model]

private def withInput (a : ℚ) : State ℚ :=
  { initial with mem := fun _ _ => .mk .real a }

-- The domain is inhabited for arbitrary signed inputs, including the fallback
-- branch; successful comparisons impose neither branch interval on the inputs.
set_option maxHeartbeats 1600000 in
theorem finite_input_domain (a : ℚ) (B : Nat) :
    (LogExp.FPEquiv.domain "x" B).Holds (LogExp.FPEquiv.engine model)
      (fun _ _ => True) (withInput a) := by
  simp [FP.Guarded.Precondition.Holds, LogExp.FPEquiv.domain, LogExp.optimizedKernel,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    ComputeDType.eraseDType, LogExp.FPEquiv.engine, Algebra.withDefaultPrecision,
    resolvePrecision, numericLt, numericLe, numeric, bop, model]

-- The actual source executes the negative identity branch and supports in-place
-- writes. Memory outside this tile is preserved, even when input and output alias.
theorem negative_in_place_kernel :
    ∃ t, exec (LogExp.FPEquiv.engine model) (LogExp.optimizedKernel "x" "x" 4)
        (withInput (-1)) = some t ∧
      (∀ i : Fin 4, t.mem "x" i.val = .mk .real (-1)) ∧
      t.mem "x" 4 = .mk .real (-1) := by
  obtain ⟨t, ht, hv, hf⟩ := LogExp.FPEquiv.optimized_run model
    (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
    (by simp [model]) (by simp [model]) "x" "x" 4
    (withInput (-1)) (fun _ => -1) (fun _ => rfl)
  refine ⟨t, ht, ?_, ?_⟩
  · norm_num [withInput, initial, LogExp.FPEquiv.output, LogExp.FPEquiv.loaded,
      FP.LogExp.value, FP.LogExp.Backend.logOp, model] at hv
    exact hv
  · have hmiss : ∀ i : Fin 4, 4 ≠ (withInput (-1)).pids 0 * 4 + i.val := by
      intro i
      simp only [withInput, initial, Nat.zero_mul, Nat.zero_add]
      omega
    exact hf "x" 4 (Or.inr hmiss)

-- The fixed reference retains its fp32 output conversion.
theorem reference_retains_output_cast (a : ℚ) :
    ∃ t, exec (LogExp.FPEquiv.engine { model with cast := fun _ _ _ x => x + 7 })
        (LogExp.originalKernel "x" "y" 1) (withInput a) = some t ∧
      t.mem "y" 0 = .mk .real ((a + 11000) + 7) := by
  obtain ⟨t, ht, hv, _⟩ := LogExp.FPEquiv.original_run
    { model with cast := fun _ _ _ x => x + 7 }
    "x" "y" 1 (withInput a) (fun _ => a) (fun _ => rfl)
  refine ⟨t, ht, ?_⟩
  have h := hv ⟨0, by decide⟩
  norm_num [withInput, initial, LogExp.FPEquiv.output, LogExp.FPEquiv.loaded,
    FP.LogExp.referenceValue, FP.LogExp.Backend.logOp, model, add_assoc] at h ⊢
  exact h

-- Both candidate branches retain the output conversion, even when the
-- identity path eliminates the log and exp calls.
theorem candidate_retains_output_cast (a : ℚ) :
    ∃ t, exec (LogExp.FPEquiv.engine { model with cast := fun _ _ _ x => x + 7 })
        (LogExp.optimizedKernel "x" "y" 1) (withInput a) = some t ∧
      t.mem "y" 0 = .mk .real
        (FP.LogExp.value .libdevice model (fun a b => decide (a < b))
          (fun a b => decide (a ≤ b)) a + 7) := by
  obtain ⟨t, ht, hv, _⟩ := LogExp.FPEquiv.optimized_run
    { model with cast := fun _ _ _ x => x + 7 }
    (fun a b => decide (a < b)) (fun a b => decide (a ≤ b))
    (by simp [model]) (by simp [model])
    "x" "y" 1 (withInput a) (fun _ => a) (fun _ => rfl)
  refine ⟨t, ht, ?_⟩
  simpa [withInput, initial, LogExp.FPEquiv.output, LogExp.FPEquiv.loaded,
    FP.LogExp.value, FP.LogExp.Backend.logOp, model] using hv ⟨0, by decide⟩

-- The public proof is about both source kernels, with symbolic dimensions.
open scoped VeriTile.Spec in
theorem source_kernel_spec (R : FP.LogExp.Rules)
    (x y : RegionName) (B : Nat) :
    LogExp.FPEquiv.originalIO x y B ≡[R] LogExp.FPEquiv.optimizedIO x y B :=
  LogExp.FPEquiv.log_exp_equiv R x y B

-- Changing the default precision changes the contract, even for identical code.
theorem precision_is_in_signature (B : Nat) :
    Spec.ProgramSyntax.signature (LogExp.FPEquiv.originalIO "x" "y" B) ≠
      Spec.ProgramSyntax.signature
        { LogExp.FPEquiv.originalIO "x" "y" B with defaultPrecision := some .fp64 } := by
  intro h
  have hp := congrArg (fun s => s.2.2) h
  cases hp

#axiomsClean LogExp.FPEquiv.log_exp_equiv
#axiomsClean FP.LogExp.apply_log_exp_cancel
#axiomsClean FP.LogExpCounterexample.plain_log_exp_not_identity
#axiomsClean FP.LogExpCounterexample.singleton_lse_not_exact
#guard_msgs (drop info) in
#auditModuleAxioms

end FPLogExpTests
