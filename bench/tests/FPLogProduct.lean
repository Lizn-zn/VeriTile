import VeriTile.Triton.Float.LogExp
import VeriTile.Triton.DSL
import Mathlib.Tactic.NormNum

/- The rational operations below test control flow, precision and domains.
They are deliberately not real logarithms and supply no numerical evidence. -/
noncomputable section
namespace FPLogProductTests
open VeriTile Triton FP Structural Guarded LogExp
set_option maxHeartbeats 1600000

-- Preserve the actual sequence of assignments and masks in the measured
-- candidate, with scalar loads and a store around the atomic computation.
def measuredCandidate : ComputeKernel := triton {
  a := tl.load($(("x" : RegionName)) + 0, dtype=tl.float32)
  b := tl.load($(("y" : RegionName)) + 0, dtype=tl.float32)
  p := a * b
  keep_product := (p >= 0.5) & (p <= 2.0)
  direct := libdevice.log(tl.where(keep_product, p, 1.0))
  split_a := tl.where(keep_product, 1.0, a)
  split_b := tl.where(keep_product, 1.0, b)
  split := libdevice.log(split_a) + libdevice.log(split_b)
  out := tl.where(keep_product, direct, split)
  tl.store($(("z" : RegionName)) + 0, out)
}

private theorem half_eq : (0.5 : ℝ) = 1 / 2 := by norm_num
private theorem one_eq : (1.0 : ℝ) = 1 := by norm_num
private theorem two_eq : (2.0 : ℝ) = 2 := by norm_num

theorem measured_candidate_runs {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (le : α → α → Bool)
    (hle : M.compareLe (some .fp32) .real = some le) :
    ∃ t, Structural.exec (M.withDefaultPrecision .fp32) measuredCandidate s = some t ∧
      t.mem "z" 0 = Cell.mk .real
        (splitProductValue M le (M.fp32Load ((s.mem "x" 0).read .real))
          (M.fp32Load ((s.mem "y" 0).read .real))) ∧
      ∀ (r : RegionName) o, (r ≠ "z" ∨ o ≠ 0) → t.mem r o = s.mem r o := by
  simp [measuredCandidate, Structural.exec, run, step, evalExpr, evalComputeOp,
    Algebra.withDefaultPrecision, resolvePrecision,
    evalOp_unfold, numeric, numericLe, hle, bop, store, Region.cast,
    ComputeDType.eraseDType, TileShape.allIndices, splitProductValue, half_eq, one_eq, two_eq]
  intro r o hmiss
  exact (State.write_other _ "z" r 0 o _ hmiss).trans rfl

private def model : Algebra ℚ where
  literal := fun _ _ r => if r = 1 / 2 then 1 / 2 else if r = 1 then 1 else if r = 2 then 2 else 0
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .mul => a * b
    | .add => a + b
    | _ => 0
  unary := fun _ op a => if op = .libdeviceLog then a + 10 else 0
  compareLt := fun _ _ => none
  compareLe := fun p _ => if p = some .fp32 then some (fun a b => decide (a ≤ b)) else none
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def le (a b : ℚ) : Bool := decide (a ≤ b)

-- Both endpoints retain the product; both exterior intervals split the logs.
theorem branch_endpoints_and_exterior :
    splitProductValue model le (1 / 2) 1 = 21 / 2 ∧
    splitProductValue model le 2 1 = 12 ∧
    splitProductValue model le (1 / 4) 1 = 85 / 4 ∧
    splitProductValue model le 3 1 = 24 := by
  norm_num [splitProductValue, model, le]

-- The selector observes the model's multiplication result, not a separately
-- computed real product. Here multiplication returns one even for inputs 3,1.
private def roundedModel : Algebra ℚ :=
  { model with binary := fun p d op a b =>
      if op = .mul then 1 else model.binary p d op a b }

theorem selector_uses_computed_product :
    splitProductValue roundedModel le 3 1 = 11 := by
  norm_num [splitProductValue, roundedModel, model, le]

private def initial : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem missing_fp32_comparison_fails :
    evalOp { model with compareLe := fun _ _ => none } (some .fp32)
      (splitProduct (.const 1) (.const 1)) initial = none := by
  simp [splitProduct, keepProduct, evalOp_unfold, numeric, numericLe]

theorem wrong_precision_cannot_execute_comparison :
    evalOp model (some .fp64) (keepProduct (.const 1)) initial = none := by
  simp [keepProduct, evalOp_unfold, numericLe, model]

private def operands (a b : ℚ) : State ℚ :=
  (initial.setReg "a" .real [] (fun _ => a)).setReg "b" .real [] (fun _ => b)

private def domain : Domain ℚ
  | .finite => fun _ => True
  | .positive => fun a => 0 < a
  | .nonzero => fun a => a ≠ 0

-- No branch interval is smuggled into the input condition. A positive product
-- formed from negative operands still does not meet the experiment's domain.
theorem domain_has_only_positive_operand_conditions (a b : ℚ) :
    ScalarDomain domain Atom.log_mul_split.guards (operands a b) ↔ 0 < a ∧ 0 < b := by
  simp [ScalarDomain, Atom.guards, productGuards, operands, State.setReg, domain]

#axiomsClean measured_candidate_runs
#axiomsClean FP.LogExp.apply_log_mul_split
end FPLogProductTests
