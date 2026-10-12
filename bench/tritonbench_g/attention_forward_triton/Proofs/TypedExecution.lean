import bench.tritonbench_g.attention_forward_triton.Proofs.FP
import VeriTile.Triton.Float.Countermodel

/-! Nonempty executions of the typed full source and its candidate, with real
arithmetic for scalar operations and total opaque matrix/reduction backends.
This checks interpreter coverage and the actual normalization domain. It is
not a GPU emulation or an additional numerical admission. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

private def model : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b
    | .max => max a b | .pow => 0
  unary := fun _ _ _ => 1
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 1
  dotTyped := fun _ _ _ out _ {_} {_} {_} {_} _ _ => some (fun _ => ofFloat out 0)

private def initial : State ℝ where
  mem := fun r _ =>
    if r = "QS" ∨ r = "KS" then .mk .real 1 else .mk .fp16 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def domain : FP.Guarded.Domain ℝ
  | .finite, _ => True
  | .nonzero, x => x ≠ 0
  | _, x => 0 < x

set_option maxHeartbeats 3200000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private theorem finite_register (s : State ℝ) (gs : List FP.OperandGuard)
    (hd : FP.Guarded.ScalarDomain domain gs s) (name : RegName)
    (hm : ⟨name, .finite⟩ ∈ gs) :
    ∃ a : ℝ, s.regs .real [] name = some (fun _ => a) := by
  obtain ⟨v, hv, _⟩ := hd _ hm
  refine ⟨v PUnit.unit, hv.trans (congrArg some ?_)⟩
  funext i
  cases i
  rfl

private theorem nonzero_register (s : State ℝ) (gs : List FP.OperandGuard)
    (hd : FP.Guarded.ScalarDomain domain gs s) (name : RegName)
    (hm : ⟨name, .nonzero⟩ ∈ gs) (a : ℝ)
    (ha : s.regs .real [] name = some (fun _ => a)) : a ≠ 0 := by
  obtain ⟨v, hv, hn⟩ := hd _ hm
  have eqv := Option.some.inj (hv.symm.trans ha)
  simpa only [domain, eqv] using hn

/-- The execution witness satisfies all scalar candidates, with their guards.
Real arithmetic here establishes consistency of the abstract assumptions; the
kernel equivalence above still uses the selected FP atom, not real algebra. -/
private theorem arithmetic_atom (atom : FP.ScalarArithmetic.Atom) :
    FP.FragmentSound model domain (FP.ScalarArithmetic.lhs atom)
      (FP.ScalarArithmetic.rhs atom) := by
  open FP.ScalarArithmetic in
    refine ⟨rfl, fun s hd => ?_⟩
    change FP.Guarded.ScalarDomain domain (guards atom) s at hd
    cases atom
    all_goals
      obtain ⟨a, ha⟩ := finite_register s _ hd "a" (by decide)
      try
        have h : ∃ b : ℝ, s.regs .real [] "b" = some (fun _ => b) := by
          apply finite_register s _ hd
          decide
        obtain ⟨b, hb⟩ := h
      try
        have h : ∃ c : ℝ, s.regs .real [] "c" = some (fun _ => c) := by
          apply finite_register s _ hd
          decide
        obtain ⟨c, hc⟩ := h
      try have hn := nonzero_register s _ hd "b" (by decide) b hb
      try have hn := nonzero_register s _ hd "a" (by decide) a ha
      simp only [lhs, rhs, lhsCode, rhsCode, fragment, FP.ScalarArithmetic.ref,
        FP.ScalarArithmetic.plus, FP.ScalarArithmetic.minus, FP.ScalarArithmetic.times,
        FP.ScalarArithmetic.divide, FP.Reciprocal.lhs, FP.Reciprocal.rhs, FP.Reciprocal.div,
        FP.Reciprocal.inv, FP.Reciprocal.mul, FP.Reciprocal.ref,
        run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
        ComputeDType.eraseDType, numeric, ha, State.setReg_same]
      simp_all [State.setReg, model, bop, Function.comp_def, _root_.add_comm, _root_.add_assoc,
        _root_.mul_comm, _root_.mul_assoc, mul_add, div_eq_mul_inv]
      all_goals
        funext i
        dsimp [FP.Structural.bop]
        simp_all [_root_.add_comm, _root_.add_assoc, _root_.mul_comm, _root_.mul_assoc,
          mul_add, div_eq_mul_inv]
        all_goals ring

theorem model_satisfies_rules (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain := by
  apply FP.models_of_atoms
  intro entry member
  obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp member
  unfold FP.ScalarArithmetic.Atom.entry? at he
  cases hr : atom.report? with
  | none => simp [hr] at he
  | some row =>
      simp only [hr, Option.map_some, Option.some.injEq] at he
      subst entry
      exact .cons (arithmetic_atom atom) .nil

/-- One nonempty 16-row, 16-key loop, preserving the source's 128/96 head layout. -/
theorem original_executes : ∃ t, FP.Structural.exec (engine model)
    (TypedSource.original "Q" "K" "V" "QS" "KS" "Out"
      2048 2048 128 1 2048 2048 128 1 2048 2048 128 1 2048 2048 128 1
      1 1 16 128 16 16 128 96 1) initial = some t := by
  simp (config := { maxSteps := 300000 }) [TypedSource.original, FP.Structural.exec,
    run, step, FP.Structural.range, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, natLt, bop, FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]

theorem reciprocal_executes : ∃ t, FP.Structural.exec (engine model)
    (TypedSource.reciprocal "Q" "K" "V" "QS" "KS" "Out"
      2048 2048 128 1 2048 2048 128 1 2048 2048 128 1 2048 2048 128 1
      1 1 16 128 16 16 128 96 1) initial = some t := by
  simp (config := { maxSteps := 300000 }) [TypedSource.reciprocal, FP.Structural.exec,
    run, step, FP.Structural.range, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, natLt, bop, FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]

theorem normalization_domain_inhabited :
    (FPEquivProofs.divisionSite "Q" "K" "V" "QS" "KS" "Out"
      2048 2048 128 1 2048 2048 128 1 2048 2048 128 1 2048 2048 128 1
      1 1 16 128 16 16 128 96 1).Holds model domain initial := by
  intro t ht a b ha hb
  simp (config := { maxSteps := 300000 }) [FPEquivProofs.divisionSite, Source.prefixBefore,
    TypedSource.original, ComputeKernel.surfaceBody, run, step, FP.Structural.range,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, initial,
    ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, numericLt, natLt,
    bop, State.setReg, TileShape.axisDim, Region.cast] at ht
  subst t
  simp [evalOp_unfold, State.setReg, TileShape.axisDim] at ha hb
  subst a
  rw [← Option.some.inj hb]
  norm_num [domain, bop, Pi.one_apply]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.TypedExecution
