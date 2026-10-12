import bench.tritonbench_g.attention_fwd_triton1.Proofs.FP
import VeriTile.Triton.Float.Countermodel

/-! A nonempty model of the admitted scalar assumptions. Real scalar
arithmetic establishes consistency here; the program-equivalence proof uses
only the selected FP atom. Dots are total opaque operations returning zero,
so this is an abstract interpreter witness, not an IEEE or GPU execution. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.FPModel
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def model : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b
    | .max => max a b | .pow => 0
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0
  dotWithOptions := fun _ _ {_} {_} {_} {_} _ _ => some (fun _ => 0)

def domain : FP.Guarded.Domain ℝ
  | .finite, _ => True
  | .nonzero, x => x ≠ 0
  | _, x => 0 < x

set_option maxHeartbeats 3200000
set_option maxRecDepth 8000
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

/-- Every scalar in this model is finite, including values loaded after H
stores or changed by earlier iterations. The nonzero/positive guards of
other scalar candidates retain their usual meanings. -/
theorem requirements_hold (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) (s : State ℝ) :
    ∀ site ∈ FPEquivProofs.requirements q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND, site.Holds model domain s := by
  intro site member
  unfold FPEquivProofs.requirements at member
  split at member
  · simp at member
  · simp only [List.mem_singleton] at member
    subst site
    intro t ht a b d ha hb hd cur u hr hz hc z hp va vb hva hvb
    exact ⟨fun _ => trivial, trivial⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.FPModel
