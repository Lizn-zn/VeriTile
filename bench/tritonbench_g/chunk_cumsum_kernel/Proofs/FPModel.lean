import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.FP
import VeriTile.Triton.Float.Countermodel
import VeriTile.Triton.Float.Integer

/-! One consistent abstract backend for the scalar assumptions. The scan
returns its input and the reduction returns one: neither is asserted to be
Triton's numerical implementation. Successful complete execution with this
backend witnesses that the FP domain and execution premises are inhabited. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.FPModel
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def model : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b
    | .max => max a b | .pow => 0
  -- One consistent interpretation of atomic addition; no universal atomic
  -- law is asserted by choosing this backend for execution witnesses.
  atomicAdd := fun _ => some (· + ·)
  unary := fun _ _ _ => 1
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareEq := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 1
  reduceSumTyped := fun _ d {_} _ _ _ => some (fun _ => ofFloat d 1)
  scan := fun _ {_} _ _ _ v => some v
  dotTyped := fun _ _ _ out _ {_} {_} {_} {_} _ _ => some (fun _ => ofFloat out 0)

def domain : FP.Guarded.Domain ℝ
  | .finite, _ => True
  | .nonzero, x => x ≠ 0
  | _, x => 0 < x

set_option maxHeartbeats 3200000
set_option maxRecDepth 3000
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
kernel equivalence still uses the selected FP atom, not real algebra. -/
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


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.FPModel
