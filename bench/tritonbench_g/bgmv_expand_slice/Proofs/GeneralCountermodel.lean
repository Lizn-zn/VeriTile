import bench.tritonbench_g.bgmv_expand_slice.Kernels
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel
import VeriTile.Triton.Float.Countermodel

/-! A logical countermodel to the pending fp32-rules-only general BGMV target.
This is not an IEEE model or a GPU numerical counterexample. The model changes
only fp16 addition; every admitted fp32 scalar fragment keeps its interpretation.
The exact current source is executed on a nonempty one-lane accumulation. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.GeneralCountermodel
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

def model : Algebra ℝ :=
  { AttentionFwdTriton3.FPModel.model with
    binary := fun p dtype op a b => match dtype, op with
      | .fp16, .add => a
      | _, _ => AttentionFwdTriton3.FPModel.model.binary p dtype op a b }

abbrev domain := AttentionFwdTriton3.FPModel.domain

/-- All fp32-profiled scalar primitives retain their original interpretation. -/
theorem fp32_binary_unchanged (p : Option ComputeDType) (op : Binary) (a b : ℝ) :
    model.binary p .real op a b = AttentionFwdTriton3.FPModel.model.binary p .real op a b :=
  rfl

/-- The only changed operation ignores its second operand. -/
theorem fp16_add_left (p : Option ComputeDType) (a b : ℝ) :
    model.binary p .fp16 .add a b = a := rfl

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

/-- The modified model satisfies every supported fp32 scalar candidate, with
its guards. This proves no fp16 law and asserts no identity about hardware. -/
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
      simp_all [State.setReg, model, AttentionFwdTriton3.FPModel.model, bop,
        Function.comp_def, _root_.add_comm, _root_.add_assoc,
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


-- These two locators deliberately reproduce the pending FP.lean definitions.
-- That module cannot be imported because its old general theorem does not compile.
def additionSite? (code : List ComputeStmt) : Option Site :=
  let before := Source.prefixBefore code "accumulator"
  match (code.drop before.length).head? with
  | some (.assign _ _ _ (.alg (.add .real (.consSame .nil) left right))) =>
      some (.add before _ left right)
  | some (.assign _ _ _ (.compute (.alg .fp32 (.add .real (.consSame .nil) left right)))) =>
      some (.addFP32 before _ left right)
  | _ => none

def requirementsFor (kernel : ComputeKernel) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody
      (fun body => (Source.dynamicRange? body "n").isSome)
    let loop ← Source.dynamicRange? active.yes "n"
    let accumulation ← Source.conditional? loop.body (fun _ => Bool.true)
    let addition ← additionSite? accumulation.yes
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "n" loop.start loop.stop loop.stride loop.body
        (.branch accumulation.before accumulation.condition Bool.true addition))).toList


def source (swapped : Bool) : ComputeKernel :=
  TypedSource.specialize .fp16 .fp16 Bool.false swapped .fp16
    "X" "W" "Y" 1 1 ⟨"Index"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.true Bool.true

def original : Program where
  kernel := source Bool.false
  domain := requirementsFor (source Bool.false)

def candidate : Program := { original with kernel := source Bool.true }

theorem original_source : original.kernel =
    TypedSource.original .fp16 .fp16 Bool.false .fp16
      "X" "W" "Y" 1 1 ⟨"Index"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.true Bool.true := rfl

theorem candidate_source : candidate.kernel =
    TypedSource.outputFirst .fp16 .fp16 Bool.false .fp16
      "X" "W" "Y" 1 1 ⟨"Index"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.true Bool.true := rfl

/-- The old locator fails on the actual half-addition assignment. -/
theorem original_domain_empty : original.domain = [] := rfl

def initial : State ℝ where
  mem := fun r _ => if r = "Index" then .mk .int 0 else .mk .fp16 2
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def program (optimized : Bool) := if optimized then candidate else original
private def code (optimized : Bool) := (program optimized).kernel.surfaceBody
private def active (optimized : Bool) := ((Source.conditional? (code optimized)
  (fun b => (Source.dynamicRange? b "n").isSome)).map (·.yes)).getD []
private def prologue := ((Source.dynamicRange? (active Bool.false) "n").map (·.before)).getD []
private def body (optimized : Bool) :=
  ((Source.dynamicRange? (active optimized) "n").map (·.body)).getD []

private def preguard : {t : State ℝ //
    run (engine model) ((code Bool.false).take 3) (initial) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [code, program, original, source, TypedSource.original,
    TypedSource.specialize, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, initial,
    engine, State.setReg, Cell.read, Region.cast]
  rfl

private def preloop : {t : State ℝ //
    run (engine model) prologue preguard.val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [prologue, active, code, program, original, source,
    TypedSource.original, TypedSource.specialize, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
    Option.map_some, Option.getD_some, beq_self_eq_true]
  dsimp [preguard]
  simp (config := { maxSteps := 100000 }) [Source.dynamicRange?, Option.map,
      Option.getD, run, step,
    FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine,
    Algebra.withDefaultPrecision, resolvePrecision, model, AttentionFwdTriton3.FPModel.model, numeric,
    integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
    State.setReg, Cell.read, ofFloat, toFloat, Region.cast]
  try rfl

private def firstEntry := preloop.val.setReg "n" .nat [] (fun _ => 0)
private def firstIteration (optimized : Bool) : {t : State ℝ //
    run (engine model) (body optimized) firstEntry = some t} := by
  cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [body, active, code, program, original, candidate, source, TypedSource.original,
      TypedSource.outputFirst, TypedSource.specialize, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
      Option.map_some, Option.getD_some, beq_self_eq_true]
    dsimp [firstEntry, preloop]
    simp (config := { maxSteps := 100000 }) [Source.dynamicRange?, Option.map,
      Option.getD, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, AttentionFwdTriton3.FPModel.model,
      ComputeDType.eraseDType,
      numeric, integralFloorDiv, integralMod, numericLt, numericLe, numericEq,
      natLt, bop, FP.Structural.store, State.setReg, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl


private def execution (swapped : Bool) : {t : State ℝ //
    FP.Structural.exec (engine model) (source swapped) initial = some t} := by
  refine ⟨(firstIteration swapped).val, ?_⟩
  have hbound : preloop.val.regs .nat [] "split_n_length" = some (fun _ => 1) := by
    simp [preloop, integralFloorDiv, FP.Structural.bop]
    funext i
    cases i
    rfl
  have hbody : code swapped = (code Bool.false).take 3 ++
      [.ifThen (.ne .int .nil (.ref .int [] "lora_index") (.constInt (-1)))
        (prologue ++ [.forRangeDyn "n" (.constNat 0)
          (.ref .nat [] "split_n_length") (.constNat 1) (body swapped)])] := by
    cases swapped <;> rfl
  have hcode : code swapped = (source swapped).surfaceBody := by cases swapped <;> rfl
  change run (engine model) (source swapped).surfaceBody initial = _
  rw [← hcode]
  rw [hbody, run_append, preguard.property]
  simp only [Option.bind_some, run, step]
  have hc : FP.Structural.evalOp (engine model) none
      (.ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))) preguard.val =
        some (fun _ => Bool.true) := by
    simp [preguard, evalOp_unfold, numericEq, bop]
    rfl
  rw [hc]
  simp only [bind, Option.bind_some, Option.bind_fun_some, ↓reduceIte, run_append,
    preloop.property, run, step, evalOp_unfold, hbound]
  change FP.Structural.range (engine model) "n" 0 1 1 (body swapped) preloop.val = _
  rw [FP.Structural.range]
  change (run (engine model) (body swapped) firstEntry).bind
    (FP.Structural.range (engine model) "n" 1 1 1 (body swapped)) = _
  simp only [(firstIteration swapped).property, Option.bind_some]
  simp [FP.Structural.range]

/-- Both runs succeed; the separating observation is not a failure artifact. -/
theorem both_execute :
    (∃ t, FP.Structural.exec (engine model) original.kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) candidate.kernel initial = some t) :=
  ⟨⟨(execution Bool.false).val, (execution Bool.false).property⟩,
    ⟨(execution Bool.true).val, (execution Bool.true).property⟩⟩

/-- Public profiled equivalence preserves the complete execution in every model
of its displayed assumptions, including through symmetry and transitivity. -/
theorem exec_eq_of_equiv {α : Type} [Inhabited α]
    (R : Spec.Assumptions FP.GuardedFragment) (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models R M D) (s : State α) {lhs rhs : Program}
    (h : lhs ≡[R] rhs) (hd : ∀ site ∈ lhs.domain, site.Holds M D s) :
    FP.Structural.exec (engine M) lhs.kernel s =
      FP.Structural.exec (engine M) rhs.kernel s := by
  have derivation := h.derivation
  change Spec.ProgramDerivation (fun _ _ : Program => False) R lhs rhs at derivation
  have sound {a b : Program}
      (proof : Spec.ProgramDerivation (fun _ _ : Program => False) R a b) :
      Spec.ProgramSyntax.signature a = Spec.ProgramSyntax.signature b ∧
      ((∀ site ∈ a.domain, site.Holds M D s) →
        FP.Structural.exec (engine M) a.kernel s =
          FP.Structural.exec (engine M) b.kernel s) := by
    induction proof with
    | refl => exact ⟨rfl, fun _ => rfl⟩
    | «syntax» hs hc _ => exact ⟨hs, fun _ => by cases hc; rfl⟩
    | structural _ hf => exact hf.elim
    | numerical hs hn => exact ⟨hs, fun hd => hn α M D hM s hd⟩
    | @symm a b _ ih =>
      refine ⟨ih.1.symm, fun hd => ?_⟩
      have domains := congrArg (fun sig => sig.2) ih.1
      change a.domain = b.domain at domains
      exact (ih.2 (domains.symm ▸ hd)).symm
    | @trans a b c _ _ ih₁ ih₂ =>
      refine ⟨ih₁.1.trans ih₂.1, fun hd => ?_⟩
      have domains := congrArg (fun sig => sig.2) ih₁.1
      change a.domain = b.domain at domains
      exact (ih₁.2 hd).trans (ih₂.2 (domains ▸ hd))
  exact (sound derivation).2 hd

/-- Observe an actual stored half value, retaining failure as None. -/
def observation (swapped : Bool) : Option ℝ :=
  (FP.Structural.exec (engine model) (source swapped) initial).map
    (fun t => (t.mem "Y" 0).read .fp16)

theorem original_stores_one : observation Bool.false = some 1 := by
  rw [observation, (execution Bool.false).property]
  simp [execution, firstIteration, TileShape.allIndices, List.finRange_succ, State.write, Cell.read]

theorem candidate_stores_two : observation Bool.true = some 2 := by
  rw [observation, (execution Bool.true).property]
  simp [execution, firstIteration, TileShape.allIndices, List.finRange_succ, State.write, Cell.read]

/-- The fp32-only rule table cannot prove the current general target: a legal,
nonempty half instance has different stored values in a model of every rule.
This is a logical countermodel, not evidence of a GPU numerical discrepancy. -/
theorem not_output_first_equiv (R : FP.ScalarArithmetic.Rules) :
    ¬ (original ≡[R] candidate) := by
  intro h
  have he := exec_eq_of_equiv R.assumptions model domain (model_satisfies_rules R)
    initial h (by simp [original_domain_empty])
  have ho : observation Bool.false = observation Bool.true :=
    congrArg (fun result => result.map (fun t => (t.mem "Y" 0).read .fp16)) he
  rw [original_stores_one, candidate_stores_two] at ho
  norm_num at ho

#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.GeneralCountermodel
