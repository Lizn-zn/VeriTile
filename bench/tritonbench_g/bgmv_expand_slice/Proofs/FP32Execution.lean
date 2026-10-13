import bench.tritonbench_g.bgmv_expand_slice.Proofs.FP32
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! A nonempty execution of the complete typed source, in a consistent model
of the admitted atoms. The fixture uses fp32 inputs cast to bf16 weights,
fp32 in-place output, a partial K tile and two n-loop iterations with a tail.
This is a semantic consistency witness, not a GPU result or a real GEMV proof. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FP32Execution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 2200000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

private def initial (slot : Int) : State ℝ where
  mem := fun r _ => if r = "Index" then .mk .int slot
    else if r = "X" then .mk .fp32 2
    else if r = "W" then .mk .bf16 2
    else .mk .real 2
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

private def original := FP32Proofs.original .fp32 .bf16 Bool.true
  "X" "W" "Y" 5 3 ⟨"Index"⟩ 3 1 15 3 1 5 1 0 2 4 2 Bool.false Bool.true
private def candidate := FP32Proofs.outputFirst .fp32 .bf16 Bool.true
  "X" "W" "Y" 5 3 ⟨"Index"⟩ 3 1 15 3 1 5 1 0 2 4 2 Bool.false Bool.true
private def program (optimized : Bool) := if optimized then candidate else original
private def code (optimized : Bool) := (program optimized).kernel.surfaceBody
private def active (optimized : Bool) := ((Source.conditional? (code optimized)
  (fun b => (Source.dynamicRange? b "n").isSome)).map (·.yes)).getD []
private def prologue := ((Source.dynamicRange? (active Bool.false) "n").map (·.before)).getD []
private def body (optimized : Bool) :=
  ((Source.dynamicRange? (active optimized) "n").map (·.body)).getD []

private def preguard (slot : Int) : {t : State ℝ //
    run (engine model) ((code Bool.false).take 3) (initial slot) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [code, program, original, FP32Proofs.original, TypedSource.original,
    TypedSource.specialize, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, initial,
    engine, State.setReg, Cell.read, Region.cast]
  rfl

private def preloop : {t : State ℝ //
    run (engine model) prologue (preguard 1).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [prologue, active, code, program, original, FP32Proofs.original,
    TypedSource.original, TypedSource.specialize, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
    Option.map_some, Option.getD_some, beq_self_eq_true]
  dsimp [preguard]
  simp (config := { maxSteps := 100000 }) [Source.dynamicRange?, Option.map, Option.getD, run, step,
    FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine,
    Algebra.withDefaultPrecision, resolvePrecision, model, numeric,
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
    simp only [body, active, code, program, original, candidate,
      FP32Proofs.original, FP32Proofs.outputFirst, TypedSource.original,
      TypedSource.outputFirst, TypedSource.specialize, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
      Option.map_some, Option.getD_some, beq_self_eq_true]
    dsimp [firstEntry, preloop]
    simp (config := { maxSteps := 100000 }) [Source.dynamicRange?, Option.map, Option.getD, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, integralFloorDiv, integralMod, numericLt, numericLe, numericEq,
      natLt, bop, FP.Structural.store, State.setReg, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

private def secondEntry (optimized : Bool) :=
  (firstIteration optimized).val.setReg "n" .nat [] (fun _ => 2)
private def secondIteration (optimized : Bool) : {t : State ℝ //
    run (engine model) (body optimized) (secondEntry optimized) = some t} := by
  cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [body, active, code, program, original, candidate,
      FP32Proofs.original, FP32Proofs.outputFirst, TypedSource.original,
      TypedSource.outputFirst, TypedSource.specialize, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
      Option.map_some, Option.getD_some, beq_self_eq_true]
    dsimp [secondEntry, firstIteration]
    simp (config := { maxSteps := 100000 }) [Source.dynamicRange?, Option.map, Option.getD, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, integralFloorDiv, integralMod, numericLt, numericLe, numericEq,
      natLt, bop, FP.Structural.store, State.setReg, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

private def loopStmt (optimized : Bool) : ComputeStmt :=
  .forRangeDyn "n" (.constNat 0) (.ref .nat [] "split_n_length") (.constNat 2) (body optimized)

private theorem loop_executes (optimized : Bool) :
    step (engine model) (loopStmt optimized) preloop.val = some (secondIteration optimized).val := by
  have hb : preloop.val.regs .nat [] "split_n_length" = some (fun _ => 3) := by
    simp [preloop, integralFloorDiv, FP.Structural.bop]
    funext i
    change (6 : Nat) / 2 = 3
    decide
  simp only [loopStmt, step, evalOp_unfold, hb, bind, Option.bind]
  rw [FP.Structural.range]
  change (run (engine model) (body optimized) firstEntry).bind
    (FP.Structural.range (engine model) "n" 2 3 2 (body optimized)) = _
  rw [(firstIteration optimized).property, Option.bind_some, FP.Structural.range]
  change (run (engine model) (body optimized) (secondEntry optimized)).bind
    (FP.Structural.range (engine model) "n" 4 3 2 (body optimized)) = _
  simp only [(secondIteration optimized).property, Option.bind_some]
  simp [FP.Structural.range]

private theorem active_executes (optimized : Bool) :
    run (engine model) (active optimized) (preguard 1).val = some (secondIteration optimized).val := by
  have hs : active optimized = prologue ++ [loopStmt optimized] := by cases optimized <;> rfl
  rw [hs, run_append, preloop.property, Option.bind_some]
  simp only [run, loop_executes, bind, Option.bind]

private def condition : Op .bool [] :=
  .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))

private theorem condition_eval (slot : Int) :
    FP.Structural.evalOp (engine model) none condition (preguard slot).val =
      some (fun _ => decide (slot ≠ -1)) := by
  simp [condition, preguard, evalOp_unfold, numericEq, bop]
  rfl

private theorem body_split (optimized : Bool) :
    code optimized = (code Bool.false).take 3 ++ [.ifThen condition (active optimized)] := by
  cases optimized <;> rfl

theorem source_executes (optimized : Bool) :
    ∃ t, FP.Structural.exec (engine model) (program optimized).kernel (initial 1) = some t := by
  refine ⟨(secondIteration optimized).val, ?_⟩
  change run (engine model) (code optimized) (initial 1) = _
  rw [body_split, run_append, (preguard 1).property, Option.bind_some]
  simp [run, step, condition_eval, active_executes]

/-- The signed sentinel bypasses the complete active body, including all
old-output reads and stores. Only the three metadata-prefix assignments run. -/
theorem sentinel_skips (optimized : Bool) :
    FP.Structural.exec (engine model) (program optimized).kernel (initial (-1)) =
      some (preguard (-1)).val := by
  change run (engine model) (code optimized) (initial (-1)) = _
  rw [body_split, run_append]
  simp only [(preguard (-1)).property, Option.bind_some]
  simp [run, step, condition_eval]

private def addArm := ((Source.conditional? (body Bool.false) (fun _ => Bool.true)).map (·.yes)).getD []

private def sumOperand : Op .real [2] :=
  .castFloat .bf16 .real (.reduceSumTyped .bf16 ⟨1, by decide⟩ Bool.false
    (.mul .bf16 (.leadL (.consSame .nil)) (.ref .bf16 [4] "tiled_a")
      (.ref .bf16 [2,4] "tiled_b")))

/-- This definitional check exposes the concrete fixture's actual guard. It
also prevents the consistency proof from succeeding on an empty locator. -/
private theorem original_domain : original.domain =
    [.branch ((code Bool.false).take 3) condition Bool.true
      (.range prologue "n" (.constNat 0) (.ref .nat [] "split_n_length") (.constNat 2)
        (body Bool.false)
        (.branch ((body Bool.false).take 4) (.constBool Bool.true) Bool.true
          (.addFP32 (Source.prefixBefore addArm "accumulator") [2] sumOperand
            (.ref .real [2] "tiled_out"))))] := by rfl

theorem domain_inhabited : ∀ site ∈ original.domain, site.Holds model domain (initial 1) := by
  rw [original_domain]
  simp [Site.Holds, domain]

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) original.kernel (initial 1) = some t) ∧
    (∃ t, FP.Structural.exec (engine model) candidate.kernel (initial 1) = some t) ∧
    (∀ site ∈ original.domain, site.Holds model domain (initial 1)) := by
  exact ⟨model_satisfies_rules R, source_executes Bool.false,
    source_executes Bool.true, domain_inhabited⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FP32Execution
