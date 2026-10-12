import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import bench.tritonbench_g.context_attn_nopad.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def initial : State ℝ where
  mem := fun r _ =>
    if r = "start" then .mk .int 1
    else if r = "sequence" then .mk .int 3
    else if r = "q" ∨ r = "k" ∨ r = "v" then .mk .fp16 1
    else .mk .fp16 99
  regs := fun _ _ _ => none
  pids := fun d => if d = 1 then 3 else 1
  numPids := fun _ => 8
  undef := fun d _ _ => defaultValue d

def original := FPProof.original .fp16 "q" "k" "v" 2 "start" "sequence" "out"
  8 2 1 8 2 1 8 2 1 8 2 1 2 2 4

def candidate := FPProof.scaleFirst .fp16 "q" "k" "v" 2 "start" "sequence" "out"
  8 2 1 8 2 1 8 2 1 8 2 1 2 2 4

def parts : Source.IntRange :=
  match Source.intRange? original.kernel.surfaceBody "start_n" with
  | some p => p
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split : original.kernel.surfaceBody =
    parts.before ++ [.forRangeInt "start_n" parts.start parts.stop parts.stride parts.body] ++ parts.after := rfl

def preloop : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,Source.intRange?,original,FPProof.original,VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody]
  simp (config := {maxSteps := 100000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    integralFloorDiv,integralMod]
  try rfl

def first : {t : State ℝ // run (engine model) parts.body
    (preloop.val.setReg "start_n" .int [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,Source.intRange?,original,FPProof.original,VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody]
  simp (config := {maxSteps := 100000}) [preloop,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    integralFloorDiv,integralMod]
  try rfl


theorem loop_step : step (engine model)
    (.forRangeInt "start_n" parts.start parts.stop parts.stride parts.body) preloop.val = some first.val := by
  have hs : FP.Structural.evalOp (engine model) none parts.start preloop.val = some (fun _ => (0 : Int)) := rfl
  have he : FP.Structural.evalOp (engine model) none parts.stop preloop.val = some (fun _ => (4 : Int)) := rfl
  have hd : FP.Structural.evalOp (engine model) none parts.stride preloop.val = some (fun _ => (4 : Nat)) := rfl
  simp only [step, hs, he, hd, bind, Option.bind_some]
  rw [rangeInt]
  change (run (engine model) parts.body (preloop.val.setReg "start_n" .int [] (fun _ => 0))).bind
    (rangeInt (engine model) "start_n" 4 4 4 parts.body) = _
  simp only [first.property, Option.bind_some]
  simp [rangeInt]

def postloop : {t : State ℝ // run (engine model) parts.after first.val = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,Source.intRange?,original,FPProof.original,VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody]
  simp (config := {maxSteps := 100000}) [first,preloop,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    integralFloorDiv,integralMod,FP.Structural.store]
  try rfl

theorem executes : FP.Structural.exec (engine model) original.kernel initial = some postloop.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  rw [source_split,run_append,run_append]
  simp only [preloop.property,Option.bind_some,run,loop_step,postloop.property]
  exact postloop.property

theorem domain_holds : ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original,FPProof.original,VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody,
    FPProof.sites,Source.intRange?,Source.prefixBefore,Site.Holds,domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initial = some postloop.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.equivalent .fp16 "q" "k" "v" 2 "start" "sequence" "out"
    8 2 1 8 2 1 8 2 1 8 2 1 2 2 4 R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_holds).symm.trans executes


/-- The one reached block writes both channels of its active row and preserves the
masked next row; the backend is the named abstract consistency witness. -/
theorem output_cells :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "out" 30,t.mem "out" 31,t.mem "out" 38)) =
      some (.mk .fp16 0,.mk .fp16 0,.mk .fp16 99) := by
  simp only [executes,Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.TypedExecution
