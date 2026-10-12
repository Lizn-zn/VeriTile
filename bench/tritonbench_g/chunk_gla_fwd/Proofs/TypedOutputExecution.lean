import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPOutput
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! A nonempty complete typed execution in a named abstract model. Two K
iterations, partial time/K/V tiles and nonzero program IDs retain mixed storage,
query casts, both dot options and the final store. Dot and exp return one in
this model; the witness is neither a GPU measurement nor a real specification. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutputExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

/-- Algorithm fp32 loads use `.real` cells; the computation profile carries
fp32 precision. Other storage types keep their own cell tags. -/
def initial : State ℝ where
  mem := fun r _ =>
    if r = "q" then .mk .fp16 1
    else if r = "v" ∨ r = "h" then .mk .bf16 1
    else if r = "g" then .mk .bf16 0
    else if r = "o" then .mk .fp16 99
    else .mk .real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := FPOutput.original .fp16 .bf16 .bf16 .bf16 .fp16 "q" "v" "g" "h" "o" "A"
  9 3 9 3 18 3 1 3 3 3 2 2 2

def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPOutput.original,TypedOutput.original,TypedOutput.source,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

def first : {t : State ℝ // run (engine model) parts.body (entry.val.setReg "i_k" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPOutput.original,TypedOutput.original,TypedOutput.source,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod,entry]
  try rfl

def second : {t : State ℝ // run (engine model) parts.body (first.val.setReg "i_k" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPOutput.original,TypedOutput.original,TypedOutput.source,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod,entry,first]
  try rfl

def finish : {t : State ℝ // run (engine model) parts.after second.val = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPOutput.original,TypedOutput.original,TypedOutput.source,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod,entry,first,second]
  try rfl

theorem source_parts : original.kernel.surfaceBody = parts.before ++
    [.forRangeDyn "i_k" parts.start parts.stop parts.stride parts.body] ++ parts.after := rfl

theorem loop_execution : step (engine model) (.forRangeDyn "i_k" parts.start parts.stop parts.stride parts.body) entry.val = some second.val := by
  have hs : parts.start = .constNat 0 := rfl
  have ht : parts.stop = .div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat 3) (.constNat 2)) (.constNat 1)) (.constNat 2) := rfl
  have hd : parts.stride = .constNat 1 := rfl
  simp only [step,hs,ht,hd,evalOp_unfold,numeric,bop,Option.bind,Option.bind_some]
  change range (engine model) "i_k" 0 2 1 parts.body entry.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (0 : Nat) < 2 from by decide,
    ↓reduceIte,first.property,bind,Option.bind_some]
  change range (engine model) "i_k" 1 2 1 parts.body first.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (1 : Nat) < 2 from by decide,
    ↓reduceIte,second.property,bind,Option.bind_some]
  change range (engine model) "i_k" 2 2 1 parts.body second.val = _
  rw [range]
  simp

theorem execution : FP.Structural.exec (engine model) original.kernel initial = some finish.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  simp only [source_parts,List.append_assoc,run_append,entry.property,Option.bind_some,List.singleton_append]
  rw [run]
  simp only [loop_execution,bind,Option.bind_some,finish.property]

theorem domain_holds : ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original,FPOutput.original,FPOutput.requirements,TypedOutput.original,TypedOutput.source,
    ComputeKernel.surfaceBody,Source.prefixBefore,Site.Holds,domain]

theorem values :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "o" 17,t.mem "o" 16,t.mem "q" 17)) =
      some (.mk .fp16 3,.mk .fp16 99,.mk .fp16 1) := by
  rw [execution]
  simp only [Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model)
      (FPOutput.sumLeft .fp16 .bf16 .bf16 .bf16 .fp16 "q" "v" "g" "h" "o" "A"
        9 3 9 3 18 3 1 3 3 3 2 2 2).kernel initial = some finish.val := by
  have he := FP.ProfiledRewrite.sound
    (FPOutput.output_equiv .fp16 .bf16 .bf16 .bf16 .fp16 "q" "v" "g" "h" "o" "A"
      9 3 9 3 18 3 1 3 3 3 2 2 2 R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_holds).symm.trans execution

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutputExecution
