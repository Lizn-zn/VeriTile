import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPIntra
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Nonempty complete typed executions with three streamed columns, partial
row/feature tiles, mixed storage and a nonzero chunk/subblock/head. The model
returns one for exp/sum and is used only as a consistent execution witness,
not as GPU evidence or the real output formula. Both causal arms execute;
the masked fourth row and unvisited fourth column retain their tagged cells. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedIntraExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxRecDepth 28000
set_option maxHeartbeats 20000000
set_option linter.unusedSimpArgs false

def initial (skipped : Bool) : State ℝ where
  mem := fun r off => if r = "q" then .mk .fp16 1
    else if r = "k" then .mk .bf16 off
    else if r = "g" then .mk .fp16 off
    else .mk .nat 99
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then (if skipped then 2 else 1) else 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := FPIntra.original .fp16 .bf16 .fp16 "q" "k" "g" "A" 40 5 2 15 3 8 4 4

def active : Source.Conditional :=
  (Source.conditional? original.kernel.surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def parts : Source.DynamicRange :=
  (Source.dynamicRange? active.yes "j").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def initialized (skipped : Bool) :
    {t : State ℝ // run (engine model) active.before (initial skipped) = some t} := by
  cases skipped
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [active,original,FPIntra.original,TypedIntra.original,TypedIntra.source,ComputeKernel.surfaceBody,Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
    simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,initial]
    try rfl

def entry : {t : State ℝ // run (engine model) parts.before (initialized Bool.false).val = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPIntra.original,TypedIntra.original,TypedIntra.source,ComputeKernel.surfaceBody,Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte,Source.dynamicRange?,String.reduceEq,beq_self_eq_true]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,initialized,initial]
  try rfl

def first : {t : State ℝ // run (engine model) parts.body
    (entry.val.setReg "j" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPIntra.original,TypedIntra.original,TypedIntra.source,ComputeKernel.surfaceBody,Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte,Source.dynamicRange?,String.reduceEq,beq_self_eq_true]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,entry,initialized,initial]
  try rfl

def second : {t : State ℝ // run (engine model) parts.body
    (first.val.setReg "j" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPIntra.original,TypedIntra.original,TypedIntra.source,ComputeKernel.surfaceBody,Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte,Source.dynamicRange?,String.reduceEq,beq_self_eq_true]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,first,entry,initialized,initial]
  try rfl

def third : {t : State ℝ // run (engine model) parts.body
    (second.val.setReg "j" .nat [] (fun _ => 2)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPIntra.original,TypedIntra.original,TypedIntra.source,ComputeKernel.surfaceBody,Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte,Source.dynamicRange?,String.reduceEq,beq_self_eq_true]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,second,first,entry,initialized,initial]
  try rfl

theorem loop_execution :
    step (engine model) (.forRangeDyn "j" parts.start parts.stop parts.stride parts.body) entry.val = some third.val := by
  have hs : parts.start = .constNat 0 := rfl
  have hd : parts.stride = .constNat 1 := rfl
  have ht : FP.Structural.evalOp (engine model) none parts.stop entry.val = some (fun _ => (3 : Nat)) := by
    simp [parts,active,original,FPIntra.original,TypedIntra.original,TypedIntra.source,ComputeKernel.surfaceBody,Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte,Source.dynamicRange?,String.reduceEq,beq_self_eq_true,entry,initialized,initial,evalOp_unfold,numeric,bop,natLt,natLe,
      Broadcast.leftIndex,Broadcast.rightIndex]
  simp only [step,hs,hd,ht,evalOp_unfold,Option.bind,Option.bind_some]
  change range (engine model) "j" 0 3 1 parts.body entry.val = some third.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (0 : Nat) < 3 from by decide,
    ↓reduceIte,first.property,bind,Option.bind_some]
  change range (engine model) "j" 1 3 1 parts.body first.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (1 : Nat) < 3 from by decide,
    ↓reduceIte,second.property,bind,Option.bind_some]
  change range (engine model) "j" 2 3 1 parts.body second.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (2 : Nat) < 3 from by decide,
    ↓reduceIte,third.property,bind,Option.bind_some]
  change range (engine model) "j" 3 3 1 parts.body third.val = _
  rw [range]
  simp

theorem source_parts : original.kernel.surfaceBody = active.before ++
    [.ifThen active.condition (parts.before ++
      [.forRangeDyn "j" parts.start parts.stop parts.stride parts.body])] := rfl

theorem condition_eval (skipped : Bool) :
    FP.Structural.evalOp (engine model) none active.condition (initialized skipped).val =
      some (fun _ => !skipped) := by
  cases skipped
  all_goals simp [active,original,FPIntra.original,TypedIntra.original,TypedIntra.source,ComputeKernel.surfaceBody,Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte,evalOp_unfold,numeric,bop,initialized,initial,natLe,natLt]

def final (skipped : Bool) : State ℝ := if skipped then (initialized Bool.true).val else third.val

theorem execution (skipped : Bool) :
    FP.Structural.exec (engine model) original.kernel (initial skipped) = some (final skipped) := by
  change run (engine model) original.kernel.surfaceBody (initial skipped) = _
  rw [source_parts,run_append,(initialized skipped).property]
  simp only [Option.bind_some]
  rw [run,step,condition_eval]
  cases skipped
  · simp only [Bool.not_false,↓reduceIte,Option.bind_some,run_append,entry.property,bind]
    rw [run,loop_execution]
    simp [run,final]
  · simp [final,run]

theorem domain_holds (skipped : Bool) :
    ∀ site ∈ original.domain, site.Holds model domain (initial skipped) := by
  simp [original,FPIntra.original,FPIntra.requirements,TypedIntra.original,TypedIntra.source,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Site.Holds,domain]

theorem values : (FP.Structural.exec (engine model) original.kernel (initial Bool.false)).map
    (fun t => (t.mem "A" 220,t.mem "A" 221,t.mem "A" 229,t.mem "A" 238,t.mem "A" 223,t.mem "A" 244)) =
    some (.mk .real 2,.mk .real 0,.mk .real 2,.mk .real 2,.mk .nat 99,.mk .nat 99) := by
  rw [execution]
  simp only [Option.map_some,final,Bool.false_eq_true,↓reduceIte,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,List.ofFn_succ,
      String.reduceEq,Cell.read,initial,engine,Algebra.withDefaultPrecision,model,BlockPtr.address,BlockPtr.inBounds,
      TileShape.indexToList]

/-- Streamed flat loads advance by K=3, while the cached matrix gate uses
its time stride 5. The explicit casts preserve the loaded values in this model. -/
theorem key_gate_layout :
    (first.val.regs .real [4] "b_k").map (fun v => v (0,PUnit.unit)) = some 76 ∧
    (second.val.regs .real [4] "b_k").map (fun v => v (0,PUnit.unit)) = some 79 ∧
    (third.val.regs .real [4] "b_gk").map (fun v => v (0,PUnit.unit)) = some 82 ∧
    (entry.val.regs .fp16 [4,4] "b_g").map (fun v => v (0,0,PUnit.unit)) = some 100 := by
  refine ⟨?_,?_,?_,?_⟩
  all_goals
    (conv_lhs => whnf)
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,List.ofFn_succ,
      String.reduceEq,Cell.read,initial,engine,Algebra.withDefaultPrecision,model,BlockPtr.address,BlockPtr.inBounds,
      TileShape.indexToList]
    try rfl

theorem skipped_memory : (final Bool.true).mem = (initial Bool.true).mem := rfl

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) (skipped : Bool) :
    FP.Structural.exec (engine model)
      (FPIntra.scaleFirst .fp16 .bf16 .fp16 "q" "k" "g" "A" 40 5 2 15 3 8 4 4).kernel
      (initial skipped) = some (final skipped) := by
  have he := FP.ProfiledRewrite.sound
    (FPIntra.intra_equiv .fp16 .bf16 .fp16 "q" "k" "g" "A" 40 5 2 15 3 8 4 4 R)
  exact (he ℝ model domain (model_satisfies_rules R) (initial skipped) (domain_holds skipped)).symm.trans
    (execution skipped)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedIntraExecution
