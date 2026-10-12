import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPInter
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Nonempty complete typed executions for the inter-subblock producer. The
active case has two K blocks, mixed input storage, fp16 gate subtraction,
partial rows/K lanes and a bf16 output. Row stride differs from K, retaining
the source's separate gate-reference layout. Both early returns and the empty
K loop also execute. The consistent abstract model returns one for dot/exp;
these witnesses are not GPU measurements or the real mathematical formula. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedInterExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxRecDepth 28000
set_option maxHeartbeats 20000000
set_option linter.unusedSimpArgs false

inductive Mode where
  | active | triangularSkip | timeSkip
  deriving DecidableEq

def initial (mode : Mode) : State ℝ where
  mem := fun r off => if r = "q" then .mk .fp16 1
    else if r = "k" then .mk .bf16 1
    else if r = "g" then .mk .fp16 off
    else .mk .bf16 99
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then (if mode = .timeSkip then 2 else 1)
    else if ax = 1 then (if mode = .triangularSkip then 3 else 2) else 1
  numPids := fun _ => 3
  undef := fun d _ _ => defaultValue d

def original (empty : Bool) :=
  FPInter.original .fp16 .bf16 .fp16 .bf16 "q" "k" "g" "A" 40 5 2 7 (if empty then 0 else 3) 4 2 2 2

def outerBranch (empty : Bool) : Source.Conditional :=
  (Source.conditional? (original empty).kernel.surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def innerBranch (empty : Bool) : Source.Conditional :=
  (Source.conditional? (outerBranch empty).yes (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def parts (empty : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (innerBranch empty).yes "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def initialized (mode : Mode) :
    {t : State ℝ // run (engine model) (outerBranch Bool.false).before (initial mode) = some t} := by
  cases mode
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [outerBranch,original,FPInter.original,TypedInter.original,TypedInter.source,ComputeKernel.surfaceBody,
      Source.conditional?,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
    simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
      initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod]
    try rfl

def entry : {t : State ℝ // run (engine model) (parts Bool.false).before (initialized .active).val = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,innerBranch,outerBranch,original,FPInter.original,TypedInter.original,TypedInter.source,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,Bool.false_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,engine,Algebra.withDefaultPrecision,
    resolvePrecision,model,ComputeDType.eraseDType,ofFloat,toFloat]
  rfl

def first : {t : State ℝ // run (engine model) (parts Bool.false).body
    (entry.val.setReg "i_k" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,innerBranch,outerBranch,original,FPInter.original,TypedInter.original,TypedInter.source,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,Bool.false_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    entry,initialized,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

def second : {t : State ℝ // run (engine model) (parts Bool.false).body
    (first.val.setReg "i_k" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,innerBranch,outerBranch,original,FPInter.original,TypedInter.original,TypedInter.source,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,Bool.false_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    first,entry,initialized,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

def loopResult (empty : Bool) : State ℝ := if empty then entry.val else second.val

def finish (empty : Bool) :
    {t : State ℝ // run (engine model) (parts empty).after (loopResult empty) = some t} := by
  cases empty
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [parts,innerBranch,outerBranch,original,FPInter.original,TypedInter.original,TypedInter.source,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,Bool.false_eq_true,↓reduceIte]
    dsimp [Source.dynamicRange?,Option.map,Option.getD]
    simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
      loopResult,second,first,entry,initialized,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod]
    try rfl

theorem loop_execution (empty : Bool) :
    step (engine model) (.forRangeDyn "i_k" (parts empty).start (parts empty).stop
      (parts empty).stride (parts empty).body) entry.val = some (loopResult empty) := by
  have hs : (parts empty).start = .constNat 0 := by cases empty <;> rfl
  have ht : (parts empty).stop = .div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat (if empty then 0 else 3)) (.constNat 2)) (.constNat 1))
      (.constNat 2) := by cases empty <;> rfl
  have hd : (parts empty).stride = .constNat 1 := by cases empty <;> rfl
  simp only [step,hs,ht,hd,evalOp_unfold,numeric,bop,Option.bind,Option.bind_some]
  cases empty
  · change range (engine model) "i_k" 0 2 1 (parts Bool.false).body entry.val = some second.val
    rw [range]
    simp only [show (1 : Nat) ≠ 0 from by decide,show (0 : Nat) < 2 from by decide,
      ↓reduceIte,first.property,bind,Option.bind_some]
    change range (engine model) "i_k" 1 2 1 (parts Bool.false).body first.val = _
    rw [range]
    simp only [show (1 : Nat) ≠ 0 from by decide,show (1 : Nat) < 2 from by decide,
      ↓reduceIte,second.property,bind,Option.bind_some]
    change range (engine model) "i_k" 2 2 1 (parts Bool.false).body second.val = _
    rw [range]
    simp
  · change range (engine model) "i_k" 0 0 1 (parts Bool.true).body entry.val = some entry.val
    rw [range]
    simp

theorem source_parts (empty : Bool) : (original empty).kernel.surfaceBody =
    (outerBranch Bool.false).before ++ [.ifThen (outerBranch empty).condition
      [.ifThen (innerBranch empty).condition ((parts Bool.false).before ++
        [.forRangeDyn "i_k" (parts empty).start (parts empty).stop (parts empty).stride (parts empty).body] ++
        (parts empty).after)]] := by cases empty <;> rfl

theorem outer_eval (empty : Bool) (mode : Mode) :
    FP.Structural.evalOp (engine model) none (outerBranch empty).condition (initialized mode).val =
      some (fun _ => decide (mode ≠ .timeSkip)) := by
  cases empty <;> cases mode
  all_goals simp [outerBranch,original,FPInter.original,TypedInter.original,TypedInter.source,ComputeKernel.surfaceBody,
    Source.conditional?,evalOp_unfold,numeric,bop,initialized,initial,natLe,natLt]
  all_goals
    funext i
    cases i
    norm_num [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem inner_eval (empty : Bool) :
    FP.Structural.evalOp (engine model) none (innerBranch empty).condition (initialized .active).val = some (fun _ => Bool.true) := by
  cases empty
  all_goals simp [innerBranch,outerBranch,original,FPInter.original,TypedInter.original,TypedInter.source,
    ComputeKernel.surfaceBody,Source.conditional?,evalOp_unfold,numeric,bop,initialized,initial,natLe,natLt]
  all_goals
    funext i
    cases i
    norm_num [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem active_execution (empty : Bool) :
    FP.Structural.exec (engine model) (original empty).kernel (initial .active) = some (finish empty).val := by
  change run (engine model) (original empty).kernel.surfaceBody (initial .active) = _
  rw [source_parts,run_append,(initialized .active).property]
  simp only [Option.bind_some]
  rw [run,step,outer_eval]
  simp only [ne_eq,reduceCtorEq,not_false_eq_true,decide_true,↓reduceIte,Option.bind_some]
  rw [run,step,inner_eval]
  simp only [Option.bind_some,↓reduceIte,List.append_assoc,run_append,entry.property]
  simp only [bind,Option.bind_some,List.singleton_append]
  rw [run,loop_execution]
  simp only [bind,Option.bind_some,(finish empty).property,run,↓reduceIte]

def result (mode : Mode) (empty : Bool) :
    {t : State ℝ // FP.Structural.exec (engine model) (original empty).kernel (initial mode) = some t} := by
  cases mode with
  | active => exact ⟨(finish empty).val,active_execution empty⟩
  | triangularSkip | timeSkip =>
    cases empty
    all_goals
      refine ⟨?_,?_⟩
      swap
      simp only [original,FPInter.original,TypedInter.original,TypedInter.source,FP.Structural.exec,
        ComputeKernel.surfaceBody,Bool.false_eq_true,↓reduceIte]
      simp (config := {maxSteps := 200000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
        engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
        initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod]
      try rfl

theorem domain_holds (mode : Mode) (empty : Bool) :
    ∀ site ∈ (original empty).domain, site.Holds model domain (initial mode) := by
  simp [original,FPInter.original,FPInter.requirements,TypedInter.original,TypedInter.source,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Site.Holds,domain]

theorem values (empty : Bool) :
    (FP.Structural.exec (engine model) (original empty).kernel (initial .active)).map
      (fun t => (t.mem "A" 52,t.mem "A" 53,t.mem "A" 54)) =
      some (.mk .bf16 (if empty then 0 else 2),.mk .bf16 (if empty then 0 else 2),.mk .bf16 99) := by
  rw [(result .active empty).property]
  cases empty
  all_goals
    simp only [Option.map_some,Bool.false_eq_true,↓reduceIte,Option.some.injEq,Prod.mk.injEq]
    repeat' apply And.intro
    all_goals
      (conv_lhs => whnf) <;>
      norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
        TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,List.ofFn_succ,
        String.reduceEq,Cell.read,initial,engine,Algebra.withDefaultPrecision,model,BlockPtr.address,BlockPtr.inBounds,
        TileShape.indexToList]

/-- The flat reference and strided matrix load read different actual cells.
The gate array stores its address, so the result exposes a layout mismatch. -/
theorem reference_layout :
    (first.val.regs .fp16 [2] "b_gn").map (fun v => v (0,PUnit.unit)) = some 58 ∧
    (second.val.regs .fp16 [2] "b_gn").map (fun v => v (0,PUnit.unit)) = some 60 ∧
    (first.val.regs .fp16 [2,2] "b_g").map (fun v => v (0,0,PUnit.unit)) = some 70 := by
  refine ⟨?_,?_,?_⟩ <;> rfl

theorem triangular_skipped (empty : Bool) :
    (result .triangularSkip empty).val.mem = (initial .triangularSkip).mem := by
  cases empty <;> rfl

theorem time_skipped (empty : Bool) :
    (result .timeSkip empty).val.mem = (initial .timeSkip).mem := by
  cases empty <;> rfl

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) (mode : Mode) (empty : Bool) :
    FP.Structural.exec (engine model)
      (FPInter.dotFirst .fp16 .bf16 .fp16 .bf16 "q" "k" "g" "A" 40 5 2 7 (if empty then 0 else 3) 4 2 2 2).kernel
      (initial mode) = some (result mode empty).val := by
  have he := FP.ProfiledRewrite.sound
    (FPInter.inter_equiv .fp16 .bf16 .fp16 .bf16 "q" "k" "g" "A" 40 5 2 7 (if empty then 0 else 3) 4 2 2 2 R)
  exact (he ℝ model domain (model_satisfies_rules R) (initial mode) (domain_holds mode empty)).symm.trans
    (result mode empty).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedInterExecution
