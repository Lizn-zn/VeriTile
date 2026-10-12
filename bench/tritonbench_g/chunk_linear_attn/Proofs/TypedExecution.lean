import bench.tritonbench_g.chunk_linear_attn.Proofs.FP
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Nonempty complete forwardH execution with two iterations, partial tails,
nonzero program IDs and explicit storage. Dot returns one in this abstract
model; this witness is neither a GPU test nor a mathematical contraction. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedForwardH
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
    if r = "h0" then .mk .bf16 1
    else if r = "h" ∨ r = "ht" then .mk .fp16 99
    else .mk .fp16 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := FPProof.forwardH.original .fp16 .bf16 .fp16 "k" "v" "h" "h0" "ht" 9 3 1 9 3 1 18 3 3 3 3 2 2 2 2 Bool.true Bool.true

def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "i_t").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.forwardH.original,ChunkLinearAttn.forwardH,ChunkLinearAttn.forwardHSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

def first : {t : State ℝ // run (engine model) parts.body (entry.val.setReg "i_t" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.forwardH.original,ChunkLinearAttn.forwardH,ChunkLinearAttn.forwardHSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod,entry]
  try rfl

def second : {t : State ℝ // run (engine model) parts.body (first.val.setReg "i_t" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.forwardH.original,ChunkLinearAttn.forwardH,ChunkLinearAttn.forwardHSource,
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
  simp only [parts,original,FPProof.forwardH.original,ChunkLinearAttn.forwardH,ChunkLinearAttn.forwardHSource,
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
    [.forRangeDyn "i_t" parts.start parts.stop parts.stride parts.body] ++ parts.after := rfl

theorem loop_execution : step (engine model) (.forRangeDyn "i_t" parts.start parts.stop parts.stride parts.body) entry.val = some second.val := by
  have hs : parts.start = .constNat 0 := rfl
  have ht : parts.stop = .constNat 2 := rfl
  have hd : parts.stride = .constNat 1 := rfl
  simp only [step,hs,ht,hd,evalOp_unfold,numeric,bop,Option.bind,Option.bind_some]
  change range (engine model) "i_t" 0 2 1 parts.body entry.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (0 : Nat) < 2 from by decide,
    ↓reduceIte,first.property,bind,Option.bind_some]
  change range (engine model) "i_t" 1 2 1 parts.body first.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (1 : Nat) < 2 from by decide,
    ↓reduceIte,second.property,bind,Option.bind_some]
  change range (engine model) "i_t" 2 2 1 parts.body second.val = _
  rw [range]
  simp

theorem execution : FP.Structural.exec (engine model) original.kernel initial = some finish.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  simp only [source_parts,List.append_assoc,run_append,entry.property,Option.bind_some,List.singleton_append]
  rw [run]
  simp only [loop_execution,bind,Option.bind_some,finish.property]

theorem domain_holds : ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original,FPProof.forwardH.original,FPProof.forwardHRequirements,ChunkLinearAttn.forwardH,ChunkLinearAttn.forwardHSource,
    ComputeKernel.surfaceBody,Source.prefixBefore,Source.dynamicRange?,Site.Holds,domain]

theorem values :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "h" 26,t.mem "h" 35,t.mem "ht" 17,t.mem "ht" 16)) = some (.mk .fp16 1,.mk .fp16 2,.mk .fp16 3,.mk .fp16 99) := by
  rw [execution]
  simp only [Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) (FPProof.forwardH.dotLeft .fp16 .bf16 .fp16 "k" "v" "h" "h0" "ht" 9 3 1 9 3 1 18 3 3 3 3 2 2 2 2 Bool.true Bool.true).kernel initial = some finish.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.forwardH.equivalent .fp16 .bf16 .fp16 "k" "v" "h" "h0" "ht" 9 3 1 9 3 1 18 3 3 3 3 2 2 2 2 Bool.true Bool.true R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_holds).symm.trans execution

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedForwardH

/-! Nonempty complete forwardO execution with two iterations, partial tails,
nonzero program IDs and explicit storage. Dot returns one in this abstract
model; this witness is neither a GPU test nor a mathematical contraction. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedForwardO
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
    if r = "v" then .mk .bf16 1
    else if r = "o" then .mk .fp16 99
    else .mk .fp16 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := FPProof.forwardO.original .fp16 .bf16 .fp16 "q" "k" "v" "h" "o" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2

def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.forwardO.original,ChunkLinearAttn.forwardO,ChunkLinearAttn.forwardOSource,
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
  simp only [parts,original,FPProof.forwardO.original,ChunkLinearAttn.forwardO,ChunkLinearAttn.forwardOSource,
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
  simp only [parts,original,FPProof.forwardO.original,ChunkLinearAttn.forwardO,ChunkLinearAttn.forwardOSource,
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
  simp only [parts,original,FPProof.forwardO.original,ChunkLinearAttn.forwardO,ChunkLinearAttn.forwardOSource,
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
  simp [original,FPProof.forwardO.original,FPProof.forwardORequirements,ChunkLinearAttn.forwardO,ChunkLinearAttn.forwardOSource,
    ComputeKernel.surfaceBody,Source.prefixBefore,Source.dynamicRange?,Site.Holds,domain]

theorem values :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "o" 17,t.mem "o" 16,t.mem "q" 17)) = some (.mk .fp16 6,.mk .fp16 99,.mk .fp16 1) := by
  rw [execution]
  simp only [Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) (FPProof.forwardO.scaleFirst .fp16 .bf16 .fp16 "q" "k" "v" "h" "o" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2).kernel initial = some finish.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.forwardO.equivalent .fp16 .bf16 .fp16 "q" "k" "v" "h" "o" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2 R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_holds).symm.trans execution

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedForwardO

/-! Nonempty complete backwardH execution with two iterations, partial tails,
nonzero program IDs and explicit storage. Dot returns one in this abstract
model; this witness is neither a GPU test nor a mathematical contraction. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedBackwardH
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
    if r = "do" then .mk .bf16 1
    else if r = "dh" then .mk .fp16 99
    else .mk .fp16 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := FPProof.backwardH.original .fp16 .bf16 .fp16 "q" "do" "dh" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2 2

def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "j").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.backwardH.original,ChunkLinearAttn.backwardH,ChunkLinearAttn.backwardHSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

def first : {t : State ℝ // run (engine model) parts.body (entry.val.setReg "j" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.backwardH.original,ChunkLinearAttn.backwardH,ChunkLinearAttn.backwardHSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod,entry]
  try rfl

def second : {t : State ℝ // run (engine model) parts.body (first.val.setReg "j" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.backwardH.original,ChunkLinearAttn.backwardH,ChunkLinearAttn.backwardHSource,
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
  simp only [parts,original,FPProof.backwardH.original,ChunkLinearAttn.backwardH,ChunkLinearAttn.backwardHSource,
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
    [.forRange "j" 0 2 1 parts.body] ++ parts.after := rfl

theorem loop_execution : step (engine model) (.forRange "j" 0 2 1 parts.body) entry.val = some second.val := by
  simp only [step]
  change range (engine model) "j" 0 2 1 parts.body entry.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (0 : Nat) < 2 from by decide,
    ↓reduceIte,first.property,bind,Option.bind_some]
  change range (engine model) "j" 1 2 1 parts.body first.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (1 : Nat) < 2 from by decide,
    ↓reduceIte,second.property,bind,Option.bind_some]
  change range (engine model) "j" 2 2 1 parts.body second.val = _
  rw [range]
  simp

theorem execution : FP.Structural.exec (engine model) original.kernel initial = some finish.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  simp only [source_parts,List.append_assoc,run_append,entry.property,Option.bind_some,List.singleton_append]
  rw [run]
  simp only [loop_execution,bind,Option.bind_some,finish.property]

theorem domain_holds : ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original,FPProof.backwardH.original,FPProof.backwardHRequirements,ChunkLinearAttn.backwardH,ChunkLinearAttn.backwardHSource,
    ComputeKernel.surfaceBody,Source.prefixBefore,Source.dynamicRange?,Site.Holds,domain]

theorem values :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "dh" 26,t.mem "dh" 35,t.mem "dh" 34)) = some (.mk .fp16 1,.mk .fp16 0,.mk .fp16 99) := by
  rw [execution]
  simp only [Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) (FPProof.backwardH.dotLeft .fp16 .bf16 .fp16 "q" "do" "dh" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2 2).kernel initial = some finish.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.backwardH.equivalent .fp16 .bf16 .fp16 "q" "do" "dh" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2 2 R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_holds).symm.trans execution

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedBackwardH

/-! Nonempty complete backwardQKV execution with two iterations, partial tails,
nonzero program IDs and explicit storage. Dot returns one in this abstract
model; this witness is neither a GPU test nor a mathematical contraction. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedBackwardQKV
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
    if r = "dq" ∨ r = "dk" ∨ r = "dv" then .mk .bf16 99
    else .mk .bf16 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := FPProof.backwardQKV.original .bf16 "q" "k" "v" "h" "do" "dh" "dq" "dk" "dv" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2 2

def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "i_v").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.backwardQKV.original,ChunkLinearAttn.backwardQKV,ChunkLinearAttn.backwardQKVSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

def first : {t : State ℝ // run (engine model) parts.body (entry.val.setReg "i_v" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.backwardQKV.original,ChunkLinearAttn.backwardQKV,ChunkLinearAttn.backwardQKVSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  dsimp [Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod,entry]
  try rfl

def second : {t : State ℝ // run (engine model) parts.body (first.val.setReg "i_v" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,original,FPProof.backwardQKV.original,ChunkLinearAttn.backwardQKV,ChunkLinearAttn.backwardQKVSource,
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
  simp only [parts,original,FPProof.backwardQKV.original,ChunkLinearAttn.backwardQKV,ChunkLinearAttn.backwardQKVSource,
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
    [.forRangeDyn "i_v" parts.start parts.stop parts.stride parts.body] ++ parts.after := rfl

theorem loop_execution : step (engine model) (.forRangeDyn "i_v" parts.start parts.stop parts.stride parts.body) entry.val = some second.val := by
  have hs : parts.start = .constNat 0 := rfl
  have ht : parts.stop = .div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat 3) (.constNat 2)) (.constNat 1)) (.constNat 2) := rfl
  have hd : parts.stride = .constNat 1 := rfl
  simp only [step,hs,ht,hd,evalOp_unfold,numeric,bop,Option.bind,Option.bind_some]
  change range (engine model) "i_v" 0 2 1 parts.body entry.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (0 : Nat) < 2 from by decide,
    ↓reduceIte,first.property,bind,Option.bind_some]
  change range (engine model) "i_v" 1 2 1 parts.body first.val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide,show (1 : Nat) < 2 from by decide,
    ↓reduceIte,second.property,bind,Option.bind_some]
  change range (engine model) "i_v" 2 2 1 parts.body second.val = _
  rw [range]
  simp

theorem execution : FP.Structural.exec (engine model) original.kernel initial = some finish.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  simp only [source_parts,List.append_assoc,run_append,entry.property,Option.bind_some,List.singleton_append]
  rw [run]
  simp only [loop_execution,bind,Option.bind_some,finish.property]

theorem domain_holds : ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original,FPProof.backwardQKV.original,FPProof.backwardQKVRequirements,ChunkLinearAttn.backwardQKV,ChunkLinearAttn.backwardQKVSource,
    ComputeKernel.surfaceBody,Source.prefixBefore,Source.dynamicRange?,Site.Holds,domain]

theorem values :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "dq" 17,t.mem "dk" 17,t.mem "dv" 35,t.mem "dv" 32)) = some (.mk .bf16 5,.mk .bf16 3,.mk .bf16 2,.mk .bf16 99) := by
  rw [execution]
  simp only [Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq,transposeValues]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) (FPProof.backwardQKV.dotLeft .bf16 "q" "k" "v" "h" "do" "dh" "dq" "dk" "dv" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2 2).kernel initial = some finish.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.backwardQKV.equivalent .bf16 "q" "k" "v" "h" "do" "dh" "dq" "dk" "dv" 9 3 1 9 3 1 18 3 2 3 3 3 2 2 2 2 R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_holds).symm.trans execution

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.TypedBackwardQKV

#guard_msgs (drop info) in
#auditModuleAxioms
