import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPMerge
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Complete typed merge executions in a consistent abstract scalar model:
three NK contributions at runtime head count three, partial rows and a bf16
store. The skipped branch and zero-slice loop also execute. These witnesses
establish a nonempty domain, not measurements of GPU rounding. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMergeExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- Algorithm fp32 loads use `.real` cells; the computation profile carries
fp32 precision. Other storage types keep their own cell tags. -/
def initial (enabled : Bool) : State ℝ where
  mem := fun r off => if r = "A" then .mk .real off else .mk .bf16 99
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 1 then 0 else if enabled then 1 else 2
  numPids := fun _ => 3
  undef := fun d _ _ => defaultValue d

def original (NK : Nat) := FPMerge.originalProgram .bf16 "A" "out" 5 4 2 NK

/-- Locate the literal-return branch and the dynamic NK loop by source structure. -/
def active (NK : Nat) : Source.Conditional :=
  (Source.conditional? (original NK).kernel.surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def parts (NK : Nat) : Source.DynamicRange :=
  (Source.dynamicRange? (active NK).yes "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def initialized (enabled : Bool) :
    {t : State ℝ // run (engine model) (active 0).before (initial enabled) = some t} := by
  cases enabled
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [parts,active,original,FPMerge.originalProgram,TypedMerge.original,ComputeKernel.surfaceBody,
      Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
    simp (config := {maxSteps := 50000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,initial]
    try rfl

def entry : {t : State ℝ // run (engine model) (parts 0).before (initialized Bool.true).val = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPMerge.originalProgram,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
  simp (config := {maxSteps := 50000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,initialized,initial]
  try rfl

def first : {t : State ℝ // run (engine model) (parts 3).body
    (entry.val.setReg "i_k" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPMerge.originalProgram,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
  simp (config := {maxSteps := 50000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,entry,initialized,initial]
  try rfl

def second : {t : State ℝ // run (engine model) (parts 3).body
    (first.val.setReg "i_k" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPMerge.originalProgram,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
  simp (config := {maxSteps := 50000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,first,entry,initialized,initial]
  try rfl

def third : {t : State ℝ // run (engine model) (parts 3).body
    (second.val.setReg "i_k" .nat [] (fun _ => 2)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,active,original,FPMerge.originalProgram,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
  simp (config := {maxSteps := 50000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,second,first,entry,initialized,initial]
  try rfl

def stored (threeSlices : Bool) : {t : State ℝ // run (engine model) (parts 0).after
    (if threeSlices then third.val else entry.val) = some t} := by
  cases threeSlices
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [parts,active,original,FPMerge.originalProgram,TypedMerge.original,ComputeKernel.surfaceBody,
      Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,Bool.false_eq_true,↓reduceIte]
    simp (config := {maxSteps := 50000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLe,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,integralFloorDiv,integralMod,third,second,first,entry,initialized,initial]
    try rfl

theorem loop_execution (threeSlices : Bool) :
    step (engine model) (.forRangeDyn "i_k" (parts (if threeSlices then 3 else 0)).start
      (parts (if threeSlices then 3 else 0)).stop (parts (if threeSlices then 3 else 0)).stride (parts 3).body)
      entry.val = some (if threeSlices then third.val else entry.val) := by
  have hs (n : Nat) : (parts n).start = .constNat 0 := rfl
  have hd (n : Nat) : (parts n).stride = .constNat 1 := rfl
  have hn (n : Nat) : (parts n).stop = .constNat n := rfl
  simp only [step,hs,hd,hn,evalOp_unfold,Option.bind,Option.bind_some]
  cases threeSlices
  · change range (engine model) "i_k" 0 0 1 (parts 3).body entry.val = some entry.val
    rw [range]
    simp
  · change range (engine model) "i_k" 0 3 1 (parts 3).body entry.val = some third.val
    rw [range]
    simp only [show (1 : Nat) ≠ 0 from by decide,show (0 : Nat) < 3 from by decide,
      ↓reduceIte,first.property,bind,Option.bind_some]
    change range (engine model) "i_k" 1 3 1 (parts 3).body first.val = _
    rw [range]
    simp only [show (1 : Nat) ≠ 0 from by decide,show (1 : Nat) < 3 from by decide,
      ↓reduceIte,second.property,bind,Option.bind_some]
    change range (engine model) "i_k" 2 3 1 (parts 3).body second.val = _
    rw [range]
    simp only [show (1 : Nat) ≠ 0 from by decide,show (2 : Nat) < 3 from by decide,
      ↓reduceIte,third.property,bind,Option.bind_some]
    change range (engine model) "i_k" 3 3 1 (parts 3).body third.val = _
    rw [range]
    simp

theorem source_parts (NK : Nat) : (original NK).kernel.surfaceBody = (active 0).before ++
    [.ifThen (active NK).condition ((parts 0).before ++
      [.forRangeDyn "i_k" (parts NK).start (parts NK).stop (parts NK).stride (parts 3).body] ++ (parts 0).after)] := rfl

theorem condition_eval (NK : Nat) (enabled : Bool) :
    FP.Structural.evalOp (engine model) none (active NK).condition (initialized enabled).val =
      some (fun _ => enabled) := by
  cases enabled
  all_goals simp [active,original,FPMerge.originalProgram,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,evalOp_unfold,numeric,bop,initialized,initial,natLe,natLt]

def final (enabled threeSlices : Bool) : State ℝ :=
  if enabled then (stored threeSlices).val else (initialized Bool.false).val

theorem execution (enabled threeSlices : Bool) :
    FP.Structural.exec (engine model) (original (if threeSlices then 3 else 0)).kernel (initial enabled) =
      some (final enabled threeSlices) := by
  change run (engine model) (original _).kernel.surfaceBody (initial enabled) = _
  rw [source_parts,run_append,(initialized enabled).property]
  simp only [Option.bind_some]
  rw [run,step,condition_eval]
  cases enabled
  · simp [run,final]
  · simp only [↓reduceIte,Option.bind_some,run_append,entry.property,bind]
    rw [run,loop_execution]
    simp only [Option.bind_some,run,final,↓reduceIte]
    simpa [bind] using (stored threeSlices).property

/-- Successful full execution, now composed from prefix, loop steps and store. -/
def result (enabled threeSlices : Bool) :
    {t : State ℝ // FP.Structural.exec (engine model) (original (if threeSlices then 3 else 0)).kernel (initial enabled) = some t} :=
  ⟨final enabled threeSlices,execution enabled threeSlices⟩

theorem domain_holds (NK : Nat) (enabled : Bool) :
    ∀ site ∈ (original NK).domain, site.Holds model domain (initial enabled) := by
  simp [original,FPMerge.originalProgram,FPMerge.requirements,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Site.Holds,domain]

theorem values (threeSlices : Bool) :
    (FP.Structural.exec (engine model) (original (if threeSlices then 3 else 0)).kernel (initial Bool.true)).map
      (fun t => (t.mem "out" 36,t.mem "out" 37,t.mem "out" 38)) =
      some (.mk .bf16 (if threeSlices then 144 else 0),.mk .bf16 (if threeSlices then 147 else 0),.mk .bf16 99) := by
  rw [(result Bool.true threeSlices).property]
  cases threeSlices
  all_goals
    simp only [Option.map_some,Bool.false_eq_true,↓reduceIte,Option.some.injEq,Prod.mk.injEq]
    repeat' apply And.intro
    all_goals
      (conv_lhs => whnf) <;>
      norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
        TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,List.ofFn_succ,String.reduceEq,Cell.read,initial,engine,Algebra.withDefaultPrecision,model,BlockPtr.address,BlockPtr.inBounds,TileShape.indexToList]

theorem skipped (threeSlices : Bool) : (result Bool.false threeSlices).val.mem = (initial Bool.false).mem := by
  cases threeSlices <;> rfl

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) (enabled threeSlices : Bool) :
    FP.Structural.exec (engine model)
      (FPMerge.loadFirstProgram .bf16 "A" "out" 5 4 2 (if threeSlices then 3 else 0)).kernel (initial enabled) =
      some (result enabled threeSlices).val := by
  have he := FP.ProfiledRewrite.sound
    (FPMerge.merge_equiv .bf16 "A" "out" 5 4 2 (if threeSlices then 3 else 0) R)
  exact (he ℝ model domain (model_satisfies_rules R) (initial enabled) (domain_holds _ enabled)).symm.trans
    (result enabled threeSlices).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMergeExecution
