import bench.tritonbench_g.rms_norm_triton.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Execution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-! Exact-real numeric execution backend. The actual source operations retain
mathematical sqrt, sum, division and multiplication; casts are identities. This
is a nonempty execution witness, not an assertion about IEEE rounding. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  atomicAdd := fun _ => some WithBot.realAdd
  unary := fun _ op a => match op with
    | .exp2 => WithBot.realExp2 a
    | .log2 => WithBot.realLog2 a
    | .sqrt => WithBot.realSqrt a
    | _ => some 0
  compareLe := fun _ _ => some (fun a b => decide (a ≤ b))
  compareLt := fun _ _ => some (fun a b => decide (a < b))
  compareEq := fun _ _ => some (fun a b => decide (a = b))
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v => ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data
  reduceSumTyped := fun _ d {_} ax keep v =>
    some (fun i => ofFloat d ((Tile.reduceSum ax keep ⟨fun j => toFloat d (v j)⟩).data i))
  reduceMaxTyped := fun _ d {_} ax keep v =>
    some (fun i => ofFloat d (((Tile.reduceMax ax keep ⟨fun j => toFloat d (v j)⟩).getD ⟨fun _ => none⟩).data i))
  dot := fun _ {batch} {_} {_} {_} a b => some (Tile.dot batch ⟨a⟩ ⟨b⟩).data
  dotTyped := fun _ l r out _ {batch} {_} {_} {_} a b =>
    some (fun i => ofFloat out ((Tile.dot batch ⟨fun j => toFloat l (a j)⟩
      ⟨fun j => toFloat r (b j)⟩).data i))

def initial : State (WithBot ℝ) where
  mem := fun r o => if r="X" then .mk .fp16 (some (if o=3 then 3 else 97))
    else if r="W" then .mk .bf16 (some (if o=0 then 2 else 91))
    else .mk .fp16 (some 99)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- One active lane and one masked tail lane, nonzero row, noncontiguous stride. -/
def executed (commuted : Bool) :
    {t : State (WithBot ℝ) // FP.Structural.exec (engine model)
      (typed .fp16 .bf16 commuted "Y" "X" "W" 6 2 3 2 1 7 2) initial = some t} := by
  cases commuted
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (maxSteps := 300000) [FP.Structural.exec,typed,ComputeKernel.surfaceBody,run,step,
      FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,model,
      ComputeDType.eraseDType,numeric,bop,numericLe,numericLt,numericEq,natLt,
      initial,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    try rfl

theorem output_values (commuted : Bool) :
    (executed commuted).val.mem "Y" 6 = Cell.mk .fp16 (some (3/2 : ℝ)) ∧
    (executed commuted).val.mem "Y" 8 = Cell.mk .fp16 (some 99) := by
  have hmul : (3 : WithBot ℝ) * 3 = some 9 := by
    change some ((3 : ℝ) * 3) = some 9
    norm_num
  have hmulr (r : ℝ) : (3 : WithBot ℝ) * (r : WithBot ℝ) = some (3*r) := rfl
  have hmulout (r : ℝ) : (r : WithBot ℝ) * (2 : WithBot ℝ) = some (r*2) := rfl
  cases commuted
  all_goals simp [executed,initial,model,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.insertAxisIndex,Fin.sum_univ_succ,
    WithBot.realAdd,WithBot.realMul,WithBot.realDiv,WithBot.realSqrt,
    FP.Structural.store,State.write,TileShape.allIndices,List.finRange_succ,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  all_goals norm_num [WithBot.some_eq_coe,WithBot.none_eq_bot,
    ←WithBot.coe_add,Option.map₂,Option.map,Option.bind]
  all_goals erw [Fin.sum_univ_two]
  all_goals norm_num [WithBot.some_eq_coe,WithBot.none_eq_bot,
    ←WithBot.coe_add,Option.map₂,Option.map,Option.bind]
  all_goals erw [hmul]
  all_goals norm_num [hmul,WithBot.some_eq_coe,WithBot.none_eq_bot,
    ←WithBot.coe_add,Option.map₂,Option.map,Option.bind]
  all_goals erw [hmulr]
  all_goals norm_num [numeric]
  all_goals erw [hmulout]
  all_goals norm_num
  all_goals exact ⟨rfl,rfl⟩

theorem full_numerical_execution (commuted : Bool) :
    ∃ t, FP.Structural.exec (engine model)
      (typed .fp16 .bf16 commuted "Y" "X" "W" 6 2 3 2 1 7 2) initial = some t ∧
      t.mem "Y" 6 = Cell.mk .fp16 (some (3/2 : ℝ)) ∧
      t.mem "Y" 8 = Cell.mk .fp16 (some 99) := by
  exact ⟨(executed commuted).val,(executed commuted).property,output_values commuted⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Execution
