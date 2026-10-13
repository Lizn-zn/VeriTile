import bench.tritonbench_g.index_select_cat.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat.Execution
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

private def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def channel : Storage → FloatDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .f8e4 => .f8e4
  | .f8e5 => .f8e5

private def initial (input output : Storage) : State Nat where
  mem := fun r o => if r = "idx" then .mk .int 1
    else if r = "inp" then .mk (channel input).toTileDType (ofFloat (channel input) (7+o))
    else .mk (channel output).toTileDType (ofFloat (channel output) 99)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private noncomputable def result (input output : Storage) (index : IndexStorage) (v : Bool) :=
  (exec model (source input output index v "out" "idx" "inp" 2 2 3 1 2 3)
    (initial input output)).map (fun s =>
      (toFloat (channel output) ((s.mem "out" 3).read (channel output).toTileDType),
       toFloat (channel output) ((s.mem "out" 4).read (channel output).toTileDType),
       toFloat (channel output) ((s.mem "out" 5).read (channel output).toTileDType)))

/-- Complete nonempty execution with signed duplicate indices and a masked
padding column. Both destination rows receive the repeated source row; the masked column remains untouched. -/
theorem duplicate_gather_executes (input output : Storage) (index : IndexStorage) (v : Bool) :
    result input output index v = some (10,11,99) := by
  cases input <;> cases output <;> cases index <;> cases v
  all_goals simp [result,source,ComputeKernel.surfaceBody,FP.Structural.exec,run,step]
  all_goals rfl


private def aliasInitial (storage : Storage) : State Nat where
  mem := fun r o => if r = "idx" then .mk .int (if o = 0 then 1 else 0)
    else .mk (channel storage).toTileDType (ofFloat (channel storage) (7+o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private noncomputable def aliasResult (storage : Storage) (index : IndexStorage) (v : Bool) :=
  (exec model (source storage storage index v "inp" "idx" "inp" 2 2 3 1 2 3)
    (aliasInitial storage)).map (fun s =>
      (toFloat (channel storage) ((s.mem "inp" 0).read (channel storage).toTileDType),
       toFloat (channel storage) ((s.mem "inp" 3).read (channel storage).toTileDType),
       toFloat (channel storage) ((s.mem "inp" 2).read (channel storage).toTileDType)))

/-- Aliased source/output swaps rows from the original snapshot. This detects
an incorrect interleaving of loads and stores, and retains the padding cell. -/
theorem aliased_gather_executes (storage : Storage) (index : IndexStorage) (v : Bool) :
    aliasResult storage index v = some (10,7,9) := by
  cases storage <;> cases index <;> cases v
  all_goals simp [aliasResult,source,ComputeKernel.surfaceBody,FP.Structural.exec,run,step]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.IndexSelectCat.Execution
