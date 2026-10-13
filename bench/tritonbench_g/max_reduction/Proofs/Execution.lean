import bench.tritonbench_g.max_reduction.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MaxReduction.Execution
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Exact finite nonnegative payloads witness successful source execution.
This interpreter instance is not an IEEE implementation or a rewrite axiom. -/
private def numericIndexed (dtype : FloatDType) {shape : TileShape}
    (ax : Fin shape.length) (v : Values Nat dtype.toTileDType shape) :
    Option (Values Nat dtype.toTileDType (TileShape.eraseAxis shape ax) ×
      Values Nat .nat (TileShape.eraseAxis shape ax)) :=
  if h : 0 < TileShape.axisDim shape ax then
    let best := fun out => (List.finRange (TileShape.axisDim shape ax)).foldl
      (fun best next => if toFloat dtype (v (TileShape.insertAxisIndex shape ax out best)) <
          toFloat dtype (v (TileShape.insertAxisIndex shape ax out next)) then next else best)
      ⟨0,h⟩
    some ((fun out => v (TileShape.insertAxisIndex shape ax out (best out))),
      fun out => (best out).val)
  else none

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
  indexedMax := fun _ dtype => numericIndexed dtype

private def channel : Storage → FloatDType
  | .fp32 => .real | .fp16 => .fp16 | .bf16 => .bf16
  | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .fp64 => .real

private def initial (storage : Storage) : State Nat where
  mem := fun r o => if r = "inp" then
    .mk (channel storage).toTileDType (ofFloat (channel storage) (if o = 0 then 7 else 9))
    else .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private noncomputable def result (storage : Storage) (v : Bool) : Option (Nat × Int) :=
  (exec model (dimensionSource storage v "inp" "value" "index" 1 2 1 1 2)
    (initial storage)).map (fun s =>
      (toFloat (channel storage) ((s.mem "value" 0).read (channel storage).toTileDType),
       (s.mem "index" 0).read .int))

/-- Every dimension-source profile and both candidates execute successfully,
returning the actual maximum nine and its signed-storage index one. -/
theorem dimension_nonempty_numeric_witness (storage : Storage) (v : Bool) :
    result storage v = some (9, 1) := by
  cases storage <;> cases v
  all_goals simp [result, dimensionSource, ComputeKernel.surfaceBody,
    FP.Structural.exec, run, step]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction.Execution
