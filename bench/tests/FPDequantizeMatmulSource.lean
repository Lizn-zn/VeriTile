import bench.tritonbench_g.dequantize_matmul.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def outputType : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def loadType? {d : TileDType} {sh : TileShape} : Op d sh → Option TileDType
  | .load dt _ _ _ => some dt
  | .castFloat _ _ e => loadType? e
  | _ => none

def assignedLoad? (code : List ComputeStmt) (name : RegName) : Option TileDType :=
  match code with
  | [] => none
  | .assign _ _ out e :: rest =>
    if out=name then e.toAlgorithm?.toOption.bind loadType? else assignedLoad? rest name
  | _ :: rest => assignedLoad? rest name

def integerConversion {d : TileDType} {sh : TileShape} : Op d sh → Bool
  | .intToReal (.ref .int _ "int_b") => Bool.true
  | .castFloat _ _ e => integerConversion e
  | .mul _ _ a b => integerConversion a || integerConversion b
  | _ => Bool.false

def output? (code : List ComputeStmt) : Option (TileDType × Bool) :=
  match code.getLast? with
  | some (.store d _ _ e _) =>
    e.toAlgorithm?.toOption.map fun v => (d,integerConversion v)
  | _ => none

/-- The actual int8 storage load stays on the signed integer channel, and its
conversion participates in the product before the explicit destination cast. -/
theorem source_types (storage : Storage) (swapped : Bool) :
    let code := (kernel storage swapped "b" "scale" "out" 3 3 3 1 3 1 2 2).surfaceBody
    assignedLoad? code "int_b" = some .int ∧
    assignedLoad? code "scale_b" = some .real ∧
    output? code = some (outputType storage,Bool.true) := by
  cases storage <;> cases swapped <;> exact ⟨rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.SourceTypes
