import bench.tritonbench_g.dequantize_rowwise.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

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
  | .intToReal (.ref .int _ "x") => Bool.true
  | .castFloat _ _ e => integerConversion e
  | .mul _ _ a b => integerConversion a || integerConversion b
  | _ => Bool.false

def output? (code : List ComputeStmt) : Option (TileDType × Bool) :=
  match code.getLast? with
  | some (.store d _ _ e _) =>
    e.toAlgorithm?.toOption.map fun v => (d,integerConversion v)
  | _ => none

def assignedConversion? (code : List ComputeStmt) : Bool :=
  match code with
  | [] => Bool.false
  | .assign _ _ "output" e :: _ => (e.toAlgorithm?.toOption.map integerConversion).getD Bool.false
  | _ :: rest => assignedConversion? rest

/-- Signed integer input, fp32 scale and fp16 output conversion all remain
in both complete kernels. The masked load omits its default exactly as in source. -/
theorem source_types (swapped : Bool) :
    let code := ((if swapped then indexCommuted else original) "x" "scale" "out" (1/127) 6 3 4).surfaceBody
    assignedLoad? code "x" = some .int ∧
    assignedLoad? code "max_val" = some .real ∧
    output? code = some (.fp16,Bool.false) ∧ assignedConversion? code = Bool.true := by
  cases swapped <;> exact ⟨rfl,rfl,rfl,rfl⟩

def loadPolicy? {d : TileDType} {sh : TileShape} : Op d sh → Option String
  | .load _ _ .none _ => some "unmasked"
  | .load _ _ (.mask _) _ => some "masked-undef"
  | .load _ _ (.maskOther _ _) _ => some "masked-other"
  | .castFloat _ _ e => loadPolicy? e
  | _ => none

def assignedPolicy? (code : List ComputeStmt) (name : RegName) : Option String :=
  match code with
  | [] => none
  | .assign _ _ out e :: rest =>
    if out=name then e.toAlgorithm?.toOption.bind loadPolicy? else assignedPolicy? rest name
  | _ :: rest => assignedPolicy? rest name

theorem load_policies (swapped : Bool) :
    let code := ((if swapped then indexCommuted else original) "x" "scale" "out" (1/127) 6 3 4).surfaceBody
    assignedPolicy? code "x" = some "masked-undef" ∧
    assignedPolicy? code "max_val" = some "unmasked" := by
  cases swapped <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.SourceTypes
