import bench.tritonbench_g.destindex_copy_kv1.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv1.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

def storageType : Storage → TileDType
  | .fp32 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

def loadInfo? {d : TileDType} {sh : TileShape} : Op d sh → Option (TileDType × String)
  | .load dt _ .none _ => some (dt,"unmasked")
  | .load dt _ (.maskOther _ _) _ => some (dt,"masked-other")
  | .load dt _ _ _ => some (dt,"masked-undef")
  | .castFloat _ _ e => loadInfo? e
  | _ => none

def assignedLoad? (code : List ComputeStmt) (name : RegName) : Option (TileDType × String) :=
  match code with
  | [] => none
  | .assign _ _ out e :: rest =>
    if out=name then e.toAlgorithm?.toOption.bind loadInfo? else assignedLoad? rest name
  | _ :: rest => assignedLoad? rest name

def storeInfo (code : List ComputeStmt) : List (TileDType × Bool) :=
  match code with
  | [] => []
  | .store d _ _ _ .none :: rest => (d,Bool.true) :: storeInfo rest
  | .store d _ _ _ _ :: rest => (d,Bool.false) :: storeInfo rest
  | _ :: rest => storeInfo rest

def signedOffset {d : TileDType} {sh : TileShape} : Op d sh → Bool
  | .ptrAdd _ _ (.castIntToNat _) => Bool.true
  | _ => Bool.false

def assignedSignedOffset? (code : List ComputeStmt) (name : RegName) : Bool :=
  match code with
  | [] => Bool.false
  | .assign _ _ out e :: rest =>
    if out=name then (e.toAlgorithm?.toOption.map signedOffset).getD Bool.false
    else assignedSignedOffset? rest name
  | _ :: rest => assignedSignedOffset? rest name

/-- Signed destination load, both payload masks, retained zero-default load,
independent input/output dtype and signed destination arithmetic. -/
theorem types_and_accesses (input output : Storage) (swapped : Bool) :
    let code := (kernel input output swapped "input" ⟨"dest"⟩ "output"
      16 4 1 16 4 1 3 3 4 4).surfaceBody
    assignedLoad? code "dest_index" = some (.int,"unmasked") ∧
    assignedLoad? code "k" = some (storageType input,"masked-other") ∧
    storeInfo code = [(storageType output,Bool.false)] ∧
    assignedSignedOffset? code "o_ptrs" = Bool.true := by
  cases input <;> cases output <;> cases swapped <;> exact ⟨rfl,rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopyKv1.SourceTypes
