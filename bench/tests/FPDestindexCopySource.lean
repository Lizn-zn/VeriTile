import bench.tritonbench_g.destindex_copy.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

def storageType : Storage → TileDType
  | .fp32 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

def loadInfo? {d : TileDType} {sh : TileShape} : Op d sh → Option (TileDType × Bool)
  | .load dt _ .none _ => some (dt,Bool.true)
  | .load dt _ _ _ => some (dt,Bool.false)
  | .castFloat _ _ e => loadInfo? e
  | _ => none

def assignedLoad? (code : List ComputeStmt) (name : RegName) : Option (TileDType × Bool) :=
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

/-- Every storage pair retains signed destination metadata, unmasked accesses,
signed destination-address arithmetic and the two independent payload dtypes. -/
theorem types_and_accesses (nopeType ropeType : Storage) (swapped : Bool) :
    let code := (kernel nopeType ropeType swapped "nope" "rope" ⟨"dest"⟩ "outNope" "outRope"
      4 7 1 2 9 1 4 7 1 2 9 1 2 3 4 2).surfaceBody
    assignedLoad? code "dest_index" = some (.int,Bool.true) ∧
    assignedLoad? code "kv_nope" = some (storageType nopeType,Bool.true) ∧
    assignedLoad? code "kv_rope" = some (storageType ropeType,Bool.true) ∧
    storeInfo code = [(storageType nopeType,Bool.true),(storageType ropeType,Bool.true)] ∧
    assignedSignedOffset? code "o_nope_ptrs" = Bool.true ∧
    assignedSignedOffset? code "o_rope_ptrs" = Bool.true := by
  cases nopeType <;> cases ropeType <;> cases swapped <;> exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopy.SourceTypes
