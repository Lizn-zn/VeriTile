import bench.tritonbench_g.cross_entropy_ops.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CrossEntropyOps.SourceTypes
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def dtype : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def fwd (storage : Storage) (swapped smooth split : Bool) :=
  forward storage swapped "loss" "lse" "z_loss" "logits" ⟨"label"⟩
    1 3 2 (-100) 11 0 9 5 11 8 smooth split

def bwd (storage : Storage) (swapped smooth : Bool) :=
  backward storage swapped "gradient" "dloss" "logits" "lse" ⟨"label"⟩
    1 3 2 (-100) 11 0 9 11 11 1 8 smooth

def assignType? (code : List ComputeStmt) (name : RegName) : Option TileDType :=
  match code with
  | [] => none
  | .assign d _ out _ :: rest => if out = name then some d else assignType? rest name
  | _ :: rest => assignType? rest name

def loadType? {d : TileDType} {sh : TileShape} : Op d sh → Option TileDType
  | .load dtype _ _ _ => some dtype
  | .castFloat _ _ e => loadType? e
  | .mul _ _ a _ => loadType? a
  | _ => none

def assignedLoadType? (code : List ComputeStmt) (name : RegName) : Option TileDType :=
  match code with
  | [] => none
  | .assign _ _ out e :: rest =>
    if out = name then e.toAlgorithm?.toOption.bind loadType? else assignedLoadType? rest name
  | _ :: rest => assignedLoadType? rest name

def stores (code : List ComputeStmt) : List TileDType :=
  match code with
  | [] => []
  | .store d _ _ _ _ :: rest => d :: stores rest
  | .ifThen _ body :: rest => stores body ++ stores rest
  | .ifThenElse _ a b :: rest => stores a ++ stores b ++ stores rest
  | _ :: rest => stores rest

def zStoreGuard? (code : List ComputeStmt) : Option Bool :=
  match code.getLast? with
  | some (.ifThen cond _) =>
    match cond with
    | .boolNot (.constBool b) => some (!b)
    | _ => none
  | _ => none

/-- Loads retain source storage before the explicit fp32 conversion; signed
labels, all three fp32 destinations and the split-gated z-loss remain visible. -/
theorem forward_types (storage : Storage) (swapped smooth split : Bool) :
    let code := (fwd storage swapped smooth split).surfaceBody
    assignedLoadType? code "logits" = some (dtype storage) ∧
    assignType? code "logits" = some .real ∧
    assignType? code "label_idx" = some .int ∧
    stores code = [.real,.real,.real] ∧ zStoreGuard? code = some (!split) := by
  cases storage <;> cases swapped <;> cases smooth <;> cases split
  all_goals refine ⟨?_,rfl,rfl,?_,rfl⟩
  all_goals simp [stores,fwd,forward,ComputeKernel.surfaceBody,assignedLoadType?,loadType?,dtype,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

theorem backward_types (storage : Storage) (swapped smooth : Bool) :
    let code := (bwd storage swapped smooth).surfaceBody
    assignedLoadType? code "logits" = some (dtype storage) ∧
    assignedLoadType? code "lse" = some .real ∧
    assignType? code "logits" = some .real ∧
    assignType? code "label_idx" = some .int ∧
    stores code = [dtype storage] := by
  cases storage <;> cases swapped <;> cases smooth
  all_goals refine ⟨?_,rfl,rfl,rfl,?_⟩
  all_goals simp [stores,bwd,backward,ComputeKernel.surfaceBody,assignedLoadType?,loadType?,dtype,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption,Option.bind]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropyOps.SourceTypes
