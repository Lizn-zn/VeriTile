import bench.tritonbench_g.embedding_triton_kernel.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def storageType : Storage → TileDType
  | .fp32 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

def loads {d : TileDType} {sh : TileShape} : Op d sh → List (TileDType × Bool × Bool)
  | .load dt _ (.mask _) _ => [(dt,Bool.true,Bool.false)]
  | .load dt _ (.maskOther _ _) _ => [(dt,Bool.true,Bool.true)]
  | .load dt _ _ _ => [(dt,Bool.false,Bool.false)]
  | .castFloat _ _ e => loads e
  | _ => []

def accesses : List ComputeStmt → List (String × TileDType × Bool × Bool)
  | [] => []
  | .assign _ _ _ e :: rest =>
      ((e.toAlgorithm?.toOption.map loads).getD []).map (fun (d,m,o) => ("load",d,m,o)) ++ accesses rest
  | .store d _ _ _ .none :: rest => ("store",d,Bool.false,Bool.false) :: accesses rest
  | .store d _ _ _ _ :: rest => ("store",d,Bool.true,Bool.false) :: accesses rest
  | .forRange _ _ _ _ body :: rest => accesses body ++ accesses rest
  | _ :: rest => accesses rest

/-- Inspect the actual full typed loop: signed int32 token IDs, matching
payload/output precision, both explicit load defaults and the output mask. -/
theorem source_accesses (storage : Storage) (swapped : Bool) :
    accesses (source storage swapped "weight" "ids" "out" (-2) 1 2 2 4 2 4 3 2).surfaceBody =
      [("load",.int,Bool.true,Bool.true),
       ("load",storageType storage,Bool.true,Bool.true),
       ("store",storageType storage,Bool.true,Bool.false)] := by
  cases storage <;> cases swapped
  all_goals simp [source,ComputeKernel.surfaceBody,accesses,loads,storageType,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

/-- Inspect the signed subtraction/comparisons rather than a Nat surrogate. -/
def signedOps {d : TileDType} {sh : TileShape} : Op d sh → List (String × TileDType)
  | .sub _ _ _ _ => [("sub",d)]
  | .ge .int _ _ _ => [("ge",.int)]
  | .lt .int _ _ _ => [("lt",.int)]
  | .boolAnd _ a b => signedOps a ++ signedOps b
  | _ => []

def assignments : List ComputeStmt → List (String × List (String × TileDType))
  | [] => []
  | .assign _ _ name e :: rest =>
      (name,(e.toAlgorithm?.toOption.map signedOps).getD []) :: assignments rest
  | .forRange _ _ _ _ body :: rest => assignments body ++ assignments rest
  | _ :: rest => assignments rest

theorem signed_id_operations (storage : Storage) (swapped : Bool) :
    (assignments (source storage swapped "weight" "ids" "out" (-2) 1 2 2 4 2 4 3 2).surfaceBody).filter
      (fun x => x.1=="id_mask" || x.1=="token_ids") =
      [("token_ids",[]),("id_mask",[("ge",.int),("lt",.int)]),("token_ids",[("sub",.int)])] := by
  cases storage <;> cases swapped
  all_goals simp [source,ComputeKernel.surfaceBody,assignments,signedOps,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.SourceTypes
