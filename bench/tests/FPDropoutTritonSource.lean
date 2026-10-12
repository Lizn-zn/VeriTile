import bench.tritonbench_g.dropout_triton.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def storageType : Storage → TileDType
  | .fp32 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

def keepType : KeepStorage → TileDType
  | .int32 => .int
  | .bool => .bool

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
  | _ :: rest => accesses rest

theorem source_accesses (storage : Storage) (keep : KeepStorage) (swapped : Bool) :
    accesses (source storage keep swapped "x" "keep" "out" 3 4 (1/2)).surfaceBody =
      [("load",storageType storage,Bool.true,Bool.false),
       ("load",keepType keep,Bool.true,Bool.false),
       ("store",storageType storage,Bool.true,Bool.false)] := by
  cases storage <;> cases keep <;> cases swapped
  all_goals simp [source,ComputeKernel.surfaceBody,accesses,loads,storageType,keepType,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

/-- Inspect actual division and condition nodes, after preserving casts. -/
def arithmetic {d : TileDType} {sh : TileShape} : Op d sh → List (String × TileDType)
  | .div _ _ a b => ("div",d) :: (arithmetic a ++ arithmetic b)
  | .sub _ _ a b => ("sub",d) :: (arithmetic a ++ arithmetic b)
  | .castFloat _ dst e => ("cast",dst.toTileDType) :: arithmetic e
  | .where c a b => ("where",d) :: (arithmetic c ++ arithmetic a ++ arithmetic b)
  | .ne .int _ (.ref .int _ _) (.broadcast (.constInt 0) _) => [("nonzero",.int)]
  | _ => []

def assignments : List ComputeStmt → List (String × List (String × TileDType))
  | [] => []
  | .assign _ _ name e :: rest => (name, (e.toAlgorithm?.toOption.map arithmetic).getD []) :: assignments rest
  | _ :: rest => assignments rest

theorem source_arithmetic (storage : Storage) (keep : KeepStorage) (swapped : Bool) :
    assignments (source storage keep swapped "x" "keep" "out" 3 4 (1/2)).surfaceBody =
      [("pid",[]),("block_start",[]),("offsets",[]),("mask",[]),("x",[]),("x_keep",[]),
       ("output", [("where",.real)] ++ (match keep with | .int32 => [("nonzero",.int)] | .bool => []) ++
         [("div",.real)] ++ (match storage with | .fp32 => [] | _ => [("cast",.real)]) ++ [("sub",.real)])] := by
  cases storage <;> cases keep <;> cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DropoutTriton.SourceTypes
