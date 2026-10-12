import bench.tritonbench_g.diag_ssm_triton.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def storageType : Storage → TileDType
  | .fp32 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

def loaded {d : TileDType} {sh : TileShape} : Op d sh → List (TileDType × Bool)
  | .load dt _ (.maskOther _ _) _ => [(dt,Bool.true)]
  | .load dt _ _ _ => [(dt,Bool.false)]
  | .castFloat _ _ e => loaded e
  | .sub _ _ _ b => loaded b
  | _ => []

def accesses (code : List ComputeStmt) : List (String × TileDType × Bool) :=
  match code with
  | [] => []
  | .assign _ _ _ e :: rest =>
      ((e.toAlgorithm?.toOption.map loaded).getD []).map (fun (d,m) => ("load",d,m)) ++ accesses rest
  | .store d _ _ _ .none :: rest => ("store",d,Bool.false) :: accesses rest
  | .store d _ _ _ _ :: rest => ("store",d,Bool.true) :: accesses rest
  | .forLoop _ _ body :: rest => accesses body ++ accesses rest
  | .forRange _ _ _ _ body :: rest => accesses body ++ accesses rest
  | .ifThenElse _ left right :: rest => accesses left ++ accesses right ++ accesses rest
  | _ :: rest => accesses rest
termination_by sizeOf code

/-- Storage precision and masks survive the typed transcription and loops. -/
theorem real_forward_accesses (storage : Storage) (swapped : Bool) :
    accesses (forward storage swapped "s" "x" "lambda" "y" 2 1 3 4).surfaceBody =
      [("load",storageType storage,Bool.true),("load",storageType storage,Bool.true),
       ("load",storageType storage,Bool.true),("store",storageType storage,Bool.true)] := by
  cases storage <;> cases swapped <;> simp [forward,ComputeKernel.surfaceBody,accesses,loaded,storageType,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

theorem real_backward_accesses (storage : Storage) (swapped : Bool) :
    accesses (backward storage swapped "s" "lambda" "y" "gs" "gx" "gl" "gy" 2 1 3 4).surfaceBody =
      List.replicate 4 ("load",storageType storage,Bool.true) ++
      List.replicate 3 ("store",storageType storage,Bool.true) := by
  cases storage <;> cases swapped <;> simp [backward,ComputeKernel.surfaceBody,accesses,loaded,storageType,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

theorem complex_forward_accesses (swapped : Bool) :
    accesses (complexForward swapped "s" "x" "y" "lambda" 2 1 3 4).surfaceBody =
      List.replicate 6 ("load",TileDType.real,Bool.true) ++
      List.replicate 2 ("store",TileDType.real,Bool.true) := by
  cases swapped <;> simp [complexForward,ComputeKernel.surfaceBody,accesses,loaded,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

theorem complex_backward_accesses (swapped : Bool) :
    accesses (complexBackward swapped "s" "lambda" "y" "gs" "gx" "gl" "gy" 2 1 3 4).surfaceBody =
      List.replicate 8 ("load",TileDType.real,Bool.true) ++
      List.replicate 6 ("store",TileDType.real,Bool.true) := by
  cases swapped <;> simp [complexBackward,ComputeKernel.surfaceBody,accesses,loaded,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.SourceTypes
