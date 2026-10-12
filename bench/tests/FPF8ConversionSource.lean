import bench.tritonbench_g.f8_conversion_utils.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.SourceTypes
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def formatType : Format → TileDType
  | .e4 => .f8e4
  | .e5 => .f8e5

def inputType : Input → TileDType
  | .fp16 => .fp16
  | .fp32 => .real

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

theorem to_half_accesses (format : Format) (swapped : Bool) :
    accesses (toHalf format swapped "y" "x" 2 4).surfaceBody =
      [("load",formatType format,Bool.true,Bool.false),
       ("store",.fp16,Bool.true,Bool.false),("store",.fp16,Bool.true,Bool.false)] := by
  cases format <;> cases swapped
  all_goals simp [toHalf,ComputeKernel.surfaceBody,accesses,loads,formatType,
    ComputeExpr.toAlgorithm?,Except.toOption]

theorem to_fp8_accesses (input : Input) (format : Format) (swapped : Bool) :
    accesses (toFP8 input format swapped "y" "x" 2 4).surfaceBody =
      [("load",inputType input,Bool.true,Bool.false),
       ("store",formatType format,Bool.true,Bool.false)] := by
  cases input <;> cases format <;> cases swapped
  all_goals simp [toFP8,ComputeKernel.surfaceBody,accesses,loads,formatType,inputType,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

def casts {d : TileDType} {sh : TileShape} : Op d sh → List (TileDType × TileDType)
  | .castFloat src dst e => (src.toTileDType,dst.toTileDType) :: casts e
  | _ => []

def storeCasts : List ComputeStmt → List (TileDType × TileDType)
  | [] => []
  | .store _ _ _ e _ :: rest => (e.toAlgorithm?.toOption.map casts).getD [] ++ storeCasts rest
  | _ :: rest => storeCasts rest

/-- Both original forward stores perform the same actual fp8-to-fp16 cast. -/
theorem to_half_casts (format : Format) (swapped : Bool) :
    storeCasts (toHalf format swapped "y" "x" 2 4).surfaceBody =
      [(formatType format,.fp16),(formatType format,.fp16)] := by
  cases format <;> cases swapped <;> rfl

/-- The source fp16/fp32 channel and selected fp8 destination survive lowering. -/
theorem to_fp8_casts (input : Input) (format : Format) (swapped : Bool) :
    storeCasts (toFP8 input format swapped "y" "x" 2 4).surfaceBody =
      [(inputType input,formatType format)] := by
  cases input <;> cases format <;> cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.SourceTypes
