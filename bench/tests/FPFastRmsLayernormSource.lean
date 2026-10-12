import bench.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.SourceChecks
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1500000
set_option maxRecDepth 15000

def storageType : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def types (body : List ComputeStmt) (name : RegName) : List TileDType :=
  body.filterMap fun st => match st with
    | .assign d _ n _ => if n=name then some d else none
    | _ => none

def storeTypes (body : List ComputeStmt) : List TileDType :=
  body.filterMap fun st => match st with
    | .store d _ _ _ _ => some d
    | _ => none

variable (input weight : Storage) (swapped isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ)

/-- Regular forward really casts into weight storage before multiplication. -/
theorem regular_precision :
    (types (forward input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody "X_row",
     types (forward input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody "W_row",
     types (forward input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody "normed",
     types (forward input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody "output")=
      ([.real],[storageType weight],[.real,storageType weight],[storageType weight]) := by
  cases input <;> cases weight <;> cases swapped <;> rfl

theorem gemma_precision :
    (types (gemma input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody "W_row",
     types (gemma input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody "output")=
      ([.real],[.real]) := by
  cases input <;> cases weight <;> cases swapped <;> rfl

theorem stores :
    storeTypes (forward input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody=
      [.real,storageType input] ∧
    storeTypes (gemma input weight swapped Y X W r YR XR WR RR N B eps).surfaceBody=
      [.real,storageType input] ∧
    storeTypes (backward input weight swapped Y X W r dW YR XR WR RR DWR N B eps isGemma).surfaceBody=
      [storageType input] := by
  cases input <;> cases weight <;> cases swapped <;> cases isGemma <;> exact ⟨rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.SourceChecks
