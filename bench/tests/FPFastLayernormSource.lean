import bench.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.SourceChecks
open VeriTile.Triton TypedSource
set_option maxHeartbeats 3000000
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

variable (input weight bias : Storage) (swapped : Bool) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ)

theorem forward_loads :
    (types (forward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "X_row",
     types (forward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "W_row",
     types (forward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "b_row")=
      ([.real],[.real],[.real]) := by
  cases input <;> cases weight <;> cases bias <;> cases swapped <;> rfl

theorem backward_loads :
    (types (backward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "dY_row",
     types (backward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "X_row",
     types (backward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "W_row",
     types (backward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "b_row",
     types (backward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "inv_var",
     types (backward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody "mean")=
      ([.real],[.real],[.real],[.real],[.real],[.real]) := by
  cases input <;> cases weight <;> cases bias <;> cases swapped <;> rfl

theorem stores :
    storeTypes (forward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody=
      [.real,.real,storageType input] ∧
    storeTypes (backward input weight bias swapped Y X W b r mu YR XR N B eps).surfaceBody=
      [storageType input] := by
  cases input <;> cases weight <;> cases bias <;> cases swapped <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastLayernorm.SourceChecks
