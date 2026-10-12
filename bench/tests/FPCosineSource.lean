import bench.tritonbench_g.cosine_compute.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CosineCompute.SourceTypes
open VeriTile.Triton
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

def selected (storage : Storage) (swapped : Bool) (n B : Nat) :=
  kernel storage swapped "a" "b" n B

def dtype : Storage → TileDType
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .fp32 => .real

def assignType? (code : List ComputeStmt) (name : RegName) : Option TileDType :=
  match code with
  | [] => none
  | .assign d _ out _ :: rest => if out = name then some d else assignType? rest name
  | _ :: rest => assignType? rest name

def storeType? (code : List ComputeStmt) := code.getLast?.bind fun st => match st with
  | .store d _ _ _ _ => some d
  | _ => none

theorem preserved_types (storage : Storage) (swapped : Bool) (n B : Nat) :
    let code := (selected storage swapped n B).surfaceBody
    assignType? code "offset" = some .nat ∧
    assignType? code "a_value" = some (dtype storage) ∧
    assignType? code "b_value" = some .real ∧
    storeType? code = some (dtype storage) := by
  cases storage <;> cases swapped <;> exact ⟨rfl,rfl,rfl,rfl⟩

/-- Source omits other=0: MaskOpt.mask retains unspecified masked values.
The padding tag is unused for an ordinary region pointer. -/
theorem load_padding (swapped : Bool) (n B : Nat) :
    ((selected .fp16 swapped n B).surfaceBody.drop 2).head? =
      some (.assign .fp16 [B] "a_value" (.alg
        (.load .fp16 (.region (⟨"a"⟩ : Region .fp16) (.ref .nat [B] "offset"))
          (.mask (.ref .bool [B] "mask")) .zero))) := by
  cases swapped <;> rfl

/-- The intrinsic receives the explicit fp16-to-fp32 conversion. -/
theorem cosine_conversion (swapped : Bool) (n B : Nat) :
    ((selected .fp16 swapped n B).surfaceBody.drop 3).head? =
      some (.assign .real [B] "b_value" (.alg
        (.cos (.castFloat .fp16 .real (.ref .fp16 [B] "a_value"))))) := by
  cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CosineCompute.SourceTypes
