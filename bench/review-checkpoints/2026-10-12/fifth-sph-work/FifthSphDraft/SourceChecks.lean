import FifthSphDraft.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.SourceChecks
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

def storageType : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

/-- Triton's binary tensor promotion for the three supported storage types:
fp32 dominates; equal-width fp16/bf16 operands prefer fp16. -/
def promote : Storage → Storage → Storage
  | .fp32,_ | _,.fp32 => .fp32
  | .fp16,_ | _,.fp16 => .fp16
  | .bf16,.bf16 => .bf16

def types (body : List ComputeStmt) (name : RegName) : List TileDType :=
  body.filterMap fun st => match st with
    | .assign d _ n _ => if n=name then some d else none
    | _ => none

def storeTypes (body : List ComputeStmt) : List TileDType :=
  body.filterMap fun st => match st with
    | .store d _ _ _ _ => some d
    | _ => none

/-- Plain named decimal assignments produce fp32 scalar tensors. Input-only
powers retain input precision; coefficient expressions promote to fp32. -/
theorem forward_precision (input output : Storage) (swapped : Bool) (X Y : RegionName) (B N M off stride : Nat) :
    (types (forward input output swapped X Y B N M off stride).surfaceBody "x",
      types (forward input output swapped X Y B N M off stride).surfaceBody "VAR05",
      ["Y00","Y01","Y02","Y03","Y04","Y05","Y06","Y07","Y08","Y09","Y10"].map
        (types (forward input output swapped X Y B N M off stride).surfaceBody),
      storeTypes (forward input output swapped X Y B N M off stride).surfaceBody)=
      ([storageType input],[storageType input],List.replicate 11 [.real],List.replicate 11 (storageType output)) := by
  cases input <;> cases output <;> cases swapped <;> rfl

theorem backward_precision (input gradient upstream : Storage) (swapped : Bool) (X dX g : RegionName) (B N M off stride : Nat) :
    (types (backward input gradient upstream swapped X dX g B N M off stride).surfaceBody "x",
      types (backward input gradient upstream swapped X dX g B N M off stride).surfaceBody "g_0",
      ["g_x","g_y","g_z"].map (types (backward input gradient upstream swapped X dX g B N M off stride).surfaceBody),
      storeTypes (backward input gradient upstream swapped X dX g B N M off stride).surfaceBody)=
      ([storageType input],[storageType upstream],
        List.replicate 3 [storageType gradient,.real],
        List.replicate 3 (storageType gradient)) := by
  cases input <;> cases gradient <;> cases upstream <;> cases swapped <;> rfl

/-- Coefficient assignments materialize literals, including negative literals;
negating an already-assigned named coefficient remains a separate operation. -/
def literalCoefficients (body : List ComputeStmt) : List Bool :=
  body.map fun st => match st with
    | .assign .real [] _ (.alg (.const _)) => Bool.true
    | _ => Bool.false

/-- The complete coefficient blocks start after the seven forward setup/load
statements and the twenty backward setup/load statements, respectively. -/
theorem forward_coefficient_assignments (input output : Storage) (swapped : Bool) (X Y : RegionName) (B N M off stride : Nat) :
    literalCoefficients ((forward input output swapped X Y B N M off stride).surfaceBody.drop 7 |>.take 30)=List.replicate 30 Bool.true := by
  cases input <;> cases output <;> cases swapped <;> rfl

theorem backward_coefficient_assignments (input gradient upstream : Storage) (swapped : Bool) (X dX g : RegionName) (B N M off stride : Nat) :
    literalCoefficients ((backward input gradient upstream swapped X dX g B N M off stride).surfaceBody.drop 20 |>.take 54)=List.replicate 54 Bool.true := by
  cases input <;> cases gradient <;> cases upstream <;> cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.SourceChecks
