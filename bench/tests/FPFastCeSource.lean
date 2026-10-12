import bench.tritonbench_g.fast_ce_loss.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.SourceChecks
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

def storageType : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def types (body : List ComputeStmt) (name : RegName) : List TileDType :=
  body.filterMap fun st => match st with
    | .assign d _ n _ => if n=name then some d else none
    | _ => none

def yes (body : List ComputeStmt) : List ComputeStmt :=
  ((Source.conditional? body (fun _ => Bool.true)).map (·.yes)).getD []

def vectorTypes (storage : Storage) (cap scale : Bool) : List TileDType :=
  [storageType storage] ++ (if scale then [storageType storage] else []) ++
    (if cap then [.real] else [])

variable (storage : Storage) (cap scale swapped : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B C DR : Nat) (SC SL : ℝ)

/-- Scaling happens in the input precision, and softcap division promotes to fp32
before the libdevice operation. The final explicit logits cast is retained. -/
theorem forward_logit_types :
    types (forward storage cap scale swapped X Y Z L V R B SC SL).surfaceBody "logits" =
      vectorTypes storage cap scale ++ [.real] := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped <;> rfl

theorem chunked_logit_types :
    types (chunked storage cap scale swapped X Y Z L V C R B SC SL).surfaceBody "logits" =
      vectorTypes storage cap scale ++ [.real] := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped <;> rfl

/-- Forward gathers scale before the cast in the loss expression. -/
theorem forward_gather_types :
    types (yes (forward storage cap scale swapped X Y Z L V R B SC SL).surfaceBody) "x" =
      vectorTypes storage cap scale := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped <;> rfl

/-- The chunked gather explicitly casts to fp32 before optional scaling. -/
theorem chunked_gather_types :
    types (yes (yes (chunked storage cap scale swapped X Y Z L V C R B SC SL).surfaceBody)) "x" =
      vectorTypes .fp32 cap scale := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped <;> rfl

theorem backward_logit_types :
    types (backward storage cap scale swapped X Y Z L V R DR B SC SL).surfaceBody "x" =
      vectorTypes storage cap scale := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped <;> rfl

theorem backward_partial_type :
    types (backward storage cap scale swapped X Y Z L V R DR B SC SL).surfaceBody "partial" =
      (if cap then [.real] else []) := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped <;> rfl

/-- All three complete sources keep the signed -100 label channel. -/
theorem signed_labels :
    types (forward storage cap scale swapped X Y Z L V R B SC SL).surfaceBody "label_idx" = [.int] ∧
    types (chunked storage cap scale swapped X Y Z L V C R B SC SL).surfaceBody "label_idx" = [.int] ∧
    types (backward storage cap scale swapped X Y Z L V R DR B SC SL).surfaceBody "label_idx" = [.int] := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped <;> exact ⟨rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastCeLoss.SourceChecks
