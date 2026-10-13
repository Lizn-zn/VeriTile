import bench.tritonbench_g.rotary_emb.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def storageCarrier : Storage → TileDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .f8e4 => .f8e4
  | .f8e5 => .f8e5

def storedCarrier : ComputeStmt → Option TileDType
  | .store dtype _ _ _ _ => some dtype
  | _ => none

def promotedStorage (a b : Storage) : Storage :=
  if a=.fp64 ∨ b=.fp64 then .fp64 else
  if a=.fp32 ∨ b=.fp32 then .fp32 else
  if a=.fp16 ∨ b=.fp16 then .fp16 else
  if a=.bf16 ∧ b=.bf16 then .bf16 else
  if a=.bf16 ∨ b=.bf16 then .fp32 else
  if a=b then a else .fp16

def assignedCarrier : ComputeStmt → Option TileDType
  | .assign dtype _ _ _ => some dtype
  | _ => none

def assignedPrecision : ComputeStmt → Option ComputeDType
  | .assign _ _ _ (.compute (.alg precision _)) => some precision
  | _ => none

def storagePrecision : Storage → Option ComputeDType
  | .fp32 => some .fp32
  | .fp64 => some .fp64
  | _ => none

local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

theorem template_shape (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).surfaceBody.length = 34 := by
  cases data <;> cases cos <;> cases sin <;> rfl

theorem template_stores (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    let body := (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).surfaceBody
    storedCarrier body[18]! = some (storageCarrier data) ∧
    storedCarrier body[19]! = some (storageCarrier data) ∧
    storedCarrier body[32]! = some (storageCarrier data) ∧
    storedCarrier body[33]! = some (storageCarrier data) := by
  cases data <;> cases cos <;> cases sin <;> exact ⟨rfl,rfl,rfl,rfl⟩

/-- The two products promote independently before the final sum/difference.
This also checks the fp32/fp64 compute tag, whose erased tile carrier is shared. -/
theorem template_result_precision (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    let body := (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd
      ssb ssd L HQ HK BH BS D).surfaceBody
    let dtype := promotedStorage (promotedStorage data cos) (promotedStorage data sin)
    assignedCarrier body[16]! = some (storageCarrier dtype) ∧
    assignedCarrier body[17]! = some (storageCarrier dtype) ∧
    assignedPrecision body[16]! = storagePrecision dtype ∧
    assignedPrecision body[17]! = storagePrecision dtype := by
  cases data <;> cases cos <;> cases sin <;> exact ⟨rfl,rfl,rfl,rfl⟩

theorem original_shape (q k cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (original q k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).surfaceBody.length = 34 := by
  simp [original, ComputeKernel.surfaceBody, List.length_append, List.length_take,
    List.length_drop, template_shape]

theorem io_roles (q k cos sin : Storage) (v : Bool) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (kernel q k cos sin v Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).inputs = [Q,K,Cos,Sin] ∧
    (kernel q k cos sin v Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).outputs = [Q,K] := by
  cases v <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
