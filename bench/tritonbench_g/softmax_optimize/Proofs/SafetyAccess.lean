import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton

theorem pointer_load_safe (bounds : RegionBounds) (B : Nat) (name : RegName)
    (s : BlockState) (addresses : Tile .ptr [B])
    (hp : s.regs .ptr [B] name = some addresses)
    (hb : ∀ lane, (addresses.data lane).2 < bounds (addresses.data lane).1) :
    (Op.load .real (.ptr (.ref .ptr [B] name)) .none).SafeAt bounds s := by
  simpa [Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, evalOp.eq_def, hp] using hb

theorem masked_pointer_load_safe (bounds : RegionBounds) (B : Nat)
    (pointerName maskName : RegName) (s : BlockState)
    (addresses : Tile .ptr [B]) (active : Tile .bool [B])
    (hp : s.regs .ptr [B] pointerName = some addresses)
    (hm : s.regs .bool [B] maskName = some active)
    (hb : ∀ lane, active.data lane = Bool.true →
      (addresses.data lane).2 < bounds (addresses.data lane).1) :
    (Op.load .real (.ptr (.ref .ptr [B] pointerName))
      (.maskOther (.ref .bool [B] maskName) (Op.negInf.broadcast [B]))).SafeAt bounds s := by
  simpa [Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, evalOp.eq_def, hp, hm] using hb

theorem pointer_store_safe (bounds : RegionBounds) (B : Nat)
    (pointerName valueName : RegName) (s : BlockState) (addresses : Tile .ptr [B])
    (hp : s.regs .ptr [B] pointerName = some addresses)
    (hb : ∀ lane, (addresses.data lane).2 < bounds (addresses.data lane).1) :
    Stmt.TraceSafe bounds (.store .real [B] (.ptr (.ref .ptr [B] pointerName))
      (.ref .real [B] valueName) .none) s := by
  simpa [Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, evalOp.eq_def, hp] using hb

theorem masked_pointer_store_safe (bounds : RegionBounds) (B : Nat)
    (pointerName valueName maskName : RegName) (s : BlockState)
    (addresses : Tile .ptr [B]) (active : Tile .bool [B])
    (hp : s.regs .ptr [B] pointerName = some addresses)
    (hm : s.regs .bool [B] maskName = some active)
    (hb : ∀ lane, active.data lane = Bool.true →
      (addresses.data lane).2 < bounds (addresses.data lane).1) :
    Stmt.TraceSafe bounds (.store .real [B] (.ptr (.ref .ptr [B] pointerName))
      (.ref .real [B] valueName) (.mask (.ref .bool [B] maskName))) s := by
  simpa [Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, evalOp.eq_def, hp, hm] using hb

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
