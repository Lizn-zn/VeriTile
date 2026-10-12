import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Meta.StatementAudit

/-! Atomic FP execution uses a storage-typed opaque operation. These checks
exercise sequential read-modify-write, masks, cell types and failure without
assuming any floating algebraic law or cross-program execution order. -/
namespace FPAtomicExecutionTests
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 800000

variable {α : Type} [Inhabited α]

theorem repeated_address_reads_updated_value (M : Algebra α)
    (s : State α) (a b : α) (f : α → α → α) (hf : M.atomicAdd .fp16 = some f) :
    atomicAdd M .fp16 (.region ⟨"Y"⟩ (.full [2] (.constNat 7))) .none
      (fun i : TileIndex [2] => if i.1.val = 0 then a else b) s =
    some (s.write "Y" 7 (.mk .fp16 (f (f ((s.mem "Y" 7).read .fp16) a) b))) := by
  have hi : TileShape.allIndices [2] = [(0,PUnit.unit), (1,PUnit.unit)] := by decide
  simp [atomicAdd, atomicAddition, hf, evalOp_unfold, hi,
    Region.cast, State.write, Cell.read]
  funext r o
  by_cases h : r = "Y" ∧ o = 7 <;> simp [h]

theorem masked_lane_preserves_cell (M : Algebra α)
    (s : State α) (v : Values α .bf16 [2]) (f : α → α → α)
    (hf : M.atomicAdd .bf16 = some f) :
    (atomicAdd M .bf16 (.region ⟨"Y"⟩ (.add .nat .scalarL (.constNat 7) (.arange 2)))
      (.mask (.lt .nat .scalarR (.arange 2) (.constNat 1))) v s).map
        (fun t => (t.mem "Y" 7, t.mem "Y" 8, t.mem "Z" 7, t.regs)) =
      some (.mk .bf16 (f ((s.mem "Y" 7).read .bf16) (v (0,PUnit.unit))),
        s.mem "Y" 8, s.mem "Z" 7, s.regs) := by
  have hi : TileShape.allIndices [2] = [(0,PUnit.unit), (1,PUnit.unit)] := by decide
  simp [atomicAdd, atomicAddition, hf, evalOp_unfold, numeric, numericLt, natLt, bop,
    hi, Region.cast, State.write]


theorem missing_backend_fails (M : Algebra α)
    (s : State α) (mem : MemAccess .fp16 shape) (mask : MaskOpt .fp16 shape)
    (v : Values α .fp16 shape) (h : M.atomicAdd .fp16 = none) :
    atomicAdd M .fp16 mem mask v s = none := by
  simp [atomicAdd, atomicAddition, h]

theorem missing_pointer_fails (M : Algebra α)
    (s : State α) (v : Values α .fp16 [2]) (f : α → α → α)
    (hf : M.atomicAdd .fp16 = some f) (h : s.regs .ptr [2] "p" = none) :
    atomicAdd M .fp16 (.ptr (.ref .ptr [2] "p")) .none v s = none := by
  simp [atomicAdd, atomicAddition, hf, evalOp_unfold, h]

theorem pointer_channel_keeps_storage (M : Algebra α)
    (s : State α) (v : Values α .bf16 [2]) (f : α → α → α)
    (hf : M.atomicAdd .bf16 = some f)
    (h : s.regs .ptr [2] "p" = some (fun i : TileIndex [2] => (("Y" : RegionName), 7+i.1.val))) :
    atomicAdd M .bf16 (.ptr (.ref .ptr [2] "p")) .none v s =
    atomicAdd M .bf16
      (.region ⟨"Y"⟩ (.add .nat .scalarL (.constNat 7) (.arange 2))) .none v s := by
  simp [atomicAdd, atomicAddition, hf, evalOp_unfold, numeric, bop, h, Region.cast]


omit [Inhabited α] in
/-- The default arithmetic profile does not change the storage-selected
atomic backend. -/
theorem default_profile_keeps_atomic (M : Algebra α) (p : ComputeDType) :
    (M.withDefaultPrecision p).atomicAdd = M.atomicAdd := rfl

theorem compute_statement_keeps_cast (M : Algebra α) (s : State α)
    (v : Values α .fp32 [2]) (h : s.regs .fp32 [2] "v" = some v) :
    step M (.atomicAdd .bf16 [2] (.ptr (.ref .ptr [2] "p"))
      (.alg (.castFloat .fp32 .bf16 (.ref .fp32 [2] "v")))
      (.mask (.ref .bool [2] "mask"))) s =
    atomicAdd M .bf16 (.ptr (.ref .ptr [2] "p"))
      (.mask (.ref .bool [2] "mask"))
      (fun i => M.cast none .fp32 .bf16 (v i)) s := by
  simp [step, evalExpr, evalOp_unfold, h, ofFloat, toFloat]

theorem algorithm_statement_reads_value (M : Algebra α) (s : State α)
    (v : Values α .fp16 [2]) (h : s.regs .fp16 [2] "v" = some v) :
    step M (.alg (.atomicAdd .fp16 [2] (.ptr (.ref .ptr [2] "p"))
      (.ref .fp16 [2] "v") .none)) s =
    atomicAdd M .fp16 (.ptr (.ref .ptr [2] "p")) .none v s := by
  simp [step, stepAlg, evalOp_unfold, h]

theorem missing_value_fails (M : Algebra α) (s : State α)
    (h : s.regs .fp16 [2] "v" = none) :
    step M (.atomicAdd .fp16 [2] (.ptr (.ref .ptr [2] "p"))
      (.alg (.ref .fp16 [2] "v")) .none) s = none := by
  simp [step, evalExpr, evalOp_unfold, h]

theorem missing_mask_fails (M : Algebra α) (s : State α)
    (f : α → α → α) (hf : M.atomicAdd .fp16 = some f)
    (v : Values α .fp16 [2]) (h : s.regs .bool [2] "mask" = none) :
    atomicAdd M .fp16 (.region ⟨"Y"⟩ (.arange 2))
      (.mask (.ref .bool [2] "mask")) v s = none := by
  simp [atomicAdd, atomicAddition, hf, evalOp_unfold, h]

/-- Checked block-pointer padding preserves the outside lane, even though
both supplied values and the atomic backend exist. -/
theorem checked_block_pointer_preserves_tail (M : Algebra α) (s : State α)
    (v : Values α .fp16 [2]) (f : α → α → α)
    (hf : M.atomicAdd .fp16 = some f) :
    (atomicAdd M .fp16
      (.blockPtr (.makeBlockPtrDynOffsets "Y" (.constNat 5) [3] [2] [2]
        [.constNat 2]) [0]) .none v s).map
        (fun t => (t.mem "Y" 9, t.mem "Y" 11, t.mem "Z" 9)) =
      some (.mk .fp16 (f ((s.mem "Y" 9).read .fp16) (v (0,PUnit.unit))),
        s.mem "Y" 11, s.mem "Z" 9) := by
  have hi : TileShape.allIndices [2] = [(0,PUnit.unit), (1,PUnit.unit)] := by decide
  simp [atomicAdd, atomicAddition, hf, evalOp_unfold, evalOffsets, hi,
    TileShape.indexToList, State.write]

#guard_msgs (drop info) in
#auditModuleAxioms
end FPAtomicExecutionTests
