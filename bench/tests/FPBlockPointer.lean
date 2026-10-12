import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Block-pointer execution keeps discrete address calculations separate from
opaque floating values. These are semantic regressions, not GPU experiments.
In particular, no boundary check is inferred from a pointer's parent shape. -/
namespace FPBlockPointerTests
open VeriTile.Triton FP.Structural

variable {α : Type} [Inhabited α]

theorem dynamic_transposed_load (M : Algebra α) (s : State α)
    (base chunk BT BD NT : Nat) :
    FP.Structural.evalOp M none
      (.load .fp16 (.blockPtr
        (.makeBlockPtrDynOffsets "K" (.constNat base) [BD, NT * BT] [BD, BT]
          [1, BD] [.constNat 0, .constNat (chunk * BT)]) []) .none) s =
      some (fun i : TileIndex [BD, BT] =>
        (s.mem "K" (base + i.1.val + (chunk * BT + i.2.1.val) * BD)).read .fp16) := by
  simp only [FP.Structural.evalOp, evalOffsets, bind, pure, Option.bind_some, BlockPtr.inBounds_nil_checkedAxes, if_true]
  congr 1
  funext i
  simp [TileShape.indexToList, BlockPtr.address_2d_offsets, Nat.add_assoc]

private def ptr (offset : Nat) : Op .blockPtr [2] :=
  .makeBlockPtrDynOffsets "x" (.constNat 5) [3] [2] [2] [.constNat offset]

-- Parent shape alone does not mask memory. Lane 1 addresses offset 11 even
-- though its logical coordinate is 3, outside the declared shape [3].
theorem unchecked_tail_reads (M : Algebra α) (s : State α) :
    (FP.Structural.evalOp M none (.load .bf16 (.blockPtr (ptr 2) []) .none) s).map
      (fun v => v (1, PUnit.unit)) = some ((s.mem "x" 11).read .bf16) := by
  simp [FP.Structural.evalOp, evalOffsets, ptr, TileShape.indexToList]

-- This raw AST load explicitly selects zero padding. Checking axis 0 alone
-- does not choose zero; the carrier's arbitrary default cannot substitute.
theorem checked_tail_pads (M : Algebra α) (s : State α) :
    (FP.Structural.evalOp M (some .fp32)
      (.load .bf16 (.blockPtr (ptr 2) [0]) .none .zero) s).map
        (fun v => (v (0, PUnit.unit), v (1, PUnit.unit))) =
      some ((s.mem "x" 9).read .bf16, M.literal (some .fp32) .bf16 0) := by
  simp [FP.Structural.evalOp, evalOffsets, ptr, TileShape.indexToList, boundaryZero]

theorem explicit_other_wins (M : Algebra α) (s : State α) :
    (FP.Structural.evalOp M none (.load .bf16 (.blockPtr (ptr 2) [0])
      (.maskOther (.full [2] (.constBool Bool.false))
        (.full [2] (.constFloat .bf16 7)))) s).map
          (fun v => v (1, PUnit.unit)) = some (M.literal none .bf16 7) := by
  simp [FP.Structural.evalOp, evalOffsets, ptr, TileShape.indexToList, boundaryZero]
  rfl

theorem masked_undefined_and_padding_differ (M : Algebra α) (s : State α) :
    (FP.Structural.evalOp M none (.load .bf16 (.blockPtr (ptr 2) [0])
      (.mask (.full [2] (.constBool Bool.false))) .zero) s).map
        (fun v => (v (0, PUnit.unit), v (1, PUnit.unit))) =
      some (s.undef .bf16 "x" 9, M.literal none .bf16 0) := by
  simp [FP.Structural.evalOp, evalOffsets, ptr, TileShape.indexToList, boundaryZero]

theorem missing_dynamic_offset_fails (M : Algebra α) (s : State α)
    (h : s.regs .nat [] "offset" = none) :
    FP.Structural.evalOp M none
      (.makeBlockPtrDynOffsets "x" (.constNat 5) [3] [2] [2]
        [.ref .nat [] "offset"]) s = none := by
  simp [FP.Structural.evalOp, evalOffsets, h]

theorem advance_preserves_layout (M : Algebra α) (s : State α) :
    FP.Structural.evalOp M none (.advanceBlockPtr (ptr 2) [-1]) s =
      FP.Structural.evalOp M none (ptr 1) s := by
  simp [FP.Structural.evalOp, evalOffsets, ptr]
  rfl

theorem dynamic_constants_match_static (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (ptr : Op .blockPtr shape) (ds : List Int) :
    FP.Structural.evalOp M p (.advanceBlockPtrDyn ptr (ds.map Op.constInt)) s =
      FP.Structural.evalOp M p (.advanceBlockPtr ptr ds) s := by
  have h : evalDeltas M p (ds.map Op.constInt) s = some ds := by
    induction ds with
    | nil => rfl
    | cons d ds ih => simp [evalDeltas, FP.Structural.evalOp, ih]
  simp only [FP.Structural.evalOp, h]
  cases FP.Structural.evalOp M p ptr s <;> rfl

theorem dynamic_rewind_preserves_layout (M : Algebra α) (s : State α)
    (h : s.regs .int [] "delta" = some (fun _ => (-1 : Int))) :
    FP.Structural.evalOp M none
      (.advanceBlockPtrDyn (ptr 2) [.ref .int [] "delta"]) s =
      FP.Structural.evalOp M none (ptr 1) s := by
  simp [FP.Structural.evalOp, evalOffsets, evalDeltas, ptr, h]
  rfl

theorem missing_advance_coordinate_fails (M : Algebra α) (s : State α)
    (h : s.regs .int [] "delta" = none) :
    FP.Structural.evalOp M none
      (.advanceBlockPtrDyn (ptr 2) [.ref .int [] "delta"]) s = none := by
  simp [FP.Structural.evalOp, evalOffsets, evalDeltas, ptr, h]

theorem checked_store_preserves_tail (M : Algebra α) (s : State α)
    (values : Values α .fp16 [2]) :
    (store M (.blockPtr (ptr 2) [0]) .none values s).map
      (fun t => (t.mem "x" 9, t.mem "x" 11, t.mem "other" 9)) =
      some (.mk .fp16 (values (0, PUnit.unit)), s.mem "x" 11, s.mem "other" 9) := by
  have hi : TileShape.allIndices [2] = [(0, PUnit.unit), (1, PUnit.unit)] := by decide
  simp [store, ptr, FP.Structural.evalOp, evalOffsets, hi, TileShape.indexToList, State.write]

-- A false mask suppresses a valid block-pointer write as well.
theorem masked_store_preserves_memory (M : Algebra α) (s : State α)
    (values : Values α .fp16 [2]) :
    store M (.blockPtr (ptr 2) [])
      (.mask (.full [2] (.constBool Bool.false))) values s = some s := by
  simp [store, FP.Structural.evalOp, evalOffsets, ptr]

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBlockPointerTests
