import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealSafetySource

/-! The complete source uses the fragment supported by the existing
region-to-flat-memory bridge. This structural check preserves dynamic block
pointers, signed range bounds, both carry branches and every store. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFlatten
open VeriTile.Triton

theorem append_ok (xs ys : List Stmt) :
    StmtList.FlattenOk (xs ++ ys) ↔ StmtList.FlattenOk xs ∧ StmtList.FlattenOk ys := by
  induction xs with
  | nil => simp [StmtList.FlattenOk]
  | cons x xs ih => simp [StmtList.FlattenOk, ih, and_assoc]

@[simp] theorem dynamic_offsets_ok (region : RegionName) (base : Op .nat [])
    (parent shape strides : List Nat) (a b : Op .nat []) :
    (Op.makeBlockPtrDynOffsets region base parent shape strides [a, b]).FlattenOk ↔
      base.FlattenOk ∧ a.FlattenOk ∧ b.FlattenOk := by
  rw [Op.FlattenOk.eq_def]
  simp only [List.forall_mem_cons]
  simp

@[simp] theorem dynamic_advance_ok {shape : TileShape}
    (ptr : Op .blockPtr shape) (a b : Op .int []) :
    (Op.advanceBlockPtrDyn ptr [a, b]).FlattenOk ↔
      ptr.FlattenOk ∧ a.FlattenOk ∧ b.FlattenOk := by
  rw [Op.FlattenOk.eq_def]
  simp only [List.forall_mem_cons]
  simp

theorem header_ok (heads kvHeads : Nat) (strides : RealHeader.Strides) :
    StmtList.FlattenOk (RealHeader.code heads kvHeads strides) := by
  simp [RealHeader.code, RealHeader.coordinateCode, RealHeader.offsetCode,
    RealHeader.offsetExpr, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem pointers_ok (BM BN BD : Nat) (p : RealPointers.Layout) :
    StmtList.FlattenOk (RealPointers.code BM BN BD p) := by
  simp only [RealPointers.code, StmtList.FlattenOk, Stmt.FlattenOk,
    RealPointers.queryExpr, RealPointers.valueExpr, RealPointers.keyExpr,
    RealPointers.outputExpr, dynamic_offsets_ok]
  simp [RealPointers.rowPointerExpr, RealPointers.rowOffsetExpr, Op.FlattenOk.eq_def]

theorem window_ok (BM BN BD context window complement : Nat) (offset size : Int) :
    Stmt.FlattenOk (RealWindow.code BM BN BD context window complement offset size) := by
  simp only [RealWindow.code, RealWindow.advance, List.cons_append, List.nil_append,
    StmtList.FlattenOk, Stmt.FlattenOk, dynamic_advance_ok]
  simp [RealWindow.complementUpperExpr, RealWindow.lowerExpr, RealWindow.upperExpr,
    RealWindow.clampLower, RealWindow.clampUpper,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem prologue_ok (BM BN BD : Nat) (p : RealPrologue.Parameters) :
    StmtList.FlattenOk (RealPrologue.code BM BN BD p) := by
  simp only [RealPrologue.code, append_ok, header_ok, pointers_ok, true_and]
  simp [RealInit.carryCode, RealInit.scaleCode, RealInit.queryCode, window_ok,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem loop_ok (BM BN BD context window complement even : Nat) (offset size : Int) :
    Stmt.FlattenOk (RealLoopSource.statement BM BN BD context window complement even offset size) := by
  simp [RealLoopSource.statement, RealBody.body, RealBody.loadKey,
    RealBody.scores, RealBody.probabilities, RealBody.rescale, RealBody.accumulate,
    RealBody.tailMask, RealExpressions.distanceExpr, RealExpressions.maskStmt,
    RealExpressions.regularMaskExpr, RealExpressions.complementMaskExpr,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem epilogue_ok (BM BD finish : Nat) (reciprocal : Bool) :
    StmtList.FlattenOk (RealEpilogue.code BM BD finish reciprocal) := by
  cases reciprocal <;>
    simp [RealEpilogue.code, RealEpilogue.endBranch, RealEpilogue.normalizeCode,
      RealEpilogue.denominatorStore, RealEpilogue.centerStore, RealEpilogue.outputStore,
      RealEpilogue.quotientExpr, RealEpilogue.denominatorExpr,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem code_ok (BM BN BD finish : Nat) (p : RealPrologue.Parameters) (reciprocal : Bool) :
    StmtList.FlattenOk (RealSafety.code BM BN BD finish p reciprocal) := by
  simp only [RealSafety.code, append_ok, StmtList.FlattenOk,
    prologue_ok, loop_ok, epilogue_ok, and_self]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFlatten
