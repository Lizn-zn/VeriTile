import bench.tritonbench_g.f8_conversion_utils.Proofs.Erasure
import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.RealProof
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

def original (doubleStore : Bool) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  if doubleStore then Erasure.toHalf Bool.false Y X N B else Erasure.toFP8 Bool.false Y X N B

def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val

/-- Erasing storage precision leaves the identity on each active lane, for
both full source bodies, including the forward kernel's duplicate store. -/
theorem readback (doubleStore : Bool) (Y X : RegionName) (N B : Nat)
    (u v : BlockState) (he : exec (original doubleStore Y X N B).toAlgKernel u=some v)
    (j : Fin B) (hj : offset u B j<N) :
    v.readMem Y (offset u B j)=u.readMem X (offset u B j) := by
  cases doubleStore
  all_goals
    simp [original,Erasure.toHalf,Erasure.toFP8,exec,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
      ComputeDType.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType,Region.cast,Option.bind,
      Except.bind,bind,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,
      NumericDType.add,NumericDType.mul,ComparableDType.lt] at he
    subst v
    simp only [offset]
    rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ (j,PUnit.unit)
      (by simpa only [offset] using hj)
      (fun k _ he => BlockState.tileIndex1d_base_offset_injective _ he)]
    simp [offset] at hj
    simp [hj]

theorem frame (doubleStore : Bool) (Y X : RegionName) (N B : Nat)
    (u v : BlockState) (he : exec (original doubleStore Y X N B).toAlgKernel u=some v)
    (r : RegionName) (o : Nat)
    (ho : ∀ j : Fin B, offset u B j<N → ¬ (Y=r ∧ offset u B j=o)) :
    v.mem r o=u.mem r o := by
  cases doubleStore
  all_goals
    simp [original,Erasure.toHalf,Erasure.toFP8,exec,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
      ComputeDType.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType,Region.cast,Option.bind,
      Except.bind,bind,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,
      NumericDType.add,NumericDType.mul,ComparableDType.lt] at he
    subst v
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) ?_
    · intro i _ hi hc
      exact ho i.1 (by simpa [offset] using hi) hc
    · first
      | rfl
      | refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
        intro i _ hi hc
        exact ho i.1 (by simpa [offset] using hi) hc

/-- Both whole kernels execute from arbitrary registers, including aliased
input/output regions; the value is loaded before either store. -/
theorem run (doubleStore : Bool) (Y X : RegionName) (N B : Nat) (u : BlockState) :
    ∃ v, exec (original doubleStore Y X N B).toAlgKernel u=some v ∧
      (∀ j : Fin B, offset u B j<N → v.readMem Y (offset u B j)=u.readMem X (offset u B j)) ∧
      (∀ r o, (r≠Y ∨ ∀ j : Fin B, offset u B j<N → o≠offset u B j) → v.mem r o=u.mem r o) := by
  cases he : exec (original doubleStore Y X N B).toAlgKernel u with
  | none =>
    cases doubleStore
    all_goals simp [original,Erasure.toHalf,Erasure.toFP8,exec,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
      ComputeDType.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType,Region.cast,Option.bind,
      Except.bind,bind,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,
      NumericDType.add,NumericDType.mul,ComparableDType.lt] at he
  | some v =>
    refine ⟨v,rfl,?_,?_⟩
    · intro j hj
      exact readback doubleStore Y X N B u v he j hj
    · intro r o ho
      apply frame doubleStore Y X N B u v he r o
      intro j hj ⟨hr,ha⟩
      rcases ho with hn | hm
      · exact hn hr.symm
      · exact hm j hj ha.symm

theorem trace_safe (doubleStore : Bool) (Y X : RegionName) (N B : Nat)
    (bounds : RegionBounds) (u : BlockState)
    (hx : ∀ j : Fin B, offset u B j<N → offset u B j<bounds X)
    (hy : ∀ j : Fin B, offset u B j<N → offset u B j<bounds Y) :
    (original doubleStore Y X N B).toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  cases doubleStore
  all_goals simp [original,Erasure.toHalf,Erasure.toFP8,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
      ComputeDType.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType,Region.cast,Option.bind,
      Except.bind,bind,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,
      NumericDType.add,NumericDType.mul,ComparableDType.lt,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MemAccess.SafeAt,MaskOpt.Active,BlockState.setReg]
  · exact ⟨fun j hj => hx j hj,fun j hj => hy j hj⟩
  · exact ⟨fun j hj => hx j hj,fun j hj => hy j hj⟩

theorem flatten_ok (doubleStore : Bool) (Y X : RegionName) (N B : Nat) :
    (original doubleStore Y X N B).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  cases doubleStore
  all_goals simp [original,Erasure.toHalf,Erasure.toFP8,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
      ComputeDType.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType,Region.cast,
      Except.bind,bind,
      StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.RealProof
