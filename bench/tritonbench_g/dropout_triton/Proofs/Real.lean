import bench.tritonbench_g.dropout_triton.Proofs.Erasure
import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.RealProof
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

def original (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  Erasure.source keep Bool.false x k y n B p

def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val

def readKeep (keep : KeepStorage) (u : BlockState) (r : RegionName) (o : Nat) : Bool :=
  match keep with
  | .int32 => decide (u.readMemValue .int r o≠0)
  | .bool => u.readMemValue .bool r o

/-- Given keep-mask inverted dropout on the original input arrays. -/
def expected (keep : KeepStorage) (u : BlockState) (x k : RegionName) (p : ℝ) (o : Nat) : ℝ :=
  if readKeep keep u k o then u.readMem x o/(1-p) else 0

theorem readback (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ)
    (u v : BlockState) (he : exec (original keep x k y n B p).toAlgKernel u=some v)
    (j : Fin B) :
    v.readMem y (offset u B j) =
      if offset u B j<n then expected keep u x k p (offset u B j) else u.readMem y (offset u B j) := by
  cases keep
  all_goals
    simp [original,Erasure.source,exec,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,
      stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.select,
      NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.ne] at he
    subst v
    simp only [offset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (BlockState.tileIndex1d_base_offset_injective _) (j,PUnit.unit)]
    by_cases hj : u.pid*B+j.val<n
    · simp [hj,expected,readKeep]
      split <;> norm_num [*]
    · simp [hj]

theorem frame (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ)
    (u v : BlockState) (he : exec (original keep x k y n B p).toAlgKernel u=some v)
    (r : RegionName) (o : Nat)
    (ho : ∀ j : Fin B, offset u B j<n → ¬ (y=r ∧ offset u B j=o)) :
    v.mem r o=u.mem r o := by
  cases keep
  all_goals
    simp [original,Erasure.source,exec,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,
      stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.select,
      NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.ne] at he
    subst he
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro i _ hi hc
    exact ho i.1 (by simpa [offset] using hi) hc

/-- Complete regional execution, with values read before the store. No
input/output separation is needed, including aliases with the keep buffer. -/
theorem run (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) (u : BlockState) :
    ∃ v, exec (original keep x k y n B p).toAlgKernel u=some v ∧
      (∀ j : Fin B, offset u B j<n → v.readMem y (offset u B j)=expected keep u x k p (offset u B j)) ∧
      (∀ r o, (r≠y ∨ ∀ j : Fin B, offset u B j<n → o≠offset u B j) → v.mem r o=u.mem r o) := by
  cases he : exec (original keep x k y n B p).toAlgKernel u with
  | none =>
    cases keep
    all_goals simp [original,Erasure.source,exec,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,
      stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.select,
      NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.ne] at he
  | some v =>
    refine ⟨v,rfl,?_,?_⟩
    · intro j hj
      simpa only [hj,if_true] using readback keep x k y n B p u v he j
    · intro r o ho
      apply frame keep x k y n B p u v he r o
      intro j hj ⟨hr,ha⟩
      rcases ho with hn | hm
      · exact hn hr.symm
      · exact hm j hj ha.symm

/-- Actual-access bounds for both input loads and the store. -/
theorem trace_safe (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ)
    (bounds : RegionBounds) (u : BlockState)
    (hx : ∀ j : Fin B, offset u B j<n → offset u B j<bounds x)
    (hk : ∀ j : Fin B, offset u B j<n → offset u B j<bounds k)
    (hy : ∀ j : Fin B, offset u B j<n → offset u B j<bounds y) :
    (original keep x k y n B p).toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  cases keep
  all_goals simp [original,Erasure.source,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt,MaskOpt.SafeAt,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.select,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,ComparableDType.ne,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MemAccess.SafeAt,
    MaskOpt.Active,BlockState.setReg]
  all_goals exact ⟨fun j hj => hx j hj,fun j hj => hk j hj,fun j hj => hy j hj⟩

theorem flatten_ok (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) :
    (original keep x k y n B p).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  cases keep
  all_goals simp [original,Erasure.source,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DropoutTriton.RealProof
