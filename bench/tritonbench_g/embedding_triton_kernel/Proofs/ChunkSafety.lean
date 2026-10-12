import bench.tritonbench_g.embedding_triton_kernel.Proofs.Chunk
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.ChunkSafety
open VeriTile.Triton Chunk
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

structure Bounds (u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN start : Nat) (bounds : RegionBounds) : Prop where
  idAccess : ∀ j : Fin NN, start+(u.pid*N+j.val)<n → start+(u.pid*N+j.val)<bounds ids
  weightAccess : ∀ j : TileIndex [NN,D], Active u n h N start j →
    first≤Reference.token u ids (row u N start j) → Reference.token u ids (row u N start j)<last →
    (Reference.token u ids (row u N start j)-first).toNat*sw+j.2.1.val<bounds weight
  outputAccess : ∀ j : TileIndex [NN,D], Active u n h N start j → address u so N start j<bounds out

/-- Bounds refer only to actual reads/writes. An inactive row loads last as
its token ID, so its weight mask is false without a weight allocation. -/
theorem trace_safe (weight ids out : RegionName) (first last : Int) (sw so n h D N NN start : Nat)
    (bounds : RegionBounds) (u : BlockState) (hc : Context u N NN D)
    (hb : Bounds u weight ids out first last sw so n h D N NN start bounds) :
    Stmt.TraceSafeList bounds (body weight ids out first last sw so n h D N NN)
      (u.setReg "start_nn" .nat [] (Tile.scalar start)) := by
  rcases hc with ⟨hr,hc⟩
  simp [body,code,Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,Option.bind,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MemAccess.SafeAt,MaskOpt.Active,
    stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.expandDim,TileShape.dropInsertedIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,ComparableDType.ge,
    Bool.and_eq_true,hr,hc]
  refine ⟨?_,?_,?_⟩
  · intro j hj
    exact hb.idAccess j hj
  · intro jr jd hlo hhi hd
    have hn : start+(u.pid*N+jr.val)<n := by
      by_contra hnot
      simp [hnot] at hhi
    simp only [hn,if_true] at hlo hhi ⊢
    have hidx := hb.weightAccess (jr,jd,PUnit.unit) ⟨hn,hd⟩ hlo hhi
    simpa only [Reference.token,row,Int.toNat_add
      (mul_nonneg (sub_nonneg.mpr hlo) (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
      Int.toNat_mul (sub_nonneg.mpr hlo) (Int.natCast_nonneg _),Int.toNat_natCast] using hidx
  · intro jr jd hn hd
    exact hb.outputAccess (jr,jd,PUnit.unit) ⟨hn,hd⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.ChunkSafety
