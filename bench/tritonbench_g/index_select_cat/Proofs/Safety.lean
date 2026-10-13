import bench.tritonbench_g.index_select_cat.Proofs.Source
import bench.tritonbench_g.index_select_cat.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem source_flattenOk (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    (mathematicalSource Bool.false out idx inp n c s0 s1 BI BC).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, mathematicalSource,source,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

theorem source_traceSafe (out idx inp : RegionName) (n c s0 s1 BI BC : Nat)
    (bounds : RegionBounds) (s : BlockState) (ids : Fin BI → Int)
    (hi : ∀ (i : Fin BI), s.pids 0 * BI + i.val < n →
      s.readMemValue .int idx (s.pids 0 * BI + i.val) = ids i)
    (hn : ∀ (i : Fin BI), s.pids 0 * BI + i.val < n → 0 ≤ ids i)
    (hr : ∀ (i : Fin BI) (j : Fin BC), s.pids 0 * BI + i.val < n →
      s.pids 1 * BC + j.val < c →
      (ids i).toNat * s0 + (s.pids 1 * BC + j.val) * s1 < bounds inp)
    (hx : ∀ (i : Fin BI), s.pids 0 * BI + i.val < n →
      s.pids 0 * BI + i.val < bounds idx)
    (hw : ∀ (i : Fin BI) (j : Fin BC), s.pids 0 * BI + i.val < n →
      s.pids 1 * BC + j.val < c →
      (s.pids 0 * BI + i.val) * s0 + (s.pids 1 * BC + j.val) * s1 < bounds out) :
    Kernel.TraceSafe bounds
      (mathematicalSource Bool.false out idx inp n c s0 s1 BI BC).toAlgKernel s := by
  simp [Kernel.TraceSafe,mathematicalSource,source,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MaskOpt.SafeAt,MemAccess.SafeAt,stepStmt,evalOp.eq_def,
    Op.PointerAddressesSafeOn,Op.MemorySafe,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MaskOpt.Active,BlockState.setReg,tile_elementwise,Bool.and_eq_true,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.expandDim,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,TileShape.insertAxis,TileShape.dropInsertedIndex]
  refine ⟨hx,?_,hw⟩
  intro i j hir hjc
  have hadd := signed_gather_address (ids i) (hn i hir) s0 (s.pids 1 * BC + j.val) s1
  simp only [Int.natCast_mul,Int.natCast_add] at hadd
  change Int.toNat ((if s.pids 0 * BI + i.val < n then
    s.readMemValue .int idx (s.pids 0 * BI + i.val) * (s0 : Int)
    else BlockState.defaultCarrier .int * (s0 : Int)) +
    ((s.pids 1 : Int) * BC + (j.val : Int)) * s1) < bounds inp
  simpa only [if_pos hir,hi i hir,hadd] using hr i j hir hjc

end VeriTile.Bench.TritonBenchG.IndexSelectCat
