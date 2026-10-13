import bench.tritonbench_g.index_select_cat.Proofs.Source
import bench.tritonbench_g.index_select_cat.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem source_terminates (out idx inp : RegionName) (n c s0 s1 BI BC : Nat)
    (s : BlockState) :
    ∃ t, exec (mathematicalSource Bool.false out idx inp n c s0 s1 BI BC) s = some t := by
  simp [exec, mathematicalSource, source, ComputeKernel.eraseDType,ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType, FloatDType.toTileDType,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind, Except.bind,bind,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.expandDim,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,TileShape.insertAxis,TileShape.dropInsertedIndex]

theorem source_memory (out idx inp : RegionName) (n c s0 s1 BI BC : Nat)
    (s t : BlockState) (ids : Fin BI → Int) (xs : Fin BI → Fin BC → ℝ)
    (hi : ∀ (i : Fin BI), s.pids 0 * BI + i.val < n →
      s.readMemValue .int idx (s.pids 0 * BI + i.val) = ids i)
    (hx : ∀ (i : Fin BI) (j : Fin BC), s.pids 0 * BI + i.val < n → s.pids 1 * BC + j.val < c →
      s.readMem inp ((ids i).toNat * s0 + (s.pids 1 * BC + j.val) * s1) = xs i j)
    (hn : ∀ (i : Fin BI), s.pids 0 * BI + i.val < n → 0 ≤ ids i)
    (he : exec (mathematicalSource Bool.false out idx inp n c s0 s1 BI BC) s = some t)
    (r : RegionName) (o : Nat) :
    t.mem r o = gatherCell BI BC (s.pids 0) (s.pids 1) n c s0 s1 xs out r o (s.mem r o) := by
  simp [exec, mathematicalSource, source, ComputeKernel.eraseDType,ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType, FloatDType.toTileDType,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind, Except.bind,bind,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.expandDim,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,TileShape.insertAxis,TileShape.dropInsertedIndex] at he
  cases he
  rw [fold_gather_cell, gatherCell_tile]
  congr 1
  funext cell i
  by_cases hr : s.pids 0 * BI + i.1.val < n
  · by_cases hc : s.pids 1 * BC + i.2.1.val < c
    · have hadd := signed_gather_address (ids i.1) (hn i.1 hr) s0
        (s.pids 1 * BC + i.2.1.val) s1
      simp only [Int.natCast_mul, Int.natCast_add] at hadd
      simp [hr, hc, hi i.1 hr, hx i.1 i.2.1 hr hc, hadd]
    · simp [hc]
  · simp [hr]

end VeriTile.Bench.TritonBenchG.IndexSelectCat
