import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerOneTileRun
import bench.tritonbench_g.softmax_flaggems.Proofs.RealTrace

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem oneTile_body_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N K B C row kbase : Nat) (hB : 0 < B) (s : BlockState)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hk : s.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩)
    (hin : ∀ i : TileIndex [B, C], i.1.val < N → kbase + i.2.1.val < K →
      tileOffset row N K kbase 0 i < bounds X)
    (hout : ∀ i : TileIndex [B, C], i.1.val < N → kbase + i.2.1.val < K →
      tileOffset row N K kbase 0 i < bounds Y) :
    Stmt.TraceSafeList bounds (oneTileBody Y X M N K B C) s := by
  simp [oneTileBody, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp.eq_def, hrow, hk,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, hB]
  constructor
  · intro i k hn hk'
    simpa [tileOffset] using hin (i, k, PUnit.unit) hn hk'
  · intro i k hn hk'
    simpa [tileOffset] using hout (i, k, PUnit.unit) hn hk'

 theorem oneTile_prefix_safe (bounds : RegionBounds) (C : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (oneTilePrefix C) s := by
  simp [oneTilePrefix, RealCandidate.indexPrefix, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt.eq_def, stepStmt, evalOp.eq_def]

 theorem oneTile_source_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hB : 0 < B) (hr : s.pids 0 < M)
    (hX : M * N * K ≤ bounds X) (hY : M * N * K ≤ bounds Y) :
    (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨st, hpre, _, _, _, _, hrow, hk⟩ := oneTile_prefix_run s C
  obtain ⟨no, hsource⟩ := oneTile_decomposition Y X M N K B C
  unfold Kernel.TraceSafe
  rw [hsource]
  apply RealTrace.append bounds _ _ s st (oneTile_prefix_safe bounds C s) hpre
  apply RealTrace.singleton
  simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, Tile.scalar_data,
    Bool.true_eq, ↓reduceIte, true_and]
  apply oneTile_body_safe bounds Y X M N K B C (s.pids 0) (s.pids 1 * C) hB st hrow hk
  · intro i hn hk'
    have h := Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk') hX
    simpa [tileOffset] using h
  · intro i hn hk'
    have h := Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk') hY
    simpa [tileOffset] using h

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
