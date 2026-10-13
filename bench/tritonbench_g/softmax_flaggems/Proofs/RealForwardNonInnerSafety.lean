import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerWriteback
import bench.tritonbench_g.softmax_flaggems.Proofs.RealLoopSafety

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem pass1_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N K B C row kbase start : Nat) (s : BlockState)
    (mf : TileIndex [B, C] → WithBot ℝ) (zf : TileIndex [B, C] → ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hk : s.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩)
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .real [B, C] "m" = some ⟨mf⟩)
    (hz : s.regs .real [B, C] "z" = some ⟨fun i ↦ some (zf i)⟩)
    (hin : ∀ i : TileIndex [B, C], start + i.1.val < N → kbase + i.2.1.val < K →
      tileOffset row N K kbase start i < bounds X) :
    Stmt.TraceSafeList bounds (pass1Body Y X M N K B C) s := by
  rw [pass1Body_extract]
  simp [softmax_kernel_non_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hrow, hk, hstart, hm, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt]
  intro i k hn hk'
  exact hin (i, k, PUnit.unit) hn hk'


theorem pass2_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N K B C row kbase previous start : Nat) (s : BlockState)
    (mx : Fin C → WithBot ℝ) (dz : Fin C → ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hk : s.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩)
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hprev : s.regs .nat [] "previous_multiple" = some (Tile.scalar previous))
    (hm : s.regs .real [C] "m" = some ⟨fun i ↦ mx i.1⟩)
    (hz : s.regs .real [C] "z" = some ⟨fun i ↦ some (dz i.1)⟩)
    (hin : ∀ i : TileIndex [B, C], previous - start + i.1.val < N → kbase + i.2.1.val < K →
      tileOffset row N K kbase (previous - start) i < bounds X)
    (hout : ∀ i : TileIndex [B, C], previous - start + i.1.val < N → kbase + i.2.1.val < K →
      tileOffset row N K kbase (previous - start) i < bounds Y) :
    Stmt.TraceSafeList bounds (pass2Body Y X M N K B C) s := by
  simp [pass2Body, softmax_kernel_non_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hrow, hk, hstart, hprev, hm, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt]
  constructor
  · intro i k hn hk'
    exact hin (i, k, PUnit.unit) hn hk'
  · intro i k hn hk'
    exact hout (i, k, PUnit.unit) hn hk'

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
