import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerWriteback
import bench.tritonbench_g.softmax_flaggems.Proofs.RealLoopSafety

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem pass1_safe (bounds : RegionBounds) (P G Y : RegionName)
    (M N K B C row kbase start : Nat) (s : BlockState)
    (z : TileIndex [B,C] → ℝ)
    (hk : s.regs .nat [C] "offsets_k" = some ⟨fun i ↦ kbase+i.1.val⟩)
    (hn : s.regs .nat [B] "offsets_n" = some ⟨fun i ↦ start+i.1.val⟩)
    (ho : s.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase start⟩)
    (hz : s.regs .real [B,C] "scale" = some ⟨fun i ↦ some (z i)⟩)
    (hp : ∀ i : TileIndex [B,C], start+i.1.val<N → kbase+i.2.1.val<K →
      tileOffset row N K kbase start i < bounds P)
    (hg : ∀ i : TileIndex [B,C], start+i.1.val<N → kbase+i.2.1.val<K →
      tileOffset row N K kbase start i < bounds G)
    : Stmt.TraceSafeList bounds (pass1Body P G Y M N K B C) s := by
  simp [pass1Body, branch, source, softmax_backward_kernel_non_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hk, hn, ho, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  constructor
  · intro i k u hi hk
    exact hp (i,k,u) hi hk
  · intro i k u hi hk
    exact hg (i,k,u) hi hk

theorem pass2_safe (bounds : RegionBounds) (P G Y : RegionName)
    (M N K B C row kbase start : Nat) (s : BlockState)
    (z : Fin C → ℝ)
    (hk : s.regs .nat [C] "offsets_k" = some ⟨fun i ↦ kbase+i.1.val⟩)
    (hn : s.regs .nat [B] "offsets_n" = some ⟨fun i ↦ start+i.1.val⟩)
    (ho : s.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase start⟩)
    (hz : s.regs .real [C] "scale" = some ⟨fun i ↦ some (z i.1)⟩)
    (hp : ∀ i : TileIndex [B,C], start+i.1.val<N → kbase+i.2.1.val<K →
      tileOffset row N K kbase start i < bounds P)
    (hg : ∀ i : TileIndex [B,C], start+i.1.val<N → kbase+i.2.1.val<K →
      tileOffset row N K kbase start i < bounds G)
    (hy : ∀ i : TileIndex [B,C], start+i.1.val<N → kbase+i.2.1.val<K →
      tileOffset row N K kbase start i < bounds Y)
    : Stmt.TraceSafeList bounds (pass2Body P G Y M N K B C) s := by
  simp [pass2Body, branch, source, softmax_backward_kernel_non_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hk, hn, ho, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  constructor
  · intro i k u hi hk
    exact hp (i,k,u) hi hk
  constructor
  · intro i k u hi hk
    exact hg (i,k,u) hi hk
  · intro i k u hi hk
    exact hy (i,k,u) hi hk

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
