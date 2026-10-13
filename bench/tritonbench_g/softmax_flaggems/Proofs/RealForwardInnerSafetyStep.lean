import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerWriteLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.RealLoopSafety

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem pass1_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B row start : Nat) (s : BlockState)
    (mf : TileIndex [B] → WithBot ℝ) (zf : TileIndex [B] → ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hp : s.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N)))
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .real [B] "m" = some ⟨mf⟩)
    (hz : s.regs .real [B] "z" = some ⟨fun i ↦ some (zf i)⟩)
    (hin : ∀ i : TileIndex [B], start + i.1.val < N →
      row * N + (start + i.1.val) < bounds X) :
    Stmt.TraceSafeList bounds (pass1Body Y X M N B) s := by
  rw [pass1Body_extract]
  simp [softmax_kernel_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hrow, hp, hstart, hm, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt]
  intro i hn
  exact hin (i, PUnit.unit) hn


theorem pass2_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B row previous start : Nat) (s : BlockState)
    (mx : WithBot ℝ) (dz : ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hp : s.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N)))
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hprev : s.regs .nat [] "previous_multiple" = some (Tile.scalar previous))
    (ho : s.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N)))
    (hm : s.regs .real [] "m" = some (Tile.scalar mx))
    (hz : s.regs .real [] "z" = some (Tile.scalar (some dz)))
    (hin : ∀ i : TileIndex [B], previous - start + i.1.val < N →
      row * N + (previous - start + i.1.val) < bounds X)
    (hout : ∀ i : TileIndex [B], previous - start + i.1.val < N →
      row * N + (previous - start + i.1.val) < bounds Y) :
    Stmt.TraceSafeList bounds (pass2Body Y X M N B) s := by
  simp [pass2Body, fallback, source, softmax_kernel_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hrow, hp, hstart, hprev, ho, hm, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt]
  constructor
  · intro i hn
    exact hin (i, PUnit.unit) hn
  · intro i hn
    exact hout (i, PUnit.unit) hn

theorem full_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B row start : Nat) (s : BlockState)
    (mf : TileIndex [B] → WithBot ℝ) (zf : TileIndex [B] → ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hp : s.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N)))
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .real [B] "m" = some ⟨mf⟩)
    (hz : s.regs .real [B] "z" = some ⟨fun i ↦ some (zf i)⟩)
    (hin : ∀ i : TileIndex [B], row * N + (start + i.1.val) < bounds X) :
    Stmt.TraceSafeList bounds (fullBody Y X M N B) s := by
  rw [fullBody_extract]
  simp [softmax_kernel_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hrow, hp, hstart, hm, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt]
  intro i
  exact hin (i, PUnit.unit)

theorem full_store_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B row previous start : Nat) (s : BlockState)
    (mx : WithBot ℝ) (dz : ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hp : s.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N)))
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hprev : s.regs .nat [] "previous_multiple" = some (Tile.scalar previous))
    (ho : s.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N)))
    (hm : s.regs .real [] "m" = some (Tile.scalar mx))
    (hz : s.regs .real [] "z" = some (Tile.scalar (some dz)))
    (hin : ∀ i : TileIndex [B], row * N + (previous - start + i.1.val) < bounds X)
    (hout : ∀ i : TileIndex [B], row * N + (previous - start + i.1.val) < bounds Y) :
    Stmt.TraceSafeList bounds (fullStoreBody Y X M N B) s := by
  simp [fullStoreBody, fallback, source, softmax_kernel_inner,
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
    stepStmts, stepStmt, evalOp.eq_def, hrow, hp, hstart, hprev, ho, hm, hz,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt]
  constructor
  · intro i
    exact hin (i, PUnit.unit)
  · intro i
    exact hout (i, PUnit.unit)

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

#guard_msgs (drop info) in
#auditModuleAxioms
