import bench.tritonbench_g.var_len_copy.Proofs.RealInvariant
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem body_safe (bounds : RegionBounds) (os ol ns : Region .int) (src dst : RegionName) (B : Nat)
    (s : BlockState) (length oldStart newStart current : Int)
    (hL : s.regs .int [] "length" = some (Tile.scalar length))
    (hO : s.regs .int [] "old_start" = some (Tile.scalar oldStart))
    (hN : s.regs .int [] "new_start" = some (Tile.scalar newStart))
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hOld : s.regs .nat [B] "old_offset" = some (Tile.vec fun i ↦ i.val))
    (hNew : s.regs .nat [B] "new_offset" = some (Tile.vec fun i ↦ i.val))
    (hin : ∀ i : TileIndex [B], (i.1.val : Int) < length →
      address oldStart current i < bounds src)
    (hout : ∀ i : TileIndex [B], (i.1.val : Int) < length →
      address newStart current i < bounds dst) :
    Stmt.TraceSafeList bounds (body os ol ns src dst B) s := by
  simp [body, source, kernel, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
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
    stepStmts, stepStmt, evalOp.eq_def, hL, hO, hN, hI, hOld, hNew,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.uop, Tile.vec,
    NumericDType.add, ComparableDType.lt]
  constructor
  · intro i hi
    exact hin (i,PUnit.unit) hi
  · intro i hi
    exact hout (i,PUnit.unit) hi

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
