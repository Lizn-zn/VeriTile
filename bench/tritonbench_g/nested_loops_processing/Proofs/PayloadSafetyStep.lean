import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetyStep
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadLoop
import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton TypedSource FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- The source's inner iteration accesses one fresh input tile and writes three
unmasked output tiles. Cached input values require no additional memory access. -/
theorem inner_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 v2 : Payload dtype)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs dtype [2,2] "a2" = some ⟨v2⟩)
    (hin : ∀ lane, (a lane).2 + (2 * sn).toNat < bounds (a lane).1)
    (hout0 : ∀ lane, (c lane).2 < bounds (c lane).1)
    (hout1 : ∀ lane, (c lane).2 + (2 * sn).toNat < bounds (c lane).1)
    (hout2 : ∀ lane, (c lane).2 + (2 * sn).toNat + (2 * sn).toNat < bounds (c lane).1) :
    Stmt.TraceSafeList bounds (PayloadAlgorithm.inner dtype src dst sm sn) s := by
  simp [PayloadAlgorithm.inner, PayloadAlgorithm.middle, PayloadAlgorithm.outer,
    PayloadAlgorithm.load, PayloadAlgorithm.store, PayloadAlgorithm.inputAdvance,
    PayloadAlgorithm.outputAdvance, FullReal.inner, FullReal.middle, FullReal.outer, FullReal.source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp.eq_def, ha, hc, h1, h2,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.vec, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, scatter, loaded, bump,
    hin, hout0, hout1, hout2]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
