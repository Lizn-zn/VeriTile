import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetyOuter
import bench.tritonbench_g.nested_loops_processing.Proofs.RealOuterInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.RealBounds

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem prefix_safe (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) : Stmt.TraceSafeList bounds (prefixBody src dst sm sn) s := by
  simp [prefixBody, source, kernel, ComputeKernel.surfaceBody,
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
    stepStmts, stepStmt, evalOp.eq_def, Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.vec, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul]/-- Full nested-loop access safety. Bounds cover precisely the largest input and
output offsets, without requiring bounds for the unused final pointer cursors. -/
theorem source_safe (bounds : RegionBounds) (src dst : RegionName) (sm sn : Nat)
    (s : BlockState) (hsrc : sm + 27 * sn < bounds src)
    (hdst : sm + 47 * sn < bounds dst) :
    (source src dst sm sn).TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  rw [source_decomposition]
  apply TraceProofs.append_safe
  · exact prefix_safe bounds src dst sm sn s
  · intro t ht
    have he := prefix_step src dst sm sn s
    have hteq : t = prefixState s src dst sm sn := Option.some.inj (ht.symm.trans he)
    subst t
    apply singleton_safe
    rw [Stmt.TraceSafe]
    apply outer_loop_safe bounds src dst sm sn _ (initialA src sm sn) (initialC dst sm sn)
      (by simp [prefixState]) (by simp [prefixState])
    · intro lane n hn
      have hd : (2 * (sn : Int)).toNat = 2 * sn := by omega
      rw [initialA_nonnegative, hd]
      have h := tilePointer_bound bounds src sm sn (2 * n) 26 (by omega) hsrc lane
      dsimp [tilePointer] at h ⊢
      nlinarith
    · intro lane n hn
      have hd : (2 * (sn : Int)).toNat = 2 * sn := by omega
      rw [initialC_nonnegative, hd]
      have h := tilePointer_bound bounds dst sm sn (2 * n) 46 (by omega) hdst lane
      dsimp [tilePointer] at h ⊢
      nlinarith

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
