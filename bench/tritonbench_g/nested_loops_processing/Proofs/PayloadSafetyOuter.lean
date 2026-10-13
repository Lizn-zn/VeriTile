import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetyOuter
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadSafetyMiddle
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadFinish

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton TypedSource FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem outer_prefix_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hin : ∀ lane, (a lane).2 < bounds (a lane).1) :
    Stmt.TraceSafeList bounds (outerPrefix dtype src dst sm sn) s := by
  simp [outerPrefix, PayloadAlgorithm.middle, PayloadAlgorithm.outer, PayloadAlgorithm.load,
    PayloadAlgorithm.inputAdvance, FullReal.inner, FullReal.middle, FullReal.outer, FullReal.source, kernel, ComputeKernel.surfaceBody,
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
    stepStmts, stepStmt, evalOp.eq_def, ha, Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.vec, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, hin]
theorem outer_suffix_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) : Stmt.TraceSafeList bounds (outerSuffix dtype src dst sm sn) s := by
  simp [outerSuffix, PayloadAlgorithm.middle, PayloadAlgorithm.outer, PayloadAlgorithm.load,
    PayloadAlgorithm.inputAdvance, FullReal.inner, FullReal.middle, FullReal.outer, FullReal.source, kernel, ComputeKernel.surfaceBody,
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
    NumericDType.add, NumericDType.mul]
  split <;> trivial
theorem outer_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (hin : ∀ lane n, n < 7 →
      (a lane).2 + n * (2 * sn).toNat < bounds (a lane).1)
    (hout : ∀ lane n, n < 12 →
      (c lane).2 + n * (2 * sn).toNat < bounds (c lane).1) :
    Stmt.TraceSafeList bounds (PayloadAlgorithm.outer dtype src dst sm sn) s := by
  rw [outer_decomposition]
  apply TraceProofs.append_safe
  · apply TraceProofs.append_safe
    · apply outer_prefix_safe dtype bounds src dst sm sn s a ha
      intro lane; have h := hin lane 0 (by decide); omega
    · intro t ht
      have he := outer_prefix_step dtype src dst sm sn s a ha
      have hteq : t = s.setReg "a1" dtype [2,2] ⟨loaded dtype s a⟩ := Option.some.inj (ht.symm.trans he)
      subst t
      apply singleton_safe
      rw [Stmt.TraceSafe]
      apply middle_loop_safe dtype bounds src dst sm sn _ a c (loaded dtype s a)
        (by simpa using ha) (by simpa using hc) (by simp)
      · intro lane n hn; exact hin lane (n + 1) (by omega)
      · exact hout
  · intro t _
    exact outer_suffix_safe dtype bounds src dst sm sn t

theorem outer_loop_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (hin : ∀ lane n, n < 14 →
      (a lane).2 + n * (2 * sn).toNat < bounds (a lane).1)
    (hout : ∀ lane n, n < 24 →
      (c lane).2 + n * (2 * sn).toNat < bounds (c lane).1) :
    Stmt.forRangeTraceSafe bounds "i" 0 2 1 (PayloadAlgorithm.outer dtype src dst sm sn) s := by
  let s0 := s.setReg "i" .nat [] (Tile.scalar 0)
  let t := outerState dtype s0 a c (2 * sn).toNat
  have e0 : stepStmts (PayloadAlgorithm.outer dtype src dst sm sn) s0 = some t :=
    outer_step dtype src dst sm sn s0 a c
      (by simpa [s0] using ha) (by simpa [s0] using hc)
  refine range_two_safe bounds "i" (PayloadAlgorithm.outer dtype src dst sm sn) s t ?_ e0 ?_
  · apply outer_safe dtype bounds src dst sm sn s0 a c
      (by simpa [s0] using ha) (by simpa [s0] using hc)
    · intro lane n hn; exact hin lane n (by omega)
    · intro lane n hn; exact hout lane n (by omega)
  · have regs := outer_registers dtype s0 a c (2 * sn).toNat
    apply outer_safe dtype bounds src dst sm sn (t.setReg "i" .nat [] (Tile.scalar 1))
      (advances 7 a (2 * sn).toNat) (advances 12 c (2 * sn).toNat)
      (by simpa [t] using regs.1) (by simpa [t] using regs.2)
    · intro lane n hn
      have h := hin lane (n + 7) (by omega)
      simp only [advances_value]
      nlinarith
    · intro lane n hn
      have h := hout lane (n + 12) (by omega)
      simp only [advances_value]
      nlinarith

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
