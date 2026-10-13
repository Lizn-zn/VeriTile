import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetyMiddle
import bench.tritonbench_g.nested_loops_processing.Proofs.RealFinish

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem outer_prefix_safe (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hin : ∀ lane, (a lane).2 < bounds (a lane).1) :
    Stmt.TraceSafeList bounds (outerPrefix src dst sm sn) s := by
  simp [outerPrefix, outer, source, kernel, ComputeKernel.surfaceBody,
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
theorem outer_suffix_safe (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) : Stmt.TraceSafeList bounds (outerSuffix src dst sm sn) s := by
  simp [outerSuffix, outer, source, kernel, ComputeKernel.surfaceBody,
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
theorem outer_safe (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (hin : ∀ lane n, n < 7 →
      (a lane).2 + n * (2 * sn).toNat < bounds (a lane).1)
    (hout : ∀ lane n, n < 12 →
      (c lane).2 + n * (2 * sn).toNat < bounds (c lane).1) :
    Stmt.TraceSafeList bounds (outer src dst sm sn) s := by
  rw [outer_decomposition]
  apply TraceProofs.append_safe
  · apply TraceProofs.append_safe
    · apply outer_prefix_safe bounds src dst sm sn s a ha
      intro lane; have h := hin lane 0 (by decide); omega
    · intro t ht
      have he := outer_prefix_step src dst sm sn s a ha
      have hteq : t = s.setReg "a1" .int [2,2] ⟨loaded s a⟩ := Option.some.inj (ht.symm.trans he)
      subst t
      apply singleton_safe
      rw [Stmt.TraceSafe]
      apply middle_loop_safe bounds src dst sm sn _ a c (loaded s a)
        (by simpa using ha) (by simpa using hc) (by simp)
      · intro lane n hn; exact hin lane (n + 1) (by omega)
      · exact hout
  · intro t _
    exact outer_suffix_safe bounds src dst sm sn t

theorem advances_value (a : PointerTile) (delta n : Nat) (lane : TileIndex [2,2]) :
    advances n a delta lane = ((a lane).1, (a lane).2 + n * delta) := by
  induction n with
  | zero => simp [advances]
  | succ n ih => simp [advances, bump, ih, Nat.succ_mul, Nat.add_assoc]

theorem outer_loop_safe (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (hin : ∀ lane n, n < 14 →
      (a lane).2 + n * (2 * sn).toNat < bounds (a lane).1)
    (hout : ∀ lane n, n < 24 →
      (c lane).2 + n * (2 * sn).toNat < bounds (c lane).1) :
    Stmt.forRangeTraceSafe bounds "i" 0 2 1 (outer src dst sm sn) s := by
  let s0 := s.setReg "i" .nat [] (Tile.scalar 0)
  let t := outerState s0 a c (2 * sn).toNat
  have e0 : stepStmts (outer src dst sm sn) s0 = some t :=
    outer_step src dst sm sn s0 a c
      (by simpa [s0] using ha) (by simpa [s0] using hc)
  refine range_two_safe bounds "i" (outer src dst sm sn) s t ?_ e0 ?_
  · apply outer_safe bounds src dst sm sn s0 a c
      (by simpa [s0] using ha) (by simpa [s0] using hc)
    · intro lane n hn; exact hin lane n (by omega)
    · intro lane n hn; exact hout lane n (by omega)
  · have regs := outer_registers s0 a c (2 * sn).toNat
    apply outer_safe bounds src dst sm sn (t.setReg "i" .nat [] (Tile.scalar 1))
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

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
