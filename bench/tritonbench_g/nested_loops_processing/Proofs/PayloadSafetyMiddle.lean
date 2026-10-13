import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadSafetyLoop
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadMiddle

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton TypedSource FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem middle_prefix_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hin : ∀ lane, (a lane).2 + (2 * sn).toNat < bounds (a lane).1) :
    Stmt.TraceSafeList bounds (middlePrefix dtype src dst sm sn) s := by
  simp [middlePrefix, PayloadAlgorithm.middle, PayloadAlgorithm.outer, PayloadAlgorithm.load,
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

theorem middle_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 : Payload dtype)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩)
    (hin : ∀ lane n, n < 3 →
      (a lane).2 + (n + 1) * (2 * sn).toNat < bounds (a lane).1)
    (hout : ∀ lane n, n < 6 →
      (c lane).2 + n * (2 * sn).toNat < bounds (c lane).1) :
    Stmt.TraceSafeList bounds (PayloadAlgorithm.middle dtype src dst sm sn) s := by
  rw [middle_decomposition]
  apply TraceProofs.append_safe
  · apply middle_prefix_safe dtype bounds src dst sm sn s a ha
    intro lane; have h := hin lane 0 (by decide); omega
  · intro t ht
    have he := middle_prefix_step dtype src dst sm sn s a ha
    have hteq : t = middlePrefixState dtype s a (2 * sn).toNat := Option.some.inj (ht.symm.trans he)
    subst t
    apply singleton_safe
    rw [Stmt.TraceSafe]
    apply inner_loop_safe dtype bounds src dst sm sn _ (bump a (2 * sn).toNat) c v1
      (loaded dtype s (bump a (2 * sn).toNat))
      (by simp [middlePrefixState]) (by simpa [middlePrefixState] using hc)
      (by simpa [middlePrefixState] using h1) (by simp [middlePrefixState])
    · intro lane n hn
      have h := hin lane (n + 1) (by omega)
      dsimp [bump]
      nlinarith
    · exact hout

theorem middle_loop_safe (dtype : TileDType) (bounds : RegionBounds) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 : Payload dtype)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩)
    (hin : ∀ lane n, n < 6 →
      (a lane).2 + (n + 1) * (2 * sn).toNat < bounds (a lane).1)
    (hout : ∀ lane n, n < 12 →
      (c lane).2 + n * (2 * sn).toNat < bounds (c lane).1) :
    Stmt.forRangeTraceSafe bounds "j" 0 2 1 (PayloadAlgorithm.middle dtype src dst sm sn) s := by
  let s0 := s.setReg "j" .nat [] (Tile.scalar 0)
  let t := middleState dtype s0 a c v1 (2 * sn).toNat
  have e0 : stepStmts (PayloadAlgorithm.middle dtype src dst sm sn) s0 = some t :=
    middle_step dtype src dst sm sn s0 a c v1
      (by simpa [s0] using ha) (by simpa [s0] using hc) (by simpa [s0] using h1)
  refine range_two_safe bounds "j" (PayloadAlgorithm.middle dtype src dst sm sn) s t ?_ e0 ?_
  · apply middle_safe dtype bounds src dst sm sn s0 a c v1
      (by simpa [s0] using ha) (by simpa [s0] using hc) (by simpa [s0] using h1)
    · intro lane n hn; exact hin lane n (by omega)
    · intro lane n hn; exact hout lane n (by omega)
  · have regs := middle_registers dtype s0 a c v1 (2 * sn).toNat (by simpa [s0] using h1)
    apply middle_safe dtype bounds src dst sm sn (t.setReg "j" .nat [] (Tile.scalar 1))
      (bump (bump (bump a (2 * sn).toNat) (2 * sn).toNat) (2 * sn).toNat)
      (bump (bump (bump (bump (bump (bump c (2 * sn).toNat) (2 * sn).toNat)
        (2 * sn).toNat) (2 * sn).toNat) (2 * sn).toNat) (2 * sn).toNat) v1
      (by simpa [t] using regs.1) (by simpa [t] using regs.2.1)
      (by simpa [t] using regs.2.2)
    · intro lane n hn
      have h := hin lane (n + 3) (by omega)
      dsimp [bump]
      nlinarith
    · intro lane n hn
      have h := hout lane (n + 6) (by omega)
      dsimp [bump]
      nlinarith

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
