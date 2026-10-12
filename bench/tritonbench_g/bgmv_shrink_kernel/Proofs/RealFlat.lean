import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealSafety

/-! Successful flat-memory execution of the complete source, its independent
real formula, and preservation of every cell outside the actual output window. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealFlat
open VeriTile.Triton RealSetupSource RealSafety
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private theorem flat_frame (A : FlatAlloc) (p : Parameters) (initial final : BlockState)
    (hf : ∀ r o, RealProgram.Misses p initial r o → final.mem r o = initial.mem r o) :
    ∀ r o, (r ≠ A.flat ∨ ∀ n : Fin p.BN, n.val < p.N →
      o ≠ A.addr p.output (RealTail.offset p (initial.pids 1) n.val)) →
      (A.flattenState final).mem r o = (A.flattenState initial).mem r o := by
  intro r o hmiss
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState, if_pos rfl, FlatAlloc.readFlat]
    cases hd : A.decode o with
    | none => rfl
    | some cell =>
      obtain ⟨r', o'⟩ := cell
      obtain ⟨_, heq, _⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro n hn hbad
      rcases hmiss with hbad' | hmiss
      · exact hbad' rfl
      · apply hmiss n hn
        rw [heq, ← hbad.1, ← hbad.2]
  · simp only [FlatAlloc.flattenState, if_neg hr]

/-- All output lanes satisfy the rank-slice formula in one allocated flat
address space. The bounds are on actual masked accesses, not output values. -/
theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType) (swapped : Bool)
    (initial : BlockState) (slot : Nat)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = Int.ofNat slot)
    (hu : initial.undef = (fun _ _ => 0))
    (hmeta : initial.pids 1 < A.extent (Region.cast p.indices))
    (hb : ActiveBounds A.extent p initial slot)
    (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) (hN : p.N ≤ p.BN) :
    ∃ final, exec (A.flattenKernel (p.source storage out_dtype swapped).eraseDType.toAlgKernel)
        (A.flattenState initial) = some final ∧
      (∀ n, n < p.N → final.readMem A.flat
        (A.addr p.output (RealTail.offset p (initial.pids 1) n)) = RealProgram.formula p initial slot n) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ n : Fin p.BN, n.val < p.N →
        o ≠ A.addr p.output (RealTail.offset p (initial.pids 1) n.val)) →
        final.mem r o = (A.flattenState initial).mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := RealProgram.source_executes p storage out_dtype swapped
    initial slot hm hS hstride hN
  have hflat := A.execR_flatten hd hcov .triv _ initial
    (source_active_safe A.extent p storage out_dtype swapped initial slot hm hmeta hS hb)
    (source_flatten_ok p storage out_dtype swapped) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState final, hflat, ?_, flat_frame A p initial final hf⟩
  intro n hn
  have hbound := hb.output ⟨n, lt_of_lt_of_le hn hN⟩ hn
  have hout : p.output ∈ A.regions := by
    by_contra hnot
    rw [hcov p.output hnot] at hbound
    exact Nat.not_lt_zero _ hbound
  rw [A.flattenState_readMem hd final hout hbound]
  exact hv n hn

/-- The sentinel performs only its signed metadata read. No active-array
bounds or positive dimensions are required to preserve all flat memory. -/
theorem sentinel_skips (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType) (swapped : Bool)
    (initial : BlockState)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = -1)
    (hu : initial.undef = (fun _ _ => 0))
    (hmeta : initial.pids 1 < A.extent (Region.cast p.indices)) :
    ∃ final, exec (A.flattenKernel (p.source storage out_dtype swapped).eraseDType.toAlgKernel)
        (A.flattenState initial) = some final ∧ final.mem = (A.flattenState initial).mem := by
  obtain ⟨final, he, hf⟩ := RealProgram.sentinel_skips p storage out_dtype swapped initial hm
  have hflat := A.execR_flatten hd hcov .triv _ initial
    (source_sentinel_safe A.extent p storage out_dtype swapped initial hm hmeta)
    (source_flatten_ok p storage out_dtype swapped) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState final, hflat, ?_⟩
  funext r o
  exact A.flattenState_mem_congr (fun r o => congrFun (congrFun hf r) o) r o

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealFlat
