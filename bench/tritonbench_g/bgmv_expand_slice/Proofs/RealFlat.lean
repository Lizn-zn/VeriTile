import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealSafety

/-! Successful flat-memory execution, the independent rank-K formula and
full-cell preservation for the complete actual source. The numeric theorem
uses real erasure; the FP equivalence proof remains independent. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealFlat
open VeriTile.Triton RealLoop RealSetupSource RealSafety
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private theorem flat_frame (A : FlatAlloc) (p : Geometry) (initial final : BlockState)
    (hf : ∀ r o, p.Misses r o → final.mem r o = initial.mem r o) :
    ∀ r o, (r ≠ A.flat ∨ ∀ g, g < p.length → o ≠ A.addr p.output (p.outputAddress g)) →
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
      intro g hg hbad
      rcases hmiss with hbad' | hmiss
      · exact hbad' rfl
      · apply hmiss g hg
        rw [heq, ← hbad.1, ← hbad.2]
  · simp only [FlatAlloc.flattenState, if_neg hr]

/-- The full source's active path executes in one flat address space, writes
the complete mathematical GEMV values, and preserves every other flat cell
and region. All array bounds are explicit; no input/output equation is hidden
inside them. Initial output appears in the formula only when adding. -/
theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Parameters) (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) (initial : BlockState) (slot : Nat)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = Int.ofNat slot)
    (hu : initial.undef = (fun _ _ => 0))
    (hmeta : initial.pids 1 < A.extent (Region.cast p.indices))
    (hb : ActiveBounds A.extent p initial slot even)
    (hBN : 0 < p.BN) (hstride : 0 < p.outputStride) (hK : p.K ≤ p.BK)
    (hsplits : 0 < p.splits) (hw : p.weights ≠ p.output) :
    ∃ final, exec (A.flattenKernel (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel)
        (A.flattenState initial) = some final ∧
      (∀ g, g < p.length → final.readMem A.flat
        (A.addr p.output ((p.geometry initial slot).outputAddress g)) = RealProgram.formula p initial slot add g) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ g, g < p.length →
        o ≠ A.addr p.output ((p.geometry initial slot).outputAddress g)) →
        final.mem r o = (A.flattenState initial).mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := RealProgram.source_executes p input weight out_dtype
    cast swapped even add initial slot hm hBN hstride hK hsplits hw
  have hflat := A.execR_flatten hd hcov .triv _ initial
    (source_active_safe A.extent p input weight out_dtype cast swapped even add initial slot hm hmeta hb)
    (source_flatten_ok p input weight out_dtype cast swapped even add) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState final, hflat, ?_, flat_frame A (p.geometry initial slot) initial final hf⟩
  intro g hg
  have hout : p.output ∈ A.regions := by
    by_contra hnot
    have hbound := hb.output g hg
    rw [hcov p.output hnot] at hbound
    exact Nat.not_lt_zero _ hbound
  rw [A.flattenState_readMem hd final hout (hb.output g hg)]
  exact hv g hg

/-- The sentinel path needs only the metadata-cell bound. It executes in flat
memory and preserves every full cell, including all typed input/output data. -/
theorem sentinel_skips (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Parameters) (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) (initial : BlockState)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = -1)
    (hu : initial.undef = (fun _ _ => 0))
    (hmeta : initial.pids 1 < A.extent (Region.cast p.indices)) :
    ∃ final, exec (A.flattenKernel (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel)
        (A.flattenState initial) = some final ∧ final.mem = (A.flattenState initial).mem := by
  obtain ⟨final, he, hf⟩ := RealProgram.sentinel_skips p input weight out_dtype cast swapped even add initial hm
  have hflat := A.execR_flatten hd hcov .triv _ initial
    (source_sentinel_safe A.extent p input weight out_dtype cast swapped even add initial hm hmeta)
    (source_flatten_ok p input weight out_dtype cast swapped even add) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState final, hflat, ?_⟩
  funext r o
  exact A.flattenState_mem_congr (fun r o => congrFun (congrFun hf r) o) r o

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealFlat
