import bench.tritonbench_g.max_reduction.Proofs.DimensionFrame
import bench.tritonbench_g.max_reduction.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Full typed value/index IO: all masked input cells, both output channels,
successful flat execution, and every cell outside both active output windows.
The independently supplied input window contains original scalar values. -/
def DimensionIO (storage : Storage) (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) : Prop :=
  ∀ (A : FlatAlloc), A.Disjoint → A.regions = [inp,ov,oi] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (xs : TileIndex [BM,BN] → ℝ),
    s.undef = (fun _ _ => 0) →
    (∀ i : Fin BM, ∀ j : Fin BN, s.pids 0 * BM + i.val < M → j.val < N →
      (s.pids 0 * BM + i.val) * N * K + j.val * K + s.pids 1 < A.extent inp) →
    (∀ i : Fin BM, s.pids 0 * BM + i.val < M →
      maxKernelOutOffset s K BM i < A.extent ov) →
    (∀ i : Fin BM, s.pids 0 * BM + i.val < M →
      maxKernelOutOffset s K BM i < A.extent oi) →
    (∀ i : Fin BM, ∀ j : Fin BN, s.pids 0 * BM + i.val < M → j.val < N →
      s.readMem inp ((s.pids 0 * BM + i.val) * N * K + j.val * K + s.pids 1) =
        xs (i,j,PUnit.unit)) →
    ∃ t,
      exec (A.flattenKernel (dimensionSource storage v inp ov oi M N K BM BN).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ i : Fin BM, s.pids 0 * BM + i.val < M →
        t.readMem A.flat (A.addr ov (maxKernelOutOffset s K BM i)) =
          rowMaximum M N BM BN (s.pids 0) xs i ∧
        t.readMemValue .int A.flat (A.addr oi (maxKernelOutOffset s K BM i)) =
          Int.ofNat (rowMaximumIndex M N BM BN (s.pids 0) xs i)) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i : Fin BM, s.pids 0 * BM + i.val < M →
        o ≠ A.addr ov (maxKernelOutOffset s K BM i) ∧
        o ≠ A.addr oi (maxKernelOutOffset s K BM i)) →
        t.mem r o = (A.flattenState s).mem r o)

/-- Assembly uses the real execution of the entire source and the proved
flat-memory bridge; no output or precomputed reduction is assumed. -/
theorem dimensionIO_correct (storage : Storage) (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) (hN : 0 < BN) (hRegions : ov ≠ oi)
    (hOutInj : Function.Injective (fun i : Fin BM => i.val * K)) :
    DimensionIO storage v inp ov oi M N K BM BN := by
  intro A hd hr hc s xs hu hi hv hx hpins
  have ho : Function.Injective (fun i : Fin BM => maxKernelOutOffset s K BM i) := by
    intro i j hij
    apply hOutInj
    simpa [maxKernelOutOffset, Nat.add_mul] using hij
  obtain ⟨t, he⟩ := dimension_terminates inp ov oi M N K BM BN hN s
  have htrace := dimension_traceSafe inp ov oi M N K BM BN hN A.extent s hi hv hx
  have hok := dimension_flattenOk inp ov oi M N K BM BN
  refine ⟨A.flattenState t, ?_, ?_, ?_⟩
  · rw [dimension_flat_exec,
      A.exec_flatten hd hc _ s htrace hok hu, he, Option.map_some]
  · intro i ha
    have hov : ov ∈ A.regions := by simp [hr]
    have hoi : oi ∈ A.regions := by simp [hr]
    rw [A.flattenState_readMem hd t hov (hv i ha),
      A.flattenState_readMemValue hd t hoi (hx i ha) .int ⟨by decide, by decide⟩]
    obtain ⟨hvalue, hindex⟩ := dimension_readback inp ov oi M N K BM BN hN s t he ho hRegions i ha
    rw [dimension_value_formula inp M N K BM BN hN s xs hpins i] at hvalue
    rw [dimension_index_formula inp M N K BM BN s xs hpins i] at hindex
    exact ⟨hvalue,hindex⟩
  · intro r' o' hcond
    by_cases hr' : r' = A.flat
    · subst hr'
      show (A.flattenState t).mem A.flat o' = (A.flattenState s).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some pair =>
        obtain ⟨r,o⟩ := pair
        obtain ⟨hrmem,hoeq,holt⟩ := A.decode_sound hdec
        show A.trCell (t.mem r o) = A.trCell (s.mem r o)
        apply congrArg A.trCell
        apply dimension_frame inp ov oi M N K BM BN hN s t he r o
        · intro i ha
          by_cases hro : r = ov
          · subst hro
            apply Or.inr
            intro hoj
            rcases hcond with hf | hn
            · exact hf rfl
            · exact (hn i ha).1 (by rw [hoeq, hoj])
          · exact Or.inl hro
        · intro i ha
          by_cases hro : r = oi
          · subst hro
            apply Or.inr
            intro hoj
            rcases hcond with hf | hn
            · exact hf rfl
            · exact (hn i ha).2 (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr']

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction
