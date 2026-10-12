import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardSafety
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardCandidate

/-! Both complete forward sources in a single flat allocation. The independent
input formula covers all chunks and the frame preserves complete tagged cells.
The zero oracle used internally by the existing flattening bridge adds no input
premise: flattenState already resets that field, and all source loads here are
unmasked. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardFlat
open VeriTile.Triton RealProgram
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) (p : Layout) (S D O seed : RegionName) (hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) : ComputeKernel :=
  if swapped then TypedSource.forwardSumLeft storage gate seedStorage S D O seed H
    p.N p.K p.V p.BK p.BV hasSeed
  else TypedSource.forward storage gate seedStorage S D O seed H
    p.N p.K p.V p.BK p.BV hasSeed

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (O : RegionName) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ (t : Nat) (e : Fin p.BK) (j : Fin p.BV),
    t < p.N ∧ A.addr O (address p l t e.val j.val) = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (O : RegionName)
    (u v : BlockState) (hf : BlockState.WriteWithin (RealForwardLoop.footprint p l O) u v) :
    BlockState.WriteWithin (footprint A p l O) (A.flattenState u) (A.flattenState v) := by
  intro rg off hmiss
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨rg',off'⟩ := cell
      obtain ⟨_,heq,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      rintro ⟨rfl,t,e,j,ht,rfl⟩
      exact hmiss ⟨rfl,t,e,j,ht,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (S D O seed : RegionName) (a : Inputs p) (swapped hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (u : BlockState)
    (hN : 0 < p.N) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V) (hSO : S ≠ O) (hDO : D ≠ O)
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem S (address p (launch u) t e.val j.val) = a.source t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p (launch u) t) = a.gate t)
    (hSeed : hasSeed = Bool.true → ∀ e j,
      u.readMem seed (seedAddress p (launch u) e.val j.val) = a.seed e j)
    (hb : RealForwardSafety.Bounds p (launch u) S D O seed hasSeed A.extent) :
    ∃ v, exec (A.flattenKernel (source swapped p S D O seed hasSeed storage gate seedStorage H).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧
      (∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
        v.readMem A.flat (A.addr O (address p (launch u) t e.val j.val)) = RealMath.forward a hasSeed t e j) ∧
      BlockState.WriteWithin (footprint A p (launch u) O) (A.flattenState u) v := by
  let u0 : BlockState := { u with undef := fun _ _ => 0 }
  obtain ⟨v,he,ho,hf⟩ := RealForward.original_executes p S D O seed a hasSeed u0
    storage gate seedStorage H hN hBK hBV hSO hDO hS hD hSeed
  have hs := RealForwardSafety.source_safe p S D O seed a hasSeed storage gate seedStorage H u0
    hN hSO hDO (noCollision p (launch u0) hBK hBV) hS hD hSeed A.extent hb
  have hflat := A.exec_flatten hd hcov _ u0 hs
    (RealForwardSafety.source_flatten_ok p S D O seed hasSeed storage gate seedStorage H) rfl
  rw [he,Option.map_some] at hflat
  refine ⟨A.flattenState v,?_,?_,flatten_frame A p (launch u) O u0 v hf⟩
  · cases swapped
    · exact hflat
    · change exec (A.flattenKernel (TypedSource.forwardSumLeft storage gate seedStorage S D O seed H
        p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel) (A.flattenState u) = _
      rw [← RealForwardCandidate.flat_exec_equal]
      exact hflat
  · intro t e j ht
    have hoff := hb.output t e j ht
    have hr : O ∈ A.regions := by
      by_contra hn
      rw [hcov O hn] at hoff
      exact Nat.not_lt_zero _ hoff
    rw [A.flattenState_readMem hd v hr hoff]
    exact ho t e j ht

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardFlat
