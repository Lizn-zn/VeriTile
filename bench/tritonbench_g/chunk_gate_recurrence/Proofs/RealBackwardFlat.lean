import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardFlatExecution
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardCandidate

/-! Complete backward flat-memory correctness for both sources. DI is the next
reverse suffix, DG is the complete tile contraction and DL is suffix zero.
All other tagged cells are preserved. Allocation placement is arbitrary subject
to disjointness and the actual access bounds; DL may alias an input region. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlat
open VeriTile.Triton RealProgram RealBackwardPointers RealBackwardLoop
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) : ComputeKernel :=
  if swapped then TypedSource.backwardSumLeft storage gate incoming S D DI DG DL DS H
    p.N p.K p.V p.BK p.BV
  else TypedSource.backward storage gate incoming S D DI DG DL DS H
    p.N p.K p.V p.BK p.BV

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (DI DG DL : RegionName) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ r off,
    RealBackward.footprint p l DI DG DL (r,off) ∧ A.addr r off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (DI DG DL : RegionName)
    (u v : BlockState) (hf : BlockState.WriteWithin (RealBackward.footprint p l DI DG DL) u v) :
    BlockState.WriteWithin (footprint A p l DI DG DL) (A.flattenState u) (A.flattenState v) := by
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
      intro hw
      exact hmiss ⟨rfl,rg',off',hw,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (S D DI DG DL DS : RegionName) (a : Inputs p) (swapped : Bool)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (u : BlockState)
    (hN : 0 < p.N) (hBKpos : 0 < p.BK) (hBVpos : 0 < p.BV)
    (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V)
    (sep : Separation S D DI DG DS) (hIDL : DI ≠ DL) (hGDL : DG ≠ DL)
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
      u.readMem S (address p (launch u) t e.val j.val) = a.source t e j)
    (hDS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem DS (address p (launch u) t e.val j.val) = a.incoming t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p (launch u) t) = a.gate t)
    (hb : RealBackwardSafety.Bounds p (launch u) S D DI DG DL DS A.extent) :
    ∃ v, exec (A.flattenKernel (source swapped p S D DI DG DL DS storage gate incoming H).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧
      (∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
        v.readMem A.flat (A.addr DI (address p (launch u) t e.val j.val)) = RealMath.reverse a p.N (t+1) e j) ∧
      (∀ t, t < p.N-1 → v.readMem A.flat (A.addr DG (gradientAddress p (launch u) t)) = RealMath.gateGradient a p.N t) ∧
      (∀ e j, v.readMem A.flat (A.addr DL (seedAddress p (launch u) e.val j.val)) = RealMath.reverse a p.N 0 e j) ∧
      BlockState.WriteWithin (footprint A p (launch u) DI DG DL) (A.flattenState u) v := by
  let u0 : BlockState := { u with undef := fun _ _ => 0 }
  obtain ⟨v,he,hi,hg,hl,hframe⟩ := RealBackward.original_executes p S D DI DG DL DS a u0
    storage gate incoming H hN hBKpos hBVpos hBK hBV sep hIDL hGDL hS hDS hD
  obtain ⟨v',flatFinal,he',hflat,ha⟩ := RealBackwardFlatExecution.paired_executes A hd hcov
    p S D DI DG DL DS a storage gate incoming H u0 hN sep
    (noCollision p (launch u0) hBK hBV)
    (gradient_injective p (launch u0) (Nat.div_pos hBK hBKpos) (Nat.div_pos hBV hBVpos)) hS hDS hD hb rfl
  have hv : v' = v := Option.some.inj (he'.symm.trans he)
  subst v'
  have hread : ∀ r off, off < A.extent r → flatFinal.readMem A.flat (A.addr r off) = v.readMem r off := by
    intro r off hoff
    have hr : r ∈ A.regions := by
      by_contra hn
      rw [hcov r hn] at hoff
      exact Nat.not_lt_zero _ hoff
    calc flatFinal.readMem A.flat (A.addr r off) = (A.flattenState v).readMem A.flat (A.addr r off) := by
           simp only [BlockState.readMem,ha.mem]
         _ = v.readMem r off := A.flattenState_readMem hd v hr hoff
  refine ⟨flatFinal,?_,?_,?_,?_,?_⟩
  · cases swapped
    · exact hflat
    · change exec (A.flattenKernel (TypedSource.backwardSumLeft storage gate incoming S D DI DG DL DS H
        p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel) (A.flattenState u) = _
      rw [← RealBackwardCandidate.flat_exec_equal]
      exact hflat
  · intro t e j ht
    rw [hread DI _ (hb.outputDI t e j ht)]
    exact hi t e j ht
  · intro t ht
    rw [hread DG _ (hb.outputDG t ht)]
    exact hg t ht
  · intro e j
    rw [hread DL _ (hb.outputDL e j)]
    exact hl e j
  · intro r off hn
    rw [ha.mem]
    exact flatten_frame A p (launch u) DI DG DL u0 v hframe r off hn

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlat
