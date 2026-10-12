import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardSafety
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardIO
open VeriTile VeriTile.Triton ComplexForwardLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Component addresses are relocated individually. No alignment condition
on the allocation base is needed. -/
def readFlat (A : FlatAlloc) (v : BlockState) (r : RegionName) (o : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (o*2)),v.readMem A.flat (A.addr r (o*2+1))⟩

def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r ≠ A.flat ∨ ∀ t, t<length → ∀ j, mask p u j=Bool.true →
    A.addr p.y (offset p u t j*2) ≠ o ∧ A.addr p.y (offset p u t j*2+1) ≠ o

/-- Full finite real IO for the two component stores. Static and temporal
access bounds, successful flat execution, the independent recurrence and
the complete tagged-cell frame are all part of this contract. -/
def Implements (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint →
    (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) →
    ComplexForwardSafety.Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        readFlat A v p.y (offset p u t j)=state p u (t+1) j) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)

theorem original_correct (p : Layout) (length : Nat) :
    Spec.Real (Implements
      (TypedSource.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType p length) := by
  intro A hd hc u hu hb
  have hs := ComplexForwardSafety.trace_safe p u length A.extent hb
  have hok := ComplexForwardSafety.flatten_ok p length
  obtain ⟨v,he,hv⟩ := ComplexForwardLoop.run p u length
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  refine ⟨?_,A.flattenState v,?_,?_,?_⟩
  · simpa only [Kernel.TraceSafe,Erasure.complexForward_body] using hs
  · have htyped : exec (A.flattenKernel
        (TypedSource.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType.toAlgKernel)
        (A.flattenState u) = exec (A.flattenKernel
        (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel)
        (A.flattenState u) := by
      simp only [exec,FlatAlloc.flattenKernel,Erasure.complexForward_body]
    rw [htyped,hbridge,he,Option.map_some]
  · intro t ht j hj
    have hb1 := hb.output t ht j hj
    have hb0 : offset p u t j*2 < A.extent p.y := by omega
    have hr : p.y ∈ A.regions := by
      by_contra hn
      rw [hc p.y hn] at hb1
      omega
    unfold readFlat
    rw [A.flattenState_readMem hd v hr hb0,A.flattenState_readMem hd v hr hb1]
    exact hv.outputs t ht j hj
  · intro r o ho
    by_cases hr : r=A.flat
    · subst r
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdcd : A.decode o with
      | none => rfl
      | some pair =>
        obtain ⟨q,z⟩ := pair
        obtain ⟨_,hz,_⟩ := A.decode_sound hdcd
        show A.trCell (v.mem q z)=A.trCell (u.mem q z)
        apply congrArg A.trCell (hv.frame q z ?_)
        by_cases hq : q=p.y
        · subst q
          right
          intro t ht j hj
          rcases ho with hne | hno
          · exact False.elim (hne rfl)
          · constructor
            · intro heq
              exact (hno t ht j hj).1 (by rw [heq]; exact hz.symm)
            · intro heq
              exact (hno t ht j hj).2 (by rw [heq]; exact hz.symm)
        · exact Or.inl hq
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem candidate_correct (p : Layout) (length : Nat) :
    Spec.Real (Implements
      (TypedSource.complexForward Bool.true p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType p length) := by
  intro A hd hc u hu hb
  obtain ⟨hs,v,he,hval,hframe⟩ := original_correct p length A hd hc u hu hb
  refine ⟨?_,v,?_,hval,hframe⟩
  · unfold Kernel.TraceSafe at hs ⊢
    rw [Erasure.complexForward_body,Candidate.complexForward_body Bool.true,← ZeroSafety.index_trace]
    rw [← Candidate.complexForward_body Bool.false]
    simpa only [Erasure.complexForward_body] using hs
  · rw [← Candidate.complexForward_flat_exec]
    exact he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardIO
