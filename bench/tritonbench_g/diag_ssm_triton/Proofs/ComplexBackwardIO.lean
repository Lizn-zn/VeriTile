import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardSafety
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardRun
import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroIO
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardIO
open VeriTile VeriTile.Triton ComplexBackwardLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Relocate each component separately, allowing odd allocation bases. -/
def readFlat (A : FlatAlloc) (v : BlockState) (r : RegionName) (o : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (o*2)),v.readMem A.flat (A.addr r (o*2+1))⟩

def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)

/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
def Implements (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → ComplexBackwardRun.Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ComplexBackwardSafety.Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        readFlat A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → readFlat A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → readFlat A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)

theorem read_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r=0) (v : BlockState) (r : RegionName) (o : Nat)
    (hb : o*2+1 < A.extent r) : readFlat A (A.flattenState v) r o=ComplexMemory.read v r o := by
  have hb0 : o*2 < A.extent r := by omega
  have hr : r ∈ A.regions := by
    by_contra hn
    rw [hc r hn] at hb
    omega
  unfold readFlat ComplexMemory.read
  rw [A.flattenState_readMem hd v hr hb0,A.flattenState_readMem hd v hr hb]

theorem original_correct (p : Layout) (length : Nat) :
    Spec.Real (Implements (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) := by
  intro hsep A hd hc u hu hb
  by_cases hz : length=0
  · subst length
    have hstatic : ZeroSafety.complexBackwardBounds A.extent u p.l p.gs p.gl p.batch p.dim p.B := by
      refine ⟨?_,?_,?_⟩
      · intro j hj; exact hb.transition ⟨j,PUnit.unit⟩ (decide_eq_true_eq.mpr hj)
      · intro j hj; exact hb.gradInitial ⟨j,PUnit.unit⟩ (decide_eq_true_eq.mpr hj)
      · intro j hj; exact hb.gradTransition ⟨j,PUnit.unit⟩ (decide_eq_true_eq.mpr hj)
    obtain ⟨hs,v,he,hz,hf⟩ := ZeroIO.complexBackward_correct Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      p.batch p.dim p.B A hd hc u hstatic
    refine ⟨hs,v,he,(fun t ht => by omega),?_,?_,?_⟩
    · intro j hj
      apply Complex.ext
      · apply ZeroIO.zero_readback (hz _ _ ?_)
        exact ⟨rfl,Or.inl ⟨j.1,of_decide_eq_true hj,rfl⟩⟩
      · apply ZeroIO.zero_readback (hz _ _ ?_)
        exact ⟨rfl,Or.inr (Or.inl ⟨j.1,of_decide_eq_true hj,rfl⟩)⟩
    · intro j hj
      apply Complex.ext
      · apply ZeroIO.zero_readback (hz _ _ ?_)
        exact ⟨rfl,Or.inr (Or.inr (Or.inl ⟨j.1,of_decide_eq_true hj,rfl⟩))⟩
      · apply ZeroIO.zero_readback (hz _ _ ?_)
        exact ⟨rfl,Or.inr (Or.inr (Or.inr ⟨j.1,of_decide_eq_true hj,rfl⟩))⟩
    · intro r o ho
      apply hf r o
      rintro ⟨hr',hw⟩
      rcases ho with hn | ⟨_,hs,hl⟩
      · exact hn hr'
      · rcases hw with ⟨j,hj,hw⟩ | ⟨j,hj,hw⟩ | ⟨j,hj,hw⟩ | ⟨j,hj,hw⟩
        · exact (hs ⟨j,PUnit.unit⟩ (decide_eq_true_eq.mpr hj)).1 hw.symm
        · exact (hs ⟨j,PUnit.unit⟩ (decide_eq_true_eq.mpr hj)).2 hw.symm
        · exact (hl ⟨j,PUnit.unit⟩ (decide_eq_true_eq.mpr hj)).1 hw.symm
        · exact (hl ⟨j,PUnit.unit⟩ (decide_eq_true_eq.mpr hj)).2 hw.symm
  · have sep := hsep (by omega : 0<length)
    have hs := ComplexBackwardSafety.trace_safe p u length A.extent hb sep.toReadSeparation
    have hok := ComplexBackwardSafety.flatten_ok p length
    obtain ⟨v,he,hgx,hgs,hgl,hframe⟩ := ComplexBackwardRun.run p u length sep
    have hbridge := A.exec_flatten hd hc _ u hs hok hu
    refine ⟨?_,A.flattenState v,?_,?_,?_,?_,?_⟩
    · simpa only [Kernel.TraceSafe,Erasure.complexBackward_body] using hs
    · have htyped : exec (A.flattenKernel (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
          length p.batch p.dim p.B).eraseDType.toAlgKernel) (A.flattenState u) =
          exec (A.flattenKernel (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
          length p.batch p.dim p.B).toAlgKernel) (A.flattenState u) := by
        simp only [exec,FlatAlloc.flattenKernel,Erasure.complexBackward_body]
      rw [htyped,hbridge,he,Option.map_some]
    · intro t ht j hj
      rw [read_flatten A hd hc v p.gx (offset p u t j) (hb.gradInput t ht j hj)]
      exact hgx t ht j hj
    · intro j hj
      rw [read_flatten A hd hc v p.gs (columns p u j) (hb.gradInitial j hj)]
      exact hgs j hj
    · intro j hj
      rw [read_flatten A hd hc v p.gl (columns p u j) (hb.gradTransition j hj)]
      exact hgl j hj
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
          apply congrArg A.trCell (hframe q z ?_)
          rcases ho with hne | ⟨hx,hs,hl⟩
          · exact False.elim (hne rfl)
          · refine ⟨?_,?_,?_⟩
            · by_cases hq : q=p.gx
              · subst q
                right
                intro t _ ht j hj
                constructor
                · intro heq; exact (hx t ht j hj).1 (by rw [heq]; exact hz.symm)
                · intro heq; exact (hx t ht j hj).2 (by rw [heq]; exact hz.symm)
              · exact Or.inl hq
            · by_cases hq : q=p.gs
              · subst q
                right
                intro j hj
                constructor
                · intro heq; exact (hs j hj).1 (by rw [heq]; exact hz.symm)
                · intro heq; exact (hs j hj).2 (by rw [heq]; exact hz.symm)
              · exact Or.inl hq
            · by_cases hq : q=p.gl
              · subst q
                right
                intro j hj
                constructor
                · intro heq; exact (hl j hj).1 (by rw [heq]; exact hz.symm)
                · intro heq; exact (hl j hj).2 (by rw [heq]; exact hz.symm)
              · exact Or.inl hq
      · simp only [FlatAlloc.flattenState,if_neg hr]

theorem candidate_correct (p : Layout) (length : Nat) :
    Spec.Real (Implements (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length) := by
  intro sep A hd hc u hu hb
  obtain ⟨hs,v,he,hx,hgs,hgl,hframe⟩ := original_correct p length sep A hd hc u hu hb
  refine ⟨?_,v,?_,hx,hgs,hgl,hframe⟩
  · unfold Kernel.TraceSafe at hs ⊢
    rw [Erasure.complexBackward_body,Candidate.complexBackward_body Bool.true,← ZeroSafety.index_trace]
    rw [← Candidate.complexBackward_body Bool.false]
    simpa only [Erasure.complexBackward_body] using hs
  · rw [← Candidate.complexBackward_flat_exec]
    exact he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardIO
