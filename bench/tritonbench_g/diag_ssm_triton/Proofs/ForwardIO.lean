import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardCorrect
import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.ReferenceCongruence
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardIO
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

abbrev Layout := ForwardLayout

def column (p : Layout) (u : BlockState) (j : Fin p.B) : Nat := u.pids 0*p.B+j.val
def active (p : Layout) (u : BlockState) (j : Fin p.B) : Prop := column p u j<p.batch*p.dim
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : Nat := t*(p.batch*p.dim)+column p u j

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : ∀ j, active p u j → column p u j < bounds p.s
  transition : ∀ j, active p u j → column p u j%p.dim < bounds p.l
  input : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.x
  output : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.y

/-- Independent scalar recurrence over the initial input memory. -/
def value (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : ℝ :=
  Reference.forwardState (u.readMem p.s (column p u j)) (u.readMem p.l (column p u j%p.dim))
    (fun q => u.readMem p.x (offset p u q j)) (t+1)

def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ t, t<length → ∀ j, active p u j → o≠A.addr p.y (offset p u t j)

/-- Uniform real contract, including zero length. The initial-state and
transition bounds remain required because their loads precede the loop. -/
def Implements (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → p.x≠p.y) →
  ∀ A : FlatAlloc, A.Disjoint → A.regions=[p.s,p.l,p.x,p.y] →
    (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, active p u j → v.readMem A.flat (A.addr p.y (offset p u t j))=value p u t j) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)

theorem original_correct (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (Implements (TypedSource.forward storage Bool.false p.s p.x p.l p.y
      length p.batch p.dim p.B).eraseDType p length) := by
  intro hsep A hd hr hc u hu hb
  by_cases hz : length=0
  · subst length
    have hs : ZeroSafety.forwardBounds A.extent u p.s p.l p.batch p.dim p.B :=
      ⟨hb.initial,hb.transition⟩
    obtain ⟨_,v,he,_,hf⟩ := ZeroIO.forward_correct storage Bool.false p.s p.x p.l p.y
      p.batch p.dim p.B A hd hc u hs
    exact ⟨v,he,fun t ht => by omega,fun r o _ => hf r o (by simp)⟩
  · have hp : 0<length := by omega
    have h := ForwardCorrect.original_correct storage p.s p.x p.l p.y length p.batch p.dim p.B hp (hsep hp)
    dsimp only [Spec.Real,StreamEmitMasked2DKernelIO₃.Implements,ForwardCorrect.originalIO] at h
    obtain ⟨v,he,hv,hf⟩ := h A hd hr hc (u.pids 0) (u.pids 1)
      (fun _ j => u.readMem p.s (column p u j))
      (fun _ j => u.readMem p.l (column p u j%p.dim))
      (fun t j => u.readMem p.x (offset p u t.val j)) u rfl rfl hu
      (fun _ j hj => hb.initial j hj)
      (fun _ j hj => hb.transition j hj)
      (fun t j hj => hb.input t.val t.isLt j hj)
      (fun t j hj => hb.output t.val t.isLt j hj)
      (by intros; rfl) (by intros; rfl) (by intros; rfl)
    refine ⟨v,he,?_,?_⟩
    · intro t ht j hj
      refine (hv ⟨t,ht⟩ j hj).trans ?_
      unfold ForwardCorrect.scan value
      apply ReferenceCongruence.forward_prefix
      intro q hq
      have hql : q<length := by omega
      simp [hql]
    · intro r o ho
      apply hf r o
      rcases ho with hn | ho
      · exact Or.inl hn
      · exact Or.inr (fun t j hj => ho t.val t.isLt j hj)

theorem candidate_correct (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (Implements (TypedSource.forward storage Bool.true p.s p.x p.l p.y
      length p.batch p.dim p.B).eraseDType p length) := by
  intro hsep A hd hr hc u hu hb
  obtain ⟨v,he,hv,hf⟩ := original_correct storage p length hsep A hd hr hc u hu hb
  refine ⟨v,?_,hv,hf⟩
  rw [← Candidate.forward_flat_exec]
  exact he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardIO
