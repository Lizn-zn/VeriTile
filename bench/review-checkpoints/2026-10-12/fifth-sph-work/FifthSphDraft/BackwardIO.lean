import FifthSphDraft.BackwardReadback
import FifthSphDraft.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardIO
open VeriTile VeriTile.Triton TypedSource BackwardBase
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

/-- Every active coordinate belongs to a complete xyz row with all eleven
upstream channels available. This is the domain of the unmasked polynomial
formula; masked loads remain present in the implementation. -/
def Cover (pid B N M off stride : Nat) : Prop := ∀ j : Fin B,
  coord pid B j.val 0<N → coord pid B j.val 2<N ∧ output pid B stride off j.val 10<M

def Outside (A : FlatAlloc) (u : BlockState) (dX : RegionName) (B N : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N →
    o≠A.addr dX (coord u.pid B j.val c.val)

def Implements (kernel : Kernel) (X dX g : RegionName) (B N M off stride : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Bounds X dX g B N M off stride u A.extent →
  Cover u.pid B N M off stride →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N →
        v.readMem A.flat (A.addr dX (coord u.pid B j.val c.val))=
          RealMath.gradient c (u.readMem X (coord u.pid B j.val 0))
            (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))
            (u.readMem dX (coord u.pid B j.val c.val))
            (fun d => u.readMem g (output u.pid B stride off j.val d.val))) ∧
      (∀ r o,Outside A u dX B N r o → v.mem r o=(A.flattenState u).mem r o)

theorem canonical_correct (X dX g : RegionName) (B N M off stride : Nat) :
    Spec.Real (Implements (BackwardBase.source X dX g B N M off stride) X dX g B N M off stride) := by
  intro A hd ha u hu hb hcover
  have hs := BackwardBase.trace_safe X dX g B N M off stride u A.extent hb
  have hok := BackwardBase.flatten_ok X dX g B N M off stride
  obtain ⟨v,he⟩ := BackwardBase.executes X dX g B N M off stride u
  have hbridge := A.exec_flatten hd ha _ u hs hok hu
  have hmem (r : RegionName) (o : Nat) (ho : o<A.extent r) : r∈A.regions := by
    by_contra hn
    rw [ha r hn] at ho
    omega
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro c j hj
    rw [A.flattenState_readMem hd v (hmem dX _ (hb.gradient c j hj)) (hb.gradient c j hj)]
    have hc0 : coord u.pid B j.val 0<N := by unfold coord at *;omega
    obtain ⟨hc2,hg10⟩ := hcover j hc0
    exact BackwardReadback.all_components X dX g B N M off stride u v he j
      (by intro d;have hd := d.isLt;unfold coord at *;omega)
      (by intro d;have hd := d.isLt;unfold output at *;omega) c
  · intro r o ho
    by_cases hr : r=A.flat
    · subst r
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o with
      | none => rfl
      | some pair =>
        obtain ⟨r,z⟩ := pair
        obtain ⟨_,hz,_⟩ := A.decode_sound hdec
        show A.trCell (v.mem r z)=A.trCell (u.mem r z)
        apply congrArg A.trCell
        apply BackwardBase.frame X dX g B N M off stride u v he r z
        intro c j hj heq
        obtain ⟨hr,hz'⟩ := heq
        subst r
        have houtside := ho.elim (fun hn => False.elim (hn rfl)) (fun hn => hn c j hj)
        apply houtside
        simpa only [hz'] using hz
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem typed_correct (input gradient upstream : Storage) (X dX g : RegionName) (B N M off stride : Nat) :
    Spec.Real (Implements (backward input gradient upstream Bool.false X dX g B N M off stride).eraseDType.toAlgKernel X dX g B N M off stride) := by
  rw [Erasure.backward_body]
  exact canonical_correct X dX g B N M off stride

theorem candidate_correct (input gradient upstream : Storage) (X dX g : RegionName) (B N M off stride : Nat) :
    Spec.Real (Implements (backward input gradient upstream Bool.true X dX g B N M off stride).eraseDType.toAlgKernel X dX g B N M off stride) := by
  rw [Erasure.backward_body]
  intro A hd ha u hu hb hcover
  obtain ⟨v,he,hv,hf⟩ := canonical_correct X dX g B N M off stride A hd ha u hu hb hcover
  exact ⟨v,(Candidate.flat_exec A .backward X dX g B N M off stride (A.flattenState u)).symm.trans he,hv,hf⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardIO
