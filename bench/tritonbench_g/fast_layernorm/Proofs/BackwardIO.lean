import bench.tritonbench_g.fast_layernorm.Proofs.MathBridge
import bench.tritonbench_g.fast_layernorm.Proofs.BackwardRun
import bench.tritonbench_g.fast_layernorm.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.BackwardIO
open VeriTile VeriTile.Triton TypedSource RealExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

structure Bounds (dY X W b r mu : RegionName) (YR XR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  dy : ∀ j : Fin B, j.val<N → u.pids 0*YR+j.val<bounds dY
  input : ∀ j : Fin B, j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B, j.val<N → j.val<bounds W
  bias : ∀ j : Fin B, j.val<N → j.val<bounds b
  inv : u.pids 0<bounds r
  mean : u.pids 0<bounds mu

def Outside (A : FlatAlloc) (u : BlockState) (dY : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr dY (u.pids 0*YR+j.val)

def Implements (kernel : Kernel) (dY X W b r mu : RegionName) (YR XR N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds dY X W b r mu YR XR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr dY (u.pids 0*YR+j.val))=
        RealMath.gradient N (u.readMem r (u.pids 0)) (u.readMem mu (u.pids 0))
          (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem dY (u.pids 0*YR+j.val))
          (fun j : Fin B => u.readMem W j.val) j) ∧
      (∀ q o,Outside A u dY YR N B q o → v.mem q o=(A.flattenState u).mem q o)

theorem canonical_correct (dY X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (layernorm_backward dY X W b r mu YR XR N eps B).toAlgKernel
      dY X W b r mu YR XR N B) := by
  intro A hd hc u hu hb hN hB
  have hs := BackwardRun.layernorm_backward_traceSafe dY X W b r mu YR XR N B eps A.extent u
    hb.input hb.weight hb.bias hb.dy hb.inv hb.mean
  have hok := BackwardRun.layernorm_backward_flattenOk dY X W b r mu YR XR N eps B
  obtain ⟨v,he⟩ := BackwardRun.layernorm_backward_exec_isSome dY X W b r mu YR XR N B eps u
  have hi : Function.Injective (fun j : Fin B => dXOutOffset u YR j) := by
    intro a b h
    exact Fin.ext (Nat.add_left_cancel h)
  have hy := layernorm_backward_dx_correct dY X W b r mu YR XR N B eps u v hi he
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  have hmem (q : RegionName) (o : Nat) (hq : o<A.extent q) : q∈A.regions := by
    by_contra hn
    rw [hc q hn] at hq
    omega
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro j hj
    rw [A.flattenState_readMem hd v (hmem dY _ (hb.dy j hj)) (hb.dy j hj)]
    have h := hy j
    rw [if_pos hj] at h
    exact h.trans (MathBridge.gradient_eq u dY X W r mu YR XR N B j hj)
  · intro q o ho
    by_cases hq : q=A.flat
    · subst q
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o with
      | none => rfl
      | some pair =>
        obtain ⟨q,z⟩ := pair
        obtain ⟨_,hz,_⟩ := A.decode_sound hdec
        show A.trCell (v.mem q z)=A.trCell (u.mem q z)
        apply congrArg A.trCell
        apply BackwardRun.layernorm_backward_frame dY X W b r mu YR XR N B eps u v he q z
        intro j hj ⟨hq,hzval⟩
        subst q; subst z
        exact ho.elim (fun hn => hn rfl) (fun hn => hn j hj hz)
    · simp only [FlatAlloc.flattenState,if_neg hq]

theorem typed_correct (input weight bias : Storage) (dY X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (backward input weight bias Bool.false dY X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      dY X W b r mu YR XR N B) := by
  rw [Erasure.backward_body]
  exact canonical_correct dY X W b r mu YR XR N B eps

theorem candidate_correct (input weight bias : Storage) (dY X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (backward input weight bias Bool.true dY X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      dY X W b r mu YR XR N B) := by
  rw [Erasure.backward_body]
  intro A hd hc u hu hb hN hB
  obtain ⟨v,he,hy,hf⟩ := canonical_correct dY X W b r mu YR XR N B eps A hd hc u hu hb hN hB
  refine ⟨v,?_,hy,hf⟩
  have hp := Candidate.flat_exec A .backward dY X W b r mu YR XR N B eps (A.flattenState u)
  exact hp.symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.BackwardIO
