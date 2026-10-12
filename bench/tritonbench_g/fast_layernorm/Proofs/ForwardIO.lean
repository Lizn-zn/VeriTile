import bench.tritonbench_g.fast_layernorm.Proofs.MathBridge
import bench.tritonbench_g.fast_layernorm.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.ForwardIO
open VeriTile VeriTile.Triton TypedSource RealExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

structure Bounds (Y X W b r mu : RegionName) (YR XR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B, j.val<N → j.val<bounds W
  bias : ∀ j : Fin B, j.val<N → j.val<bounds b
  output : ∀ j : Fin B, j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0<bounds r
  mean : u.pids 0<bounds mu

def Outside (A : FlatAlloc) (u : BlockState) (Y r mu : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0) ∧ o≠A.addr mu (u.pids 0))

def Implements (kernel : Kernel) (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds Y X W b r mu YR XR N B u A.extent →
  0<N → N≤B → Y≠r → Y≠mu → r≠mu →
  0<RealMath.variance N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        RealMath.output N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
          (fun j : Fin B => u.readMem W j.val) (fun j : Fin B => u.readMem b j.val) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0))=
        RealMath.invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      v.readMem A.flat (A.addr mu (u.pids 0))=
        RealMath.mean N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,Outside A u Y r mu YR N B q o → v.mem q o=(A.flattenState u).mem q o)

theorem canonical_correct (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (layernorm_forward Y X W b r mu YR XR N eps B).toAlgKernel
      Y X W b r mu YR XR N B eps) := by
  intro A hd hc u hu hb hN hB hYr hYmu hRmu hpos
  have hs := layernorm_forward_traceSafe Y X W b r mu YR XR N B eps A.extent u
    hb.input hb.weight hb.bias hb.output hb.inv hb.mean
  have hok := layernorm_forward_flattenOk Y X W b r mu YR XR N eps B
  obtain ⟨v,he,hy,hr,hm,_⟩ := layernorm_forward_region_run Y X W b r mu YR XR N B eps
    hYr hYmu hRmu u (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))
    (fun j : Fin B => u.readMem W j.val) (fun j : Fin B => u.readMem b j.val)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  have hmem (q : RegionName) (o : Nat) (hq : o<A.extent q) : q∈A.regions := by
    by_contra hn
    rw [hc q hn] at hq
    omega
  refine ⟨A.flattenState v,?_,?_,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro j hj
    rw [A.flattenState_readMem hd v (hmem Y _ (hb.output j hj)) (hb.output j hj)]
    exact (hy j hj).trans (MathBridge.output_eq _ _ _ _ _ _ _)
  · rw [A.flattenState_readMem hd v (hmem r _ hb.inv) hb.inv]
    exact hr.trans (MathBridge.inv_eq _ _ _ _)
  · rw [A.flattenState_readMem hd v (hmem mu _ hb.mean) hb.mean]
    exact hm.trans (MathBridge.mean_eq _ _ _)
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
        apply layernorm_forward_frame Y X W b r mu YR XR N B eps u v he q z
        · intro j hj ⟨hq,hzval⟩
          subst q; subst z
          exact ho.elim (fun hn => hn rfl) (fun hn => hn.1 j hj hz)
        · intro ⟨hq,hzval⟩
          subst q; subst z
          exact ho.elim (fun hn => hn rfl) (fun hn => hn.2.1 hz)
        · intro ⟨hq,hzval⟩
          subst q; subst z
          exact ho.elim (fun hn => hn rfl) (fun hn => hn.2.2 hz)
    · simp only [FlatAlloc.flattenState,if_neg hq]

theorem typed_correct (input weight bias : Storage) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (forward input weight bias Bool.false Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      Y X W b r mu YR XR N B eps) := by
  rw [Erasure.forward_body]
  exact canonical_correct Y X W b r mu YR XR N B eps

theorem candidate_correct (input weight bias : Storage) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (forward input weight bias Bool.true Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel
      Y X W b r mu YR XR N B eps) := by
  rw [Erasure.forward_body]
  intro A hd hc u hu hb hN hB hYr hYmu hRmu hpos
  obtain ⟨v,he,hy,hr,hm,hf⟩ := canonical_correct Y X W b r mu YR XR N B eps A hd hc u hu hb hN hB hYr hYmu hRmu hpos
  refine ⟨v,?_,hy,hr,hm,hf⟩
  have hp := Candidate.flat_exec A .forward Y X W b r mu YR XR N B eps (A.flattenState u)
  exact hp.symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.ForwardIO
