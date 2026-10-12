import bench.tritonbench_g.fast_rms_layernorm.Proofs.ForwardRun
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.ForwardIO
open VeriTile VeriTile.Triton TypedSource ForwardRun
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

structure Bounds (isGemma : Bool) (Y X W r : RegionName) (YR XR WR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → weightOffset isGemma WR j.val<bounds W
  output : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  inv : u.pids 0*RR<bounds r

def Outside (A : FlatAlloc) (u : BlockState) (Y r : RegionName) (YR RR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ((∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)) ∧
    o≠A.addr r (u.pids 0*RR))

def Implements (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Bounds isGemma Y X W r YR XR WR RR N B u A.extent →
  0<N → N≤B → Y≠r →
  0<RealMath.meanSquare N (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val))+eps →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        RealMath.output isGemma N eps (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem W (weightOffset isGemma WR k.val)) j) ∧
      v.readMem A.flat (A.addr r (u.pids 0*RR))=
        RealMath.invStd N eps (fun j : Fin B => u.readMem X (u.pids 0*XR+j.val)) ∧
      (∀ q o,Outside A u Y r YR RR N B q o → v.mem q o=(A.flattenState u).mem q o)

theorem canonical_correct (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (original isGemma Y X W r YR XR WR RR N B eps)
      isGemma Y X W r YR XR WR RR N B eps) := by
  intro A hd hc u hu hb hN hB hne hpos
  have hs := trace_safe isGemma Y X W r YR XR WR RR N B eps A.extent u hb.input hb.weight hb.output hb.inv
  have hok := flatten_ok isGemma Y X W r YR XR WR RR N B eps
  obtain ⟨v,he,hy,hr⟩ := run isGemma Y X W r YR XR WR RR N B eps u hne
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  have hmem (q : RegionName) (o : Nat) (hq : o<A.extent q) : q∈A.regions := by
    by_contra hn
    rw [hc q hn] at hq
    omega
  refine ⟨A.flattenState v,?_,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro j hj
    rw [A.flattenState_readMem hd v (hmem Y _ (hb.output j hj)) (hb.output j hj)]
    exact hy j hj
  · rw [A.flattenState_readMem hd v (hmem r _ hb.inv) hb.inv]
    exact hr
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
        apply congrArg A.trCell (frame isGemma Y X W r YR XR WR RR N B eps u v he q z ?_ ?_)
        · intro j hj ⟨hq,hzval⟩
          subst q;subst z
          exact ho.elim (fun hn => hn rfl) (fun hn => hn.1 j hj hz)
        · intro ⟨hq,hzval⟩
          subst q;subst z
          exact ho.elim (fun hn => hn rfl) (fun hn => hn.2 hz)
    · simp only [FlatAlloc.flattenState,if_neg hq]

def source (input weight : Storage) (isGemma swapped : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : ComputeKernel :=
  if isGemma then gemma input weight swapped Y X W r YR XR WR RR N B eps
  else forward input weight swapped Y X W r YR XR WR RR N B eps

theorem erased (input weight : Storage) (isGemma swapped : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    (source input weight isGemma swapped Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel=
    (source .fp32 .fp32 isGemma swapped Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel := by
  cases isGemma
  · exact Erasure.forward_body input weight swapped Y X W r YR XR WR RR N B eps
  · exact Erasure.gemma_body input weight swapped Y X W r YR XR WR RR N B eps

theorem typed_correct (input weight : Storage) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (source input weight isGemma Bool.false Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      isGemma Y X W r YR XR WR RR N B eps) := by
  rw [erased]
  exact canonical_correct isGemma Y X W r YR XR WR RR N B eps

theorem candidate_correct (input weight : Storage) (isGemma : Bool) (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (source input weight isGemma Bool.true Y X W r YR XR WR RR N B eps).eraseDType.toAlgKernel
      isGemma Y X W r YR XR WR RR N B eps) := by
  rw [erased]
  intro A hd hc u hu hb hN hB hne hpos
  obtain ⟨v,he,hy,hr,hf⟩ := canonical_correct isGemma Y X W r YR XR WR RR N B eps A hd hc u hu hb hN hB hne hpos
  refine ⟨v,?_,hy,hr,hf⟩
  cases isGemma
  · exact (Candidate.flat_exec A .forward Bool.false Y X W r "unused" YR XR WR RR 0 N B eps (A.flattenState u)).symm.trans he
  · exact (Candidate.flat_exec A .gemma Bool.true Y X W r "unused" YR XR WR RR 0 N B eps (A.flattenState u)).symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.ForwardIO
