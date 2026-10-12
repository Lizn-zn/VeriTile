import bench.tritonbench_g.fast_rms_layernorm.Proofs.BackwardMath
import bench.tritonbench_g.fast_rms_layernorm.Proofs.BackwardRun
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.BackwardIO
open VeriTile VeriTile.Triton TypedSource BackwardRun
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

structure Bounds (Y X W r : RegionName) (YR XR RR N B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  dy : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y
  input : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X
  weight : ∀ j : Fin B,j.val<N → j.val<bounds W
  inv : u.pids 0*RR<bounds r

def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (YR N B : Nat)
    (q : RegionName) (o : Nat) : Prop :=
  q≠A.flat ∨ ∀ j : Fin B,j.val<N → o≠A.addr Y (u.pids 0*YR+j.val)

def Implements (kernel : Kernel) (isGemma : Bool) (Y X W r : RegionName) (YR XR RR N B : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ q,q∉A.regions → A.extent q=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Bounds Y X W r YR XR RR N B u A.extent →
  0<N → N≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B,j.val<N → v.readMem A.flat (A.addr Y (u.pids 0*YR+j.val))=
        RealMath.gradient isGemma N (u.readMem r (u.pids 0*RR))
          (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
          (fun k : Fin B => u.readMem Y (u.pids 0*YR+k.val))
          (fun k : Fin B => u.readMem W k.val) j) ∧
      (∀ q o,Outside A u Y YR N B q o → v.mem q o=(A.flattenState u).mem q o)

theorem canonical_correct (isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (original isGemma Y X W r dW YR XR WR RR DWR N B eps)
      isGemma Y X W r YR XR RR N B) := by
  intro A hd hc u hu hb hN hB
  have hs := trace_safe isGemma Y X W r dW YR XR WR RR DWR N B eps A.extent u hb.dy hb.input hb.weight hb.inv
  have hok := flatten_ok isGemma Y X W r dW YR XR WR RR DWR N B eps
  obtain ⟨v,he⟩ := executes isGemma Y X W r dW YR XR WR RR DWR N B eps u
  have hi : Function.Injective (fun j : Fin B => RealExecution.dyOutOffset u YR j) := by
    intro a b h
    exact Fin.ext (Nat.add_left_cancel h)
  have hy (j : Fin B) (hj : j.val<N) : v.readMem Y (u.pids 0*YR+j.val)=
      RealMath.gradient isGemma N (u.readMem r (u.pids 0*RR))
        (fun k : Fin B => u.readMem X (u.pids 0*XR+k.val))
        (fun k : Fin B => u.readMem Y (u.pids 0*YR+k.val))
        (fun k : Fin B => u.readMem W k.val) j := by
    cases isGemma
    · have h := RealExecution.rms_layernorm_backward_dy_correct Y X W r dW YR XR WR RR DWR N B eps u v hi he j
      rw [if_pos hj] at h
      exact h.trans (BackwardMath.plain_gradient u Y X W r YR XR RR N B j hj)
    · have h := RealExecution.gemma_rms_layernorm_backward_dy_correct Y X W r dW YR XR WR RR DWR N B eps u v hi he j
      rw [if_pos hj] at h
      exact h.trans (BackwardMath.gemma_gradient u Y X W r YR XR RR N B j hj)
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  have hmem (q : RegionName) (o : Nat) (hq : o<A.extent q) : q∈A.regions := by
    by_contra hn
    rw [hc q hn] at hq
    omega
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro j hj
    rw [A.flattenState_readMem hd v (hmem Y _ (hb.dy j hj)) (hb.dy j hj)]
    exact hy j hj
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
        apply congrArg A.trCell (frame isGemma Y X W r dW YR XR WR RR DWR N B eps u v he q z ?_)
        intro j hj ⟨hq,hzval⟩
        subst q;subst z
        exact ho.elim (fun hn => hn rfl) (fun hn => hn j hj hz)
    · simp only [FlatAlloc.flattenState,if_neg hq]

theorem typed_correct (input weight : Storage) (isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (backward input weight Bool.false Y X W r dW YR XR WR RR DWR N B eps isGemma).eraseDType.toAlgKernel
      isGemma Y X W r YR XR RR N B) := by
  rw [Erasure.backward_body]
  exact canonical_correct isGemma Y X W r dW YR XR WR RR DWR N B eps

theorem candidate_correct (input weight : Storage) (isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    Spec.Real (Implements (backward input weight Bool.true Y X W r dW YR XR WR RR DWR N B eps isGemma).eraseDType.toAlgKernel
      isGemma Y X W r YR XR RR N B) := by
  rw [Erasure.backward_body]
  intro A hd hc u hu hb hN hB
  obtain ⟨v,he,hy,hf⟩ := canonical_correct isGemma Y X W r dW YR XR WR RR DWR N B eps A hd hc u hu hb hN hB
  refine ⟨v,?_,hy,hf⟩
  exact (Candidate.flat_exec A .backward isGemma Y X W r dW YR XR WR RR DWR N B eps (A.flattenState u)).symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.BackwardIO
