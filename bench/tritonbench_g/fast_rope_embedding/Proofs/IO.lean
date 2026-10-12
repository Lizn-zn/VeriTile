import bench.tritonbench_g.fast_rope_embedding.Proofs.Real
import bench.tritonbench_g.fast_rope_embedding.Proofs.Safety
import bench.tritonbench_g.fast_rope_embedding.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.IOProof
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000

def Outside (A : FlatAlloc) (u : BlockState) (Q : RegionName) (QR H heads B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ k,Reference.start (u.pids 1)≤k → k<Reference.stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 →
      o≠A.addr Q (Reference.first u.pid QR H k j.val) ∧ o≠A.addr Q (Reference.second u.pid QR H k j.val)

def Implements (kernel : Kernel) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Safety.Bounds u Q C S QR CR SR seq H heads B A.extent →
  0<seq → H/2≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ k,Reference.start (u.pids 1)≤k → k<Reference.stop (u.pids 1) heads →
        ∀ j : Fin B,j.val<H/2 →
        v.readMem A.flat (A.addr Q (Reference.first u.pid QR H k j.val))=
          Reference.firstValue backwards
            (u.readMem Q (Reference.first u.pid QR H k j.val))
            (u.readMem Q (Reference.second u.pid QR H k j.val))
            (u.readMem C (Reference.coefficientAddress u.pid CR seq j.val))
            (u.readMem S (Reference.coefficientAddress u.pid SR seq j.val)) ∧
        v.readMem A.flat (A.addr Q (Reference.second u.pid QR H k j.val))=
          Reference.secondValue backwards
            (u.readMem Q (Reference.first u.pid QR H k j.val))
            (u.readMem Q (Reference.second u.pid QR H k j.val))
            (u.readMem C (Reference.coefficientAddress u.pid CR seq j.val))
            (u.readMem S (Reference.coefficientAddress u.pid SR seq j.val))) ∧
      (∀ r o,Outside A u Q QR H heads B r o → v.mem r o=(A.flattenState u).mem r o)

theorem canonical_correct (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Spec.Real (Implements (Code.original Q C S QR CR SR seq H heads B backwards) Q C S QR CR SR seq H heads B backwards) := by
  intro A hd ha u hu hb _ _
  have hs := Safety.trace_safe Q C S QR CR SR seq H heads B backwards A.extent u hb
  have hok := Safety.flatten_ok Q C S QR CR SR seq H heads B backwards
  obtain ⟨v,he,hv,hf⟩ := RealProof.run Q C S QR CR SR seq H heads B backwards u
  have hbridge := A.exec_flatten hd ha _ u hs hok hu
  have hmem (r : RegionName) (o : Nat) (ho : o<A.extent r) : r∈A.regions := by
    by_contra hn
    rw [ha r hn] at ho
    omega
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro k hk ht j hj
    rw [A.flattenState_readMem hd v (hmem Q _ (hb.first k hk ht j hj)) (hb.first k hk ht j hj),
      A.flattenState_readMem hd v (hmem Q _ (hb.second k hk ht j hj)) (hb.second k hk ht j hj)]
    exact hv k hk ht j hj
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
        apply hf r z
        by_cases hr : r=Q
        · subst r
          right
          rintro ⟨k,hk,ht,j,hj,heq⟩
          have houtside := ho.elim (fun hn => False.elim (hn rfl)) (fun hn => hn k hk ht j hj)
          rcases heq with heq|heq
          · subst z
            exact houtside.1 hz
          · subst z
            exact houtside.2 hz
        · exact Or.inl hr
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem typed_correct (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Spec.Real (Implements (rope input cosine sine Bool.false Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel
      Q C S QR CR SR seq H heads B backwards) := by
  rw [Erasure.body]
  exact canonical_correct Q C S QR CR SR seq H heads B backwards

theorem candidate_correct (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Spec.Real (Implements (rope input cosine sine Bool.true Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel
      Q C S QR CR SR seq H heads B backwards) := by
  rw [Erasure.body]
  intro A hd ha u hu hb hseq hB
  obtain ⟨v,he,hv,hf⟩ := canonical_correct Q C S QR CR SR seq H heads B backwards A hd ha u hu hb hseq hB
  refine ⟨v,?_,hv,hf⟩
  exact (Candidate.flat_exec A Q C S QR CR SR seq H heads B backwards (A.flattenState u)).symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.IOProof
