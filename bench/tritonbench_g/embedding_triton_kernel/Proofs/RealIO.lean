import bench.tritonbench_g.embedding_triton_kernel.Proofs.Real
import bench.tritonbench_g.embedding_triton_kernel.Proofs.Safety
import bench.tritonbench_g.embedding_triton_kernel.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.RealIO
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def Outside (A : FlatAlloc) (u : BlockState) (out : RegionName)
    (so n h D N NN : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ z, Reference.Written u.pid N NN D n h so z → o≠A.addr out z

/-- Actual finite allocation and successful flat execution are part of the
contract, together with the independent lookup and full tagged-cell frame. -/
def Implements (kernel : ComputeKernel) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : Prop :=
  weight≠out → ids≠out → min D h≤so →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Safety.Bounds u weight ids out first last sw so n h D N NN A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
        Reference.row u.pid N NN t jr.val<n → jd.val<h →
        v.readMem A.flat (A.addr out (Reference.row u.pid N NN t jr.val*so+jd.val))=
          Reference.value u weight ids first last sw (Reference.row u.pid N NN t jr.val) jd.val) ∧
      (∀ r o, Outside A u out so n h D N NN r o → v.mem r o=(A.flattenState u).mem r o)

theorem original_correct (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) :
    Spec.Real (Implements (source storage Bool.false weight ids out first last sw so n h D N NN).eraseDType
      weight ids out first last sw so n h D N NN) := by
  intro hw hid hstride A hd hc u hu hb
  have hsafe := Safety.trace_safe weight ids out first last sw so n h D N NN A.extent u hw hid hstride hb
  have hok := Chunk.flatten_ok weight ids out first last sw so n h D N NN
  obtain ⟨v,he,hv,hf⟩ := RealProof.run weight ids out first last sw so n h D N NN u hw hid hstride
  have hbridge := A.exec_flatten hd hc _ u hsafe hok hu
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · have htyped : exec (A.flattenKernel (source storage Bool.false weight ids out first last sw so n h D N NN).eraseDType.toAlgKernel)
        (A.flattenState u)=
        exec (A.flattenKernel (Erasure.source Bool.false weight ids out first last sw so n h D N NN).toAlgKernel) (A.flattenState u) := by
      simp only [exec,FlatAlloc.flattenKernel,Erasure.body]
    rw [htyped,hbridge,he,Option.map_some]
  · intro t ht jr jd hn hdcol
    have hy := hb.outputAccess t ht jr jd hn hdcol
    have hr : out ∈ A.regions := by
      by_contra hn
      rw [hc out hn] at hy
      omega
    rw [A.flattenState_readMem hd v hr hy]
    exact hv t ht jr jd hn hdcol
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
        apply congrArg A.trCell (hf q z ?_)
        by_cases hq : q=out
        · subst q
          right
          intro hwritten
          rcases ho with hn | hn
          · exact hn rfl
          · exact hn z hwritten hz
        · exact Or.inl hq
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem candidate_correct (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) :
    Spec.Real (Implements (source storage Bool.true weight ids out first last sw so n h D N NN).eraseDType
      weight ids out first last sw so n h D N NN) := by
  intro hw hid hs A hd hc u hu hb
  obtain ⟨v,he,hv,hf⟩ := original_correct storage weight ids out first last sw so n h D N NN hw hid hs A hd hc u hu hb
  refine ⟨v,?_,hv,hf⟩
  rw [← Candidate.flat_exec]
  exact he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.RealIO
