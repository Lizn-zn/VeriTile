import bench.tritonbench_g.dropout_triton.Proofs.Real
import bench.tritonbench_g.dropout_triton.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.RealIO
open VeriTile VeriTile.Triton TypedSource RealProof
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- All three buffers use the same lane window. Only active lanes access memory. -/
structure Bounds (x k y : RegionName) (n B : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, offset u B j<n → offset u B j<bounds x
  keep : ∀ j : Fin B, offset u B j<n → offset u B j<bounds k
  output : ∀ j : Fin B, offset u B j<n → offset u B j<bounds y

def Outside (A : FlatAlloc) (u : BlockState) (y : RegionName) (n B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<n → o≠A.addr y (offset u B j)

/-- Finite flat execution, independent initial-input dropout values, and a
full tagged-cell frame; region names may alias. -/
def Implements (kernel : ComputeKernel) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds x k y n B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<n →
        v.readMem A.flat (A.addr y (offset u B j))=expected keep u x k p (offset u B j)) ∧
      (∀ r o, Outside A u y n B r o → v.mem r o=(A.flattenState u).mem r o)

theorem original_correct (storage : Storage) (keep : KeepStorage)
    (x k y : RegionName) (n B : Nat) (p : ℝ) :
    Spec.Real (Implements (source storage keep Bool.false x k y n B p).eraseDType keep x k y n B p) := by
  intro A hd hc u hu hb
  have hs := trace_safe keep x k y n B p A.extent u hb.input hb.keep hb.output
  have hok := flatten_ok keep x k y n B p
  obtain ⟨v,he,hv,hf⟩ := run keep x k y n B p u
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · have htyped : exec (A.flattenKernel (source storage keep Bool.false x k y n B p).eraseDType.toAlgKernel)
        (A.flattenState u) =
        exec (A.flattenKernel (RealProof.original keep x k y n B p).toAlgKernel) (A.flattenState u) := by
      simp only [exec,FlatAlloc.flattenKernel,Erasure.body,RealProof.original]
    rw [htyped,hbridge,he,Option.map_some]
  · intro j hj
    have hy := hb.output j hj
    have hr : y ∈ A.regions := by
      by_contra hn
      rw [hc y hn] at hy
      omega
    rw [A.flattenState_readMem hd v hr hy]
    exact hv j hj
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
        by_cases hq : q=y
        · subst q
          right
          intro j hj hjz
          rcases ho with hn | hn
          · exact hn rfl
          · exact hn j hj (by rw [← hjz]; exact hz)
        · exact Or.inl hq
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem candidate_correct (storage : Storage) (keep : KeepStorage)
    (x k y : RegionName) (n B : Nat) (p : ℝ) :
    Spec.Real (Implements (source storage keep Bool.true x k y n B p).eraseDType keep x k y n B p) := by
  intro A hd hc u hu hb
  obtain ⟨v,he,hv,hf⟩ := original_correct storage keep x k y n B p A hd hc u hu hb
  refine ⟨v,?_,hv,hf⟩
  rw [← Candidate.flat_exec]
  exact he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DropoutTriton.RealIO
