import bench.tritonbench_g.f8_conversion_utils.Proofs.Real
import bench.tritonbench_g.f8_conversion_utils.Proofs.Candidate
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.RealIO
open VeriTile VeriTile.Triton TypedSource RealProof
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

structure Bounds (Y X : RegionName) (N B : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, offset u B j<N → offset u B j<bounds X
  output : ∀ j : Fin B, offset u B j<N → offset u B j<bounds Y

def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<N → o≠A.addr Y (offset u B j)

/-- Erased real copying: actual finite bounds, successful full execution,
initial-input values and the full tagged-cell frame. -/
def Implements (kernel : ComputeKernel) (Y X : RegionName) (N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds Y X N B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<N → v.readMem A.flat (A.addr Y (offset u B j))=u.readMem X (offset u B j)) ∧
      (∀ r o, Outside A u Y N B r o → v.mem r o=(A.flattenState u).mem r o)

theorem canonical_correct (doubleStore : Bool) (Y X : RegionName) (N B : Nat) :
    Spec.Real (Implements (original doubleStore Y X N B) Y X N B) := by
  intro A hd hc u hu hb
  have hs := trace_safe doubleStore Y X N B A.extent u hb.input hb.output
  have hok := flatten_ok doubleStore Y X N B
  obtain ⟨v,he,hv,hf⟩ := run doubleStore Y X N B u
  have hbridge := A.exec_flatten hd hc _ u hs hok hu
  refine ⟨A.flattenState v,?_,?_,?_⟩
  · rw [hbridge,he,Option.map_some]
  · intro j hj
    have hy := hb.output j hj
    have hr : Y ∈ A.regions := by
      by_contra hn
      rw [hc Y hn] at hy
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
        by_cases hq : q=Y
        · subst q
          right
          intro j hj hjz
          rcases ho with hn|hn
          · exact hn rfl
          · exact hn j hj (by rw [←hjz]; exact hz)
        · exact Or.inl hq
    · simp only [FlatAlloc.flattenState,if_neg hr]

theorem to_half_correct (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (Implements (toHalf format Bool.false Y X N B).eraseDType Y X N B) := by
  simpa only [Spec.Real,Implements,exec,FlatAlloc.flattenKernel,Erasure.to_half_body,original,↓reduceIte]
    using canonical_correct Bool.true Y X N B

theorem to_fp8_correct (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (Implements (toFP8 input format Bool.false Y X N B).eraseDType Y X N B) := by
  simpa only [Spec.Real,Implements,exec,FlatAlloc.flattenKernel,Erasure.to_fp8_body,original,↓reduceIte]
    using canonical_correct Bool.false Y X N B

theorem to_half_candidate_correct (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (Implements (toHalf format Bool.true Y X N B).eraseDType Y X N B) := by
  intro A hd hc u hu hb
  obtain ⟨v,he,hv,hf⟩ := to_half_correct format Y X N B A hd hc u hu hb
  refine ⟨v,?_,hv,hf⟩
  rw [← Candidate.to_half_flat_exec]
  exact he

theorem to_fp8_candidate_correct (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (Implements (toFP8 input format Bool.true Y X N B).eraseDType Y X N B) := by
  intro A hd hc u hu hb
  obtain ⟨v,he,hv,hf⟩ := to_fp8_correct input format Y X N B A hd hc u hu hb
  refine ⟨v,?_,hv,hf⟩
  rw [← Candidate.to_fp8_flat_exec]
  exact he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.RealIO
