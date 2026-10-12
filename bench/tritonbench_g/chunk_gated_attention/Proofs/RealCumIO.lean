import VeriTile.Triton.Memory.KernelSpec.TileND
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealCumFlat

/-! The cumulative kernel's complete BT-by-BS IO window, at feature/chunk/head
program coordinates. All source loads precede the store, permitting aliasing.
The formula is the independent prefix of initial inputs inside this chunk. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumIO
open VeriTile.Triton RealCumProgram RealCumExecution
open scoped BigOperators VeriTile.Triton.Masked3DTileKernelIO₁
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def io (swapped : Bool) (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) : Masked3DTileKernelIO₁ where
  kernel := (RealCumSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,RealCumSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  inp := s
  out := o
  shape := [p.BT,p.BS]
  read := fun feature c head i => offset p ⟨feature,head⟩ c i
  write := fun feature c head i => offset p ⟨feature,head⟩ c i
  mask := fun feature c head i => tileActive p ⟨feature,head⟩ c i

def expected (p : Layout) (_feature _chunk : Nat) (xs : TileIndex [p.BT,p.BS] → ℝ)
    (i : TileIndex [p.BT,p.BS]) : ℝ :=
  ∑ k : Fin p.BT, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0

theorem expected_eq (p : Layout) (u : BlockState) (s : RegionName)
    (xs : TileIndex [p.BT,p.BS] → ℝ)
    (hx : ∀ i, tileActive p (launch u) (chunk u) i → u.readMem s (offset p (launch u) (chunk u) i) = xs i)
    (i : TileIndex [p.BT,p.BS]) (hi : tileActive p (launch u) (chunk u) i) :
    RealCumExecution.expected p (launch u) (chunk u) (u.readMem s) i =
      expected p (u.pids 0) (u.pids 1) xs i := by
  unfold RealCumExecution.expected expected
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val ≤ i.1.val
  · simp only [hk,ite_true]
    apply hx (k,i.2.1,PUnit.unit)
    exact ⟨by change chunk u*p.BT+k.val < p.T; have := hi.1; omega,hi.2⟩
  · simp only [hk,ite_false]

theorem implements (swapped : Bool) (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (hinj : ∀ feature c head, NoCollision p ⟨feature,head⟩ c) :
    io swapped p s o dtype out_dtype ⊨ expected p := by
  apply Masked3DTileKernelIO₁.Implements.intro
  · exact RealCumSafety.source_flatten_ok p s o swapped dtype out_dtype
  · intro bounds u hb hw
    exact RealCumSafety.source_safe p s o swapped dtype out_dtype u bounds ⟨hb,hw⟩
  · intro u xs hx
    obtain ⟨v,he,hv,hf⟩ := RealCumExecution.source_executes p s o swapped dtype out_dtype u (hinj _ _ _)
    refine ⟨v,he,?_,?_⟩
    · intro i hi
      exact (hv i hi).trans (expected_eq p u s xs hx i hi)
    · intro r off hn
      apply (hf r off ?_).symm
      rintro ⟨hr,i,hi,hoff⟩
      rcases hn with hr' | ho
      · exact hr' hr
      · exact ho i hi hoff

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumIO
