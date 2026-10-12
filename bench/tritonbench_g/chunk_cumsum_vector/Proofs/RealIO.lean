import VeriTile.Triton.Memory.KernelSpec.TileND
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealSafety
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealCandidate

/-! Flat IO for the entire logical T-by-BS output window. The algorithm still
executes BT-by-BS chunks; this window describes its complete observable result. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealIO
open VeriTile.Triton RealProgram
open scoped BigOperators VeriTile.Triton.Masked3DTileKernelIO₁
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def io (swapped : Bool) (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) :
    Masked3DTileKernelIO₁ where
  kernel := (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType, RealSource.projects]
    simp only [ComputeKernel.fromKernelBody, ComputeKernel.toAlgorithm?_mk_map_alg,
      ComputeKernel.toAlgKernel]
  inp := s
  out := o
  shape := [p.T, p.BS]
  read := fun feature head _ i => RealStep.address p ⟨feature,head⟩ i.1.val i.2.1.val
  write := fun feature head _ i => RealStep.address p ⟨feature,head⟩ i.1.val i.2.1.val
  mask := fun feature _ _ i => feature * p.BS + i.2.1.val < p.S

def expected (p : Layout) (_feature _head : Nat) (xs : TileIndex [p.T,p.BS] → ℝ)
    (i : TileIndex [p.T,p.BS]) : ℝ :=
  ∑ k : Fin p.T, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0

theorem expected_eq (p : Layout) (u : BlockState) (s : RegionName)
    (xs : TileIndex [p.T,p.BS] → ℝ)
    (hx : ∀ i : TileIndex [p.T,p.BS], u.pids 0 * p.BS + i.2.1.val < p.S →
      u.readMem s (RealStep.address p (RealInit.launch u) i.1.val i.2.1.val) = xs i)
    (i : TileIndex [p.T,p.BS]) (hi : u.pids 0 * p.BS + i.2.1.val < p.S) :
    RealLoop.expected p (RealInit.launch u) (u.readMem s) i.1.val i.2.1 =
      expected p (u.pids 0) (u.pids 1) xs i := by
  unfold RealLoop.expected RealMath.prefixValue expected
  rw [Finset.sum_filter, ← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val ≤ i.1.val
  · simp only [hk, ite_true, RealStep.column, RealInit.launch, hi]
    exact hx (k,i.2.1,PUnit.unit) hi
  · simp only [hk, ite_false]

theorem original_implements (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hinj : ∀ feature head, RealLoop.NoCollision p ⟨feature,head⟩) :
    io Bool.false p s o dtype out_dtype ⊨ expected p := by
  apply Masked3DTileKernelIO₁.Implements.intro
  · exact RealSafety.source_flatten_ok p s o dtype out_dtype
  · intro bounds u hb hw
    apply RealSafety.source_safe p s o dtype out_dtype u hSO hBT (hinj _ _) bounds ?_
    constructor
    · intro t j ha
      exact hb (⟨t,ha.1⟩,j,PUnit.unit) ha.2
    · intro t j ha
      exact hw (⟨t,ha.1⟩,j,PUnit.unit) ha.2
  · intro u xs hx
    obtain ⟨v,he,hv,hf⟩ := RealInit.original_executes p u s o dtype out_dtype
      (u.readMem s) (fun _ => rfl) hSO hBT (hinj _ _)
    refine ⟨v,he,?_,?_⟩
    · intro i hi
      exact (hv i.1.val i.2.1 ⟨i.1.isLt,hi⟩).trans (expected_eq p u s xs hx i hi)
    · intro r off hn
      apply (hf r off ?_).symm
      rintro ⟨hr,t,j,ha,hoff⟩
      rcases hn with hr' | ho
      · exact hr' hr
      · exact ho (⟨t,ha.1⟩,j,PUnit.unit) ha.2 hoff

theorem candidate_implements (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hinj : ∀ feature head, RealLoop.NoCollision p ⟨feature,head⟩) :
    io Bool.true p s o dtype out_dtype ⊨ expected p := by
  intro A hd hr hc p₀ p₁ p₂ hb hw xs u hp₀ hp₁ hp₂ hu hx
  obtain ⟨v,he,hv,hf⟩ := original_implements p s o dtype out_dtype hSO hBT hinj
    A hd hr hc p₀ p₁ p₂ hb hw xs u hp₀ hp₁ hp₂ hu hx
  refine ⟨v,?_,hv,hf⟩
  exact (RealCandidate.flat_exec_equal A p s o dtype out_dtype (A.flattenState u)).symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealIO
