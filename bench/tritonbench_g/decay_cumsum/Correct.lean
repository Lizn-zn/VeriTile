import bench.tritonbench_g.decay_cumsum.Kernels
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.PrepareCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.BackwardCorrect
/-! Real specifications for all three complete kernels and their index-commuted
candidates. Each contract requires bounded active flat accesses and zero
unspecified values, and concludes successful execution, the independent output
formula and preservation of every other tagged memory cell. Storage erasure
is used only here; floating-point equivalence is in FPEquiv.lean. -/

namespace VeriTile.Bench.TritonBenchG.DecayCumsum.Correctness.Forward
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamEmitMasked3DKernelIO₁
noncomputable section
/-- Complete streamed read/write windows. Active accesses must fit the finite
allocation; the shared real adapter requires zero unspecified values. -/
def originalIO (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) : StreamEmitMasked3DKernelIO₁ where
  kernel := (forward storage Bool.false g go sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,ForwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  inp1 := g
  out := go
  T := BT
  B1 := BK
  C := BK
  read1 := fun p₀ p₁ p₂ t j => p₂*sH+p₁*BT*DK+p₀*BK+j.val+t.val*DK
  write := fun p₀ p₁ p₂ t j => p₂*sH+p₁*BT*DK+p₀*BK+j.val+t.val*DK
  mask1 := fun p₀ _ _ _ j => p₀*BK+j.val<DK
  writeMask := fun p₀ _ _ _ j => p₀*BK+j.val<DK

def indexCommutedIO (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) : StreamEmitMasked3DKernelIO₁ :=
  { originalIO storage g go sH sT sD B H T BT BK DK scale with
    kernel := (forward storage Bool.true g go sH sT sD B H T BT BK DK scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,ForwardCorrect.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

/-- Independent prefix sum of the initially loaded rows. -/
def scan (BT BK : Nat) (xs : Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  1.44269504 * ∑ k : Fin (t.val+1), xs (Fin.castLE t.isLt k) j


specification forward_original_correct (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ)
    (hne : g≠go) (hBK : BK≤DK) :
    Spec.Real (originalIO storage g go sH sT sD B H T BT BK DK scale ⊨
      fun _ _ _ xs t j => scan BT BK xs t j) := by
  exact ForwardCorrect.original_correct storage g go sH sT sD B H T BT BK DK scale hne hBK

specification forward_index_commuted_correct (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ)
    (hne : g≠go) (hBK : BK≤DK) :
    Spec.Real (indexCommutedIO storage g go sH sT sD B H T BT BK DK scale ⊨
      fun _ _ _ xs t j => scan BT BK xs t j) := by
  exact ForwardCorrect.index_commuted_correct storage g go sH sT sD B H T BT BK DK scale hne hBK

end
end VeriTile.Bench.TritonBenchG.DecayCumsum.Correctness.Forward

namespace VeriTile.Bench.TritonBenchG.DecayCumsum.Correctness.Prepare
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO
noncomputable section
/-- Q/K loads and both stores are lane-masked. The gate stream additionally
includes every final-row lane because last_decay is loaded without a mask.
All these actual accesses must fit the finite allocation. -/
def originalIO (source gate : Storage) (Q K G QG KG : RegionName)
    (s_qk_h sT sD B H T DK BT BK : Nat) (scale : ℝ) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := (prepare source gate Bool.false Q K G QG KG s_qk_h sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,PrepareCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 3
  nOut := 2
  bufs := [Q, K, G, QG, KG]
  inp := fun i => match i with
    | ⟨0, _⟩ => Q
    | ⟨1, _⟩ => K
    | ⟨2, _⟩ => G
    | ⟨n + 3, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => QG
    | ⟨1, _⟩ => KG
    | ⟨n + 2, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => p₀ * BK + j.val < DK
    | ⟨1, _⟩ => p₀ * BK + j.val < DK
    | ⟨2, _⟩ => p₀ * BK + j.val < DK ∨ t.val = BT - 1
    | ⟨n + 3, h⟩ => absurd h (by omega)
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK

def indexCommutedIO (source gate : Storage) (Q K G QG KG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO :=
  { originalIO source gate Q K G QG KG sH sT sD B H T DK BT BK scale with
    kernel := (prepare source gate Bool.true Q K G QG KG sH sT sD B H T BT BK DK scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,PrepareCorrect.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

noncomputable def values (BT BK : Nat) (scale : ℝ)
    (xs : Fin 3 → Fin BT → Fin BK → ℝ) (o : Fin 2) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ =>
      xs (⟨0, by omega⟩ : Fin 3) t j
        * Real.exp (xs (⟨2, by omega⟩ : Fin 3) t j * Real.log 2) * scale
  | ⟨1, _⟩ =>
      if h : 0 < BT then
        xs (⟨1, by omega⟩ : Fin 3) t j
          * Real.exp ((xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
              - xs (⟨2, by omega⟩ : Fin 3) t j) * Real.log 2)
      else 0
  | ⟨n + 2, h2⟩ => absurd h2 (by omega)



specification prepare_original_correct (source gate : Storage) (Q K G QG KG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q≠QG) (hQ_KG : Q≠KG) (hK_QG : K≠QG) (hK_KG : K≠KG)
    (hG_QG : G≠QG) (hG_KG : G≠KG) (hQG_KG : QG≠KG)
    (hBK : BK≤DK) (hBT : 0<BT) :
    Spec.Real (originalIO source gate Q K G QG KG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK scale xs o t j) := by
  exact PrepareCorrect.original_correct source gate Q K G QG KG sH sT sD B H T DK BT BK scale hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT

specification prepare_index_commuted_correct (source gate : Storage) (Q K G QG KG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q≠QG) (hQ_KG : Q≠KG) (hK_QG : K≠QG) (hK_KG : K≠KG)
    (hG_QG : G≠QG) (hG_KG : G≠KG) (hQG_KG : QG≠KG)
    (hBK : BK≤DK) (hBT : 0<BT) :
    Spec.Real (indexCommutedIO source gate Q K G QG KG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK scale xs o t j) := by
  exact PrepareCorrect.index_commuted_correct source gate Q K G QG KG sH sT sD B H T DK BT BK scale hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT

end
end VeriTile.Bench.TritonBenchG.DecayCumsum.Correctness.Prepare

namespace VeriTile.Bench.TritonBenchG.DecayCumsum.Correctness.Backward
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO
noncomputable section
/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK


def indexCommutedIO (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO :=
  { originalIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale with
    kernel := (backward source gate Bool.true DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

noncomputable def dq (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)

noncomputable def dk (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0

noncomputable def dg (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (dq BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - dk BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)

noncomputable def values (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => dq BT BK xs t j
  | ⟨1, _⟩ => dk BT BK xs t j
  | ⟨2, _⟩ => dg BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)



specification backward_original_correct (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    Spec.Real (originalIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK xs o t j) := by
  exact BackwardCorrect.original_correct source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG hBK hBT

specification backward_index_commuted_correct (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    Spec.Real (indexCommutedIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK xs o t j) := by
  exact BackwardCorrect.index_commuted_correct source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG hBK hBT

end
end VeriTile.Bench.TritonBenchG.DecayCumsum.Correctness.Backward


#guard_msgs (drop info) in
#auditModuleAxioms
