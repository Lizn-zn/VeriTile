import bench.tritonbench_g.decay_cumsum.Proofs.BackwardExecution
import bench.tritonbench_g.decay_cumsum.Proofs.MaskCandidate
import bench.tritonbench_g.decay_cumsum.Proofs.GroupedRealIO
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardCorrect
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem source_projects (source gate : Storage) (swapped : Bool) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) :
    (backward source gate swapped DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).toAlgorithm? =
      .ok (backward source gate swapped DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).toAlgKernel := by
  cases source <;> cases gate <;> cases swapped <;> rfl

/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,source_projects,
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
      simp only [ComputeKernel.eraseDType,source_projects,
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


theorem original_correct (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
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
  have h := StreamGroupedEmitMasked3DKernelIO.Implements.of_identityRounding
    (RealReference.bwdDecayCumsumKernelIO DQInner DQInter DKInner DKInter Q K G DG sH DK BT BK) rfl
    (fun _ _ _ xs o t j => values BT BK xs o t j)
    (RealReference.bwd_decay_global_cumsum_io_correctness .triv DQInner DQInter DKInner DKInter Q K G DG sH DK BT BK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG hBK hBT)
  intro A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  obtain ⟨v,he,hv,hf⟩ := h A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  change exec (A.flattenKernel (RealReference.bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG sH DK BT BK).toAlgKernel)
    (A.flattenState u) = some v at he
  rw [BackwardExecution.flat A source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale] at he
  obtain ⟨w,hw,rfl⟩ := Option.map_eq_some_iff.mp he
  refine ⟨w,hw,?_,?_⟩
  · simpa only [BackwardRegisterFrame.setOffs,BlockState.setReg_readMem] using hv
  · simpa only [BackwardRegisterFrame.setOffs,BlockState.setReg_mem] using hf

theorem index_commuted_correct (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
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
  have h := original_correct source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale
    hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG hBK hBT
  simpa only [Spec.Real,StreamGroupedEmitMasked3DKernelIO.Implements,
    indexCommutedIO,originalIO,BackwardCandidate.flat_exec_equal] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardCorrect
