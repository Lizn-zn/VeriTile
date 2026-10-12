import bench.tritonbench_g.decay_cumsum.Proofs.PrepareExecution
import bench.tritonbench_g.decay_cumsum.Proofs.MaskCandidate
import bench.tritonbench_g.decay_cumsum.Proofs.GroupedRealIO
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareCorrect
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem source_projects (source gate : Storage) (swapped : Bool) (Q K G QG KG : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (prepare source gate swapped Q K G QG KG sH sT sD B H T BT BK DK scale).toAlgorithm? =
      .ok (prepare source gate swapped Q K G QG KG sH sT sD B H T BT BK DK scale).toAlgKernel := by
  cases source <;> cases gate <;> cases swapped <;> rfl

/-- Q/K loads and both stores are lane-masked. The gate stream additionally
includes every final-row lane because last_decay is loaded without a mask.
All these actual accesses must fit the finite allocation. -/
def originalIO (source gate : Storage) (Q K G QG KG : RegionName)
    (s_qk_h sT sD B H T DK BT BK : Nat) (scale : ℝ) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := (prepare source gate Bool.false Q K G QG KG s_qk_h sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,source_projects,
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
      simp only [ComputeKernel.eraseDType,source_projects,
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


theorem original_correct (source gate : Storage) (Q K G QG KG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q≠QG) (hQ_KG : Q≠KG) (hK_QG : K≠QG) (hK_KG : K≠KG)
    (hG_QG : G≠QG) (hG_KG : G≠KG) (hQG_KG : QG≠KG)
    (hBK : BK≤DK) (hBT : 0<BT) :
    Spec.Real (originalIO source gate Q K G QG KG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK scale xs o t j) := by
  have h := StreamGroupedEmitMasked3DKernelIO.Implements.of_identityRounding
    (RealReference.prepareQgKgKernelIO Q K G QG KG sH DK BT BK scale) rfl
    (fun _ _ _ xs o t j => values BT BK scale xs o t j)
    (RealReference.prepare_qg_kg_io_correctness .triv Q K G QG KG sH DK BT BK scale
      hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT)
  intro A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  obtain ⟨v,he,hv,hf⟩ := h A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
  change exec (A.flattenKernel (RealReference.prepare_qg_kg_surface Q K G QG KG sH DK BT BK scale).toAlgKernel)
    (A.flattenState u) = some v at he
  rw [PrepareExecution.flat A source gate Q K G QG KG sH sT sD B H T BT BK DK scale] at he
  obtain ⟨w,hw,rfl⟩ := Option.map_eq_some_iff.mp he
  refine ⟨w,hw,?_,?_⟩
  · simpa only [PrepareRegisterFrame.setOffs,BlockState.setReg_readMem] using hv
  · simpa only [PrepareRegisterFrame.setOffs,BlockState.setReg_mem] using hf

theorem index_commuted_correct (source gate : Storage) (Q K G QG KG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q≠QG) (hQ_KG : Q≠KG) (hK_QG : K≠QG) (hK_KG : K≠KG)
    (hG_QG : G≠QG) (hG_KG : G≠KG) (hQG_KG : QG≠KG)
    (hBK : BK≤DK) (hBT : 0<BT) :
    Spec.Real (indexCommutedIO source gate Q K G QG KG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK scale xs o t j) := by
  have h := original_correct source gate Q K G QG KG sH sT sD B H T DK BT BK scale
    hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT
  simpa only [Spec.Real,StreamGroupedEmitMasked3DKernelIO.Implements,
    indexCommutedIO,originalIO,PrepareCandidate.flat_exec_equal] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareCorrect
