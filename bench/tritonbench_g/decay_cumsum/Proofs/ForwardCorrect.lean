import bench.tritonbench_g.decay_cumsum.Proofs.ForwardProjection
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardCandidate
import bench.tritonbench_g.decay_cumsum.Proofs.RealIO
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardCorrect
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamEmitMasked3DKernelIO₁
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem source_projects (storage : Storage) (swapped : Bool) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    (forward storage swapped g go sH sT sD B H T BT BK DK scale).toAlgorithm? =
    .ok (forward storage swapped g go sH sT sD B H T BT BK DK scale).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

/-- Complete streamed read/write windows. Active accesses must fit the finite
allocation; the shared real adapter requires zero unspecified values. -/
def originalIO (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) : StreamEmitMasked3DKernelIO₁ where
  kernel := (forward storage Bool.false g go sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,source_projects,
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
      simp only [ComputeKernel.eraseDType,source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

/-- Independent prefix sum of the initially loaded rows. -/
def scan (BT BK : Nat) (xs : Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  1.44269504 * ∑ k : Fin (t.val+1), xs (Fin.castLE t.isLt k) j

theorem original_correct (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ)
    (hne : g≠go) (hBK : BK≤DK) :
    Spec.Real (originalIO storage g go sH sT sD B H T BT BK DK scale ⊨
      fun _ _ _ xs t j => scan BT BK xs t j) := by
  have h := StreamEmitMasked3DKernelIO₁.Implements.of_identityRounding
    (RealReference.fwdDecayCumsumKernelIO g go sH sT sD B H T scale BT BK DK) rfl
    (fun _ _ _ xs t j => RealReference.fwdDecayStreamSpec BT BK xs t j)
    (RealReference.fwd_decay_cumsum_io_correctness .triv g go sH sT sD B H T scale BT BK DK hne hBK)
  simpa only [Spec.Real,StreamEmitMasked3DKernelIO₁.Implements,
    originalIO,RealReference.fwdDecayCumsumKernelIO,
    ForwardProjection.original_flat_exec,scan,RealReference.fwdDecayStreamSpec] using h

theorem index_commuted_correct (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ)
    (hne : g≠go) (hBK : BK≤DK) :
    Spec.Real (indexCommutedIO storage g go sH sT sD B H T BT BK DK scale ⊨
      fun _ _ _ xs t j => scan BT BK xs t j) := by
  have h := original_correct storage g go sH sT sD B H T BT BK DK scale hne hBK
  simpa only [Spec.Real,StreamEmitMasked3DKernelIO₁.Implements,indexCommutedIO,
    originalIO,ForwardCandidate.flat_exec_equal] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardCorrect
