import bench.tritonbench_g.relu_triton_kernel.Kernels
import bench.tritonbench_g.relu_triton_kernel.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Complete guarded-store fusion at fp32. The input/output windows contain B lanes;
loads are active below N and stores additionally require pid = 0. The supporting
execution theorem proves the stronger exact gated/tail frame. No numerical law
or finite-input condition is required, but the interpreter must support comparison. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.Equivalence
open VeriTile Triton FP.Structural FPProofs
open scoped VeriTile.Spec

def originalIO (x y : RegionName) (N B : Nat) : KernelIO₁ where
  kernel := relu_kernel x y N B
  inp := x
  out := y
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def fusedIO (x y : RegionName) (N B : Nat) : KernelIO₁ :=
  (originalIO x y N B).withKernel (relu_fused_kernel x y N B)

def original (x y : RegionName) (N B : Nat) : FP.Kernel₁ where
  io := originalIO x y N B
  profile := FP.fp32
  domain := fun _ => .top
  comparisons := [.le .fp32 .real]

def fused (x y : RegionName) (N B : Nat) : FP.Kernel₁ :=
  { original x y N B with io := fusedIO x y N B }

def assumptions : Spec.Assumptions FP.GuardedFragment := []

specification relu_fused_equiv (x y : RegionName) (N B : Nat) (_hB : 0 < B) :
    original x y N B ≡[assumptions] fused x y N B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := original x y N B) (rhs := fused x y N B)
    (structural := fun _ _ => False) rfl rfl
  refine ⟨by simp [IO₁PrivateScratch, original, originalIO],
    by simp [IO₁PrivateScratch, fused, fusedIO, originalIO], ?_⟩
  intro α _ M D _ hc plans s _
  have hcmp := hc (.le .fp32 .real) (by simp [original])
  simp only [FP.Scheduled.Comparison.Supported] at hcmp
  obtain ⟨cmp, hcmp⟩ := Option.isSome_iff_exists.mp hcmp
  let A := FP.fp32.algebra M plans
  have hcA : A.compareLe none .real = some cmp := hcmp
  obtain ⟨a, ha, hma⟩ := original_run A cmp hcA x y N B s
  obtain ⟨b, hb, hmb⟩ := fused_run A cmp hcA x y N B s
  refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
  · intro i
    exact congrFun (congrFun (hma.trans hmb.symm) y) (s.pids 0 * B + i.val)
  · intro r o ho _
    rw [hma]
    apply outputState_frame
    rcases ho with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ => ho i)
  · intro r o ho _
    rw [hmb]
    apply outputState_frame
    rcases ho with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ => ho i)

#print_fp_assumptions relu_fused_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.Equivalence
