import bench.tritonbench_g.mul_exponent_compensator.Kernels
import bench.tritonbench_g.mul_exponent_compensator.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Independent mathematical constant-scaling specifications for both sources.
Each includes successful flat execution, output values and the complete frame. -/
namespace VeriTile.Bench.TritonBenchG.MulExponentCompensator
open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.KernelIO₁

/-! ## The spec: `mulIO ⊨` the constant scaling -/

/-- `mul_kernel`'s **IO signature** — the whole kernel-specific audit
surface of the headline: `src` in, `dst` out; tile lengths
`Bin = Bout = B`; program `pid` reads its tile at `pid * B` of `src` and
writes the same window of `dst`. The windows are declared, not parsed from
the kernel: they formalize the host-side launch convention
(`idxs = pid * BLOCK_SIZE + arange`), and the headline **proves** the
kernel's actual addressing matches them. Buffer sizes are not signature
content: the headline quantifies over every allocation large enough for the
windows. -/
noncomputable def mulIO (src dst : RegionName) (B : Nat) : KernelIO₁ where
  kernel := mul_kernel src dst B
  inp := src
  out := dst
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- **The headline**: `mul_kernel` implements the elementwise constant
scaling `xs i * exponentCompensator` on its IO signature. Spelled out (see
`KernelIO₁.Implements`): for every disjoint flat placement of the two
buffers, every program id whose windows are in bounds, and every launch
state whose input window holds `xs` — everything else arbitrary — the
translated pointer kernel terminates, its output window holds
`xs i * exponentCompensator`, and every other memory cell is unchanged.
Proof: `Implements.intro` assembles the region-model triple with the bridge
side conditions. -/
specification mul_kernel_correctness (src dst : RegionName) (B : Nat)
    (hB : 0 < B) :
    mulIO src dst B ⊨ fun xs i => xs i * exponentCompensator := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul_kernel_flattenOk src dst B
  · intro bounds s h1 h2 _
    exact mul_kernel_traceSafe src dst B bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul_kernel_region_run src dst B hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

noncomputable def commutedIO (src dst : RegionName) (B : Nat) : KernelIO₁ :=
  { mulIO src dst B with
    kernel := mul_commuted_kernel src dst B
    projection := by rfl }

specification mul_commuted_correctness (src dst : RegionName) (B : Nat)
    (hB : 0 < B) :
    Spec.Real (commutedIO src dst B ⊨ fun xs i => xs i * exponentCompensator) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact commuted_flattenOk src dst B
  · intro bounds s hs hd _
    exact commuted_traceSafe src dst B bounds s hs hd
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := mul_kernel_region_run src dst B hB s xs hx
    exact ⟨s1, (commuted_exec src dst B s).trans he, hv, fun r o hmiss _ => hf r o hmiss⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MulExponentCompensator
