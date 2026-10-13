import bench.tritonbench_g.relu_triton_kernel.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel
open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁
/-- `relu_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B` — the window length each program owns;
* `read`/`write` — where program `pid` reads its input tile / writes its
  output tile (`pid * block_size`, the launch convention
  `offsets = pid * block_size + arange`);
* `mask` — program `pid`'s **read-active lanes**, `pid * block_size + j < N`:
  the kernel's load mask (every program loads);
* `writeMask` — program `pid`'s **write-active lanes**,
  `pid = 0 ∧ pid * block_size + j < N`: the kernel's `pid == 0`-gated store
  mask. Programs with `pid ≠ 0` have no write-active lanes; the frame asserts
  they write nothing.

Inactive lanes carry no obligations on either side. The windows and masks are
declared, not parsed from the kernel; the headline **proves** the kernel's
actual addressing, masking, and store gate match them. -/
def reluIO (x_ptr out_ptr : RegionName) (N block_size : Nat) :
    MaskedKernelIO₁ where
  kernel := relu_kernel x_ptr out_ptr N block_size
  inp := x_ptr
  out := out_ptr
  B := block_size
  read := fun pid => pid * block_size
  write := fun pid => pid * block_size
  mask := fun pid j => pid * block_size + j.val < N
  writeMask := fun pid j => pid = 0 ∧ pid * block_size + j.val < N

/-- **The headline**: `relu_kernel` implements lane-wise
`TiledActivation.relu` (`max 0 ·`) on its masked IO signature — the full
masked Hoare triple over flat pointer memory (see the module docstring),
**general over `N`**: the kernel's `pid == 0` store gate is part of the
audited IO signature (`reluIO.writeMask`), so programs with `pid ≠ 0` carry
a pure frame obligation and no value claim. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the bridge side conditions. -/
specification relu_kernel_correctness
    (x_ptr out_ptr : RegionName) (N block_size : Nat)
    (hB : 0 < block_size) :
    reluIO x_ptr out_ptr N block_size ⊨
      fun xs i => TiledActivation.relu (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact relu_kernel_flattenOk x_ptr out_ptr N block_size
  · intro bounds s h1 h2 _
    exact relu_kernel_traceSafe x_ptr out_ptr N block_size bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      relu_kernel_region_run x_ptr out_ptr N block_size hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩

/-- `relu_fused_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B` — the window length each program owns;
* `read`/`write` — where program `pid` reads its input tile / writes its
  output tile (`pid * block_size`, the launch convention
  `offsets = pid * block_size + arange`);
* `mask` — program `pid`'s **read-active lanes**, `pid * block_size + j < N`:
  the kernel's load mask (every program loads);
* `writeMask` — program `pid`'s **write-active lanes**,
  `pid = 0 ∧ pid * block_size + j < N`: the kernel's `pid == 0`-gated store
  mask. Programs with `pid ≠ 0` have no write-active lanes; the frame asserts
  they write nothing.

Inactive lanes carry no obligations on either side. The windows and masks are
declared, not parsed from the kernel; the headline **proves** the kernel's
actual addressing, masking, and store gate match them. -/
def reluFusedIO (x_ptr out_ptr : RegionName) (N block_size : Nat) :
    MaskedKernelIO₁ where
  kernel := relu_fused_kernel x_ptr out_ptr N block_size
  inp := x_ptr
  out := out_ptr
  B := block_size
  read := fun pid => pid * block_size
  write := fun pid => pid * block_size
  mask := fun pid j => pid * block_size + j.val < N
  writeMask := fun pid j => pid = 0 ∧ pid * block_size + j.val < N

/-- **The headline**: `relu_fused_kernel` implements lane-wise
`TiledActivation.relu` (`max 0 ·`) on its masked IO signature — the full
masked Hoare triple over flat pointer memory (see the module docstring),
**general over `N`**: the kernel's `pid == 0` store gate is part of the
audited IO signature (`reluFusedIO.writeMask`), so programs with `pid ≠ 0` carry
a pure frame obligation and no value claim. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the bridge side conditions. -/
specification relu_fused_kernel_correctness
    (x_ptr out_ptr : RegionName) (N block_size : Nat)
    (hB : 0 < block_size) :
    reluFusedIO x_ptr out_ptr N block_size ⊨
      fun xs i => TiledActivation.relu (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact relu_fused_kernel_flattenOk x_ptr out_ptr N block_size
  · intro bounds s h1 h2 _
    exact relu_fused_kernel_traceSafe x_ptr out_ptr N block_size bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      relu_fused_kernel_region_run x_ptr out_ptr N block_size hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel
