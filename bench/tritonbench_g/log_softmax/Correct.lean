import bench.tritonbench_g.log_softmax.Proofs.Real
import bench.tritonbench_g.log_softmax.Proofs.Execution
import bench.tritonbench_g.log_softmax.Proofs.Contracts
import bench.tritonbench_g.log_softmax.Proofs.Region
import bench.tritonbench_g.log_softmax.Proofs.Precision

namespace VeriTile.Bench.TritonBenchG.LogSoftmax

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₁

/-- **The backward headline**: `log_softmax_backward_kernel` implements the
log-softmax VJP `in_grad = out_grad − exp(out) · (Σ_row out_grad)` on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program-id pair whose active lanes are in bounds, and every launch state whose
input windows hold `xs`/`ys` at the active lanes, the translated pointer kernel
terminates, every active output lane `j` holds `logSoftmaxBackwardOf` applied to
the `out`/`out_grad` tiles rebuilt from `xs`/`ys`, and every other memory cell,
including the masked-off lanes of the output window, is unchanged.

The kernel's two loads are masked **without** an `other=` default, so at the
masked-off lanes the register tiles carry `s₀.undef` — which
`Masked2DKernelIO₂.Implements` pins to `0`. The row reduction
`scale = tl.sum(out_grad, 1)` genuinely consumes those pinned zeros, so the
honest per-lane spec sums the tile with masked-off lanes set to `0`
(`logSoftmaxBackwardPure{Out,Grad}Tile`). Because the pin is a hypothesis of the
`⊨` obligation, this headline is discharged via
`Masked2DKernelIO₂.Implements.intro_undef` (the `undef`-threading sibling of
`intro`), not the plain `intro`.

One honest, **truth-required** side hypothesis: `hOutInj` (distinct lanes of a
program's tile must not alias the same output cell — otherwise only the last
store survives). Unlike the forward headline, **no** `0 < BLOCK_N` is needed:
the backward kernel's only reduction is `tl.sum`, which is total on an empty
axis (`= 0`), so termination never fails. -/
specification log_softmax_backward_kernel_correctness
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_M * BLOCK_N) =>
        outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j))) :
    Masked2DKernelIO₂.Implements
      (logSoftmaxBackwardIO out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M BLOCK_N)
      (fun p₀ _ xs ys j =>
          logSoftmaxBackwardOf BLOCK_M BLOCK_N
            (logSoftmaxBackwardPureOutTile p₀ M N BLOCK_M BLOCK_N xs)
            (logSoftmaxBackwardPureGradTile p₀ M N BLOCK_M BLOCK_N ys)
            (laneIdx BLOCK_M BLOCK_N j)) := by
  refine Masked2DKernelIO₂.Implements.intro_undef _ ?_ ?_ ?_
  · exact log_softmax_backward_kernel_flattenOk out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N
  · intro bounds s hin1 hin2 hout _
    exact log_softmax_backward_kernel_traceSafe out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N bounds s hin1 hin2 hout
  · intro s₀ xs ys hundef hx hy
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      log_softmax_backward_kernel_region_run out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N s₀ xs ys hundef
        (inj_of_lane_inj (hOutInj (s₀.pids 0) (s₀.pids 1))) hx hy
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩

/-- **The headline**: `log_softmax_kernel` implements the row-wise log-softmax
on its masked IO signature — for every disjoint flat placement of
`input_ptr`/`output_ptr`, every program-id pair whose active lanes are in
bounds, and every launch state whose input window holds the tile `xs` at the
active lanes, the translated pointer kernel terminates, every active output
lane `j` holds `log(exp(x − rowmax) / ∑ exp(x − rowmax))` — the reduction core
`logSoftmaxOf` applied to the tile rebuilt from `xs` (with the kernel's own
`other = -inf` padding on the masked-off lanes, which is what makes the
row reduction independent of memory outside the window) — and every other
memory cell, including the masked-off lanes of the output window, is
unchanged.

Two honest, **truth-required** side hypotheses: `hOutInj` (distinct lanes of a
program's tile must not alias the same output cell — otherwise only the last
store survives) and `hBN : 0 < BLOCK_N` (`tl.max` over a zero-length reduction
axis has no value and `exec` returns `none`, so termination genuinely fails). -/
specification log_softmax_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_M * BLOCK_N) =>
        outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j))) :
    logSoftmaxIO output_ptr input_ptr M N K BLOCK_M BLOCK_N
      ⊨ fun p₀ _ xs j =>
          logSoftmaxOf BLOCK_M BLOCK_N
            (logSoftmaxPureTile p₀ M N BLOCK_M BLOCK_N xs)
            (laneIdx BLOCK_M BLOCK_N j) := by
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact log_softmax_kernel_flattenOk output_ptr input_ptr M N K BLOCK_M BLOCK_N
  · intro bounds s hin hout _
    exact log_softmax_kernel_traceSafe output_ptr input_ptr M N K BLOCK_M BLOCK_N
      hBN bounds s hin hout
  · intro s₀ xs hx
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      log_softmax_kernel_region_run output_ptr input_ptr M N K BLOCK_M BLOCK_N
        hBN s₀ xs (inj_of_lane_inj (hOutInj (s₀.pids 0) (s₀.pids 1))) hx
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩

/-- Independent full-row log-softmax for both complete typed programs, with successful
flat execution, active address bounds and the complete unwritten-memory frame. -/
specification forward_typed_correct (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat)
    (hC : 0 < C) (hK : 0 < K) (hNC : N ≤ C) :
    Masked2DKernelIO₁.Implements (typedForwardIO input output v OUT X M N K R C)
      (fun _ _ xs j ↦ forwardExpected N R C hNC xs j) :=
  forward_typed_formula input output v OUT X M N K R C hC hK hNC

/-- Independent full-row VJP for both complete typed programs under the flat
interface's explicit zero-undefined launch invariant. -/
specification backward_typed_correct (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (hK : 0 < K) (hNC : N ≤ C) :
    Masked2DKernelIO₂.Implements (typedBackwardIO input upstream output v Y G DX M N K R C)
      (fun _ _ xs ys j ↦ backwardExpected N R C hNC xs ys j) :=
  backward_typed_formula input upstream output v Y G DX M N K R C hK hNC

/-- Complete source execution with independent full-row log-softmax and raw memory
frame, allowing arbitrary undefined values and input/output aliases. -/
specification forward_region_correct (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat)
    (hC : 0 < C) (hK : 0 < K) (hNC : N ≤ C) (s : BlockState) :
    ∃ t, exec (typedForward input output v OUT X M N K R C).eraseDType.toAlgKernel s = some t ∧
      (∀ (r : Fin R) (i : Fin C) (hr : s.pids 0*R+r.val<M) (hi : i.val<N),
        t.readMem OUT (outOffset (s.pids 0) (s.pids 1) N K R (r,i,PUnit.unit)) =
          Math.forward (fun j : Fin N ↦
            s.readMem X ((s.pids 0*R+r.val)*N*K+j.val*K+s.pids 1)) ⟨i.val,hi⟩) ∧
      (∀ r o, (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        ¬(OUT = r ∧ outOffset (s.pids 0) (s.pids 1) N K R i = o)) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := SourceRegion.forward_run input output v OUT X M N K R C hC hK s
  refine ⟨t, he, ?_, hf⟩
  intro r i hr hi
  exact (hv (r,i,PUnit.unit) ⟨hr,hi⟩).trans
    (Formula.forward_memory s X (s.pids 0) (s.pids 1) M N K R C hNC r i hr hi)

/-- Complete backward execution with its actual padding-dependent mathematical
formula and full raw frame, allowing arbitrary undefined values and aliases. -/
specification backward_region_correct (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (hK : 0 < K) (s : BlockState) :
    ∃ t, exec (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgKernel s = some t ∧
      (∀ (r : Fin R) (i : Fin C), s.pids 0*R+r.val<M → i.val<N →
        t.readMem DX (outOffset (s.pids 0) (s.pids 1) N K R (r,i,PUnit.unit)) =
          Math.backwardPadded C
            (Formula.paddedSample s Y (s.pids 0*R+r.val) N K (s.pids 1))
            (Formula.paddedSample s G (s.pids 0*R+r.val) N K (s.pids 1)) i.val) ∧
      (∀ r o, (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        ¬(DX = r ∧ outOffset (s.pids 0) (s.pids 1) N K R i = o)) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := SourceRegion.backward_run input upstream output v Y G DX M N K R C hK s
  refine ⟨t, he, ?_, hf⟩
  intro r i hr hi
  exact (hv (r,i,PUnit.unit) ⟨hr,hi⟩).trans
    (Formula.backward_memory s Y G M N K R C r i hr)

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.LogSoftmax
