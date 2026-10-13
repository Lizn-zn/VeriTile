import bench.tritonbench_g.swiglu_fwd.Proofs.Precision
import bench.tritonbench_g.swiglu_fwd.Proofs.Math
import bench.tritonbench_g.swiglu_fwd.Proofs.Candidate
import bench.tritonbench_g.swiglu_fwd.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.SwigluFwd

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- **The headline**: `_swiglu_fwd_kernel` implements the oracle SwiGLU
`TiledActivation.swiglu` (i.e. `x · σ(x) · y`) lane-wise on its masked 2-D IO
signature — for every disjoint flat placement of the three buffers, every
program-id pair `(row, col_block)` whose active lanes are in bounds, and every
launch state whose active input-window lanes hold `xs`/`ys`, the translated
pointer kernel terminates, every active output lane `j` holds
`TiledActivation.swiglu (xs j) (ys j)`, and every other memory cell is
unchanged. Proof: `Masked2DKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification swiglu_fwd_kernel_correctness
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    swigluIO X Y OUT stride_x_row stride_y_row stride_out_row ncols BLOCK_N
      ⊨ fun _ _ xs ys i => TiledActivation.swiglu (xs i) (ys i) := by
  refine Masked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact swiglu_fwd_kernel_flattenOk X Y OUT stride_x_row stride_y_row
      stride_out_row ncols BLOCK_N
  · intro bounds s h1 h2 h3 _
    exact swiglu_fwd_kernel_traceSafe X Y OUT stride_x_row stride_y_row
      stride_out_row ncols BLOCK_N bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := swiglu_fwd_kernel_region_run
      X Y OUT stride_x_row stride_y_row stride_out_row ncols BLOCK_N s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- Independent scalar formula for every active output of the complete source. -/
specification formula_correct (X Y OUT : RegionName) (sx sy so N B : Nat) :
    swigluIO X Y OUT sx sy so N B ⊨ fun _ _ xs ys i => Math.forward (xs i) (ys i) := by
  simpa only [Math.forward_eq] using swiglu_fwd_kernel_correctness X Y OUT sx sy so N B

def typedIO (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : Masked2DKernelIO₂ :=
  { swigluIO X Y OUT sx sy so N B with
    kernel := (typedKernel input partner output v X Y OUT sx sy so N B).eraseDType
    projection := typed_supported input partner output v X Y OUT sx sy so N B }

/-- Every source storage triple and both variants satisfy the independent scalar
formula, successful flat execution, active bounds and the complete memory frame.
Both program axes and all three independent row strides remain symbolic. -/
specification typed_real_correct (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    VeriTile.Spec.Real (typedIO input partner output v X Y OUT sx sy so N B
      ⊨ fun _ _ xs ys i => Math.forward (xs i) (ys i)) := by
  intro A hd hr hc p0 p1 hx hy ho hscr xs ys s hp0 hp1 hu hxs hys
  obtain ⟨t, he, hv, hf⟩ := formula_correct X Y OUT sx sy so N B
    A hd hr hc p0 p1 hx hy ho hscr xs ys s hp0 hp1 hu hxs hys
  refine ⟨t, ?_, hv, hf⟩
  exact (typed_flat_exec A input partner output v X Y OUT sx sy so N B
    (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd
