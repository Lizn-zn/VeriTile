import bench.tritonbench_g.swiglu_backward.Proofs.Precision
import bench.tritonbench_g.swiglu_backward.Proofs.Math
import bench.tritonbench_g.swiglu_backward.Proofs.Candidate
import bench.tritonbench_g.swiglu_backward.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.SwigluBackward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₃ₓ₃

/-- **The headline**: `_swiglu_bwd_kernel` implements the SwiGLU backward
oracles on its masked three-input / three-output IO signature — for every
disjoint flat placement of the six buffers, every program id whose active
lanes are in bounds, and every launch state whose active input lanes hold
`xs` (the gate), `ys` (the value) and `douts` (the upstream gradient), the
translated pointer kernel terminates, every active `DX` lane `j` holds
`TiledActivation.swigluBwdA (douts j) (xs j) (ys j)`, every active `DY` lane
holds `TiledActivation.swigluBwdB (douts j) (xs j)`, every `OUT` lane that is
active **and** `RECOMPUTE_OUTPUT`-gated holds the recomputed forward
`TiledActivation.swiglu (xs j) (ys j)`, and every other memory cell is
unchanged — one statement covering both heuristic outcomes of the constexpr
flag. The output buffers must be pairwise distinct (`DX ≠ DY`, `OUT ≠ DX`,
`OUT ≠ DY`) so each readback sees through the later stores. Proof:
`Masked2DKernelIO₃ₓ₃.Implements.intro` assembles the region-model masked
triple with the flat-memory bridge side conditions. -/
specification swiglu_bwd_kernel_correctness
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY) :
    swigluBackwardIO X Y DOUT OUT DX DY stride_x_row stride_y_row
        stride_dout_row stride_out_row stride_dx_row stride_dy_row ncols
        BLOCK_N RECOMPUTE_OUTPUT ⊨
      fun _ _ xs ys douts =>
        (fun j => TiledActivation.swigluBwdA (douts j) (xs j) (ys j),
         fun j => TiledActivation.swigluBwdB (douts j) (xs j),
         fun j => TiledActivation.swiglu (xs j) (ys j)) := by
  refine Masked2DKernelIO₃ₓ₃.Implements.intro _ ?_ ?_ ?_
  · exact swiglu_bwd_kernel_flattenOk X Y DOUT OUT DX DY
      stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
  · intro bounds s h1 h2 h3 h4 h5 h6
    exact swiglu_bwd_kernel_traceSafe X Y DOUT OUT DX DY
      stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
      bounds s h1 h2 h3 h4 h5 h6
  · intro s₀ xs ys douts hx hy hdout
    exact swiglu_bwd_kernel_region_run X Y DOUT OUT DX DY
      stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
      hDXDY hOUTDX hOUTDY s₀ xs ys douts hx hy hdout

/-- Independent scalar gradients and optional forward recomputation, with full
flat execution and frame for both values of the source constexpr flag. -/
specification formula_correct (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy N B : Nat) (R : Bool)
    (hxy : DX ≠ DY) (hox : OUT ≠ DX) (hoy : OUT ≠ DY) :
    swigluBackwardIO X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R ⊨
      fun _ _ xs ys ds => (fun j => Math.dx (xs j) (ys j) (ds j),
        fun j => Math.dy (xs j) (ds j), fun j => Math.forward (xs j) (ys j)) := by
  simpa only [Math.dx_eq, Math.dy_eq, Math.forward_eq] using
    swiglu_bwd_kernel_correctness X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R hxy hox hoy

def typedIO (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    Masked2DKernelIO₃ₓ₃ :=
  { swigluBackwardIO X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R with
    kernel := (typedKernel input upstream gradient output v X Y DOUT OUT DX DY
      sx sy sd so sdx sdy N B R).eraseDType
    projection := typed_supported input upstream gradient output v X Y DOUT OUT DX DY
      sx sy sd so sdx sdy N B R }

/-- Both full typed variants satisfy independent gradient and optional-forward
formulas from the original input windows, with successful execution and the
complete flat frame. Output regions are pairwise distinct; inputs may alias
outputs because all input loads precede every store. -/
specification typed_real_correct (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool)
    (hxy : DX ≠ DY) (hox : OUT ≠ DX) (hoy : OUT ≠ DY) :
    VeriTile.Spec.Real (typedIO input upstream gradient output v X Y DOUT OUT DX DY
      sx sy sd so sdx sdy N B R ⊨
        fun _ _ xs ys ds => (fun j => Math.dx (xs j) (ys j) (ds j),
          fun j => Math.dy (xs j) (ds j), fun j => Math.forward (xs j) (ys j))) := by
  intro A hd hr hc p0 p1 h1 h2 h3 h4 h5 h6 xs ys ds s hp0 hp1 hu hx hy hz
  obtain ⟨t,he,hv1,hv2,hv3,hf⟩ := formula_correct X Y DOUT OUT DX DY
    sx sy sd so sdx sdy N B R hxy hox hoy
    A hd hr hc p0 p1 h1 h2 h3 h4 h5 h6 xs ys ds s hp0 hp1 hu hx hy hz
  refine ⟨t,?_,hv1,hv2,hv3,hf⟩
  exact (typed_flat_exec A input upstream gradient output v X Y DOUT OUT DX DY
    sx sy sd so sdx sdy N B R (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.SwigluBackward
