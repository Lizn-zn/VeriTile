import bench.tritonbench_g.log_softmax.Proofs.IO
import bench.tritonbench_g.log_softmax.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.LogSoftmax
open VeriTile.Triton

def typedForwardIO (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) : Masked2DKernelIO₁ :=
  { logSoftmaxIO OUT X M N K R C with
    kernel := (typedForward input output v OUT X M N K R C).eraseDType
    projection := typedForward_supported input output v OUT X M N K R C }

def typedBackwardIO (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) : Masked2DKernelIO₂ :=
  { logSoftmaxBackwardIO Y G DX M N K R C with
    kernel := (typedBackward input upstream output v Y G DX M N K R C).eraseDType
    projection := typedBackward_supported input upstream output v Y G DX M N K R C }

/-- Typed full flat execution and frame, with active-lane rather than whole-tile injectivity. -/
theorem forward_typed_io (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) (hC : 0 < C) (hK : 0 < K) :
    Masked2DKernelIO₁.Implements (typedForwardIO input output v OUT X M N K R C)
      (fun p₀ _ xs j ↦ logSoftmaxOf R C (logSoftmaxPureTile p₀ M N R C xs)
        (laneIdx R C j)) := by
  intro A hd hr hc p0 p1 hx ho hscr xs s hp0 hp1 hu hxs
  obtain ⟨t, he, hv, hf⟩ := SourceIO.forward_io OUT X M N K R C hC hK
    A hd hr hc p0 p1 hx ho hscr xs s hp0 hp1 hu hxs
  refine ⟨t, ?_, hv, hf⟩
  exact (forward_typed_flat_exec A input output v OUT X M N K R C
    (A.flattenState s)).trans he

/-- The flat bridge's zero-undef premise is retained explicitly by this IO relation. -/
theorem backward_typed_io (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (hK : 0 < K) :
    Masked2DKernelIO₂.Implements (typedBackwardIO input upstream output v Y G DX M N K R C)
      (fun p₀ _ xs ys j ↦ logSoftmaxBackwardOf R C
        (logSoftmaxBackwardPureOutTile p₀ M N R C xs)
        (logSoftmaxBackwardPureGradTile p₀ M N R C ys) (laneIdx R C j)) := by
  intro A hd hr hc p0 p1 hx hy ho hscr xs ys s hp0 hp1 hu hxs hys
  obtain ⟨t, he, hv, hf⟩ := SourceIO.backward_io Y G DX M N K R C hK
    A hd hr hc p0 p1 hx hy ho hscr xs ys s hp0 hp1 hu hxs hys
  refine ⟨t, ?_, hv, hf⟩
  exact (backward_typed_flat_exec A input upstream output v Y G DX M N K R C
    (A.flattenState s)).trans he

end VeriTile.Bench.TritonBenchG.LogSoftmax
