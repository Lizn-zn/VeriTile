import bench.tritonbench_g.log_softmax.Proofs.TypedIO
import bench.tritonbench_g.log_softmax.Proofs.FormulaForward
import bench.tritonbench_g.log_softmax.Proofs.FormulaBackward

namespace VeriTile.Bench.TritonBenchG.LogSoftmax
open VeriTile.Triton

/-- A full logical row of independent mathematical log-softmax values. -/
noncomputable def forwardExpected (N R C : Nat) (hNC : N ≤ C)
    (xs : Fin (R*C) → ℝ) (j : Fin (R*C)) : ℝ :=
  let idx := laneIdx R C j
  if hi : idx.2.1.val < N then
    Math.forward (fun k : Fin N ↦
      xs (laneOf R C (idx.1, ⟨k.val, lt_of_lt_of_le k.isLt hNC⟩, PUnit.unit)))
      ⟨idx.2.1.val, hi⟩
  else 0

/-- Full logical-row VJP for the flat interface's explicitly zero-padded inputs. -/
noncomputable def backwardExpected (N R C : Nat) (hNC : N ≤ C)
    (xs ys : Fin (R*C) → ℝ) (j : Fin (R*C)) : ℝ :=
  let idx := laneIdx R C j
  Math.backward N
    (fun k ↦ if hk : k < C then xs (laneOf R C (idx.1, ⟨k,hk⟩, PUnit.unit)) else 0)
    (fun k ↦ if hk : k < N then
      ys (laneOf R C (idx.1, ⟨k,lt_of_lt_of_le hk hNC⟩, PUnit.unit)) else 0)
    idx.2.1.val

theorem forward_typed_formula (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat)
    (hC : 0 < C) (hK : 0 < K) (hNC : N ≤ C) :
    Masked2DKernelIO₁.Implements (typedForwardIO input output v OUT X M N K R C)
      (fun _ _ xs j ↦ forwardExpected N R C hNC xs j) := by
  intro A hd hr hc p0 p1 hx ho hscr xs s hp0 hp1 hu hxs
  obtain ⟨t, he, hv, hf⟩ := forward_typed_io input output v OUT X M N K R C hC hK
    A hd hr hc p0 p1 hx ho hscr xs s hp0 hp1 hu hxs
  refine ⟨t, he, ?_, hf⟩
  intro j hj
  have ha : active p0 M N R (laneIdx R C j) := hj
  rw [hv j hj]
  simp only [forwardExpected, dif_pos ha.2]
  exact Formula.forward_pure_tile p0 M N R C hNC xs
    (laneIdx R C j).1 (laneIdx R C j).2.1 ha.1 ha.2

theorem backward_typed_formula (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (hK : 0 < K) (hNC : N ≤ C) :
    Masked2DKernelIO₂.Implements (typedBackwardIO input upstream output v Y G DX M N K R C)
      (fun _ _ xs ys j ↦ backwardExpected N R C hNC xs ys j) := by
  intro A hd hr hc p0 p1 hx hy ho hscr xs ys s hp0 hp1 hu hxs hys
  obtain ⟨t, he, hv, hf⟩ := backward_typed_io input upstream output v Y G DX M N K R C hK
    A hd hr hc p0 p1 hx hy ho hscr xs ys s hp0 hp1 hu hxs hys
  refine ⟨t, he, ?_, hf⟩
  intro j hj
  have ha : active p0 M N R (laneIdx R C j) := hj
  rw [hv j hj]
  exact Formula.backward_pure_tile p0 M N R C hNC xs ys
    (laneIdx R C j).1 (laneIdx R C j).2.1 ha.1 ha.2

end VeriTile.Bench.TritonBenchG.LogSoftmax
