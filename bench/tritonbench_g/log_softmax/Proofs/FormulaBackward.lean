import bench.tritonbench_g.log_softmax.Proofs.Real
import bench.tritonbench_g.log_softmax.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.Formula
open VeriTile.Triton
open scoped BigOperators
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

/-- A finite loaded row retains every padded gradient in its reduction. -/
theorem backward_row (R C : Nat) (out grad : Tile .real [R, C])
    (r : Fin R) (i : Fin C) (y dy : Nat → ℝ)
    (hy : ∀ j : Fin C, out.data (r, j, PUnit.unit) = some (y j.val))
    (hdy : ∀ j : Fin C, grad.data (r, j, PUnit.unit) = some (dy j.val)) :
    logSoftmaxBackwardOf R C out grad (r, i, PUnit.unit) =
      Math.backwardPadded C y dy i.val := by
  have hsum : (@Finset.sum (Fin C) (WithBot ℝ) _ Finset.univ
      (fun j ↦ some (dy j.val))) =
      (some (∑ j ∈ Finset.range C, dy j) : WithBot ℝ) := by
    rw [WithBot.sum_someTerm_eq_some]
    exact congrArg some (Fin.sum_univ_eq_sum_range dy C)
  simp only [logSoftmaxBackwardOf, Tile.bop, Tile.uop,
    Tile.reduceSum, Tile.reduceSumKeep_data]
  change (WithBot.realSub (grad.data (r, i, PUnit.unit))
    (WithBot.realMul (WithBot.realExp (out.data (r, i, PUnit.unit)))
      (∑ j : Fin C, grad.data (r, j, PUnit.unit)))).unbotD 0 = _
  simp only [hy, hdy, hsum]
  rfl

/-- Removing padded terms is a separate, explicit hypothesis. -/
theorem backward_row_zero_padding (R C N : Nat) (hNC : N ≤ C)
    (out grad : Tile .real [R, C]) (r : Fin R) (i : Fin C)
    (y dy : Nat → ℝ)
    (hy : ∀ j : Fin C, out.data (r, j, PUnit.unit) = some (y j.val))
    (hdy : ∀ j : Fin C, grad.data (r, j, PUnit.unit) = some (dy j.val))
    (hpad : ∀ j, N ≤ j → j < C → dy j = 0) :
    logSoftmaxBackwardOf R C out grad (r, i, PUnit.unit) =
      Math.backward N y dy i.val := by
  rw [backward_row R C out grad r i y dy hy hdy]
  have hC : C = N + (C - N) := by omega
  simpa only [← hC] using
    Math.backwardPadded_eq_of_zero N (C-N) y dy i.val
      (fun j hj ↦ hpad (N+j) (by omega) (by omega))

/-- The IO tile builders explicitly use zero padding; this corollary bridges
that narrower interface to the conventional memory-independent VJP. -/
theorem backward_pure_tile (p M N R C : Nat) (hNC : N ≤ C)
    (xs ys : Fin (R*C) → ℝ) (r : Fin R) (i : Fin C)
    (hr : p*R+r.val < M) (hi : i.val < N) :
    logSoftmaxBackwardOf R C
      (logSoftmaxBackwardPureOutTile p M N R C xs)
      (logSoftmaxBackwardPureGradTile p M N R C ys) (r,i,PUnit.unit) =
      Math.backward N
        (fun j ↦ if hj : j<C then xs (laneOf R C (r,⟨j,hj⟩,PUnit.unit)) else 0)
        (fun j ↦ if hj : j<N then ys (laneOf R C (r,⟨j,lt_of_lt_of_le hj hNC⟩,PUnit.unit)) else 0)
        i.val := by
  let y : Nat → ℝ := fun j ↦ if hj : j<N then
    xs (laneOf R C (r,⟨j,lt_of_lt_of_le hj hNC⟩,PUnit.unit)) else 0
  let dy : Nat → ℝ := fun j ↦ if hj : j<N then
    ys (laneOf R C (r,⟨j,lt_of_lt_of_le hj hNC⟩,PUnit.unit)) else 0
  have h := backward_row_zero_padding R C N hNC
    (logSoftmaxBackwardPureOutTile p M N R C xs)
    (logSoftmaxBackwardPureGradTile p M N R C ys) r i y dy
    (by intro j; simp [logSoftmaxBackwardPureOutTile, active, hr, y]; split_ifs <;> rfl)
    (by intro j; simp [logSoftmaxBackwardPureGradTile, active, hr, dy]; split_ifs <;> rfl)
    (by intro j hn _; simp [dy, not_lt.mpr hn])
  simpa [Math.backward, y, dy, hi, i.isLt] using h

/-- Actual finite loaded samples, retaining the source's omitted `other`. -/
noncomputable def paddedSample (s : BlockState) (X : RegionName)
    (row N K col j : Nat) : ℝ :=
  if j<N then s.readMem X (row*N*K+j*K+col)
  else s.undef X (row*N*K+j*K+col)

/-- State-facing backward readback uses arbitrary undefined-padding samples. -/
theorem backward_memory (s : BlockState) (P G : RegionName)
    (M N K R C : Nat) (r : Fin R) (i : Fin C)
    (hr : s.pids 0*R+r.val<M) :
    logSoftmaxBackwardSpec s P G M N K R C (r,i,PUnit.unit) =
      Math.backwardPadded C
        (paddedSample s P (s.pids 0*R+r.val) N K (s.pids 1))
        (paddedSample s G (s.pids 0*R+r.val) N K (s.pids 1)) i.val := by
  exact backward_row R C
    (logSoftmaxBackwardOutTile s P M N K R C)
    (logSoftmaxBackwardGradTile s G M N K R C) r i _ _
    (by intro j; simp [logSoftmaxBackwardOutTile, active, hr, outOffset, paddedSample];
        split_ifs <;> rfl)
    (by intro j; simp [logSoftmaxBackwardGradTile, active, hr, outOffset, paddedSample];
        split_ifs <;> rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax.Formula
