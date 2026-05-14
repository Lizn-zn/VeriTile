import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Activation

namespace VeriTile.Bench.TritonBenchG.SwigluFwd

open VeriTile.Triton

/-- Faithful transcription of `swiglu_fwd.py`'s `_swiglu_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
def swiglu_fwd_kernel
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  OUT += row * $(stride_out_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  out = x * tl.sigmoid(x) * y
  tl.store(OUT + cols, out, mask=cols < $(ncols))
}

def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val

/-- Algorithm-layer correctness for `_swiglu_fwd_kernel`. -/
theorem swiglu_fwd_kernel_correct
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (s s' : BlockState)
    (xs ys : Fin BLOCK_N → ℝ)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i)
    (hExec : exec (swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
          ncols BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_out_row BLOCK_N i
      s'.readMem OUT outAddr =
        if s.pids 1 * BLOCK_N + i.val < ncols then
          TiledActivation.swiglu (xs i) (ys i)
        else s.readMem OUT outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [exec, swiglu_fwd_kernel, stepStmts, stepStmt, evalOp,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [swigluOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
  · have hx := h_x i
    have hy := h_y i
    simp [swigluOffset, Nat.add_assoc] at hx hy
    simp [hi, TiledActivation.swiglu, TiledActivation.silu,
          tile_elementwise, hx, hy]
  · simp [hi]

/-- Compute-facing correctness for `_swiglu_fwd_kernel`. -/
theorem swiglu_fwd_kernel_compute_correct
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (s : BlockState)
    (xs ys : Fin BLOCK_N → ℝ)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i) :
    ComputeCorrect.Realizes
      (kernel := swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
        ncols BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => s.pids 1 * BLOCK_N + i.val < ncols)
        (fun i => (OUT, swigluOffset s stride_out_row BLOCK_N i)))
      (expected := fun i => TiledActivation.swiglu (xs i) (ys i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [swiglu_fwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := swiglu_fwd_kernel_correct X Y OUT stride_x_row stride_y_row
    stride_out_row ncols BLOCK_N s s' xs ys h_x h_y hExec i
  simp [hActive] at hi
  exact hi

end VeriTile.Bench.TritonBenchG.SwigluFwd
