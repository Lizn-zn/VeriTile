import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Activation

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton

open VeriTile.Triton

/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters.
- Python `from triton.language.extra.libdevice import tanh` is represented by
  the DSL surface function `tanh`. -/
def geglu_tanh_forward_kernel
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a += program_id * $(stride)
  b += program_id * $(stride)
  c += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  tl.store(c + col_offsets, c_row, mask=mask)
}

/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_backward_kernel`.

The Python kernel overwrites `a` and `b` with `da` and `db`; the Lean port keeps
the same region arguments. -/
def geglu_tanh_backward_kernel
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc += program_id * $(stride)
  a += program_id * $(stride)
  b += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  tl.store(a + col_offsets, da_row, mask=mask)
  tl.store(b + col_offsets, db_row, mask=mask)
}

def gegluTanhOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val

/-- Algorithm-layer correctness for `_geglu_tanh_forward_kernel`. -/
theorem geglu_tanh_forward_kernel_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i)
    (hExec : exec (geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := gegluTanhOffset s stride i
      s'.readMem C outAddr =
        if i.val < n_cols then
          TiledActivation.geluTanhFwd (as i) (bs i)
        else s.readMem C outAddr := by
  intro i
  simp [exec, geglu_tanh_forward_kernel, stepStmts, stepStmt, evalOp,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [gegluTanhOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · have ha := h_a i
    have hb := h_b i
    simp [gegluTanhOffset] at ha hb
    simp [hi, TiledActivation.geluTanhFwd, TiledActivation.geluTanhCore,
          TiledActivation.geluTanhArg, ha, hb]
  · simp [hi]

/-- Compute-facing correctness for `_geglu_tanh_forward_kernel`. -/
theorem geglu_tanh_forward_kernel_compute_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i) :
    ComputeCorrect.Realizes
      (kernel := geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (C, gegluTanhOffset s stride i)))
      (expected := fun i => TiledActivation.geluTanhFwd (as i) (bs i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [geglu_tanh_forward_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := geglu_tanh_forward_kernel_correct A B C stride n_cols BLOCK_SIZE
    s s' as bs h_a h_b hExec i
  simp [hActive] at hi
  exact hi

/-- Algorithm-layer correctness for `_geglu_tanh_backward_kernel`.

The kernel stores `A` first and `B` second. We assume `A ≠ B` so the second
store cannot overwrite the first output channel. -/
theorem geglu_tanh_backward_kernel_correct
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (gegluTanhOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i)
    (hExec : exec (geglu_tanh_backward_kernel DC A B stride n_cols BLOCK_SIZE) s = some s') :
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := gegluTanhOffset s stride i
      s'.readMem A outAddr =
        if i.val < n_cols then
          TiledActivation.geluTanhBwdA (dcs i) (as i) (bs i)
        else s.readMem A outAddr) ∧
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := gegluTanhOffset s stride i
      s'.readMem B outAddr =
        if i.val < n_cols then
          TiledActivation.geluTanhBwdB (dcs i) (as i)
        else s.readMem B outAddr) := by
  simp [exec, geglu_tanh_backward_kernel, stepStmts, stepStmt, evalOp,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  constructor
  · intro i
    simp only [gegluTanhOffset]
    rw [BlockState.scatter_prop_masked_preserves_other_region
      (region := B) (R := A) (h_ne := hAB)
      (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
      (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      have hb := h_b i
      simp [gegluTanhOffset] at hdc ha hb
      norm_num [hi, TiledActivation.geluTanhBwdA, TiledActivation.geluTanhArg,
            hdc, ha, hb]
    · simp [hi]
  · intro i
    simp only [gegluTanhOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      simp [gegluTanhOffset] at hdc ha
      simp [hi, TiledActivation.geluTanhBwdB, TiledActivation.geluTanhCore,
            TiledActivation.geluTanhArg, hdc, ha]
    · rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := A) (R := B) (h_ne := Ne.symm hAB)
        (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
        (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
      simp [hi]

/-- Compute-facing correctness for `_geglu_tanh_backward_kernel`. -/
theorem geglu_tanh_backward_kernel_compute_correct
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (gegluTanhOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i) :
    ComputeCorrect.Realizes
      (kernel := geglu_tanh_backward_kernel DC A B stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Sum (Fin BLOCK_SIZE) (Fin BLOCK_SIZE) =>
        match i with
        | .inl lane =>
            if lane.val < n_cols then some (A, gegluTanhOffset s stride lane) else none
        | .inr lane =>
            if lane.val < n_cols then some (B, gegluTanhOffset s stride lane) else none)
      (expected := fun i =>
        match i with
        | .inl lane => TiledActivation.geluTanhBwdA (dcs lane) (as lane) (bs lane)
        | .inr lane => TiledActivation.geluTanhBwdB (dcs lane) (as lane)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [geglu_tanh_backward_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := geglu_tanh_backward_kernel_correct DC A B stride n_cols BLOCK_SIZE
    s s' dcs as bs hAB h_dc h_a h_b hExec
  cases i with
  | inl lane =>
      by_cases hActive : lane.val < n_cols
      · have hi := h.1 lane
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]
  | inr lane =>
      by_cases hActive : lane.val < n_cols
      · have hi := h.2 lane
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]

end VeriTile.Bench.TritonBenchG.GegluTanhTriton
