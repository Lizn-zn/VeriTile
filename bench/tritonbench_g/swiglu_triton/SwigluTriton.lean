import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Activation

namespace VeriTile.Bench.TritonBenchG.SwigluTriton

open VeriTile.Triton

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters.
- Python pointer mutation `a += ...` / `b += ...` / `c += ...` -> explicit
  base pointer registers.
- The helper `silu(x)` is inlined as `x * tl.sigmoid(x)`. -/
def swiglu_forward_kernel
    (A B C : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  A_base = A + program_id * $(stride)
  B_base = B + program_id * $(stride)
  C_base = C + program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(A_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
  b_row = tl.load(B_base + col_offsets, mask=mask, other=0.0)
  c_row = a_row * tl.sigmoid(a_row) * b_row
  tl.store(C_base + col_offsets, c_row, mask=mask)
}

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_backward_kernel`.

Allowed mechanical Lean-syntax-only changes match `swiglu_forward_kernel`: base
pointer mutation is made explicit. -/
def swiglu_backward_kernel
    (DC A B : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  DC_base = DC + program_id * $(stride)
  A_base = A + program_id * $(stride)
  B_base = B + program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(DC_base + col_offsets, mask=mask, other=0.0)
  a_row = tl.load(A_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
  b_row = tl.load(B_base + col_offsets, mask=mask, other=0.0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(A_base + col_offsets, da_row, mask=mask)
  tl.store(B_base + col_offsets, db_row, mask=mask)
}

def swigluOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val

/-- Algorithm-layer correctness for `_swiglu_forward_kernel`. -/
theorem swiglu_forward_kernel_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i)
    (hExec : exec (swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem C outAddr =
        if i.val < n_cols then
          TiledActivation.swiglu (as i) (bs i)
        else s.readMem C outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * stride + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, swiglu_forward_kernel, stepStmts, stepStmt, evalOp,
        tile_elementwise] at hExec
  subst s'
  simp only [swigluOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · have ha := h_a i
    have hb := h_b i
    simp [swigluOffset] at ha hb
    simp [hi, TiledActivation.swiglu, TiledActivation.silu, ha, hb]
  · simp [hi]

/-- Compute-facing correctness for `_swiglu_forward_kernel`. -/
theorem swiglu_forward_kernel_compute_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i) :
    ComputeCorrect.Realizes
      (kernel := swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (C, swigluOffset s stride i)))
      (expected := fun i => TiledActivation.swiglu (as i) (bs i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [swiglu_forward_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := swiglu_forward_kernel_correct A B C stride n_cols BLOCK_SIZE
    s s' as bs h_a h_b hExec i
  simp [hActive] at hi
  exact hi

/-- Algorithm-layer correctness for `_swiglu_backward_kernel`.

The two stores write `A` first and `B` second. We assume `A ≠ B` so the second
store cannot overwrite the first output channel. -/
theorem swiglu_backward_kernel_correct
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (swigluOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i)
    (hExec : exec (swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE) s = some s') :
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem A outAddr =
        if i.val < n_cols then
          TiledActivation.swigluBwdA (dcs i) (as i) (bs i)
        else s.readMem A outAddr) ∧
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem B outAddr =
        if i.val < n_cols then
          TiledActivation.swigluBwdB (dcs i) (as i)
        else s.readMem B outAddr) := by
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * stride + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, swiglu_backward_kernel, stepStmts, stepStmt, evalOp,
        tile_elementwise] at hExec
  subst s'
  constructor
  · intro i
    simp only [swigluOffset]
    rw [BlockState.scatter_prop_masked_preserves_other_region
      (region := B) (R := A) (h_ne := hAB)
      (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
      (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      have hb := h_b i
      simp [swigluOffset] at hdc ha hb
      simp [hi, TiledActivation.swigluBwdA, TiledActivation.silu, hdc, ha, hb]
    · simp [hi]
  · intro i
    simp only [swigluOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      simp [swigluOffset] at hdc ha
      simp [hi, TiledActivation.swigluBwdB, TiledActivation.silu, hdc, ha]
    · rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := A) (R := B) (h_ne := Ne.symm hAB)
        (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
        (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
      simp [hi]

/-- Compute-facing correctness for `_swiglu_backward_kernel`. -/
theorem swiglu_backward_kernel_compute_correct
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (swigluOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i) :
    ComputeCorrect.Realizes
      (kernel := swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Sum (Fin BLOCK_SIZE) (Fin BLOCK_SIZE) =>
        match i with
        | .inl lane =>
            if lane.val < n_cols then some (A, swigluOffset s stride lane) else none
        | .inr lane =>
            if lane.val < n_cols then some (B, swigluOffset s stride lane) else none)
      (expected := fun i =>
        match i with
        | .inl lane => TiledActivation.swigluBwdA (dcs lane) (as lane) (bs lane)
        | .inr lane => TiledActivation.swigluBwdB (dcs lane) (as lane)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [swiglu_backward_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := swiglu_backward_kernel_correct DC A B stride n_cols BLOCK_SIZE
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

end VeriTile.Bench.TritonBenchG.SwigluTriton
