import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SwigluBackward

open VeriTile.Triton

/-- Faithful transcription of `swiglu_backward.py`'s `_swiglu_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` / `RECOMPUTE_OUTPUT: tl.constexpr` -> Lean
  parameters.
- Python pointer mutation `X += ...` / `Y += ...` / ... -> explicit base
  pointer registers.
- `OUT_base` is computed unconditionally; the store using it remains guarded by
  `RECOMPUTE_OUTPUT`, so this does not change memory behavior. -/
def swiglu_bwd_kernel
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X_base = X + row * $(stride_x_row)
  Y_base = Y + row * $(stride_y_row)
  DOUT_base = DOUT + row * $(stride_dout_row)
  OUT_base = OUT + row * $(stride_out_row)
  DX_base = DX + row * $(stride_dx_row)
  DY_base = DY + row * $(stride_dy_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X_base + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y_base + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  dout = tl.load(DOUT_base + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  x_sigmoid = tl.sigmoid(x)
  dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
  dy = x * x_sigmoid * dout
  tl.store(DX_base + cols, dx, mask=cols < $(ncols))
  tl.store(DY_base + cols, dy, mask=cols < $(ncols))
  if RECOMPUTE_OUTPUT {
    out = x * x_sigmoid * y
    tl.store(OUT_base + cols, out, mask=cols < $(ncols))
  }
}

noncomputable def swigluBwdDXSpec (x y dout : ℝ) : ℝ :=
  let sig := Real.sigmoid x
  sig * (1 + x * (1 - sig)) * y * dout

noncomputable def swigluBwdDYSpec (x dout : ℝ) : ℝ :=
  x * Real.sigmoid x * dout

noncomputable def swigluBwdOutSpec (x y : ℝ) : ℝ :=
  x * Real.sigmoid x * y

def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val

/-- Algorithm-layer correctness for `_swiglu_bwd_kernel`.

The kernel stores `DX`, then `DY`, then optionally `OUT`. We assume these output
regions are distinct so the later stores cannot overwrite earlier channels. -/
theorem swiglu_bwd_kernel_correct
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (s s' : BlockState)
    (xs ys douts : Fin BLOCK_N → ℝ)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i)
    (h_dout : ∀ i : Fin BLOCK_N, s.readMem DOUT (swigluOffset s stride_dout_row BLOCK_N i) = douts i)
    (hExec : exec (swiglu_bwd_kernel X Y DOUT OUT DX DY
          stride_x_row stride_y_row stride_dout_row stride_out_row
          stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT) s = some s') :
    (∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_dx_row BLOCK_N i
      s'.readMem DX outAddr =
        if s.pids 1 * BLOCK_N + i.val < ncols then
          swigluBwdDXSpec (xs i) (ys i) (douts i)
        else s.readMem DX outAddr) ∧
    (∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_dy_row BLOCK_N i
      s'.readMem DY outAddr =
        if s.pids 1 * BLOCK_N + i.val < ncols then
          swigluBwdDYSpec (xs i) (douts i)
        else s.readMem DY outAddr) ∧
    (∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_out_row BLOCK_N i
      s'.readMem OUT outAddr =
        if RECOMPUTE_OUTPUT then
          if s.pids 1 * BLOCK_N + i.val < ncols then
            swigluBwdOutSpec (xs i) (ys i)
          else s.readMem OUT outAddr
        else s.readMem OUT outAddr) := by
  have h_inj_dx : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_dx_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_dx_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  have h_inj_dy : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_dy_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_dy_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  have h_inj_out : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  cases RECOMPUTE_OUTPUT <;>
    simp [exec, swiglu_bwd_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot] at hExec
  · subst s'
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DY) (R := DX) (h_ne := hDXDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dx (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hy := h_y i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hy hdout
        simp [hi, swigluBwdDXSpec, hx, hy, hdout, FloatDType.toWithBot]
      · simp [hi]
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dy (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hdout
        simp [hi, swigluBwdDYSpec, hx, hdout, FloatDType.toWithBot]
      · rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DX) (R := DY) (h_ne := Ne.symm hDXDY)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        simp [hi]
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DY) (R := OUT) (h_ne := hOUTDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DX) (R := OUT) (h_ne := hOUTDX)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      simp
  · subst s'
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := OUT) (R := DX) (h_ne := Ne.symm hOUTDX)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      simp
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DY) (R := DX) (h_ne := hDXDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dx (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hy := h_y i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hy hdout
        simp [hi, swigluBwdDXSpec, hx, hy, hdout, FloatDType.toWithBot]
      · simp [hi]
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := OUT) (R := DY) (h_ne := Ne.symm hOUTDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      simp
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dy (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hdout
        simp [hi, swigluBwdDYSpec, hx, hdout, FloatDType.toWithBot]
      · rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DX) (R := DY) (h_ne := Ne.symm hDXDY)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        simp [hi]
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_out (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hy := h_y i
        simp [swigluOffset, Nat.add_assoc] at hx hy
        simp [hi, swigluBwdOutSpec, hx, hy, FloatDType.toWithBot]
      · simp [hi]
        rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DY) (R := OUT) (h_ne := hOUTDY)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DX) (R := OUT) (h_ne := hOUTDX)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        simp

/-- Compute-facing correctness for `_swiglu_bwd_kernel`. -/
theorem swiglu_bwd_kernel_compute_correct
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (s : BlockState)
    (xs ys douts : Fin BLOCK_N → ℝ)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i)
    (h_dout : ∀ i : Fin BLOCK_N, s.readMem DOUT (swigluOffset s stride_dout_row BLOCK_N i) = douts i) :
    ComputeCorrect.Realizes
      (kernel := swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT)
      (initialState := s)
      (write := fun i : Sum (Sum (Fin BLOCK_N) (Fin BLOCK_N)) (Fin BLOCK_N) =>
        match i with
        | .inl (.inl lane) =>
            if s.pids 1 * BLOCK_N + lane.val < ncols then
              some (DX, swigluOffset s stride_dx_row BLOCK_N lane)
            else none
        | .inl (.inr lane) =>
            if s.pids 1 * BLOCK_N + lane.val < ncols then
              some (DY, swigluOffset s stride_dy_row BLOCK_N lane)
            else none
        | .inr lane =>
            if RECOMPUTE_OUTPUT then
              if s.pids 1 * BLOCK_N + lane.val < ncols then
                some (OUT, swigluOffset s stride_out_row BLOCK_N lane)
              else none
            else none)
      (expected := fun i =>
        match i with
        | .inl (.inl lane) => swigluBwdDXSpec (xs lane) (ys lane) (douts lane)
        | .inl (.inr lane) => swigluBwdDYSpec (xs lane) (douts lane)
        | .inr lane => swigluBwdOutSpec (xs lane) (ys lane)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [swiglu_bwd_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := swiglu_bwd_kernel_correct X Y DOUT OUT DX DY
    stride_x_row stride_y_row stride_dout_row stride_out_row
    stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
    s s' xs ys douts hDXDY hOUTDX hOUTDY h_x h_y h_dout hExec
  cases i with
  | inl lr =>
      cases lr with
      | inl lane =>
          by_cases hActive : s.pids 1 * BLOCK_N + lane.val < ncols
          · have hi := h.1 lane
            simp [hActive] at hi ⊢
            exact hi
          · simp [hActive]
      | inr lane =>
          by_cases hActive : s.pids 1 * BLOCK_N + lane.val < ncols
          · have hi := h.2.1 lane
            simp [hActive] at hi ⊢
            exact hi
          · simp [hActive]
  | inr lane =>
      by_cases hRecompute : RECOMPUTE_OUTPUT
      · by_cases hActive : s.pids 1 * BLOCK_N + lane.val < ncols
        · have hi := h.2.2 lane
          simp [hRecompute, hActive] at hi ⊢
          exact hi
        · simp [hRecompute, hActive]
      · simp [hRecompute]

end VeriTile.Bench.TritonBenchG.SwigluBackward
