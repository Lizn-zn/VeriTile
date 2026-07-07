import VeriTile.Triton

/-!
# `swiglu_triton` — strict per-kernel correctness

Two SwiGLU kernels. `_swiglu_forward_kernel`: program `row` loads gate `a` and
value `b` for its row and stores `silu(a) · b` to `c`. `_swiglu_backward_kernel`:
recomputes `silu(a)` from the loaded gate `a`, then combines it with the upstream
gradient `dc` and writes the two input gradients in place — `da` into `a_ptr`,
`db` into `b_ptr`. Both are masked by
`col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`_swiglu_*_kernel[(n_rows,)](...)`, the
1-D grid, the `calculate_settings` choice of `BLOCK_SIZE`/`num_warps`, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `pid` is universally quantified,
each per-program statement covers every program of the grid.

## Proof architecture

```
swiglu_forward_kernel_output_summary          ← TOP THEOREM (forward)
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ swiglu_forward_kernel_compute_correct    ← ComputeCorrect over the masked store
       └─ swiglu_forward_kernel_correct       ← algorithm-layer readback per lane

swiglu_backward_kernel_output_summary         ← TOP THEOREM (backward)
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ swiglu_backward_kernel_compute_correct   ← ComputeCorrect over two masked stores
       └─ swiglu_backward_kernel_correct      ← algorithm-layer readback per lane, per channel
```

## Modeling boundary

The specs are **oracle wrappers** over `VeriTile.Triton.Math.Activation`: the
SwiGLU math (`TiledActivation.swiglu`, `TiledActivation.swigluBwdA/B`, built on
`TiledActivation.silu`) lives once in `Math.Activation` and is reused here, so
this file only checks that the kernels realize those oracles lane-wise.
Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `tl.program_id(0)
.to(tl.int64)` cast and `.to(tl.float32)` load casts reduce to the identity at
the algorithm layer (post-erasure all dtypes unify to `ℝ`); `calculate_settings`
is not modeled. The backward kernel writes two channels (`da`→`a_ptr`,
`db`→`b_ptr`) and overwrites its own inputs in place; the readback theorem assumes
the two output regions are distinct (`A ≠ B`) so the second store cannot clobber
the first channel, and is stated as a `Sum`-indexed multi-output `ComputeCorrect`
(not a single `writeIf`). Inputs are presented as `s.readMem`-resolved tiles.
-/

namespace VeriTile.Bench.TritonBenchG.SwigluTriton

open VeriTile.Triton

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters. -/
def swiglu_forward_kernel
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  c_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
}

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_backward_kernel`.

Allowed mechanical Lean-syntax-only changes match `swiglu_forward_kernel`. -/
def swiglu_backward_kernel
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc_ptr += program_id * $(stride)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(a_ptr + col_offsets, da_row, mask=mask)
  tl.store(b_ptr + col_offsets, db_row, mask=mask)
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
  simp [exec, swiglu_forward_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [swigluOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
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
  · simp [swiglu_forward_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := swiglu_forward_kernel_correct A B C stride n_cols BLOCK_SIZE
    s s' as bs h_a h_b hExec i
  simp [hActive] at hi
  exact hi

/-- Per-kernel output summary for `_swiglu_forward_kernel`: the DSL surface lowers
to the algorithm layer, and the masked store to `C` is compute-correct — every
active lane holds `TiledActivation.swiglu (as i) (bs i)`, out-of-bounds lanes are
preserved. -/
theorem swiglu_forward_kernel_output_summary
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i) :
    (∃ alg, (swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (C, swigluOffset s stride i)))
      (expected := fun i => TiledActivation.swiglu (as i) (bs i)) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact swiglu_forward_kernel_compute_correct A B C stride n_cols BLOCK_SIZE
    s as bs h_a h_b

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
  simp [exec, swiglu_backward_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  constructor
  · intro i
    simp only [swigluOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
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
      simp [swigluOffset] at hdc ha hb
      simp [hi, TiledActivation.swigluBwdA, TiledActivation.silu, hdc, ha, hb]
    · simp [hi]
  · intro i
    simp only [swigluOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
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
  · simp [swiglu_backward_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
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

/-- Per-kernel output summary for `_swiglu_backward_kernel`: the DSL surface
lowers to the algorithm layer, and the two masked stores are compute-correct —
every active lane writes `swigluBwdA` to `A` and `swigluBwdB` to `B`, with the two
output channels indexed by `Sum`; out-of-bounds lanes are preserved. Assumes the
two output regions are distinct (`A ≠ B`). -/
theorem swiglu_backward_kernel_output_summary
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (swigluOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i) :
    (∃ alg, (swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
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
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact swiglu_backward_kernel_compute_correct DC A B stride n_cols BLOCK_SIZE
    s dcs as bs hAB h_dc h_a h_b

end VeriTile.Bench.TritonBenchG.SwigluTriton
