import VeriTile.Triton

/-!
# `geglu_tanh_triton` — strict per-kernel correctness

Two GeGLU kernels using the tanh approximation of GELU. `_geglu_tanh_forward_kernel`:
program `row` loads gate `a` and value `b` and stores
`0.5·a·(1 + tanh(√(2/π)·(a + 0.044715·a³)))·b` to `c`.
`_geglu_tanh_backward_kernel`: recomputes the activation from upstream gradient
`dc` and writes the two input gradients in place — `da` into `a`, `db` into `b`.
Both are masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`_geglu_tanh_*_kernel[(n_rows,)](...)`,
the 1-D grid, the `calculate_settings` choice of `BLOCK_SIZE`/`num_warps`, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `pid` is universally quantified,
each per-program statement covers every program of the grid.

## Proof architecture

```
geglu_tanh_forward_kernel_output_summary        ← TOP THEOREM (forward)
  ├─ (toAlgorithm? = Except.ok _)               surface lowers to the algorithm layer
  └─ geglu_tanh_forward_kernel_compute_correct  ← ComputeCorrect over the masked store
       └─ geglu_tanh_forward_kernel_correct     ← algorithm-layer readback per lane

geglu_tanh_backward_kernel_output_summary       ← TOP THEOREM (backward)
  ├─ (toAlgorithm? = Except.ok _)               surface lowers to the algorithm layer
  └─ geglu_tanh_backward_kernel_compute_correct ← ComputeCorrect over two masked stores
       └─ geglu_tanh_backward_kernel_correct    ← algorithm-layer readback per lane, per channel
```

## Modeling boundary

The specs are **oracle wrappers** over `VeriTile.Triton.Math.Activation`: the
tanh-GeGLU math (`TiledActivation.geluTanhFwd`, `geluTanhBwdA/B`, built on
`TiledActivation.geluTanhCore` / `geluTanhArg`) lives once in `Math.Activation`
and is reused here, so this file only checks that the kernels realize those
oracles lane-wise. The `from triton.language.extra.libdevice import tanh` is
represented by the DSL surface function `tanh`. Arithmetic is over `ℝ` (not
bit-accurate IEEE float); the `tl.program_id(0).to(tl.int64)` and `.to(tl.float32)`
casts reduce to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`); `calculate_settings` is not modeled. The float literals
(`0.7978845608028654`, `0.044715`) are interpreted as exact reals, not their
rounded float values. The backward kernel writes two channels (`da`→`a`,
`db`→`b`) and overwrites its own inputs in place; the readback theorem assumes
the two output regions are distinct (`A ≠ B`) so the second store cannot clobber
the first channel, and is stated as a `Sum`-indexed multi-output `ComputeCorrect`
(not a single `writeIf`). Inputs are presented as `s.readMem`-resolved tiles.
-/

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
  simp [exec, geglu_tanh_forward_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [gegluTanhOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
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
    ComputeCorrect.Realizes_without_Rounding
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

/-- Per-kernel output summary for `_geglu_tanh_forward_kernel`: the DSL surface
lowers to the algorithm layer, and the masked store to `C` is compute-correct —
every active lane holds `TiledActivation.geluTanhFwd (as i) (bs i)`, out-of-bounds
lanes are preserved. -/
theorem geglu_tanh_forward_kernel_output_summary
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i) :
    (∃ alg, (geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (C, gegluTanhOffset s stride i)))
      (expected := fun i => TiledActivation.geluTanhFwd (as i) (bs i)) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact geglu_tanh_forward_kernel_compute_correct A B C stride n_cols BLOCK_SIZE
    s as bs h_a h_b

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
  simp [exec, geglu_tanh_backward_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  constructor
  · intro i
    simp only [gegluTanhOffset]
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
      simp [gegluTanhOffset] at hdc ha hb
      norm_num [hi, TiledActivation.geluTanhBwdA, TiledActivation.geluTanhArg,
            hdc, ha, hb]
    · simp [hi]
  · intro i
    simp only [gegluTanhOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
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
    ComputeCorrect.Realizes_without_Rounding
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

/-- Per-kernel output summary for `_geglu_tanh_backward_kernel`: the DSL surface
lowers to the algorithm layer, and the two masked stores are compute-correct —
every active lane writes `geluTanhBwdA` to `A` and `geluTanhBwdB` to `B`, with the
two output channels indexed by `Sum`; out-of-bounds lanes are preserved. Assumes
the two output regions are distinct (`A ≠ B`). -/
theorem geglu_tanh_backward_kernel_output_summary
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (gegluTanhOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i) :
    (∃ alg, (geglu_tanh_backward_kernel DC A B stride n_cols BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
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
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact geglu_tanh_backward_kernel_compute_correct DC A B stride n_cols BLOCK_SIZE
    s dcs as bs hAB h_dc h_a h_b

end VeriTile.Bench.TritonBenchG.GegluTanhTriton
