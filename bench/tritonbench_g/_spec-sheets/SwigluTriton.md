# Spec sheet — `bench/tritonbench_g/swiglu_triton/SwigluTriton.lean`

**Python source:** `bench/tritonbench_g/swiglu_triton/swiglu_triton.py`

## Public theorem: `swiglu_forward_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_swiglu_forward_kernel`: the DSL surface lowers
to the algorithm layer, and the masked store to `C` is compute-correct — every
active lane holds `TiledActivation.swiglu (as i) (bs i)`, out-of-bounds lanes are
preserved. -/
```
</details>

**Statement:**
```lean
theorem swiglu_forward_kernel_output_summary
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i) :
    (∃ alg, (swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (C, swigluOffset s stride i)))
      (expected := fun i => TiledActivation.swiglu (as i) (bs i))
```

**Assumptions / layout contracts:**
- `as bs : Fin BLOCK_SIZE → ℝ`
- `h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i`
- `h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `swigluOffset`, `swiglu_forward_kernel`

<details><summary><code>swigluOffset</code></summary>

```lean
def swigluOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val
```
</details>

<details><summary><code>swiglu_forward_kernel</code></summary>

```
/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters. -/
```
```lean
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
```
</details>

## Public theorem: `swiglu_backward_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_swiglu_backward_kernel`: the DSL surface
lowers to the algorithm layer, and the two masked stores are compute-correct —
every active lane writes `swigluBwdA` to `A` and `swigluBwdB` to `B`, with the two
output channels indexed by `Sum`; out-of-bounds lanes are preserved. Assumes the
two output regions are distinct (`A ≠ B`). -/
```
</details>

**Statement:**
```lean
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
    ComputeCorrect.Realizes_without_Rounding
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
        | .inr lane => TiledActivation.swigluBwdB (dcs lane) (as lane))
```

**Assumptions / layout contracts:**
- `dcs as bs : Fin BLOCK_SIZE → ℝ`
- `hAB : A ≠ B`
- `h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (swigluOffset s stride i) = dcs i`
- `h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i`
- `h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i`

**Closed-form spec defs (transitive):** `swigluOffset`, `swiglu_backward_kernel`

<details><summary><code>swigluOffset</code></summary>

```lean
def swigluOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val
```
</details>

<details><summary><code>swiglu_backward_kernel</code></summary>

```
/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_backward_kernel`.

Allowed mechanical Lean-syntax-only changes match `swiglu_forward_kernel`. -/
```
```lean
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
```
</details>

## Also present (pinned special-case summaries)
- `swiglu_forward_kernel_compute_correct`
- `swiglu_backward_kernel_compute_correct`
