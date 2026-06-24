# Spec sheet — `bench/tritonbench_g/geglu_tanh_triton/GegluTanhTriton.lean`

**Python source:** `bench/tritonbench_g/geglu_tanh_triton/geglu_tanh_triton.py`

## Public theorem: `geglu_tanh_forward_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_geglu_tanh_forward_kernel`: the DSL surface
lowers to the algorithm layer, and the masked store to `C` is compute-correct —
every active lane holds `TiledActivation.geluTanhFwd (as i) (bs i)`, out-of-bounds
lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem geglu_tanh_forward_kernel_output_summary
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i) :
    (∃ alg, (geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (C, gegluTanhOffset s stride i)))
      (expected := fun i => TiledActivation.geluTanhFwd (as i) (bs i))
```

**Assumptions / layout contracts:**
- `as bs : Fin BLOCK_SIZE → ℝ`
- `h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i`
- `h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i`
- `kernel : = geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`
- `expected : = fun i => TiledActivation.geluTanhFwd (as i) (bs i)`

**Closed-form spec defs (transitive):** `gegluTanhOffset`, `geglu_tanh_forward_kernel`

<details><summary><code>gegluTanhOffset</code></summary>

```lean
def gegluTanhOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val
```
</details>

<details><summary><code>geglu_tanh_forward_kernel</code></summary>

```
/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters.
- Python `from triton.language.extra.libdevice import tanh` is represented by
  the DSL surface function `tanh`. -/
```
```lean
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
```
</details>

## Public theorem: `geglu_tanh_backward_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_geglu_tanh_backward_kernel`: the DSL surface
lowers to the algorithm layer, and the two masked stores are compute-correct —
every active lane writes `geluTanhBwdA` to `A` and `geluTanhBwdB` to `B`, with the
two output channels indexed by `Sum`; out-of-bounds lanes are preserved. Assumes
the two output regions are distinct (`A ≠ B`). -/
```
</details>

**Statement:**
```lean
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
        | .inr lane => TiledActivation.geluTanhBwdB (dcs lane) (as lane))
```

**Assumptions / layout contracts:**
- `dcs as bs : Fin BLOCK_SIZE → ℝ`
- `hAB : A ≠ B`
- `h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (gegluTanhOffset s stride i) = dcs i`
- `h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i`
- `h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i`
- `kernel : = geglu_tanh_backward_kernel DC A B stride n_cols BLOCK_SIZE`
- `initialState : = s`
- `write : = fun i : Sum (Fin BLOCK_SIZE) (Fin BLOCK_SIZE) =>
        match i with
        | .inl lane =>
            if lane.val < n_cols then some (A, gegluTanhOffset s stride lane) else none
        | .inr lane =>
            if lane.val < n_cols then some (B, gegluTanhOffset s stride lane) else none`
- `expected : = fun i =>
        match i with
        | .inl lane => TiledActivation.geluTanhBwdA (dcs lane) (as lane) (bs lane)
        | .inr lane => TiledActivation.geluTanhBwdB (dcs lane) (as lane)`

**Closed-form spec defs (transitive):** `gegluTanhOffset`, `geglu_tanh_backward_kernel`

<details><summary><code>gegluTanhOffset</code></summary>

```lean
def gegluTanhOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val
```
</details>

<details><summary><code>geglu_tanh_backward_kernel</code></summary>

```
/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_backward_kernel`.

The Python kernel overwrites `a` and `b` with `da` and `db`; the Lean port keeps
the same region arguments. -/
```
```lean
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
```
</details>

## Also present (pinned special-case summaries)
- `geglu_tanh_forward_kernel_compute_correct`
- `geglu_tanh_backward_kernel_compute_correct`
