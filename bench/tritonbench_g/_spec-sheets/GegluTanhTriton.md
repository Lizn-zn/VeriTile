# Spec sheet — `bench/tritonbench_g/geglu_tanh_triton/GegluTanhTriton.lean`

**Python source:** `bench/tritonbench_g/geglu_tanh_triton/geglu_tanh_triton.py`

## Public theorem: `geglu_tanh_forward_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The forward headline**: `_geglu_tanh_forward_kernel` implements the
tanh-GeGLU forward oracle on its masked IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `as`/`bs`,
the translated pointer kernel terminates, every active output-row lane `j`
holds `TiledActivation.geluTanhFwd (as j) (bs j)`, and every other memory cell
is unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification geglu_tanh_forward_kernel_correctness
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    gegluTanhFwdIO A B C stride n_cols BLOCK_SIZE
      ⊨ fun as bs i => TiledActivation.geluTanhFwd (as i) (bs i)
```

**Closed-form spec defs (transitive):** `gegluTanhFwdIO`, `geglu_tanh_forward_kernel`

<details><summary><code>gegluTanhFwdIO</code></summary>

```
/-- `_geglu_tanh_forward_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the forward `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (gate `a`, value `b`,
  output `c`);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` reads and writes its row at
  `pid * stride` in all three buffers (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes carry no
  obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def gegluTanhFwdIO (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE
  in1 := A
  in2 := B
  out := C
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols
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

## Public theorem: `geglu_tanh_backward_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The backward headline**: `_geglu_tanh_backward_kernel` implements the
tanh-GeGLU backward oracles on its masked in-place IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input-row lanes hold
`dcs`/`as`/`bs`, the translated pointer kernel terminates, every active lane
of the gate buffer ends up holding `geluTanhBwdA` and of the value buffer
`geluTanhBwdB` — applied to the *originally loaded* windows (the standard
before/after reading of an in-place Hoare triple) — and every other memory
cell is unchanged. Assumes the two output regions are distinct (`A ≠ B`) so
the second store cannot clobber the first channel. Proof:
`MaskedKernelIO₃ₓ₂.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification geglu_tanh_backward_kernel_correctness
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (hAB : A ≠ B) :
    gegluTanhBwdIO DC A B stride n_cols BLOCK_SIZE ⊨ fun dcs as bs =>
      (fun i => TiledActivation.geluTanhBwdA (dcs i) (as i) (bs i),
       fun i => TiledActivation.geluTanhBwdB (dcs i) (as i))
```

**Assumptions / layout contracts:**
- `hAB : A ≠ B`

**Closed-form spec defs (transitive):** `gegluTanhBwdIO`, `geglu_tanh_backward_kernel`

<details><summary><code>gegluTanhBwdIO</code></summary>

```
/-- `_geglu_tanh_backward_kernel`'s masked in-place **IO signature** — the
whole kernel-specific audit surface of the backward `⊨` headline:

* `bufs` — the allocation list: three buffers, each exactly once;
* `in1`/`in2`/`in3` — upstream gradient `dc`, gate `a`, value `b` (the
  wiring);
* `out1 = in2`, `out2 = in3` — the **in-place** roles: the kernel rewrites
  the gate and value buffers it read (`da`→`a`, `db`→`b`);
* `read1..3`/`write1..2` — every window is the same row `pid * stride` (the
  host-side one-program-per-row launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**.
  Inactive lanes carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
```
```lean
def gegluTanhBwdIO (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := geglu_tanh_backward_kernel DC A B stride n_cols BLOCK_SIZE
  bufs := [DC, A, B]  -- a and b are updated in place
  in1 := DC
  in2 := A
  in3 := B
  out1 := A   -- = in2: in-place `da` into `a`
  out2 := B   -- = in3: in-place `db` into `b`
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  read3 := fun pid => pid * stride
  write1 := fun pid => pid * stride
  write2 := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols
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
