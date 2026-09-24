# Spec sheet — `bench/tritonbench_g/swiglu_triton/SwigluTriton.lean`

**Python source:** `bench/tritonbench_g/swiglu_triton/swiglu_triton.py`

## Public theorem: `swiglu_forward_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline (forward)**: `_swiglu_forward_kernel` implements the SwiGLU
oracle `TiledActivation.swiglu` (i.e. `silu(a) · b`) lane-wise on its masked IO
signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
active input-row lanes hold `as`/`bs`, the translated pointer kernel
terminates, every active output-row lane `j` holds
`TiledActivation.swiglu (as j) (bs j)`, and every other memory cell is
unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification swiglu_forward_kernel_correctness
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    swigluFwdIO A B C stride n_cols BLOCK_SIZE
      ⊨ fun as bs i => TiledActivation.swiglu (as i) (bs i)
```

**Closed-form spec defs (transitive):** `swigluFwdIO`, `swiglu_forward_kernel`

<details><summary><code>swigluFwdIO</code></summary>

```
/-- `_swiglu_forward_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: gate `A`,
  value `B`, output `C`);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` (the row) reads and writes its row at
  `pid * stride` in all three buffers (the host-side one-program-per-row launch
  convention `x_ptr += program_id * stride`);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def swigluFwdIO (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE
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

## Public theorem: `swiglu_backward_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline (backward)**: `_swiglu_backward_kernel` implements the
SwiGLU backward oracles on its masked in-place IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input-row lanes hold
`dcs`/`as`/`bs`, the translated pointer kernel terminates, every active lane
of the gate buffer ends up holding `swigluBwdA` and of the value buffer
`swigluBwdB`, applied to the *originally loaded* windows; every other flat
cell is untouched. The side condition `A ≠ B` rules out aliasing between the
two output buffers (the second masked store would otherwise clobber the first
output). Proof: `MaskedKernelIO₃ₓ₂.Implements.intro` assembles the
region-model triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification swiglu_backward_kernel_correctness
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (hAB : A ≠ B) :
    swigluBwdIO DC A B stride n_cols BLOCK_SIZE
      ⊨ fun dcs as bs =>
        (fun i => TiledActivation.swigluBwdA (dcs i) (as i) (bs i),
         fun i => TiledActivation.swigluBwdB (dcs i) (as i))
```

**Assumptions / layout contracts:**
- `hAB : A ≠ B`

**Closed-form spec defs (transitive):** `swigluBwdIO`, `swiglu_backward_kernel`

<details><summary><code>swigluBwdIO</code></summary>

```
/-- `_swiglu_backward_kernel`'s masked in-place **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `bufs` — the allocation list: three buffers, each exactly once;
* `in1`/`in2`/`in3` — upstream gradient `DC`, gate `A`, value `B` (the wiring);
* `out1 = in2`, `out2 = in3` — the **in-place** roles: the kernel overwrites
  the gate and value buffers it read with `da` and `db`;
* `read1..3`/`write1..2` — every window is the same row `pid * stride` (the
  host-side one-program-per-row launch convention
  `x_ptr += program_id * stride`);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def swigluBwdIO (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE
  bufs := [DC, A, B]  -- A and B are updated in place
  in1 := DC
  in2 := A
  in3 := B
  out1 := A    -- = in2: `da` overwrites the gate input in place
  out2 := B    -- = in3: `db` overwrites the value input in place
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  read3 := fun pid => pid * stride
  write1 := fun pid => pid * stride
  write2 := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols
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
