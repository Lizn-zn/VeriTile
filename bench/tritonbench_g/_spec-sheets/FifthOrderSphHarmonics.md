# Spec sheet — `bench/tritonbench_g/fifth_order_sph_harmonics/FifthOrderSphHarmonics.lean`

**Python source:** `bench/tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics.py`

## Public theorem: `fifth_order_fwd_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `fifth_order_fwd` implements the eleven fifth-order real
spherical harmonics on its grouped three-input / eleven-output IO signature —
for every disjoint flat placement of the coordinate and output buffers, every
program id whose active lanes are in bounds, and every launch state whose
read-active lanes hold the coordinates `xs`, the translated pointer kernel
terminates, every write-active lane `j` of every channel `o` holds
`sphY o (x j) (y j) (z j)`, and every other memory cell is unchanged. One
statement covers all eleven strided stores, and the frame is a single
channel-quantified leg.

Side conditions: `hStride : 10 < output_stride` (the eleven channel columns fit
inside one output row — this also makes the per-lane store offsets injective),
and `hCover`, the host-layout coupling that a lane whose `Y00` store is in
range has its three coordinates in range (both extents count the same rows in
the `calculate_lastdim_num_blocks` launch). Proof:
`GroupedMasked2DKernelIO.Implements.intro` assembles the region-model grouped
triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification fifth_order_fwd_correctness
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (hStride : 10 < output_stride)
    (hCover : ∀ (pid₀ : Nat) (j : Fin block_size),
      j.val * output_stride + block_size * output_stride * pid₀ + col_offset
        < output_numel →
      j.val * 3 + block_size * 3 * pid₀ + 2 < coord_numel) :
    fifthOrderFwdIO coord_ptr output_ptr block_size coord_numel output_numel
        col_offset output_stride ⊨
      fun _ _ xs o j =>
        sphY o (xs ⟨0, by show 0 < 3; decide⟩ j) (xs ⟨1, by show 1 < 3; decide⟩ j)
          (xs ⟨2, by show 2 < 3; decide⟩ j)
```

**Assumptions / layout contracts:**
- `hStride : 10 < output_stride`
- `hCover : ∀ (pid₀ : Nat) (j : Fin block_size),
      j.val * output_stride + block_size * output_stride * pid₀ + col_offset
        < output_numel →
      j.val * 3 + block_size * 3 * pid₀ + 2 < coord_numel`

**Closed-form spec defs (transitive):** `fifthOrderFwdIO`, `sphY`, `fifth_order_fwd_surface`, `sphInWin`, `sphOutWin`

<details><summary><code>fifthOrderFwdIO</code></summary>

```
/-- `fifth_order_fwd`'s **grouped IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `nIn = 3` input channels, all reading the one coordinate buffer `coord_ptr`
  (`inp` is constant); `nOut = 11` output channels, all writing the one output
  buffer `output_ptr` (`out` is constant). `bufs = [coord_ptr, output_ptr]` is
  the decoupled allocation list — two buffers, fourteen channels.
* `B = block_size` — the lane window each program owns.
* `read i` (`sphInWin`) — the interleaved coordinate layout: lane `j` of
  program `pid₀` reads `x`, `y`, `z` at `3j + 3·block_size·pid₀ + i`
  (`coord_stride = 3`).
* `write o` (`sphOutWin`) — the strided channel layout: lane `j` writes channel
  `k` at `j·output_stride + block_size·output_stride·pid₀ + col_offset + k`, so
  the eleven channels share one region but own **different windows**, one
  column apart inside each row.
* `readMask` / `writeMask` — the kernel's own `< coord_numel` /
  `< output_numel` guards, per channel.

Neither program id is used beyond `pid₀` (the kernel launches a 1-D grid), so
`pid₁` is ignored by every window. The windows and masks are declared, not
parsed from the kernel; the headline **proves** the kernel's actual addressing
and masking match them. Buffer sizes are not signature content: the headline
quantifies over every allocation whose extents cover the active lanes. -/
```
```lean
def fifthOrderFwdIO (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    GroupedMasked2DKernelIO where
  kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride
  nIn := 3
  nOut := 11
  bufs := [coord_ptr, output_ptr]
  inp := fun _ => coord_ptr
  out := fun _ => output_ptr
  B := block_size
  read := fun i pid₀ _ j => sphInWin block_size pid₀ i j
  readMask := fun i pid₀ _ j => sphInWin block_size pid₀ i j < coord_numel
  write := fun o pid₀ _ j => sphOutWin block_size col_offset output_stride pid₀ o j
  writeMask := fun o pid₀ _ j =>
    sphOutWin block_size col_offset output_stride pid₀ o j < output_numel
```
</details>

<details><summary><code>sphY</code></summary>

```
/-- The eleven real fifth-order spherical-harmonic polynomials, as pure
functions of one lane's coordinates `(x, y, z)`. Output channel `o` of the
kernel writes `sphY o`; these are exactly the `Y00..Y10` expressions of
`fifth_order_sph_harmonics.py`, with the shared monomial intermediates
(`VAR05 = x⁵`, `VAR26 = z²`, …) inlined. -/
```
```lean
noncomputable def sphY (o : Fin 11) (x y z : ℝ) : ℝ :=
  match o with
  | ⟨0, _⟩ => 2.32681380862329 * x^5 + 11.6340690431164 * z^4 * x
      - 23.2681380862329 * x^3 * z^2
  | ⟨1, _⟩ => -29.4321253055229 * y * (x^3 * z - x * z^3)
  | ⟨2, _⟩ => 1.73430461568895 * x^5
      + x^3 * (-13.8744369255116 * y^2 - 3.46860923137790 * z^2)
      + x * (41.6233107765348 * y^2 * z^2 - 5.20291384706685 * z^4)
  | ⟨3, _⟩ => -16.9926454679664 * x^3 * y * z
      + x * (33.9852909359329 * y^3 * z - 16.9926454679664 * z^3 * y)
  | ⟨4, _⟩ => 1.60565407233314 * x^5
      + x^3 * (3.21130814466628 * z^2 - 19.2678488679977 * y^2)
      + x * (1.60565407233314 * z^4 + 12.8452325786651 * y^4
          - 19.2678488679977 * y^2 * z^2)
  | ⟨5, _⟩ => 3.31662479035540 * y^5
      + y^3 * (-16.5831239517770 * x^2 - 16.5831239517770 * z^2)
      + y * (6.21867148191637 * x^4 + 6.21867148191637 * z^4
          + 12.4373429638327 * x^2 * z^2)
  | ⟨6, _⟩ => 1.60565407233314 * z^5
      + z^3 * (3.21130814466628 * x^2 - 19.2678488679977 * y^2)
      + z * (1.60565407233314 * x^4 + 12.8452325786651 * y^4
          - 19.2678488679977 * x^2 * y^2)
  | ⟨7, _⟩ => 16.9926454679664 * y^3 * (z^2 - x^2)
      + 8.49632273398321 * y * (x^4 - z^4)
  | ⟨8, _⟩ => -1.73430461568895 * z^5
      + z^3 * (13.8744369255116 * y^2 + 3.46860923137790 * x^2)
      + z * (-41.6233107765348 * x^2 * y^2 + 5.20291384706685 * x^4)
  | ⟨9, _⟩ => y * (7.35803132638072 * x^4 + 7.35803132638072 * z^4
      - 44.1481879582843 * x^2 * z^2)
  | ⟨_ + 10, _⟩ => 2.32681380862329 * z^5 + 11.6340690431164 * x^4 * z
      - 23.2681380862329 * x^2 * z^3
```
</details>

<details><summary><code>fifth_order_fwd_surface</code></summary>

```
/-- Faithful transcription of `fifth_order_sph_harmonics.py`'s
`fifth_order_fwd`.

This preserves the full forward kernel: coordinate loads, all fifth-order
polynomial intermediates, and the eleven strided `Y00..Y10` stores. The proved
`fifth_order_fwd_y00` kernel below remains the proof-oriented projection for
the first output channel. -/
```
```lean
def fifth_order_fwd_surface
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    ComputeKernel := triton {
  coord_stride = $(3)
  block_id = tl.program_id(0)
  coord_striding = tl.arange(0, $(block_size)) * coord_stride
  coord_row_offset = coord_striding + $(block_size) * coord_stride * block_id
  x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  y = tl.load(coord_ptr + coord_row_offset + $(1),
    mask=coord_row_offset + $(1) < $(coord_numel))
  z = tl.load(coord_ptr + coord_row_offset + $(2),
    mask=coord_row_offset + $(2) < $(coord_numel))
  CONST000 = 1.73430461568895
  CONST001 = 2.32681380862329
  CONST002 = 1.60565407233314
  CONST003 = 3.21130814466628
  CONST004 = 3.31662479035540
  CONST005 = 6.21867148191637
  CONST006 = 6.21867148191637
  CONST007 = 1.60565407233314
  CONST009 = 11.6340690431164
  CONST010 = 12.8452325786651
  CONST011 = 12.4373429638327
  CONST012 = 12.8452325786651
  CONST013 = 13.8744369255116
  CONST017 = 33.9852909359329
  CONST018 = 7.35803132638072
  CONST020 = -44.1481879582843
  CONST021 = -41.6233107765348
  CONST022 = -29.4321253055229
  CONST023 = -23.2681380862329
  CONST024 = -19.2678488679977
  CONST025 = -19.2678488679977
  CONST026 = -16.9926454679664
  CONST027 = -16.9926454679664
  CONST028 = -13.8744369255116
  CONST029 = -16.5831239517770
  CONST030 = 3.46860923137790
  CONST031 = -8.49632273398321
  CONST032 = -5.20291384706685
  CONST033 = -3.46860923137790
  CONST034 = -1.73430461568895
  VAR05 = x * x * x * x * x
  VAR06 = x * x * x * x
  VAR07 = x * x * x
  VAR08 = x * x
  VAR14 = y * y * y * y * y
  VAR15 = y * y * y * y
  VAR16 = y * y * y
  VAR17 = y * y
  VAR23 = z * z * z * z * z
  VAR24 = z * z * z * z
  VAR25 = z * z * z
  VAR26 = z * z
  Y00 = CONST001 * VAR05 + CONST009 * VAR24 * x + CONST023 * VAR07 * VAR26
  Y01 = y * (CONST022 * VAR07 * z - CONST022 * VAR25 * x)
  Y02 = CONST000 * VAR05 +
    VAR07 * (CONST028 * VAR17 + CONST033 * VAR26) +
    x * (-CONST021 * VAR17 * VAR26 + CONST032 * VAR24)
  Y03 = CONST027 * VAR07 * y * z +
    x * (CONST017 * VAR16 * z + CONST026 * VAR25 * y)
  Y04 = CONST002 * VAR05 +
    VAR07 * (CONST003 * VAR26 + CONST025 * VAR17) +
    x * (CONST002 * VAR24 + CONST010 * VAR15 + CONST024 * VAR17 * VAR26)
  Y05 = CONST004 * VAR14 +
    VAR16 * (CONST029 * VAR08 + CONST029 * VAR26) +
    y * (CONST005 * VAR06 + CONST006 * VAR24 + CONST011 * VAR08 * VAR26)
  Y06 = CONST002 * VAR23 +
    VAR25 * (CONST003 * VAR08 + CONST024 * VAR17) +
    z * (CONST007 * VAR06 + CONST012 * VAR15 + CONST024 * VAR08 * VAR17)
  Y07 = VAR16 * ((0 - 16.9926454679664) * VAR08 + 16.9926454679664 * VAR26) +
    y * (8.49632273398321 * VAR06 + (0 - 8.49632273398321) * VAR24)
  Y08 = CONST034 * VAR23 +
    VAR25 * (CONST013 * VAR17 + CONST030 * VAR08) +
    z * (CONST021 * VAR08 * VAR17 - CONST032 * VAR06)
  Y09 = y * (CONST018 * VAR06 + CONST018 * VAR24 + (0 - 44.1481879582843) * VAR08 * VAR26)
  Y10 = CONST001 * VAR23 + CONST009 * VAR06 * z + CONST023 * VAR08 * VAR25
  output_striding = tl.arange(0, $(block_size)) * $(output_stride)
  output_row_offset = output_striding + $(block_size) * $(output_stride) * block_id +
    $(col_offset)
  tl.store(output_ptr + output_row_offset, Y00, mask=output_row_offset < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(1), Y01,
    mask=output_row_offset + $(1) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(2), Y02,
    mask=output_row_offset + $(2) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(3), Y03,
    mask=output_row_offset + $(3) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(4), Y04,
    mask=output_row_offset + $(4) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(5), Y05,
    mask=output_row_offset + $(5) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(6), Y06,
    mask=output_row_offset + $(6) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(7), Y07,
    mask=output_row_offset + $(7) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(8), Y08,
    mask=output_row_offset + $(8) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(9), Y09,
    mask=output_row_offset + $(9) < $(output_numel))
  tl.store(output_ptr + output_row_offset + $(10), Y10,
    mask=output_row_offset + $(10) < $(output_numel))
}
```
</details>

<details><summary><code>sphInWin</code></summary>

```
/-- Input channel `i`'s lane-`j` load address for program `pid₀`: the three
coordinate reads share the row window `3j + 3·block_size·pid₀` and differ only
in the component column `+ i` (`x`, `y`, `z`). -/
```
```lean
def sphInWin (block_size pid₀ : Nat) (i : Fin 3) (j : Fin block_size) : Nat :=
  match i with
  | ⟨0, _⟩ => j.val * 3 + block_size * 3 * pid₀
  | ⟨1, _⟩ => j.val * 3 + block_size * 3 * pid₀ + 1
  | ⟨_ + 2, _⟩ => j.val * 3 + block_size * 3 * pid₀ + 2
```
</details>

<details><summary><code>sphOutWin</code></summary>

```
/-- Output channel `o`'s lane-`j` store address for program `pid₀`: the eleven
`Y0k` stores share the row window
`j·output_stride + block_size·output_stride·pid₀ + col_offset` and differ only
in the channel column `+ k`. -/
```
```lean
def sphOutWin (block_size col_offset output_stride pid₀ : Nat)
    (o : Fin 11) (j : Fin block_size) : Nat :=
  match o with
  | ⟨0, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset
  | ⟨1, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 1
  | ⟨2, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 2
  | ⟨3, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 3
  | ⟨4, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 4
  | ⟨5, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 5
  | ⟨6, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 6
  | ⟨7, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 7
  | ⟨8, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 8
  | ⟨9, _⟩ => j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 9
  | ⟨_ + 10, _⟩ =>
      j.val * output_stride + block_size * output_stride * pid₀ + col_offset + 10
```
</details>

## Also present (pinned special-case summaries)
- `fifth_order_fwd_y00_compute_correct`
- `fifth_order_fwd_channel_store_slice_compute_correct`
- `fifth_order_fwd_surface_y00_compute_correct`
- `fifth_order_fwd_surface_y01_compute_correct`
- `fifth_order_fwd_surface_y02_compute_correct`
- `fifth_order_fwd_surface_y03_compute_correct`
- `fifth_order_fwd_surface_y04_compute_correct`
- `fifth_order_fwd_surface_y05_compute_correct`
- `fifth_order_fwd_surface_y06_compute_correct`
- `fifth_order_fwd_surface_y07_compute_correct`
- `fifth_order_fwd_surface_y08_compute_correct`
- `fifth_order_fwd_surface_y09_compute_correct`
- `fifth_order_fwd_surface_y10_compute_correct`
