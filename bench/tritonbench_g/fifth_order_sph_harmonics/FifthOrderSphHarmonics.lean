import VeriTile.Triton

/-!
# `fifth_order_sph_harmonics` — strict per-kernel correctness

`fifth_order_fwd` evaluates the eleven real fifth-order spherical-harmonic
basis polynomials elementwise: each program loads a `block_size` tile of
`(x, y, z)` coordinates, computes the shared monomial intermediates and the
eleven polynomials `Y00..Y10` (each a fixed-constant polynomial in `x, y, z`),
and writes each channel with a strided store into `output_ptr`.

## Scope

This file verifies **the Triton kernel itself** — the per-program
`@triton.jit` body. The host launch (grid over coordinate blocks via
`calculate_lastdim_num_blocks`, and how the runtime composes per-program
strided writes) is the *trusted boundary*. Because `tl.program_id(0)` is
universally quantified, the per-program statement covers every program.

## Proof architecture

```
fifth_order_fwd_surface_y00_output_summary       ← TOP THEOREM (channel Y00)
  ├─ (toAlgorithm? = Except.ok _)                 surface lowers to the algorithm layer
  └─ fifth_order_fwd_surface_y00_compute_correct  ComputeCorrect over the Y00 strided store
       └─ fifth_order_fwd_surface_y00_correct      per-lane readback of y00Spec

fifth_order_fwd_surface_y0k_compute_correct       ← per-channel (k = 0..10)
  └─ fifth_order_fwd_surface_y0k_correct           per-lane readback of y0kSpec
```

The eleven `fifth_order_fwd_surface_y0{0..10}_compute_correct` theorems each
verify one output channel of the full surface kernel against its polynomial
spec `y0kSpec`. `fifth_order_fwd_y00_compute_correct` is the proof-oriented
single-channel projection.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. The kernel is purely elementwise (no recurrence): each
output is a polynomial in the loaded `(x, y, z)` lane values, modeled with
`Option.map₂` over the loaded coordinates so out-of-range loads propagate.
The eleven forward channels `Y00..Y10` are each verified; the backward kernel
`fifth_order_bwd` is transcribed, but the ~30-constant-per-dimension
gradient polynomials are **not** verified (left for future work, stated
honestly in the backward-section doc). Side conditions: per-channel
output-store-offset injectivity (`hOutInj`) and a stride bound (`hStride :
10 < output_stride`), with the offset-disjointness helpers
(`y0k_offset_disjoint` / `y0jk_offset_disjoint`) showing the eleven channel
columns are disjoint.
-/

namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `fifth_order_sph_harmonics.py`'s
`fifth_order_fwd`.

This preserves the full forward kernel: coordinate loads, all fifth-order
polynomial intermediates, and the eleven strided `Y00..Y10` stores. The proved
`fifth_order_fwd_y00` kernel below remains the proof-oriented projection for
the first output channel. -/
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

/-- The full fifth-order spherical harmonics forward surface lowers to the
algorithm layer, including all eleven output channels. -/
theorem fifth_order_fwd_surface_toAlgorithm_supported
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    ∃ alg, (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
      output_numel col_offset output_stride).toAlgorithm? = Except.ok alg := by
  simp [fifth_order_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of `fifth_order_sph_harmonics.py`'s
`fifth_order_bwd`. -/
def fifth_order_bwd_surface
    (coord_ptr coord_grad_ptr sph_grad_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    ComputeKernel := triton {
  block_id = tl.program_id(0)
  coord_striding = tl.arange(0, $(block_size)) * $(3)
  coord_row_offset = coord_striding + $(block_size) * $(3) * block_id
  x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  y = tl.load(coord_ptr + coord_row_offset + $(1),
    mask=coord_row_offset + $(1) < $(coord_numel))
  z = tl.load(coord_ptr + coord_row_offset + $(2),
    mask=coord_row_offset + $(2) < $(coord_numel))
  output_striding = tl.arange(0, $(block_size)) * $(output_stride)
  output_row_offset = output_striding + $(block_size) * $(output_stride) * block_id +
    $(col_offset)
  g_0 = tl.load(sph_grad_ptr + output_row_offset,
    mask=output_row_offset < $(output_numel))
  g_1 = tl.load(sph_grad_ptr + output_row_offset + $(1),
    mask=output_row_offset + $(1) < $(output_numel))
  g_2 = tl.load(sph_grad_ptr + output_row_offset + $(2),
    mask=output_row_offset + $(2) < $(output_numel))
  g_3 = tl.load(sph_grad_ptr + output_row_offset + $(3),
    mask=output_row_offset + $(3) < $(output_numel))
  g_4 = tl.load(sph_grad_ptr + output_row_offset + $(4),
    mask=output_row_offset + $(4) < $(output_numel))
  g_5 = tl.load(sph_grad_ptr + output_row_offset + $(5),
    mask=output_row_offset + $(5) < $(output_numel))
  g_6 = tl.load(sph_grad_ptr + output_row_offset + $(6),
    mask=output_row_offset + $(6) < $(output_numel))
  g_7 = tl.load(sph_grad_ptr + output_row_offset + $(7),
    mask=output_row_offset + $(7) < $(output_numel))
  g_8 = tl.load(sph_grad_ptr + output_row_offset + $(8),
    mask=output_row_offset + $(8) < $(output_numel))
  g_9 = tl.load(sph_grad_ptr + output_row_offset + $(9),
    mask=output_row_offset + $(9) < $(output_numel))
  g_10 = tl.load(sph_grad_ptr + output_row_offset + $(10),
    mask=output_row_offset + $(10) < $(output_numel))
  CONST000 = 1.60565407233314
  CONST001 = 3.00000000000000
  CONST002 = 3.21130814466628
  CONST003 = 1.60565407233314
  CONST004 = 6.42261628933256
  CONST005 = 6.42261628933256
  CONST006 = 8.67152307844476
  CONST007 = 8.02827036166571
  CONST008 = 6.93721846275580
  CONST009 = 11.6340690431164
  CONST010 = 12.8452325786651
  CONST011 = 6.21867148191637
  CONST012 = 6.21867148191637
  CONST014 = 12.4373429638327
  CONST017 = 12.8452325786651
  CONST018 = 13.8744369255116
  CONST019 = 24.8746859276655
  CONST020 = 24.8746859276655
  CONST021 = 27.7488738510232
  CONST024 = 29.4321253055229
  CONST027 = 7.35803132638072
  CONST029 = 46.5362761724657
  CONST030 = 51.3809303146605
  CONST031 = 51.3809303146605
  CONST034 = 101.955872807799
  CONST036 = -8.67152307844475
  CONST037 = 3.46860923137790
  CONST038 = -88.2963759165686
  CONST039 = -83.2466215530696
  CONST040 = -69.8044142586986
  CONST041 = -50.9779364038993
  CONST042 = -50.9779364038993
  CONST043 = -46.5362761724657
  CONST044 = -44.1481879582843
  CONST045 = -41.6233107765348
  CONST046 = -38.5356977359954
  CONST047 = -38.5356977359954
  CONST048 = -33.1662479035540
  CONST049 = -33.9852909359329
  CONST050 = 6.42261628933257
  CONST051 = -33.9852909359329
  CONST052 = -29.4321253055229
  CONST053 = -27.7488738510232
  CONST054 = -20.8116553882674
  CONST055 = -19.2678488679977
  CONST056 = -19.2678488679977
  CONST057 = -16.9926454679664
  CONST058 = -16.9926454679664
  CONST059 = -13.8744369255116
  CONST060 = -16.5831239517770
  CONST061 = -8.49632273398321
  CONST062 = -6.93721846275580
  CONST063 = -5.20291384706685
  CONST064 = -3.46860923137790
  VAR06 = x * x * x * x
  VAR07 = x * x * x
  VAR08 = x * x
  VAR15 = y * y * y * y
  VAR16 = y * y * y
  VAR17 = y * y
  VAR24 = z * z * z * z
  VAR25 = z * z * z
  VAR26 = z * z
  g_x = tl.load(coord_grad_ptr + coord_row_offset,
    mask=coord_row_offset < $(coord_numel))
  g_y = tl.load(coord_grad_ptr + coord_row_offset + $(1),
    mask=coord_row_offset + $(1) < $(coord_numel))
  g_z = tl.load(coord_grad_ptr + coord_row_offset + $(2),
    mask=coord_row_offset + $(2) < $(coord_numel))
  g_x += g_0 * (CONST009 * VAR06 + CONST009 * VAR24 + CONST040 * VAR08 * VAR26) +
    g_1 * y * (CONST038 * VAR08 * z - CONST052 * VAR25) +
    g_10 * (CONST029 * VAR07 * z + CONST043 * VAR25 * x) +
    g_2 * (CONST001 * VAR08 * (CONST059 * VAR17 + CONST064 * VAR26) +
      CONST006 * VAR06 - CONST045 * VAR17 * VAR26 + CONST063 * VAR24) +
    g_3 * (CONST041 * VAR08 * y * z - CONST049 * VAR16 * z +
      CONST057 * VAR25 * y) +
    g_4 * (CONST000 * VAR24 +
      CONST001 * VAR08 * (CONST002 * VAR26 + CONST055 * VAR17) +
      CONST007 * VAR06 + CONST010 * VAR15 + CONST056 * VAR17 * VAR26) +
    g_5 * (CONST048 * VAR16 * x + y * (CONST019 * VAR07 + CONST019 * VAR26 * x)) +
    g_6 * (CONST005 * VAR25 * x + z * (CONST004 * VAR07 + CONST046 * VAR17 * x)) +
    g_7 * (CONST049 * VAR16 * x - CONST051 * VAR07 * y) +
    g_8 * (CONST008 * VAR25 * x + z * (CONST039 * VAR17 * x - CONST054 * VAR07)) +
    g_9 * y * (CONST024 * VAR07 + CONST038 * VAR26 * x)
  g_y += g_1 * (CONST052 * VAR07 * z - CONST052 * VAR25 * x) +
    g_2 * (-CONST039 * VAR26 * x * y + CONST053 * VAR07 * y) +
    g_3 * (CONST058 * VAR07 * z + x * (CONST034 * VAR17 * z + CONST057 * VAR25)) +
    g_4 * (CONST047 * VAR07 * y + x * (CONST030 * VAR16 + CONST046 * VAR26 * y)) +
    g_5 * (CONST001 * VAR17 * (CONST060 * VAR08 + CONST060 * VAR26) +
      CONST011 * VAR06 + CONST012 * VAR24 + CONST014 * VAR08 * VAR26 -
      CONST060 * VAR15) +
    g_6 * (CONST046 * VAR25 * y + z * (CONST031 * VAR16 + CONST046 * VAR08 * y)) +
    g_7 * (CONST001 * VAR17 * (CONST057 * VAR08 - CONST057 * VAR26) -
      CONST061 * VAR06 + CONST061 * VAR24) +
    g_8 * (CONST021 * VAR25 * y + CONST039 * VAR08 * y * z) +
    g_9 * (CONST027 * VAR06 + CONST027 * VAR24 + CONST044 * VAR08 * VAR26)
  g_z += g_0 * (CONST029 * VAR25 * x + CONST043 * VAR07 * z) +
    g_1 * y * (-CONST038 * VAR26 * x + CONST052 * VAR07) +
    g_10 * (CONST009 * VAR06 + CONST009 * VAR24 + CONST040 * VAR08 * VAR26) +
    g_2 * (CONST062 * VAR07 * z + x * (-CONST039 * VAR17 * z + CONST054 * VAR25)) +
    g_3 * (CONST058 * VAR07 * y + x * (CONST042 * VAR26 * y - CONST049 * VAR16)) +
    g_4 * (CONST005 * VAR07 * z + x * (CONST046 * VAR17 * z + CONST050 * VAR25)) +
    g_5 * (CONST048 * VAR16 * z + y * (CONST019 * VAR08 * z + CONST020 * VAR25)) +
    g_6 * (CONST001 * VAR26 * (CONST002 * VAR08 + CONST056 * VAR17) +
      CONST003 * VAR06 + CONST007 * VAR24 + CONST017 * VAR15 +
      CONST056 * VAR08 * VAR17) +
    g_7 * (-CONST049 * VAR16 * z + CONST051 * VAR25 * y) +
    g_8 * (CONST001 * VAR26 * (CONST018 * VAR17 + CONST037 * VAR08) +
      CONST036 * VAR24 + CONST045 * VAR08 * VAR17 - CONST063 * VAR06) +
    g_9 * y * (CONST024 * VAR25 + CONST038 * VAR08 * z)
  tl.store(coord_grad_ptr + coord_row_offset, g_x,
    mask=coord_row_offset < $(coord_numel))
  tl.store(coord_grad_ptr + coord_row_offset + $(1), g_y,
    mask=coord_row_offset + $(1) < $(coord_numel))
  tl.store(coord_grad_ptr + coord_row_offset + $(2), g_z,
    mask=coord_row_offset + $(2) < $(coord_numel))
}

/-- The full fifth-order spherical harmonics backward surface lowers to the
algorithm layer, including all coordinate-gradient updates. -/
theorem fifth_order_bwd_surface_toAlgorithm_supported
    (coord_ptr coord_grad_ptr sph_grad_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    ∃ alg, (fifth_order_bwd_surface coord_ptr coord_grad_ptr sph_grad_ptr
      block_size coord_numel output_numel col_offset output_stride).toAlgorithm?
        = Except.ok alg := by
  simp [fifth_order_bwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented `Y00` slice of `fifth_order_sph_harmonics.py`'s
`fifth_order_fwd`.

The full source kernel writes eleven fifth-order spherical harmonics channels.
This slice covers the common coordinate/block pointer arithmetic, masked
coordinate loads, the `Y00` fifth-order polynomial, and the first strided output
store. The full forward and backward surfaces above cover the remaining
channels and gradient writeback; this kernel remains as the proved
single-channel projection. -/
def fifth_order_fwd_y00
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    ComputeKernel := triton {
  block_id = tl.program_id(0)
  coord_striding = tl.arange(0, $(block_size)) * $(3)
  coord_row_offset = coord_striding + $(block_size) * $(3) * block_id
  x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < $(coord_numel))
  z = tl.load(coord_ptr + coord_row_offset + $(2), mask=coord_row_offset + $(2) < $(coord_numel))
  x2 = x * x
  x3 = x2 * x
  x5 = x3 * x2
  z2 = z * z
  z4 = z2 * z2
  y00 = 2.32681380862329 * x5 + 11.6340690431164 * z4 * x - 23.2681380862329 * x3 * z2
  output_striding = tl.arange(0, $(block_size)) * $(output_stride)
  output_row_offset = output_striding + $(block_size) * $(output_stride) * block_id +
    $(col_offset)
  tl.store(output_ptr + output_row_offset, y00, mask=output_row_offset < $(output_numel))
}

def coordOffset (s : BlockState) (block_size : Nat) (i : Fin block_size) : Nat :=
  i.val * 3 + block_size * 3 * s.pid

def outOffset
    (s : BlockState) (block_size col_offset output_stride : Nat)
    (i : Fin block_size) : Nat :=
  i.val * output_stride + block_size * output_stride * s.pid + col_offset

noncomputable def coordX
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : WithBot ℝ :=
  if coordOffset s block_size i < coord_numel then
    some (s.readMem coord_ptr (coordOffset s block_size i))
  else some (s.undef coord_ptr (coordOffset s block_size i))

noncomputable def coordY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : WithBot ℝ :=
  if coordOffset s block_size i + 1 < coord_numel then
    some (s.readMem coord_ptr (coordOffset s block_size i + 1))
  else some (s.undef coord_ptr (coordOffset s block_size i + 1))

noncomputable def coordZ
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : WithBot ℝ :=
  if coordOffset s block_size i + 2 < coord_numel then
    some (s.readMem coord_ptr (coordOffset s block_size i + 2))
  else some (s.undef coord_ptr (coordOffset s block_size i + 2))

noncomputable def y00Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun x1 x2 => x1 * x2) x x
  let x3 := Option.map₂ (fun x1 x2 => x1 * x2) x2 x
  let x5 := Option.map₂ (fun x1 x2 => x1 * x2) x3 x2
  let z2 := Option.map₂ (fun x1 x2 => x1 * x2) z z
  let z4 := Option.map₂ (fun x1 x2 => x1 * x2) z2 z2
  WithBot.unbotD 0
    (Option.map₂ (fun x1 x2 => x1 - x2)
      (Option.map₂ (fun x1 x2 => x1 + x2)
        (Option.map (fun b => 2.32681380862329 * b) x5)
        (Option.map₂ (fun x1 x2 => x1 * x2)
          (Option.map (fun b => 11.6340690431164 * b) z4)
          x))
      (Option.map₂ (fun x1 x2 => x1 * x2)
        (Option.map (fun b => 23.2681380862329 * b) x3)
        z2))

/-- Algorithm-layer correctness for the `Y00` forward slice. -/
theorem fifth_order_fwd_y00_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_y00 coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i) =
        if outOffset s block_size col_offset output_stride i < output_numel then
          y00Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_y00, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset < output_numel
    · simp [hActive, y00Spec, coordX, coordZ, coordOffset, NumericDType.mul]
      rfl
    · simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `Y00` forward slice. -/
theorem fifth_order_fwd_y00_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_y00 coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i)))
      (expected := fun i => y00Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_y00]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_y00_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented per-channel `Yk` store slice of
`fifth_order_sph_harmonics.py`'s `fifth_order_fwd`. Parameterized over the
channel column offset, takes a precomputed `YkPre` row and proves the masked
strided writeback into `output_ptr`. Used to factor the 11 output-channel
stores out of the per-polynomial computation. -/
def fifth_order_fwd_channel_store_slice
    (YkPre output_ptr : RegionName)
    (block_size output_numel col_offset output_stride : Nat) :
    ComputeKernel := triton {
  block_id = tl.program_id(0)
  output_striding = tl.arange(0, $(block_size)) * $(output_stride)
  output_row_offset = output_striding + $(block_size) * $(output_stride) * block_id +
    $(col_offset)
  yk = tl.load(YkPre + output_row_offset, mask=output_row_offset < $(output_numel))
  tl.store(output_ptr + output_row_offset, yk, mask=output_row_offset < $(output_numel))
}

theorem fifth_order_fwd_channel_store_slice_correct
    (YkPre output_ptr : RegionName)
    (block_size output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_channel_store_slice YkPre output_ptr
        block_size output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i) =
        if outOffset s block_size col_offset output_stride i < output_numel then
          s.readMem YkPre (outOffset s block_size col_offset output_stride i)
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, fifth_order_fwd_channel_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [outOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : i.val * output_stride + block_size * output_stride * s.pid + col_offset < output_numel
  · simp [outOffset, hi]
  · simp [outOffset, hi]

theorem fifth_order_fwd_channel_store_slice_compute_correct
    (YkPre output_ptr : RegionName)
    (block_size output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_channel_store_slice YkPre output_ptr
        block_size output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i)))
      (expected := fun i =>
        s.readMem YkPre (outOffset s block_size col_offset output_stride i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_channel_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_channel_store_slice_correct YkPre output_ptr
    block_size output_numel col_offset output_stride s s' hOutInj hExec i
  simpa [hActive] using h

/-- Disjointness helper: the Y00 readback offset at lane `i` differs from
any `Y0k` store offset at lane `idx`, for `k ∈ [1, 10]`. -/
private theorem y0k_offset_disjoint
    (s : BlockState) (block_size col_offset output_stride : Nat)
    (hStride : 10 < output_stride)
    (k : Nat) (hk_pos : 0 < k) (hk_le : k ≤ 10)
    (i : Fin block_size) (idx : TileIndex [block_size]) :
    i.val * output_stride + block_size * output_stride * s.pid + col_offset ≠
      idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + k := by
  -- Reduce to: i.val * output_stride ≠ idx.1.val * output_stride + k.
  have hk_lt_stride : k < output_stride := Nat.lt_of_le_of_lt hk_le hStride
  intro heq
  -- Strip the common `base = block_size * output_stride * s.pid + col_offset`.
  have hbase :
      i.val * output_stride + (block_size * output_stride * s.pid + col_offset) =
      idx.1.val * output_stride + k + (block_size * output_stride * s.pid + col_offset) := by
    have hl :
        i.val * output_stride + block_size * output_stride * s.pid + col_offset =
        i.val * output_stride + (block_size * output_stride * s.pid + col_offset) := by
      ring
    have hr :
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + k =
        idx.1.val * output_stride + k + (block_size * output_stride * s.pid + col_offset) := by
      ring
    rw [hl, hr] at heq
    exact heq
  have hkey : i.val * output_stride = idx.1.val * output_stride + k :=
    Nat.add_right_cancel hbase
  -- Case-split on i.val vs idx.1.val.
  rcases Nat.lt_trichotomy i.val idx.1.val with hlt | heqv | hgt
  · -- i.val < idx.1.val: i.val * stride < idx.1.val * stride ≤ idx.1.val * stride + k.
    have h1 : i.val * output_stride < idx.1.val * output_stride := by
      have hpos : 0 < output_stride := Nat.lt_of_lt_of_le hk_pos (Nat.le_of_lt hk_lt_stride)
      exact (Nat.mul_lt_mul_right hpos).mpr hlt
    have h2 : idx.1.val * output_stride ≤ idx.1.val * output_stride + k := Nat.le_add_right _ _
    exact absurd hkey (Nat.ne_of_lt (Nat.lt_of_lt_of_le h1 h2))
  · -- i.val = idx.1.val: hkey reduces to 0 = k, contradicts hk_pos.
    rw [heqv] at hkey
    have : k = 0 := by omega
    exact absurd this (Nat.pos_iff_ne_zero.mp hk_pos)
  · -- i.val > idx.1.val: i.val * stride ≥ (idx.1.val + 1) * stride = idx.1.val * stride + stride > idx.1.val * stride + k.
    have h1 : idx.1.val + 1 ≤ i.val := hgt
    have h2 : (idx.1.val + 1) * output_stride ≤ i.val * output_stride :=
      Nat.mul_le_mul_right _ h1
    have h3 : (idx.1.val + 1) * output_stride = idx.1.val * output_stride + output_stride := by ring
    have h4 : idx.1.val * output_stride + output_stride ≤ i.val * output_stride := by
      rw [← h3]; exact h2
    have h5 : idx.1.val * output_stride + k < idx.1.val * output_stride + output_stride := by
      exact Nat.add_lt_add_left hk_lt_stride _
    have h6 : idx.1.val * output_stride + k < i.val * output_stride :=
      Nat.lt_of_lt_of_le h5 h4
    exact absurd hkey.symm (Nat.ne_of_lt h6)

/-- Generalized disjointness helper: the `Yk` readback offset at lane `i`
differs from any `Yj` store offset at lane `idx`, when `j, k ∈ [0, 10]` and
`j ≠ k`, given `output_stride > 10`. -/
private theorem y0jk_offset_disjoint
    (s : BlockState) (block_size col_offset output_stride : Nat)
    (hStride : 10 < output_stride)
    (j k : Nat) (hj_le : j ≤ 10) (hk_le : k ≤ 10) (hjk_ne : j ≠ k)
    (i : Fin block_size) (idx : TileIndex [block_size]) :
    i.val * output_stride + block_size * output_stride * s.pid + col_offset + k ≠
      idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + j := by
  have hj_lt_stride : j < output_stride := Nat.lt_of_le_of_lt hj_le hStride
  have hk_lt_stride : k < output_stride := Nat.lt_of_le_of_lt hk_le hStride
  intro heq
  -- Strip the common `base = block_size * output_stride * s.pid + col_offset`.
  have hbase :
      i.val * output_stride + k + (block_size * output_stride * s.pid + col_offset) =
      idx.1.val * output_stride + j + (block_size * output_stride * s.pid + col_offset) := by
    have hl :
        i.val * output_stride + block_size * output_stride * s.pid + col_offset + k =
        i.val * output_stride + k + (block_size * output_stride * s.pid + col_offset) := by
      ring
    have hr :
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + j =
        idx.1.val * output_stride + j + (block_size * output_stride * s.pid + col_offset) := by
      ring
    rw [hl, hr] at heq
    exact heq
  have hkey : i.val * output_stride + k = idx.1.val * output_stride + j :=
    Nat.add_right_cancel hbase
  -- Case-split on i.val vs idx.1.val.
  rcases Nat.lt_trichotomy i.val idx.1.val with hlt | heqv | hgt
  · -- i.val < idx.1.val: i.val * stride + k < (i.val+1) * stride ≤ idx.1.val * stride ≤ idx.1.val * stride + j.
    have hstep : i.val + 1 ≤ idx.1.val := hlt
    have h1 : (i.val + 1) * output_stride ≤ idx.1.val * output_stride :=
      Nat.mul_le_mul_right _ hstep
    have h2 : (i.val + 1) * output_stride = i.val * output_stride + output_stride := by ring
    have h3 : i.val * output_stride + k < i.val * output_stride + output_stride :=
      Nat.add_lt_add_left hk_lt_stride _
    have h4 : i.val * output_stride + output_stride ≤ idx.1.val * output_stride := by
      rw [← h2]; exact h1
    have h5 : i.val * output_stride + k < idx.1.val * output_stride :=
      Nat.lt_of_lt_of_le h3 h4
    have h6 : idx.1.val * output_stride ≤ idx.1.val * output_stride + j := Nat.le_add_right _ _
    exact absurd hkey (Nat.ne_of_lt (Nat.lt_of_lt_of_le h5 h6))
  · -- i.val = idx.1.val: hkey reduces to k = j, contradicts hjk_ne.
    rw [heqv] at hkey
    have hkj : k = j := Nat.add_left_cancel hkey
    exact hjk_ne hkj.symm
  · -- i.val > idx.1.val: symmetric.
    have hstep : idx.1.val + 1 ≤ i.val := hgt
    have h1 : (idx.1.val + 1) * output_stride ≤ i.val * output_stride :=
      Nat.mul_le_mul_right _ hstep
    have h2 : (idx.1.val + 1) * output_stride = idx.1.val * output_stride + output_stride := by ring
    have h3 : idx.1.val * output_stride + j < idx.1.val * output_stride + output_stride :=
      Nat.add_lt_add_left hj_lt_stride _
    have h4 : idx.1.val * output_stride + output_stride ≤ i.val * output_stride := by
      rw [← h2]; exact h1
    have h5 : idx.1.val * output_stride + j < i.val * output_stride :=
      Nat.lt_of_lt_of_le h3 h4
    have h6 : i.val * output_stride ≤ i.val * output_stride + k := Nat.le_add_right _ _
    exact absurd hkey.symm (Nat.ne_of_lt (Nat.lt_of_lt_of_le h5 h6))

/-- Algorithm-layer correctness for the full forward `Y00` store: even when
the kernel proceeds to write `Y01..Y10` at offsets `+1..+10`, those writes
target offsets disjoint from the `Y00` readback offset as long as
`output_stride > 10`. -/
theorem fifth_order_fwd_surface_y00_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i) =
        if outOffset s block_size col_offset output_stride i < output_numel then
          y00Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  -- Disjointness packets for k = 1..10.
  have hd1 := y0k_offset_disjoint s block_size col_offset output_stride hStride 1
    (by norm_num) (by norm_num) i
  have hd2 := y0k_offset_disjoint s block_size col_offset output_stride hStride 2
    (by norm_num) (by norm_num) i
  have hd3 := y0k_offset_disjoint s block_size col_offset output_stride hStride 3
    (by norm_num) (by norm_num) i
  have hd4 := y0k_offset_disjoint s block_size col_offset output_stride hStride 4
    (by norm_num) (by norm_num) i
  have hd5 := y0k_offset_disjoint s block_size col_offset output_stride hStride 5
    (by norm_num) (by norm_num) i
  have hd6 := y0k_offset_disjoint s block_size col_offset output_stride hStride 6
    (by norm_num) (by norm_num) i
  have hd7 := y0k_offset_disjoint s block_size col_offset output_stride hStride 7
    (by norm_num) (by norm_num) i
  have hd8 := y0k_offset_disjoint s block_size col_offset output_stride hStride 8
    (by norm_num) (by norm_num) i
  have hd9 := y0k_offset_disjoint s block_size col_offset output_stride hStride 9
    (by norm_num) (by norm_num) i
  have hd10 := y0k_offset_disjoint s block_size col_offset output_stride hStride 10
    (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    -- Strip the 10 outer foldls (Y10, Y09, ..., Y01) one at a time.
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 7 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
          (hOff := fun k _ _ => hd7 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 6 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
          (hOff := fun k _ _ => hd6 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 5 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
          (hOff := fun k _ _ => hd5 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 4 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
          (hOff := fun k _ _ => hd4 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 3 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
          (hOff := fun k _ _ => hd3 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 2 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
          (hOff := fun k _ _ => hd2 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 1 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
          (hOff := fun k _ _ => hd1 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset < output_numel
    · -- Active branch. The surface kernel writes
      -- `Y00 = 2.32681 * x^5 + 11.634 * z^4 * x + (0 - 23.268) * x^3 * z^2`
      -- via the unary-minus expansion of `CONST023`, while `y00Spec` uses
      -- `... - 23.268 * x^3 * z^2`. After splitting the coordinate `if`s,
      -- close by `norm_num` (to fold `0.0 = 0`) plus `ring`.
      simp only [hActive, if_true, y00Spec, coordX, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs with hx hz hz
      all_goals (simp only [id]; norm_num; ring)
    · simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the full forward `Y00` store. -/
theorem fifth_order_fwd_surface_y00_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i)))
      (expected := fun i => y00Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y00_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

noncomputable def y01Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun x1 x2 => x1 * x2) x x
  let x3 := Option.map₂ (fun x1 x2 => x1 * x2) x2 x
  let z2 := Option.map₂ (fun x1 x2 => x1 * x2) z z
  let z3 := Option.map₂ (fun x1 x2 => x1 * x2) z2 z
  WithBot.unbotD 0
    (Option.map₂ (fun x1 x2 => x1 * x2)
      y
      (Option.map₂ (fun x1 x2 => x1 - x2)
        (Option.map₂ (fun x1 x2 => x1 * x2)
          (Option.map (fun b => (-29.4321253055229 : ℝ) * b) x3) z)
        (Option.map₂ (fun x1 x2 => x1 * x2)
          (Option.map (fun b => (-29.4321253055229 : ℝ) * b) z3) x)))

theorem fifth_order_fwd_surface_y01_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 1) =
        if outOffset s block_size col_offset output_stride i + 1 < output_numel then
          y01Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 1) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 1) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 1
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 1
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 7 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
          (hOff := fun k _ _ => hd7 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 6 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
          (hOff := fun k _ _ => hd6 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 5 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
          (hOff := fun k _ _ => hd5 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 4 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
          (hOff := fun k _ _ => hd4 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 3 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
          (hOff := fun k _ _ => hd3 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 2 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
          (hOff := fun k _ _ => hd2 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 1 < output_numel
    · simp only [hActive, if_true, y01Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y01_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 1 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 1)))
      (expected := fun i => y01Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y01_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

noncomputable def y02Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=

  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x3 := Option.map₂ (fun a b => a * b) x2 x
  let x5 := Option.map₂ (fun a b => a * b) x3 x2
  let y2 := Option.map₂ (fun a b => a * b) y y
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z4 := Option.map₂ (fun a b => a * b) z2 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a + b)
        (Option.map (fun b => (1.73430461568895 : ℝ) * b) x5)
        (Option.map₂ (fun a b => a * b) x3
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (-13.8744369255116 : ℝ) * b) y2)
            (Option.map (fun b => (-3.46860923137790 : ℝ) * b) z2))))
      (Option.map₂ (fun a b => a * b) x
        (Option.map₂ (fun a b => a + b)
          (Option.map₂ (fun a b => a * b)
            (Option.map (fun b => (41.6233107765348 : ℝ) * b) y2) z2)
          (Option.map (fun b => (-5.20291384706685 : ℝ) * b) z4))))

theorem fifth_order_fwd_surface_y02_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 2) =
        if outOffset s block_size col_offset output_stride i + 2 < output_numel then
          y02Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 2) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 2) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 2
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 2
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    -- Strip Y10..Y03.
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 7 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
          (hOff := fun k _ _ => hd7 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 6 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
          (hOff := fun k _ _ => hd6 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 5 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
          (hOff := fun k _ _ => hd5 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 4 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
          (hOff := fun k _ _ => hd4 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 3 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
          (hOff := fun k _ _ => hd3 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 2 < output_numel
    · simp only [hActive, if_true, y02Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · -- Inactive: strip Y01, Y00 inner folds.
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y02_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 2 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 2)))
      (expected := fun i => y02Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y02_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

noncomputable def y03Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x3 := Option.map₂ (fun a b => a * b) x2 x
  let y2 := Option.map₂ (fun a b => a * b) y y
  let y3 := Option.map₂ (fun a b => a * b) y2 y
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z3 := Option.map₂ (fun a b => a * b) z2 z
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a * b)
        (Option.map₂ (fun a b => a * b)
          (Option.map (fun b => (-16.9926454679664 : ℝ) * b) x3) y) z)
      (Option.map₂ (fun a b => a * b) x
        (Option.map₂ (fun a b => a + b)
          (Option.map₂ (fun a b => a * b)
            (Option.map (fun b => (33.9852909359329 : ℝ) * b) y3) z)
          (Option.map₂ (fun a b => a * b)
            (Option.map (fun b => (-16.9926454679664 : ℝ) * b) z3) y))))

noncomputable def y04Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x3 := Option.map₂ (fun a b => a * b) x2 x
  let x5 := Option.map₂ (fun a b => a * b) x3 x2
  let y2 := Option.map₂ (fun a b => a * b) y y
  let y4 := Option.map₂ (fun a b => a * b) y2 y2
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z4 := Option.map₂ (fun a b => a * b) z2 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a + b)
        (Option.map (fun b => (1.60565407233314 : ℝ) * b) x5)
        (Option.map₂ (fun a b => a * b) x3
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (3.21130814466628 : ℝ) * b) z2)
            (Option.map (fun b => (-19.2678488679977 : ℝ) * b) y2))))
      (Option.map₂ (fun a b => a * b) x
        (Option.map₂ (fun a b => a + b)
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (1.60565407233314 : ℝ) * b) z4)
            (Option.map (fun b => (12.8452325786651 : ℝ) * b) y4))
          (Option.map₂ (fun a b => a * b)
            (Option.map (fun b => (-19.2678488679977 : ℝ) * b) y2) z2))))

noncomputable def y05Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x4 := Option.map₂ (fun a b => a * b) x2 x2
  let y2 := Option.map₂ (fun a b => a * b) y y
  let y3 := Option.map₂ (fun a b => a * b) y2 y
  let y4 := Option.map₂ (fun a b => a * b) y2 y2
  let y5 := Option.map₂ (fun a b => a * b) y4 y
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z4 := Option.map₂ (fun a b => a * b) z2 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a + b)
        (Option.map (fun b => (3.31662479035540 : ℝ) * b) y5)
        (Option.map₂ (fun a b => a * b) y3
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (-16.5831239517770 : ℝ) * b) x2)
            (Option.map (fun b => (-16.5831239517770 : ℝ) * b) z2))))
      (Option.map₂ (fun a b => a * b) y
        (Option.map₂ (fun a b => a + b)
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (6.21867148191637 : ℝ) * b) x4)
            (Option.map (fun b => (6.21867148191637 : ℝ) * b) z4))
          (Option.map₂ (fun a b => a * b)
            (Option.map (fun b => (12.4373429638327 : ℝ) * b) x2) z2))))

noncomputable def y06Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x4 := Option.map₂ (fun a b => a * b) x2 x2
  let y2 := Option.map₂ (fun a b => a * b) y y
  let y4 := Option.map₂ (fun a b => a * b) y2 y2
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z3 := Option.map₂ (fun a b => a * b) z2 z
  let z5 := Option.map₂ (fun a b => a * b) z3 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a + b)
        (Option.map (fun b => (1.60565407233314 : ℝ) * b) z5)
        (Option.map₂ (fun a b => a * b) z3
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (3.21130814466628 : ℝ) * b) x2)
            (Option.map (fun b => (-19.2678488679977 : ℝ) * b) y2))))
      (Option.map₂ (fun a b => a * b) z
        (Option.map₂ (fun a b => a + b)
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (1.60565407233314 : ℝ) * b) x4)
            (Option.map (fun b => (12.8452325786651 : ℝ) * b) y4))
          (Option.map₂ (fun a b => a * b)
            (Option.map (fun b => (-19.2678488679977 : ℝ) * b) x2) y2))))

noncomputable def y07Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x4 := Option.map₂ (fun a b => a * b) x2 x2
  let y2 := Option.map₂ (fun a b => a * b) y y
  let y3 := Option.map₂ (fun a b => a * b) y2 y
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z4 := Option.map₂ (fun a b => a * b) z2 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a * b) y3
        (Option.map₂ (fun a b => a - b)
          (Option.map (fun b => (-16.9926454679664 : ℝ) * b) x2)
          (Option.map (fun b => (-16.9926454679664 : ℝ) * b) z2)))
      (Option.map₂ (fun a b => a * b) y
        (Option.map₂ (fun a b => a + b)
          (Option.map (fun b => -((-8.49632273398321 : ℝ)) * b) x4)
          (Option.map (fun b => (-8.49632273398321 : ℝ) * b) z4))))

noncomputable def y08Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x4 := Option.map₂ (fun a b => a * b) x2 x2
  let y2 := Option.map₂ (fun a b => a * b) y y
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z3 := Option.map₂ (fun a b => a * b) z2 z
  let z5 := Option.map₂ (fun a b => a * b) z3 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a + b)
        (Option.map (fun b => (-1.73430461568895 : ℝ) * b) z5)
        (Option.map₂ (fun a b => a * b) z3
          (Option.map₂ (fun a b => a + b)
            (Option.map (fun b => (13.8744369255116 : ℝ) * b) y2)
            (Option.map (fun b => (3.46860923137790 : ℝ) * b) x2))))
      (Option.map₂ (fun a b => a * b) z
        (Option.map₂ (fun a b => a - b)
          (Option.map₂ (fun a b => a * b)
            (Option.map (fun b => (-41.6233107765348 : ℝ) * b) x2) y2)
          (Option.map (fun b => (-5.20291384706685 : ℝ) * b) x4))))

noncomputable def y09Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let y := coordY s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x4 := Option.map₂ (fun a b => a * b) x2 x2
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z4 := Option.map₂ (fun a b => a * b) z2 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a * b) y
      (Option.map₂ (fun a b => a + b)
        (Option.map₂ (fun a b => a + b)
          (Option.map (fun b => (7.35803132638072 : ℝ) * b) x4)
          (Option.map (fun b => (7.35803132638072 : ℝ) * b) z4))
        (Option.map₂ (fun a b => a * b)
          (Option.map (fun b => (-44.1481879582843 : ℝ) * b) x2) z2)))

noncomputable def y10Spec
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) : ℝ :=
  let x := coordX s coord_ptr block_size coord_numel i
  let z := coordZ s coord_ptr block_size coord_numel i
  let x2 := Option.map₂ (fun a b => a * b) x x
  let x4 := Option.map₂ (fun a b => a * b) x2 x2
  let z2 := Option.map₂ (fun a b => a * b) z z
  let z3 := Option.map₂ (fun a b => a * b) z2 z
  let z5 := Option.map₂ (fun a b => a * b) z3 z2
  WithBot.unbotD 0
    (Option.map₂ (fun a b => a + b)
      (Option.map₂ (fun a b => a + b)
        (Option.map (fun b => (2.32681380862329 : ℝ) * b) z5)
        (Option.map₂ (fun a b => a * b)
          (Option.map (fun b => (11.6340690431164 : ℝ) * b) x4) z))
      (Option.map₂ (fun a b => a * b)
        (Option.map (fun b => (-23.2681380862329 : ℝ) * b) x2) z3))

theorem fifth_order_fwd_surface_y03_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 3) =
        if outOffset s block_size col_offset output_stride i + 3 < output_numel then
          y03Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 3) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 3) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 3
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 3
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 7 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
          (hOff := fun k _ _ => hd7 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 6 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
          (hOff := fun k _ _ => hd6 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 5 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
          (hOff := fun k _ _ => hd5 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 4 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
          (hOff := fun k _ _ => hd4 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 3 < output_numel
    · simp only [hActive, if_true, y03Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y03_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 3 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 3)))
      (expected := fun i => y03Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y03_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

theorem fifth_order_fwd_surface_y04_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 4) =
        if outOffset s block_size col_offset output_stride i + 4 < output_numel then
          y04Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 4) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 4) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 4
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 4
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 7 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
          (hOff := fun k _ _ => hd7 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 6 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
          (hOff := fun k _ _ => hd6 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 5 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
          (hOff := fun k _ _ => hd5 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 4 < output_numel
    · simp only [hActive, if_true, y04Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 3 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
            (hOff := fun k _ _ => hd3 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y04_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 4 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 4)))
      (expected := fun i => y04Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y04_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

theorem fifth_order_fwd_surface_y05_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 5) =
        if outOffset s block_size col_offset output_stride i + 5 < output_numel then
          y05Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 5) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 5) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 5
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 5
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 7 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
          (hOff := fun k _ _ => hd7 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 6 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
          (hOff := fun k _ _ => hd6 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 5 < output_numel
    · simp only [hActive, if_true, y05Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 4 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
            (hOff := fun k _ _ => hd4 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 3 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
            (hOff := fun k _ _ => hd3 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y05_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 5 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 5)))
      (expected := fun i => y05Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y05_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

theorem fifth_order_fwd_surface_y06_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 6) =
        if outOffset s block_size col_offset output_stride i + 6 < output_numel then
          y06Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 6) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 6) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 6
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 6
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 7 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
          (hOff := fun k _ _ => hd7 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 6 < output_numel
    · simp only [hActive, if_true, y06Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 5 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
            (hOff := fun k _ _ => hd5 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 4 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
            (hOff := fun k _ _ => hd4 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 3 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
            (hOff := fun k _ _ => hd3 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y06_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 6 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 6)))
      (expected := fun i => y06Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y06_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

theorem fifth_order_fwd_surface_y07_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 7) =
        if outOffset s block_size col_offset output_stride i + 7 < output_numel then
          y07Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 7) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 7) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 7
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 7
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 8 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
          (hOff := fun k _ _ => hd8 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 7 < output_numel
    · simp only [hActive, if_true, y07Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; try simp; try ring_nf; try norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 6 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
            (hOff := fun k _ _ => hd6 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 5 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
            (hOff := fun k _ _ => hd5 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 4 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
            (hOff := fun k _ _ => hd4 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 3 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
            (hOff := fun k _ _ => hd3 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y07_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 7 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 7)))
      (expected := fun i => y07Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y07_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

theorem fifth_order_fwd_surface_y08_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 8) =
        if outOffset s block_size col_offset output_stride i + 8 < output_numel then
          y08Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 8) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 8) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 8
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 8
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 9 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
          (hOff := fun k _ _ => hd9 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 8 < output_numel
    · simp only [hActive, if_true, y08Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 7 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
            (hOff := fun k _ _ => hd7 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 6 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
            (hOff := fun k _ _ => hd6 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 5 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
            (hOff := fun k _ _ => hd5 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 4 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
            (hOff := fun k _ _ => hd4 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 3 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
            (hOff := fun k _ _ => hd3 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y08_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 8 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 8)))
      (expected := fun i => y08Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y08_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

theorem fifth_order_fwd_surface_y09_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 9) =
        if outOffset s block_size col_offset output_stride i + 9 < output_numel then
          y09Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 9) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 9) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 9
    (by norm_num) (by norm_num) (by norm_num) i
  have hd10 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 10 9
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (P := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
              col_offset + 10 < output_numel)
          (offsetFn := fun idx : TileIndex [block_size] =>
            idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 10)
          (hOff := fun k _ _ => hd10 k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 9 < output_numel
    · simp only [hActive, if_true, y09Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; try simp; try ring_nf; try norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 8 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
            (hOff := fun k _ _ => hd8 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 7 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
            (hOff := fun k _ _ => hd7 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 6 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
            (hOff := fun k _ _ => hd6 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 5 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
            (hOff := fun k _ _ => hd5 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 4 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
            (hOff := fun k _ _ => hd4 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 3 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
            (hOff := fun k _ _ => hd3 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y09_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 9 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 9)))
      (expected := fun i => y09Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y09_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

theorem fifth_order_fwd_surface_y10_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s' : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i))
    (hExec : exec (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride) s = some s') :
    ∀ i : Fin block_size,
      s'.readMem output_ptr (outOffset s block_size col_offset output_stride i + 10) =
        if outOffset s block_size col_offset output_stride i + 10 < output_numel then
          y10Spec s coord_ptr block_size coord_numel i
        else s.readMem output_ptr (outOffset s block_size col_offset output_stride i + 10) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [block_size] =>
        idx.1.val * output_stride + block_size * output_stride * s.pid + col_offset + 10) := by
    intro a b h
    have h' : a.1.val * output_stride + block_size * output_stride * s.pid + col_offset =
              b.1.val * output_stride + block_size * output_stride * s.pid + col_offset :=
      Nat.add_right_cancel h
    have hab : a.1 = b.1 := by apply hOutInj; simpa [outOffset] using h'
    cases a; cases b; simp only at hab; cases hab; rfl
  have hd0 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 0 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd1 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 1 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd2 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 2 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd3 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 3 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd4 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 4 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd5 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 5 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd6 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 6 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd7 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 7 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd8 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 8 10
    (by norm_num) (by norm_num) (by norm_num) i
  have hd9 := y0jk_offset_disjoint s block_size col_offset output_stride hStride 9 10
    (by norm_num) (by norm_num) (by norm_num) i
  by_cases hB : 0 < block_size
  · simp [exec, fifth_order_fwd_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hActive : i.val * output_stride + block_size * output_stride * s.pid +
        col_offset + 10 < output_numel
    · simp only [hActive, if_true, y10Spec, coordX, coordY, coordZ, coordOffset,
                 NumericDType.mul, NumericDType.add, NumericDType.sub,
                 WithBot.realSub, WithBot.realAdd, WithBot.realMul,
                 Option.map₂, Option.map, WithBot.unbotD,
                 Option.bind, Option.bind_some, WithBot.recBotCoe]
      split_ifs
      all_goals (simp only [id]; first | (norm_num; ring) | norm_num)
    · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 9 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 9)
            (hOff := fun k _ _ => hd9 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 8 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 8)
            (hOff := fun k _ _ => hd8 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 7 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 7)
            (hOff := fun k _ _ => hd7 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 6 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 6)
            (hOff := fun k _ _ => hd6 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 5 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 5)
            (hOff := fun k _ _ => hd5 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 4 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 4)
            (hOff := fun k _ _ => hd4 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 3 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 3)
            (hOff := fun k _ _ => hd3 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 2 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 2)
            (hOff := fun k _ _ => hd2 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset + 1 < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset + 1)
            (hOff := fun k _ _ => hd1 k)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (P := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 +
                col_offset < output_numel)
            (offsetFn := fun idx : TileIndex [block_size] =>
              idx.1.val * output_stride + block_size * output_stride * s.pids 0 + col_offset)
            (hOff := fun k _ _ => by have := hd0 k; simpa using this)]
      simp [hActive]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem fifth_order_fwd_surface_y10_compute_correct
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i + 10 < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i + 10)))
      (expected := fun i => y10Spec s coord_ptr block_size coord_numel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fifth_order_fwd_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fifth_order_fwd_surface_y10_correct coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride s s' hStride hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `fifth_order_fwd` (representative channel
`Y00`): the DSL surface lowers to the algorithm layer, and the strided `Y00`
store to `output_ptr` is compute-correct — every active lane holds the
spherical-harmonic polynomial value `y00Spec`, inactive lanes are preserved.
Mirrors `add_kernel_output_summary`. The remaining channels `Y01..Y10` are
covered by the sibling `fifth_order_fwd_surface_y0k_compute_correct`
theorems. -/
theorem fifth_order_fwd_surface_y00_output_summary
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState)
    (hStride : 10 < output_stride)
    (hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s block_size col_offset output_stride i)) :
    (∃ alg, (fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin block_size =>
          outOffset s block_size col_offset output_stride i < output_numel)
        (fun i => (output_ptr, outOffset s block_size col_offset output_stride i)))
      (expected := fun i => y00Spec s coord_ptr block_size coord_numel i) := by
  refine ⟨⟨(fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
      output_numel col_offset output_stride).toAlgKernel, ?_⟩, ?_⟩
  · simp [fifth_order_fwd_surface]
  · exact fifth_order_fwd_surface_y00_compute_correct coord_ptr output_ptr block_size
      coord_numel output_numel col_offset output_stride s hStride hOutInj

end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics
