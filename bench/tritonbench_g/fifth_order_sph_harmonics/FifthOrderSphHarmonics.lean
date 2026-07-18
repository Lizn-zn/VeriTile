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
fifth_order_fwd_correctness                    ← TOP SPECIFICATION (fifthOrderFwdIO ⊨ sphY)
  ├─ fifth_order_fwd_surface_flattenOk         bridge fragment membership
  ├─ fifth_order_fwd_surface_traceSafe         per-execution lane-wise safety walk
  └─ fifth_order_fwd_surface_region_run        region-model grouped Hoare triple
       ├─ fifth_order_fwd_surface_exec_isSome  termination
       ├─ fifth_order_fwd_surface_frame        eleven-scatter cell frame
       ├─ fifth_order_fwd_surface_y0k_correct  algorithm-layer readback, channel k = 0..10
       └─ y0kSpec_eq_sphY                      loaded-value spec = pure polynomial `sphY k`

fifth_order_fwd_surface_y0k_compute_correct    ← per-channel `Realizes` family (k = 0..10)
  └─ fifth_order_fwd_surface_y0k_correct
```

The headline is stated on the kernel's **grouped IO signature**
`fifthOrderFwdIO` (`GroupedMasked2DKernelIO`): three float input channels
(`x`/`y`/`z`, all reading one coordinate buffer at strided offsets `3j`,
`3j+1`, `3j+2`) and **eleven** output channels (`Y00..Y10`, all writing one
output buffer at strided offsets `col_offset + k`), with `bufs` the decoupled
two-buffer allocation list. `⊨` (`GroupedMasked2DKernelIO.Implements`) is the
audit-once grouped Hoare-triple combinator: for **every** disjoint flat
placement of the two buffers, **every** program id whose active lanes are in
bounds, and **every** launch state whose read-active lanes hold `xs`, the
translated pointer kernel terminates, every write-active lane of every one of
the eleven channels holds its polynomial value, and every other memory cell is
unchanged. Because the channels are indexed rather than named, the frame is a
**single channel-quantified leg**, not eleven conjuncts.

The eleven `fifth_order_fwd_surface_y0{0..10}_compute_correct` theorems remain
as the per-channel compute-facing `Realizes` family (now subsumed by the
headline); `fifth_order_fwd_y00_compute_correct` is the proof-oriented
single-channel projection.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. The kernel is purely elementwise (no recurrence): each
output is a polynomial in the loaded `(x, y, z)` lane values, modeled with
`Option.map₂` over the loaded coordinates so out-of-range loads propagate.
The eleven forward channels `Y00..Y10` are each verified; the backward kernel
`fifth_order_bwd` is transcribed, but the ~30-constant-per-dimension
gradient polynomials are **not** verified (left for future work, stated
honestly in the backward-section doc). Side conditions of the headline: the stride bound
`hStride : 10 < output_stride` (the eleven channel columns fit inside one
output row, and the per-lane store offsets are injective — `hOutInj` is
*derived* from it, not assumed), and `hCover`, the host-layout coupling
"a lane whose `Y00` store is in range has its three coordinates in range"
(satisfied by the `calculate_lastdim_num_blocks` launch, where both extents
count the same rows). Without `hCover` a store-active but load-inactive lane
would write an `undef`-derived value that no spec over the pinned inputs can
name. The offset-disjointness helpers (`y0k_offset_disjoint` /
`y0jk_offset_disjoint`) show the eleven channel columns are disjoint.
-/

namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics

open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ### The `⊨` specification -/

/-- The eleven real fifth-order spherical-harmonic polynomials, as pure
functions of one lane's coordinates `(x, y, z)`. Output channel `o` of the
kernel writes `sphY o`; these are exactly the `Y00..Y10` expressions of
`fifth_order_sph_harmonics.py`, with the shared monomial intermediates
(`VAR05 = x⁵`, `VAR26 = z²`, …) inlined. -/
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

/-- Input channel `i`'s lane-`j` load address for program `pid₀`: the three
coordinate reads share the row window `3j + 3·block_size·pid₀` and differ only
in the component column `+ i` (`x`, `y`, `z`). -/
def sphInWin (block_size pid₀ : Nat) (i : Fin 3) (j : Fin block_size) : Nat :=
  match i with
  | ⟨0, _⟩ => j.val * 3 + block_size * 3 * pid₀
  | ⟨1, _⟩ => j.val * 3 + block_size * 3 * pid₀ + 1
  | ⟨_ + 2, _⟩ => j.val * 3 + block_size * 3 * pid₀ + 2

theorem sphInWin_eq (block_size pid₀ : Nat) (i : Fin 3) (j : Fin block_size) :
    sphInWin block_size pid₀ i j
      = j.val * 3 + block_size * 3 * pid₀ + i.val := by
  fin_cases i <;> rfl

/-- Output channel `o`'s lane-`j` store address for program `pid₀`: the eleven
`Y0k` stores share the row window
`j·output_stride + block_size·output_stride·pid₀ + col_offset` and differ only
in the channel column `+ k`. -/
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

theorem sphOutWin_eq (block_size col_offset output_stride pid₀ : Nat)
    (o : Fin 11) (j : Fin block_size) :
    sphOutWin block_size col_offset output_stride pid₀ o j
      = j.val * output_stride + block_size * output_stride * pid₀ + col_offset
        + o.val := by
  fin_cases o <;> rfl

/-- Channel `Y00`'s loaded-value spec is the pure polynomial `sphY 0` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y00Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y00Spec s coord_ptr block_size coord_numel i = sphY ⟨0, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y00Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y01`'s loaded-value spec is the pure polynomial `sphY 1` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y01Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y01Spec s coord_ptr block_size coord_numel i = sphY ⟨1, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y01Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y02`'s loaded-value spec is the pure polynomial `sphY 2` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y02Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y02Spec s coord_ptr block_size coord_numel i = sphY ⟨2, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y02Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y03`'s loaded-value spec is the pure polynomial `sphY 3` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y03Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y03Spec s coord_ptr block_size coord_numel i = sphY ⟨3, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y03Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y04`'s loaded-value spec is the pure polynomial `sphY 4` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y04Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y04Spec s coord_ptr block_size coord_numel i = sphY ⟨4, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y04Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y05`'s loaded-value spec is the pure polynomial `sphY 5` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y05Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y05Spec s coord_ptr block_size coord_numel i = sphY ⟨5, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y05Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y06`'s loaded-value spec is the pure polynomial `sphY 6` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y06Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y06Spec s coord_ptr block_size coord_numel i = sphY ⟨6, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y06Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y07`'s loaded-value spec is the pure polynomial `sphY 7` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y07Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y07Spec s coord_ptr block_size coord_numel i = sphY ⟨7, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y07Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y08`'s loaded-value spec is the pure polynomial `sphY 8` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y08Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y08Spec s coord_ptr block_size coord_numel i = sphY ⟨8, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y08Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y09`'s loaded-value spec is the pure polynomial `sphY 9` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y09Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y09Spec s coord_ptr block_size coord_numel i = sphY ⟨9, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y09Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- Channel `Y10`'s loaded-value spec is the pure polynomial `sphY 10` of the
lane's three coordinates, whenever the lane's coordinate window is in range
(so all three masked loads return memory, not `undef`). -/
theorem y10Spec_eq_sphY
    (s : BlockState) (coord_ptr : RegionName)
    (block_size coord_numel : Nat) (i : Fin block_size) (x y z : ℝ)
    (hbound : coordOffset s block_size i + 2 < coord_numel)
    (hxv : s.readMem coord_ptr (coordOffset s block_size i) = x)
    (hyv : s.readMem coord_ptr (coordOffset s block_size i + 1) = y)
    (hzv : s.readMem coord_ptr (coordOffset s block_size i + 2) = z) :
    y10Spec s coord_ptr block_size coord_numel i = sphY ⟨10, by decide⟩ x y z := by
  subst hxv; subst hyv; subst hzv
  have hx : coordOffset s block_size i < coord_numel := by omega
  have hy : coordOffset s block_size i + 1 < coord_numel := by omega
  have ex : coordX s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i)) := by
    simp [coordX, hx]
  have ey : coordY s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 1)) := by
    simp [coordY, hy]
  have ez : coordZ s coord_ptr block_size coord_numel i
      = some (s.readMem coord_ptr (coordOffset s block_size i + 2)) := by
    simp [coordZ, hbound]
  simp only [y10Spec, sphY, ex, ey, ez, Option.map₂, Option.map, WithBot.unbotD,
    Option.bind, Option.bind_some, WithBot.recBotCoe, id]
  ring

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the eleven masked stores). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- The kernel sits inside the flat-memory bridge's covered fragment: pointer
arithmetic, three masked coordinate loads, elementwise polynomial arithmetic,
and eleven masked strided stores. -/
theorem fifth_order_fwd_surface_flattenOk
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) :
    ((fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [fifth_order_fwd_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Termination: the straight-line elementwise body executes to completion from
any state (no loops, no data-dependent control flow). -/
theorem fifth_order_fwd_surface_exec_isSome
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s : BlockState) :
    ∃ s1, exec ((fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride).toAlgKernel) s = some s1 := by
  simp [exec, fifth_order_fwd_surface, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the three masked coordinate loads and the eleven
masked strided stores address the windows `sphInWin` / `sphOutWin`, active only
under the kernel's own `< coord_numel` / `< output_numel` masks, so the bounds
contract is **lane-wise and channel-wise** — every active lane's address is
below the region bound of the buffer it touches. -/
theorem fifth_order_fwd_surface_traceSafe
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ (i : Fin 3) (j : Fin block_size),
      sphInWin block_size (s.pids 0) i j < coord_numel →
      sphInWin block_size (s.pids 0) i j < bounds coord_ptr)
    (hout : ∀ (o : Fin 11) (j : Fin block_size),
      sphOutWin block_size col_offset output_stride (s.pids 0) o j < output_numel →
      sphOutWin block_size col_offset output_stride (s.pids 0) o j < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride).toAlgKernel) s := by
  simp only [sphInWin_eq, sphOutWin_eq] at hin hout
  unfold Kernel.TraceSafe
  simp (maxSteps := 2000000) [fifth_order_fwd_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, tile_elementwise,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt]
  exact ⟨fun a ha => hin ⟨0, by decide⟩ a ha,
    fun a ha => hin ⟨1, by decide⟩ a ha,
    fun a ha => hin ⟨2, by decide⟩ a ha,
    fun a ha => hout ⟨0, by decide⟩ a ha,
    fun a ha => hout ⟨1, by decide⟩ a ha,
    fun a ha => hout ⟨2, by decide⟩ a ha,
    fun a ha => hout ⟨3, by decide⟩ a ha,
    fun a ha => hout ⟨4, by decide⟩ a ha,
    fun a ha => hout ⟨5, by decide⟩ a ha,
    fun a ha => hout ⟨6, by decide⟩ a ha,
    fun a ha => hout ⟨7, by decide⟩ a ha,
    fun a ha => hout ⟨8, by decide⟩ a ha,
    fun a ha => hout ⟨9, by decide⟩ a ha,
    fun a ha => hout ⟨10, by decide⟩ a ha⟩

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell the eleven masked stores do not actively hit —
every cell of every region other than `output_ptr`, and the inactive lanes of
the eleven channel windows themselves — is preserved by the run. One
channel-quantified exclusion condition covers all eleven stores. -/
theorem fifth_order_fwd_surface_frame
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (s s1 : BlockState)
    (hExec : exec ((fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
        output_numel col_offset output_stride).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ (c : Fin 11) (j : Fin block_size),
      j.val * output_stride + block_size * output_stride * s.pids 0 + col_offset
          + c.val < output_numel →
      ¬(output_ptr = r ∧ j.val * output_stride + block_size * output_stride * s.pids 0
          + col_offset + c.val = o)) :
    s1.mem r o = s.mem r o := by
  simp (maxSteps := 2000000) [exec, fifth_order_fwd_surface, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl))))))))))
  · intro k _ hmk hc; exact hmiss ⟨10, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨9, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨8, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨7, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨6, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨5, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨4, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨3, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨2, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨1, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)
  · intro k _ hmk hc; exact hmiss ⟨0, by decide⟩ k.1 (by simpa using hmk) (by simpa using hc)

/-- **The region-model grouped Hoare triple** — termination, write-active-lane
values of all eleven output channels, and frame off every write-active lane,
from any launch state whose read-active coordinate lanes hold `xs`. This is the
`hrun` obligation of the `⊨` headline; the value half reuses the eleven
`fifth_order_fwd_surface_y0k_correct` readbacks composed with the
`y0kSpec_eq_sphY` bridges, and the frame half is
`fifth_order_fwd_surface_frame`. -/
theorem fifth_order_fwd_surface_region_run
    (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat)
    (hStride : 10 < output_stride)
    (hCover : ∀ (pid₀ : Nat) (j : Fin block_size),
      j.val * output_stride + block_size * output_stride * pid₀ + col_offset
        < output_numel →
      j.val * 3 + block_size * 3 * pid₀ + 2 < coord_numel)
    (s₀ : BlockState) (xs : Fin 3 → Fin block_size → ℝ)
    (hpin : ∀ (i : Fin 3) (j : Fin block_size),
      sphInWin block_size (s₀.pids 0) i j < coord_numel →
      s₀.readMem coord_ptr (sphInWin block_size (s₀.pids 0) i j) = xs i j) :
    ∃ s1, exec ((fifth_order_fwd_surface coord_ptr output_ptr block_size coord_numel
          output_numel col_offset output_stride).toAlgKernel) s₀ = some s1
      ∧ (∀ (o : Fin 11) (j : Fin block_size),
          sphOutWin block_size col_offset output_stride (s₀.pids 0) o j < output_numel →
          s1.readMem output_ptr
              (sphOutWin block_size col_offset output_stride (s₀.pids 0) o j)
            = sphY o (xs ⟨0, by decide⟩ j) (xs ⟨1, by decide⟩ j) (xs ⟨2, by decide⟩ j))
      ∧ (∀ (r : RegionName) (o' : Nat),
          (∀ (oc : Fin 11) (j : Fin block_size),
            sphOutWin block_size col_offset output_stride (s₀.pids 0) oc j < output_numel →
            r ≠ output_ptr ∨
              o' ≠ sphOutWin block_size col_offset output_stride (s₀.pids 0) oc j) →
          s1.mem r o' = s₀.mem r o') := by
  simp only [sphInWin_eq, sphOutWin_eq] at hpin ⊢
  obtain ⟨s1, hs1⟩ := fifth_order_fwd_surface_exec_isSome coord_ptr output_ptr
    block_size coord_numel output_numel col_offset output_stride s₀
  have hOutInj : Function.Injective
      (fun i : Fin block_size => outOffset s₀ block_size col_offset output_stride i) := by
    intro a b hab
    simp only [outOffset] at hab
    have h : a.val * output_stride = b.val * output_stride := by omega
    exact Fin.ext (Nat.eq_of_mul_eq_mul_right (by omega) h)
  have hpin0 : ∀ j : Fin block_size,
      (j.val : Nat) * 3 + block_size * 3 * s₀.pids 0 < coord_numel →
      s₀.readMem coord_ptr ((j.val : Nat) * 3 + block_size * 3 * s₀.pids 0)
        = xs ⟨0, by decide⟩ j :=
    fun j hj => hpin ⟨0, by decide⟩ j hj
  have hpin1 : ∀ j : Fin block_size,
      (j.val : Nat) * 3 + block_size * 3 * s₀.pids 0 + 1 < coord_numel →
      s₀.readMem coord_ptr ((j.val : Nat) * 3 + block_size * 3 * s₀.pids 0 + 1)
        = xs ⟨1, by decide⟩ j :=
    fun j hj => hpin ⟨1, by decide⟩ j hj
  have hpin2 : ∀ j : Fin block_size,
      (j.val : Nat) * 3 + block_size * 3 * s₀.pids 0 + 2 < coord_numel →
      s₀.readMem coord_ptr ((j.val : Nat) * 3 + block_size * 3 * s₀.pids 0 + 2)
        = xs ⟨2, by decide⟩ j :=
    fun j hj => hpin ⟨2, by decide⟩ j hj
  refine ⟨s1, hs1, ?_, ?_⟩
  · rintro ⟨ov, hov⟩ j hact
    interval_cases ov
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y00Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y00_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y01Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y01_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 1
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y02Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y02_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 2
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y03Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y03_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 3
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y04Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y04_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 4
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y05Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y05_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 5
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y06Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y06_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 6
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y07Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y07_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 7
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y08Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y08_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 8
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y09Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y09_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 9
        < output_numel from hact), hb] at h
      exact h
    · have hc0 : (j.val : Nat) * output_stride + block_size * output_stride * s₀.pids 0
          + col_offset < output_numel := by omega
      have hcv := hCover (s₀.pids 0) j hc0
      have hb := y10Spec_eq_sphY s₀ coord_ptr block_size coord_numel j _ _ _
        (by simp only [coordOffset, BlockState.pid]; omega)
        (hpin0 j (by omega)) (hpin1 j (by omega)) (hpin2 j (by omega))
      have h := fifth_order_fwd_surface_y10_correct coord_ptr output_ptr block_size
        coord_numel output_numel col_offset output_stride s₀ s1 hStride hOutInj hs1 j
      rw [if_pos (show outOffset s₀ block_size col_offset output_stride j + 10
        < output_numel from hact), hb] at h
      exact h
  · intro r o' hno
    refine fifth_order_fwd_surface_frame coord_ptr output_ptr block_size coord_numel
      output_numel col_offset output_stride s₀ s1 hs1 r o' ?_
    intro c j hj hc
    rcases hno c j hj with hne | hne
    · exact hne hc.1.symm
    · exact hne hc.2.symm

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

set_option maxRecDepth 8000

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
          (xs ⟨2, by show 2 < 3; decide⟩ j) := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro _
    show output_ptr ∈ [coord_ptr, output_ptr]
    exact List.mem_cons_of_mem _ List.mem_cons_self
  · exact fifth_order_fwd_surface_flattenOk coord_ptr output_ptr block_size
      coord_numel output_numel col_offset output_stride
  · intro bounds s hin hout
    exact fifth_order_fwd_surface_traceSafe coord_ptr output_ptr block_size
      coord_numel output_numel col_offset output_stride bounds s hin hout
  · intro s₀ xs hpin
    exact fifth_order_fwd_surface_region_run coord_ptr output_ptr block_size
      coord_numel output_numel col_offset output_stride hStride hCover s₀ xs hpin

end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics
