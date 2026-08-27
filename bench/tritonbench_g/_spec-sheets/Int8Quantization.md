# Spec sheet — `bench/tritonbench_g/int8_quantization/Int8Quantization.lean`

**Python source:** `bench/tritonbench_g/int8_quantization/int8_quantization.py`

## Public theorem: `per_block_int8_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: the per-block int8 quantization kernel implements the
exact quantized-value / scale pair on its masked two-output IO signature. For
every disjoint flat placement of the four buffers, every program id pair
`(off_blk, off_b)` whose active lanes and scalar scale cell are in bounds, and
every launch state whose active `X` lanes hold `xs` and whose `ScalePre` cell
holds `ys`, the translated pointer kernel terminates, every active lane `j` of
the `[BLK, C]` block holds `perBlockInt8ValSpec … = (preScale · xs j) / ys j`,
the cell `Scale[off_b · scale_stride + off_blk]` holds the scale `ys`, and every
other memory cell is unchanged.

`preScale` is symbolic, so this one theorem covers **both** Python kernels:
`preScale = C**-0.5 · 1.44269504` is `q_kernel_per_block_int8` and
`preScale = 1` is `k_kernel_per_block_int8`.

Side conditions, all genuinely forced: `0 < BLK * C` (the `Scale` store is
unmasked in the kernel, so its safety bound and single-cell frame exclusion are
carried by the lane-`0` gate `writeMask2`, which needs a lane) and
`XInt8 ≠ Scale` (that unmasked scalar store must not alias the masked block
store). No separate `0 < C` is needed: the `Lane2D` row-major bijection
`j ↦ (j / C, j % C)` gets its positivity from the lane itself. The per-block
scale is an
**input**, not a computed value — see the module docstring: the Python
reduction reads uninitialized memory at a partial tail block, so it has no pure
spec. Proof: `Masked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification per_block_int8_correctness
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (hB : 0 < BLK * C) (hRegions : XInt8 ≠ Scale) :
    perBlockInt8IO X ScalePre XInt8 Scale L C BLK scale_stride preScale ⊨
      fun _ _ xs ys =>
        (fun j => perBlockInt8ValSpec BLK C preScale xs ys j, fun j => ys j)
```

**Assumptions / layout contracts:**
- `hB : 0 < BLK * C`
- `hRegions : XInt8 ≠ Scale`

**Closed-form spec defs (transitive):** `perBlockInt8IO`, `perBlockInt8ValSpec`, `per_block_int8_store_slice`

<details><summary><code>perBlockInt8IO</code></summary>

```
/-- The verified kernel's masked two-output **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out1`/`out2` — the block buffer `X`, the per-block scale input
  `ScalePre`, the quantized block buffer `XInt8`, and the scale vector `Scale`;
* `B = BLK * C` — the block window each program owns, indexed by the flat
  **row-major lane** `j ↦ (j / C, j % C)` of the kernel's `[BLK, C]` tile;
* `read1`/`write1` — program `(off_blk, off_b)` reads and writes its block at
  `off_b · L · C + off_blk · BLK · C + j` (the Python row-major addressing
  `x_offset + offs_m[:, None] · C + offs_k[None, :]`);
* `read2`/`write2` — the **scalar** cell `off_b · scale_stride + off_blk`, the
  same for every lane;
* `mask` — the active lanes `off_blk · BLK + j / C < L`, i.e. the Python row
  mask `offs_m[:, None] < L`; the load mask and the block-store mask coincide,
  so `writeMask1` keeps its `mask` default;
* `read2Mask` — `True`: the `ScalePre` load is unmasked;
* `writeMask2` — lane `0` carries the scalar scale; the other lanes are
  write-inactive.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
```
```lean
def perBlockInt8IO (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) : Masked2DKernelIO₂ₓ₂ where
  kernel := per_block_int8_store_slice X ScalePre XInt8 Scale L C BLK
    scale_stride preScale
  in1 := X
  in2 := ScalePre
  out1 := XInt8
  out2 := Scale
  B := BLK * C
  read1 := fun off_blk off_b j => off_b * L * C + off_blk * BLK * C + j.val
  read2 := fun off_blk off_b _ => off_b * scale_stride + off_blk
  write1 := fun off_blk off_b j => off_b * L * C + off_blk * BLK * C + j.val
  write2 := fun off_blk off_b _ => off_b * scale_stride + off_blk
  mask := fun off_blk _ j => off_blk * BLK + j.val / C < L
  read2Mask := fun _ _ _ => True
  writeMask2 := fun _ _ j => j.val = 0
```
</details>

<details><summary><code>perBlockInt8ValSpec</code></summary>

```
/-- The value written to active lane `j`: `(preScale · x j) / scale`. The
second channel `ys` is the per-block scale, read from `ScalePre` at the single
address `off_b · scale_stride + off_blk` — every lane reads the same cell, so
`ys` is constant. -/
```
```lean
noncomputable def perBlockInt8ValSpec (BLK C : Nat) (preScale : ℝ)
    (xs ys : Fin (BLK * C) → ℝ) (j : Fin (BLK * C)) : ℝ :=
  preScale * xs j / ys j
```
</details>

<details><summary><code>per_block_int8_store_slice</code></summary>

```
/-- **The verified kernel.** `q_kernel_per_block_int8` /
`k_kernel_per_block_int8` with the per-block scale taken from a separate input
buffer `ScalePre` (see the module docstring for why this is forced) and the
half-ULP rounding bias / int8 cast dropped. The block addressing, the row mask,
the value expression `(preScale · x) / scale`, the masked block store and the
unmasked scalar scale store are transcribed 1:1. `preScale` is
`C**-0.5 · 1.44269504` for the Q kernel and `1` for the K kernel. -/
```
```lean
def per_block_int8_store_slice
    (X ScalePre XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale = tl.load(ScalePre + off_b * $(scale_stride) + off_blk)
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_int8 = ($(preScale) * x) / scale
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}
```
</details>
