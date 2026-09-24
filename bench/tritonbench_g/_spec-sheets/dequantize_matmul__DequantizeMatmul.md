# Spec sheet — `bench/tritonbench_g/dequantize_matmul/DequantizeMatmul.lean`

**Python source:** `bench/tritonbench_g/dequantize_matmul/dequantize_matmul.py`

## Public theorem: `dequantize_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `dequantize_kernel` implements the elementwise
dequantization `int_b * scale_b` on its masked IO signature — for every
disjoint flat placement of the three buffers, every program-id pair whose
gated lanes are in bounds, and every launch state whose weight window holds
`xs` at the full mask and whose column-scale window holds `ys` at the wider
column gate, the translated pointer kernel terminates, every write-active lane
`j` of `fp_b` holds `xs j * ys j`, and every other memory cell — including the
masked-off lanes of the output window — is unchanged.

Two honest, **truth-required** side hypotheses: `hOutInj` (distinct lanes of a
program's output tile must not alias the same `fp_b` cell — otherwise only the
last store survives) and `0 < BLOCK_SIZE_K` (with an empty K block the lane set
is empty, so the signature can state no bounds hypothesis at all for the
`[1, BLOCK_SIZE_N]`-shaped column-scale load, which the kernel performs
regardless). -/
```
</details>

**Statement:**
```lean
specification dequantize_kernel_correctness
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (hBK : 0 < BLOCK_SIZE_K)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N) =>
        fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
          (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j))) :
    dequantizeIO b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      ⊨ fun _ _ xs ys j => xs j * ys j
```

**Assumptions / layout contracts:**
- `hBK : 0 < BLOCK_SIZE_K`

**Closed-form spec defs (transitive):** `fpbOffset`, `laneIdx`, `dequantizeIO`, `dequantize_kernel`, `bOffset`, `nOffset`, `dequantizeActive`

<details><summary><code>fpbOffset</code></summary>

```
/-- Output write address of tile cell `idx` for program `(p₀, p₁)`. -/
```
```lean
def fpbOffset (p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (p₀ * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (p₁ * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
```
</details>

<details><summary><code>laneIdx</code></summary>

```
/-- Lane `j` of the row-major flattening of an `R × C` tile. -/
```
```lean
def laneIdx (R C : Nat) (j : Fin (R * C)) : TileIndex [R, C] :=
  (⟨j.val / C, Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm R C)))⟩,
   ⟨j.val % C, Nat.mod_lt _ (Nat.pos_of_ne_zero (by
      rintro rfl
      exact absurd j.isLt (by simp)))⟩,
   PUnit.unit)
```
</details>

<details><summary><code>dequantizeIO</code></summary>

```
/-- `dequantize_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (weight `b_ptr`, column
  scale `b_scale_ptr`, dequantized output `fpb_ptr`);
* `B = BLOCK_SIZE_K * BLOCK_SIZE_N` — the tile flattened row-major into lanes,
  lane `j` = tile cell `(j / BSN, j % BSN)` (`laneIdx`);
* `read1`/`write` — the strided weight / output windows at row
  `pid₀ * BSK + i` and column `pid₁ * BSN + j`;
* `read2` — the column-scale window `pid₁ * BSN + j` (row-independent: the
  scale is broadcast down the K axis);
* `mask` — the store gate `pid₀ * BSK + i < K ∧ pid₁ * BSN + j < N`, which is
  also the weight-load gate; `writeMask` keeps the struct default;
* `read2Mask` — the **wider** column-only gate `pid₁ * BSN + j < N`, exactly
  the kernel's `n_mask`.

The windows and gates are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
```
```lean
def dequantizeIO (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    Masked2DKernelIO₂ where
  kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
    stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
  in1 := b_ptr
  in2 := b_scale_ptr
  out := fpb_ptr
  B := BLOCK_SIZE_K * BLOCK_SIZE_N
  read1 := fun p₀ p₁ j =>
    bOffset p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  read2 := fun _ p₁ j =>
    nOffset p₁ BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1
  write := fun p₀ p₁ j =>
    fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  mask := fun p₀ p₁ j =>
    dequantizeActive p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  read2Mask := fun _ p₁ j =>
    nOffset p₁ BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1 < N
```
</details>

<details><summary><code>dequantize_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
```
```lean
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}
```
</details>

<details><summary><code>bOffset</code></summary>

```
/-- Weight-tile read address of tile cell `idx` for program `(p₀, p₁)`. -/
```
```lean
def bOffset (p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (p₀ * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (p₁ * BLOCK_SIZE_N + idx.2.1.val) * stride_bn
```
</details>

<details><summary><code>nOffset</code></summary>

```
/-- Column-scale read address of tile column `j` for program `(_, p₁)`. -/
```
```lean
def nOffset (p₁ BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  p₁ * BLOCK_SIZE_N + j.val
```
</details>

<details><summary><code>dequantizeActive</code></summary>

```
/-- The kernel's store gate `mask` = `(k·BSK + i < K) & (n·BSN + j < N)`. -/
```
```lean
def dequantizeActive (p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  p₀ * BLOCK_SIZE_K + idx.1.val < K ∧ p₁ * BLOCK_SIZE_N + idx.2.1.val < N
```
</details>
