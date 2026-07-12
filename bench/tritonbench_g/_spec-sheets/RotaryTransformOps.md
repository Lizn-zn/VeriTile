# Spec sheet — `bench/tritonbench_g/rotary_transform_ops/RotaryTransformOps.lean`

**Python source:** `bench/tritonbench_g/rotary_transform_ops/rotary_transform_ops.py`

## Public theorem: `rotary_transform_ops_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general public output summary for `rotary_transform_ops.py`**
(genuine, not self-referential).

Every sequence length, rotary
dimension, block size, and stride is a `Nat` parameter rather than a pinned
Python literal, and the per-lane output-offset injectivity plus the
`stride_out_headdim ≠ 0` / `BLOCK_HALF ≤ rotary_dim_half` disjointness
side-conditions are taken as hypotheses.

For ANY shape, the full `rotary_kernel_surface` (with all four prologue
branches and the conjugate/interleaved flags) lowers to the algorithm layer,
and on the non-interleaved rotation body run on a `[BLOCK_M, BLOCK_HALF]` row
tile BOTH output halves are written so that every active lane of the
first-half (`o0`) store equals the genuine closed form `x0·cos − x1·sin`
(`rotaryO0Spec`) and every active lane of the second-half (`o1`) store equals
`x0·sin + x1·cos` (`rotaryO1Spec`) — the actual rotary embedding read from the
precomputed `COS`/`SIN` cache, NOT the kernel's own re-executed value.

The host launch remains the trusted boundary. -/
```
</details>

**Statement:**
```lean
specification rotary_transform_ops_output_summary_general
    (OUT X COS SIN : RegionName) (CU_SEQLENS SEQLEN_OFFSETS : Region .nat)
    (SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool)
    (body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
      body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
      body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
      body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s body_stride_out_batch body_stride_out_seqlen
          body_stride_out_nheads body_stride_out_headdim body_BLOCK_M i))
    (hOut1Inj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s body_stride_out_batch body_stride_out_seqlen
          body_stride_out_nheads body_stride_out_headdim body_rotary_dim_half
          body_BLOCK_M i))
    (hStrideHd : body_stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ body_rotary_dim_half) :
    (∃ alg, (rotary_kernel_surface OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS
      SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch
      stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch
      stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M
      IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN
        body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
        body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
        body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
        body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF =>
          active s body_seqlen body_rotary_dim_half body_BLOCK_M i)
        (fun i => (OUT,
          outOffset s body_stride_out_batch body_stride_out_seqlen
            body_stride_out_nheads body_stride_out_headdim body_BLOCK_M i)))
      (expected := fun i =>
        rotaryO0Spec s X COS SIN body_SEQLEN_OFFSETS body_seqlen_ro
          body_stride_x_batch body_stride_x_seqlen body_stride_x_nheads
          body_stride_x_headdim body_rotary_dim_half body_BLOCK_M i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN
        body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
        body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
        body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
        body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF =>
          active s body_seqlen body_rotary_dim_half body_BLOCK_M i)
        (fun i => (OUT,
          out1Offset s body_stride_out_batch body_stride_out_seqlen
            body_stride_out_nheads body_stride_out_headdim body_rotary_dim_half
            body_BLOCK_M i)))
      (expected := fun i =>
        rotaryO1Spec s X COS SIN body_SEQLEN_OFFSETS body_seqlen_ro
          body_stride_x_batch body_stride_x_seqlen body_stride_x_nheads
          body_stride_x_headdim body_rotary_dim_half body_BLOCK_M i))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s body_stride_out_batch body_stride_out_seqlen
          body_stride_out_nheads body_stride_out_headdim body_BLOCK_M i)`
- `hOut1Inj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s body_stride_out_batch body_stride_out_seqlen
          body_stride_out_nheads body_stride_out_headdim body_rotary_dim_half
          body_BLOCK_M i)`
- `hStrideHd : body_stride_out_headdim ≠ 0`
- `hHalfBound : BLOCK_HALF ≤ body_rotary_dim_half`
- `fun i : Fin BLOCK_HALF =>
          active s body_seqlen body_rotary_dim_half body_BLOCK_M i`
- `fun i : Fin BLOCK_HALF =>
          active s body_seqlen body_rotary_dim_half body_BLOCK_M i`

**Closed-form spec defs (transitive):** `outOffset`, `out1Offset`, `rotary_kernel_surface`, `rotary_kernel_o0o1_row`, `active`, `rotaryO0Spec`, `rotaryO1Spec`, `rowIndex`, `dimIndex`, `rotOffset`, `x0Offset`, `x1Offset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    rowIndex s BLOCK_M * stride_out_seqlen + dimIndex i * stride_out_headdim
```
</details>

<details><summary><code>out1Offset</code></summary>

```lean
def out1Offset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    rowIndex s BLOCK_M * stride_out_seqlen +
    (dimIndex i + rotary_dim_half) * stride_out_headdim
```
</details>

<details><summary><code>rotary_kernel_surface</code></summary>

```
/-- Faithful DSL port of `rotary_transform_ops.py`'s `rotary_kernel`.

Python's `SEQLEN_OFFSETS` argument is a union of scalar offset and tensor
pointer. The surface keeps `SEQLEN_OFFSETS` as the tensor region used by the
tensor-offset path and uses `SEQLEN_OFFSETS_SCALAR` for the scalar-offset path. -/
```
```lean
def rotary_kernel_surface
    (OUT X COS SIN : RegionName) (CU_SEQLENS SEQLEN_OFFSETS : Region .nat)
    (SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
  rotary_dim_half = rotary_dim // $(2)

  if not IS_VARLEN {
    X = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
    OUT = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
  } else {
    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + $(1)) - start_idx
    X = X + start_idx * $(stride_x_seqlen) + pid_head * $(stride_x_nheads)
    OUT = OUT + start_idx * $(stride_out_seqlen) + pid_head * $(stride_out_nheads)
  }

  if pid_m * $(BLOCK_M) >= seqlen {
    return
  }
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  if not IS_SEQLEN_OFFSETS_TENSOR {
    rm_cs = rm + $(SEQLEN_OFFSETS_SCALAR)
  } else {
    rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
  }
  rk = tl.arange(0, $(BLOCK_K))
  rk_half = tl.arange(0, $(BLOCK_K) // $(2))

  if not INTERLEAVED {
    X = X + (rm[:, None] * $(stride_x_seqlen) +
      rk_half[None, :] * $(stride_x_headdim))
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    cos = tl.load(COS,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_half[None, :] < rotary_dim_half),
      other=1.0).to(tl.float32)
    sin = tl.load(SIN,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x0 = tl.load(X,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x1 = tl.load(X + rotary_dim_half * $(stride_x_headdim),
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    if CONJUGATE {
      sin = -sin
    }
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    OUT = OUT + (rm[:, None] * $(stride_out_seqlen) +
      rk_half[None, :] * $(stride_out_headdim))
    tl.store(OUT, o0,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    tl.store(OUT + rotary_dim_half * $(stride_out_headdim), o1,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
  } else {
    rk_swap = rk + ((rk + $(1)) % $(2)) * $(2) - $(1)
    rk_repeat = tl.arange(0, $(BLOCK_K)) // $(2)
    X0 = X + (rm[:, None] * $(stride_x_seqlen) + rk[None, :] * $(stride_x_headdim))
    X1 = X + (rm[:, None] * $(stride_x_seqlen) + rk_swap[None, :] * $(stride_x_headdim))
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
    cos = tl.load(COS,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_repeat[None, :] < rotary_dim_half),
      other=1.0).to(tl.float32)
    sin = tl.load(SIN,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_repeat[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x0 = tl.load(X0,
      mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim),
      other=0.0).to(tl.float32)
    x1 = tl.load(X1,
      mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim),
      other=0.0).to(tl.float32)
    if CONJUGATE {
      sin = -sin
    }
    x0_cos = x0 * cos
    x1_sin = x1 * sin
    out = tl.where(rk[None, :] % $(2) == $(0), x0_cos - x1_sin, x0_cos + x1_sin)
    OUT = OUT + (rm[:, None] * $(stride_out_seqlen) + rk[None, :] * $(stride_out_headdim))
    tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))
  }
}
```
</details>

<details><summary><code>rotary_kernel_o0o1_row</code></summary>

```
/-- Combined one-row first-and-second-half slice of
`rotary_transform_ops.py`'s `rotary_kernel`.

Performs BOTH the `o0 = x0 * cos - x1 * sin` first-half store and the
`o1 = x0 * sin + x1 * cos` second-half store in a single kernel, matching
the non-varlen, non-interleaved, non-conjugate branch of the Python source. -/
```
```lean
def rotary_kernel_o0o1_row
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
  rm = pid_m * $(BLOCK_M)
  rm_cs = rm + $(SEQLEN_OFFSETS)
  rk_half = tl.arange(0, $(BLOCK_HALF))
  x_base = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
  out_base = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
  cos = tl.load(COS + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=1.0)
  sin = tl.load(SIN + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=0.0)
  x0 = tl.load(x_base + rm * $(stride_x_seqlen) + rk_half * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0)
  x1 = tl.load(x_base + rm * $(stride_x_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0)
  o0 = x0 * cos - x1 * sin
  o1 = x0 * sin + x1 * cos
  tl.store(out_base + rm * $(stride_out_seqlen) + rk_half * $(stride_out_headdim),
    o0, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
  tl.store(out_base + rm * $(stride_out_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_out_headdim),
    o1, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Prop :=
  rowIndex s BLOCK_M < seqlen ∧ dimIndex i < rotary_dim_half
```
</details>

<details><summary><code>rotaryO0Spec</code></summary>

```lean
noncomputable def rotaryO0Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem COS (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem SIN (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (x0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    cosVal -
  s.readMem X
      (x1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    sinVal
```
</details>

<details><summary><code>rotaryO1Spec</code></summary>

```lean
noncomputable def rotaryO1Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem COS (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem SIN (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (x0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    sinVal +
  s.readMem X
      (x1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    cosVal
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) : Nat :=
  s.pids 0 * BLOCK_M
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val
```
</details>

<details><summary><code>rotOffset</code></summary>

```lean
def rotOffset (s : BlockState) (SEQLEN_OFFSETS rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  (rowIndex s BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + dimIndex i
```
</details>

<details><summary><code>x0Offset</code></summary>

```lean
def x0Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    rowIndex s BLOCK_M * stride_x_seqlen + dimIndex i * stride_x_headdim
```
</details>

<details><summary><code>x1Offset</code></summary>

```lean
def x1Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    rowIndex s BLOCK_M * stride_x_seqlen +
    (dimIndex i + rotary_dim_half) * stride_x_headdim
```
</details>

## Also present (pinned special-case summaries)
- `rotary_kernel_o0o1_row_o0_compute_correct`
- `rotary_kernel_o0o1_row_o1_compute_correct`
- `rotary_kernel_o0o1_row_all_outputs_compute_correct`
