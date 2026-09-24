# Spec sheet — `bench/tritonbench_g/l2_norm_triton2/L2NormTriton2.lean`

**Python source:** `bench/tritonbench_g/l2_norm_triton2/l2_norm_triton2.py`

## Public theorem: `l2_norm_fwd_1pass_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The forward headline**: `_l2_norm_fwd_1pass_kernel` implements the exact
L2-normalization `l2FwdSpec` (the `Math.TiledL2Norm.l2Norm` oracle over the
zero-padded active prefix) on its masked IO signature — for every disjoint
flat placement of the two buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `j` holds
`l2FwdSpec N BLOCK_N eps xs j`, and every other memory cell is unchanged.
Proof: `Implements.intro` assembles the region-model masked triple with the
bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification l2_norm_fwd_1pass_kernel_correctness
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2FwdIO X Y stride_x_row N eps BLOCK_N ⊨
      fun xs i => l2FwdSpec N BLOCK_N eps xs i
```

**Closed-form spec defs (transitive):** `l2FwdIO`, `l2FwdSpec`, `l2_norm_fwd_1pass_kernel`

<details><summary><code>l2FwdIO</code></summary>

```
/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `pid` reads and writes its row at
  `pid * stride_x_row` (the host-side one-program-per-row launch convention;
  `Y` shares `X`'s row stride upstream);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = next_power_of_2(N)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def l2FwdIO (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₁ where
  kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N
  inp := X
  out := Y
  B := BLOCK_N
  read := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N
```
</details>

<details><summary><code>l2FwdSpec</code></summary>

```
/-- Exact L2-normalized value at lane `idx`, as a pure function of the active
row prefix `xs j`, `j < N`: masked lanes enter the oracle sum-of-squares as `0`
(matching `other=0.0` + `tl.where`). -/
```
```lean
noncomputable def l2FwdSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs : Fin BLOCK_N → ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (fun j => if j.val < N then xs j else 0) eps idx
```
</details>

<details><summary><code>l2_norm_fwd_1pass_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_fwd_1pass_kernel`. -/
```
```lean
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

## Public theorem: `l2_norm_bwd_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The backward headline**: `_l2_norm_bwd_kernel` implements the exact
L2-norm input gradient `l2BwdSpec` (the `Math.TiledL2Norm.l2NormBwd` oracle
over the zero-padded active prefixes) on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active `X`/`DY` row lanes
hold `xs`/`dys`, the translated pointer kernel terminates, every active
output-row lane `j` holds `l2BwdSpec N BLOCK_N eps xs dys j`, and every other
memory cell is unchanged. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification l2_norm_bwd_kernel_correctness
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2BwdIO X DY DX stride_x_row N eps BLOCK_N ⊨
      fun xs dys i => l2BwdSpec N BLOCK_N eps xs dys i
```

**Closed-form spec defs (transitive):** `l2BwdIO`, `l2BwdSpec`, `l2_norm_bwd_kernel`

<details><summary><code>l2BwdIO</code></summary>

```
/-- `_l2_norm_bwd_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: `X`, `DY`
  in, `DX` out);
* `B = BLOCK_N` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` reads and writes all three rows at
  `pid * stride_x_row` (the host-side one-program-per-row launch convention;
  `DY`/`DX` share `X`'s row stride upstream);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes carry no
  obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
```
```lean
def l2BwdIO (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₂ where
  kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N
  in1 := X
  in2 := DY
  out := DX
  B := BLOCK_N
  read1 := fun pid => pid * stride_x_row
  read2 := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N
```
</details>

<details><summary><code>l2BwdSpec</code></summary>

```
/-- Exact L2-norm input gradient at lane `idx`, as a pure function of the
active row prefixes `xs j` / `dys j`, `j < N`: masked lanes enter the oracle
sum-of-squares and dot product as `0`. -/
```
```lean
noncomputable def l2BwdSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs dys : Fin BLOCK_N → ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2NormBwd
    (fun j => if j.val < N then xs j else 0)
    (fun j => if j.val < N then dys j else 0)
    eps idx
```
</details>

<details><summary><code>l2_norm_bwd_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_bwd_kernel`. -/
```
```lean
def l2_norm_bwd_kernel
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  DX += row * $(stride_x_row)
  DY += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  dy = tl.load(DY + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + $(eps))) * rstd * x
  tl.store(DX + cols, dx, mask=mask)
}
```
</details>
