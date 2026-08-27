# Spec sheet — `bench/tritonbench_g/l2_norm_triton1/L2NormTriton1.lean`

**Python source:** `bench/tritonbench_g/l2_norm_triton1/l2_norm_triton1.py`

## Public theorem: `l2_norm_fwd_1pass_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_l2_norm_fwd_1pass_kernel` implements the exact L2
normalization over the active row prefix on its masked IO signature — for
every disjoint flat placement of the two buffers, every program id whose
active lanes are in bounds, and every launch state whose active input-row
lanes hold `xs`, the translated pointer kernel terminates, every active
output-row lane `j` holds `l2NormSpec N BLOCK_N eps xs j` (the `Math.*`
oracle `l2Norm` over the masked row), and every other memory cell is
unchanged. No `0 < BLOCK_N` side condition: the kernel's only reduction is a
`sum`, total on empty tiles. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification l2_norm_fwd_1pass_kernel_correctness
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2NormIO X Y stride_x_row N eps BLOCK_N ⊨
      fun xs i => l2NormSpec N BLOCK_N eps xs i
```

**Closed-form spec defs (transitive):** `l2NormIO`, `l2NormSpec`, `l2_norm_fwd_1pass_kernel`

<details><summary><code>l2NormIO</code></summary>

```
/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `row` reads its row of `X` and writes its row of
  `Y`, both at `row * stride_x_row` (the host passes the same row stride for
  both buffers; the one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))`) carry no obligations on
  either side. The store mask equals the load mask, so `writeMask` keeps its
  default.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def l2NormIO (X Y : RegionName)
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

<details><summary><code>l2NormSpec</code></summary>

```
/-- Exact L2-normalization value computed by the kernel at lane `idx`, as a
pure function of the active row prefix `xs j`, `j < N`: the `Math.*` oracle
`l2Norm` over the masked row (lanes `≥ N` enter the sum-of-squares as `0`,
matching `mask=cols < N, other=0.0`). -/
```
```lean
noncomputable def l2NormSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs : Fin BLOCK_N → ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (fun j : Fin BLOCK_N => if j.val < N then xs j else 0) eps idx
```
</details>

<details><summary><code>l2_norm_fwd_1pass_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
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
