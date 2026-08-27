# Spec sheet — `bench/tritonbench_g/l2_norm_bwd/L2NormBwd.lean`

**Python source:** `bench/tritonbench_g/l2_norm_bwd/l2_norm_bwd.py`

## Public theorem: `l2_norm_bwd_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_l2_norm_bwd_kernel` implements the oracle L2-norm
backward `l2BwdSpec` (i.e. `dx = dy·rstd − (Σ dy·x)·(1/(var+eps))·rstd·x` over
the zero-filled active prefix) on its masked IO signature — for every disjoint
flat placement of the three buffers, every program id whose active lanes are
in bounds, and every launch state whose active input-row lanes hold
`xs`/`dys`, the translated pointer kernel terminates, every active output-row
lane `j` holds `l2BwdSpec N BLOCK_N eps xs dys j`, and every other memory cell
is unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification l2_norm_bwd_kernel_correctness
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2BwdIO X DY DX stride_x_row N eps BLOCK_N
      ⊨ fun xs dys i => l2BwdSpec N BLOCK_N eps xs dys i
```

**Closed-form spec defs (transitive):** `l2BwdIO`, `l2BwdSpec`, `l2_norm_bwd_kernel`

<details><summary><code>l2BwdIO</code></summary>

```
/-- `_l2_norm_bwd_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: input `X`,
  upstream gradient `DY`, input gradient `DX`);
* `B = BLOCK_N` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` (the row) reads and writes its row
  at `pid * stride_x_row` in all three buffers (the host-side
  one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = next_power_of_2(N)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def l2BwdIO (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : MaskedKernelIO₂ where
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
/-- Oracle L2-norm backward value at lane `i`, as a **pure** function of the
active row prefixes `xs j` / `dys j`, `j < N` (masked lanes enter both
reductions as `0`, matching the kernel's `other=0.0` loads and
`tl.where(cols < N, ·, 0.0)` zero-fill). -/
```
```lean
noncomputable def l2BwdSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs dys : Fin BLOCK_N → ℝ) (i : Fin BLOCK_N) : ℝ :=
  l2NormBwd
    (fun j => if j.val < N then xs j else 0)
    (fun j => if j.val < N then dys j else 0)
    eps i
```
</details>

<details><summary><code>l2_norm_bwd_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_bwd.py`'s `_l2_norm_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` -> Lean `Nat` parameter. -/
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
