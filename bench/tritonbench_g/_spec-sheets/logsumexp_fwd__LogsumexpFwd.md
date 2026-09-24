# Spec sheet — `bench/tritonbench_g/logsumexp_fwd/LogsumexpFwd.lean`

**Python source:** `bench/tritonbench_g/logsumexp_fwd/logsumexp_fwd.py`

## Public theorem: `logsumexp_fwd_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `logsumexp_fwd_kernel` implements the per-block
log-sum-exp on its masked two-axis IO signature — for every disjoint flat
placement of the two buffers, every program `(i_n, i_d)` whose active lanes
and store cell are in bounds, and every launch state whose active input lanes
hold `xs`, the translated pointer kernel terminates, the scalar cell
`z[i_n·cdiv(D,B) + i_d]` holds `blockLSE_local D B i_d HAS_SCALE scale xs`
(the exact LSE over the block's active lanes) when the block meets the row
(`i_d·B < D`), and the `⊥`-store fallback `0` otherwise (programs outside the
real launch grid), and every other memory cell is unchanged. `0 < B` is
required: the kernel's `max` reduce (like `Finset.sup'`) is only defined on
non-empty tiles. Proof: `Implements.intro` assembles the region-model masked
triple with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification logsumexp_fwd_kernel_correctness
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (hB : 0 < B) :
    logsumexpIO x z D B HAS_SCALE scale ⊨
      fun _ i_d xs _ =>
        if i_d * B < D then blockLSE_local D B i_d HAS_SCALE scale xs else 0
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `logsumexpIO`, `blockLSE_local`, `logsumexp_fwd_kernel`

<details><summary><code>logsumexpIO</code></summary>

```
/-- `logsumexp_fwd_kernel`'s masked two-axis **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B` — the block window each program owns;
* `read` — program `(i_n, i_d)`'s lane `j` reads `x[i_n·D + i_d·B + j]`
  (row-major rows of length `D`, `B`-sized blocks along the row);
* `write` — the **scalar** store cell `z[i_n·cdiv(D,B) + i_d]`, the same for
  every lane;
* `mask` — the active read lanes `i_d·B + j < D`: the part of the block that
  actually lies inside the row;
* `writeMask` — lane `0` carries the scalar; the other lanes are
  write-inactive and carry no obligations on either side.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def logsumexpIO (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    Masked2DKernelIO₁ where
  kernel := logsumexp_fwd_kernel x z D B HAS_SCALE scale
  inp := x
  out := z
  B := B
  read := fun i_n i_d j => i_n * D + (i_d * B + j.val)
  write := fun i_n i_d _ => i_n * ((D + B - 1) / B) + i_d
  mask := fun _ i_d j => i_d * B + j.val < D
  writeMask := fun _ _ j => j.val = 0
```
</details>

<details><summary><code>blockLSE_local</code></summary>

```
/-- Exact log-sum-exp over the **active lanes** of one `B`-sized block: lane
`j` of block `i_d` is active when `i_d * B + j < D` (the kernel's load mask).
This is the pure, block-local form of the tiled `blockLSE`; the bridge below
proves the two agree on loaded blocks. -/
```
```lean
noncomputable def blockLSE_local (D B i_d : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => i_d * B + i.val < D),
    Real.exp (if HAS_SCALE then xs i * scale else xs i))
```
</details>

<details><summary><code>logsumexp_fwd_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `logsumexp_fwd.py`'s `logsumexp_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `D: tl.constexpr` / `B: tl.constexpr` / `HAS_SCALE: tl.constexpr` →
  Lean parameters; the `tl.constexpr` annotation is implicit on Lean params.
- Python `if cond: body` → `tl.if cond { body }`, the DSL-side gate equivalent.
- `scale` (Lean `ℝ` parameter) injected via `$(...)`. -/
```
```lean
def logsumexp_fwd_kernel
    (x z : RegionName)
    (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    ComputeKernel := triton {
  i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
  o_d = i_d * $(B) + tl.arange(0, $(B))
  m_d = o_d < $(D)
  b_x = tl.load(x + i_n * $(D) + o_d, mask=m_d, other=-float("inf"))
  if HAS_SCALE {
    b_x = b_x * $(scale)
  }
  b_m = tl.max(b_x, 0)
  b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
  tl.store(z + i_n * tl.cdiv($(D), $(B)) + i_d, b_z)
}
```
</details>
