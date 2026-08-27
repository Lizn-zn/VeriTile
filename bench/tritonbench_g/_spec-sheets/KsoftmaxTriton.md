# Spec sheet — `bench/tritonbench_g/ksoftmax_triton/KsoftmaxTriton.lean`

**Python source:** `bench/tritonbench_g/ksoftmax_triton/ksoftmax_triton.py`

## Public theorem: `ksoftmax_forward_plain_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: the plain forward slice of `_softmax` (`LOG=false`, no
mask, non-causal, no fp16 cast) implements the exact stable softmax over the
active row prefix on its masked 2D IO signature — for every disjoint flat
placement of the two buffers, every program `(m, n)` whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `k` holds
`ksoftmaxSpec K DEPTH xs k`, and every other memory cell is unchanged.
`0 < DEPTH` is required: the kernel's `max` reduce (like `Finset.sup'`) is
only defined on non-empty tiles. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification ksoftmax_forward_plain_correctness
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (hD : 0 < DEPTH) :
    ksoftmaxIO Y X stride_ym stride_yn stride_xm stride_xn K DEPTH ⊨
      fun _ _ xs k => ksoftmaxSpec K DEPTH xs k
```

**Assumptions / layout contracts:**
- `hD : 0 < DEPTH`

**Closed-form spec defs (transitive):** `ksoftmaxIO`, `ksoftmaxSpec`, `ksoftmax_forward_plain`, `ksoftmaxInputTile`

<details><summary><code>ksoftmaxIO</code></summary>

```
/-- `ksoftmax_forward_plain`'s masked **2D IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = DEPTH` — the last-dimension row window each program owns;
* `read`/`write` — program `(m, n)` reads lane `k` of its row at
  `m * stride_xm + n * stride_xn + k` and writes it at
  `m * stride_ym + n * stride_yn + k` (the host-side one-program-per-`(m, n)`
  launch convention over the leading two dimensions);
* `mask` — the active lanes `k < K`, **the same for every program**: the row
  prefix that actually exists in the tensor. Inactive lanes (the padding of
  `DEPTH = next_power_of_2(K)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def ksoftmaxIO (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat) :
    Masked2DKernelIO₁ where
  kernel := ksoftmax_forward_plain Y X stride_ym stride_yn stride_xm
    stride_xn K DEPTH
  inp := X
  out := Y
  B := DEPTH
  read := fun m n j => m * stride_xm + n * stride_xn + j.val
  write := fun m n j => m * stride_ym + n * stride_yn + j.val
  mask := fun _ _ j => j.val < K
```
</details>

<details><summary><code>ksoftmaxSpec</code></summary>

```
/-- Exact stable-softmax value computed by the kernel at lane `idx`, as a pure
function of the active row prefix `xs k`, `k < K` (masked lanes enter the
reductions as `⊥`, neutral for both `max` and the `exp`-sum). -/
```
```lean
noncomputable def ksoftmaxSpec (K DEPTH : Nat)
    (xs : Fin DEPTH → ℝ) (idx : Fin DEPTH) : ℝ :=
  let row := ksoftmaxInputTile K DEPTH xs
  match Tile.reduceMax (shape := [DEPTH]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let num := Tile.uop WithBot.realExp shifted
      let denom := Tile.reduceSum (shape := [DEPTH]) ⟨0, by simp⟩ Bool.false num
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR num denom).data
          (idx, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>ksoftmax_forward_plain</code></summary>

```
/-- Proof-oriented forward softmax slice of `ksoftmax_triton.py`'s `_softmax`.

This specializes the constexpr branches to:
- `LOG = false`
- `MASK_TYPE = None`
- `CAUSAL = false`
- `IS_FP16 = false`

It preserves the 2D `(m, n)` program ids, strided 3D row addressing, masked
load over the last dimension, stable softmax normalization, and masked store. -/
```
```lean
def ksoftmax_forward_plain
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  x = tl.load(x_ptrs, mask=io_mask, other=-inf)
  z = x - tl.max(x, axis=0)
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  y = num / denom
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}
```
</details>

<details><summary><code>ksoftmaxInputTile</code></summary>

```
/-- Masked input row tile used by `ksoftmax_forward_plain`: lane `k < K` holds
`xs k`, masked lanes are `⊥`, matching `other=-inf`. -/
```
```lean
noncomputable def ksoftmaxInputTile (K DEPTH : Nat)
    (xs : Fin DEPTH → ℝ) : Tile .real [DEPTH] :=
  { data := fun idx => if idx.1.val < K then some (xs idx.1) else none }
```
</details>
