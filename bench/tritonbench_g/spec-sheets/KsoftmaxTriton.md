# Spec sheet — `bench/tritonbench_g/ksoftmax_triton/KsoftmaxTriton.lean`

**Python source:** `bench/tritonbench_g/ksoftmax_triton/ksoftmax_triton.py`

## Public theorem: `ksoftmax_forward_plain_output_summary`

<details><summary>docstring</summary>

```
/-- Public plain-forward summary: the Python surface for
`LOG=false`, no mask, non-causal, and no fp16 accumulator cast lowers, and the
corresponding checked proof realizes the softmax computation over the output row
from `x` to `y`.
-/
```
</details>

**Statement:**
```lean
theorem ksoftmax_forward_plain_output_summary
    (Y X M : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin DEPTH => yOffset s stride_ym stride_yn i)) :
    (∃ alg, (ksoftmax_forward_surface Y X M stride_ym stride_yn stride_xm
      stride_xn stride_m K DEPTH Bool.false Bool.false Bool.false Bool.false
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := ksoftmax_forward_plain Y X
        stride_ym stride_yn stride_xm stride_xn K DEPTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin DEPTH => i.val < K)
        (fun i => (Y, yOffset s stride_ym stride_yn i)))
      (expected := fun i => ksoftmaxSpec s X stride_xm stride_xn K DEPTH i))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin DEPTH => yOffset s stride_ym stride_yn i)`
- `fun i : Fin DEPTH => i.val < K`

**Closed-form spec defs (transitive):** `yOffset`, `ksoftmax_forward_surface`, `ksoftmax_forward_plain`, `ksoftmaxSpec`, `ksoftmaxInputTile`, `xOffset`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset
    (s : BlockState) (stride_ym stride_yn : Nat) (i : Fin DEPTH) : Nat :=
  s.pids 0 * stride_ym + s.pids 1 * stride_yn + i.val
```
</details>

<details><summary><code>ksoftmax_forward_surface</code></summary>

```
/-- Lean transcription of `ksoftmax_triton.py`'s `_softmax`.

`MASK_TYPE` records whether the optional mask is present; `MASK_QK` selects
Python's `qk` layout when true and `bk` layout when false. -/
```
```lean
def ksoftmax_forward_surface
    (Y X M : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG MASK_TYPE CAUSAL IS_FP16 MASK_QK : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
  if CAUSAL {
    off = float("-inf")
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  if MASK_TYPE {
    if MASK_QK {
      mask_ptrs = M + n * $(stride_m) + k
    } else {
      mask_ptrs = M + m * $(stride_m) + k
    }
    add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
    x += add_mask
  }
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}
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

<details><summary><code>ksoftmaxSpec</code></summary>

```lean
noncomputable def ksoftmaxSpec
    (s : BlockState) (X : RegionName)
    (stride_xm stride_xn K DEPTH : Nat) (idx : Fin DEPTH) : ℝ :=
  let row := ksoftmaxInputTile s X stride_xm stride_xn K DEPTH
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

<details><summary><code>ksoftmaxInputTile</code></summary>

```lean
noncomputable def ksoftmaxInputTile
    (s : BlockState) (X : RegionName)
    (stride_xm stride_xn K DEPTH : Nat) :
    Tile .real [DEPTH] :=
  { data := fun idx =>
      if idx.1.val < K then
        some (s.readMem X (xOffset s stride_xm stride_xn idx.1))
      else none }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset
    (s : BlockState) (stride_xm stride_xn : Nat) (i : Fin DEPTH) : Nat :=
  s.pids 0 * stride_xm + s.pids 1 * stride_xn + i.val
```
</details>

## Also present (pinned special-case summaries)
- `ksoftmax_forward_plain_compute_correct`
