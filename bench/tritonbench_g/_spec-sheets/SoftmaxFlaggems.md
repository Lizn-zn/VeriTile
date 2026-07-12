# Spec sheet — `bench/tritonbench_g/softmax_flaggems/SoftmaxFlaggems.lean`

**Python source:** `bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py`

## Public theorem: `softmax_kernel_inner_one_tile_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the inner one-tile FlagGems softmax. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_inner_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_inner_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, linearOffset s N i)))
      (expected := fun i => softmaxFlaggemsSpec s input_ptr N TILE_N i)
```

**Assumptions / layout contracts:**
- `fun i : Fin TILE_N => i.val < N`

**Closed-form spec defs (transitive):** `softmax_kernel_inner_one_tile`, `softmaxFlaggemsSpec`, `softmaxFlaggemsInputTile`

<details><summary><code>softmax_kernel_inner_one_tile</code></summary>

```
/-- Proof-oriented `ONE_TILE_PER_CTA=true` slice of
`softmax_flaggems.py`'s `softmax_kernel_inner`.

This covers the inner-dimension fast path where one CTA covers a full row. It
preserves the source kernel's row program id, masked load with `-inf`, stable
softmax max/sum normalization, output-dtype load cast, and masked store. The
multi-tile online fallback remains future work. -/
```
```lean
def softmax_kernel_inner_one_tile
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) + n_offsets
  mask = n_offsets < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
  m = tl.max(inp, 0)
  e = tl.exp(inp - m)
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}
```
</details>

<details><summary><code>softmaxFlaggemsSpec</code></summary>

```lean
noncomputable def softmaxFlaggemsSpec
    (s : BlockState) (input_ptr : RegionName)
    (N TILE_N : Nat) (idx : Fin TILE_N) : ℝ :=
  let row := softmaxFlaggemsInputTile s input_ptr N TILE_N
  match Tile.reduceMax (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let e := Tile.uop WithBot.realExp shifted
      let z := Tile.reduceSum (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false e
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR e z).data
          (idx, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>softmaxFlaggemsInputTile</code></summary>

```lean
noncomputable def softmaxFlaggemsInputTile
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat) :
    Tile .real [TILE_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem input_ptr (linearOffset s N idx.1))
      else none }
```
</details>

## Public theorem: `softmax_kernel_non_inner_one_tile_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the non-inner one-tile FlagGems softmax. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_non_inner_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat)
    (s : BlockState)
    (hRange : s.pids 1 * TILE_K + TILE_K ≤ K) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_non_inner_one_tile_surface
        output_ptr input_ptr N K TILE_N TILE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [TILE_N, TILE_K] =>
          idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K)
        (fun idx => (output_ptr, nonInnerOffset s N K TILE_K idx)))
      (expected := fun idx =>
        softmaxFlaggemsNonInnerSpec s input_ptr N K TILE_N TILE_K idx)
```

**Assumptions / layout contracts:**
- `hRange : s.pids 1 * TILE_K + TILE_K ≤ K`
- `fun idx : TileIndex [TILE_N, TILE_K] =>
          idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K`

**Closed-form spec defs (transitive):** `softmax_kernel_non_inner_one_tile_surface`, `nonInnerOffset`, `softmaxFlaggemsNonInnerSpec`, `softmaxFlaggemsNonInnerInputTile`

<details><summary><code>softmax_kernel_non_inner_one_tile_surface</code></summary>

```
/-- Proof-oriented specialization of `softmax_kernel_non_inner` to
`ONE_TILE_PER_CTA=true`.

This covers the `K > 1` forward path used when the softmax dimension is not the
innermost physical dimension: each CTA handles one `(m, k-block)`, loads the
`[TILE_N, TILE_K]` tile, reduces along `N`, and stores the normalized tile.
The multi-tile fallback's `tl.float32` running `m/z` state is outside this
specialized surface. -/
```
```lean
def softmax_kernel_non_inner_one_tile_surface
    (output_ptr input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) :
    ComputeKernel := triton {
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)
  k_offsets = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
  mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
  input_ptrs = input_ptr + offset
  inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
  m = tl.max(inp, 0)
  e = tl.exp(inp - m[None, :])
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}
```
</details>

<details><summary><code>nonInnerOffset</code></summary>

```
/-! ## Non-inner one-tile forward — store-side proof skeleton

This block provides offset/value defs and a substantive `_writes_at_idx`
lemma showing that, for in-range lanes, the kernel writes exactly the
softmax-of-column value at every `(n, k)` lane. The full
`ComputeCorrect.Realizes_without_Rounding` lift to a math-level spec for the 2D non-inner
softmax is left as future work — the underlying difficulty is the per-column
streaming reduction along axis 0 with a precomputed-tile dependence. -/
```
```lean
def nonInnerOffset (s : BlockState) (N K TILE_K : Nat)
    (idx : TileIndex [TILE_N, TILE_K]) : Nat :=
  s.pids 0 * N * K + idx.1.val * K + (s.pids 1 * TILE_K + idx.2.1.val)
```
</details>

<details><summary><code>softmaxFlaggemsNonInnerSpec</code></summary>

```
/-- Spec for the non-inner one-tile softmax output at lane `(n, k)`:
softmax along the `n` axis (axis 0) at column `k`. -/
```
```lean
noncomputable def softmaxFlaggemsNonInnerSpec
    (s : BlockState) (input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) (idx : TileIndex [TILE_N, TILE_K]) : ℝ :=
  let row := softmaxFlaggemsNonInnerInputTile s input_ptr N K TILE_N TILE_K
  match Tile.reduceMax (shape := [TILE_N, TILE_K]) ⟨0, by simp⟩ Bool.false row with
  | some colMax =>
      let bc : Broadcast [TILE_N, TILE_K] [TILE_K] [TILE_N, TILE_K] :=
        Broadcast.leadR (Broadcast.consSame Broadcast.nil)
      let shifted := Tile.bop (NumericDType.sub .real) bc row colMax
      let e := Tile.uop WithBot.realExp shifted
      let z := Tile.reduceSum (shape := [TILE_N, TILE_K]) ⟨0, by simp⟩ Bool.false e
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) bc e z).data idx)
  | none => 0
```
</details>

<details><summary><code>softmaxFlaggemsNonInnerInputTile</code></summary>

```
/-- Per-`(n, k)` input tile for the 2D non-inner softmax: along the `n`
axis the column at fixed `k` ranges over `n ∈ [0, N)`; out-of-range lanes
default to `-∞` to match `tl.load(..., other=-float("inf"))`. -/
```
```lean
noncomputable def softmaxFlaggemsNonInnerInputTile
    (s : BlockState) (input_ptr : RegionName) (N K TILE_N TILE_K : Nat) :
    Tile .real [TILE_N, TILE_K] :=
  { data := fun idx =>
      if idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K then
        some (s.readMem input_ptr (nonInnerOffset s N K TILE_K idx))
      else ⊥ }
```
</details>

## Public theorem: `softmax_backward_kernel_inner_one_tile_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the inner one-tile FlagGems softmax
backward. -/
```
</details>

**Statement:**
```lean
specification softmax_backward_kernel_inner_one_tile_compute_correct
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N TILE_M TILE_N : Nat)
    (s : BlockState)
    (hOffInj : Function.Injective
      (fun idx : TileIndex [TILE_M, TILE_N] => innerBwdOffset s N TILE_M idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_backward_kernel_inner_one_tile_surface
        out_ptr out_grad_ptr in_grad_ptr M N TILE_M TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [TILE_M, TILE_N] =>
          innerBwdActive s M N TILE_M idx)
        (fun idx => (in_grad_ptr, innerBwdOffset s N TILE_M idx)))
      (expected := fun idx =>
        innerBwdSpec s out_ptr out_grad_ptr M N TILE_M TILE_N idx)
```

**Assumptions / layout contracts:**
- `hOffInj : Function.Injective
      (fun idx : TileIndex [TILE_M, TILE_N] => innerBwdOffset s N TILE_M idx)`
- `fun idx : TileIndex [TILE_M, TILE_N] =>
          innerBwdActive s M N TILE_M idx`

**Closed-form spec defs (transitive):** `innerBwdOffset`, `softmax_backward_kernel_inner_one_tile_surface`, `innerBwdActive`, `innerBwdSpec`, `innerBwdRowIndex`, `innerBwdColIndex`, `innerBwdOutTile`, `innerBwdOutGradTile`

<details><summary><code>innerBwdOffset</code></summary>

```lean
def innerBwdOffset (s : BlockState) (N TILE_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) : Nat :=
  innerBwdRowIndex s TILE_M idx * N + innerBwdColIndex idx
```
</details>

<details><summary><code>softmax_backward_kernel_inner_one_tile_surface</code></summary>

```
/-- Surface transcription of `softmax_flaggems.py`'s
`softmax_backward_kernel_inner`, specialized to `ONE_TILE_PER_CTA=true`.

This is the contiguous-row backward path: a CTA handles `[TILE_M, TILE_N]`,
reduces along `N`, and writes the masked input-gradient tile. -/
```
```lean
def softmax_backward_kernel_inner_one_tile_surface
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N TILE_M TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  m_offsets = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
  n_offsets = tl.arange(0, $(TILE_N))
  offsets = m_offsets[:, None] * $(N) + n_offsets
  mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
  out_tile = tl.load(out_ptr + offsets, mask=mask)
  out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
  scale = tl.sum(out_tile * out_grad_tile, 1)
  in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
  tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
}
```
</details>

<details><summary><code>innerBwdActive</code></summary>

```lean
def innerBwdActive (s : BlockState) (M N TILE_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) : Prop :=
  innerBwdRowIndex s TILE_M idx < M ∧ innerBwdColIndex idx < N
```
</details>

<details><summary><code>innerBwdSpec</code></summary>

```lean
noncomputable def innerBwdSpec
    (s : BlockState) (out_ptr out_grad_ptr : RegionName)
    (M N TILE_M TILE_N : Nat) (idx : TileIndex [TILE_M, TILE_N]) : ℝ :=
  let outT := innerBwdOutTile s out_ptr M N TILE_M TILE_N
  let gradT := innerBwdOutGradTile s out_grad_ptr M N TILE_M TILE_N
  let sameBc : Broadcast [TILE_M, TILE_N] [TILE_M, TILE_N] [TILE_M, TILE_N] :=
    Broadcast.consSame (Broadcast.consSame Broadcast.nil)
  let prod := Tile.bop (NumericDType.mul .real) sameBc outT gradT
  let scale :=
    Tile.reduceSum (shape := [TILE_M, TILE_N]) ⟨1, by simp⟩ Bool.true prod
  let rowBc : Broadcast [TILE_M, TILE_N] [TILE_M, 1] [TILE_M, TILE_N] :=
    Broadcast.consSame (Broadcast.consR Broadcast.nil)
  let diff :=
    Tile.bop (NumericDType.sub .real) rowBc gradT scale
  WithBot.unbotD 0
    ((Tile.bop (NumericDType.mul .real) sameBc outT diff).data idx)
```
</details>

<details><summary><code>innerBwdRowIndex</code></summary>

```lean
def innerBwdRowIndex (s : BlockState) (TILE_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) : Nat :=
  s.pid * TILE_M + idx.1.val
```
</details>

<details><summary><code>innerBwdColIndex</code></summary>

```lean
def innerBwdColIndex (idx : TileIndex [TILE_M, TILE_N]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>innerBwdOutTile</code></summary>

```lean
noncomputable def innerBwdOutTile
    (s : BlockState) (out_ptr : RegionName)
    (M N TILE_M TILE_N : Nat) :
    Tile .real [TILE_M, TILE_N] :=
  { data := fun idx =>
      if innerBwdActive s M N TILE_M idx then
        some (s.readMem out_ptr (innerBwdOffset s N TILE_M idx))
      else some (s.undef out_ptr (innerBwdOffset s N TILE_M idx)) }
```
</details>

<details><summary><code>innerBwdOutGradTile</code></summary>

```lean
noncomputable def innerBwdOutGradTile
    (s : BlockState) (out_grad_ptr : RegionName)
    (M N TILE_M TILE_N : Nat) :
    Tile .real [TILE_M, TILE_N] :=
  { data := fun idx =>
      if innerBwdActive s M N TILE_M idx then
        some (s.readMem out_grad_ptr (innerBwdOffset s N TILE_M idx))
      else some (s.undef out_grad_ptr (innerBwdOffset s N TILE_M idx)) }
```
</details>

## Public theorem: `softmax_backward_kernel_non_inner_one_tile_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the non-inner one-tile FlagGems softmax
backward. -/
```
</details>

**Statement:**
```lean
specification softmax_backward_kernel_non_inner_one_tile_compute_correct
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (N K TILE_N TILE_K : Nat)
    (s : BlockState)
    (hRange : s.pids 1 * TILE_K + TILE_K ≤ K) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_backward_kernel_non_inner_one_tile_surface
        out_ptr out_grad_ptr in_grad_ptr N K TILE_N TILE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [TILE_N, TILE_K] =>
          idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K)
        (fun idx => (in_grad_ptr, nonInnerOffset s N K TILE_K idx)))
      (expected := fun idx =>
        nonInnerBwdSpec s out_ptr out_grad_ptr N K TILE_N TILE_K idx)
```

**Assumptions / layout contracts:**
- `hRange : s.pids 1 * TILE_K + TILE_K ≤ K`
- `fun idx : TileIndex [TILE_N, TILE_K] =>
          idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K`

**Closed-form spec defs (transitive):** `softmax_backward_kernel_non_inner_one_tile_surface`, `nonInnerOffset`, `nonInnerBwdSpec`, `nonInnerBwdOutTile`, `nonInnerBwdOutGradTile`

<details><summary><code>softmax_backward_kernel_non_inner_one_tile_surface</code></summary>

```
/-- Surface transcription of `softmax_flaggems.py`'s
`softmax_backward_kernel_non_inner`, specialized to `ONE_TILE_PER_CTA=true`.

This preserves the non-inner `(m, k-block)` addressing and the standard softmax
VJP formula `out * (out_grad - sum(out * out_grad, axis=0))`. -/
```
```lean
def softmax_backward_kernel_non_inner_one_tile_surface
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  offsets_k = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
  offsets_n = tl.arange(0, $(TILE_N))
  offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
  mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
  out_tile = tl.load(out_ptr + offsets, mask=mask)
  out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
  scale = tl.sum(out_tile * out_grad_tile, axis=0)
  in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
  tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
}
```
</details>

<details><summary><code>nonInnerOffset</code></summary>

```
/-! ## Non-inner one-tile forward — store-side proof skeleton

This block provides offset/value defs and a substantive `_writes_at_idx`
lemma showing that, for in-range lanes, the kernel writes exactly the
softmax-of-column value at every `(n, k)` lane. The full
`ComputeCorrect.Realizes_without_Rounding` lift to a math-level spec for the 2D non-inner
softmax is left as future work — the underlying difficulty is the per-column
streaming reduction along axis 0 with a precomputed-tile dependence. -/
```
```lean
def nonInnerOffset (s : BlockState) (N K TILE_K : Nat)
    (idx : TileIndex [TILE_N, TILE_K]) : Nat :=
  s.pids 0 * N * K + idx.1.val * K + (s.pids 1 * TILE_K + idx.2.1.val)
```
</details>

<details><summary><code>nonInnerBwdSpec</code></summary>

```lean
noncomputable def nonInnerBwdSpec
    (s : BlockState) (out_ptr out_grad_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) (idx : TileIndex [TILE_N, TILE_K]) : ℝ :=
  let outT := nonInnerBwdOutTile s out_ptr N K TILE_N TILE_K
  let gradT := nonInnerBwdOutGradTile s out_grad_ptr N K TILE_N TILE_K
  let sameBc : Broadcast [TILE_N, TILE_K] [TILE_N, TILE_K] [TILE_N, TILE_K] :=
    Broadcast.consSame (Broadcast.consSame Broadcast.nil)
  let prod := Tile.bop (NumericDType.mul .real) sameBc outT gradT
  let scale :=
    Tile.reduceSum (shape := [TILE_N, TILE_K]) ⟨0, by simp⟩ Bool.false prod
  let colBc : Broadcast [TILE_N, TILE_K] [TILE_K] [TILE_N, TILE_K] :=
    Broadcast.leadR (Broadcast.consSame Broadcast.nil)
  let diff :=
    Tile.bop (NumericDType.sub .real) colBc gradT scale
  WithBot.unbotD 0
    ((Tile.bop (NumericDType.mul .real) sameBc outT diff).data idx)
```
</details>

<details><summary><code>nonInnerBwdOutTile</code></summary>

```lean
noncomputable def nonInnerBwdOutTile
    (s : BlockState) (out_ptr : RegionName) (N K TILE_N TILE_K : Nat) :
    Tile .real [TILE_N, TILE_K] :=
  { data := fun idx =>
      if idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K then
        some (s.readMem out_ptr (nonInnerOffset s N K TILE_K idx))
      else some (s.undef out_ptr (nonInnerOffset s N K TILE_K idx)) }
```
</details>

<details><summary><code>nonInnerBwdOutGradTile</code></summary>

```lean
noncomputable def nonInnerBwdOutGradTile
    (s : BlockState) (out_grad_ptr : RegionName) (N K TILE_N TILE_K : Nat) :
    Tile .real [TILE_N, TILE_K] :=
  { data := fun idx =>
      if idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K then
        some (s.readMem out_grad_ptr (nonInnerOffset s N K TILE_K idx))
      else some (s.undef out_grad_ptr (nonInnerOffset s N K TILE_K idx)) }
```
</details>
