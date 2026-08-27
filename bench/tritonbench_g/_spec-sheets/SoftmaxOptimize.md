# Spec sheet — `bench/tritonbench_g/softmax_optimize/SoftmaxOptimize.lean`

**Python source:** `bench/tritonbench_g/softmax_optimize/softmax_optimize.py`

## Public theorem: `softmax_kernel_online_v2_one_tile_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for the one-tile slice of
`softmax_kernel_online_v2`: the slice lowers to the algorithm layer, and the
masked store to `output_ptr` is compute-correct — every active lane (`i < N`)
holds the stable-softmax value `softmaxOptimizeSpec`, out-of-bounds lanes are
preserved. (Covers the `N ≤ TILE_N` collapsed-loop regime; the full online
surface's exec-level value correctness is established separately, see
`softmax_kernel_online_v2_surface_exec_correct`.) -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_online_v2_one_tile_output_summary
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    (∃ alg, (softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, linearOffset s N i)))
      (expected := fun i => softmaxOptimizeSpec s input_ptr N TILE_N i)
```

**Assumptions / layout contracts:**
- `fun i : Fin TILE_N => i.val < N`

**Closed-form spec defs (transitive):** `softmax_kernel_online_v2_one_tile`, `softmaxOptimizeSpec`, `softmaxOptimizeInputTile`

<details><summary><code>softmax_kernel_online_v2_one_tile</code></summary>

```
/-- Proof-oriented one-tile specialization of `softmax_kernel_online_v2`.

When `N <= TILE_N`, the online loops collapse to one masked row tile. This kernel
is kept as the small executable target for the existing algorithm proof. -/
```
```lean
def softmax_kernel_online_v2_one_tile
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

<details><summary><code>softmaxOptimizeSpec</code></summary>

```lean
noncomputable def softmaxOptimizeSpec
    (s : BlockState) (input_ptr : RegionName)
    (N TILE_N : Nat) (idx : Fin TILE_N) : ℝ :=
  let row := softmaxOptimizeInputTile s input_ptr N TILE_N
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

<details><summary><code>softmaxOptimizeInputTile</code></summary>

```lean
noncomputable def softmaxOptimizeInputTile
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat) :
    Tile .real [TILE_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem input_ptr (linearOffset s N idx.1))
      else none }
```
</details>

## Public theorem: `softmax_kernel_online_v2_output_summary`

<details><summary>docstring</summary>

```
/-- **Public output summary (headline theorem).** The full online-softmax
surface `softmax_kernel_online_v2_surface` realizes the genuine full-row softmax
`softmaxOptimizeFullSpec` at every output column `j < N` of row `pid`: the total
write map sends column `j` to `output_ptr` at `linearOffset s N j`, and the
value read back there after any successful run is exactly the numerically
stabilized full-row softmax of the loaded input row. Stated via
`ComputeCorrect.Realizes_without_Rounding` (per `bench/MAIN_THEOREM_CONVENTIONS.md` §4), wrapping
the exec-level engine lemma `softmax_kernel_online_v2_surface_exec_correct`. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_online_v2_output_summary
    (output_ptr input_ptr : RegionName) (M N TILE_N : Nat)
    (hN : 0 < N) (hT : 0 < TILE_N) (hne : output_ptr ≠ input_ptr)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N)
      (initialState := s)
      (write := fun j : Fin N => some (output_ptr, linearOffset s N j))
      (expected := fun j : Fin N => softmaxOptimizeFullSpec s input_ptr N j)
```

**Assumptions / layout contracts:**
- `hN : 0 < N`
- `hT : 0 < TILE_N`
- `hne : output_ptr ≠ input_ptr`

**Closed-form spec defs (transitive):** `softmax_kernel_online_v2_surface`, `softmaxOptimizeFullSpec`, `softmaxOptimizeRow`

<details><summary><code>softmax_kernel_online_v2_surface</code></summary>

```
/-- Faithful transcription of `softmax_optimize.py`'s
`softmax_kernel_online_v2`.

This keeps the online max/sum recurrence over full tiles, the masked tail pass,
the final scalar normalization, and the two writeback loops. -/
```
```lean
def softmax_kernel_online_v2_surface
    (output_ptr input_ptr : RegionName)
    (M N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  m = tl.full([$(TILE_N)], value=-float("inf"), dtype=output_ptr.dtype.element_ty)
  z = tl.full([$(TILE_N)], value=0, dtype=output_ptr.dtype.element_ty)
  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  final_m = tl.max(m, 0)
  z = tl.sum(tl.exp(m - final_m) * z)
  m = final_m

  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out)
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)
  }
}
```
</details>

<details><summary><code>softmaxOptimizeFullSpec</code></summary>

```
/-- **Genuine full-row softmax closed form.** For column `j < N`, the standard
numerically-stabilized softmax over the `N` loaded inputs:
`exp(input[j] - rowMax) / Σ_{j'<N} exp(input[j'] - rowMax)`, where
`rowMax = max_{j'<N} input[j']`. This is exactly `TiledSoftmax.naiveSpec`
(`= stableSpec`, the numerically stabilized form). It is a function of the
loaded `input` memory, not of the kernel's own output. -/
```
```lean
noncomputable def softmaxOptimizeFullSpec
    (s : BlockState) (input_ptr : RegionName) (N : Nat) (j : Fin N) : ℝ :=
  Real.exp (softmaxOptimizeRow s input_ptr N j)
    / ∑ j' : Fin N, Real.exp (softmaxOptimizeRow s input_ptr N j')
```
</details>

<details><summary><code>softmaxOptimizeRow</code></summary>

```
/-- The loaded input row as a function of the genuine column index `j : Fin N`:
`input[pid_m, j]` at memory address `linearOffset s N j = s.pid·N + j`. -/
```
```lean
noncomputable def softmaxOptimizeRow
    (s : BlockState) (input_ptr : RegionName) (N : Nat) (j : Fin N) : ℝ :=
  s.readMem input_ptr (linearOffset s N j)
```
</details>

## Public theorem: `softmax_kernel_online_v2_one_tile_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `softmax_optimize.py`'s one-tile path:
for every disjoint flat placement of `input_ptr` / `output_ptr`, every program id
whose active lanes are in bounds, and every launch state whose input row holds
`xs` on the `i < N` lanes, the kernel terminates, every active lane of the output
row holds `softmaxTileSpec N TILE_N xs i`, and every other memory cell is
unchanged.

`softmaxTileSpec` is the port's own reduction chain — running max, shift,
`exp`, running sum, divide — read over the **loaded lane values** instead of over
memory, so this is a genuine input/output statement rather than a read-back of
the kernel's own buffer. Masked-out lanes stay `⊥`, exactly as the kernel's
`other = -float("inf")` load leaves them, which is what makes the face true for a
partial tile (`N < TILE_N`, the case the one-tile path is *for*) and not only for
a full one.

Dimension-general in `N` and `TILE_N`. **Zero side-conditions on addresses**: the
window is `pid·N + i`, so injectivity is discharged inline. The one honest
hypothesis is `0 < TILE_N` — an empty tile makes `tl.max` fault, exactly as in
Python. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_online_v2_one_tile_io_correctness
    (output_ptr input_ptr : RegionName) (N TILE_N : Nat) (hT : 0 < TILE_N) :
    oneTileIO output_ptr input_ptr N TILE_N
      ⊨ fun _pid xs idx => softmaxTileSpec N TILE_N xs idx.1
```

**Assumptions / layout contracts:**
- `hT : 0 < TILE_N`

**Closed-form spec defs (transitive):** `oneTileIO`, `softmaxTileSpec`, `softmax_kernel_online_v2_one_tile`

<details><summary><code>oneTileIO</code></summary>

```
/-- IO signature of the one-tile softmax. -/
```
```lean
def oneTileIO (output_ptr input_ptr : RegionName) (N TILE_N : Nat) :
    MaskedTileKernelIO₁ where
  kernel := softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N
  inp := input_ptr
  out := output_ptr
  shape := [TILE_N]
  read := fun pid idx => pid * N + idx.1.val
  write := fun pid idx => pid * N + idx.1.val
  mask := fun _pid idx => idx.1.val < N
```
</details>

<details><summary><code>softmaxTileSpec</code></summary>

```
/-- `softmaxOptimizeSpec`'s reduction chain, read over the **loaded lane values**
instead of over memory. Masked-out lanes are `⊥`, exactly as the kernel's
`other = -float("inf")` load leaves them. -/
```
```lean
noncomputable def softmaxTileSpec (N TILE_N : Nat)
    (xs : TileIndex [TILE_N] → ℝ) (idx : Fin TILE_N) : ℝ :=
  let row : Tile .real [TILE_N] :=
    { data := fun j => if j.1.val < N then some (xs j) else none }
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

<details><summary><code>softmax_kernel_online_v2_one_tile</code></summary>

```
/-- Proof-oriented one-tile specialization of `softmax_kernel_online_v2`.

When `N <= TILE_N`, the online loops collapse to one masked row tile. This kernel
is kept as the small executable target for the existing algorithm proof. -/
```
```lean
def softmax_kernel_online_v2_one_tile
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

## Also present (pinned special-case summaries)
- `softmax_kernel_online_v2_one_tile_compute_correct`
