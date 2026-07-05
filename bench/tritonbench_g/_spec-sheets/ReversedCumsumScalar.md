# Spec sheet — `bench/tritonbench_g/reversed_cumsum_scalar/ReversedCumsumScalar.lean`

**Python source:** `bench/tritonbench_g/reversed_cumsum_scalar/reversed_cumsum_scalar.py`

## Public theorem: `reversed_cumsum_scalar_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `reversed_cumsum_scalar.py`,
against the **genuine closed forms** (no self-referential read-back), for
arbitrary `T`, `BT` and strides. It packages:

* the **full reverse-traversal surface** (verbatim `range(cdiv(T,BT)−1,−1,−1)`
  loop with the carried scalar `b_z`) lowers to the algorithm layer;
* the **single-chunk path** (any shape with `T ≤ BT`, e.g. the Python
  benchmark `T = 4`, `BT = 16`, where the reverse loop runs exactly once):
  every active lane `i` receives the genuine global reversed cumulative sum
  `singleBlockRevClosed = Σ_{flat ≥ i, flat < T} s[i_bh·T + flat]`,
  end-to-end from the input region `S` — a standalone `Finset.sum`, never a
  read-back of the kernel output;
* the **per-chunk reversed carry-fold face**: for *every* state whose carry
  buffer holds the genuine post-update suffix total
  (`Carry[i_bh] = Σ_{flat ≥ i_t·BT, flat < T} s[i_bh·T + flat]` — the value
  `b_z` has after the kernel's `b_z += b_zz`), one chunk iteration realizes
  the genuine global reversed cumulative sum `globalRevCumsumClosed`.

Honest side-conditions only: output-offset injectivity of each store
footprint. The **cross-chunk loop scheduling** that threads the carried `b_z`
register from chunk `c` to `c−1` — i.e. that the materialized carry buffer
holds the genuine suffix total when the body of chunk `c` runs — is the
*trusted runtime boundary*, exactly as in the sibling `reversed_cumsum`
(#290-style carried register); its algebraic content is
`revStoreValue_eq_globalRevCumsumClosed`. -/
```
</details>

**Statement:**
```lean
theorem reversed_cumsum_scalar_output_summary_general
    (S Carry O : RegionName) (T BT : Nat) (hT : T ≤ BT) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    -- (1) the full reverse-traversal surface lowers to the algorithm layer
    (∃ alg, (reversed_cumsum_scalar_surface S O T BT).toAlgorithm?
      = Except.ok alg) ∧
    -- (2) single-chunk genuine suffix sum, end-to-end from `S`
    (ComputeCorrect.Realizes
      (kernel := reversed_cumsum_scalar_single_block_surface S O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => singleBlockActive s T i)
        (fun i => (O, singleBlockVecOffset s T i)))
      (expected := fun i : Fin BT => singleBlockRevClosed s S T BT i)) ∧
    -- (3) per-chunk reversed carry-fold face, for every carry-invariant state
    (∀ (s' : BlockState),
      Function.Injective (fun i : Fin BT => vecOffset s' T BT i) →
      s'.readMem Carry (s'.pids 0)
        = (∑ flat ∈ (Finset.range T).filter
            (fun flat => s'.pids 1 * BT ≤ flat), rowElem s' S T flat) →
      ComputeCorrect.Realizes
        (kernel := reversed_cumsum_scalar_rev_slice S Carry O T BT)
        (initialState := s')
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BT => active s' T BT i)
          (fun i => (O, vecOffset s' T BT i)))
        (expected := fun i : Fin BT => globalRevCumsumClosed s' S T BT i))
```

**Assumptions / layout contracts:**
- `hT : T ≤ BT`
- `hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)`
- `fun i : Fin BT => singleBlockActive s T i`
- `fun i : Fin BT => vecOffset s' T BT i`
- `fun i : Fin BT => active s' T BT i`

**Closed-form spec defs (transitive):** `singleBlockVecOffset`, `reversed_cumsum_scalar_surface`, `reversed_cumsum_scalar_single_block_surface`, `singleBlockActive`, `singleBlockRevClosed`, `vecOffset`, `rowElem`, `reversed_cumsum_scalar_rev_slice`, `active`, `globalRevCumsumClosed`, `tIndex`

<details><summary><code>singleBlockVecOffset</code></summary>

```lean
def singleBlockVecOffset (s : BlockState) (T : Nat) (i : Fin BT) : Nat :=
  s.pids 0 * T + i.val
```
</details>

<details><summary><code>reversed_cumsum_scalar_surface</code></summary>

```
/-- Faithful transcription of `reversed_cumsum_scalar.py`'s
`chunk_global_reversed_cumsum_scalar_kernel`.

The chunk loop runs in reverse (`range(cdiv(T,BT)−1, −1, −1)`), the scalar
carry `b_z` is updated with the current chunk total *before* the store, and
the final cast targets the block pointer destination dtype. -/
```
```lean
def reversed_cumsum_scalar_surface
  (S O : RegionName) (T BT : Nat) : ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_z = tl.zeros([], dtype=tl.float32)
  for i_t in range(tl.cdiv($(T), $(BT)) - $(1), -$(1), -$(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    p_o = tl.make_block_ptr(base=O + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    b_s = tl.load(p_s, boundary_check=([0] : List Nat)).to(tl.float32)
    b_zz = tl.sum(b_s, axis=0)
    b_z += b_zz
    b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0] : List Nat))
  }
}
```
</details>

<details><summary><code>reversed_cumsum_scalar_single_block_surface</code></summary>

```
/-! ## Single Python-chunk computed surface

For the checked Python shape `T = 4` and autotuned `BT = 16`, the reverse loop
has one iteration and the carried scalar `b_z` at store time is exactly the
current chunk total (`0 + b_zz`). This surface keeps the Python-observable
path from `S` to `O` without a precomputed output tile or external carry
buffer: `b_o = b_s − tl.cumsum(b_s) + tl.sum(b_s)`. -/
```
```lean
def reversed_cumsum_scalar_single_block_surface
    (S O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  offs_t = tl.arange(0, $(BT))
  mask = offs_t < $(T)
  b_s = tl.load(S + i_bh * $(T) + offs_t, mask=mask, other=0.0).to(tl.float32)
  b_zz = tl.sum(b_s, axis=0)
  b_o = b_s - tl.cumsum(b_s, axis=0) + b_zz[None]
  tl.store(O + i_bh * $(T) + offs_t, (b_o).to(O.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>singleBlockActive</code></summary>

```lean
def singleBlockActive (_s : BlockState) (T : Nat) (i : Fin BT) : Prop :=
  i.val < T
```
</details>

<details><summary><code>singleBlockRevClosed</code></summary>

```
/-- Genuine closed form for the single-Python-chunk path (single iteration,
carry `= 0 + chunk total`): the suffix sum of all source entries from flat
index `i` to the end of the row. -/
```
```lean
noncomputable def singleBlockRevClosed
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  ∑ flat ∈ (Finset.range T).filter (fun flat => i.val ≤ flat),
    rowElem s S T flat
```
</details>

<details><summary><code>vecOffset</code></summary>

```lean
def vecOffset (s : BlockState) (T BT : Nat) (i : Fin BT) : Nat :=
  s.pids 0 * T + tIndex s BT i
```
</details>

<details><summary><code>rowElem</code></summary>

```
/-- Row element `s[i_bh, flat]` of the row-major `[B·H, T]` input: this
program's row (`i_bh = pids 0`, row stride `T`) at flat time index `flat`
(unit column stride). Every load/spec read in this file uses this layout. -/
```
```lean
noncomputable def rowElem (s : BlockState) (S : RegionName) (T flat : Nat) : ℝ :=
  s.readMem S (s.pids 0 * T + flat)
```
</details>

<details><summary><code>reversed_cumsum_scalar_rev_slice</code></summary>

```
/-- Proof-oriented single-iteration reversed-cumsum slice of
`reversed_cumsum_scalar.py`'s `chunk_global_reversed_cumsum_scalar_kernel`.

This models one loop iteration after the kernel's `b_z += b_zz` update for the
current chunk: `Carry` materializes the **post-update** carry — the suffix
total of the current chunk *and* all later chunks. The slice loads the current
source block, computes `b_s − tl.cumsum(b_s, axis=0) + b_z`, and stores the
masked output block. -/
```
```lean
def reversed_cumsum_scalar_rev_slice
    (S Carry O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  i_t = tl.program_id(1)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  mask = offs_t < $(T)
  b_s = tl.load(S + i_bh * $(T) + offs_t, mask=mask, other=0.0).to(tl.float32)
  b_z = tl.load(Carry + i_bh).to(tl.float32)
  b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
  tl.store(O + i_bh * $(T) + offs_t, (b_o).to(O.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (T BT : Nat) (i : Fin BT) : Prop :=
  tIndex s BT i < T
```
</details>

<details><summary><code>globalRevCumsumClosed</code></summary>

```
/-! ## Genuine reversed chunked-cumsum closed form

`globalRevCumsumClosed` is the genuine mathematical specification — a
`Finset.sum` over all *flat* time indices `flat ≥ chunk·BT + i` (with
`flat < T`) of the source value at `i_bh·T + flat`. It is **not** a read-back
of the kernel's own output, so realizing it is a true correctness statement.

For program `(i_bh, i_t)` (so `i_bh = pids 0`, `i_t = pids 1`) and active lane
`i`, the global flat output index is `i_t·BT + i`, and the output must hold
the suffix sum of all source entries from that flat index to the end of the
row. -/
```
```lean
noncomputable def globalRevCumsumClosed
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  ∑ flat ∈ (Finset.range T).filter
      (fun flat => s.pids 1 * BT + i.val ≤ flat),
    rowElem s S T flat
```
</details>

<details><summary><code>tIndex</code></summary>

```lean
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val
```
</details>

## Also present (pinned special-case summaries)
- `reversed_cumsum_scalar_store_slice_compute_correct`
- `reversed_cumsum_scalar_rev_slice_compute_correct`
- `reversed_cumsum_scalar_single_block_surface_compute_correct`
