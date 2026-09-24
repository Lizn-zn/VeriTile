# Spec sheet — `bench/tritonbench_g/max_reduction/MaxReduction.lean`

**Python source:** `bench/tritonbench_g/max_reduction/max_reduction.py`

## Public theorem: `max_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for the 2D value/index `max_kernel`: the DSL
surface lowers to the algorithm layer, and the masked value/index stores are
compute-correct — every active row lane `i` holds the row-wise maximum
(`maxKernelValueSpec`) in `out_value` and its argmax (`maxKernelIndexSpec`) in
`out_index`. Carries the existing side conditions of
`max_kernel_compute_correct`: `hOutInj` (injective output offsets) and
`hOutRegions` (`out_value ≠ out_index`). -/
```
</details>

**Statement:**
```lean
specification max_kernel_output_summary
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => maxKernelOutOffset s K BLOCK_M i))
    (hOutRegions : out_value ≠ out_index) :
    (∃ alg, (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.OutputPairWhere
      (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N)
      s out_value out_index
      (maxKernelOutOffset s K BLOCK_M)
      (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M)
      (maxKernelValueSpec s inp M N K BLOCK_M BLOCK_N)
      (maxKernelIndexSpec s inp M N K BLOCK_M BLOCK_N)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => maxKernelOutOffset s K BLOCK_M i)`
- `hOutRegions : out_value ≠ out_index`

**Closed-form spec defs (transitive):** `maxKernelOutOffset`, `max_kernel`, `maxKernelValueSpec`, `maxKernelIndexSpec`, `maxKernelInputTile`, `maxInpElem`

<details><summary><code>maxKernelOutOffset</code></summary>

```
/-- Output offset for the 2D value/index max kernel at row lane `i`. -/
```
```lean
def maxKernelOutOffset (s : BlockState) (K BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  (s.pids 0 * BLOCK_M + i.val) * K + s.pids 1
```
</details>

<details><summary><code>max_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel`
(autotuned, returns value + index via `tl.max(..., return_indices=True)`).

Allowed mechanical Lean-syntax-only changes apply. The `@triton.autotune`
and `@triton.heuristics` decorators that wrap `max_kernel` are not DSL-side
constructs (they configure the launch grid, not the kernel body); they have
no in-kernel transcription. -/
```
```lean
def max_kernel
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index, mask=mask1)
}
```
</details>

<details><summary><code>maxKernelValueSpec</code></summary>

```
/-- Exact max value written by `max_kernel` at row lane `i`. -/
```
```lean
noncomputable def maxKernelValueSpec
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      (maxKernelInputTile s inp M N K BLOCK_M BLOCK_N) with
  | some out => WithBot.unbotD 0 (out.data (i, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>maxKernelIndexSpec</code></summary>

```
/-- Exact argmax index written by `max_kernel` at row lane `i`. -/
```
```lean
noncomputable def maxKernelIndexSpec
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩
    (maxKernelInputTile s inp M N K BLOCK_M BLOCK_N)).data (i, PUnit.unit)
```
</details>

<details><summary><code>maxKernelInputTile</code></summary>

```
/-- Input tile for the 2D value/index max kernel. Masked lanes are `⊥`,
matching `other=-float("inf")`. -/
```
```lean
noncomputable def maxKernelInputTile
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let m := s.pids 0 * BLOCK_M + idx.1.val
      let n := idx.2.1.val
      if m < M ∧ n < N then
        some (maxInpElem s inp N K m n)
      else none }
```
</details>

<details><summary><code>maxInpElem</code></summary>

```
/-- Input element `inp[m, n, pid1]` of the row-major `[M, N, K]` input tensor
(this program reduces along `n` at fixed trailing index `k = pids 1`). -/
```
```lean
noncomputable def maxInpElem (s : BlockState) (inp : RegionName)
    (N K m n : Nat) : ℝ :=
  s.readMem inp (m * N * K + n * K + s.pids 1)
```
</details>

## Public theorem: `max_kernel_1_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `max_reduction.py`'s `max_kernel_1`:
for every disjoint flat placement of the two buffers, every program id whose
active lanes are in bounds, and every launch state whose input window holds `xs`
at the active lanes, the translated pointer kernel terminates, `mid[pid]` holds
the genuine block max — `Tile.reduceMax` of `xs` over the active lanes, with the
`other=-float("inf")` padding modeled as `⊥` — and every other memory cell is
unchanged.

This is the first *reduction* on an `io ⊨ f` face, and it is what the spec `f`'s
program-id argument is for: the reduced index set is `{i | pid·BLOCK_SIZE + i < M}`,
so the value is irreducibly pid-dependent (the tail block reduces fewer lanes
than a full block) and a pid-independent spec would be falsifiable.

Dimension-general in `M` and `BLOCK_SIZE`. Honest side-condition:
`0 < BLOCK_SIZE` — with an empty tile `tl.max` has no axis to reduce and the
kernel faults, so termination genuinely fails there. -/
```
</details>

**Statement:**
```lean
specification max_kernel_1_io_correctness
    (inp mid : RegionName) (M BLOCK_SIZE : Nat) (hB : 0 < BLOCK_SIZE) :
    maxKernel1IO inp mid M BLOCK_SIZE
      ⊨ fun pid xs _ => maxTileSpecOf M BLOCK_SIZE pid xs
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `maxKernel1IO`, `maxTileSpecOf`, `max_kernel_1`

<details><summary><code>maxKernel1IO</code></summary>

```
/-- IO signature of `max_kernel_1` on the tile-indexed surface: every lane of
the `BLOCK_SIZE` window reads `inp` at `pid·BLOCK_SIZE + i` and is read-active on
the `< M` guard, while **only lane 0** is write-active and it writes the single
cell `mid[pid]` — the block's max. A single-cell store expressed as a one-lane
write mask over the read tile is exactly what a reduction's IO signature is. -/
```
```lean
def maxKernel1IO (inp mid : RegionName) (M BLOCK_SIZE : Nat) :
    MaskedTileKernelIO₁ where
  kernel := max_kernel_1 inp mid M BLOCK_SIZE
  inp := inp
  out := mid
  shape := [BLOCK_SIZE]
  read := fun pid idx => pid * BLOCK_SIZE + idx.1.val
  write := fun pid _ => pid
  mask := fun pid idx => pid * BLOCK_SIZE + idx.1.val < M
  writeMask := fun _ idx => idx.1.val = 0
```
</details>

<details><summary><code>maxTileSpecOf</code></summary>

```
/-- Value-level first-stage max spec: the `Tile.reduceMax` of the tile that holds
`xs` on the active lanes (`pid·BLOCK_SIZE + i < M`) and `⊥` elsewhere — the
`other=-float("inf")` padding. Written over the *loaded values* rather than over
memory, which is what the IO surface quantifies. -/
```
```lean
noncomputable def maxTileSpecOf (M BLOCK_SIZE pid : Nat)
    (xs : TileIndex [BLOCK_SIZE] → ℝ) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      ⟨fun idx =>
        if pid * BLOCK_SIZE + idx.1.val < M then some (xs idx) else none⟩ with
  | some out => WithBot.unbotD 0 (out.data PUnit.unit)
  | none => 0
```
</details>

<details><summary><code>max_kernel_1</code></summary>

```
/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def max_kernel_1
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}
```
</details>

## Also present (pinned special-case summaries)
- `max_kernel_1_compute_correct`
- `max_kernel_2_compute_correct`
- `max_kernel_compute_correct`
