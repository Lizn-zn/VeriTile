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
theorem softmax_kernel_online_v2_one_tile_output_summary
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    (∃ alg, (softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, linearOffset s N i)))
      (expected := fun i => softmaxOptimizeSpec s input_ptr N TILE_N i)
```

**Assumptions / layout contracts:**
- `kernel : = softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N`
- `initialState : = s`
- `fun i : Fin TILE_N => i.val < N`
- `expected : = fun i => softmaxOptimizeSpec s input_ptr N TILE_N i`

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

## Also present (pinned special-case summaries)
- `softmax_kernel_online_v2_one_tile_compute_correct`
