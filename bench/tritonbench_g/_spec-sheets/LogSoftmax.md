# Spec sheet — `bench/tritonbench_g/log_softmax/LogSoftmax.lean`

**Python source:** `bench/tritonbench_g/log_softmax/log_softmax.py`

## Public theorem: `log_softmax_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `log_softmax_kernel`: the DSL surface lowers to
the algorithm layer, and the masked store to `output_ptr` is compute-correct —
every active cell holds `logSoftmaxSpec ...`, inactive cells are preserved.
Conditional on the output-offset injectivity side hypothesis `hOutInj`. -/
```
</details>

**Statement:**
```lean
specification log_softmax_kernel_output_summary
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] => outOffset s N K BLOCK_M idx)) :
    (∃ alg, (log_softmax_kernel output_ptr input_ptr M N K BLOCK_M BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := log_softmax_kernel output_ptr input_ptr M N K BLOCK_M BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_N] => active s M N BLOCK_M idx)
        (fun idx => (output_ptr, outOffset s N K BLOCK_M idx)))
      (expected := fun idx => logSoftmaxSpec s input_ptr M N K BLOCK_M BLOCK_N idx)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] => outOffset s N K BLOCK_M idx)`
- `fun idx : TileIndex [BLOCK_M, BLOCK_N] => active s M N BLOCK_M idx`

**Closed-form spec defs (transitive):** `outOffset`, `log_softmax_kernel`, `active`, `logSoftmaxSpec`, `mIndex`, `nIndex`, `logSoftmaxInputTile`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (N K BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  mIndex s BLOCK_M idx * N * K + nIndex idx * K + s.pids 1
```
</details>

<details><summary><code>log_softmax_kernel</code></summary>

```
/-- Faithful transcription of `log_softmax.py`'s `log_softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` -> Lean `Nat`
  parameters. -/
```
```lean
def log_softmax_kernel
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(tl.float32)
  row_minus_max = inp - tl.max(inp, axis=1)[:, None]
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=1)[:, None]
  softmax_output = tl.log(numerator / denominator)
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, softmax_output, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (M N BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  mIndex s BLOCK_M idx < M ∧ nIndex idx < N
```
</details>

<details><summary><code>logSoftmaxSpec</code></summary>

```lean
noncomputable def logSoftmaxSpec
    (s : BlockState) (input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  let inp := logSoftmaxInputTile s input_ptr M N K BLOCK_M BLOCK_N
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true inp with
  | some rowMax =>
      let rowBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, 1] [BLOCK_M, BLOCK_N] :=
        Broadcast.consSame (Broadcast.consR Broadcast.nil)
      let shifted := Tile.bop (NumericDType.sub .real) rowBroadcast inp rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true numerator
      WithBot.unbotD 0
        ((Tile.uop WithBot.realLog
          (Tile.bop (NumericDType.div .real) rowBroadcast numerator denominator)).data idx)
  | none => 0
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  s.pids 0 * BLOCK_M + idx.1.val
```
</details>

<details><summary><code>nIndex</code></summary>

```lean
def nIndex (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>logSoftmaxInputTile</code></summary>

```lean
noncomputable def logSoftmaxInputTile
    (s : BlockState) (input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active s M N BLOCK_M idx then
        some (s.readMem input_ptr (outOffset s N K BLOCK_M idx))
      else none }
```
</details>

## Also present (pinned special-case summaries)
- `log_softmax_kernel_compute_correct`
- `log_softmax_backward_kernel_compute_correct`
