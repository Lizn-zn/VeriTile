# Spec sheet — `bench/tritonbench_g/embedding_triton_kernel/EmbeddingTritonKernel.lean`

**Python source:** `bench/tritonbench_g/embedding_triton_kernel/embedding_triton_kernel.py`

## Public theorem: `embedding_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `embedding_kernel`: the DSL surface lowers to
the algorithm layer, and the full embedding-gather loop is compute-correct — under
the no-duplicate-destination hypothesis `hOutInj`, the chunking `hOne`, and the
alias side conditions, every store-active cell holds the gathered weight row
`embeddingSpecFull`. -/
```
</details>

**Statement:**
```lean
theorem embedding_kernel_output_summary
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hOne : BLOCK_NN = 1)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out) :
    (∃ alg, (embedding_kernel weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN).toAlgorithm? = Except.ok alg) ∧
    embedding_kernel_correct_target weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx)`
- `hOne : BLOCK_NN = 1`
- `hInputOutNe : input_ids ≠ out`
- `hWeightOutNe : weight ≠ out`

**Closed-form spec defs (transitive):** `outOffsetFull`, `embedding_kernel`, `embedding_kernel_correct_target`, `fullSeqIndex`, `dimIndex`, `storeActiveFull`, `embeddingSpecFull`, `tokenRawFull`, `weightOffsetFull`, `tokenIndexFull`

<details><summary><code>outOffsetFull</code></summary>

```lean
def outOffsetFull
    (s : BlockState) (stride_out_seq BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Nat :=
  fullSeqIndex s BLOCK_N idx.1 * stride_out_seq + dimIndex idx.2.1
```
</details>

<details><summary><code>embedding_kernel</code></summary>

```
/-- Faithful transcription of `embedding_triton_kernel.py`'s
`embedding_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N` / `BLOCK_NN` / `BLOCK_DMODEL` / `hiden_size: tl.constexpr`
  → Lean `Nat` parameters.
- Python `[:, None]` / `[None, :]` dimension annotations preserved.

The proof below connects the full `range(0, BLOCK_N, BLOCK_NN)` embedding loop
to `ComputeCorrect.Realizes_without_Rounding` under the stated no-collision/no-alias
hypotheses. -/
```
```lean
def embedding_kernel
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0) * $(BLOCK_N)
  offs_nn = start_n + tl.arange(0, $(BLOCK_NN))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  for start_nn in range(0, $(BLOCK_N), $(BLOCK_NN)) {
    start_nn = tl.multiple_of(start_nn, $(BLOCK_NN))
    offs_seq = start_nn + offs_nn
    n_ctx_mask = offs_seq < $(n_ctx)
    token_ids = tl.load($((input_ids : Region .nat)) + offs_seq, mask=n_ctx_mask, other=$(vob_end_id))
      id_mask = (token_ids >= $(vob_start_id)) & (token_ids < $(vob_end_id))
      token_ids = token_ids - $(vob_start_id)
      dim_mask = offs_d < $(hiden_size)
      load_mask = id_mask[:, None] & dim_mask[None, :]
      store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
    vecs = tl.load(weight + token_ids[:, None] * $(stride_weight_seq) + offs_d[None, :],
      mask=load_mask, other=0.0)
    tl.store(out + offs_seq[:, None] * $(stride_out_seq) + offs_d[None, :], vecs, mask=store_mask)
  }
}
```
</details>

<details><summary><code>embedding_kernel_correct_target</code></summary>

```lean
def embedding_kernel_correct_target
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState) : Prop :=
  ComputeCorrect.Realizes_without_Rounding
    (kernel := embedding_kernel weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL)
      (fun idx => (out, outOffsetFull s stride_out_seq BLOCK_N idx)))
    (expected := fun idx =>
      embeddingSpecFull s weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N BLOCK_DMODEL idx)
```
</details>

<details><summary><code>fullSeqIndex</code></summary>

```lean
def fullSeqIndex
    (s : BlockState) (BLOCK_N : Nat) (lane : Fin BLOCK_N) : Nat :=
  s.pids 0 * BLOCK_N + lane.val
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (i : Fin BLOCK_DMODEL) : Nat :=
  i.val
```
</details>

<details><summary><code>storeActiveFull</code></summary>

```lean
def storeActiveFull
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Prop :=
  fullSeqIndex s BLOCK_N idx.1 < n_ctx ∧ dimIndex idx.2.1 < hiden_size
```
</details>

<details><summary><code>embeddingSpecFull</code></summary>

```lean
noncomputable def embeddingSpecFull
    (s : BlockState) (weight input_ids : RegionName)
    (vob_start_id vob_end_id stride_weight_seq BLOCK_N BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if vob_start_id ≤ tokenRawFull s input_ids BLOCK_N idx.1 ∧
        tokenRawFull s input_ids BLOCK_N idx.1 < vob_end_id then
      some (s.readMem weight
        (weightOffsetFull s input_ids vob_start_id stride_weight_seq BLOCK_N idx))
    else
      some (0.0 : ℝ))
```
</details>

<details><summary><code>tokenRawFull</code></summary>

```lean
def tokenRawFull
    (s : BlockState) (input_ids : RegionName) (BLOCK_N : Nat)
    (lane : Fin BLOCK_N) : Nat :=
  s.readMemValue .nat input_ids (fullSeqIndex s BLOCK_N lane)
```
</details>

<details><summary><code>weightOffsetFull</code></summary>

```lean
def weightOffsetFull
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id stride_weight_seq BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Nat :=
  tokenIndexFull s input_ids vob_start_id BLOCK_N idx.1 * stride_weight_seq +
    dimIndex idx.2.1
```
</details>

<details><summary><code>tokenIndexFull</code></summary>

```lean
def tokenIndexFull
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id BLOCK_N : Nat) (lane : Fin BLOCK_N) : Nat :=
  tokenRawFull s input_ids BLOCK_N lane - vob_start_id
```
</details>

## Also present (pinned special-case summaries)
- `embedding_kernel_compute_correct`
