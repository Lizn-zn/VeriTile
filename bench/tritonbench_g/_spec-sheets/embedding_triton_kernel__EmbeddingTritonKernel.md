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
specification embedding_kernel_output_summary
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

## Public theorem: `embedding_body_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for one body of `embedding_kernel`'s
`range(0, BLOCK_N, BLOCK_NN)` loop: for every disjoint flat placement of
`input_ids` / `weight` / `out`, every program id whose windows are in bounds,
every token-id row `ids` and weight tile `xs` the launch state holds on the
active lanes, the body terminates, every store-active cell of `out` holds

    if vob_start_id ≤ ids[row] < vob_end_id then xs[row, col] else 0

and every other memory cell is unchanged.

The token ids are a **channel**, not an assumption: the face quantifies over what
`input_ids` may hold and the `weight` window is a *gather* through it
(`(ids[row] − vob_start_id)·stride_weight_seq + col`), which is what
`GatherTileKernelIO` exists for. The out-of-vocabulary branch is Python's
`id_mask`, and it is genuine: an out-of-context lane loads the `other =
vob_end_id` sentinel, which fails `id_mask` outright, so no out-of-bounds row of
`weight` is ever addressed.

Dimension-general in `n_ctx`, `hiden_size`, `BLOCK_N`, `BLOCK_NN`,
`BLOCK_DMODEL`, both strides and the vocabulary window, and general in the loop
variable `start_nn`. Honest side-condition: destination injectivity at every
program id, the same hypothesis the launched kernel's summary takes. -/
```
</details>

**Statement:**
```lean
specification embedding_body_io_correctness
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOutInj : ∀ pid : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        (pid * BLOCK_N + start_nn + idx.1.val) * stride_out_seq
          + idx.2.1.val)) :
    bodyIO weight input_ids out vob_start_id vob_end_id stride_weight_seq
        stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn
      ⊨ fun _pid ids xs j =>
          if vob_start_id ≤ ids (j.1, PUnit.unit) ∧
              ids (j.1, PUnit.unit) < vob_end_id then xs j else 0
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        (pid * BLOCK_N + start_nn + idx.1.val) * stride_out_seq
          + idx.2.1.val`

**Closed-form spec defs (transitive):** `bodyIO`, `embedding_body_slice`

<details><summary><code>bodyIO</code></summary>

```
/-- IO signature of one loop body: the `[BLOCK_NN]` token-id row is a `.nat`
channel, and the `weight` window is a **gather** through it. -/
```
```lean
def bodyIO (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    GatherTileKernelIO where
  kernel := embedding_body_slice weight input_ids out vob_start_id vob_end_id
    stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
    BLOCK_NN start_nn
  idxbuf := input_ids
  inp := weight
  out := out
  shapeIdx := [BLOCK_NN]
  shape := [BLOCK_NN, BLOCK_DMODEL]
  readx := fun pid i => pid * BLOCK_N + start_nn + i.1.val
  read := fun _pid ids j =>
    (ids (j.1, PUnit.unit) - vob_start_id) * stride_weight_seq + j.2.1.val
  write := fun pid j =>
    (pid * BLOCK_N + start_nn + j.1.val) * stride_out_seq + j.2.1.val
  maskx := fun pid i => pid * BLOCK_N + start_nn + i.1.val < n_ctx
  readMask := fun _pid ids j =>
    (vob_start_id ≤ ids (j.1, PUnit.unit) ∧
      ids (j.1, PUnit.unit) < vob_end_id) ∧ j.2.1.val < hiden_size
  writeMask := fun pid j =>
    pid * BLOCK_N + start_nn + j.1.val < n_ctx ∧ j.2.1.val < hiden_size
```
</details>

<details><summary><code>embedding_body_slice</code></summary>

```
/-- One body of the `range(0, BLOCK_N, BLOCK_NN)` loop of `embedding_kernel`,
with the loop variable `start_nn` as a parameter. Statement for statement the
Python loop body, including the `other = vob_end_id` sentinel on the token load
(which makes an out-of-context lane fail `id_mask`) and the ℕ-truncated
`token_ids - vob_start_id`.

Two mechanical changes, both value-preserving: the pinned loop offset is folded
into the base (`start_n = pid·BLOCK_N + start_nn`, so `offs_seq = start_n +
tl.arange(0, BLOCK_NN)` is Python's `start_nn + offs_nn` lane for lane — the
resulting lane address is exactly `seqLaneIndex`), and the intermediate `offs_nn`
register disappears with it. -/
```
```lean
def embedding_body_slice
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0) * $(BLOCK_N) + $(start_nn)
  offs_seq = start_n + tl.arange(0, $(BLOCK_NN))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
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
```
</details>

## Public theorem: `embedding_body_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for one body of `embedding_kernel`'s
`range(0, BLOCK_N, BLOCK_NN)` loop: for **every** rounding model `R`, the same
gather Hoare triple as `embedding_body_io_correctness`, but run under `execR R`
and read back as `.real`-typed cells holding `R.round .real (f …)`.

The body carries no float cast and no arithmetic on the gathered row, so the
slice is cast-free and the exact run transports verbatim. The content of the
rounding face here is exactly that: *this body introduces no rounding event of
its own*, at any `R` — an embedding gather moves weight rows bit-for-bit. -/
```
</details>

**Statement:**
```lean
specification embedding_body_io_correctnessR (R : RoundingModel)
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOutInj : ∀ pid : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        (pid * BLOCK_N + start_nn + idx.1.val) * stride_out_seq
          + idx.2.1.val)) :
    bodyIO weight input_ids out vob_start_id vob_end_id stride_weight_seq
        stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn
      ⊨[R, FloatDType.real] fun _pid ids xs j =>
          if vob_start_id ≤ ids (j.1, PUnit.unit) ∧
              ids (j.1, PUnit.unit) < vob_end_id then xs j else 0
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        (pid * BLOCK_N + start_nn + idx.1.val) * stride_out_seq
          + idx.2.1.val`

**Closed-form spec defs (transitive):** `bodyIO`, `embedding_body_slice`

<details><summary><code>bodyIO</code></summary>

```
/-- IO signature of one loop body: the `[BLOCK_NN]` token-id row is a `.nat`
channel, and the `weight` window is a **gather** through it. -/
```
```lean
def bodyIO (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    GatherTileKernelIO where
  kernel := embedding_body_slice weight input_ids out vob_start_id vob_end_id
    stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
    BLOCK_NN start_nn
  idxbuf := input_ids
  inp := weight
  out := out
  shapeIdx := [BLOCK_NN]
  shape := [BLOCK_NN, BLOCK_DMODEL]
  readx := fun pid i => pid * BLOCK_N + start_nn + i.1.val
  read := fun _pid ids j =>
    (ids (j.1, PUnit.unit) - vob_start_id) * stride_weight_seq + j.2.1.val
  write := fun pid j =>
    (pid * BLOCK_N + start_nn + j.1.val) * stride_out_seq + j.2.1.val
  maskx := fun pid i => pid * BLOCK_N + start_nn + i.1.val < n_ctx
  readMask := fun _pid ids j =>
    (vob_start_id ≤ ids (j.1, PUnit.unit) ∧
      ids (j.1, PUnit.unit) < vob_end_id) ∧ j.2.1.val < hiden_size
  writeMask := fun pid j =>
    pid * BLOCK_N + start_nn + j.1.val < n_ctx ∧ j.2.1.val < hiden_size
```
</details>

<details><summary><code>embedding_body_slice</code></summary>

```
/-- One body of the `range(0, BLOCK_N, BLOCK_NN)` loop of `embedding_kernel`,
with the loop variable `start_nn` as a parameter. Statement for statement the
Python loop body, including the `other = vob_end_id` sentinel on the token load
(which makes an out-of-context lane fail `id_mask`) and the ℕ-truncated
`token_ids - vob_start_id`.

Two mechanical changes, both value-preserving: the pinned loop offset is folded
into the base (`start_n = pid·BLOCK_N + start_nn`, so `offs_seq = start_n +
tl.arange(0, BLOCK_NN)` is Python's `start_nn + offs_nn` lane for lane — the
resulting lane address is exactly `seqLaneIndex`), and the intermediate `offs_nn`
register disappears with it. -/
```
```lean
def embedding_body_slice
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0) * $(BLOCK_N) + $(start_nn)
  offs_seq = start_n + tl.arange(0, $(BLOCK_NN))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
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
```
</details>

## Public theorem: `embedding_kernel_whole_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The whole-kernel `⊨[R]` headline** for `embedding_kernel` — the launched
kernel, `for start_nn in range(0, BLOCK_N, BLOCK_NN)` loop included, not a
hand-cut body.

For every rounding model `R`, every disjoint flat placement of
`input_ids` / `weight` / `out`, every program id whose index and gather windows
are in bounds, and every launch state whose index row holds `ids` and whose
gathered weight rows hold `xs`: the translated pointer kernel run under
`execR R` terminates, every write-active lane of the output tile reads back at
`.real` holding the gathered row (`xs j` on an in-vocabulary lane, `0` outside —
the kernel's `id_mask` branch), and every other memory cell is unchanged.

Honest side conditions, all inherited from the port's whole-kernel value result
rather than added here:

* `hOne` is discharged by instantiating the signature at `BLOCK_NN = 1` — the
  chunking `embedding_kernel_compute_correct` assumes;
* `hOutInj` — no two lanes share an output cell;
* `input_ids ≠ out` and `weight ≠ out` — the loop writes only `out`, and these
  are what keep the token ids and weight rows stable across iterations.

The kernel is cast-free, so the `R.round .real` the surface applies is the
identity: this face's rounding content is that *the embedding gather introduces
no rounding event of its own*, which is the truthful reading for a kernel that
only moves weight rows. -/
```
</details>

**Statement:**
```lean
specification embedding_kernel_whole_io_correctnessR (R : RoundingModel)
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N : Nat)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hOutInj : ∀ pid : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        (pid * BLOCK_N + idx.1.val) * stride_out_seq + idx.2.1.val)) :
    wholeIO weight input_ids out vob_start_id vob_end_id stride_weight_seq
        stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      ⊨[R, FloatDType.real] fun _pid ids xs j =>
          if vob_start_id ≤ ids (j.1, PUnit.unit) ∧
              ids (j.1, PUnit.unit) < vob_end_id then xs j else 0
```

**Assumptions / layout contracts:**
- `hInputOutNe : input_ids ≠ out`
- `hWeightOutNe : weight ≠ out`
- `fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        (pid * BLOCK_N + idx.1.val) * stride_out_seq + idx.2.1.val`

**Closed-form spec defs (transitive):** `wholeIO`, `embedding_kernel`

<details><summary><code>wholeIO</code></summary>

```
/-- IO signature of the **whole** `embedding_kernel` on the row-gather surface:
the index tile is the program's full `[BLOCK_N]` slice of `input_ids`, the data
and output tiles are `[BLOCK_N, BLOCK_DMODEL]`, and the gather address is built
from the loaded token ids. Compare `bodyIO`, which is the same skin at one
chunk. -/
```
```lean
def wholeIO (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N : Nat) : GatherTileKernelIO where
  kernel := embedding_kernel weight input_ids out vob_start_id vob_end_id
    stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N 1
  idxbuf := input_ids
  inp := weight
  out := out
  shapeIdx := [BLOCK_N]
  shape := [BLOCK_N, BLOCK_DMODEL]
  readx := fun pid i => pid * BLOCK_N + i.1.val
  read := fun _pid ids j =>
    (ids (j.1, PUnit.unit) - vob_start_id) * stride_weight_seq + j.2.1.val
  write := fun pid j => (pid * BLOCK_N + j.1.val) * stride_out_seq + j.2.1.val
  maskx := fun pid i => pid * BLOCK_N + i.1.val < n_ctx
  readMask := fun _pid ids j =>
    (vob_start_id ≤ ids (j.1, PUnit.unit) ∧
      ids (j.1, PUnit.unit) < vob_end_id) ∧ j.2.1.val < hiden_size
  writeMask := fun pid j => pid * BLOCK_N + j.1.val < n_ctx ∧ j.2.1.val < hiden_size
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

## Also present (pinned special-case summaries)
- `embedding_kernel_compute_correct`
