# Spec sheet — `bench/tritonbench_g/fast_ce_loss/FastCeLoss.lean`

**Python source:** `bench/tritonbench_g/fast_ce_loss/fast_ce_loss.py`

## Public theorem: `cross_entropy_forward_output_summary`

<details><summary>docstring</summary>

```
/-- **Per-kernel forward output summary for `cross_entropy_forward_surface`
(genuine, end-to-end, no softcapping).**

Stated as a conjunction of `ComputeCorrect.Realizes` claims with
`DO_SOFTCAPPING = false`, at least one valid lane (`0 < VOCAB_SIZE`), and
`logsumexp_ptr ≠ loss_ptr`, bundling:
1. **genuine LSE output**: `logsumexp_ptr[row]` holds exactly `fastCeLseSpec` —
   the masked-lane stable log-sum-exp of the per-lane transformed INPUT row
   logits (transform = optional `LOGIT_SCALE * ·`);
2. **genuine loss output**: `loss_ptr[row]` holds exactly
   `fastCeLseSpec − transform(label logit)`, every term read from INPUT memory.

Each `ComputeCorrect.Realizes` internalizes the execution (`exec ... = some s'`)
and the lowering to the algorithm layer. All value specs read INPUT memory, never
`exec(...).readMem`, so this summary is non-self-referential. The
region-distinctness and one-valid-lane hypotheses are the only side-conditions.
The softcapping branch is out of scope (see `fastCeTransform`'s `⊥`-propagation
note); the `-100` ignore label is dead under the cast-to-`Nat` erasure. -/
```
</details>

**Statement:**
```lean
theorem cross_entropy_forward_output_summary
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s : BlockState)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr) :
    (ComputeCorrect.Realizes
      (kernel := cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (logsumexp_ptr, fceOutOffset s))
      (expected := fun _ =>
        fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x))) ∧
    (ComputeCorrect.Realizes
      (kernel := cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr labels_ptr
        VOCAB_SIZE logits_row_stride (n+1) SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, fceOutOffset s))
      (expected := fun _ =>
        fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x)
        - fceLabelLogit s logits_ptr labels_ptr logits_row_stride
            SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING))
```

**Assumptions / layout contracts:**
- `h_tail : 0 * (n+1) < VOCAB_SIZE`
- `hne : logsumexp_ptr ≠ loss_ptr`

**Closed-form spec defs (transitive):** `cross_entropy_forward_surface`, `fceOutOffset`, `fastCeLseSpec`, `fastCeRowLogits`, `fceLabelLogit`, `fastCeTransform`, `fceLabelNat`

<details><summary><code>cross_entropy_forward_surface</code></summary>

```
/-- Faithful transcription of `fast_ce_loss.py`'s `_cross_entropy_forward`.

Python's hard-coded `label_idx != -100` sentinel is preserved as the literal
`-100`. -/
```
```lean
def cross_entropy_forward_surface
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
  loss_ptr += row_idx
  logsumexp_ptr += row_idx
  labels_ptr += row_idx
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = (tl.load(labels_ptr)).to(tl.int32)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $(LOGIT_SCALE) * logits
  }
  if DO_SOFTCAPPING {
    logits = $(SOFTCAP) * triton_tanh(logits / $(SOFTCAP))
  }
  logits = (logits).to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if label_idx != $((-100 : Int)) {
    x = tl.load(logits_ptr + label_idx)
    if DO_LOGIT_SCALING {
      x = $(LOGIT_SCALE) * x
    }
    if DO_SOFTCAPPING {
      x = $(SOFTCAP) * triton_tanh(x / $(SOFTCAP))
    }
    loss = logsumexp - (x).to(tl.float32)
  } else {
    loss = 0.0
  }
  tl.store(logsumexp_ptr, logsumexp)
  tl.store(loss_ptr, loss)
}
```
</details>

<details><summary><code>fceOutOffset</code></summary>

```
/-- Output offset for the single-program forward kernel: `logsumexp_ptr` and
`loss_ptr` are both indexed at `row_idx = pid`. -/
```
```lean
def fceOutOffset (s : BlockState) : Nat := s.pids 0
```
</details>

<details><summary><code>fastCeLseSpec</code></summary>

```
/-- Genuine stable log-sum-exp of the transformed masked logits for the tail
block `i_d` (block size `n+1`): `m + log(∑ exp(transform(raw) - m))` over the
valid lanes, where `m` is the max over valid lanes. Mirrors `partialLSE_full`
but with an arbitrary per-lane transform `g` (here the `mul`/`id` part of
`fastCeTransform`). -/
```
```lean
noncomputable def fastCeLseSpec
    {VOCAB_SIZE : Nat} (xs : Fin VOCAB_SIZE → ℝ) {n : Nat} (i_d : Nat)
    (h_tail : i_d * (n+1) < VOCAB_SIZE)
    (g : ℝ → ℝ) : ℝ :=
  let vl := validLanes n VOCAB_SIZE i_d
  let lane : Fin (n+1) → ℝ := fun i =>
    if h : i_d * (n+1) + i.val < VOCAB_SIZE then g (xs ⟨i_d * (n+1) + i.val, h⟩) else 0
  let m := vl.sup' (validLanes_nonempty h_tail) lane
  m + Real.log (∑ i ∈ vl, Real.exp (lane i - m))
```
</details>

<details><summary><code>fastCeRowLogits</code></summary>

```
/-- Row-logits function for the single-program forward kernel: lane `i` reads
INPUT memory `logits_ptr` at `row_idx * logits_row_stride + i`. -/
```
```lean
noncomputable def fastCeRowLogits
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride VOCAB_SIZE : Nat) (j : Fin VOCAB_SIZE) : ℝ :=
  s.readMem logits_ptr (s.pids 0 * logits_row_stride + j.val)
```
</details>

<details><summary><code>fceLabelLogit</code></summary>

```
/-- The transformed label logit `transform(tl.load(logits_ptr + label_idx))`
read from INPUT memory at the data-dependent (Nat) label position. -/
```
```lean
noncomputable def fceLabelLogit
    (s : BlockState) (logits_ptr : RegionName) (labels_ptr : Region .int)
    (logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) : ℝ :=
  fastCeTransform SOFTCAP LOGIT_SCALE DO_SOFTCAPPING DO_LOGIT_SCALING
    (s.readMem logits_ptr (s.pids 0 * logits_row_stride + fceLabelNat s labels_ptr))
```
</details>

<details><summary><code>fastCeTransform</code></summary>

```
/-- The per-lane logit transform `fast_ce_loss.py` applies before the reduction:
optional `LOGIT_SCALE * x`, then optional `SOFTCAP * tanh(x / SOFTCAP)`. -/
```
```lean
noncomputable def fastCeTransform
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) (x : ℝ) : ℝ :=
  let scaled := if DO_LOGIT_SCALING then LOGIT_SCALE * x else x
  if DO_SOFTCAPPING then SOFTCAP * Real.tanh (scaled / SOFTCAP) else scaled
```
</details>

<details><summary><code>fceLabelNat</code></summary>

```
/-- The label value loaded by the forward kernel:
`label_idx = (tl.load(labels_ptr + row_idx)).to(tl.int32)`, read from INPUT
memory and (per the algorithm-layer cast erasure) carried as a `Nat` offset. -/
```
```lean
noncomputable def fceLabelNat (s : BlockState) (labels_ptr : Region .int) : Nat :=
  s.readMemValue .nat (Region.cast labels_ptr) (s.pids 0)
```
</details>

## Public theorem: `chunked_cross_entropy_forward_output_summary`

<details><summary>docstring</summary>

```
/-- **Per-kernel chunked-forward output summary for
`chunked_cross_entropy_forward_surface` (genuine, end-to-end, chunk 0, no
softcapping).**

The chunked surface stores both side outputs only under `chunk_idx == 0`. Stated
as a conjunction of `ComputeCorrect.Realizes` claims for chunk `0` with
`DO_SOFTCAPPING = false`, at least one valid lane, and `logsumexp_ptr ≠ loss_ptr`,
bundling:
1. **genuine per-chunk LSE output**: `logsumexp_ptr[row * N_CHUNKS + 0]` holds
   exactly `fastCeLseSpec` of the per-lane transformed INPUT chunk-0 logits;
2. **genuine chunk-0 partial loss output**: `loss_ptr[row]` holds exactly
   `-1 * transform(label logit)`, read from INPUT memory.

Each `ComputeCorrect.Realizes` internalizes the execution (`exec ... = some s'`)
and the lowering to the algorithm layer. All value specs read INPUT memory;
non-self-referential. The softcapping branch is out of scope; the `-100` ignore
label is dead under cast-to-`Nat` erasure. -/
```
</details>

**Statement:**
```lean
theorem chunked_cross_entropy_forward_output_summary
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s : BlockState)
    (hchunk : s.pids 1 = 0)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr) :
    (ComputeCorrect.Realizes
      (kernel := chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
        labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
        Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (logsumexp_ptr, fceChunkLseOffset s N_CHUNKS))
      (expected := fun _ =>
        fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
        labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
        Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, fceOutOffset s))
      (expected := fun _ =>
        (-1 : ℝ) * fceLabelLogit s logits_ptr labels_ptr logits_row_stride
          SOFTCAP LOGIT_SCALE Bool.false DO_LOGIT_SCALING))
```

**Assumptions / layout contracts:**
- `hchunk : s.pids 1 = 0`
- `h_tail : 0 * (n+1) < VOCAB_SIZE`
- `hne : logsumexp_ptr ≠ loss_ptr`

**Closed-form spec defs (transitive):** `chunked_cross_entropy_forward_surface`, `fceChunkLseOffset`, `fastCeLseSpec`, `fastCeRowLogits`, `fceOutOffset`, `fceLabelLogit`, `fastCeTransform`, `fceLabelNat`

<details><summary><code>chunked_cross_entropy_forward_surface</code></summary>

```
/-- Surface transcription of `fast_ce_loss.py`'s
`_chunked_cross_entropy_forward`.

Python's hard-coded `label_idx != -100` sentinel is preserved as the literal
`-100`. -/
```
```lean
def chunked_cross_entropy_forward_surface
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ)
    (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  chunk_idx = tl.program_id(1)
  logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
  loss_ptr += row_idx
  logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
  labels_ptr += row_idx
  col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = (tl.load(labels_ptr)).to(tl.int32)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $(LOGIT_SCALE) * logits
  }
  if DO_SOFTCAPPING {
    logits = $(SOFTCAP) * triton_tanh(logits / $(SOFTCAP))
  }
  logits = (logits).to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if chunk_idx == 0 {
    if label_idx != $((-100 : Int)) {
      x = (tl.load(logits_ptr + label_idx)).to(tl.float32)
      if DO_LOGIT_SCALING {
        x = $(LOGIT_SCALE) * x
      }
      if DO_SOFTCAPPING {
        x = $(SOFTCAP) * triton_tanh(x / $(SOFTCAP))
      }
      loss = -1.0 * (x).to(tl.float32)
    } else {
      loss = 0.0
    }
    tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)
  }
}
```
</details>

<details><summary><code>fceChunkLseOffset</code></summary>

```
/-- Chunked logsumexp output offset `row_idx * N_CHUNKS + chunk_idx`. -/
```
```lean
def fceChunkLseOffset (s : BlockState) (N_CHUNKS : Nat) : Nat :=
  s.pids 0 * N_CHUNKS + s.pids 1
```
</details>

<details><summary><code>fastCeLseSpec</code></summary>

```
/-- Genuine stable log-sum-exp of the transformed masked logits for the tail
block `i_d` (block size `n+1`): `m + log(∑ exp(transform(raw) - m))` over the
valid lanes, where `m` is the max over valid lanes. Mirrors `partialLSE_full`
but with an arbitrary per-lane transform `g` (here the `mul`/`id` part of
`fastCeTransform`). -/
```
```lean
noncomputable def fastCeLseSpec
    {VOCAB_SIZE : Nat} (xs : Fin VOCAB_SIZE → ℝ) {n : Nat} (i_d : Nat)
    (h_tail : i_d * (n+1) < VOCAB_SIZE)
    (g : ℝ → ℝ) : ℝ :=
  let vl := validLanes n VOCAB_SIZE i_d
  let lane : Fin (n+1) → ℝ := fun i =>
    if h : i_d * (n+1) + i.val < VOCAB_SIZE then g (xs ⟨i_d * (n+1) + i.val, h⟩) else 0
  let m := vl.sup' (validLanes_nonempty h_tail) lane
  m + Real.log (∑ i ∈ vl, Real.exp (lane i - m))
```
</details>

<details><summary><code>fastCeRowLogits</code></summary>

```
/-- Row-logits function for the single-program forward kernel: lane `i` reads
INPUT memory `logits_ptr` at `row_idx * logits_row_stride + i`. -/
```
```lean
noncomputable def fastCeRowLogits
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride VOCAB_SIZE : Nat) (j : Fin VOCAB_SIZE) : ℝ :=
  s.readMem logits_ptr (s.pids 0 * logits_row_stride + j.val)
```
</details>

<details><summary><code>fceOutOffset</code></summary>

```
/-- Output offset for the single-program forward kernel: `logsumexp_ptr` and
`loss_ptr` are both indexed at `row_idx = pid`. -/
```
```lean
def fceOutOffset (s : BlockState) : Nat := s.pids 0
```
</details>

<details><summary><code>fceLabelLogit</code></summary>

```
/-- The transformed label logit `transform(tl.load(logits_ptr + label_idx))`
read from INPUT memory at the data-dependent (Nat) label position. -/
```
```lean
noncomputable def fceLabelLogit
    (s : BlockState) (logits_ptr : RegionName) (labels_ptr : Region .int)
    (logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) : ℝ :=
  fastCeTransform SOFTCAP LOGIT_SCALE DO_SOFTCAPPING DO_LOGIT_SCALING
    (s.readMem logits_ptr (s.pids 0 * logits_row_stride + fceLabelNat s labels_ptr))
```
</details>

<details><summary><code>fastCeTransform</code></summary>

```
/-- The per-lane logit transform `fast_ce_loss.py` applies before the reduction:
optional `LOGIT_SCALE * x`, then optional `SOFTCAP * tanh(x / SOFTCAP)`. -/
```
```lean
noncomputable def fastCeTransform
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_SOFTCAPPING DO_LOGIT_SCALING : Bool) (x : ℝ) : ℝ :=
  let scaled := if DO_LOGIT_SCALING then LOGIT_SCALE * x else x
  if DO_SOFTCAPPING then SOFTCAP * Real.tanh (scaled / SOFTCAP) else scaled
```
</details>

<details><summary><code>fceLabelNat</code></summary>

```
/-- The label value loaded by the forward kernel:
`label_idx = (tl.load(labels_ptr + row_idx)).to(tl.int32)`, read from INPUT
memory and (per the algorithm-layer cast erasure) carried as a `Nat` offset. -/
```
```lean
noncomputable def fceLabelNat (s : BlockState) (labels_ptr : Region .int) : Nat :=
  s.readMemValue .nat (Region.cast labels_ptr) (s.pids 0)
```
</details>

## Also present (pinned special-case summaries)
- `cross_entropy_backward_store_slice_compute_correct`
