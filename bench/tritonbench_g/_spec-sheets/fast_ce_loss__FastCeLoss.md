# Spec sheet — `bench/tritonbench_g/fast_ce_loss/FastCeLoss.lean`

**Python source:** `bench/tritonbench_g/fast_ce_loss/fast_ce_loss.py`

## Public theorem: `cross_entropy_forward_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_cross_entropy_forward` implements the pure per-row
cross-entropy pair on its metadata-genre IO signature — for every disjoint
flat placement of the four buffers, every program `row_idx = pid₀` whose
declared cells/lanes are in bounds, and every launch state pinning the label
`lab` at the `.int` slot cell, the block logits `xs` on the active lanes, and
the gather cell `g` under its `lab ≠ -100` gate, the translated pointer kernel
terminates and writes

* `logsumexp_ptr[pid₀] = fceLseLocal … xs` — the log-sum-exp of the valid
  transformed lanes, and
* `loss_ptr[pid₀] = fceLossLocal … lab xs g` — `0` for the ignored `-100`
  label, else `LSE − transform(g)`,

and every other memory cell is unchanged. The `-100` sentinel branch is
**genuine**: the label rides the `.int` channel, so an ignored row really
takes the `loss = 0` path. `0 < VOCAB_SIZE` (at least one valid lane) and
`0 < BLOCK_SIZE` feed the `max` reduce; `logsumexp_ptr ≠ loss_ptr` is the one
output-distinctness side condition (matching the old summary's `hne`).
`DO_SOFTCAPPING = false` is pinned (softcap breaks `⊥`-propagation on masked
lanes; see `fastCeTransform`); `DO_LOGIT_SCALING` parametrizes the spec.
Proof: `MetaGatherMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification cross_entropy_forward_correctness
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (hV : 0 < VOCAB_SIZE) (hB : 0 < BLOCK_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr) :
    fastCeForwardIO logits_ptr loss_ptr logsumexp_ptr labels_ptr VOCAB_SIZE
        logits_row_stride BLOCK_SIZE SOFTCAP LOGIT_SCALE DO_LOGIT_SCALING ⊨
      fun _ _ lab xs g =>
        (fceLseLocal VOCAB_SIZE BLOCK_SIZE LOGIT_SCALE DO_LOGIT_SCALING xs,
         fceLossLocal VOCAB_SIZE BLOCK_SIZE LOGIT_SCALE DO_LOGIT_SCALING
           lab xs g)
```

**Assumptions / layout contracts:**
- `hV : 0 < VOCAB_SIZE`
- `hB : 0 < BLOCK_SIZE`
- `hne : logsumexp_ptr ≠ loss_ptr`

**Closed-form spec defs (transitive):** `fastCeForwardIO`, `fceLseLocal`, `fceLossLocal`, `cross_entropy_forward_surface`

<details><summary><code>fastCeForwardIO</code></summary>

```
/-- `cross_entropy_forward_surface`'s metadata-genre **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`MetaGatherMasked2DKernelIO₂ₓ₂`, the cross-entropy metadata shape):

* `mbufL` — the `.int` label slot: program `row_idx = pid₀` loads
  `labels_ptr[pid₀]` (`mwinL`), yielding the named `Int` scalar `lab`;
* `inp = logits_ptr` — read twice: the masked row block (`read`/`mask`: lane
  `j` at `pid₀·stride + j`, active while `j < VOCAB_SIZE`) and the
  sentinel-gated single-cell gather (`gwin`/`gmask`: cell
  `pid₀·stride + lab.toNat`, read exactly when `lab ≠ -100`);
* `out1 = logsumexp_ptr`, `out2 = loss_ptr` — the two per-row cells, both at
  offset `pid₀`, written unconditionally (`writeMask` defaults).

`DO_SOFTCAPPING` is pinned `false` (see the softcap modeling note above);
`DO_LOGIT_SCALING` stays a spec parameter. The slot cell, windows, gates, and
masks are declared, not parsed from the kernel; the headline **proves** the
kernel's actual slot load, addressing, gating, and masking match them. -/
```
```lean
def fastCeForwardIO
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool) :
    MetaGatherMasked2DKernelIO₂ₓ₂ where
  kernel := cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
    labels_ptr VOCAB_SIZE logits_row_stride BLOCK_SIZE SOFTCAP LOGIT_SCALE
    Bool.false DO_LOGIT_SCALING
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := logsumexp_ptr
  out2 := loss_ptr
  B := BLOCK_SIZE
  mwinL := fun pid₀ _ => pid₀
  read := fun pid₀ _ _ j => pid₀ * logits_row_stride + j.val
  mask := fun _ _ _ j => j.val < VOCAB_SIZE
  gwin := fun pid₀ _ lab => pid₀ * logits_row_stride + lab.toNat
  gmask := fun _ _ lab => lab ≠ -100
  write1 := fun pid₀ _ _ => pid₀
  write2 := fun pid₀ _ _ => pid₀
```
</details>

<details><summary><code>fceLseLocal</code></summary>

```
/-- Pure per-program value of the `logsumexp` output cell: the plain
log-sum-exp `log (∑ exp)` over the valid lanes (`j < VOCAB_SIZE`) of the
per-lane transformed block values (transform = optional `LOGIT_SCALE * ·`,
the `DO_SOFTCAPPING = false` regime). The kernel's max-shifted stable form
collapses to it via `fastCeLseSpec_eq_log_sum`. Built only from the pinned
block values `xs`. -/
```
```lean
noncomputable def fceLseLocal (VOCAB_SIZE B : Nat) (LOGIT_SCALE : ℝ)
    (DO_LOGIT_SCALING : Bool) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j ∈ Finset.univ.filter (fun j : Fin B => j.val < VOCAB_SIZE),
    Real.exp (if DO_LOGIT_SCALING then LOGIT_SCALE * xs j else xs j))
```
</details>

<details><summary><code>fceLossLocal</code></summary>

```
/-- Pure per-program value of the `loss` output cell: `0` for the ignored
`-100` sentinel label, otherwise `LSE − transform(g)` with `g` the gathered
label-logit cell. Built only from the pinned label, block values, and gather
cell. -/
```
```lean
noncomputable def fceLossLocal (VOCAB_SIZE B : Nat) (LOGIT_SCALE : ℝ)
    (DO_LOGIT_SCALING : Bool) (lab : Int) (xs : Fin B → ℝ) (g : ℝ) : ℝ :=
  if lab = -100 then 0
  else fceLseLocal VOCAB_SIZE B LOGIT_SCALE DO_LOGIT_SCALING xs
    - (if DO_LOGIT_SCALING then LOGIT_SCALE * g else g)
```
</details>

<details><summary><code>cross_entropy_forward_surface</code></summary>

```
/-- Faithful transcription of `fast_ce_loss.py`'s `_cross_entropy_forward`.

Python's hard-coded `label_idx != -100` sentinel is preserved as the literal
`-100`, and the label load rides the typed `.int` region channel so the
sentinel comparison is a genuine signed comparison. The two label-channel
deviations — the pointer bump folded into the load offset, and the dropped
`.to(tl.int32)` — are the registered `Translation-surface blocker:` in the
module preamble above; both are forced by the `MetaGatherMasked2DKernelIO₂ₓ₂`
skin's `mwinL` address shape, not by the `.int` channel itself. -/
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
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr + row_idx)
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

## Public theorem: `chunked_cross_entropy_forward_output_summary`

<details><summary>docstring</summary>

```
/-- **Per-kernel chunked-forward output summary for
`chunked_cross_entropy_forward_surface` (genuine, end-to-end, chunk 0, no
softcapping).**

The chunked surface stores both side outputs only under `chunk_idx == 0`. Stated
as a conjunction of `ComputeCorrect.Realizes_without_Rounding` claims for chunk `0` with
`DO_SOFTCAPPING = false`, at least one valid lane, and `logsumexp_ptr ≠ loss_ptr`,
bundling:
1. **genuine per-chunk LSE output**: `logsumexp_ptr[row * N_CHUNKS + 0]` holds
   exactly `fastCeLseSpec` of the per-lane transformed INPUT chunk-0 logits;
2. **genuine chunk-0 partial loss output**: `loss_ptr[row]` holds exactly
   `-1 * transform(label logit)`, read from INPUT memory.

Each `ComputeCorrect.Realizes_without_Rounding` internalizes the execution (`exec ... = some s'`)
and the lowering to the algorithm layer. All value specs read INPUT memory;
non-self-referential. The softcapping branch is out of scope; the `-100` ignore
label is dead under cast-to-`Nat` erasure. -/
```
</details>

**Statement:**
```lean
specification chunked_cross_entropy_forward_output_summary
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride : Nat)
    (SOFTCAP LOGIT_SCALE : ℝ) (DO_LOGIT_SCALING : Bool)
    (n : Nat)
    (s : BlockState)
    (hchunk : s.pids 1 = 0)
    (h_tail : 0 * (n+1) < VOCAB_SIZE)
    (hne : logsumexp_ptr ≠ loss_ptr) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cross_entropy_forward_surface logits_ptr loss_ptr logsumexp_ptr
        labels_ptr VOCAB_SIZE N_CHUNKS logits_row_stride (n+1) SOFTCAP LOGIT_SCALE
        Bool.false DO_LOGIT_SCALING)
      (initialState := s)
      (write := fun _ : PUnit => some (logsumexp_ptr, fceChunkLseOffset s N_CHUNKS))
      (expected := fun _ =>
        fastCeLseSpec (fastCeRowLogits s logits_ptr logits_row_stride VOCAB_SIZE)
          0 h_tail (fun x => if DO_LOGIT_SCALING then LOGIT_SCALE * x else x))) ∧
    (ComputeCorrect.Realizes_without_Rounding
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
read from INPUT memory at the data-dependent (Nat) label position — the
chunked kernel's gather value. -/
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
/-- The label value loaded by the **chunked** forward kernel (whose label
load still goes through a dynamic pointer register and is therefore carried
as a `Nat` offset under the algorithm-layer cast erasure — see the modeling
boundary note; the non-chunked forward now rides the `.int` channel). -/
```
```lean
noncomputable def fceLabelNat (s : BlockState) (labels_ptr : Region .int) : Nat :=
  s.readMemValue .nat (Region.cast labels_ptr) (s.pids 0)
```
</details>

## Also present (pinned special-case summaries)
- `cross_entropy_backward_store_slice_compute_correct`
