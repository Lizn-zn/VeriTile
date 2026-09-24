# Spec sheet — `bench/tritonbench_g/bgmv_expand_slice/BgmvExpandSlice.lean`

**Python source:** `bench/tritonbench_g/bgmv_expand_slice/bgmv_expand_slice.py`

## Public theorem: `bgmv_full_output_summary`

<details><summary>docstring</summary>

```
/-- **Full output summary** for the general `bgmv_expand_slice` kernel (arbitrary
`split_n_length`, multi-block `for n` loop, signed `-1` sentinel guard): the DSL
surface lowers to the algorithm layer, and the masked GEMV store to the
slice-offset output realizes the genuine rank-`K` reduction
`bgmvFullSpec g = Σ_k (k<K ? A[k] : 0)·(g<split_n_length ∧ k<K ? B[g,k] : 0)` at
every output lane `g < split_n_length`. Requires the active LoRA index
(`lora_index = Int.ofNat li ≥ 0`, so the `-1` early return is not taken),
out-of-place output (`out ≠ input`, `out ≠ lora`), and per-lane output-offset
injectivity. -/
```
</details>

**Statement:**
```lean
specification bgmv_full_output_summary
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (li K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N)
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride)
    (hlx : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = Int.ofNat li) :
    (∃ alg, (bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bgmv_full input_ptr lora_ptr out_ptr lora_indices K split_n_length xm_stride xk_stride
        l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin split_n_length => True)
        (fun g => (out_ptr, cOff s split_n_length cm_stride cn_stride slice_offset g.val)))
      (expected := fun g : Fin split_n_length =>
        bgmvFullSpec s input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K g.val)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hoi : out_ptr ≠ input_ptr`
- `hol : out_ptr ≠ lora_ptr`
- `hcn : 0 < cn_stride`
- `hlx : s.readMemValue .int (Region.cast lora_indices) (s.pids 1) = Int.ofNat li`

**Closed-form spec defs (transitive):** `bgmv_full`, `cOff`, `bgmvFullSpec`, `prodGK`, `aElem`, `bElem`

<details><summary><code>bgmv_full</code></summary>

```lean
def bgmv_full
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_k = tl.arange(0, $(BLOCK_K))
    offset_n = tl.arange(0, $(BLOCK_N))
    tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
      mask=offset_k < $(K), other=0)
    b_ptr = lora_ptr + $(l0_stride) * lora_index +
      pid_sn * $(split_n_length) * $(lora_k_stride)
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length) +
      $(slice_offset) * $(cn_stride)
    for n in range($(0), $(split_n_length), $(BLOCK_N)) {
      current_n = n + offset_n
      b_ptr_mask = (current_n[:, None] < $(split_n_length)) & (offset_k[None, :] < $(K))
      c_mask = current_n < $(split_n_length)
      tiled_b = tl.load(
        b_ptr + current_n[:, None] * $(lora_k_stride) +
          offset_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      accumulator = tl.sum(tiled_a * tiled_b, 1)
      tl.store(c_ptr + current_n * $(cn_stride), accumulator, mask=c_mask)
    }
  }
}
```
</details>

<details><summary><code>cOff</code></summary>

```lean
def cOff (s0 : BlockState) (split_n_length cm_stride cn_stride slice_offset : Nat) (g : Nat) : Nat :=
  s0.pids 1 * cm_stride + s0.pids 0 * split_n_length + slice_offset * cn_stride + g * cn_stride

-- masked product for global lane g, key k (over Fin BLOCK_K)
```
</details>

<details><summary><code>bgmvFullSpec</code></summary>

```lean
noncomputable def bgmvFullSpec (s0 : BlockState) (input_ptr lora_ptr : RegionName) (li : Nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_K : Nat) (g : Nat) : ℝ :=
  ∑ k : Fin BLOCK_K, prodGK s0 input_ptr lora_ptr li K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride g k.val

set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

-- matmul-style expandDim helpers
```
</details>

<details><summary><code>prodGK</code></summary>

```lean
noncomputable def prodGK (s0 : BlockState) (input_ptr lora_ptr : RegionName) (li : Nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride : Nat) (g k : Nat) : ℝ :=
  (if k < K then aElem s0 input_ptr xm_stride xk_stride k else 0) *
  (if g < split_n_length ∧ k < K then bElem s0 lora_ptr li split_n_length l0_stride lora_k_stride lora_n_stride g k else 0)

-- full spec for global lane g : sum over k : Fin BLOCK_K of prodGK
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- Input-vector element `A[cur_batch, k]`: this program's batch row
(`cur_batch = pid1`, row stride `xm_stride`) at rank lane `k` (lane stride
`xk_stride`). -/
```
```lean
noncomputable def aElem (s0 : BlockState) (input_ptr : RegionName) (xm_stride xk_stride : Nat) (k : Nat) : ℝ :=
  s0.readMem input_ptr (s0.pids 1 * xm_stride + k * xk_stride)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- LoRA-B weight element `B[li][pid_sn·split_n_length + g, k]`: LoRA index
`li` selects the `l0_stride` slab, the split (`pid_sn = pid0`) plus in-split
output lane `g` select the row (stride `lora_k_stride`), rank lane `k` the
column (stride `lora_n_stride`). -/
```
```lean
noncomputable def bElem (s0 : BlockState) (lora_ptr : RegionName) (li : Nat)
    (split_n_length l0_stride lora_k_stride lora_n_stride : Nat) (g k : Nat) : ℝ :=
  s0.readMem lora_ptr (l0_stride * li + s0.pids 0 * split_n_length * lora_k_stride + g * lora_k_stride + k * lora_n_stride)

-- output offset for global lane g (relative to out_ptr region)
```
</details>

## Public theorem: `bgmv_expand_slice_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming metadata emit headline (wave-5, MetaEmit
genre).** For every rounding model `R`, the verified `bgmv_full` config of
`_bgmv_expand_slice_kernel` (`EVEN_K = false`, `ADD_INPUTS = false`, no
`CAST_TYPE` — exactly the config the exact numeric stack above verifies)
implements, on its `StreamMetaEmitMasked3DKernelIO₂` signature, the
**ideal ℝ masked rank-`K` GEMV dot** at every write-active emitted lane:
step `t`, lane `j` of the output stream holds

```
Σ_{k < BLOCK_K} ([k < K]·a-tile[k]) · ([t·BLOCK_N + j < split_n_length ∧ k < K]·b-tile[t](j, k))
```

— exact real arithmetic over the pinned streams. The `.int` metadata slot
`m 0 = lora_indices[cur_batch]` enters the value contract **only** through
the sentinel gate and the `read2` base geometry:

* **Sentinel branch is genuine.** `writeMask` carries `m 0 ≠ -1`, so at the
  `-1` sentinel the write-active window family is empty, the readback is
  vacuous, and the frame clause proves the program leaves **all** of memory
  untouched (the guard skips the body). This is the io headline's added
  value over the exact `bgmv_full_output_summary`, whose `hlx` hypothesis
  pins `lora_index = Int.ofNat li ≥ 0` and says nothing about sentinel
  programs.
* **Negative non-sentinel values are honest, not hypothesized away.** The
  slot is quantified over all of `Int`; for `m 0 ∉ {-1} ∪ ℕ` the kernel
  *runs* the body with the `castIntToNat` clamp `(l0_stride·m 0).toNat = 0`
  on the `b_ptr` base, and `read2`'s `(m 0).toNat` reproduces exactly that
  clamp (the generalized `bptrGen_eval`), so the statement stays true
  without any sign hypothesis.

The kernel has **zero rounding events** (typed `.int` slot load, masked
`.real` loads, `.real` `tl.sum`, `.real` masked in-loop store; the only
casts are the exact int casts in `b_ptr`), so with the default
`outDType := .real` the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity and the per-step stores are
exact under `execR R` — the ∀-`R` face holds via the `RoundingModel`
`.real` identity fields, not as a `.triv` special case. Layer map: the
guarded body is cast-free, so under `execR R` it collapses verbatim onto
the exact stepper and the proven `preLoop` / `wbInv` / `wbStep` /
`forRange_inv` stack above is reused unchanged (with `li := (m 0).toNat`
threaded through `preLoopGen`); the `⊨[R]` face adds the `TraceSafeR` walk,
the per-cell memory frame (`bgmv_wbBody_step_frame`, the `mem` twin of
`wbStep`) and the stream-lane spec bridge (`bgmvSpec_eq_streamSum`).

All five hypotheses are truth-forced; provenance:

* `hBN : 0 < BLOCK_N` — the loop steps by `BLOCK_N`
  (`range(0, split_n_length, BLOCK_N)`); at `BLOCK_N = 0` the loop never
  advances and the step index `n / BLOCK_N` is meaningless. A launched
  constexpr tile is nonempty. (Same hypothesis as the exact headline.)
* `hsn : 0 < split_n_length` — `split_n_length = ⌈N / SPLIT_N⌉ ≥ 1` for
  every nonempty output (`N ≥ 1`). Needed because the trip count
  `T = ⌈split_n_length / BLOCK_N⌉` vanishes at `split_n_length = 0`, which
  erases the static `read1` window while the kernel's pre-loop `tiled_a`
  load still executes — the safety walk would have no bound to cite. The
  degenerate `N = 0` launch stores nothing and is out of scope.
* `hoi : out_ptr ≠ input_ptr`, `hol : out_ptr ≠ lora_ptr` — the loop
  stores into `out` **between** re-reads of the LoRA-B tiles (and the
  register-cached `tiled_a` was read from `input`); aliasing would let a
  later block read already-overwritten values. (Same as the exact
  headline.)
* `hcn : 0 < cn_stride` — output-lane footprint injectivity
  (`affine1D_inj`): with `cn_stride = 0` all output lanes collide on one
  cell and the per-lane readback would be last-writer-wins. Torch strides
  of a non-degenerate output are ≥ 1. (Same as the exact headline.)

The exact headline's `hlx` slot pin is **not** carried: the skin pins the
slot internally and the proof case-splits on the sentinel.

Inherited modeling boundary (unchanged from the file docstring): the
numeric face covers the main `ADD_INPUTS = false`, no-`CAST_TYPE` config
(`bgmv_full`); the `ADD_INPUTS` accumulation and `CAST_TYPE` value faces
remain future work, exercised only by the lowering theorem
`bgmv_expand_slice_surface_toAlgorithm_supported` on the general surface. -/
```
</details>

**Statement:**
```lean
specification bgmv_expand_slice_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hsn : 0 < split_n_length)
    (hoi : out_ptr ≠ input_ptr) (hol : out_ptr ≠ lora_ptr)
    (hcn : 0 < cn_stride) :
    bgmvExpandSliceIO input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
      ⊨[R] fun _ _ _ m xs ys t j =>
        ∑ k : Fin BLOCK_K,
          (if k.val < K then xs t k else 0)
            * (if t.val * BLOCK_N + j.val < split_n_length ∧ k.val < K
               then ys t (bTileLane BLOCK_N BLOCK_K j k) else 0)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hsn : 0 < split_n_length`
- `hoi : out_ptr ≠ input_ptr`
- `hol : out_ptr ≠ lora_ptr`
- `hcn : 0 < cn_stride`

**Closed-form spec defs (transitive):** `bgmvExpandSliceIO`, `bTileLane`, `bgmv_full`, `bgmvNumSteps`

<details><summary><code>bgmvExpandSliceIO</code></summary>

```
/-- **Streaming metadata emit IO signature** of the verified `bgmv_full`
config (`EVEN_K = false`, `ADD_INPUTS = false`, no `CAST_TYPE`) on the
metadata-parametrized two-stream per-step emit skin
(`StreamMetaEmitMasked3DKernelIO₂`, style S3: the store sits inside the
`for n` loop). One **`.int`** metadata slot: `lora_index =
lora_indices[cur_batch]`, read at the pid-only cell `cur_batch = pid₁`
(`mwin`) at the kernel's own signed dtype — the `-1` skip sentinel stays
visible in the slot value and gates `writeMask` (the honest sentinel gate:
at `m 0 = -1` no window is write-active and the program stores nothing).
The kernel launches on a 2-D grid `(pid_sn, cur_batch)`; the skin's `pid₂`
slot is unused — every window is constant in it, and the headline still
quantifies over all three pids (the `bgmv_shrink` precedent).

Step `t` of the loop (at `n = t·BLOCK_N`) reads the `[BLOCK_N, BLOCK_K]`
LoRA-B tile and emits the `BLOCK_N`-lane output window; the `BLOCK_K`-lane
input row is the genre's degenerate **static stream** (`read1` ignores `t`:
the kernel loads `tiled_a` once before the loop and register-caches it
across steps — the uniform per-step pin is harmless and keeps the contract
one-shaped). The windows transcribe the kernel's pointer arithmetic
verbatim, with the loaded slot value `m 0` in place of the in-state
`lora_index` read:

* `read1` lane `k` (any step): `cur_batch·xm_stride + k·xk_stride`;
  `mask1`: `k < K` — the masked `EVEN_K = false` load path.
* `read2` step `t`, lane `l = (r, k)` (row-major over
  `[BLOCK_N, BLOCK_K]`, `r = l / BLOCK_K`, `k = l % BLOCK_K`):
  `l0_stride·(m 0).toNat + pid_sn·split_n_length·lora_k_stride
  + (t·BLOCK_N + r)·lora_k_stride + k·lora_n_stride` — the `b_ptr` cell.
  **Negative non-sentinel indices are modeled honestly**: the kernel's
  `b_ptr` arithmetic clamps `l0_stride · lora_index` through `castIntToNat`
  (`Int.toNat`, so any negative product clamps to `0`), and `(m 0).toNat`
  reproduces exactly that clamp — the io does not pretend `m 0 ≥ 0`.
* `write` step `t`, lane `j`: `cur_batch·cm_stride + pid_sn·split_n_length
  + slice_offset·cn_stride + (t·BLOCK_N + j)·cn_stride` — the kernel's
  `c_ptr + current_n·cn_stride` cell (slot-independent).
* `mask2` transcribes `b_ptr_mask` (`current_n < split_n_length &
  offset_k < K`); `writeMask` is `m 0 ≠ -1 ∧ t·BLOCK_N + j <
  split_n_length` — the store's `c_mask` under the sentinel guard. -/
```
```lean
def bgmvExpandSliceIO (input_ptr lora_ptr out_ptr : RegionName)
    (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    StreamMetaEmitMasked3DKernelIO₂ where
  kernel := bgmv_full input_ptr lora_ptr out_ptr lora_indices K
    split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 1
  sty := fun _ => .int
  mbuf := fun _ => Region.cast lora_indices
  mwin := fun _ _ pid₁ _ => pid₁
  T := bgmvNumSteps split_n_length BLOCK_N
  B1 := BLOCK_K
  B2 := BLOCK_N * BLOCK_K
  C := BLOCK_N
  read1 := fun _ pid₁ _ _ _ k => pid₁ * xm_stride + k.val * xk_stride
  read2 := fun pid₀ _ _ m t l =>
    l0_stride * (m ⟨0, Nat.one_pos⟩).toNat
      + pid₀ * split_n_length * lora_k_stride
      + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
      + l.val % BLOCK_K * lora_n_stride
  write := fun pid₀ pid₁ _ _ t j =>
    pid₁ * cm_stride + pid₀ * split_n_length + slice_offset * cn_stride
      + (t.val * BLOCK_N + j.val) * cn_stride
  mask1 := fun _ _ _ _ _ k => k.val < K
  mask2 := fun _ _ _ _ t l =>
    t.val * BLOCK_N + l.val / BLOCK_K < split_n_length ∧ l.val % BLOCK_K < K
  writeMask := fun _ _ _ m t j =>
    m ⟨0, Nat.one_pos⟩ ≠ -1 ∧ t.val * BLOCK_N + j.val < split_n_length
```
</details>

<details><summary><code>bTileLane</code></summary>

```
/-- The `tiled_b`-stream lane feeding output lane `j` at rank key `k`:
`(j, k)` row-major over the `[BLOCK_N, BLOCK_K]` per-step weight tile, via
the shared `Lane2D` bridge. -/
```
```lean
def bTileLane (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (k : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_K) :=
  Lane2D.encode (j, k, PUnit.unit)
```
</details>

<details><summary><code>bgmv_full</code></summary>

```lean
def bgmv_full
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .int)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_k = tl.arange(0, $(BLOCK_K))
    offset_n = tl.arange(0, $(BLOCK_N))
    tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
      mask=offset_k < $(K), other=0)
    b_ptr = lora_ptr + $(l0_stride) * lora_index +
      pid_sn * $(split_n_length) * $(lora_k_stride)
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length) +
      $(slice_offset) * $(cn_stride)
    for n in range($(0), $(split_n_length), $(BLOCK_N)) {
      current_n = n + offset_n
      b_ptr_mask = (current_n[:, None] < $(split_n_length)) & (offset_k[None, :] < $(K))
      c_mask = current_n < $(split_n_length)
      tiled_b = tl.load(
        b_ptr + current_n[:, None] * $(lora_k_stride) +
          offset_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      accumulator = tl.sum(tiled_a * tiled_b, 1)
      tl.store(c_ptr + current_n * $(cn_stride), accumulator, mask=c_mask)
    }
  }
}
```
</details>

<details><summary><code>bgmvNumSteps</code></summary>

```
/-- Trip count of `for n in range(0, split_n_length, BLOCK_N)`:
`⌈split_n_length / BLOCK_N⌉`. -/
```
```lean
def bgmvNumSteps (snl BN : Nat) : Nat := (snl + BN - 1) / BN
```
</details>

## Also present (pinned special-case summaries)
- `bgmv_full_compute_correct`
