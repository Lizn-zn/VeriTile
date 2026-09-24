# Spec sheet — `bench/tritonbench_g/lora_expand_gemv/LoraExpandGemv.lean`

**Python source:** `bench/tritonbench_g/lora_expand_gemv/lora_expand_gemv.py`

## Public theorem: `gemv_full_output_summary`

<details><summary>docstring</summary>

```
/-- **Full output summary**: the full LoRA expand GEMV surface lowers to the
algorithm layer, and the masked store realizes the genuine matrix-vector product
`out[m] = Σ_{k<K} x[k]·W[m,k]` at every active global lane `m < split_n_length`
(general `⌈split_n_length/BLOCK_N⌉`-block loop). -/
```
</details>

**Statement:**
```lean
specification gemv_full_output_summary
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hol : out_ptr ≠ lora_ptr)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hcn : 0 < cn_stride) :
    (∃ alg, (bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
        split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hKB : K ≤ BLOCK_K`
- `hol : out_ptr ≠ lora_ptr`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hcn : 0 < cn_stride`

**Closed-form spec defs (transitive):** `bgmv_loop_surface`

<details><summary><code>bgmv_loop_surface</code></summary>

```
/-- The full loop surface: signed-sentinel guard elided (the host only launches
the active body when `lora_index ≠ -1`), `RegionName`-typed `lora_indices` so the
selected base is read back via `readMemValue .nat`, `ADD_INPUTS = false`,
`CAST_TYPE = false`, `EVEN_K = false` (masked input load). The `for n` loop runs
`⌈split_n_length / BLOCK_N⌉` blocks. -/
```
```lean
def bgmv_loop_surface
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(BLOCK_K))
  offset_n = tl.arange(0, $(BLOCK_N))
  tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
    mask=offset_k < $(K), other=0.0)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    pid_sn * $(split_n_length) * $(lora_k_stride)
  c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length)
  for n in range($(0), $(split_n_length), $(BLOCK_N)) {
    current_n = n + offset_n
    tiled_b = tl.load(
      b_ptr + current_n[:, None] * $(lora_k_stride) +
        offset_k[None, :] * $(lora_n_stride),
      mask=(current_n[:, None] < $(split_n_length)) and (offset_k[None, :] < $(K)),
      other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(cn_stride), accumulator,
      mask=current_n < $(split_n_length))
  }
}
```
</details>

## Public theorem: `gemv_full_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming metadata emit headline (wave-5 S3 genre).** For
every rounding model `R`, the verified `bgmv_loop_surface` implements, on
its `StreamMetaEmitMasked3DKernelIO₂` signature, the **ideal ℝ LoRA expand
GEMV** over the streamed tiles: every step-`t` write-active output lane `j`
(global lane `t·BLOCK_N + j < split_n_length`) holds
`Σ_{k<BLOCK_K} [k<K] xs[t,k] · ys[t, j·BLOCK_K+k]` — exact real arithmetic;
the slot vector `m` (the adapter index `lora_indices[cur_batch]`) enters the
spec only through the bank select of the `read2` window. The kernel has
**zero rounding events** (`.nat` slot load, `.real` masked loads/reduce,
`.real` masked in-loop stores; the verified `CAST_TYPE = false` path has no
`castFloat`), so the skin's boundary quantization degenerates: the readback
contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` in-loop stores are exact under `execR R` — the
∀-`R` face holds via the `RoundingModel` `.real` identity fields, not as a
`.triv` special case.

Layer map: the whole body is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven
`prefix_inv` / `gemvWbInv` / `gemvWbInv_step` / `forRange_inv` stack above
is reused unchanged; the `⊨[R]` face adds the `TraceSafeR` walk, the
per-cell memory frame (`bgmv_loopBody_step_frame`, the `mem` twin of
`gemvWbInv_step`), and the stream-lane spec bridge
(`sum_fin_ite_range` re-blocking the `Σ_{k<K}` contraction, `Lane2D`
re-blocking the `[BLOCK_N, BLOCK_K]` tile).

All five hypotheses are truth-forced:

* `hBN : 0 < BLOCK_N` — the loop steps by `BLOCK_N`
  (`range(0, split_n_length, BLOCK_N)`); at `BLOCK_N = 0` the loop never
  advances and the step index `i / BLOCK_N` is meaningless. The same
  hypothesis the exact headline `gemv_full_output_summary` carries.
* `hKB : K ≤ BLOCK_K` — the kernel's own `BLOCK_K = next_power_of_2(K)`
  choice; it is what collapses the `tl.sum` over `BLOCK_K` keys to
  `Σ_{k<K}` (`reduceSum_active_eq_spec`). Same as the exact headline.
* `hsnl : 0 < split_n_length` — `split_n_length = tl.cdiv(N, SPLIT_N) ≥ 1`
  for every real launch (`N ≥ 1`). This face needs it **only for the safety
  walk**: at `split_n_length = 0` the stream is empty (`T = 0`) while the
  kernel still performs its pre-loop static `input` load, so the skin's
  step-indexed `read1` window bounds could not cover it. (The exact headline
  does not need it because `Realizes` carries no bounds obligation.)
* `hol : out_ptr ≠ lora_ptr` — the loop stores into `out` **between** its
  re-reads of the `loraB` tile; if they aliased, later blocks would re-read
  already-overwritten weights and the closed form would be false. Same as
  the exact headline.
* `hcn : 0 < cn_stride` — the per-lane write window
  `pid₁·cm + pid₀·snl + m·cn` is injective over global lanes only when the
  output column stride is nonzero (`affine1D_inj`); with `cn_stride = 0` all
  lanes collide and the per-lane readback would be last-writer-wins. Same as
  the exact headline.

The exact headline's remaining hypothesis `hundef` (`∀ rg o, s.undef rg o =
0`) is **not** carried here: `s₀.undef = fun _ _ => 0` is a precondition of
the skin's Hoare triple itself, so it comes free from `ImplementsR.intro`.

Modeling boundary (mirrors the surface's docstring): the statement covers
the verified `bgmv_loop_surface` configuration — `ADD_INPUTS = false`,
`CAST_TYPE = false`, masked input load (general `EVEN_K` path), and the
signed `lora_index == -1` sentinel elided at the trusted host boundary (the
`.nat` slot only covers active launches; see the `loraExpandGemvIO`
docstring). An `ADD_INPUTS = true` value face (accumulating onto the
existing output, `tiled_out +` in the spec) is future work — the verified
exact stack does not cover that branch either.

Relation to the exact surface: the exact headline
`gemv_full_output_summary` (`Realizes_without_Rounding`) above is retained
unchanged; this `⊨[R]` face restates the same GEMV content on the streaming
metadata emit skin, for every `R` at once (at the `.real` grid the two faces
carry the same exact cell). Both faces are kept per the rounding-as-default
doctrine. -/
```
</details>

**Statement:**
```lean
specification gemv_full_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K : Nat)
    (hBN : 0 < BLOCK_N) (hKB : K ≤ BLOCK_K) (hsnl : 0 < split_n_length)
    (hol : out_ptr ≠ lora_ptr) (hcn : 0 < cn_stride) :
    loraExpandGemvIO input_ptr lora_ptr out_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K ⊨[R]
      fun _ _ _ _ xs ys t j =>
        ∑ k : Fin BLOCK_K, if k.val < K then
          xs t k * ys t (bLane BLOCK_N BLOCK_K j k) else 0
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hKB : K ≤ BLOCK_K`
- `hsnl : 0 < split_n_length`
- `hol : out_ptr ≠ lora_ptr`
- `hcn : 0 < cn_stride`

**Closed-form spec defs (transitive):** `loraExpandGemvIO`, `bLane`, `bgmv_loop_surface`, `bgmvNumSteps`

<details><summary><code>loraExpandGemvIO</code></summary>

```
/-- **Streaming metadata IO signature** of the verified `bgmv_loop_surface`
on the metadata-parametrized per-step emit skin (S3: in-loop store, 3-D pid
grid). The kernel is 2-pid (`pid_sn = pids 0`, `cur_batch = pids 1`); the
skin's third pid is quantified and every window ignores it. The single
`.nat` metadata slot is the kernel's own per-batch adapter index, loaded at
cell `cur_batch = pid₁` of `lora_indices`; the loaded slot value `m 0` banks
the `loraB` read window (`l0_stride · m 0`, in place of the in-state
`loraIdx` read). Step `t` of the loop (at `n = t·BLOCK_N`) reads the static
`BLOCK_K`-lane `input` tile (`read1` ignores `t` — the genre's degenerate
static stream: the kernel loads it once, pre-loop, and register-caches it)
and the `[BLOCK_N, BLOCK_K]` `loraB` tile (`read2`, row-major lane
`r·BLOCK_K + k`), and masked-stores the `BLOCK_N`-lane output window
(`write`) at the **`.real`** grid (`outDType` default — the store is an
untyped `tl.store`, no quantization event). The windows transcribe the
kernel's pointer arithmetic verbatim:

* `read1` lane `k`: `pid₁·xm_stride + k·xk_stride`; `mask1`: `k < K`
  (the kernel's `offset_k < K` load mask).
* `read2` step `t`, lane `l = (r, k)`:
  `l0_stride·m 0 + pid₀·split_n_length·lora_k_stride +
  (t·BLOCK_N + r)·lora_k_stride + k·lora_n_stride`; `mask2`:
  `t·BLOCK_N + r < split_n_length ∧ k < K` (the kernel's `b_ptr_mask`).
* `write` step `t`, lane `j`:
  `pid₁·cm_stride + pid₀·split_n_length + (t·BLOCK_N + j)·cn_stride`;
  `writeMask`: `t·BLOCK_N + j < split_n_length` (the kernel's `c_mask`).

**Sentinel disclosure** (mirrors the surface's docstring): Python's signed
`lora_index == -1` early return lives at the **trusted host boundary** — the
host only launches the active body when `lora_index ≠ -1` — so this
signature's `.nat` slot only covers active launches and `writeMask` carries
**no** sentinel gate. The skin's empty-write-window sentinel idiom is *not*
exercised here; expressing the `-1` skip as an `.int` slot gating `writeMask`
would need the guarded surface (`bgmv_expand_surface`), not the verified
active-body surface. -/
```
```lean
def loraExpandGemvIO (input_ptr lora_ptr out_ptr : RegionName)
    (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    StreamMetaEmitMasked3DKernelIO₂ where
  kernel := bgmv_loop_surface input_ptr lora_ptr out_ptr lora_indices K
    split_n_length xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride BLOCK_N BLOCK_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 1
  sty := fun _ => ChanTy.nat
  mbuf := fun _ => lora_indices.cast
  mwin := fun _ _ pid₁ _ => pid₁
  T := bgmvNumSteps split_n_length BLOCK_N
  B1 := BLOCK_K
  B2 := BLOCK_N * BLOCK_K
  C := BLOCK_N
  read1 := fun _ pid₁ _ _ _ k => pid₁ * xm_stride + k.val * xk_stride
  read2 := fun pid₀ _ _ m t l =>
    l0_stride * m (⟨0, by omega⟩ : Fin 1)
      + pid₀ * split_n_length * lora_k_stride
      + (t.val * BLOCK_N + l.val / BLOCK_K) * lora_k_stride
      + (l.val % BLOCK_K) * lora_n_stride
  write := fun pid₀ pid₁ _ _ t j =>
    pid₁ * cm_stride + pid₀ * split_n_length + (t.val * BLOCK_N + j.val) * cn_stride
  mask1 := fun _ _ _ _ _ k => k.val < K
  mask2 := fun _ _ _ _ t l =>
    t.val * BLOCK_N + l.val / BLOCK_K < split_n_length ∧ l.val % BLOCK_K < K
  writeMask := fun _ _ _ _ t j => t.val * BLOCK_N + j.val < split_n_length
```
</details>

<details><summary><code>bLane</code></summary>

```
/-- The `loraB`-stream lane feeding output lane `j` at rank key `k`: row `j`
paired with `k`, row-major over the `[BLOCK_N, BLOCK_K]` per-step `b`-tile,
via the shared `Lane2D` bridge. -/
```
```lean
def bLane (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (k : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_K) :=
  Lane2D.encode (j, k, PUnit.unit)
```
</details>

<details><summary><code>bgmv_loop_surface</code></summary>

```
/-- The full loop surface: signed-sentinel guard elided (the host only launches
the active body when `lora_index ≠ -1`), `RegionName`-typed `lora_indices` so the
selected base is read back via `readMemValue .nat`, `ADD_INPUTS = false`,
`CAST_TYPE = false`, `EVEN_K = false` (masked input load). The `for n` loop runs
`⌈split_n_length / BLOCK_N⌉` blocks. -/
```
```lean
def bgmv_loop_surface
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(BLOCK_K))
  offset_n = tl.arange(0, $(BLOCK_N))
  tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
    mask=offset_k < $(K), other=0.0)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    pid_sn * $(split_n_length) * $(lora_k_stride)
  c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length)
  for n in range($(0), $(split_n_length), $(BLOCK_N)) {
    current_n = n + offset_n
    tiled_b = tl.load(
      b_ptr + current_n[:, None] * $(lora_k_stride) +
        offset_k[None, :] * $(lora_n_stride),
      mask=(current_n[:, None] < $(split_n_length)) and (offset_k[None, :] < $(K)),
      other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(cn_stride), accumulator,
      mask=current_n < $(split_n_length))
  }
}
```
</details>

<details><summary><code>bgmvNumSteps</code></summary>

```
/-- Trip count of the kernel's `for n in range(0, split_n_length, BLOCK_N)`
loop: `⌈split_n_length / BLOCK_N⌉`. -/
```
```lean
def bgmvNumSteps (snl B : Nat) : Nat := (snl + B - 1) / B
```
</details>

## Also present (pinned special-case summaries)
- `gemv_compute_correct`
