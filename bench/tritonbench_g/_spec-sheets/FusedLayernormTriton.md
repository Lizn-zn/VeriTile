# Spec sheet — `bench/tritonbench_g/fused_layernorm_triton/FusedLayernormTriton.lean`

**Python source:** `bench/tritonbench_g/fused_layernorm_triton/fused_layernorm_triton.py`

## Public theorem: `fused_layernorm_triton_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `fused_layernorm_triton.py`
(`triton_red_fused_native_layer_norm_0`), against the **genuine closed
forms** (mean / variance / rstd / affine output as standalone `Finset.sum`
formulas over input memory — no read-back of kernel output), for arbitrary
`xnumel`, `rnumel`, `XBLOCK`, `RBLOCK` and program ids. It packages:

* the **full faithful surface** — both tiled reduction loops with the
  inlined Welford statements, the `tl.debug_barrier()` fence, and all three
  stores — lowers to the algorithm layer, all four dimensions symbolic;
* the **reduction phase** (`XBLOCK = 1` as pinned by both autotune configs;
  single reduction block `rnumel ≤ RBLOCK`): the mean cell `out_ptr0[x0]`
  and rstd cell `in_out_ptr0[x0]` genuinely realize
  `rowMeanSpec = (Σ_r X[x0,r])/rnumel` and
  `rowRstdSpec = 1/√((Σ_r (X[x0,r]−μ)²)/rnumel + 1e-05)`, end-to-end from
  `in_ptr0`;
* the **normalize face**: for *every* state whose mean/rstd cells hold the
  genuine closed forms — exactly the values the reduction phase stores —
  one normalize-loop iteration realizes the genuine
  `rowYSpec = ((X−μ)·rstd)·W + B` on its masked chunk, general over all
  chunk indices (so the second loop is covered for every `rnumel`,
  `RBLOCK`, multi-iteration included).

The multi-iteration content of the *first* loop is the pure Welford step
face `welford_reduce_step_closed` plus the moment-combine identity
`var_moment_identity`; the cross-iteration register scheduling that threads
`(tmp3_mean, tmp3_m2, tmp3_weight)` between iterations is the trusted
runtime boundary (as with `reversed_cumsum_scalar`'s carried `b_z`).
Honest side conditions only: `rnumel ≤ RBLOCK` for the single-block
reduction phase and mean/rstd output-region distinctness
(`out_ptr0 ≠ in_out_ptr0`). -/
```
</details>

**Statement:**
```lean
theorem fused_layernorm_triton_output_summary_general
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK)
    (s : BlockState)
    (hMeanRstd : out_ptr0 ≠ in_out_ptr0) :
    -- (1) the full faithful surface lowers to the algorithm layer
    (∃ alg,
      (fused_layernorm_triton_surface in_out_ptr0 in_ptr0 in_ptr1 in_ptr2
        out_ptr0 out_ptr1 xnumel rnumel XBLOCK RBLOCK).toAlgorithm?
      = Except.ok alg) ∧
    -- (2) reduction phase: genuine mean and rstd, end-to-end from `in_ptr0`
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_layernorm_triton_reduce_slice in_out_ptr0 in_ptr0
        out_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (out_ptr0, s.pids 0))
      (expected := fun _ => rowMeanSpec s in_ptr0 rnumel)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_layernorm_triton_reduce_slice in_out_ptr0 in_ptr0
        out_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr0, s.pids 0))
      (expected := fun _ => rowRstdSpec s in_ptr0 rnumel))) ∧
    -- (3) normalize face: genuine `Y` chunk for every mean/rstd-genuine state
    (∀ s' : BlockState,
      s'.readMem out_ptr0 (s'.pids 0) = rowMeanSpec s' in_ptr0 rnumel →
      s'.readMem in_out_ptr0 (s'.pids 0) = rowRstdSpec s' in_ptr0 rnumel →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_layernorm_triton_normalize_slice in_out_ptr0 in_ptr0
          in_ptr1 in_ptr2 out_ptr0 out_ptr1 rnumel RBLOCK)
        (initialState := s')
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin RBLOCK => activeLane s' rnumel RBLOCK i)
          (fun i => (out_ptr1, yOffset s' rnumel RBLOCK i)))
        (expected := fun i : Fin RBLOCK =>
          rowYSpec s' in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s' RBLOCK i)))
```

**Assumptions / layout contracts:**
- `hLe : rnumel ≤ RBLOCK`
- `hMeanRstd : out_ptr0 ≠ in_out_ptr0`
- `fun i : Fin RBLOCK => activeLane s' rnumel RBLOCK i`

**Closed-form spec defs (transitive):** `fused_layernorm_triton_surface`, `fused_layernorm_triton_reduce_slice`, `rowMeanSpec`, `rowRstdSpec`, `fused_layernorm_triton_normalize_slice`, `activeLane`, `yOffset`, `rowYSpec`, `rIndex`, `rowElem`, `rowVarSpec`

<details><summary><code>fused_layernorm_triton_surface</code></summary>

```
/-- Faithful transcription of `fused_layernorm_triton.py`'s
`triton_red_fused_native_layer_norm_0` (the only `@triton.jit` kernel; the
`@triton.autotune` decorator sweeps `(XBLOCK, RBLOCK) ∈ {(1,1024),(1,2048)}`
and is not modeled — all four parameters stay symbolic). The inlined
`welford_reduce` / `welford` helper statements are documented in the
`Translation-surface blocker:` preamble marker. -/
```
```lean
def fused_layernorm_triton_surface
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(XBLOCK)
  xindex = xoffset + tl.arange(0, $(XBLOCK))[:, None]
  xmask = xindex < $(xnumel)
  rbase = tl.arange(0, $(RBLOCK))[None, :]
  x0 = xindex
  tmp3_mean = tl.zeros([$(XBLOCK), $(RBLOCK)], tl.float32)
  tmp3_m2 = tl.zeros([$(XBLOCK), $(RBLOCK)], tl.float32)
  tmp3_weight = tl.zeros([$(XBLOCK), $(RBLOCK)], tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
    tmp3_delta = tmp2 - tmp3_mean
    tmp3_weight_next = tmp3_weight + 1.0
    tmp3_mean_next = tmp3_mean + tmp3_delta / tmp3_weight_next
    tmp3_m2_next = tmp3_m2 + tmp3_delta * (tmp2 - tmp3_mean_next)
    tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
    tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
    tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  }
  tmp5_tmp = tl.sum(tmp3_weight, 1)
  tmp3_tmp = tl.sum(tmp3_weight * tmp3_mean, 1) / tmp5_tmp
  tmp4_tmp = tl.sum(tmp3_m2 + tmp3_weight * tmp3_mean * tmp3_mean, 1) - tmp5_tmp * tmp3_tmp * tmp3_tmp
  tmp3 = tmp3_tmp[:, None]
  tmp4 = tmp4_tmp[:, None]
  tmp5 = tmp5_tmp[:, None]
  tl.store(out_ptr0 + (x0), tmp3)
  tmp6 = $(rnumel)
  tmp7 = tmp4 / tmp6
  tmp8 = 1e-05
  tmp9 = tmp7 + tmp8
  tmp10 = tl.rsqrt(tmp9)
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp10)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp11 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_first").to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp12 = (tmp11).to(tl.float32)
    tmp13 = tmp12 - tmp3
    tmp14 = tmp13 * tmp10
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = (tmp18).to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp20).to(tl.float32)
    tl.store(out_ptr1 + (r1 + ($(rnumel) * x0)), tmp21, rmask)
  }
}
```
</details>

<details><summary><code>fused_layernorm_triton_reduce_slice</code></summary>

```
/-! ## Reduction-phase slice (`XBLOCK = 1`, one reduction iteration)

Both `@triton.autotune` configs pin `XBLOCK = 1` (one row per program:
`xoffset = pid`, `xindex = x0 = pid`, the `[XBLOCK, RBLOCK]` tiles are the
row's `[RBLOCK]` lanes, `axis=1` reductions become `axis=0`). When the row
also fits one reduction block (`rnumel ≤ RBLOCK`, e.g. the benchmark's
`D = 1024/2048` rows under `RBLOCK ∈ {1024, 2048}`), the first
`for roffset` loop runs exactly once with `roffset = 0`, so this slice
transcribes the reduction phase — loop body at `roffset = 0`, the inlined
moment combine, the `tmp6..tmp10` rstd chain, `tl.debug_barrier()`, and
the two scalar stores — in kernel order, stopping before the (independent)
normalize loop, which `fused_layernorm_triton_normalize_slice` covers for
every chunk. The masked load carries `other=0.0` (the faithful surface's
mask-only load lanes are discarded by `tl.where`; here the lanes must be
total to be summed), and the identity recasts `tmp1`/`tmp2`
(`.to(tl.float32)` and `broadcast_to` on an already-`[RBLOCK]` tile) are
elided. The single inlined `welford_reduce` step runs from the all-zero
state. -/
```
```lean
def fused_layernorm_triton_reduce_slice
    (in_out_ptr0 in_ptr0 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  rbase = tl.arange(0, $(RBLOCK))
  rmask = rbase < $(rnumel)
  r1 = rbase
  tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp3_mean_init = tl.zeros([$(RBLOCK)], tl.float32)
  tmp3_m2_init = tl.zeros([$(RBLOCK)], tl.float32)
  tmp3_weight_init = tl.zeros([$(RBLOCK)], tl.float32)
  tmp3_delta = tmp0 - tmp3_mean_init
  tmp3_weight_next = tmp3_weight_init + 1.0
  tmp3_mean_next = tmp3_mean_init + tmp3_delta / tmp3_weight_next
  tmp3_m2_next = tmp3_m2_init + tmp3_delta * (tmp0 - tmp3_mean_next)
  tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean_init)
  tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2_init)
  tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight_init)
  tmp5_tmp = tl.sum(tmp3_weight, 0)
  tmp3_tmp = tl.sum(tmp3_weight * tmp3_mean, 0) / tmp5_tmp
  tmp4_tmp = tl.sum(tmp3_m2 + tmp3_weight * tmp3_mean * tmp3_mean, 0) - tmp5_tmp * tmp3_tmp * tmp3_tmp
  tl.store(out_ptr0 + (x0), tmp3_tmp)
  tmp6 = $(rnumel)
  tmp7 = tmp4_tmp / tmp6
  tmp8 = 1e-05
  tmp9 = tmp7 + tmp8
  tmp10 = tl.rsqrt(tmp9)
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp10)
}
```
</details>

<details><summary><code>rowMeanSpec</code></summary>

```
/-- Genuine row mean: `(Σ_{r < rnumel} X[x0, r]) / rnumel`. -/
```
```lean
noncomputable def rowMeanSpec (s : BlockState) (in_ptr0 : RegionName)
    (rnumel : Nat) : ℝ :=
  (∑ r ∈ Finset.range rnumel, rowElem s in_ptr0 rnumel r) / (rnumel : ℝ)
```
</details>

<details><summary><code>rowRstdSpec</code></summary>

```
/-- Genuine reciprocal standard deviation with the kernel's hardcoded
`tmp8 = 1e-05`: `1 / sqrt(var + 1e-05)` (`libdevice.rsqrt`). -/
```
```lean
noncomputable def rowRstdSpec (s : BlockState) (in_ptr0 : RegionName)
    (rnumel : Nat) : ℝ :=
  1 / Real.sqrt (rowVarSpec s in_ptr0 rnumel + 1e-05)
```
</details>

<details><summary><code>fused_layernorm_triton_normalize_slice</code></summary>

```
/-! ## Normalize-loop slice (second `for roffset` loop, one iteration)

The second loop's iterations are independent: each reads the already-stored
row mean (`out_ptr0[x0]`, the kernel's `tmp3`) and rstd (`in_out_ptr0[x0]`,
the kernel's `tmp10`) plus fresh `in_ptr0/1/2` tiles, and stores one masked
output chunk. The slice materializes one iteration with the chunk index as
`tl.program_id(1)` (`roffset = i_t · RBLOCK`); its correctness theorem takes
the honest hypotheses that the mean/rstd cells hold the genuine closed forms
— exactly the values the first phase stores — and is dimension-general over
`rnumel`, `RBLOCK`, and the chunk index. -/
```
```lean
def fused_layernorm_triton_normalize_slice
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  i_t = tl.program_id(1)
  rindex = i_t * $(RBLOCK) + tl.arange(0, $(RBLOCK))
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp3 = tl.load(out_ptr0 + (x0))
  tmp10 = tl.load(in_out_ptr0 + (x0))
  tmp11 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp15 = tl.load(in_ptr1 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp18 = tl.load(in_ptr2 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp12 = (tmp11).to(tl.float32)
  tmp13 = tmp12 - tmp3
  tmp14 = tmp13 * tmp10
  tmp16 = (tmp15).to(tl.float32)
  tmp17 = tmp14 * tmp16
  tmp19 = (tmp18).to(tl.float32)
  tmp20 = tmp17 + tmp19
  tmp21 = (tmp20).to(tl.float32)
  tl.store(out_ptr1 + (r1 + ($(rnumel) * x0)), tmp21, mask=rmask)
}
```
</details>

<details><summary><code>activeLane</code></summary>

```
/-- Lane `i` is active iff its column is in range (`rmask`). -/
```
```lean
def activeLane (s : BlockState) (rnumel RBLOCK : Nat) (i : Fin RBLOCK) : Prop :=
  rIndex s RBLOCK i < rnumel
```
</details>

<details><summary><code>yOffset</code></summary>

```
/-- Output cell of lane `i`: the kernel's store offset `r1 + rnumel * x0`. -/
```
```lean
def yOffset (s : BlockState) (rnumel RBLOCK : Nat) (i : Fin RBLOCK) : Nat :=
  rIndex s RBLOCK i + rnumel * s.pids 0
```
</details>

<details><summary><code>rowYSpec</code></summary>

```
/-- Genuine normalized output:
`Y[x0, r] = ((X[x0, r] − mean) * rstd) * W[r] + B[r]`. -/
```
```lean
noncomputable def rowYSpec (s : BlockState)
    (in_ptr0 in_ptr1 in_ptr2 : RegionName) (rnumel r : Nat) : ℝ :=
  ((rowElem s in_ptr0 rnumel r - rowMeanSpec s in_ptr0 rnumel) *
      rowRstdSpec s in_ptr0 rnumel) * s.readMem in_ptr1 r +
    s.readMem in_ptr2 r
```
</details>

<details><summary><code>rIndex</code></summary>

```
/-- Flat column index of lane `i` in chunk `i_t = pids 1`. -/
```
```lean
def rIndex (s : BlockState) (RBLOCK : Nat) (i : Fin RBLOCK) : Nat :=
  s.pids 1 * RBLOCK + i.val
```
</details>

<details><summary><code>rowElem</code></summary>

```
/-- Row element `X[x0, r]` of the row-major `[xnumel, rnumel]` input: this
program's row (`x0 = pids 0` in the `XBLOCK = 1` launch) at column `r`
(the kernel's load offset `r1 + rnumel * x0`). -/
```
```lean
noncomputable def rowElem (s : BlockState) (in_ptr0 : RegionName)
    (rnumel r : Nat) : ℝ :=
  s.readMem in_ptr0 (r + rnumel * s.pids 0)
```
</details>

<details><summary><code>rowVarSpec</code></summary>

```
/-- Genuine row (population) variance:
`(Σ_{r < rnumel} (X[x0, r] − mean)²) / rnumel`. -/
```
```lean
noncomputable def rowVarSpec (s : BlockState) (in_ptr0 : RegionName)
    (rnumel : Nat) : ℝ :=
  (∑ r ∈ Finset.range rnumel,
      (rowElem s in_ptr0 rnumel r - rowMeanSpec s in_ptr0 rnumel) ^ 2) /
    (rnumel : ℝ)
```
</details>

## Also present (pinned special-case summaries)
- `fused_layernorm_triton_normalize_slice_compute_correct`
- `fused_layernorm_triton_reduce_slice_compute_correct`
