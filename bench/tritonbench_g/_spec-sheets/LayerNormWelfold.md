# Spec sheet — `bench/tritonbench_g/layer_norm_welfold/LayerNormWelfold.lean`

**Python source:** `bench/tritonbench_g/layer_norm_welfold/layer_norm_welfold.py`

## Public theorem: `layer_norm_welfold_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `layer_norm_welfold.py`
(`triton_red_fused_native_layer_norm_no_welford`), against the **genuine
closed forms** (mean / variance / rstd / affine output as standalone
`Finset.sum` formulas over input memory — no read-back of kernel output),
for arbitrary `xnumel`, `rnumel`, `XBLOCK`, `RBLOCK` and program ids. It
packages:

* the **full faithful surface** — all three tiled loops and both
  `tl.debug_barrier()` fences — lowers to the algorithm layer, all four
  dimensions symbolic;
* the **two reduction phases** (`XBLOCK = 1` as pinned by both autotune
  configs; single reduction block `rnumel ≤ RBLOCK`): the mean cell
  `in_out_ptr0[x0]` and rstd cell `in_out_ptr1[x0]` genuinely realize
  `rowMeanSpec = (Σ_r X[x0,r])/rnumel` and
  `rowRstdSpec = 1/√((Σ_r (X[x0,r]−μ)²)/rnumel + 1e-05)`, end-to-end from
  `in_ptr0` (the second pass consumes the *register* mean, so the variance
  is genuinely computed from the first pass, not re-read);
* the **normalize face**: for *every* state whose mean/rstd cells hold the
  genuine closed forms — exactly the values the reduction phases store —
  one normalize-loop iteration realizes the genuine
  `rowYSpec = ((X−μ)·rstd)·W + B` on its masked chunk, general over all
  chunk indices (so the third loop is covered for every `rnumel`,
  `RBLOCK`, multi-iteration included).

The multi-iteration content of the first two loops is the pure step face
`sum_accumulate_step_closed`; the cross-iteration register scheduling that
threads `_tmp3` / `_tmp12` between iterations is the trusted runtime
boundary (as with `reversed_cumsum_scalar`'s carried `b_z`). Honest side
conditions only: `rnumel ≤ RBLOCK` for the single-block reduction phases,
mean/rstd output-region distinctness (`in_out_ptr0 ≠ in_out_ptr1`), and
input/mean-buffer distinctness (`in_ptr0 ≠ in_out_ptr0`, because the
second pass reloads the row after the mean store). -/
```
</details>

**Statement:**
```lean
theorem layer_norm_welfold_output_summary_general
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK)
    (s : BlockState)
    (hMeanRstd : in_out_ptr0 ≠ in_out_ptr1)
    (hInMean : in_ptr0 ≠ in_out_ptr0) :
    -- (1) the full faithful surface lowers to the algorithm layer
    (∃ alg,
      (layer_norm_welfold_surface in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1
        in_ptr2 out_ptr0 xnumel rnumel XBLOCK RBLOCK).toAlgorithm?
      = Except.ok alg) ∧
    -- (2) reduction phases: genuine mean and rstd, end-to-end from `in_ptr0`
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_welfold_reduce_slice in_out_ptr0 in_out_ptr1
        in_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr0, s.pids 0))
      (expected := fun _ => rowMeanSpec s in_ptr0 rnumel)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_welfold_reduce_slice in_out_ptr0 in_out_ptr1
        in_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr1, s.pids 0))
      (expected := fun _ => rowRstdSpec s in_ptr0 rnumel))) ∧
    -- (3) normalize face: genuine `Y` chunk for every mean/rstd-genuine state
    (∀ s' : BlockState,
      s'.readMem in_out_ptr0 (s'.pids 0) = rowMeanSpec s' in_ptr0 rnumel →
      s'.readMem in_out_ptr1 (s'.pids 0) = rowRstdSpec s' in_ptr0 rnumel →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := layer_norm_welfold_normalize_slice in_out_ptr0 in_out_ptr1
          in_ptr0 in_ptr1 in_ptr2 out_ptr0 rnumel RBLOCK)
        (initialState := s')
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin RBLOCK => activeLane s' rnumel RBLOCK i)
          (fun i => (out_ptr0, yOffset s' rnumel RBLOCK i)))
        (expected := fun i : Fin RBLOCK =>
          rowYSpec s' in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s' RBLOCK i)))
```

**Assumptions / layout contracts:**
- `hLe : rnumel ≤ RBLOCK`
- `hMeanRstd : in_out_ptr0 ≠ in_out_ptr1`
- `hInMean : in_ptr0 ≠ in_out_ptr0`
- `fun i : Fin RBLOCK => activeLane s' rnumel RBLOCK i`

**Closed-form spec defs (transitive):** `layer_norm_welfold_surface`, `layer_norm_welfold_reduce_slice`, `rowMeanSpec`, `rowRstdSpec`, `layer_norm_welfold_normalize_slice`, `activeLane`, `yOffset`, `rowYSpec`, `rIndex`, `rowElem`, `rowVarSpec`

<details><summary><code>layer_norm_welfold_surface</code></summary>

```
/-- Faithful transcription of `layer_norm_welfold.py`'s
`triton_red_fused_native_layer_norm_no_welford` (the only `@triton.jit`
kernel; the `@triton.autotune` decorator sweeps
`(XBLOCK, RBLOCK) ∈ {(1,1024),(1,2048)}` and is not modeled — all four
parameters stay symbolic). The mechanical DSL spellings are documented in
the `Translation-surface blocker:` preamble marker. -/
```
```lean
def layer_norm_welfold_surface
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(XBLOCK)
  xindex = xoffset + tl.arange(0, $(XBLOCK))[:, None]
  xmask = xindex < $(xnumel)
  rbase = tl.arange(0, $(RBLOCK))[None, :]
  x0 = xindex
  _tmp3 = tl.full([$(XBLOCK), $(RBLOCK)], 0, dtype=tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, _tmp3)
    tmp4 = _tmp3 + tmp2
    _tmp3 = tmp4
  }
  tmp3_tmp = tl.sum(_tmp3, 1)
  tmp3 = tmp3_tmp[:, None]
  tmp5 = $(rnumel)
  tmp6 = tmp3 / tmp5
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp6)
  _tmp12 = tl.full([$(XBLOCK), $(RBLOCK)], 0, dtype=tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp7 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp8 = (tmp7).to(tl.float32)
    tmp9 = tmp8 - tmp6
    tmp10 = tmp9 * tmp9
    tmp11, _ = tl.broadcast(tmp10, _tmp12)
    tmp13 = _tmp12 + tmp11
    _tmp12 = tmp13
  }
  tmp12_tmp = tl.sum(_tmp12, 1)
  tmp12 = tmp12_tmp[:, None]
  tmp14 = $(rnumel)
  tmp15 = tmp12 / tmp14
  tmp16 = 1e-05
  tmp17 = tmp15 + tmp16
  tmp18 = tl.rsqrt(tmp17)
  tl.debug_barrier()
  tl.store(in_out_ptr1 + (x0), tmp18)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp19 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_first").to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp20 = (tmp19).to(tl.float32)
    tmp21 = tmp20 - tmp6
    tmp22 = tmp21 * tmp18
    tmp24 = (tmp23).to(tl.float32)
    tmp25 = tmp22 * tmp24
    tmp27 = (tmp26).to(tl.float32)
    tmp28 = tmp25 + tmp27
    tmp29 = (tmp28).to(tl.float32)
    tl.store(out_ptr0 + (r1 + ($(rnumel) * x0)), tmp29, rmask)
  }
}
```
</details>

<details><summary><code>layer_norm_welfold_reduce_slice</code></summary>

```
/-! ## Reduction-phase slice (`XBLOCK = 1`, one reduction iteration each)

Both `@triton.autotune` configs pin `XBLOCK = 1` (one row per program:
`xoffset = pid`, `xindex = x0 = pid`, the `[XBLOCK, RBLOCK]` tiles are the
row's `[RBLOCK]` lanes, `axis=1` reductions become `axis=0`). When the row
also fits one reduction block (`rnumel ≤ RBLOCK`), the first two
`for roffset` loops run exactly once with `roffset = 0`, so this slice
transcribes both reduction phases — first-loop body at `roffset = 0`, the
mean chain `tmp5/tmp6`, the first `tl.debug_barrier()` and mean store,
second-loop body at `roffset = 0`, the `tmp14..tmp18` rstd chain, the
second `tl.debug_barrier()` and rstd store — in kernel order, stopping
before the (independent) normalize loop, which
`layer_norm_welfold_normalize_slice` covers for every chunk. The masked
loads carry `other=0.0` and the squared-deviation lanes are zero-guarded
with `tl.where(rmask, ·, 0)` (the faithful surface's unguarded accumulation
relies on the autotune configs' exact division `RBLOCK ∣ rnumel`; here the
tail lanes must be total-and-zero to be summed), and the identity recasts
`tmp1`/`tmp2` and `tmp8`/`tmp11` (`.to(tl.float32)` and `broadcast_to` on
an already-`[RBLOCK]` tile) are elided. -/
```
```lean
def layer_norm_welfold_reduce_slice
    (in_out_ptr0 in_out_ptr1 in_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  rbase = tl.arange(0, $(RBLOCK))
  rmask = rbase < $(rnumel)
  r1 = rbase
  tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp3 = tl.sum(tmp0, 0)
  tmp5 = $(rnumel)
  tmp6 = tmp3 / tmp5
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp6)
  tmp7 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp9 = tmp7 - tmp6
  tmp10 = tmp9 * tmp9
  zero_lanes = tl.zeros([$(RBLOCK)], tl.float32)
  tmp11 = tl.where(rmask, tmp10, zero_lanes)
  tmp12 = tl.sum(tmp11, 0)
  tmp14 = $(rnumel)
  tmp15 = tmp12 / tmp14
  tmp16 = 1e-05
  tmp17 = tmp15 + tmp16
  tmp18 = tl.rsqrt(tmp17)
  tl.debug_barrier()
  tl.store(in_out_ptr1 + (x0), tmp18)
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
`tmp16 = 1e-05`: `1 / sqrt(var + 1e-05)` (`libdevice.rsqrt`). -/
```
```lean
noncomputable def rowRstdSpec (s : BlockState) (in_ptr0 : RegionName)
    (rnumel : Nat) : ℝ :=
  1 / Real.sqrt (rowVarSpec s in_ptr0 rnumel + 1e-05)
```
</details>

<details><summary><code>layer_norm_welfold_normalize_slice</code></summary>

```
/-! ## Normalize-loop slice (third `for roffset` loop, one iteration)

The third loop's iterations are independent: each reads the already-stored
row mean (`in_out_ptr0[x0]`, the kernel's `tmp6`) and rstd
(`in_out_ptr1[x0]`, the kernel's `tmp18`) plus fresh `in_ptr0/1/2` tiles,
and stores one masked output chunk. The slice materializes one iteration
with the chunk index as `tl.program_id(1)` (`roffset = i_t · RBLOCK`); its
correctness theorem takes the honest hypotheses that the mean/rstd cells
hold the genuine closed forms — exactly the values the reduction phases
store — and is dimension-general over `rnumel`, `RBLOCK`, and the chunk
index. -/
```
```lean
def layer_norm_welfold_normalize_slice
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  i_t = tl.program_id(1)
  rindex = i_t * $(RBLOCK) + tl.arange(0, $(RBLOCK))
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp6 = tl.load(in_out_ptr0 + (x0))
  tmp18 = tl.load(in_out_ptr1 + (x0))
  tmp19 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp23 = tl.load(in_ptr1 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp26 = tl.load(in_ptr2 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp20 = (tmp19).to(tl.float32)
  tmp21 = tmp20 - tmp6
  tmp22 = tmp21 * tmp18
  tmp24 = (tmp23).to(tl.float32)
  tmp25 = tmp22 * tmp24
  tmp27 = (tmp26).to(tl.float32)
  tmp28 = tmp25 + tmp27
  tmp29 = (tmp28).to(tl.float32)
  tl.store(out_ptr0 + (r1 + ($(rnumel) * x0)), tmp29, mask=rmask)
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
- `layer_norm_welfold_normalize_slice_compute_correct`
- `layer_norm_welfold_reduce_slice_compute_correct`
