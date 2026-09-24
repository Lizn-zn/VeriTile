# Spec sheet — `bench/tritonbench_g/decay_cumsum/DecayCumsum.lean`

**Python source:** `bench/tritonbench_g/decay_cumsum/decay_cumsum.py`

## Public theorem: `decay_cumsum_backward_closed_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General `output_summary`.** The executed backward surface realizes all three
genuine closed forms (`bwdDQInterClosed` / `bwdDKInterClosed` / `bwdDGClosed`). -/
```
</details>

**Statement:**
```lean
specification decay_cumsum_backward_closed_output_summary_general :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i))
```

**Closed-form spec defs (transitive):** `bwd_decay_global_cumsum_surface`, `active`, `offset`, `bwdDQInterClosed`, `bwdDKInterClosed`, `bwdDGClosed`, `elemIndex`, `baseOffset`, `bwdDGSummand`

<details><summary><code>bwd_decay_global_cumsum_surface</code></summary>

```
/-- Surface transcription of `decay_cumsum.py`'s `bwd_decay_global_cumsum`.

The Python kernel traverses the chunk in reverse and decrements pointers; the
DSL surface preserves that reverse range and pointer movement directly. -/
```
```lean
def bwd_decay_global_cumsum_surface
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_k = K + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_g = G + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dg = DG + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inner = DQInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inner = DKInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inter = DQInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inter = DKInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + offs) < $(DK)
  last_g = tl.zeros([$(BK)], dtype=tl.float32)
  for t in range($(BT) - $(1), -$(1), -$(1)) {
    g_val = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    if t == $(BT) - $(1) {
      last_g = g_val
    }
    dq1 = tl.load(p_dq_inner, mask=mask, other=0)
    dq2 = tl.load(p_dq_inter, mask=mask, other=0)
    dq2 *= tl.math.exp2(g_val)
    dq = dq1 + dq2
    tl.store(p_dq_inter, dq, mask=mask)
    dk1 = tl.load(p_dk_inner, mask=mask, other=0)
    dk2 = tl.load(p_dk_inter, mask=mask, other=0)
    dk2 *= tl.math.exp2(last_g - g_val)
    dk = dk1 + dk2
    tl.store(p_dk_inter, dk, mask=mask)
    q_val = tl.load(p_q, mask=mask, other=0)
    k_val = tl.load(p_k, mask=mask, other=0)
    dg_val = dq * q_val - dk * k_val
    cum_grad_dg += dg_val
    tl.store(p_dg, (cum_grad_dg).to(p_dg.dtype.element_ty), mask=mask)
    p_g -= $(DK)
    p_k -= $(DK)
    p_q -= $(DK)
    p_dq_inner -= $(DK)
    p_dk_inner -= $(DK)
    p_dq_inter -= $(DK)
    p_dk_inter -= $(DK)
    p_dg -= $(DK)
  }
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (DK BK : Nat) (i : Fin BK) : Prop :=
  elemIndex s BK i < DK
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset
    (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : Nat :=
  baseOffset s s_qk_h DK t_rel BT BK + i.val
```
</details>

<details><summary><code>bwdDQInterClosed</code></summary>

```
/-- **Genuine `dq_inter` closed form.** At chunk row `t_rel` and lane `i`, the
backward kernel writes `dq_inner[idx] + dq_inter_in[idx] * exp2(g[idx])` into
`dq_inter`, with `exp2(x) = Real.exp (x * Real.log 2)`. -/
```
```lean
noncomputable def bwdDQInterClosed
    (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DQInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DQInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp (s.readMem G (offset s s_qk_h DK t_rel.val BT BK i) * Real.log 2)
```
</details>

<details><summary><code>bwdDKInterClosed</code></summary>

```
/-- **Genuine `dk_inter` closed form.** At chunk row `t_rel` and lane `i`, the
backward kernel writes
`dk_inner[idx] + dk_inter_in[idx] * exp2(g[row BT-1] - g[idx])` into `dk_inter`,
where `g[row BT-1]` is the captured `last_g`. -/
```
```lean
noncomputable def bwdDKInterClosed
    (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DKInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DKInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp ((s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) -
        s.readMem G (offset s s_qk_h DK t_rel.val BT BK i)) * Real.log 2)
```
</details>

<details><summary><code>bwdDGClosed</code></summary>

```
/-- **Genuine `dg` closed form.** At chunk row `t_rel` and lane `i`, the backward
kernel writes the reverse cumulative sum
`Σ_{j = t_rel}^{BT-1} (dq_inter[j]*q[j] - dk_inter[j]*k[j])` into `dg`. This is
the honest reverse-prefix-scan specification of the carried `cum_grad_dg`
accumulator (the `range(BT-1,-1,-1)` loop threads `cum_grad_dg += dq*q - dk*k`).
This is *not* the executed kernel readback. -/
```
```lean
noncomputable def bwdDGClosed
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t_rel.val),
    bwdDGSummand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      ⟨t_rel.val + d.val, by omega⟩ i

/-! ### Proof recipe (backward closed forms)

The three genuine closed forms above (`bwdDQInterClosed`, `bwdDKInterClosed`,
`bwdDGClosed`) are the honest, non self-referential specifications that replace
the (now-deleted) `decayBackwardSurfaceValue`. They are connected to the executed
`bwd_decay_global_cumsum_surface` in
`decay_cumsum_backward_closed_output_summary_general` (and its three
faces `bwd_decay_cumsum_d{q,k}_inter_closed_compute_correct_general` /
`bwd_decay_cumsum_dg_closed_compute_correct_general`) at the end of this file, following
the same closed-form recipe as the forward/prepare general stacks,
but the backward
loop body is ~25 statements with a conditional `last_g` capture and three masked
stores per iteration, traversed over the reverse `range(BT-1,-1,-1)` rows (lowered to a
forward `forRangeDyn "__rev_t" 0 BT 1` with `t := BT-1 - __rev_t`). A single
`simp [exec, …, evalOp.eq_def, stepForRangeAux.*]` blast does *not* scale to this
body (it does not terminate within ~9 min even at 8M heartbeats), so the
mandated per-statement architecture is required:

1. `exec → stepStmts toAlgKernel.body`, with the surface body decomposed by
   `bwd_body_decomp_general` into the 15-stmt prologue + the `forRangeDyn` reverse loop.
2. Drive the `forRangeDyn` loop with `forRangeAux_inv` /
   `VeriTile.Triton.forRangeDyn_inv` (carry invariant on `cum_grad_dg` =
   partial reverse prefix sum), *not* a `simp` over the whole loop.
3. Per body statement: `stepStmts.cons_some` + `simp only` over the named
   `evalOp_*` lemmas (`evalOp_add/mul/sub/ref/…`, `evalOp_ref_setReg*`) — never
   `evalOp.eq_def` whnf over the nested `setReg` literal state.
4. Read back each output with the masked-scatter lemmas
   (`scatter_readback_prop_masked_nd`,
   `scatter_prop_masked_preserves_other_{offset,region}`), peeling the later
   stores in reverse, exactly as the forward row-1 proof does.
5. Bridge to `ComputeCorrect.Realizes_without_Rounding` via `realizes_writeIf_iff` +
   `computeCorrect_of_toAlgKernel` (done; `decayBackwardSurfaceValue` deleted).

This plan is now fully realized dimension-generally: `bwd_prologue_eval_general`
runs the 15-stmt prologue, `bwd_decay_cumsum_step_general` advances the reverse-loop
invariant `bwdInvG` (one iteration, head + conditional `last_g` capture + three
masked stores), `bwd_loop_drive_general` assembles prologue + the full `range(BT)`
reverse loop, and the three `_general` readback theorems certify the closed forms
(the `dg` face uses the genuine reverse cumsum via `bwdCumPartialG`).

The `dq_inter`/`dk_inter` faces are pointwise (no carry); only `dg` needs the
reverse-scan invariant. Region-distinctness side hypotheses (`DQInter ≠ DKInter`
etc.) are needed so a later store does not clobber an earlier readback, mirroring
the forward `G ≠ GO` and `prepare` `Q ≠ QG …` hypotheses. -/

/-! ## Per-statement op-eval recipes (backward kernel, recipe layer)

These are the standalone, register-readback-abstracted `stepStmt`/`evalOp`
reduction lemmas for *each statement kind* appearing in the
`bwd_decay_global_cumsum_surface` body (15-stmt prologue + 25-stmt reverse loop
body). They are the mandated per-statement architecture building blocks: every
```
</details>

<details><summary><code>elemIndex</code></summary>

```lean
def elemIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val
```
</details>

<details><summary><code>baseOffset</code></summary>

```lean
def baseOffset (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) : Nat :=
  s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK + s.pids 0 * BK
```
</details>

<details><summary><code>bwdDGSummand</code></summary>

```
/-- The per-row `dg` summand `dq_inter[j] * q[j] - dk_inter[j] * k[j]`, written
in terms of the genuine `dq_inter`/`dk_inter` closed forms above. -/
```
```lean
noncomputable def bwdDGSummand
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (j : Fin BT) (i : Fin BK) : ℝ :=
  bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK j i *
      s.readMem Q (offset s s_qk_h DK j.val BT BK i) -
    bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK j i *
      s.readMem K (offset s s_qk_h DK j.val BT BK i)
```
</details>

## Public theorem: `fwd_decay_cumsum_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre, 3-D
grid).** For every rounding model `R`, the faithful `fwd_decay_cumsum`
surface implements, on its `StreamEmitMasked3DKernelIO₁` signature, the
**ideal ℝ decay cumulative sum** over the streamed `g` rows: emitted window
`(t, j)` holds the scan `1.44269504 · Σ_{u ≤ t} g[u, j]` — the spec `f` is
exact real arithmetic. The kernel has **zero rounding events** (the masked
load, the accumulate and the per-step masked stores are all at `.real`; the
erased `.to(tl.float32)` / `.to(p_go.dtype.element_ty)` casts are
`.real → .real`), so the skin's boundary quantization degenerates: the
readback contract's `R.round .real` is the identity by the model's defining
`round_real` — the ∀-`R` face holds via the `RoundingModel` `.real`
identity fields, not as a `.triv` special case.

Layer map: the loop body is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven `fwdInv` invariant stack
above (`fwd_decay_cumsum_step` / `forRange_inv`) is reused unchanged; the
`⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory frame
(`fwd_body_step_frame`, the `mem` twin of the step lemma), and the
stream-lane spec bridge (`fwdDecayStreamSpec_eq_closed` — the mask is
`t`-independent, so on a write-active lane every summand read is
mask-pinned and the guarded form needs no in-sum guard).

Both hypotheses are inherited from the exact headline
`fwd_decay_cumsum_full_surface_closed_general`'s side conditions:

* `hne : G ≠ GO` — the loop stores into `g_o` **between** its per-row
  re-reads of `g`; the invariant's whole-`G` frame (and hence the closed
  form over the *initial* `g` values) requires the output buffer not to
  alias the input. The launch allocates `g` and `g_o` as distinct tensors.
* `hBK : BK ≤ DK` — row separation: the row-`m` scatter must not collide
  with other rows' windows, which needs every lane index `< BK` to stay
  inside one `DK`-wide row. The launch sets `BK = min(DK, 64)`, so this
  holds for every real launch.

Relation to the exact surface: the exact headline
`fwd_decay_cumsum_full_surface_closed_general` above is retained unchanged;
this `⊨[R]` face restates the same scan content on the streaming emit skin,
for every `R` at once (at the `.real` grid the two faces carry the same
exact cell). Both faces are kept per the rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification fwd_decay_cumsum_io_correctness (R : RoundingModel)
    (G GO : RegionName) (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (hne : G ≠ GO) (hBK : BK ≤ DK) :
    fwdDecayCumsumKernelIO G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK ⊨[R]
      fun _ _ _ gs t j => fwdDecayStreamSpec BT BK gs t j
```

**Assumptions / layout contracts:**
- `hne : G ≠ GO`
- `hBK : BK ≤ DK`

**Closed-form spec defs (transitive):** `fwdDecayCumsumKernelIO`, `fwdDecayStreamSpec`, `fwd_decay_cumsum_surface`

<details><summary><code>fwdDecayCumsumKernelIO</code></summary>

```
/-- **Streaming IO signature** of `fwd_decay_cumsum` on the three-pid
single-stream per-step emit skin (S3: in-loop store). Step `t` of the
`range(BT)` loop reads the `BK`-lane `g` row (`read1`) and stores the
`BK`-lane `cum_decay` window (`write`) at the **`.real`** grid (`outDType`
default — the store's `.to(p_go.dtype.element_ty)` cast erases to `.real`,
so the per-step stores have no quantization event). The windows transcribe
the surface's effective addresses verbatim (the surface models the `+= DK`
pointer advance, so row `t`'s address is the base plus `t·DK`):

* `read1` step `t`, lane `j`:
  `i_bh·s_qk_h + i_c·BT·DK + i_k·BK + j + t·DK` — the kernel's `p_g` lane
  after `t` advances (`pid₀ = i_k`, `pid₁ = i_c`, `pid₂ = i_bh`).
* `write` step `t`, lane `j`: the same window into `g_o` — the kernel's
  `p_go` lane.

Both masks are the kernel's single, `t`-independent bound mask
`i_k·BK + j < DK`. -/
```
```lean
def fwdDecayCumsumKernelIO (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat) :
    StreamEmitMasked3DKernelIO₁ where
  kernel := fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK
  inp1 := G
  out := GO
  T := BT
  B1 := BK
  C := BK
  read1 := fun p₀ p₁ p₂ t j => p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  write := fun p₀ p₁ p₂ t j => p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  mask1 := fun p₀ _ _ _ j => p₀ * BK + j.val < DK
  writeMask := fun p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>fwdDecayStreamSpec</code></summary>

```
/-- The stream-level decay-cumsum spec (the genre's *scan* shape): output
window `(t, j)` holds the `inv_ln2`-scaled prefix sum of lane `j`'s streamed
`g` values through step `t`. Since the kernel's mask is `t`-independent, the
sum only ever touches lane `j` itself — on a write-active lane every
summand is mask-pinned, so no guard is needed inside the sum.
Algebraically `fwdDecayClosed` with the row reads re-indexed to the curried
stream. -/
```
```lean
noncomputable def fwdDecayStreamSpec (BT BK : Nat)
    (gs : Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  1.44269504 * ∑ u : Fin (t.val + 1), gs (Fin.castLE t.isLt u) j
```
</details>

<details><summary><code>fwd_decay_cumsum_surface</code></summary>

```
/-- Faithful transcription of `decay_cumsum.py`'s `fwd_decay_cumsum`.

This preserves the program-id decomposition, row base pointers, `BK` lane mask,
float32 accumulator, per-row cumulative update by `inv_ln2`, block-pointer
element dtype cast, and `DK` pointer increments through the `BT` loop. -/
```
```lean
def fwd_decay_cumsum_surface
    (G GO : RegionName)
    (s_qk_h _s_qk_t _s_qk_d _B _H _T : Nat) (_scale : ℝ)
    (BT BK DK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_g = G + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) +
    i_k * $(BK) + tl.arange(0, $(BK))
  p_go = GO + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) +
    i_k * $(BK) + tl.arange(0, $(BK))
  cum_decay = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + tl.arange(0, $(BK))) < $(DK)
  for _i in range($(0), $(BT), $(1)) {
    _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    cum_decay += _g * 1.44269504
    tl.store(p_go, (cum_decay).to(p_go.dtype.element_ty), mask=mask)
    p_g += $(DK)
    p_go += $(DK)
  }
}
```
</details>

## Public theorem: `prepare_qg_kg_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 grouped per-step emit
genre, 3-D grid).** For every rounding model `R`, the faithful
`prepare_qg_kg` surface implements, on its grouped
`StreamGroupedEmitMasked3DKernelIO` signature, the **ideal ℝ decay
scaling** of the streamed `q`/`k` rows: emitted window `(t, j)` of channel
`qg` holds `q[t,j]·exp2(g[t,j])·scale`, of channel `kg` holds
`k[t,j]·exp2(g[BT−1,j] − g[t,j])` — the spec `f` is exact real
arithmetic, and the `last_decay` row is the `g` stream's own step-`BT−1`
cells (no extra channel). The kernel has **zero rounding events** (all
loads, the `exp2` scalings and both per-step masked stores are at `.real`;
the erased `.to(...)` casts are `.real → .real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real` — the ∀-`R` face holds via
the `RoundingModel` `.real` identity fields, not as a `.triv` special
case.

Layer map: the prologue, the unmasked `last_decay` load and the two-store
loop body are cast-free, so under `execR R` they collapse verbatim onto
the exact stepper and the proven `prepInv` invariant stack above
(`prepare_qg_kg_step` / `forRange_inv`, via `prepare_qg_kg_loop_drive`'s
prefix walk re-packaged as `prep_preLoop`) is reused unchanged; the
`⊨[R]` face adds the `TraceSafeR` walk (threading register pins across
the body's first store — `stepStmtR_store_real_regs`), the per-cell
memory frame (`prep_body_step_frame`, the `mem` twin of the step lemma),
and the per-channel stream-lane spec bridges
(`prepStreamSpec_qg_eq_closed` / `prepStreamSpec_kg_eq_closed`).

All hypotheses are truth-forced:

* `hQ_QG`/`hQ_KG`/`hK_QG`/`hK_KG`/`hG_QG`/`hG_KG` — inherited from the
  exact headlines `prepare_qg_kg_full_surface_{qg,kg}_closed_general`:
  the loop stores into `qg`/`kg` **between** its per-row re-reads of
  `q`/`k`/`g`, so the invariant's whole-input frames (and hence the
  closed forms over the *initial* input values) require the output
  buffers not to alias any input. The launch allocates all five tensors
  separately.
* `hQG_KG : QG ≠ KG` — also from the exact headlines: the two per-step
  stores land at identical offsets of their buffers; if `qg` and `kg`
  aliased, the second (qg) store would clobber the kg row just written.
* `hBK : BK ≤ DK` — row separation, as in the exact headlines: the
  row-`m` scatters must not collide with other rows' windows, which needs
  every lane index `< BK` to stay inside one `DK`-wide row. The launch
  sets `BK = min(DK, 64)`.
* `hBT : 0 < BT` — truth-forced twice over: (a) the pre-loop
  `last_decay` load needs a nonempty stream — its addresses are bounded
  by the `g` window bound *at step `BT − 1`*, and its row arithmetic
  `(i_c·BT + BT − 1)·DK` only matches the stream address at `BT ≥ 1`
  (`prep_ld_addr_eq`); (b) the spec's last-row index `⟨BT − 1, _⟩ : Fin
  BT` needs `Fin BT` inhabited (the spec's `dite` else-branch is
  unreachable under this hypothesis). The exact headlines carry the same
  constraint implicitly via `t_rel : Fin BT`, and the launch sets
  `BT = 16` (`prepare_qg_kg_step` also consumes `hBT` for the
  `last_decay` register pin).

Relation to the exact surface: the exact headlines above are retained
unchanged; this `⊨[R]` face restates both closed forms at once on the
grouped streaming emit skin, for every `R` (at the `.real` grid the two
faces carry the same exact cells). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification prepare_qg_kg_io_correctness (R : RoundingModel)
    (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    prepareQgKgKernelIO Q K G QG KG s_qk_h DK BT BK scale ⊨[R]
      fun _ _ _ xs o t j => prepStreamSpec BT BK scale xs o t j
```

**Assumptions / layout contracts:**
- `hQ_QG : Q ≠ QG`
- `hQ_KG : Q ≠ KG`
- `hK_QG : K ≠ QG`
- `hK_KG : K ≠ KG`
- `hG_QG : G ≠ QG`
- `hG_KG : G ≠ KG`
- `hQG_KG : QG ≠ KG`
- `hBK : BK ≤ DK`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `prepareQgKgKernelIO`, `prepStreamSpec`, `prepare_qg_kg_surface`

<details><summary><code>prepareQgKgKernelIO</code></summary>

```
/-- **Streaming IO signature** of `prepare_qg_kg` on the grouped
multi-channel per-step emit skin (S3: two in-loop stores). Step `t` of the
`range(BT)` loop reads the `BK`-lane `q`/`k`/`g` rows (input channels
`0`/`1`/`2`) and stores the `BK`-lane `qg`/`kg` windows (output channels
`0`/`1`) at the **`.real`** grid (`outDType` default — the stores'
`.to(*.dtype.element_ty)` casts erase to `.real`, so the per-step stores
have no quantization event). All five channels share the surface's
effective row address, transcribed verbatim (`pid₀ = i_k`, `pid₁ = i_c`,
`pid₂ = i_bh`; the surface models the `+= DK` pointer advance, so row
`t`'s address is the base plus `t·DK`):

`i_bh·s_qk_h + i_c·BT·DK + i_k·BK + j + t·DK`.

Masks: the `q`/`k` read masks and both write masks are the kernel's
single, `t`-independent bound mask `i_k·BK + j < DK`. The `g` channel's
read mask carries the extra disjunct `t = BT − 1`: the pre-loop
`last_decay` load is **unmasked** in the Python (`tl.load` without
`mask=`) and reads every lane of the last `g` row — its addresses are
exactly the `g` stream's step-`BT−1` cells, so no extra channel is
needed, only the honestly widened last-row read window. -/
```
```lean
def prepareQgKgKernelIO (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale
  nIn := 3
  nOut := 2
  bufs := [Q, K, G, QG, KG]
  inp := fun i => match i with
    | ⟨0, _⟩ => Q
    | ⟨1, _⟩ => K
    | ⟨2, _⟩ => G
    | ⟨n + 3, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => QG
    | ⟨1, _⟩ => KG
    | ⟨n + 2, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => p₀ * BK + j.val < DK
    | ⟨1, _⟩ => p₀ * BK + j.val < DK
    | ⟨2, _⟩ => p₀ * BK + j.val < DK ∨ t.val = BT - 1
    | ⟨n + 3, h⟩ => absurd h (by omega)
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>prepStreamSpec</code></summary>

```
/-- The stream-level `prepare_qg_kg` spec (the genre's *emit* shape, one
clause per output channel): window `(t, j)` of channel `0` (`qg`) holds
`q[t,j] · exp2(g[t,j]) · scale`, of channel `1` (`kg`) holds
`k[t,j] · exp2(g[BT−1,j] − g[t,j])`, with `exp2(x) = exp(x·log 2)` — the
exact spellings of the proven closed forms `prepareQgClosed` /
`prepareKgClosed`, re-indexed to the curried streams. The `last_decay` row
is the `g` stream's step-`BT−1` cells (`xs 2 ⟨BT−1, _⟩ j`); the `dite`
keeps `f` total in `BT` — its `else` branch is unreachable under the
headline's `hBT : 0 < BT`. The channel `match` lives in this single shared
def (inline per-site matchers block cross-declaration `exact`). -/
```
```lean
noncomputable def prepStreamSpec (BT BK : Nat) (scale : ℝ)
    (xs : Fin 3 → Fin BT → Fin BK → ℝ) (o : Fin 2) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ =>
      xs (⟨0, by omega⟩ : Fin 3) t j
        * Real.exp (xs (⟨2, by omega⟩ : Fin 3) t j * Real.log 2) * scale
  | ⟨1, _⟩ =>
      if h : 0 < BT then
        xs (⟨1, by omega⟩ : Fin 3) t j
          * Real.exp ((xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
              - xs (⟨2, by omega⟩ : Fin 3) t j) * Real.log 2)
      else 0
  | ⟨n + 2, h2⟩ => absurd h2 (by omega)
```
</details>

<details><summary><code>prepare_qg_kg_surface</code></summary>

```
/-- Surface transcription of `decay_cumsum.py`'s `prepare_qg_kg`.

This preserves the shared q/k/g row addressing, masked loads, `last_decay`
load, exp2 decay factors, `scale` multiplication for `qg`, dtype-cast stores,
and `DK` pointer increments through the `BT` loop. -/
```
```lean
def prepare_qg_kg_surface
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat)
    (scale : ℝ) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_g = G + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_k = K + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_qg = QG + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_kg = KG + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  mask = (i_k * $(BK) + offs) < $(DK)
  last_decay = tl.load(G + i_bh * $(s_qk_h) +
    (i_c * $(BT) + $(BT) - $(1)) * $(DK) + i_k * $(BK) + offs)
  for _i in range($(0), $(BT), $(1)) {
    q_val = tl.load(p_q, mask=mask, other=0)
    k_val = tl.load(p_k, mask=mask, other=0)
    g_val = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    q_val *= tl.math.exp2(g_val) * $((scale : ℝ))
    k_val *= tl.math.exp2(last_decay - g_val)
    tl.store(p_kg, (k_val).to(p_kg.dtype.element_ty), mask=mask)
    tl.store(p_qg, (q_val).to(p_qg.dtype.element_ty), mask=mask)
    p_q += $(DK)
    p_g += $(DK)
    p_k += $(DK)
    p_kg += $(DK)
    p_qg += $(DK)
  }
}
```
</details>

## Public theorem: `bwd_decay_global_cumsum_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-6 S3 grouped per-step emit
genre, 3-D grid) — the first consumer of the dead-write-tolerant flat
bridge.** For every rounding model `R`, the faithful
`bwd_decay_global_cumsum` surface implements, on its grouped
`StreamGroupedEmitMasked3DKernelIO` signature, the **ideal ℝ backward
decay combination** of the seven streamed rows: emitted window `(t, j)`
of channel `dq_inter` holds `dq_inner + dq_inter_old·exp2(g)`, of channel
`dk_inter` holds `dk_inner + dk_inter_old·exp2(g[BT−1] − g)`, and of
channel `dg` the reverse-scan suffix sum
`Σ_{u ≥ t} (dq[u]·q[u] − dk[u]·k[u])` — the spec `f` is exact real
arithmetic, `last_g` is the `g` stream's own step-`BT−1` row (no extra
channel, no mask widening), and the `dq_inter`/`dk_inter` channels are
**in place** (the pinned inputs are the launch-state contents; sound
because each row is read before it is rewritten, once). The kernel has
**zero rounding events** (all loads, `exp2` combinations and the three
per-step masked stores are at `.real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real`.

**Why this is a direct `ImplementsR` proof and not
`ImplementsR.intro`**: the loop body's 8 trailing `p_x -= DK`
self-decrements genuinely underflow in the final iteration (row 0,
`pid₀ = pid₂ = 0` lanes have addresses `< DK`), so the kernel-wide
`TraceSafeR` walk that `intro` requires is **falsifiable** — machine
checked, not a proof gap. The flat leg instead goes through
`FlatAlloc.execR_flatten_deadPtrTail`: strict commutation for the
prologue + loop core + every non-final tail (`bwd_bodySafeR` /
`bwd_tailSafeR` at rows `≥ 1`), and observational agreement (`ObsAgree`)
across the final, dead, underflowing tail — the readback leg transports
along `ObsAgree.readMemAs_eq` and the frame leg along `ObsAgree.mem`, so
both reduce to the strict case after one rewrite.

Layer map: the region-model triple rides the exact `bwdInvG` stack
verbatim (`bwd_runR`: cast-free collapses + `forRangeAux_inv` +
`bwd_decay_cumsum_step_general`, with the per-cell frame
`bwd_body_step_frame` carried alongside); the flat leg is the tolerant
bridge fed the `bwdInvG` invariant itself as `P` (its pointer pins give
core-load/store bounds at row `BT−1−c` and tail no-underflow at rows
`≥ 1` via `bwd_core_regsR`); the spec legs are the per-channel stream
bridges (`bwdDqStream_eq_closed` / `bwdDkStream_eq_closed` /
`bwdDgStream_eq_closed`).

All hypotheses are truth-forced:

* the 19 region-distinctness hypotheses are **exactly** the exact step
  lemma `bwd_decay_cumsum_step_general`'s side-condition set (the loop
  stores into `dq_inter`/`dk_inter`/`dg` *between* its per-row re-reads
  of `g`/`q`/`k`/`dq_inner`/`dk_inner` and its own earlier outputs, so
  the invariant's whole-input frames and untouched-row conjuncts require
  the three output buffers not to alias any other buffer or each other;
  the launch allocates all eight tensors separately). The
  `dq_inter`-with-`dq_inter` / `dk_inter`-with-`dk_inter` *self*-aliasing
  is the in-place design, **not** a hypothesis.
* `hBK : BK ≤ DK` — row separation, as in the exact stack: the row-`m`
  scatters must not collide with other rows' windows, which needs every
  lane index `< BK` to stay inside one `DK`-wide row. The launch sets
  `BK = min(DK, 64)`.
* `hBT : 0 < BT` — the reverse loop's designated first row `BT − 1`
  (the `last_g` capture and the spec's `⟨BT − 1, _⟩ : Fin BT` index)
  needs a nonempty stream; the exact stack carries the same constraint
  (`bwd_prologue_eval_general`'s `hBT`), and the launch sets `BT = 16`.

Relation to the exact surface: the exact headline
`decay_cumsum_backward_closed_output_summary_general` above is retained
unchanged; this `⊨[R]` face restates all three closed forms at once on
the grouped streaming emit skin, for every `R` (at the `.real` grid the
two faces carry the same exact cells). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification bwd_decay_global_cumsum_io_correctness (R : RoundingModel)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    bwdDecayCumsumKernelIO DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK ⊨[R]
      fun _ _ _ xs o t j => bwdDecayStreamSpec BT BK xs o t j
```

**Assumptions / layout contracts:**
- `hDKInner_DQInter : DKInner ≠ DQInter`
- `hDKInter_DQInter : DKInter ≠ DQInter`
- `hQ_DQInter : Q ≠ DQInter`
- `hQ_DKInter : Q ≠ DKInter`
- `hK_DQInter : K ≠ DQInter`
- `hK_DKInter : K ≠ DKInter`
- `hDQInter_DKInter : DQInter ≠ DKInter`
- `hDQInter_DG : DQInter ≠ DG`
- `hDKInter_DG : DKInter ≠ DG`
- `hG_DQInter : G ≠ DQInter`
- `hG_DKInter : G ≠ DKInter`
- `hG_DG : G ≠ DG`
- `hQ_DG : Q ≠ DG`
- `hK_DG : K ≠ DG`
- `hDQInner_DQInter : DQInner ≠ DQInter`
- `hDQInner_DKInter : DQInner ≠ DKInter`
- `hDQInner_DG : DQInner ≠ DG`
- `hDKInner_DKInter : DKInner ≠ DKInter`
- `hDKInner_DG : DKInner ≠ DG`
- `hBK : BK ≤ DK`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `bwdDecayCumsumKernelIO`, `bwdDecayStreamSpec`, `bwd_decay_global_cumsum_surface`, `bwdDqStream`, `bwdDkStream`, `bwdDgStream`

<details><summary><code>bwdDecayCumsumKernelIO</code></summary>

```
/-- **Streaming IO signature** of `bwd_decay_global_cumsum` on the grouped
multi-channel per-step emit skin (S3: three in-loop stores, two of them
**in place**). Loop counter `m` processes PYTHON row `t = BT − 1 − m`;
the io is indexed by the *row* `t`, so step `t` of the signature is the
`(BT − 1 − t)`-th executed iteration. Step `t` reads the `BK`-lane rows
of the seven input channels (`0` = `g`, `1` = `dq_inner`, `2` =
`dq_inter` (old contents), `3` = `dk_inner`, `4` = `dk_inter` (old
contents), `5` = `q`, `6` = `k`) and stores the `BK`-lane
`dq_inter`/`dk_inter`/`dg` windows (output channels `0`/`1`/`2`) at the
**`.real`** grid (`outDType` default — the store's
`.to(p_dg.dtype.element_ty)` cast erases to `.real`).

**In-place channels**: output channels `0`/`1` name the same buffers as
input channels `2`/`4` — the skin's decoupled `bufs` allocation list
carries each region exactly once. Pinning the *old* contents on the
launch state is sound because the reverse loop visits each row exactly
once and reads its `dq_inter`/`dk_inter` cells *before* rewriting them in
the same iteration, so every read of those channels observes the launch
value (the invariant's untouched-rows conjunct `hDQIunt`/`hDKIunt` is
precisely this argument).

All ten windows share the surface's effective row address, transcribed
verbatim (`pid₀ = i_k`, `pid₁ = i_c`, `pid₂ = i_bh`; the surface's
pointers start at row `BT−1` and step `-DK`, so row `t`'s address is the
base plus `t·DK`):

`i_bh·s_qk_h + i_c·BT·DK + i_k·BK + j + t·DK`.

All masks are the kernel's single, `t`-independent bound mask
`i_k·BK + j < DK` — the `last_g` capture needs **no widening**: it is the
`g` stream's own step-`BT−1` row, loaded (masked, like every `g` row) in
the first executed iteration. -/
```
```lean
def bwdDecayCumsumKernelIO
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) : StreamGroupedEmitMasked3DKernelIO where
  kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
    s_qk_h DK BT BK
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>bwdDecayStreamSpec</code></summary>

```
/-- The stream-level `bwd_decay_global_cumsum` spec (the genre's *emit* +
*suffix-scan* shape, one clause per output channel): window `(t, j)` of
channel `0` (`dq_inter`) holds `dq_inner + dq_inter_old·exp2(g)`, of
channel `1` (`dk_inter`) holds `dk_inner + dk_inter_old·exp2(last_g − g)`
(with `last_g = g[BT−1]`, the `g` stream's own last row), of channel `2`
(`dg`) holds the suffix sum `Σ_{u ≥ t} (dq[u]·q[u] − dk[u]·k[u])` — the
exact spellings of the proven closed forms `bwdDQInterClosed` /
`bwdDKInterClosed` / `bwdDGClosed`, re-indexed to the curried streams.
The channel `match` lives in this single shared def (inline per-site
matchers block cross-declaration `exact`). -/
```
```lean
noncomputable def bwdDecayStreamSpec (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => bwdDqStream BT BK xs t j
  | ⟨1, _⟩ => bwdDkStream BT BK xs t j
  | ⟨2, _⟩ => bwdDgStream BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)
```
</details>

<details><summary><code>bwd_decay_global_cumsum_surface</code></summary>

```
/-- Surface transcription of `decay_cumsum.py`'s `bwd_decay_global_cumsum`.

The Python kernel traverses the chunk in reverse and decrements pointers; the
DSL surface preserves that reverse range and pointer movement directly. -/
```
```lean
def bwd_decay_global_cumsum_surface
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_k = K + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_g = G + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dg = DG + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inner = DQInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inner = DKInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inter = DQInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inter = DKInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + offs) < $(DK)
  last_g = tl.zeros([$(BK)], dtype=tl.float32)
  for t in range($(BT) - $(1), -$(1), -$(1)) {
    g_val = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    if t == $(BT) - $(1) {
      last_g = g_val
    }
    dq1 = tl.load(p_dq_inner, mask=mask, other=0)
    dq2 = tl.load(p_dq_inter, mask=mask, other=0)
    dq2 *= tl.math.exp2(g_val)
    dq = dq1 + dq2
    tl.store(p_dq_inter, dq, mask=mask)
    dk1 = tl.load(p_dk_inner, mask=mask, other=0)
    dk2 = tl.load(p_dk_inter, mask=mask, other=0)
    dk2 *= tl.math.exp2(last_g - g_val)
    dk = dk1 + dk2
    tl.store(p_dk_inter, dk, mask=mask)
    q_val = tl.load(p_q, mask=mask, other=0)
    k_val = tl.load(p_k, mask=mask, other=0)
    dg_val = dq * q_val - dk * k_val
    cum_grad_dg += dg_val
    tl.store(p_dg, (cum_grad_dg).to(p_dg.dtype.element_ty), mask=mask)
    p_g -= $(DK)
    p_k -= $(DK)
    p_q -= $(DK)
    p_dq_inner -= $(DK)
    p_dk_inner -= $(DK)
    p_dq_inter -= $(DK)
    p_dk_inter -= $(DK)
    p_dg -= $(DK)
  }
}
```
</details>

<details><summary><code>bwdDqStream</code></summary>

```
/-- Per-row `dq` stream value: `dq_inner + dq_inter_old · exp2(g)` (with
`exp2(x) = exp(x·log 2)`) — channel `0`'s clause and the `dg` summand's
first factor. -/
```
```lean
private noncomputable def bwdDqStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)
```
</details>

<details><summary><code>bwdDkStream</code></summary>

```
/-- Per-row `dk` stream value:
`dk_inner + dk_inter_old · exp2(last_g − g)` where `last_g` is the `g`
stream's step-`BT−1` row (the first processed row — no extra channel).
The `dite` keeps the definition total in `BT`; its `else` branch is
unreachable under the headline's `hBT : 0 < BT`. -/
```
```lean
private noncomputable def bwdDkStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0
```
</details>

<details><summary><code>bwdDgStream</code></summary>

```
/-- Per-row `dg` stream value: the reverse cumulative sum written at row
`t` is the **suffix sum** `Σ_{u = t}^{BT−1} (dq[u]·q[u] − dk[u]·k[u])`
over the per-row `dq`/`dk` values above (the executed loop folds rows
`BT−1, …, t` before storing row `t`). -/
```
```lean
private noncomputable def bwdDgStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (bwdDqStream BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - bwdDkStream BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)
```
</details>

## Also present (pinned special-case summaries)
- `fwd_decay_cumsum_surface_closed_compute_correct_general`
- `prepare_qg_kg_surface_qg_closed_compute_correct_general`
- `prepare_qg_kg_surface_kg_closed_compute_correct_general`
- `bwd_decay_cumsum_dq_inter_closed_compute_correct_general`
- `bwd_decay_cumsum_dk_inter_closed_compute_correct_general`
- `bwd_decay_cumsum_dg_closed_compute_correct_general`
