# Spec sheet — `bench/tritonbench_g/fused_recurrent_retention/FusedRecurrentRetention.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_retention/fused_recurrent_retention.py`

## Public theorem: `fused_recurrent_retention_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**Genuine, dimension-general fused recurrent retention compute-correctness,
forward and backward.** Parameterized over the symbolic head strides
`s_qk_h s_vo_h`, batch/head/time `B H T`, head extents `DK DV`, tile sizes
`BK BV`, the real `scale`, the step index `m`, and **both** flags
`USE_INITIAL_STATE STORE_FINAL_STATE`. It bundles all faces of both launched
kernels, each realized against the genuine closed forms `stateClosed` /
`outClosed` / `dqClosed` / `dStateClosed` / `dkClosed` / `dvClosed` over the
*input* regions `q, k, v, do, initial_state` (never a read-back of the
kernel's own output):

1. the full **forward** surface lowers to the algorithm layer;
2. the full **backward** surface (both loops, `tl.debug_barrier()`, pointer
   rebasing and decrements) lowers to the algorithm layer;
3. one forward **output** body realizes `outClosed(m)` — the reduction of the
   post-update state `stateClosed(m+1)` against `scale·q_m`;
4. one **state-update** body realizes `stateClosed(m+1)` (the scalar-decay
   carry-fold `h = b_b·h + k_m ⊗ v_m`), given the carry invariant
   `HPrev = stateClosed(m)` — shared by the forward and backward-phase-1 loops;
5. the **final-state** writeback realizes `stateClosed(T)` (masked), given
   `HFinal = stateClosed(T)`;
6. one backward **`dq`** body realizes `dqClosed(m)` (post-update state
   reduced against `do_m`, then `·scale`);
7. one reverse **gradient-state carry** body realizes `b_b·dStateClosed(m)`,
   given the reverse invariant `DHPrev = b_b·dStateClosed(m+1)` (which is `0`
   at the first reverse iteration `m = T−1`, matching `tl.zeros`);
8. one reverse **`dk`** body realizes `dkClosed(m)`;
9. one reverse **`dv`** body realizes `dvClosed(m)`.

Honest structural side conditions only: `BV ≤ DV` (the tile fits the logical
extents, giving state-address injectivity), `0 < BK`, `0 < BV` (nonempty
contiguous output lanes), and `m < T` for the reverse-phase faces (the reverse
loop visits exactly the rows `T−1, …, 0`). The flags flow through verbatim;
clauses 3–9 hold for every flag setting, and each Python test case is
recovered by projecting the subset of clauses its `USE_INITIAL_STATE` /
`STORE_FINAL_STATE` configuration exercises. The carry invariants are
self-propagating: clause 4 advances `stateClosed(m) ↦ stateClosed(m+1)` from
the seed `stateClosed(0) = stateSeed` (`tl.zeros` + optional `initial_state`
load), and clause 7 advances the reverse invariant downward from
`b_b·dStateClosed(T) = 0`. -/
```
</details>

**Statement:**
```lean
theorem fused_recurrent_retention_output_summary_general
    (q k v o do_ dq dk dv initial_state final_state
      HPrev HOut HFinal DHPrev DHOut : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s_qk_h s_vo_h B H T DK DV BK BV m : Nat) (scale : ℝ) (s : BlockState)
    (hBV : BV ≤ DV) (hBKpos : 0 < BK) (hBVpos : 0 < BV)
    (hmT : m < T)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
            H DK DV BK BV m jk jv)
    (hFinal : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HFinal (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
            H DK DV BK BV T jk jv)
    (hDPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (m + 1) jk jv) :
    -- (1) the full forward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_retention_fwd_surface q k v o initial_state
      final_state s_qk_h s_vo_h B H T scale BK BV DK DV USE_INITIAL_STATE
      STORE_FINAL_STATE).toAlgorithm? = Except.ok alg) ∧
    -- (2) the full backward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_retention_bwd_surface q k v do_ dq dk dv
      initial_state s_qk_h s_vo_h B H T scale BK BV DK DV
      USE_INITIAL_STATE).toAlgorithm? = Except.ok alg) ∧
    -- (3) the forward output body realizes the genuine `outClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_output_step_slice HPrev q k v o
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s DV BV jv)
        (fun jv => (o, outStepOffset s m s_vo_h B H DV BV jv)))
      (expected := fun jv : Fin BV =>
        outClosed s q k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
          H DK DV BK BV scale m jv)) ∧
    -- (4) the state-update body realizes the genuine `stateClosed(m+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_state_step_slice HPrev k v HOut
        m s_qk_h s_vo_h H DK DV BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s DK DV BK BV idx.2.1 idx.1))
      (expected := fun idx =>
        stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
          H DK DV BK BV (m + 1) idx.2.1 idx.1)) ∧
    -- (5) the final-state writeback realizes the genuine `stateClosed(T)` (masked)
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_final_state_store_slice HFinal
        final_state DK DV BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => finalActive s DK DV BK BV idx)
        (fun idx : TileIndex [BV, BK] =>
          (final_state, stateOffset s DK DV BK BV idx.2.1 idx.1)))
      (expected := fun idx : TileIndex [BV, BK] =>
        if finalActive s DK DV BK BV idx then
          stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
            H DK DV BK BV T idx.2.1 idx.1
        else 0)) ∧
    -- (6) the backward `dq` body realizes the genuine `dqClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dq_step_slice HPrev k v do_ dq
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s DK BK jk)
        (fun jk => (dq, dqStepOffset s m s_qk_h B H DK BK jk)))
      (expected := fun jk : Fin BK =>
        dqClosed s k v do_ initial_state USE_INITIAL_STATE s_qk_h s_vo_h
          H DK DV BK BV scale m jk)) ∧
    -- (7) the reverse gradient-state carry body realizes `b_b·dStateClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dstate_step_slice DHPrev q do_
        DHOut m s_qk_h s_vo_h H DK DV BK BV scale)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (DHOut, stateOffset s DK DV BK BV idx.1 idx.2.1))
      (expected := fun idx =>
        bbVal s H *
          dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale m
            idx.1 idx.2.1)) ∧
    -- (8) the reverse `dk` body realizes the genuine `dkClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dk_step_slice DHPrev q do_ v dk
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s DK BK jk)
        (fun jk => (dk, dqStepOffset s m s_qk_h B H DK BK jk)))
      (expected := fun jk : Fin BK =>
        dkClosed s q do_ v s_qk_h s_vo_h H DK DV BK BV T scale m jk)) ∧
    -- (9) the reverse `dv` body realizes the genuine `dvClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dv_step_slice DHPrev q do_ k dv
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s DV BV jv)
        (fun jv => (dv, outStepOffset s m s_vo_h B H DV BV jv)))
      (expected := fun jv : Fin BV =>
        dvClosed s q do_ k s_qk_h s_vo_h H DK DV BK BV T scale m jv))
```

**Assumptions / layout contracts:**
- `hBV : BV ≤ DV`
- `hBKpos : 0 < BK`
- `hBVpos : 0 < BV`
- `hmT : m < T`
- `hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
            H DK DV BK BV m jk jv`
- `hFinal : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HFinal (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
            H DK DV BK BV T jk jv`
- `hDPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (m + 1) jk jv`
- `fun jv : Fin BV => activeV s DV BV jv`
- `fun idx : TileIndex [BV, BK] => finalActive s DK DV BK BV idx`
- `fun idx : TileIndex [BV, BK] =>
          (final_state, stateOffset s DK DV BK BV idx.2.1 idx.1)`
- `fun jk : Fin BK => activeK s DK BK jk`
- `fun jk : Fin BK => activeK s DK BK jk`
- `fun jv : Fin BV => activeV s DV BV jv`

**Closed-form spec defs (transitive):** `stateOffset`, `stateClosed`, `bbVal`, `dStateClosed`, `fused_recurrent_retention_fwd_surface`, `fused_recurrent_retention_bwd_surface`, `fused_recurrent_retention_output_step_slice`, `activeV`, `outStepOffset`, `outClosed`, `fused_recurrent_retention_state_step_slice`, `fused_recurrent_retention_final_state_store_slice`, `finalActive`, `fused_recurrent_retention_bwd_dq_step_slice`, `activeK`, `dqStepOffset`, `dqClosed`, `fused_recurrent_retention_bwd_dstate_step_slice`, `fused_recurrent_retention_bwd_dk_step_slice`, `dkClosed`, `fused_recurrent_retention_bwd_dv_step_slice`, `dvClosed`, `kIdx`, `vIdx`, `stateSeed`, `kValR`, `vValR`

<details><summary><code>stateOffset</code></summary>

```
/-- Flattened state / `initial_state` / `final_state` address
(`i_bh·DK·DV + (i_k·BK + j_k)·DV + (i_v·BV + j_v)`). -/
```
```lean
def stateOffset (s : BlockState) (DK DV BK BV : Nat)
    (jk : Fin BK) (jv : Fin BV) : Nat :=
  s.pids 2 * DK * DV + kIdx s BK jk * DV + vIdx s BV jv
```
</details>

<details><summary><code>stateClosed</code></summary>

```
/-- **Genuine closed form for the retention state after `m` steps** (key lane
`j_k`, value lane `j_v`):
`seed · b_b^m + Σ_{t<m} k_t[j_k]·v_t[j_v] · b_b^(m−1−t)`. A standalone
specification over the input regions `k, v, initial_state` — never a read-back
of the kernel's own output. Shared by the forward loop and the backward
phase-1 loop (same recurrence, same flat state layout). -/
```
```lean
noncomputable def stateClosed
    (s : BlockState) (k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (m : Nat)
    (jk : Fin BK) (jv : Fin BV) : ℝ :=
  stateSeed s h0 USE_INITIAL_STATE DK DV BK BV jk jv * bbVal s H ^ m +
    ∑ t ∈ Finset.range m,
      kValR s k s_qk_h DK BK t jk * vValR s v s_vo_h DV BV t jv *
        bbVal s H ^ (m - 1 - t)
```
</details>

<details><summary><code>bbVal</code></summary>

```
/-- Per-head retention decay `b_b = 1 − 2^(−5 − i_h)` with `i_h = i_bh % H`,
in the kernel's `exp2` semantics (`tl.math.exp2 x = Real.exp (x · log 2)`). -/
```
```lean
noncomputable def bbVal (s : BlockState) (H : Nat) : ℝ :=
  1 - Real.exp ((-5 - ((s.pids 2 % H : Nat) : ℝ)) * Real.log 2)
```
</details>

<details><summary><code>dStateClosed</code></summary>

```
/-- **Genuine closed form for the reverse-phase gradient state at time `t`**
(the value of `d_h` used by the `dk`/`dv` writebacks at row `t`):
`d_h^(t)[j_k,j_v] = Σ_{t ≤ u < T} scale·q_u[j_k] · do_u[j_v] · b_b^(u−t)`.
Over the input regions `q, do` only. -/
```
```lean
noncomputable def dStateClosed
    (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat)
    (jk : Fin BK) (jv : Fin BV) : ℝ :=
  ∑ u ∈ Finset.Ico t T,
    kValR s q s_qk_h DK BK u jk * scale * vValR s do_ s_vo_h DV BV u jv *
      bbVal s H ^ (u - t)
```
</details>

<details><summary><code>fused_recurrent_retention_fwd_surface</code></summary>

```
/-- Faithful transcription of `fused_recurrent_retention.py`'s
`fused_recurrent_retention_fwd_kernel`.

The per-head scalar decay `b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)` and the
`0..T` recurrence (including the optional initial-state seed, per-step masked
loads/stores, pointer `+=` advances, and the optional final-state store) are
represented directly. The unused stride arguments `s_qk_t/s_qk_d/s_vo_t/s_vo_d`
of the Python signature do not appear in the kernel body and are omitted. -/
```
```lean
def fused_recurrent_retention_fwd_surface
    (q k v o initial_state final_state : RegionName)
    (s_qk_h s_vo_h B H T : Nat) (scale : ℝ)
    (BK BV DK DV : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))

  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(DK)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(DV)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]

  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)

  if USE_INITIAL_STATE {
    p_init_s = initial_state + i_bh * $(DK) * $(DV) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
  }

  for _t in range($(0), $(T), $(1)) {
    _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)

    h = b_b * h + _k[None, :] * _v[:, None]
    _o = h * _q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)

    p_q += $(DK)
    p_k += $(DK)
    p_o += $(DV)
    p_v += $(DV)
  }

  if STORE_FINAL_STATE {
    p_final_s = final_state + i_bh * $(DK) * $(DV) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_final_s, (h).to(p_final_s.dtype.element_ty), mask=mask_kv)
  }
}
```
</details>

<details><summary><code>fused_recurrent_retention_bwd_surface</code></summary>

```
/-- Faithful transcription of `fused_recurrent_retention.py`'s
`fused_recurrent_retention_bwd_kernel`.

Phase 1 rebuilds the forward state `h` (now `[BK, BV]`) and stores `dq` rows;
the `tl.debug_barrier()` fence is transcribed verbatim (an intra-program
fence — a no-op at the algorithm layer); phase 2 rebases the pointers at time
row `T−1` (reassigning the same pointer names, as the Python does) and scans
backwards with pointer `-=` decrements, carrying the gradient state `d_h`. -/
```
```lean
def fused_recurrent_retention_bwd_surface
    (q k v do_ dq dk dv initial_state : RegionName)
    (s_qk_h s_vo_h B H T : Nat) (scale : ℝ)
    (BK BV DK DV : Nat)
    (USE_INITIAL_STATE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))

  mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(DK)
  mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(DV)

  h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)

  if USE_INITIAL_STATE {
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_init_s = initial_state + i_bh * $(DK) * $(DV) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(DV) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
  }

  for _i in range($(0), $(T), $(1)) {
    _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

    h = b_b * h + _k[:, None] * _v[None, :]
    _d_q = h * _do[None, :]
    d_q = tl.sum(_d_q, axis=1) * $(scale)
    tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)

    p_k += $(DK)
    p_do += $(DV)
    p_v += $(DV)
    p_dq += $(DK)
  }

  tl.debug_barrier()

  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
  d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)

  for _i in range($(0), $(T), $(1)) {
    _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    d_h += _q[:, None] * _do[None, :]
    d_k = tl.sum(d_h * _v[None, :], axis=1)
    d_v = tl.sum(d_h * _k[:, None], axis=0)

    d_h *= b_b
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)

    p_do -= $(DV)
    p_q -= $(DK)
    p_k -= $(DK)
    p_v -= $(DV)
    p_dk -= $(DK)
    p_dv -= $(DV)
  }
}
```
</details>

<details><summary><code>fused_recurrent_retention_output_step_slice</code></summary>

```
/-! ## Output-reduction step slice (the forward per-step `o_t`)

Isolates one forward loop body's output computation: with the materialized
*pre-update* state tile `HPrev`, it recomputes `b_b`, the post-update state
`h = b_b·prev + k_t ⊗ v_t`, forms `h · (scale·q_t)`, reduces over the key axis
(`tl.sum(_, axis=1)`), and masked-stores the `[BV]` row into `o` at time row
`t`. -/
```
```lean
def fused_recurrent_retention_output_step_slice
    (HPrev q k v o : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(DV)
  prev = tl.load(HPrev + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[None, :]) * $(DV) + (i_v * $(BV) + offs_v[:, None]))
  _k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK))
  _v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  _q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK)) * $(scale)
  h = b_b * prev + _k[None, :] * _v[:, None]
  _o = h * _q[None, :]
  _o = tl.sum(_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(DV), (_o).to(o.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>activeV</code></summary>

```lean
def activeV (s : BlockState) (DV BV : Nat) (jv : Fin BV) : Prop :=
  vIdx s BV jv < DV
```
</details>

<details><summary><code>outStepOffset</code></summary>

```
/-- The kernel's per-time output-row address for the `v`-layout outputs `o`
(forward) and `dv` (backward): `(i_bh + i_k·B·H)·s_vo_h + i_v·BV + j_v + t·DV`. -/
```
```lean
def outStepOffset (s : BlockState) (t s_vo_h B H DV BV : Nat) (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV + jv.val + t * DV
```
</details>

<details><summary><code>outClosed</code></summary>

```
/-- **Genuine closed form for forward output row `m`, lane `j_v`** — the
reduction over key lanes of the **post-update** state:
`o_m[j_v] = Σ_{j_k} h^(m+1)[j_k,j_v] · scale·q_m[j_k]`. Over the input regions
`q, k, v, initial_state` only. -/
```
```lean
noncomputable def outClosed
    (s : BlockState) (q k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (m : Nat)
    (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
      (m + 1) jk jv *
    (kValR s q s_qk_h DK BK m jk * scale)
```
</details>

<details><summary><code>fused_recurrent_retention_state_step_slice</code></summary>

```
/-! ## State-update step slice (the shared forward / backward-phase-1 carry-fold body)

Isolates one loop body's state update from the cross-step loop induction: it
loads the materialized previous-state tile `HPrev` (flat state layout), the
time-`t` rows `k_t`/`v_t`, recomputes the per-head decay `b_b`, and stores
`b_b·h + k_t ⊗ v_t` into a state buffer `HOut` at the same layout. The forward
loop (`[BV,BK]` axes) and the backward phase-1 loop (`[BK,BV]` axes) carry the
*same* recurrence at the *same* flat addresses; this canonical `[BV,BK]` slice
realizes that shared carry-fold step. -/
```
```lean
def fused_recurrent_retention_state_step_slice
    (HPrev k v HOut : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  prev = tl.load(HPrev + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[None, :]) * $(DV) + (i_v * $(BV) + offs_v[:, None]))
  _k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK))
  _v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  acc = b_b * prev + _k[None, :] * _v[:, None]
  tl.store(HOut + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[None, :]) * $(DV) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(HOut.dtype.element_ty))
}
```
</details>

<details><summary><code>fused_recurrent_retention_final_state_store_slice</code></summary>

```
/-! ## Final-state store slice (the `STORE_FINAL_STATE` branch)

After the forward loop, the kernel masked-stores the final state tile `h` into
`final_state`. This slice models that writeback exactly, reading the
materialized final-state tile `HFinal` and writing the masked `[BV,BK]` face
into `Ht` at the flat state layout (with the kernel's
`mask_bk[None,:] & mask_bv[:,None]` mask orientation). -/
```
```lean
def fused_recurrent_retention_final_state_store_slice
    (HFinal Ht : RegionName) (DK DV BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask_kv = (offs_k[None, :] < $(DK)) & (offs_v[:, None] < $(DV))
  h = tl.load(HFinal + i_bh * $(DK) * $(DV) +
      offs_k[None, :] * $(DV) + offs_v[:, None],
    mask=mask_kv, other=0.0)
  tl.store(Ht + i_bh * $(DK) * $(DV) +
      offs_k[None, :] * $(DV) + offs_v[:, None],
    h, mask=mask_kv)
}
```
</details>

<details><summary><code>finalActive</code></summary>

```
/-- The `mask_kv` predicate of the forward `[BV, BK]` state faces
(`mask_bk[None,:] & mask_bv[:,None]`). -/
```
```lean
def finalActive (s : BlockState) (DK DV BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Prop :=
  activeK s DK BK idx.2.1 ∧ activeV s DV BV idx.1
```
</details>

<details><summary><code>fused_recurrent_retention_bwd_dq_step_slice</code></summary>

```
/-! ## Backward phase-1 `dq` step slice

One backward phase-1 loop body: with the materialized *pre-update* rebuilt
state `HPrev` (`[BK,BV]` axes, same flat layout), it recomputes `b_b`, the
post-update state `h = b_b·prev + k_t ⊗ v_t`, reduces `h · do_t` over the
value axis, multiplies by `scale`, and masked-stores the `[BK]` row into `dq`
at time row `t`. -/
```
```lean
def fused_recurrent_retention_bwd_dq_step_slice
    (HPrev k v do_ dq : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(DK)
  prev = tl.load(HPrev + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[:, None]) * $(DV) + (i_v * $(BV) + offs_v[None, :]))
  _k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK))
  _v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  _do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  h = b_b * prev + _k[:, None] * _v[None, :]
  _d_q = h * _do[None, :]
  d_q = tl.sum(_d_q, axis=1) * $(scale)
  tl.store(dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(DK), (d_q).to(dq.dtype.element_ty), mask=mask_bk)
}
```
</details>

<details><summary><code>activeK</code></summary>

```lean
def activeK (s : BlockState) (DK BK : Nat) (jk : Fin BK) : Prop :=
  kIdx s BK jk < DK
```
</details>

<details><summary><code>dqStepOffset</code></summary>

```
/-- The kernel's per-time output-row address for the `k`-layout gradient
outputs `dq` and `dk`: `(i_bh + i_v·B·H)·s_qk_h + i_k·BK + j_k + t·DK`. -/
```
```lean
def dqStepOffset (s : BlockState) (t s_qk_h B H DK BK : Nat) (jk : Fin BK) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK + jk.val + t * DK
```
</details>

<details><summary><code>dqClosed</code></summary>

```
/-- **Genuine closed form for the backward `dq` row `m`, lane `j_k`** — the
phase-1 reduction over value lanes of the **post-update** rebuilt state:
`dq_m[j_k] = (Σ_{j_v} h^(m+1)[j_k,j_v] · do_m[j_v]) · scale`. -/
```
```lean
noncomputable def dqClosed
    (s : BlockState) (k v do_ h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (m : Nat)
    (jk : Fin BK) : ℝ :=
  (∑ jv : Fin BV,
    stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
      (m + 1) jk jv *
    vValR s do_ s_vo_h DV BV m jv) * scale
```
</details>

<details><summary><code>fused_recurrent_retention_bwd_dstate_step_slice</code></summary>

```
/-! ## Backward phase-2 gradient-state carry step slice

One reverse loop body's `d_h` update, including the trailing `d_h *= b_b`:
with the materialized incoming carry `DHPrev` (`[BK,BV]` axes, flat state
layout), it loads `scale·q_t` and `do_t`, forms `d_h = DHPrev + q_t ⊗ do_t`,
recomputes `b_b`, and stores `d_h · b_b` — the carry handed to time row `t−1`
— into `DHOut`. -/
```
```lean
def fused_recurrent_retention_bwd_dstate_step_slice
    (DHPrev q do_ DHOut : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  prev = tl.load(DHPrev + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[:, None]) * $(DV) + (i_v * $(BV) + offs_v[None, :]))
  _do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  _q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK)) * $(scale)
  d_h = prev + _q[:, None] * _do[None, :]
  d_h = d_h * b_b
  tl.store(DHOut + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[:, None]) * $(DV) + (i_v * $(BV) + offs_v[None, :]),
    (d_h).to(DHOut.dtype.element_ty))
}
```
</details>

<details><summary><code>fused_recurrent_retention_bwd_dk_step_slice</code></summary>

```
/-! ## Backward phase-2 `dk` / `dv` step slices

One reverse loop body's writebacks: with the incoming carry `DHPrev`, form
`d_h = DHPrev + scale·q_t ⊗ do_t` (the value the kernel's `d_h += …` produces
at row `t`) and reduce against `v_t` over the value axis (`dk`, `axis=1`) /
against `k_t` over the key axis (`dv`, `axis=0`), masked-storing the rows. -/
```
```lean
def fused_recurrent_retention_bwd_dk_step_slice
    (DHPrev q do_ v dk : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(DK)
  prev = tl.load(DHPrev + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[:, None]) * $(DV) + (i_v * $(BV) + offs_v[None, :]))
  _do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  _q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK)) * $(scale)
  _v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  d_h = prev + _q[:, None] * _do[None, :]
  d_k = tl.sum(d_h * _v[None, :], axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(DK), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}
```
</details>

<details><summary><code>dkClosed</code></summary>

```
/-- **Genuine closed form for the backward `dk` row `t`, lane `j_k`**:
`dk_t[j_k] = Σ_{j_v} d_h^(t)[j_k,j_v] · v_t[j_v]`. -/
```
```lean
noncomputable def dkClosed
    (s : BlockState) (q do_ v : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat)
    (jk : Fin BK) : ℝ :=
  ∑ jv : Fin BV,
    dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale t jk jv *
      vValR s v s_vo_h DV BV t jv
```
</details>

<details><summary><code>fused_recurrent_retention_bwd_dv_step_slice</code></summary>

```lean
def fused_recurrent_retention_bwd_dv_step_slice
    (DHPrev q do_ k dv : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(DV)
  prev = tl.load(DHPrev + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + offs_k[:, None]) * $(DV) + (i_v * $(BV) + offs_v[None, :]))
  _do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(DV))
  _q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK)) * $(scale)
  _k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(DK))
  d_h = prev + _q[:, None] * _do[None, :]
  d_v = tl.sum(d_h * _k[:, None], axis=0)
  tl.store(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(DV), (d_v).to(dv.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>dvClosed</code></summary>

```
/-- **Genuine closed form for the backward `dv` row `t`, lane `j_v`**:
`dv_t[j_v] = Σ_{j_k} d_h^(t)[j_k,j_v] · k_t[j_k]`. -/
```
```lean
noncomputable def dvClosed
    (s : BlockState) (q do_ k : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat)
    (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale t jk jv *
      kValR s k s_qk_h DK BK t jk
```
</details>

<details><summary><code>kIdx</code></summary>

```
/-! ## Index / offset helpers

`s.pids 0 = i_v`, `s.pids 1 = i_k`, `s.pids 2 = i_bh`. The forward state tile
is `[BV, BK]` (`idx.1 = j_v`, `idx.2.1 = j_k`); the backward tiles are
`[BK, BV]` (`idx.1 = j_k`, `idx.2.1 = j_v`). Both share the flat
`initial_state`/`final_state` layout `i_bh·DK·DV + (i_k·BK + j_k)·DV +
(i_v·BV + j_v)`. -/
```
```lean
def kIdx (s : BlockState) (BK : Nat) (jk : Fin BK) : Nat :=
  s.pids 1 * BK + jk.val
```
</details>

<details><summary><code>vIdx</code></summary>

```lean
def vIdx (s : BlockState) (BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 0 * BV + jv.val
```
</details>

<details><summary><code>stateSeed</code></summary>

```
/-- Seeded initial state `h^(0)`: `initial_state` if `USE_INITIAL_STATE`
else `0`. -/
```
```lean
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (DK DV BK BV : Nat)
    (jk : Fin BK) (jv : Fin BV) : ℝ :=
  if USE_INITIAL_STATE then s.readMem h0 (stateOffset s DK DV BK BV jk jv) else 0
```
</details>

<details><summary><code>kValR</code></summary>

```
/-- Element `R[i_bh][t, j_k]` of the shared `[T, DK]` **k-layout** at time row
`t`, key lane `j_k` (offset `i_bh·s_qk_h + i_k·BK + j_k + t·DK`). The `q` and
`k` block pointers both use this layout. -/
```
```lean
noncomputable def kValR (s : BlockState) (k : RegionName)
    (s_qk_h DK BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * DK)
```
</details>

<details><summary><code>vValR</code></summary>

```
/-- Element `R[i_bh][t, j_v]` of the shared `[T, DV]` **v-layout** at time row
`t`, value lane `j_v` (offset `i_bh·s_vo_h + i_v·BV + j_v + t·DV`). The `v`
and `do` block pointers both use this layout. -/
```
```lean
noncomputable def vValR (s : BlockState) (v : RegionName)
    (s_vo_h DV BV : Nat) (t : Nat) (jv : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * DV)
```
</details>

## Also present (pinned special-case summaries)
- `fused_recurrent_retention_state_step_slice_compute_correct`
- `fused_recurrent_retention_output_step_slice_compute_correct`
- `fused_recurrent_retention_final_state_store_slice_compute_correct`
- `fused_recurrent_retention_bwd_dq_step_slice_compute_correct`
- `fused_recurrent_retention_bwd_dstate_step_slice_compute_correct`
- `fused_recurrent_retention_bwd_dk_step_slice_compute_correct`
- `fused_recurrent_retention_bwd_dv_step_slice_compute_correct`
