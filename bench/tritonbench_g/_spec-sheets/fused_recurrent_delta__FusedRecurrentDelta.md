# Spec sheet — `bench/tritonbench_g/fused_recurrent_delta/FusedRecurrentDelta.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_delta/fused_recurrent_delta.py`

## Public theorem: `fused_recurrent_delta_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about eight hand-cut single-step slices, not about the
launched kernels.** The launched surfaces appear only in clauses 1 and 2, which
say nothing more than "they lower to the algorithm layer". Clause 4 writes `HOut`,
a **fiction region** modelling the carried state register `h`, not a Python
tensor. Neither cross-step fold is modeled, and the `STORE_FINAL_STATE` writeback
has no face.

**Genuine, shape-general delta-rule step summary.** Parameterized
over the symbolic head strides `s_qk_h s_vo_h`, grid extents `NK B H T`,
key/value extents `K V`, tile sizes `BK BV`, the real `scale`, the step index
`m`, the final time `T`, and **all** compile-time flags (`IS_HEADWISE_BETA`,
`USE_INITIAL_STATE`, `STORE_FINAL_STATE`, `USE_DH0`, `USE_DHT`). It bundles
every stored output of both kernels, each realized against the genuine
standalone recurrence `deltaState` / the genuine per-step gradient formulas —
never a read-back of the kernel's own output:

1. the full **forward surface** lowers to the algorithm layer (all flags);
2. the full **backward surface** (both loops, `tl.debug_barrier()`, all
   flags) lowers to the algorithm layer;
3. one forward **`v_new` writeback** realizes the genuine delta
   `vNewClosed(m)` (masked), given the carry invariant
   `HPrev = deltaState(m)`;
4. one forward **state-update** body (the `IS_HEADWISE_BETA` specialization
   selected by the flag) realizes `deltaState(m+1)` — the delta-rule
   carry-fold — given `HPrev = deltaState(m)`;
5. one forward **output** body realizes `outputClosed(m)` (the key-axis
   readout of the *post-update* state), given `HNext = deltaState(m+1)`;
6. one backward loop-1 **`dk`** body (flag-selected) realizes the genuine
   `dkStepSpec` over the materialized reverse-carry `DHPrev` and input rows;
7. one backward loop-1 **`dv`** body (flag-selected) realizes `dvStepSpec`;
8. one backward loop-1 headwise **`dbeta`** body realizes `dbetaStepSpec`;
9. one backward loop-1 scalar **`dbeta`** body realizes the full-reduction
   `dbetaScalarStepSpec` (scalar cell);
10. one backward loop-2 **`dk` correction** body, *sequenced onto the loop-1
    stores*: under the two clause-local hypotheses that the `dk`/`dv` rows still
    hold clause 6's and clause 7's values (which is what Python's two-loop
    structure does — loop 2 re-walks the same `dk`/`dv` addresses forward after
    the `tl.debug_barrier()`), the in-place fixup realizes `dkCorrClosed`
    = `dkStepSpec − Σ_{j_v} dvStepSpec · HRec`, the final memory value of the
    `dk` row expressed through the loop-1 formulas rather than through
    unconstrained reads of the kernel's own output rows;
11. one backward loop-2 **`dq`** body (flag-selected) realizes `dqStepSpec`.

Clause 10 is why clause 6 and clause 10 are consistent rather than two
unrelated values for one `dk` cell: they are `Realizes` facts about *different*
slices run from the same initial state (loop 1's body and loop 2's body), and
clause 10 now names the loop-1 value it starts from instead of reading `dk` at
its own write address with nothing said about it.

The `STORE_FINAL_STATE` writeback has **no** conjunct: its face was a masked
memcpy (load address `HFinal + i_bh·K·V + offs_k·V + offs_v`, store address
`ht + …` — identical under the same mask) whose entire content was the
assumption `HFinal = deltaState(T)`, so it has been deleted rather than
presented as a result about `ht`.

Side conditions. Structural: `BV ≤ V`, `BK ≤ K` (the state tile fits the logical
extents, giving state-address injectivity); all row-address injectivities are
unconditional. There are **no** scope-fixing hypotheses left: every
slice in this file masks exactly where Python masks, and every specification is
guarded to match, so all eleven clauses hold for partial tiles. **Load-bearing:** `hPrev` and `hNext` are *assumptions* and they
carry the entire forward recurrence; every clause mentioning `deltaState`,
`vNewClosed` or `outputClosed` holds only under them.

**They are not self-propagating, and clause 4 does not discharge `hNext`.**
Clause 4 concludes about the state *after executing* the state-step slice, while
`hNext` constrains region `HOut` in the *initial* state `s` of the output slice;
sequencing one slice's post-state into the next slice's pre-state is not modeled.
(`HOut` is at least now the same region in both clauses — previously clause 4
wrote `HOut` while clause 5 read a separate `HNext` with nothing relating them, so
`deltaState(m+1)` was proved into a region nothing read and independently assumed
for the region that was read.) The cross-step folds threading `h`, `d_h` and the
recomputed `h` are not modeled (carried state presented as materialized buffers;
see the module docstring, including the in-place-`v` consequence).

By the time the backward kernel runs, region `v` holds the forward pass's stored
deltas `v_new` — the backward faces are genuine over the backward launch's actual
inputs. -/
```
</details>

**Statement:**
```lean
specification fused_recurrent_delta_output_summary_general
    (q k v beta o h0 ht dht dh0 do_ dq dk dv dbeta : RegionName)
    (HPrev HOut DHPrev HRec : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE STORE_FINAL_STATE USE_DH0 USE_DHT : Bool)
    (m s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hBV : BV ≤ V) (hBK : BK ≤ K)
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem HPrev (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx)
    -- the post-update carry buffer is the SAME region clause 4 stores into;
    -- this hypothesis is NOT discharged by clause 4 (see the docstring)
    (hNext : ∀ idx : TileIndex [BV, BK],
      s.readMem HOut (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV (m + 1) idx) :
    -- (1) the full forward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_delta_fwd_surface q k v beta o h0 ht s_qk_h s_vo_h
      B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      IS_HEADWISE_BETA).toAlgorithm? = Except.ok alg) ∧
    -- (2) the full backward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_delta_bwd_surface q k v beta dht dh0 do_ dq dk dv
      dbeta h0 s_qk_h s_vo_h NK B H T K V BK BV scale USE_INITIAL_STATE
      IS_HEADWISE_BETA USE_DH0 USE_DHT).toAlgorithm? = Except.ok alg) ∧
    -- (3) the `v_new` writeback realizes the genuine delta `vNewClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_vnew_step_slice HPrev k v
        m s_qk_h s_vo_h K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (v, vRowOffset s m s_vo_h V BV jv)))
      (expected := fun jv : Fin BV =>
        vNewClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV m jv)) ∧
    -- (4) the state-update body realizes the genuine `deltaState(m+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_state_step_slice_headwise HPrev k v beta HOut
            m s_qk_h s_vo_h T K V BK BV
        else
          fused_recurrent_delta_state_step_slice_scalarbeta HPrev k v beta HOut
            m s_qk_h s_vo_h T K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => activeKV s K V BK BV idx)
        (fun idx => (HOut, stateOffset s K V BK BV idx)))
      (expected := fun idx =>
        deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV (m + 1) idx)) ∧
    -- (5) the output body realizes the genuine `outputClosed(m)` — mask-faithful,
    --     so it holds for partial tiles and carries no full-tile antecedent
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_output_step_slice HOut q o
        m s_qk_h s_vo_h B H K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (o, outOffset s m s_vo_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        outputClosed s q k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV scale m jv)) ∧
    -- (6) the backward loop-1 `dk` body realizes the genuine `dkStepSpec` —
    --     mask-faithful, so no full-tile antecedent
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dk_step_slice_headwise DHPrev q do_ v beta dk
            m s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dk_step_slice_scalarbeta DHPrev q do_ v beta dk
            m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dk, dkRowOffset s m s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
          m s_qk_h s_vo_h T K V BK BV scale jk)) ∧
    -- (7) the backward loop-1 `dv` body realizes the genuine `dvStepSpec` —
    --     mask-faithful, so no full-tile antecedent
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dv_step_slice_headwise DHPrev q do_ k beta dv
            m s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dv_step_slice_scalarbeta DHPrev q do_ k beta dv
            m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (dv, outOffset s m s_vo_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
          m s_qk_h s_vo_h T K V BK BV scale jv)) ∧
    -- (8) the backward loop-1 headwise `dbeta` body realizes `dbetaStepSpec` —
    --     mask-faithful, so no full-tile antecedent
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dbeta_step_slice_headwise DHPrev q do_ k v
        dbeta m s_qk_h s_vo_h NK B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (dbeta, dbetaRowOffset s m s_vo_h B H NK V BV jv)))
      (expected := fun jv : Fin BV =>
        dbetaStepSpec s DHPrev q do_ k v m s_qk_h s_vo_h K V BK BV scale jv)) ∧
    -- (9) the backward loop-1 scalar `dbeta` body realizes `dbetaScalarStepSpec`
    --     — mask-faithful (guarded on both nested reductions)
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta DHPrev q do_ k v
        dbeta m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := fun _ : PUnit => some (dbeta, dbetaScalarOffset s m T B H))
      (expected := fun _ =>
        dbetaScalarStepSpec s DHPrev q do_ k v m s_qk_h s_vo_h K V BK BV scale)) ∧
    -- (10) the backward loop-2 `dk` correction body, **sequenced onto the loop-1
    --      stores**: given that the `dk`/`dv` rows still hold what clauses 6 and 7
    --      put there, the final `dk` row is `dkCorrClosed`
    ((∀ jk : Fin BK,
        s.readMem dk (dkRowOffset s m s_qk_h B H K BK jk)
          = dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
              m s_qk_h s_vo_h T K V BK BV scale jk) →
      (∀ jv : Fin BV,
        s.readMem dv (outOffset s m s_vo_h B H V BV jv)
          = dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
              m s_qk_h s_vo_h T K V BK BV scale jv) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_delta_bwd_dk_correction_step_slice HRec dv dk
          m s_qk_h s_vo_h B H K V BK BV)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun jk : Fin BK => activeK s K BK jk)
          (fun jk => (dk, dkRowOffset s m s_qk_h B H K BK jk)))
        (expected := fun jk : Fin BK =>
          dkCorrClosed s DHPrev HRec q do_ k v beta IS_HEADWISE_BETA
            m s_qk_h s_vo_h T K V BK BV scale jk)) ∧
    -- (11) the backward loop-2 `dq` body realizes the genuine `dqStepSpec` —
    --     mask-faithful, so no full-tile antecedent
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dq_step_slice_headwise HRec k v beta do_ dq
            m s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dq_step_slice_scalarbeta HRec k v beta do_ dq
            m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dq, dkRowOffset s m s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dqStepSpec s HRec k v beta do_ IS_HEADWISE_BETA
          m s_qk_h s_vo_h T K V BK BV scale jk))
```

**Assumptions / layout contracts:**
- `hBV : BV ≤ V`
- `hBK : BK ≤ K`
- `hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem HPrev (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx`
- `hNext : ∀ idx : TileIndex [BV, BK],
      s.readMem HOut (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV (m + 1) idx`

**Closed-form spec defs (transitive):** `stateOffset`, `deltaState`, `fused_recurrent_delta_fwd_surface`, `fused_recurrent_delta_bwd_surface`, `fused_recurrent_delta_vnew_step_slice`, `activeV`, `vRowOffset`, `vNewClosed`, `fused_recurrent_delta_state_step_slice_headwise`, `fused_recurrent_delta_state_step_slice_scalarbeta`, `activeKV`, `fused_recurrent_delta_output_step_slice`, `outOffset`, `outputClosed`, `fused_recurrent_delta_bwd_dk_step_slice_headwise`, `fused_recurrent_delta_bwd_dk_step_slice_scalarbeta`, `activeK`, `dkRowOffset`, `dkStepSpec`, `fused_recurrent_delta_bwd_dv_step_slice_headwise`, `fused_recurrent_delta_bwd_dv_step_slice_scalarbeta`, `dvStepSpec`, `fused_recurrent_delta_bwd_dbeta_step_slice_headwise`, `dbetaRowOffset`, `dbetaStepSpec`, `fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta`, `dbetaScalarOffset`, `dbetaScalarStepSpec`, `fused_recurrent_delta_bwd_dk_correction_step_slice`, `dkCorrClosed`, `fused_recurrent_delta_bwd_dq_step_slice_headwise`, `fused_recurrent_delta_bwd_dq_step_slice_scalarbeta`, `dqStepSpec`, `kIndex`, `vIndex`, `stateSeed`, `kVal`, `betaVal`, `vVal`, `vMinusClosed`, `qVal`, `dhOffset`, `h0Val`

<details><summary><code>stateOffset</code></summary>

```
/-- Flattened `[BV, BK]` state / `h0` / `ht` address
(row-major `i_bh·K·V + j_k·V + j_v`). -/
```
```lean
def stateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.2.1 * V + vIndex s BV idx.1
```
</details>

<details><summary><code>deltaState</code></summary>

```
/-- **Genuine delta-rule state recurrence.** Tile element `idx = (j_v, j_k)`:

```
deltaState 0       = stateSeed                    (h0 or 0)
deltaState (m+1)   = deltaState m
                     + k_m[j_k] · (β_m[j_v] · (v_m[j_v] − Σ_{j'_k} deltaState m [j_v,j'_k] · k_m[j'_k]))
```

A standalone specification over the input regions `k, v, beta, h0`. -/
```
```lean
noncomputable def deltaState (s : BlockState) (k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) :
    Nat → TileIndex [BV, BK] → ℝ
  | 0, idx => stateSeed s h0 USE_INITIAL_STATE K V BK BV idx
  | m + 1, idx =>
    deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
        s_qk_h s_vo_h T K V BK BV m idx
      + kVal s k s_qk_h K BK m idx.2.1 *
          (betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV m idx.1 *
            (vVal s v s_vo_h V BV m idx.1
              - ∑ jk : Fin BK,
                  if activeK s K BK jk then
                    deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
                        s_qk_h s_vo_h T K V BK BV m (idx.1, jk, PUnit.unit)
                      * kVal s k s_qk_h K BK m jk
                  else 0))
```
</details>

<details><summary><code>fused_recurrent_delta_fwd_surface</code></summary>

```
/-- Faithful transcription of `fused_recurrent_delta.py`'s
`fused_recurrent_fwd_kernel`.

All three `tl.constexpr` flags are `Bool` parameters gating the verbatim
branches; the `IS_HEADWISE_BETA` branches keep the Python `p_beta`/`b_beta`
shapes (`[BV]` row vs scalar). See the module docstring for the three
mechanical transcription notes (`_i` loop variable, `ℕ`-domain grouping of
trailing `- 1`, `if`-gate form of the conditional `p_beta` increment). -/
```
```lean
def fused_recurrent_delta_fwd_surface
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  if IS_HEADWISE_BETA {
    p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  } else {
    p_beta = beta + i_bh * $(T)
  }
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    if IS_HEADWISE_BETA {
      b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    } else {
      b_beta = tl.load(p_beta).to(tl.float32)
    }
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    if IS_HEADWISE_BETA {
      p_beta += $(V)
    } else {
      p_beta += $(1)
    }
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_surface</code></summary>

```
/-- Faithful transcription of `fused_recurrent_delta.py`'s
`fused_recurrent_bwd_kernel`: the reverse-time `d_h` scan storing
`dk`/`dv`/`dbeta`, the optional `dh0` store, the `tl.debug_barrier()` fence,
then the forward-time recomputation of `h` storing the `dk` correction and
`dq`. `do` is spelled `do_`; see the module docstring for the other two
mechanical notes. -/
```
```lean
def fused_recurrent_delta_bwd_surface
    (q k v beta dht dh0 do_ dq dk dv dbeta h0 : RegionName)
    (s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(K)
  mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(V)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  if IS_HEADWISE_BETA {
    p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  } else {
    p_beta = beta + i_bh * $(T) + ($(T) - $(1))
  }
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  if IS_HEADWISE_BETA {
    p_dbeta = dbeta + (i_bh + i_k * $(B) * $(H) + i_v * $(B) * $(H) * $(NK)) * $(s_vo_h) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  } else {
    p_dbeta = dbeta + (i_bh + i_v * $(B) * $(H)) * $(T) + ($(T) - $(1))
  }
  d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_DHT {
    p_ht = dht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
  }
  for _i in range($(T)) {
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA {
      b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    } else {
      b_beta = tl.load(p_beta).to(tl.float32)
    }
    d_h += b_q[:, None] * b_do[None, :]
    d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
    d_v = tl.sum(d_h * b_k[:, None], axis=0)
    if IS_HEADWISE_BETA {
      d_beta = d_v * b_v
    } else {
      d_beta = tl.sum(d_v * b_v)
    }
    d_v = d_v * b_beta
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA {
      tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty), mask=mask_bv)
    } else {
      tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty))
    }
    d_h -= b_k[:, None] * d_v[None, :]
    p_do -= $(V)
    p_q -= $(K)
    p_k -= $(K)
    p_v -= $(V)
    p_dk -= $(K)
    p_dv -= $(V)
    if IS_HEADWISE_BETA {
      p_dbeta -= $(V)
    } else {
      p_dbeta -= $(1)
    }
    if IS_HEADWISE_BETA {
      p_beta -= $(V)
    } else {
      p_beta -= $(1)
    }
  }
  if USE_DH0 {
    p_dh0 = dh0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    tl.store(p_dh0, (d_h).to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])
  }
  tl.debug_barrier()
  h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  if IS_HEADWISE_BETA {
    p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  } else {
    p_beta = beta + i_bh * $(T)
  }
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  if USE_INITIAL_STATE {
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for i in range($(0), $(T), $(1)) {
    d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
    d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
    d_k -= tl.sum(d_v[None, :] * h, axis=1)
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA {
      b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    } else {
      b_beta = tl.load(p_beta).to(tl.float32)
    }
    b_v *= b_beta
    h += b_k[:, None] * b_v[None, :]
    _d_q = h * b_do[None, :]
    d_q = tl.sum(_d_q, axis=1) * $(scale)
    tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
    p_k += $(K)
    p_do += $(V)
    p_v += $(V)
    p_dk += $(K)
    p_dv += $(V)
    p_dq += $(K)
    if IS_HEADWISE_BETA {
      p_beta += $(V)
    } else {
      p_beta += $(1)
    }
  }
}
```
</details>

<details><summary><code>fused_recurrent_delta_vnew_step_slice</code></summary>

```
/-! ## `v_new` writeback step slice (the in-place delta store)

One loop body's `v` writeback, isolated from the cross-step loop: with the
materialized pre-update state tile `HPrev`, it loads the time-`t` rows of `k`
and `v`, forms `v_minus = Σ_{j_k} HPrev·k_t` and masked-stores the delta
`v_t − v_minus` back into `v` at row `t` (the kernel's in-place update; the
row is read before it is written). -/
```
```lean
def fused_recurrent_delta_vnew_step_slice
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  tl.store(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    (b_v).to(v.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>activeV</code></summary>

```lean
def activeV (s : BlockState) (V BV : Nat) (jv : Fin BV) : Prop :=
  vIndex s BV jv < V
```
</details>

<details><summary><code>vRowOffset</code></summary>

```
/-- The `v`-layout row address at time `t`, lane `j_v` — shared by the `v`
in-place writeback and the `v`/`do`/headwise-`beta` loads. -/
```
```lean
def vRowOffset (s : BlockState) (t s_vo_h V BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V
```
</details>

<details><summary><code>vNewClosed</code></summary>

```
/-- **Genuine closed form of the in-place `v` writeback at step `m`**:
the delta `v_new = v_m − v_minus`. -/
```
```lean
noncomputable def vNewClosed (s : BlockState) (k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) (m : Nat) (jv : Fin BV) : ℝ :=
  vVal s v s_vo_h V BV m jv
    - vMinusClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
        s_qk_h s_vo_h T K V BK BV m jv
```
</details>

<details><summary><code>fused_recurrent_delta_state_step_slice_headwise</code></summary>

```
/-! ## State-update step slice (the delta-rule carry-fold body)

One loop body's state update, isolated from the cross-step loop: with the
materialized pre-update state tile `HPrev`, it loads the time-`t` rows of
`k`, `v` and `beta`, recomputes `v_minus`, the delta `b_v − v_minus`, the
beta scaling, and stores `HPrev + k_t ⊗ (β_t ⊙ v_new)` into the state buffer
`HOut`. The two `IS_HEADWISE_BETA` compile-time specializations differ in the
`b_beta` load shape (`[BV]` row vs scalar), exactly as in the surface. -/
```
```lean
def fused_recurrent_delta_state_step_slice_headwise
    (HPrev k v beta HOut : RegionName) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_v *= b_beta
  acc = prev + b_k[None, :] * b_v[:, None]
  tl.store(HOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(HOut.dtype.element_ty), mask=mask_kv)
}
```
</details>

<details><summary><code>fused_recurrent_delta_state_step_slice_scalarbeta</code></summary>

```lean
def fused_recurrent_delta_state_step_slice_scalarbeta
    (HPrev k v beta HOut : RegionName) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  b_v *= b_beta
  acc = prev + b_k[None, :] * b_v[:, None]
  tl.store(HOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(HOut.dtype.element_ty), mask=mask_kv)
}
```
</details>

<details><summary><code>activeKV</code></summary>

```
/-- Python's `mask_kv` tile predicate (`mask_bk[None, :] & mask_bv[:, None]`). -/
```
```lean
def activeKV (s : BlockState) (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : Prop :=
  activeK s K BK idx.2.1 ∧ activeV s V BV idx.1
```
</details>

<details><summary><code>fused_recurrent_delta_output_step_slice</code></summary>

```
/-! ## Output-readout step slice (the per-step `o_t` store)

One loop body's output store, isolated from the cross-step loop. The delta
rule reads out the **post-update** state, so this face consumes a materialized
post-update tile `HNext`: it loads `HNext` and the scaled `q_t` row, reduces
over the key axis and masked-stores the `[BV]` row into `o` at time row `t`.

`HNext` holding `deltaState(m+1)` is an *assumption* wherever this face is used
(the headline's `hNext`). The state-update face proves `deltaState(m+1)` about
the state *after executing* its own slice; nothing in this file sequences that
post-state into this face's pre-state. -/
```
```lean
def fused_recurrent_delta_output_step_slice
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HNext + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  _o = prev * b_q[None, :]
  _o = tl.sum(_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (_o).to(o.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>outOffset</code></summary>

```
/-- The masked output address at lane `j_v` for time row `t` — the kernel's
exact `o + (i_bh + i_k·B·H)·s_vo_h + i_v·BV + j_v + t·V` layout (also the
`dv` store layout of the backward kernel). -/
```
```lean
def outOffset (s : BlockState) (t s_vo_h B H V BV : Nat) (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV + jv.val + t * V
```
</details>

<details><summary><code>outputClosed</code></summary>

```
/-- **Genuine closed form of output row `m`, lane `j_v`** — the readout of the
*post-update* state:
`o_m[j_v] = Σ_{j_k} deltaState(m+1)[j_v,j_k] · scale·q_m[j_k]`. -/
```
```lean
noncomputable def outputClosed (s : BlockState) (q k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ) (m : Nat) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    if activeK s K BK jk then
      deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV (m + 1) (jv, jk, PUnit.unit)
        * qVal s q s_qk_h K BK scale m jk
    else 0
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dk_step_slice_headwise</code></summary>

```lean
def fused_recurrent_delta_bwd_dk_step_slice_headwise
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dk_step_slice_scalarbeta</code></summary>

```lean
def fused_recurrent_delta_bwd_dk_step_slice_scalarbeta
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}
```
</details>

<details><summary><code>activeK</code></summary>

```lean
def activeK (s : BlockState) (K BK : Nat) (jk : Fin BK) : Prop :=
  kIndex s BK jk < K
```
</details>

<details><summary><code>dkRowOffset</code></summary>

```
/-- The `dk`/`dq` store row address at time `t`, lane `j_k` — the kernel's
`(i_bh + i_v·B·H)·s_qk_h + i_k·BK + j_k + t·K` layout. -/
```
```lean
def dkRowOffset (s : BlockState) (t s_qk_h B H K BK : Nat) (jk : Fin BK) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK + jk.val + t * K
```
</details>

<details><summary><code>dkStepSpec</code></summary>

```
/-- The genuine per-lane `dk` step formula (loop 1, both beta modes):
`dk_t[j_k] = Σ_{j_v} (DHPrev + scale·q_t ⊗ do_t)[j_k,j_v] · (v_t[j_v]·β_t[j_v])`. -/
```
```lean
noncomputable def dkStepSpec (s : BlockState) (DHPrev q do_ v beta : RegionName)
    (IS_HEADWISE_BETA : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ)
    (jk : Fin BK) : ℝ :=
  ∑ jv : Fin BV,
    if activeV s V BV jv then
      (s.readMem DHPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
          + qVal s q s_qk_h K BK scale t jk * vVal s do_ s_vo_h V BV t jv)
        * (vVal s v s_vo_h V BV t jv *
            betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV t jv)
    else 0
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dv_step_slice_headwise</code></summary>

```lean
def fused_recurrent_delta_bwd_dv_step_slice_headwise
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_v = d_v * b_beta
  tl.store(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (d_v).to(dv.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dv_step_slice_scalarbeta</code></summary>

```lean
def fused_recurrent_delta_bwd_dv_step_slice_scalarbeta
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_v = d_v * b_beta
  tl.store(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (d_v).to(dv.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>dvStepSpec</code></summary>

```
/-- The genuine per-lane `dv` step formula (loop 1, both beta modes):
`dv_t[j_v] = (Σ_{j_k} (DHPrev + scale·q_t ⊗ do_t)[j_k,j_v] · k_t[j_k]) · β_t[j_v]`. -/
```
```lean
noncomputable def dvStepSpec (s : BlockState) (DHPrev q do_ k beta : RegionName)
    (IS_HEADWISE_BETA : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ)
    (jv : Fin BV) : ℝ :=
  (∑ jk : Fin BK,
      if activeK s K BK jk then
        (s.readMem DHPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
            + qVal s q s_qk_h K BK scale t jk * vVal s do_ s_vo_h V BV t jv)
          * kVal s k s_qk_h K BK t jk
      else 0)
    * betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV t jv
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dbeta_step_slice_headwise</code></summary>

```lean
def fused_recurrent_delta_bwd_dbeta_step_slice_headwise
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_beta = d_v * b_v
  tl.store(dbeta + (i_bh + i_k * $(B) * $(H) + i_v * $(B) * $(H) * $(NK)) * $(s_vo_h) +
    offs_v + $(t) * $(V), (d_beta).to(dbeta.dtype.element_ty), mask=mask_bv)
}
```
</details>

<details><summary><code>dbetaRowOffset</code></summary>

```
/-- The headwise `dbeta` store row address at time `t`, lane `j_v` — the
kernel's `(i_bh + i_k·B·H + i_v·B·H·NK)·s_vo_h + j_v + t·V` layout (a plain
`tl.arange(0, BV)`, with no `i_v·BV` term, exactly as in the Python). -/
```
```lean
def dbetaRowOffset (s : BlockState) (t s_vo_h B H NK V BV : Nat)
    (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H + s.pids 0 * B * H * NK) * s_vo_h + jv.val + t * V
```
</details>

<details><summary><code>dbetaStepSpec</code></summary>

```
/-- The genuine per-lane headwise `dbeta` step formula (loop 1):
`dbeta_t[j_v] = (Σ_{j_k} (DHPrev + scale·q_t ⊗ do_t)[j_k,j_v] · k_t[j_k]) · v_t[j_v]`
(the *pre-beta* `d_v`, as in the Python). -/
```
```lean
noncomputable def dbetaStepSpec (s : BlockState) (DHPrev q do_ k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) (jv : Fin BV) : ℝ :=
  (∑ jk : Fin BK,
      if activeK s K BK jk then
        (s.readMem DHPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
            + qVal s q s_qk_h K BK scale t jk * vVal s do_ s_vo_h V BV t jv)
          * kVal s k s_qk_h K BK t jk
      else 0)
    * vVal s v s_vo_h V BV t jv
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta</code></summary>

```lean
def fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_beta = tl.sum(d_v * b_v)
  tl.store(dbeta + (i_bh + i_v * $(B) * $(H)) * $(T) + $(t),
    (d_beta).to(dbeta.dtype.element_ty))
}
```
</details>

<details><summary><code>dbetaScalarOffset</code></summary>

```
/-- The scalar `dbeta` store address at time `t` — the kernel's
`(i_bh + i_v·B·H)·T + t` layout. -/
```
```lean
def dbetaScalarOffset (s : BlockState) (t T B H : Nat) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * T + t
```
</details>

<details><summary><code>dbetaScalarStepSpec</code></summary>

```
/-- The genuine scalar `dbeta` step formula (loop 1):
`dbeta_t = Σ_{j_v} d_v[j_v] · v_t[j_v]` (the full `tl.sum` reduction). -/
```
```lean
noncomputable def dbetaScalarStepSpec (s : BlockState) (DHPrev q do_ k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) : ℝ :=
  ∑ jv : Fin BV,
    if activeV s V BV jv then
      dbetaStepSpec s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale jv
    else 0
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dk_correction_step_slice</code></summary>

```
/-! ### Loop-2 faces: the `dk` correction and `dq` -/
```
```lean
def fused_recurrent_delta_bwd_dk_correction_step_slice
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  d_k = tl.load(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), mask=mask_bk, other=0.0)
  d_v = tl.load(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), mask=mask_bv, other=0.0)
  h = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  d_k -= tl.sum(d_v[None, :] * h, axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}
```
</details>

<details><summary><code>dkCorrClosed</code></summary>

```
/-- **The sequenced loop-2 `dk` correction**: loop 1's `dk` row minus the
recomputed-state readout of loop 1's `dv` row,
`dkStepSpec_t[j_k] − Σ_{j_v} dvStepSpec_t[j_v] · HRec[j_k,j_v]`. A formula over
the backward-launch inputs and the two materialized carries (`DHPrev`, `HRec`) —
no unconstrained read of the kernel's own output rows. -/
```
```lean
noncomputable def dkCorrClosed (s : BlockState)
    (DHPrev HRec q do_ k v beta : RegionName) (IS_HEADWISE_BETA : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ) (jk : Fin BK) : ℝ :=
  dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
      t s_qk_h s_vo_h T K V BK BV scale jk
    - ∑ jv : Fin BV,
        if activeV s V BV jv then
          dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
              t s_qk_h s_vo_h T K V BK BV scale jv
            * s.readMem HRec (dhOffset s K V BK BV (jk, jv, PUnit.unit))
        else 0
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dq_step_slice_headwise</code></summary>

```lean
def fused_recurrent_delta_bwd_dq_step_slice_headwise
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_v *= b_beta
  h = prev + b_k[:, None] * b_v[None, :]
  _d_q = h * b_do[None, :]
  d_q = tl.sum(_d_q, axis=1) * $(scale)
  tl.store(dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_q).to(dq.dtype.element_ty), mask=mask_bk)
}
```
</details>

<details><summary><code>fused_recurrent_delta_bwd_dq_step_slice_scalarbeta</code></summary>

```lean
def fused_recurrent_delta_bwd_dq_step_slice_scalarbeta
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  b_v *= b_beta
  h = prev + b_k[:, None] * b_v[None, :]
  _d_q = h * b_do[None, :]
  d_q = tl.sum(_d_q, axis=1) * $(scale)
  tl.store(dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_q).to(dq.dtype.element_ty), mask=mask_bk)
}
```
</details>

<details><summary><code>dqStepSpec</code></summary>

```
/-- The genuine per-lane `dq` step formula (loop 2, both beta modes), reading
the recomputed *post-update* state built from the materialized `HPrev`:
`dq_t[j_k] = (Σ_{j_v} (HPrev[j_k,j_v] + k_t[j_k]·(v_t[j_v]·β_t[j_v])) · do_t[j_v]) · scale`
(recall `v` here holds the forward pass's stored deltas `v_new`). -/
```
```lean
noncomputable def dqStepSpec (s : BlockState) (HPrev k v beta do_ : RegionName)
    (IS_HEADWISE_BETA : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ)
    (jk : Fin BK) : ℝ :=
  (∑ jv : Fin BV,
      if activeV s V BV jv then
        (s.readMem HPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
            + kVal s k s_qk_h K BK t jk *
                (vVal s v s_vo_h V BV t jv *
                  betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV t jv))
          * vVal s do_ s_vo_h V BV t jv
      else 0)
    * scale
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (s : BlockState) (BK : Nat) (jk : Fin BK) : Nat :=
  s.pids 1 * BK + jk.val
```
</details>

<details><summary><code>vIndex</code></summary>

```
/-! ## Index / offset helpers

`s.pids 0 = i_v`, `s.pids 1 = i_k`, `s.pids 2 = i_bh`. The forward state tile
is `[BV, BK]` (`idx.1` the value axis `j_v`, `idx.2.1` the key axis `j_k`);
the backward carried tiles (`d_h` and the recomputed `h`) are `[BK, BV]`
(`idx.1 = j_k`, `idx.2.1 = j_v`). Both share the flattened row-major state
memory layout `i_bh·K·V + j_k·V + j_v` of `h0`/`ht`. -/
```
```lean
def vIndex (s : BlockState) (BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 0 * BV + jv.val
```
</details>

<details><summary><code>stateSeed</code></summary>

```
/-- Seeded initial state `h^(0)`: `h0` if `USE_INITIAL_STATE` else `0`. -/
```
```lean
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : ℝ :=
  if USE_INITIAL_STATE then h0Val s h0 K V BK BV idx else 0
```
</details>

<details><summary><code>kVal</code></summary>

```
/-- Element `R[i_bh][t, j_k]` of the shared `[T, K]` **k-layout** at time row
`t`, key channel `j_k` (offset `i_bh·s_qk_h + (i_k·BK + j_k) + t·K`). Both `q`
and `k` block pointers use this layout. -/
```
```lean
noncomputable def kVal (s : BlockState) (r : RegionName)
    (s_qk_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  s.readMem r (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)
```
</details>

<details><summary><code>betaVal</code></summary>

```
/-- `β_t[j_v]`: the headwise v-layout row when `IS_HEADWISE_BETA`, else the
per-`(b,h,t)` scalar `beta[i_bh·T + t]` (broadcast over `j_v`). -/
```
```lean
noncomputable def betaVal (s : BlockState) (beta : RegionName)
    (IS_HEADWISE_BETA : Bool) (s_vo_h T V BV : Nat)
    (t : Nat) (jv : Fin BV) : ℝ :=
  if IS_HEADWISE_BETA then
    s.readMem beta (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)
  else
    s.readMem beta (s.pids 2 * T + t)
```
</details>

<details><summary><code>vVal</code></summary>

```
/-- Element `R[i_bh][t, j_v]` of the `[T, V]` **v-layout** at time row `t`,
value channel `j_v` (offset `i_bh·s_vo_h + (i_v·BV + j_v) + t·V`). The `v`,
`do` and headwise-`beta` block pointers use this layout. -/
```
```lean
noncomputable def vVal (s : BlockState) (r : RegionName)
    (s_vo_h V BV : Nat) (t : Nat) (jv : Fin BV) : ℝ :=
  s.readMem r (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)
```
</details>

<details><summary><code>vMinusClosed</code></summary>

```
/-- The state readout `v_minus` at step `m`, lane `j_v`:
`Σ_{j_k} deltaState(m)[j_v,j_k] · k_m[j_k]`. -/
```
```lean
noncomputable def vMinusClosed (s : BlockState) (k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) (m : Nat) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    if activeK s K BK jk then
      deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV m (jv, jk, PUnit.unit)
        * kVal s k s_qk_h K BK m jk
    else 0
```
</details>

<details><summary><code>qVal</code></summary>

```
/-- `q[t][j_k]·scale` — the kernel multiplies the loaded `q` row by `scale`. -/
```
```lean
noncomputable def qVal (s : BlockState) (q : RegionName)
    (s_qk_h K BK : Nat) (scale : ℝ) (t : Nat) (jk : Fin BK) : ℝ :=
  scale * kVal s q s_qk_h K BK t jk
```
</details>

<details><summary><code>dhOffset</code></summary>

```
/-- The same flattened state address for the backward `[BK, BV]` tile
orientation (`d_h`, loop-2 `h`). -/
```
```lean
def dhOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.1 * V + vIndex s BV idx.2.1
```
</details>

<details><summary><code>h0Val</code></summary>

```lean
noncomputable def h0Val (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  s.readMem h0 (stateOffset s K V BK BV idx)
```
</details>

## Public theorem: `fused_recurrent_delta_output_step_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `fused_recurrent_delta_rule.py`'s
output store: for every disjoint flat placement of `HNext` / `q` / `o`, every
program coordinate whose active lanes are in bounds, and every launch state whose
`HNext` state tile holds `xs` and whose `q` row holds `ys` on their active lanes,
the kernel terminates, every write-active lane of the `o` row holds the guarded
contraction `Σ_{j_k} ⟦i_k·BK + j_k < K⟧ · xs[j_v, j_k] · ys[j_k]`, and every other
memory cell is unchanged.

The `mask_bk` guard inside the sum is not a proof convenience: it is what Python's
`other = 0` loads do, and dropping it would make the face false for partial key
tiles. Dimension-general in `K`, `V`, `BK`, `BV`, `B`, `H` and the strides; stated
at `scale = 1`, where the kernel's `b_q = tl.load(...) * scale` is the loaded row
itself — the general-`scale` face is the same statement with `ys` scaled, already
carried by `fused_recurrent_delta_output_step_slice_correct`. Honest
side-condition: output-address injectivity at every program coordinate. -/
```
</details>

**Statement:**
```lean
specification fused_recurrent_delta_output_step_io_correctness
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun jv : Fin BV =>
        (p₂ + p₁ * B * H) * s_vo_h + p₀ * BV + jv.val + t * V)) :
    outputStepIO HNext q o t s_qk_h s_vo_h B H K V BK BV
      ⊨ fun _p₀ p₁ xs ys jv =>
          ∑ jk : Fin BK,
            if p₁ * BK + jk.val < K then
              xs (jv.1, jk, PUnit.unit) * ys (jk, PUnit.unit)
            else 0
```

**Closed-form spec defs (transitive):** `outputStepIO`, `fused_recurrent_delta_output_step_slice`

<details><summary><code>outputStepIO</code></summary>

```
/-- IO signature of the output store: the `[BV, BK]` state tile and the `[BK]`
`q` row contract into the `[BV]` `o` row. -/
```
```lean
noncomputable def outputStepIO (HNext q o : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) :
    Masked3DTileShapedKernelIO₂ where
  kernel := fused_recurrent_delta_output_step_slice HNext q o t s_qk_h s_vo_h B H
    K V BK BV 1
  in1 := HNext
  in2 := q
  out := o
  shape1 := [BV, BK]
  shape2 := [BK]
  shapeOut := [BV]
  read1 := fun p₀ p₁ p₂ idx =>
    p₂ * K * V + (p₁ * BK + idx.2.1.val) * V + (p₀ * BV + idx.1.val)
  read2 := fun _p₀ p₁ p₂ jk => p₂ * s_qk_h + p₁ * BK + jk.1.val + t * K
  write := fun p₀ p₁ p₂ jv =>
    (p₂ + p₁ * B * H) * s_vo_h + p₀ * BV + jv.1.val + t * V
  mask1 := fun p₀ p₁ _p₂ idx =>
    p₁ * BK + idx.2.1.val < K ∧ p₀ * BV + idx.1.val < V
  mask2 := fun _p₀ p₁ _p₂ jk => p₁ * BK + jk.1.val < K
  writeMask := fun p₀ _p₁ _p₂ jv => p₀ * BV + jv.1.val < V
```
</details>

<details><summary><code>fused_recurrent_delta_output_step_slice</code></summary>

```
/-! ## Output-readout step slice (the per-step `o_t` store)

One loop body's output store, isolated from the cross-step loop. The delta
rule reads out the **post-update** state, so this face consumes a materialized
post-update tile `HNext`: it loads `HNext` and the scaled `q_t` row, reduces
over the key axis and masked-stores the `[BV]` row into `o` at time row `t`.

`HNext` holding `deltaState(m+1)` is an *assumption* wherever this face is used
(the headline's `hNext`). The state-update face proves `deltaState(m+1)` about
the state *after executing* its own slice; nothing in this file sequences that
post-state into this face's pre-state. -/
```
```lean
def fused_recurrent_delta_output_step_slice
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HNext + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  _o = prev * b_q[None, :]
  _o = tl.sum(_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (_o).to(o.dtype.element_ty), mask=mask_bv)
}
```
</details>

## Also present (pinned special-case summaries)
- `fused_recurrent_delta_vnew_step_slice_compute_correct`
- `fused_recurrent_delta_state_step_slice_headwise_compute_correct`
- `fused_recurrent_delta_state_step_slice_scalarbeta_compute_correct`
- `fused_recurrent_delta_output_step_slice_compute_correct`
- `fused_recurrent_delta_bwd_dk_step_slice_compute_correct`
- `fused_recurrent_delta_bwd_dv_step_slice_compute_correct`
- `fused_recurrent_delta_bwd_dbeta_step_slice_headwise_compute_correct`
- `fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta_compute_correct`
- `fused_recurrent_delta_bwd_dk_correction_step_slice_compute_correct`
- `fused_recurrent_delta_bwd_dq_step_slice_compute_correct`
