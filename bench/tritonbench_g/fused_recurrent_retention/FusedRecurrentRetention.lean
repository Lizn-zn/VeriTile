import VeriTile.Triton

/-!
# `fused_recurrent_retention` — genuine retention closed-form correctness

`fused_recurrent_retention_fwd_kernel` is the flash-linear-attention fused
recurrent retention forward scan. Program `(i_v, i_k, i_bh)` carries a
`[BV, BK]` state tile `h` across the `0..T` time loop with the **scalar
per-head decay** `b_b = 1 − 2^(−5 − i_h)` (`i_h = i_bh % H`, computed with
`tl.math.exp2`). Per step `t`:

* `h[j_v,j_k] = b_b · h[j_v,j_k] + k_t[j_k] · v_t[j_v]`   (decayed outer-product update)
* `o_t[j_v]   = Σ_{j_k} h[j_v,j_k] · scale·q_t[j_k]`      (reduction over the *post-update* state)

with an optional `initial_state` seed and an optional `final_state` store.
The companion `fused_recurrent_retention_bwd_kernel` (exercised by the test
path through `loss.backward()`) runs two phases separated by a
`tl.debug_barrier()`: a forward-time scan re-building the same state `h` and
writing `dq_t = scale · Σ_{j_v} h[j_k,j_v] · do_t[j_v]`, then a reverse-time
scan carrying the gradient state `d_h += scale·q_t ⊗ do_t` (decayed by `b_b`
*after* each step's writebacks) and writing
`dk_t[j_k] = Σ_{j_v} d_h[j_k,j_v] · v_t[j_v]`,
`dv_t[j_v] = Σ_{j_k} d_h[j_k,j_v] · k_t[j_k]`.

## Scope — what is and is not verified

Verified: **six hand-cut single-step slices** — the forward output row, the
forward/backward-phase-1 state update, the backward `dq` row, the reverse
gradient-state carry, and the reverse `dk` / `dv` rows. Two of those six write
*fiction regions* (`HOut`, `DHOut`) that model the internal carry registers `h`
and `d_h`, not Python tensors.

**Not** verified: the launched kernels. Both `@triton.jit` bodies are shown only
to *lower* to the algorithm layer (`*_toAlgorithm_supported`), which is
well-formedness, not correctness. Neither cross-step fold is modeled, and the
`STORE_FINAL_STATE` writeback has no correctness face (see below).

The host launch (grid `(NV, NK, B·H)`, the `o.sum(0)` / `dq.sum(0)` / …
cross-`i_k`/`i_v` reductions, and stride computation) is the *trusted boundary*.
Program ids are universally quantified, so each per-program statement covers
every program of the grid.

## Genuine closed form (NOT a self-reference)

Unrolling the recurrence, the state after `m` steps (key lane `j_k`, value
lane `j_v`) is

```
h^(m)[j_k,j_v] = seed[j_k,j_v] · b_b^m + Σ_{t<m} k_t[j_k]·v_t[j_v] · b_b^(m−1−t)
```

— `stateClosed` below, a standalone specification over the *input* regions
`k, v, initial_state`, never a read-back of the kernel's own output. The
forward output at time row `t` reads the **post-update** state:

```
o_t[j_v] = Σ_{j_k} h^(t+1)[j_k,j_v] · scale·q_t[j_k]
         = Σ_{s ≤ t} b_b^(t−s) · (Σ_{j_k} scale·q_t[j_k]·k_s[j_k]) · v_s[j_v]  (+ seed term)
```

(`outClosed`; the second display is `outClosed_as_decayed_dots`), the
`STORE_FINAL_STATE` writeback is `stateClosed` at `m = T`, and the backward
`dq` row is the same reduction against `do`. The reverse-phase gradient state
is

```
d_h^(t)[j_k,j_v] = Σ_{t ≤ u < T} scale·q_u[j_k] · do_u[j_v] · b_b^(u−t)
```

(`dStateClosed`), with `dk`/`dv` its reductions against `v_t` / `k_t`.

## Proof architecture

```
fused_recurrent_retention_output_summary_general              ← TOP THEOREM
  ├─ fused_recurrent_retention_{fwd,bwd}_surface_toAlgorithm_supported
  ├─ fused_recurrent_retention_seed_slice_realizes_stateClosed_zero
  │      └─ stateClosed_zero                                   h^(0) = seed
  ├─ bb_mul_dStateClosed_top                                   b_b·d_h^(T) = 0
  │      └─ dStateClosed_of_top_le
  ├─ fused_recurrent_retention_output_step_closed_form         o_t   = Σ h^(t+1)·scale·q_t
  │      └─ outputStepSpec_eq_outClosed ── stateClosed_succ
  ├─ fused_recurrent_retention_state_step_closed_form          h^(t+1) = b_b·h^(t) + k_t⊗v_t
  │      └─ stateStepSpec_eq_stateClosed_succ ── stateClosed_succ
  ├─ fused_recurrent_retention_bwd_dq_step_closed_form         dq_t  = scale·Σ h^(t+1)·do_t
  ├─ fused_recurrent_retention_bwd_dstate_step_closed_form     d_h carry: b_b·d_h^(t)
  │      └─ bwdDStateStepSpec_eq ── dStateClosed_recurrence
  ├─ fused_recurrent_retention_bwd_dk_step_closed_form         dk_t  = Σ d_h^(t)·v_t
  └─ fused_recurrent_retention_bwd_dv_step_closed_form         dv_t  = Σ d_h^(t)·k_t
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. `tl.math.exp2 x` is modeled as `Real.exp (x · log 2)`,
so `b_b = 1 − 2^(−5 − i_h)` exactly (`bbVal`). The `USE_INITIAL_STATE` branch is
modeled (inside `stateSeed`); `tl.debug_barrier()` is transcribed verbatim (an
intra-program fence, a no-op at the algorithm layer).

The **`STORE_FINAL_STATE` writeback has no correctness face.** The former
`final_state_store_slice` face was a masked memcpy — load address
`HFinal + i_bh·DK·DV + offs_k·DV + offs_v`, store address
`final_state + i_bh·DK·DV + offs_k·DV + offs_v`, identical under the same mask —
so its entire content was the assumption `HFinal = stateClosed(T)`. It has been
deleted rather than presented as a result about `final_state`.

The cross-step folds over `range(0, T)` threading `h` (forward and backward phase
1) and `d_h` (backward phase 2) are **not modeled**: the carried state is
presented to each step slice as a materialized buffer (`HPrev`, `DHPrev`, at the
flat state layout). The **backward** carry invariant
`DHPrev = b_b·dStateClosed(m+1)` is still *assumed*, not proven, and is **not
self-propagating**: that body writes `DHOut` while the invariant constrains
`DHPrev`, so nothing chains step `m` into step `m+1` there.

The **forward** state carry is no longer in that position.
`fused_recurrent_retention_state_carry_fold` chains it: running the state slices
through one shared carry region `C` reaches `stateClosed(T)` from a single
assumption about the initial buffer. What that fold models is a
memory-threaded, one-launch-per-step program; the launched kernel keeps `h` in a
*register* across its loop, and that object remains unmodeled.

Both **base cases are theorems**. `stateClosed_zero` proves `h^(0) = stateSeed`, and
`fused_recurrent_retention_seed_slice_realizes_stateClosed_zero` anchors it to the
kernel's own `USE_INITIAL_STATE` prologue (`fused_recurrent_retention.py:30–36`),
for **both** flag settings, via `fused_recurrent_retention_seed_slice`.
`bb_mul_dStateClosed_top` proves the reverse invariant's top value
`b_b·d_h^(T) = 0`, matching the Python `d_h = tl.zeros([BK, BV], ...)` entering the
reverse loop. The seed slice writes the fiction region `HSeed` (the Python `h` is
a register the prologue never stores), so it is a statement about the seeded carry
register, not about a Python tensor. The forward and backward-phase-1 state loops share
one recurrence and one flat state layout, so a single state-update step slice
(canonical `[BV, BK]` axes) serves both. Following the `fused_rwkv6_kernel`
precedent, the step slices read the per-time `q/k/v/do` rows unmasked: the
kernel's in-loop masked loads (`other=0` on lanes past `DK`/`DV`) coincide on
in-bounds lanes, and the partial-tail-block regime (`i_k·BK + BK > DK` or
`i_v·BV + BV > DV`) is part of the trusted boundary. Store-offset injectivity
and the tile-fit bound `BV ≤ DV` are honest structural side conditions of the
headline.

Translation-surface blocker: the Python kernel signatures carry four stride
parameters that are never used in either kernel body (`s_qk_t`, `s_qk_d`,
`s_vo_t`, `s_vo_d` — dead arguments; addressing uses only `s_qk_h`/`s_vo_h`
and literal element strides); they are omitted from the Lean surface binders
(`fused_rwkv6_kernel`/`fused_recurrent_hgrn` precedent). Both kernel bodies
are otherwise transcribed verbatim, including `tl.debug_barrier()` and the
phase-2 pointer rebasing.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `fused_recurrent_retention_output_summary_general` —
shape-general, genuine closed forms `stateClosed` / `outClosed` / `dStateClosed`
over the input regions, but scoped to six single-step **slices** (two of which
write internal carry registers, not Python tensors), not to the launched
kernels. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `fused_recurrent_retention.py`'s
`fused_recurrent_retention_fwd_kernel`.

The per-head scalar decay `b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)` and the
`0..T` recurrence (including the optional initial-state seed, per-step masked
loads/stores, pointer `+=` advances, and the optional final-state store) are
represented directly. The unused stride arguments `s_qk_t/s_qk_d/s_vo_t/s_vo_d`
of the Python signature do not appear in the kernel body and are omitted. -/
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

/-- The full fused recurrent retention forward surface — the `exp2` decay,
optional initial-state seed, `0..T` recurrence, per-step masked output stores,
pointer advances, and optional final-state store — lowers to the algorithm
layer. -/
theorem fused_recurrent_retention_fwd_surface_toAlgorithm_supported
    (q k v o initial_state final_state : RegionName)
    (s_qk_h s_vo_h B H T : Nat) (scale : ℝ)
    (BK BV DK DV : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (fused_recurrent_retention_fwd_surface q k v o initial_state
      final_state s_qk_h s_vo_h B H T scale BK BV DK DV USE_INITIAL_STATE
      STORE_FINAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_retention_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `fused_recurrent_retention.py`'s
`fused_recurrent_retention_bwd_kernel`.

Phase 1 rebuilds the forward state `h` (now `[BK, BV]`) and stores `dq` rows;
the `tl.debug_barrier()` fence is transcribed verbatim (an intra-program
fence — a no-op at the algorithm layer); phase 2 rebases the pointers at time
row `T−1` (reassigning the same pointer names, as the Python does) and scans
backwards with pointer `-=` decrements, carrying the gradient state `d_h`. -/
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

/-- The full fused recurrent retention backward surface — both time loops, the
`tl.debug_barrier()` fence, the phase-2 pointer rebasing at row `T−1`, and the
pointer `-=` decrements — lowers to the algorithm layer. -/
theorem fused_recurrent_retention_bwd_surface_toAlgorithm_supported
    (q k v do_ dq dk dv initial_state : RegionName)
    (s_qk_h s_vo_h B H T : Nat) (scale : ℝ)
    (BK BV DK DV : Nat)
    (USE_INITIAL_STATE : Bool) :
    ∃ alg, (fused_recurrent_retention_bwd_surface q k v do_ dq dk dv
      initial_state s_qk_h s_vo_h B H T scale BK BV DK DV
      USE_INITIAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_retention_bwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Index / offset helpers

`s.pids 0 = i_v`, `s.pids 1 = i_k`, `s.pids 2 = i_bh`. The forward state tile
is `[BV, BK]` (`idx.1 = j_v`, `idx.2.1 = j_k`); the backward tiles are
`[BK, BV]` (`idx.1 = j_k`, `idx.2.1 = j_v`). Both share the flat
`initial_state`/`final_state` layout `i_bh·DK·DV + (i_k·BK + j_k)·DV +
(i_v·BV + j_v)`. -/

def kIdx (s : BlockState) (BK : Nat) (jk : Fin BK) : Nat :=
  s.pids 1 * BK + jk.val

def vIdx (s : BlockState) (BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 0 * BV + jv.val

/-- Flattened state / `initial_state` / `final_state` address
(`i_bh·DK·DV + (i_k·BK + j_k)·DV + (i_v·BV + j_v)`). -/
def stateOffset (s : BlockState) (DK DV BK BV : Nat)
    (jk : Fin BK) (jv : Fin BV) : Nat :=
  s.pids 2 * DK * DV + kIdx s BK jk * DV + vIdx s BV jv

def activeK (s : BlockState) (DK BK : Nat) (jk : Fin BK) : Prop :=
  kIdx s BK jk < DK

instance activeKDecidable (s : BlockState) (DK BK : Nat) (jk : Fin BK) :
    Decidable (activeK s DK BK jk) := by
  unfold activeK; infer_instance

def activeV (s : BlockState) (DV BV : Nat) (jv : Fin BV) : Prop :=
  vIdx s BV jv < DV

instance activeVDecidable (s : BlockState) (DV BV : Nat) (jv : Fin BV) :
    Decidable (activeV s DV BV jv) := by
  unfold activeV; infer_instance

/-- The kernel's per-time output-row address for the `v`-layout outputs `o`
(forward) and `dv` (backward): `(i_bh + i_k·B·H)·s_vo_h + i_v·BV + j_v + t·DV`. -/
def outStepOffset (s : BlockState) (t s_vo_h B H DV BV : Nat) (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV + jv.val + t * DV

/-- The kernel's per-time output-row address for the `k`-layout gradient
outputs `dq` and `dk`: `(i_bh + i_v·B·H)·s_qk_h + i_k·BK + j_k + t·DK`. -/
def dqStepOffset (s : BlockState) (t s_qk_h B H DK BK : Nat) (jk : Fin BK) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK + jk.val + t * DK

/-! ## Genuine closed-form data (over the *input* regions)

`kValR`/`vValR` read the kernel's exact per-time block-pointer layouts; the
scalar per-head decay is `bbVal = 1 − 2^(−5 − i_bh % H)` in the kernel's
`exp2` semantics. -/

/-- Element `R[i_bh][t, j_k]` of the shared `[T, DK]` **k-layout** at time row
`t`, key lane `j_k` (offset `i_bh·s_qk_h + i_k·BK + j_k + t·DK`). The `q` and
`k` block pointers both use this layout. -/
noncomputable def kValR (s : BlockState) (k : RegionName)
    (s_qk_h DK BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * DK)

/-- Element `R[i_bh][t, j_v]` of the shared `[T, DV]` **v-layout** at time row
`t`, value lane `j_v` (offset `i_bh·s_vo_h + i_v·BV + j_v + t·DV`). The `v`
and `do` block pointers both use this layout. -/
noncomputable def vValR (s : BlockState) (v : RegionName)
    (s_vo_h DV BV : Nat) (t : Nat) (jv : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * DV)

/-- Per-head retention decay `b_b = 1 − 2^(−5 − i_h)` with `i_h = i_bh % H`,
in the kernel's `exp2` semantics (`tl.math.exp2 x = Real.exp (x · log 2)`). -/
noncomputable def bbVal (s : BlockState) (H : Nat) : ℝ :=
  1 - Real.exp ((-5 - ((s.pids 2 % H : Nat) : ℝ)) * Real.log 2)

/-- Seeded initial state `h^(0)`: `initial_state` if `USE_INITIAL_STATE`
else `0`. -/
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (DK DV BK BV : Nat)
    (jk : Fin BK) (jv : Fin BV) : ℝ :=
  if USE_INITIAL_STATE then s.readMem h0 (stateOffset s DK DV BK BV jk jv) else 0

/-- **Genuine closed form for the retention state after `m` steps** (key lane
`j_k`, value lane `j_v`):
`seed · b_b^m + Σ_{t<m} k_t[j_k]·v_t[j_v] · b_b^(m−1−t)`. A standalone spec
over the input regions `k, v, initial_state` — never a read-back of the
kernel's own output. Shared by the forward loop and the backward phase-1 loop
(same recurrence, same flat state layout).

(Note: no docstring line here may begin at column 0 with a declaration keyword
such as `theorem` or `specification` — `scripts/spec_sheet.py` matches those at
column 0 and would report a phantom headline. Same hazard class as the "bench
comment lines must not start with `import`" rule.) -/
noncomputable def stateClosed
    (s : BlockState) (k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (m : Nat)
    (jk : Fin BK) (jv : Fin BV) : ℝ :=
  stateSeed s h0 USE_INITIAL_STATE DK DV BK BV jk jv * bbVal s H ^ m +
    ∑ t ∈ Finset.range m,
      kValR s k s_qk_h DK BK t jk * vValR s v s_vo_h DV BV t jv *
        bbVal s H ^ (m - 1 - t)

/-- **The state carry-fold recurrence.** Unrolling one step:
`h^(m+1) = b_b · h^(m) + k_m ⊗ v_m` — the exact closed-form counterpart of the
Python loop body `h = b_b * h + _k[None, :] * _v[:, None]`. -/
theorem stateClosed_succ
    (s : BlockState) (k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV m : Nat) (jk : Fin BK) (jv : Fin BV) :
    stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV (m + 1) jk jv
      = bbVal s H *
          stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV m jk jv
        + kValR s k s_qk_h DK BK m jk * vValR s v s_vo_h DV BV m jv := by
  unfold stateClosed
  rw [Finset.sum_range_succ,
    show m + 1 - 1 - m = 0 by omega, pow_zero, mul_one]
  have hsum :
      (∑ t ∈ Finset.range m,
          kValR s k s_qk_h DK BK t jk * vValR s v s_vo_h DV BV t jv *
            bbVal s H ^ (m + 1 - 1 - t))
        = bbVal s H *
            ∑ t ∈ Finset.range m,
              kValR s k s_qk_h DK BK t jk * vValR s v s_vo_h DV BV t jv *
                bbVal s H ^ (m - 1 - t) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun t ht => ?_
    simp only [Finset.mem_range] at ht
    rw [show m + 1 - 1 - t = (m - 1 - t) + 1 by omega, pow_succ]
    ring
  rw [hsum, pow_succ]
  ring

/-- **The state recurrence's base case.** At `m = 0` the closed form collapses to
the seed: `h^(0) = seed` (`b_b^0 = 1` and the accumulation sum is empty). This is
the closed-form counterpart of the Python prologue
`h = tl.zeros(...)` followed by the optional `h += tl.load(p_init_s, ...)`
(`fused_recurrent_retention.py:30–36`), and it is what
`fused_recurrent_retention_seed_slice_realizes_stateClosed_zero` below anchors to
the actual seed load. -/
theorem stateClosed_zero
    (s : BlockState) (k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (jk : Fin BK) (jv : Fin BV) :
    stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV 0 jk jv
      = stateSeed s h0 USE_INITIAL_STATE DK DV BK BV jk jv := by
  simp [stateClosed]

/-- **Genuine closed form for forward output row `m`, lane `j_v`** — the
reduction over key lanes of the **post-update** state:
`o_m[j_v] = Σ_{j_k} h^(m+1)[j_k,j_v] · scale·q_m[j_k]`. Over the input regions
`q, k, v, initial_state` only. -/
noncomputable def outClosed
    (s : BlockState) (q k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (m : Nat)
    (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
      (m + 1) jk jv *
    (kValR s q s_qk_h DK BK m jk * scale)

/-- Readability display of `outClosed` without an initial state: the classic
retention form `o_t[j_v] = Σ_{u ≤ t} b_b^(t−u) · (Σ_{j_k} scale·q_t[j_k]·
k_u[j_k]) · v_u[j_v]` — the decayed per-block dot products of the current
query row against every past key row, weighting the past value rows. -/
theorem outClosed_as_decayed_dots
    (s : BlockState) (q k v h0 : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (m : Nat) (jv : Fin BV) :
    outClosed s q k v h0 Bool.false s_qk_h s_vo_h H DK DV BK BV scale m jv
      = ∑ u ∈ Finset.range (m + 1),
          bbVal s H ^ (m - u) *
            (∑ jk : Fin BK,
              scale * kValR s q s_qk_h DK BK m jk * kValR s k s_qk_h DK BK u jk) *
            vValR s v s_vo_h DV BV u jv := by
  unfold outClosed stateClosed stateSeed
  simp only [Bool.false_eq_true, if_false, zero_mul, zero_add, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun u hu => ?_
  simp only [Finset.mem_range] at hu
  rw [Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun jk _ => ?_
  rw [show m + 1 - 1 - u = m - u by omega]
  ring

/-- **Genuine closed form for the backward `dq` row `m`, lane `j_k`** — the
phase-1 reduction over value lanes of the **post-update** rebuilt state:
`dq_m[j_k] = (Σ_{j_v} h^(m+1)[j_k,j_v] · do_m[j_v]) · scale`. -/
noncomputable def dqClosed
    (s : BlockState) (k v do_ h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (m : Nat)
    (jk : Fin BK) : ℝ :=
  (∑ jv : Fin BV,
    stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
      (m + 1) jk jv *
    vValR s do_ s_vo_h DV BV m jv) * scale

/-- **Genuine closed form for the reverse-phase gradient state at time `t`**
(the value of `d_h` used by the `dk`/`dv` writebacks at row `t`):
`d_h^(t)[j_k,j_v] = Σ_{t ≤ u < T} scale·q_u[j_k] · do_u[j_v] · b_b^(u−t)`.
Over the input regions `q, do` only. -/
noncomputable def dStateClosed
    (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat)
    (jk : Fin BK) (jv : Fin BV) : ℝ :=
  ∑ u ∈ Finset.Ico t T,
    kValR s q s_qk_h DK BK u jk * scale * vValR s do_ s_vo_h DV BV u jv *
      bbVal s H ^ (u - t)

/-- **The reverse gradient-state recurrence.** For `t < T`:
`d_h^(t) = scale·q_t ⊗ do_t + b_b · d_h^(t+1)` — the closed-form counterpart
of the Python reverse loop body `d_h += _q[:, None] * _do[None, :]` (entering
with the previous iteration's `d_h *= b_b` applied). -/
theorem dStateClosed_recurrence
    (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat) (ht : t < T)
    (jk : Fin BK) (jv : Fin BV) :
    dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale t jk jv
      = kValR s q s_qk_h DK BK t jk * scale * vValR s do_ s_vo_h DV BV t jv
        + bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (t + 1) jk jv := by
  unfold dStateClosed
  rw [Finset.sum_eq_sum_Ico_succ_bot ht, Nat.sub_self, pow_zero, mul_one]
  have hsum :
      (∑ u ∈ Finset.Ico (t + 1) T,
          kValR s q s_qk_h DK BK u jk * scale * vValR s do_ s_vo_h DV BV u jv *
            bbVal s H ^ (u - t))
        = bbVal s H *
            ∑ u ∈ Finset.Ico (t + 1) T,
              kValR s q s_qk_h DK BK u jk * scale * vValR s do_ s_vo_h DV BV u jv *
                bbVal s H ^ (u - (t + 1)) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun u hu => ?_
    simp only [Finset.mem_Ico] at hu
    rw [show u - t = (u - (t + 1)) + 1 by omega, pow_succ]
    ring
  rw [hsum]

/-- **The reverse recurrence's base case.** At (or past) the top of the reverse
scan the gradient state is zero: the index set `Ico t T` is empty for `T ≤ t`.
This is the closed-form counterpart of the Python prologue
`d_h = tl.zeros([BK, BV], ...)` (`fused_recurrent_retention.py:116`), entered
before the reverse loop over `T-1, …, 0`. -/
theorem dStateClosed_of_top_le
    (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat) (ht : T ≤ t)
    (jk : Fin BK) (jv : Fin BV) :
    dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale t jk jv = 0 := by
  simp [dStateClosed, Finset.Ico_eq_empty_of_le ht]

/-- **The reverse carry invariant's base case**, in exactly the shape the step
faces assume it (`DHPrev = b_b · dStateClosed (m+1)` at `m + 1 = T`):
`b_b · d_h^(T) = 0`. Restores as a theorem the claim the mechanical repair pass
deleted from the module docstring. -/
theorem bb_mul_dStateClosed_top
    (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ)
    (jk : Fin BK) (jv : Fin BV) :
    bbVal s H *
        dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale T jk jv = 0 := by
  rw [dStateClosed_of_top_le s q do_ s_qk_h s_vo_h H DK DV BK BV T scale T
    le_rfl jk jv, mul_zero]

/-- **Genuine closed form for the backward `dk` row `t`, lane `j_k`**:
`dk_t[j_k] = Σ_{j_v} d_h^(t)[j_k,j_v] · v_t[j_v]`. -/
noncomputable def dkClosed
    (s : BlockState) (q do_ v : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat)
    (jk : Fin BK) : ℝ :=
  ∑ jv : Fin BV,
    dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale t jk jv *
      vValR s v s_vo_h DV BV t jv

/-- **Genuine closed form for the backward `dv` row `t`, lane `j_v`**:
`dv_t[j_v] = Σ_{j_k} d_h^(t)[j_k,j_v] · k_t[j_k]`. -/
noncomputable def dvClosed
    (s : BlockState) (q do_ k : RegionName)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (t : Nat)
    (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale t jk jv *
      kValR s k s_qk_h DK BK t jk

/-! ## Seed slice (the `USE_INITIAL_STATE` prologue)

Models `fused_recurrent_retention.py:30–36` — `h = tl.zeros([BV, BK], ...)`
followed by the constexpr-guarded `h += tl.load(p_init_s, mask=mask_kv,
other=0)` — verbatim, including the `mask_kv = mask_bk[None, :] &
mask_bv[:, None]` lane predicate and the flat `initial_state` addressing
`i_bh·DK·DV + (i_k·BK + j_k)·DV + (i_v·BV + j_v)`. Both settings of the
constexpr flag are covered.

The Python `h` is a register, never stored by the prologue; the slice
materializes it into a scratch region `HSeed` at the *same* flat state layout the
`STORE_FINAL_STATE` writeback uses, under the *same* `mask_kv` predicate. `HSeed`
is therefore a fiction region — the face is a statement about the seeded carry
register, not about a Python tensor. What it buys is the base case of the carry
invariant: composed with `stateClosed_zero` it says the register the loop enters
with is exactly `stateClosed 0`. -/

/-- The `mask_kv` lane predicate of the seed load / final-state store, on the
canonical `[BV, BK]` state tile (`idx.1 = j_v`, `idx.2.1 = j_k`). -/
def activeKV (s : BlockState) (DK DV BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Prop :=
  activeK s DK BK idx.2.1 ∧ activeV s DV BV idx.1

instance activeKVDecidable (s : BlockState) (DK DV BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Decidable (activeKV s DK DV BK BV idx) := by
  unfold activeKV; infer_instance

def fused_recurrent_retention_seed_slice
    (initial_state HSeed : RegionName) (DK DV BK BV : Nat)
    (USE_INITIAL_STATE : Bool) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
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
  tl.store(HSeed + i_bh * $(DK) * $(DV) +
    (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
    (i_v * $(BV) + tl.arange(0, $(BV))[:, None]),
    (h).to(HSeed.dtype.element_ty), mask=mask_kv)
}

/-- The seed slice lowers to the algorithm layer for either flag setting. -/
theorem fused_recurrent_retention_seed_slice_toAlgorithm_supported
    (initial_state HSeed : RegionName) (DK DV BK BV : Nat)
    (USE_INITIAL_STATE : Bool) :
    ∃ alg, (fused_recurrent_retention_seed_slice initial_state HSeed DK DV BK BV
      USE_INITIAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_retention_seed_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

theorem fused_recurrent_retention_seed_slice_correct
    (initial_state HSeed : RegionName) (DK DV BK BV : Nat)
    (USE_INITIAL_STATE : Bool) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1)) :
    ∀ idx : TileIndex [BV, BK],
      let outAddr := stateOffset s DK DV BK BV idx.2.1 idx.1
      (exec (fused_recurrent_retention_seed_slice initial_state HSeed DK DV BK BV
            USE_INITIAL_STATE) s).map (·.readMem HSeed outAddr)
        = some (if activeKV s DK DV BK BV idx then
            stateSeed s initial_state USE_INITIAL_STATE DK DV BK BV idx.2.1 idx.1
          else s.readMem HSeed outAddr) := by
  intro idx
  have hOffsetInj :
      Function.Injective (fun idx : TileIndex [BV, BK] =>
        s.pids 2 * DK * DV + (s.pids 1 * BK + idx.2.1.val) * DV +
          (s.pids 0 * BV + idx.1.val)) := by
    simpa [stateOffset, kIdx, vIdx] using hOutInj
  cases USE_INITIAL_STATE
  -- `USE_INITIAL_STATE = false`
  · -- `h` stays `tl.zeros`, so the seeded state is `0` on every kept lane.
    simp [exec, fused_recurrent_retention_seed_slice, stepStmts, stepStmt,
      evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
      Tile.expandDim, Tile.ptrAdd, Tile.remap, NumericDType.add,
      NumericDType.mul, ComparableDType.lt, stateOffset, kIdx, vIdx,
      activeKV, activeK, activeV, stateSeed, TileShape.dropInsertedIndex,
      ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _
      (fun idx : TileIndex [BV, BK] =>
        s.pids 1 * BK + idx.2.1.val < DK ∧ s.pids 0 * BV + idx.1.val < DV)
      hOffsetInj idx]
    by_cases hAct : s.pids 1 * BK + idx.2.1.val < DK ∧
        s.pids 0 * BV + idx.1.val < DV
    · simp [hAct]
    · simp [hAct]
  -- `USE_INITIAL_STATE = true`
  · -- `h = 0 + initial_state[...]` on kept lanes.
    simp [exec, fused_recurrent_retention_seed_slice, stepStmts, stepStmt,
      evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
      Tile.expandDim, Tile.ptrAdd, Tile.remap, NumericDType.add,
      NumericDType.mul, ComparableDType.lt, stateOffset, kIdx, vIdx,
      activeKV, activeK, activeV, stateSeed, TileShape.dropInsertedIndex,
      ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _
      (fun idx : TileIndex [BV, BK] =>
        s.pids 1 * BK + idx.2.1.val < DK ∧ s.pids 0 * BV + idx.1.val < DV)
      hOffsetInj idx]
    by_cases hAct : s.pids 1 * BK + idx.2.1.val < DK ∧
        s.pids 0 * BV + idx.1.val < DV
    · simp [hAct]
    · simp [hAct]

theorem fused_recurrent_retention_seed_slice_compute_correct
    (initial_state HSeed : RegionName) (DK DV BK BV : Nat)
    (USE_INITIAL_STATE : Bool) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_seed_slice initial_state HSeed
        DK DV BK BV USE_INITIAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => activeKV s DK DV BK BV idx)
        (fun idx : TileIndex [BV, BK] =>
          (HSeed, stateOffset s DK DV BK BV idx.2.1 idx.1)))
      (expected := fun idx : TileIndex [BV, BK] =>
        stateSeed s initial_state USE_INITIAL_STATE DK DV BK BV
          idx.2.1 idx.1) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_retention_seed_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := fused_recurrent_retention_seed_slice_correct initial_state HSeed
    DK DV BK BV USE_INITIAL_STATE s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **The carry invariant's base case, on the kernel's own seed load.** The
`USE_INITIAL_STATE` prologue writes exactly `stateClosed 0` — the `m = 0`
instance of the very closed form the step faces carry forward. Together with
`stateStepSpec_eq_stateClosed_succ` this restores as a theorem the "from the seed
`stateClosed(0) = stateSeed`" claim the repair pass deleted from the docstring.

It is still **not** a proof of the cross-step fold: this face writes `HSeed`
while the step face's `hPrev` constrains `HPrev`, so the chaining from step `m`
to step `m+1` remains unmodeled (see the module docstring). -/
theorem fused_recurrent_retention_seed_slice_realizes_stateClosed_zero
    (initial_state HSeed k v : RegionName) (DK DV BK BV : Nat)
    (USE_INITIAL_STATE : Bool) (s_qk_h s_vo_h H : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_seed_slice initial_state HSeed
        DK DV BK BV USE_INITIAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => activeKV s DK DV BK BV idx)
        (fun idx : TileIndex [BV, BK] =>
          (HSeed, stateOffset s DK DV BK BV idx.2.1 idx.1)))
      (expected := fun idx : TileIndex [BV, BK] =>
        stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
          H DK DV BK BV 0 idx.2.1 idx.1) := by
  have hfun :
      (fun idx : TileIndex [BV, BK] =>
          stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
            H DK DV BK BV 0 idx.2.1 idx.1)
        = fun idx : TileIndex [BV, BK] =>
            stateSeed s initial_state USE_INITIAL_STATE DK DV BK BV
              idx.2.1 idx.1 := by
    funext idx
    exact stateClosed_zero s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
      H DK DV BK BV idx.2.1 idx.1
  rw [hfun]
  exact fused_recurrent_retention_seed_slice_compute_correct initial_state
    HSeed DK DV BK BV USE_INITIAL_STATE s hOutInj

/-! ## State-update step slice (the shared forward / backward-phase-1 carry-fold body)

Isolates one loop body's state update from the cross-step loop induction: it
loads the materialized previous-state tile `HPrev` (flat state layout), the
time-`t` rows `k_t`/`v_t`, recomputes the per-head decay `b_b`, and stores
`b_b·h + k_t ⊗ v_t` into a state buffer `HOut` at the same layout. The forward
loop (`[BV,BK]` axes) and the backward phase-1 loop (`[BK,BV]` axes) carry the
*same* recurrence at the *same* flat addresses; this canonical `[BV,BK]` slice
realizes that shared carry-fold step. -/

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

/-- The arithmetic spec of one state-update body: `b_b·prev + k_t ⊗ v_t`. -/
noncomputable def stateStepSpec
    (s : BlockState) (HPrev k v : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  bbVal s H * s.readMem HPrev (stateOffset s DK DV BK BV idx.2.1 idx.1)
    + kValR s k s_qk_h DK BK t idx.2.1 * vValR s v s_vo_h DV BV t idx.1

theorem fused_recurrent_retention_state_step_slice_correct
    (HPrev k v HOut : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1)) :
    ∀ idx : TileIndex [BV, BK],
      let outAddr := stateOffset s DK DV BK BV idx.2.1 idx.1
      (exec (fused_recurrent_retention_state_step_slice HPrev k v HOut
            t s_qk_h s_vo_h H DK DV BK BV) s).map (·.readMem HOut outAddr)
        = some (stateStepSpec s HPrev k v t s_qk_h s_vo_h H DK DV BK BV idx) := by
  intro idx
  simp [exec, fused_recurrent_retention_state_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.sub, WithBot.realExp2,
        stateOffset, kIdx, vIdx, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV, BK] → Nat :=
    fun idx => s.pids 2 * DK * DV +
      (s.pids 1 * BK + idx.2.1.val) * DV + (s.pids 0 * BV + idx.1.val)
  let valueFn : TileIndex [BV, BK] → ℝ :=
    fun idx =>
      (1 - Real.exp (((0.0 : ℝ) - 5 - ((s.pids 2 % H : ℕ) : ℝ) * 1.0) * Real.log 2)) *
        s.readMem HPrev
          (s.pids 2 * DK * DV + (s.pids 1 * BK + idx.2.1.val) * DV +
            (s.pids 0 * BV + idx.1.val))
      + s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.2.1.val + t * DK) *
          s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * DV)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, stateOffset, kIdx, vIdx] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem HOut (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BV, BK])).readMem HOut (offsetFn idx) =
    stateStepSpec s HPrev k v t s_qk_h s_vo_h H DK DV BK BV idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, stateStepSpec, stateOffset, bbVal, kValR, vValR, kIdx, vIdx]
  norm_num

theorem fused_recurrent_retention_state_step_slice_compute_correct
    (HPrev k v HOut : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_state_step_slice HPrev k v HOut
        t s_qk_h s_vo_h H DK DV BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s DK DV BK BV idx.2.1 idx.1))
      (expected := fun idx =>
        stateStepSpec s HPrev k v t s_qk_h s_vo_h H DK DV BK BV idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_retention_state_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := fused_recurrent_retention_state_step_slice_correct HPrev k v HOut
    t s_qk_h s_vo_h H DK DV BK BV s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- **State carry-fold step (genuine).** If the materialized previous-state
buffer `HPrev` holds the genuine `m`-step folded state `stateClosed(m)`, then
one loop body — `stateStepSpec`, i.e. `b_b·HPrev + k_m ⊗ v_m` — produces
exactly the genuine `(m+1)`-step folded state `stateClosed(m+1)`. -/
theorem stateStepSpec_eq_stateClosed_succ
    (s : BlockState) (HPrev k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h H DK DV BK BV : Nat)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV m jk jv)
    (idx : TileIndex [BV, BK]) :
    stateStepSpec s HPrev k v m s_qk_h s_vo_h H DK DV BK BV idx
      = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          (m + 1) idx.2.1 idx.1 := by
  rw [stateClosed_succ]
  unfold stateStepSpec
  rw [hPrev idx.2.1 idx.1]

/-! ## Cross-step carry fold — the state loop, threaded through memory

The docstring above says the induction "is missing only its step-chaining": both
base cases are theorems, but the step face constrains `HPrev` while concluding
about `HOut`, with nothing relating them. This section supplies exactly that
missing piece, by running the state slices as a `CarryFold.execChain` whose
carry-in and carry-out region are literally the same name `C`.

Because this slice's store is **unmasked**, the fold runs at the full tile index
— no write-active subtype is needed, unlike `fused_rwkv6_kernel` and
`fused_recurrent_delta`.

Honesty limits, the same ones `VeriTile.Triton.CarryFold` states: the carry
travels through a **memory region**, one launch per step, while Python keeps `h`
in a *register* across the loop — that object is still not modeled here. `hSeed`
is an assumption about the caller's buffer; what `stateClosed_zero` contributes
is that the value it must hold is exactly `stateSeed`. -/

/-- One state step leaves everything outside the carry region alone: it writes
only `HOut`, and an (unmasked) `writeMem` scatter touches neither the program
ids nor any other region. -/
theorem fused_recurrent_retention_state_step_frame
    (C k v : RegionName) (t s_qk_h s_vo_h H DK DV BK BV : Nat)
    (u u' : BlockState)
    (hExec : exec (fused_recurrent_retention_state_step_slice C k v C
      t s_qk_h s_vo_h H DK DV BK BV) u = some u') :
    u'.pids = u.pids ∧ ∀ r, r ≠ C → ∀ o, u'.mem r o = u.mem r o := by
  simp [exec, fused_recurrent_retention_state_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.sub, WithBot.realExp2,
        stateOffset, kIdx, vIdx, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  refine ⟨?_, ?_⟩
  · rw [BlockState.foldl_writeMem_pids]
    simp
  · intro r hr o
    rw [BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o]
    simp

/-- Address congruence: `stateOffset` reads the state only through `pids`. -/
theorem stateOffset_congr (s u : BlockState) (DK DV BK BV : Nat)
    (h : u.pids = s.pids) (jk : Fin BK) (jv : Fin BV) :
    stateOffset u DK DV BK BV jk jv = stateOffset s DK DV BK BV jk jv := by
  unfold stateOffset kIdx vIdx; rw [h]

/-- `stateStepSpec` transported to a state `u` agreeing with the reference state
`s` on `pids` and on the two **input** regions it reads. The carry region `C` is
deliberately *not* transported: its content at `u` is what the fold's invariant
supplies. This is what forces the `k`/`v ≠ C` side conditions below. The decay
`b_b` needs no hypothesis — it is a function of `pids` alone. -/
theorem stateStepSpec_transport
    (s u : BlockState) (C k v : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat)
    (hpids : u.pids = s.pids)
    (hk : ∀ o, u.readMem k o = s.readMem k o)
    (hv : ∀ o, u.readMem v o = s.readMem v o)
    (idx : TileIndex [BV, BK]) :
    stateStepSpec u C k v t s_qk_h s_vo_h H DK DV BK BV idx
      = bbVal s H * u.readMem C (stateOffset s DK DV BK BV idx.2.1 idx.1)
        + kValR s k s_qk_h DK BK t idx.2.1 * vValR s v s_vo_h DV BV t idx.1 := by
  simp only [stateStepSpec, bbVal, kValR, vValR, stateOffset, kIdx, vIdx,
    hpids, hk, hv]

/-- **Cross-step carry fold for the retention state recurrence (genuine).**

Running `T` state slices as a chain through the shared carry region `C`, the
final buffer holds the genuine closed form `stateClosed(T)` on **every** lane,
and the chain leaves everything outside `C` untouched.

This is the step-chaining the module docstring records as missing: the step face
constrains `HPrev` and concludes about `HOut` with nothing relating them, so a
`T`-step story needed `T` assumptions. Here there is **one**, `hSeed`, and the
identification is structural — the same region name is both the slice's `HPrev`
and its `HOut`.

Hypotheses, all necessary:

* `hkC` / `hvC` — the two input regions are distinct from the carry region,
  forced by `stateStepSpec_transport`;
* `hInj` — state-address injectivity, as in the step face;
* `hSeed` — the initial buffer holds the seed. `stateClosed_zero` is what
  identifies that value as `stateSeed`, so this is a statement about the
  caller's buffer and nothing more;
* `hRun` — the chain runs to completion (postcondition style).

This says nothing about the launched surface, which keeps `h` in a register
across its own loop. -/
theorem fused_recurrent_retention_state_carry_fold
    (C k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h H DK DV BK BV T : Nat) (s sFinal : BlockState)
    (hkC : k ≠ C) (hvC : v ≠ C)
    (hInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1))
    (hSeed : ∀ idx : TileIndex [BV, BK],
      s.readMem C (stateOffset s DK DV BK BV idx.2.1 idx.1)
        = stateSeed s h0 USE_INITIAL_STATE DK DV BK BV idx.2.1 idx.1)
    (hRun : execChain (foldStages
        (fun j => fused_recurrent_retention_state_step_slice C k v C
          j s_qk_h s_vo_h H DK DV BK BV) T) s = some sFinal) :
    AgreeOutsideRegion C s sFinal ∧
    ∀ idx : TileIndex [BV, BK],
      sFinal.readMem C (stateOffset s DK DV BK BV idx.2.1 idx.1)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
            T idx.2.1 idx.1 := by
  have key := carryFold_execChain
    (ι := TileIndex [BV, BK])
    (step := fun j => fused_recurrent_retention_state_step_slice C k v C
      j s_qk_h s_vo_h H DK DV BK BV)
    (C := C)
    (addr := fun idx => stateOffset s DK DV BK BV idx.2.1 idx.1)
    (val := fun j idx =>
      stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
        j idx.2.1 idx.1)
    (n := T) (s := s) (sFinal := sFinal)
    (fun idx => by simpa [stateClosed] using hSeed idx)
    (fun j _hj t t' hAgree hInv hExec => by
      have hpidsR : t.resetRegs.pids = s.pids := by
        rw [BlockState.resetRegs_pids]; exact hAgree.pids
      obtain ⟨hp', hm'⟩ := fused_recurrent_retention_state_step_frame C k v
        j s_qk_h s_vo_h H DK DV BK BV t.resetRegs t' hExec
      have hAgree' : AgreeOutsideRegion C s t' :=
        ⟨by rw [hp', BlockState.resetRegs_pids]; exact hAgree.pids,
         by
           intro r hr o
           rw [hm' r hr o, BlockState.resetRegs_mem]
           exact hAgree.mem r hr o⟩
      refine ⟨hAgree', ?_⟩
      intro idx
      show t'.readMem C (stateOffset s DK DV BK BV idx.2.1 idx.1)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
            (j + 1) idx.2.1 idx.1
      have hInjT : Function.Injective
          (fun i : TileIndex [BV, BK] =>
            stateOffset t.resetRegs DK DV BK BV i.2.1 i.1) := by
        intro a b hab
        exact hInj (by simpa [stateOffset_congr s t.resetRegs DK DV BK BV hpidsR] using hab)
      have h := fused_recurrent_retention_state_step_slice_correct C k v C
        j s_qk_h s_vo_h H DK DV BK BV t.resetRegs hInjT idx
      rw [hExec] at h
      have hcorr := Option.some.inj h
      simp only [stateOffset_congr s t.resetRegs DK DV BK BV hpidsR] at hcorr
      rw [hcorr, stateStepSpec_transport s t.resetRegs C k v
        j s_qk_h s_vo_h H DK DV BK BV hpidsR
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem k hkC o)
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem v hvC o)
        idx, BlockState.resetRegs_readMem, hInv idx, stateClosed_succ])
    hRun
  exact ⟨key.1, key.2⟩

/-! ## Output-reduction step slice (the forward per-step `o_t`)

Isolates one forward loop body's output computation: with the materialized
*pre-update* state tile `HPrev`, it recomputes `b_b`, the post-update state
`h = b_b·prev + k_t ⊗ v_t`, forms `h · (scale·q_t)`, reduces over the key axis
(`tl.sum(_, axis=1)`), and masked-stores the `[BV]` row into `o` at time row
`t`. -/

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

/-- The genuine per-lane output spec of one forward loop body, reading the
materialized *pre-update* state `HPrev`:
`o_t[j_v] = Σ_{j_k} (b_b·HPrev[j_k,j_v] + k_t[j_k]·v_t[j_v]) · scale·q_t[j_k]`. -/
noncomputable def outputStepSpec
    (s : BlockState) (HPrev q k v : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    (bbVal s H * s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
      + kValR s k s_qk_h DK BK t jk * vValR s v s_vo_h DV BV t jv)
    * (kValR s q s_qk_h DK BK t jk * scale)

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_retention_output_step_slice_correct
    (HPrev q k v o : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s t s_vo_h B H DV BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := outStepOffset s t s_vo_h B H DV BV jv
      (exec (fused_recurrent_retention_output_step_slice HPrev q k v o
            t s_qk_h s_vo_h B H DK DV BK BV scale) s).map (·.readMem o outAddr)
        = some (if activeV s DV BV jv then
            outputStepSpec s HPrev q k v t s_qk_h s_vo_h H DK DV BK BV scale jv
          else s.readMem o outAddr) := by
  intro jv
  simp [exec, fused_recurrent_retention_output_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        NumericDType.sub, WithBot.realExp2,
        ComparableDType.lt, stateOffset, outStepOffset, activeV, vIdx, kIdx,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV + idx.1.val + t * DV
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      ∑ jk : Fin BK,
        ((1 - Real.exp (((0.0 : ℝ) - 5 - ((s.pids 2 % H : ℕ) : ℝ) * 1.0) * Real.log 2)) *
            s.readMem HPrev
              (s.pids 2 * DK * DV + (s.pids 1 * BK + jk.val) * DV + (s.pids 0 * BV + idx.1.val))
          + s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * DK) *
              s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * DV)) *
        (s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * DK) * scale)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < DV
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : outStepOffset s t s_vo_h B H DV BV a = outStepOffset s t s_vo_h B H DV BV b := by
      simpa [offsetFn, outStepOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem o (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem o (offsetFn (jv, PUnit.unit)) =
    if activeV s DV BV jv then
      outputStepSpec s HPrev q k v t s_qk_h s_vo_h H DK DV BK BV scale jv
    else s.readMem o (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < DV
  · simp only [P, hjv, if_true, activeV, vIdx]
    simp only [valueFn, outputStepSpec, stateOffset, kValR, vValR, bbVal, kIdx, vIdx]
    apply Finset.sum_congr rfl
    intro jk _
    norm_num
  · simp only [P, activeV, vIdx, hjv, if_false, BlockState.setReg_readMem]

theorem fused_recurrent_retention_output_step_slice_compute_correct
    (HPrev q k v o : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s t s_vo_h B H DV BV jv)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_output_step_slice HPrev q k v o
        t s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s DV BV jv)
        (fun jv => (o, outStepOffset s t s_vo_h B H DV BV jv)))
      (expected := fun jv : Fin BV =>
        outputStepSpec s HPrev q k v t s_qk_h s_vo_h H DK DV BK BV scale jv) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_retention_output_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jv hActive
  have h := fused_recurrent_retention_output_step_slice_correct HPrev q k v o
    t s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj jv
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Output step realizes the genuine closed form.** When the materialized
pre-update state buffer `HPrev` holds `stateClosed(m)`, the per-lane output
reduction `outputStepSpec` equals the genuine post-update output
`outClosed(m)` over the input regions (via `stateClosed_succ`). -/
theorem outputStepSpec_eq_outClosed
    (s : BlockState) (HPrev q k v h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV m jk jv)
    (jv : Fin BV) :
    outputStepSpec s HPrev q k v m s_qk_h s_vo_h H DK DV BK BV scale jv
      = outClosed s q k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          scale m jv := by
  unfold outputStepSpec outClosed
  refine Finset.sum_congr rfl fun jk _ => ?_
  rw [stateClosed_succ, hPrev jk jv]

/-! ## Backward phase-1 `dq` step slice

One backward phase-1 loop body: with the materialized *pre-update* rebuilt
state `HPrev` (`[BK,BV]` axes, same flat layout), it recomputes `b_b`, the
post-update state `h = b_b·prev + k_t ⊗ v_t`, reduces `h · do_t` over the
value axis, multiplies by `scale`, and masked-stores the `[BK]` row into `dq`
at time row `t`. -/

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

/-- The genuine per-lane `dq` spec of one backward phase-1 body:
`dq_t[j_k] = (Σ_{j_v} (b_b·HPrev[j_k,j_v] + k_t[j_k]·v_t[j_v]) · do_t[j_v]) · scale`. -/
noncomputable def bwdDqStepSpec
    (s : BlockState) (HPrev k v do_ : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (jk : Fin BK) : ℝ :=
  (∑ jv : Fin BV,
    (bbVal s H * s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
      + kValR s k s_qk_h DK BK t jk * vValR s v s_vo_h DV BV t jv)
    * vValR s do_ s_vo_h DV BV t jv) * scale

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_retention_bwd_dq_step_slice_correct
    (HPrev k v do_ dq : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dqStepOffset s t s_qk_h B H DK BK jk)) :
    ∀ jk : Fin BK,
      let outAddr := dqStepOffset s t s_qk_h B H DK BK jk
      (exec (fused_recurrent_retention_bwd_dq_step_slice HPrev k v do_ dq
            t s_qk_h s_vo_h B H DK DV BK BV scale) s).map (·.readMem dq outAddr)
        = some (if activeK s DK BK jk then
            bwdDqStepSpec s HPrev k v do_ t s_qk_h s_vo_h H DK DV BK BV scale jk
          else s.readMem dq outAddr) := by
  intro jk
  simp [exec, fused_recurrent_retention_bwd_dq_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        NumericDType.sub, WithBot.realExp2,
        ComparableDType.lt, stateOffset, dqStepOffset, activeK, vIdx, kIdx,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK] → Nat :=
    fun idx => (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK + idx.1.val + t * DK
  let valueFn : TileIndex [BK] → ℝ :=
    fun idx =>
      (∑ jv : Fin BV,
        ((1 - Real.exp (((0.0 : ℝ) - 5 - ((s.pids 2 % H : ℕ) : ℝ) * 1.0) * Real.log 2)) *
            s.readMem HPrev
              (s.pids 2 * DK * DV + (s.pids 1 * BK + idx.1.val) * DV + (s.pids 0 * BV + jv.val))
          + s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.1.val + t * DK) *
              s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * DV)) *
        s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * DV)) * scale
  let P : TileIndex [BK] → Prop := fun idx => s.pids 1 * BK + idx.1.val < DK
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dqStepOffset s t s_qk_h B H DK BK a = dqStepOffset s t s_qk_h B H DK BK b := by
      simpa [offsetFn, dqStepOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dq (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BK])).readMem dq (offsetFn (jk, PUnit.unit)) =
    if activeK s DK BK jk then
      bwdDqStepSpec s HPrev k v do_ t s_qk_h s_vo_h H DK DV BK BV scale jk
    else s.readMem dq (offsetFn (jk, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jk, PUnit.unit)]
  by_cases hjk : s.pids 1 * BK + jk.val < DK
  · simp only [P, hjk, if_true, activeK, kIdx]
    simp only [valueFn, bwdDqStepSpec, stateOffset, kValR, vValR, bbVal, kIdx, vIdx]
    congr 1
    apply Finset.sum_congr rfl
    intro jv _
    norm_num
  · simp only [P, activeK, kIdx, hjk, if_false, BlockState.setReg_readMem]

theorem fused_recurrent_retention_bwd_dq_step_slice_compute_correct
    (HPrev k v do_ dq : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dqStepOffset s t s_qk_h B H DK BK jk)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dq_step_slice HPrev k v do_ dq
        t s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s DK BK jk)
        (fun jk => (dq, dqStepOffset s t s_qk_h B H DK BK jk)))
      (expected := fun jk : Fin BK =>
        bwdDqStepSpec s HPrev k v do_ t s_qk_h s_vo_h H DK DV BK BV scale jk) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_retention_bwd_dq_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jk hActive
  have h := fused_recurrent_retention_bwd_dq_step_slice_correct HPrev k v do_ dq
    t s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj jk
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **`dq` step realizes the genuine closed form.** When the materialized
pre-update state buffer `HPrev` holds `stateClosed(m)`, the `dq` reduction
`bwdDqStepSpec` equals the genuine `dqClosed(m)` (via `stateClosed_succ`). -/
theorem bwdDqStepSpec_eq_dqClosed
    (s : BlockState) (HPrev k v do_ h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV m jk jv)
    (jk : Fin BK) :
    bwdDqStepSpec s HPrev k v do_ m s_qk_h s_vo_h H DK DV BK BV scale jk
      = dqClosed s k v do_ h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          scale m jk := by
  unfold bwdDqStepSpec dqClosed
  congr 1
  refine Finset.sum_congr rfl fun jv _ => ?_
  rw [stateClosed_succ, hPrev jk jv]

/-! ## Backward phase-2 gradient-state carry step slice

One reverse loop body's `d_h` update, including the trailing `d_h *= b_b`:
with the materialized incoming carry `DHPrev` (`[BK,BV]` axes, flat state
layout), it loads `scale·q_t` and `do_t`, forms `d_h = DHPrev + q_t ⊗ do_t`,
recomputes `b_b`, and stores `d_h · b_b` — the carry handed to time row `t−1`
— into `DHOut`. -/

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

/-- The arithmetic spec of one gradient-state carry body:
`(DHPrev + scale·q_t ⊗ do_t) · b_b`. -/
noncomputable def bwdDStateStepSpec
    (s : BlockState) (DHPrev q do_ : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ)
    (idx : TileIndex [BK, BV]) : ℝ :=
  (s.readMem DHPrev (stateOffset s DK DV BK BV idx.1 idx.2.1)
    + kValR s q s_qk_h DK BK t idx.1 * scale * vValR s do_ s_vo_h DV BV t idx.2.1)
  * bbVal s H

theorem fused_recurrent_retention_bwd_dstate_step_slice_correct
    (DHPrev q do_ DHOut : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => stateOffset s DK DV BK BV idx.1 idx.2.1)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := stateOffset s DK DV BK BV idx.1 idx.2.1
      (exec (fused_recurrent_retention_bwd_dstate_step_slice DHPrev q do_ DHOut
            t s_qk_h s_vo_h H DK DV BK BV scale) s).map (·.readMem DHOut outAddr)
        = some (bwdDStateStepSpec s DHPrev q do_ t s_qk_h s_vo_h H DK DV BK BV
            scale idx) := by
  intro idx
  simp [exec, fused_recurrent_retention_bwd_dstate_step_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, NumericDType.sub, WithBot.realExp2,
        stateOffset, kIdx, vIdx, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * DK * DV +
      (s.pids 1 * BK + idx.1.val) * DV + (s.pids 0 * BV + idx.2.1.val)
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx =>
      (s.readMem DHPrev
          (s.pids 2 * DK * DV + (s.pids 1 * BK + idx.1.val) * DV +
            (s.pids 0 * BV + idx.2.1.val))
        + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.1.val + t * DK) * scale *
            s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.2.1.val + t * DV))
      * (1 - Real.exp (((0.0 : ℝ) - 5 - ((s.pids 2 % H : ℕ) : ℝ) * 1.0) * Real.log 2))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, stateOffset, kIdx, vIdx] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DHOut (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BK, BV])).readMem DHOut (offsetFn idx) =
    bwdDStateStepSpec s DHPrev q do_ t s_qk_h s_vo_h H DK DV BK BV scale idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, bwdDStateStepSpec, stateOffset, bbVal, kValR, vValR, kIdx, vIdx]
  norm_num

theorem fused_recurrent_retention_bwd_dstate_step_slice_compute_correct
    (DHPrev q do_ DHOut : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => stateOffset s DK DV BK BV idx.1 idx.2.1)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dstate_step_slice DHPrev q do_
        DHOut t s_qk_h s_vo_h H DK DV BK BV scale)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (DHOut, stateOffset s DK DV BK BV idx.1 idx.2.1))
      (expected := fun idx =>
        bwdDStateStepSpec s DHPrev q do_ t s_qk_h s_vo_h H DK DV BK BV scale idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_retention_bwd_dstate_step_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := fused_recurrent_retention_bwd_dstate_step_slice_correct DHPrev q do_
    DHOut t s_qk_h s_vo_h H DK DV BK BV scale s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- **Gradient-state carry step (genuine).** The reverse carry invariant is
`DHPrev = b_b · dStateClosed(t+1)` (the post-decay `d_h` handed down from time
row `t+1`; at the first reverse iteration `t = T−1` this is `b_b·0 = 0`, the
kernel's `tl.zeros` initialization). Under it, one carry body produces exactly
`b_b · dStateClosed(t)` — the invariant for time row `t−1` — via
`dStateClosed_recurrence`. -/
theorem bwdDStateStepSpec_eq
    (s : BlockState) (DHPrev q do_ : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (ht : t < T)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (t + 1) jk jv)
    (idx : TileIndex [BK, BV]) :
    bwdDStateStepSpec s DHPrev q do_ t s_qk_h s_vo_h H DK DV BK BV scale idx
      = bbVal s H *
          dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale t idx.1 idx.2.1 := by
  unfold bwdDStateStepSpec
  rw [hPrev idx.1 idx.2.1, dStateClosed_recurrence s q do_ s_qk_h s_vo_h
    H DK DV BK BV T scale t ht idx.1 idx.2.1]
  ring

/-! ## Backward phase-2 `dk` / `dv` step slices

One reverse loop body's writebacks: with the incoming carry `DHPrev`, form
`d_h = DHPrev + scale·q_t ⊗ do_t` (the value the kernel's `d_h += …` produces
at row `t`) and reduce against `v_t` over the value axis (`dk`, `axis=1`) /
against `k_t` over the key axis (`dv`, `axis=0`), masked-storing the rows. -/

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

/-- The per-lane `dk` spec of one reverse body:
`dk_t[j_k] = Σ_{j_v} (DHPrev[j_k,j_v] + scale·q_t[j_k]·do_t[j_v]) · v_t[j_v]`. -/
noncomputable def bwdDkStepSpec
    (s : BlockState) (DHPrev q do_ v : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (jk : Fin BK) : ℝ :=
  ∑ jv : Fin BV,
    (s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
      + kValR s q s_qk_h DK BK t jk * scale * vValR s do_ s_vo_h DV BV t jv)
    * vValR s v s_vo_h DV BV t jv

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_retention_bwd_dk_step_slice_correct
    (DHPrev q do_ v dk : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dqStepOffset s t s_qk_h B H DK BK jk)) :
    ∀ jk : Fin BK,
      let outAddr := dqStepOffset s t s_qk_h B H DK BK jk
      (exec (fused_recurrent_retention_bwd_dk_step_slice DHPrev q do_ v dk
            t s_qk_h s_vo_h B H DK DV BK BV scale) s).map (·.readMem dk outAddr)
        = some (if activeK s DK BK jk then
            bwdDkStepSpec s DHPrev q do_ v t s_qk_h s_vo_h H DK DV BK BV scale jk
          else s.readMem dk outAddr) := by
  intro jk
  simp [exec, fused_recurrent_retention_bwd_dk_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, stateOffset, dqStepOffset, activeK, vIdx, kIdx,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK] → Nat :=
    fun idx => (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK + idx.1.val + t * DK
  let valueFn : TileIndex [BK] → ℝ :=
    fun idx =>
      ∑ jv : Fin BV,
        (s.readMem DHPrev
            (s.pids 2 * DK * DV + (s.pids 1 * BK + idx.1.val) * DV + (s.pids 0 * BV + jv.val))
          + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.1.val + t * DK) * scale *
              s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * DV)) *
        s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * DV)
  let P : TileIndex [BK] → Prop := fun idx => s.pids 1 * BK + idx.1.val < DK
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dqStepOffset s t s_qk_h B H DK BK a = dqStepOffset s t s_qk_h B H DK BK b := by
      simpa [offsetFn, dqStepOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dk (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BK])).readMem dk (offsetFn (jk, PUnit.unit)) =
    if activeK s DK BK jk then
      bwdDkStepSpec s DHPrev q do_ v t s_qk_h s_vo_h H DK DV BK BV scale jk
    else s.readMem dk (offsetFn (jk, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jk, PUnit.unit)]
  by_cases hjk : s.pids 1 * BK + jk.val < DK
  · simp only [P, hjk, if_true, activeK, kIdx]
    simp only [valueFn, bwdDkStepSpec, stateOffset, kValR, vValR, kIdx, vIdx]
  · simp only [P, activeK, kIdx, hjk, if_false, BlockState.setReg_readMem]

theorem fused_recurrent_retention_bwd_dk_step_slice_compute_correct
    (DHPrev q do_ v dk : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dqStepOffset s t s_qk_h B H DK BK jk)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dk_step_slice DHPrev q do_ v dk
        t s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s DK BK jk)
        (fun jk => (dk, dqStepOffset s t s_qk_h B H DK BK jk)))
      (expected := fun jk : Fin BK =>
        bwdDkStepSpec s DHPrev q do_ v t s_qk_h s_vo_h H DK DV BK BV scale jk) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_retention_bwd_dk_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jk hActive
  have h := fused_recurrent_retention_bwd_dk_step_slice_correct DHPrev q do_ v dk
    t s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj jk
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **`dk` step realizes the genuine closed form.** Under the reverse carry
invariant `DHPrev = b_b · dStateClosed(t+1)`, the row-`t` `dk` reduction
equals the genuine `dkClosed(t)` (via `dStateClosed_recurrence`). -/
theorem bwdDkStepSpec_eq_dkClosed
    (s : BlockState) (DHPrev q do_ v : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (ht : t < T)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (t + 1) jk jv)
    (jk : Fin BK) :
    bwdDkStepSpec s DHPrev q do_ v t s_qk_h s_vo_h H DK DV BK BV scale jk
      = dkClosed s q do_ v s_qk_h s_vo_h H DK DV BK BV T scale t jk := by
  unfold bwdDkStepSpec dkClosed
  refine Finset.sum_congr rfl fun jv _ => ?_
  rw [hPrev jk jv, dStateClosed_recurrence s q do_ s_qk_h s_vo_h
    H DK DV BK BV T scale t ht jk jv]
  ring

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

/-- The per-lane `dv` spec of one reverse body:
`dv_t[j_v] = Σ_{j_k} (DHPrev[j_k,j_v] + scale·q_t[j_k]·do_t[j_v]) · k_t[j_k]`. -/
noncomputable def bwdDvStepSpec
    (s : BlockState) (DHPrev q do_ k : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV : Nat) (scale : ℝ) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    (s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
      + kValR s q s_qk_h DK BK t jk * scale * vValR s do_ s_vo_h DV BV t jv)
    * kValR s k s_qk_h DK BK t jk

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_retention_bwd_dv_step_slice_correct
    (DHPrev q do_ k dv : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s t s_vo_h B H DV BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := outStepOffset s t s_vo_h B H DV BV jv
      (exec (fused_recurrent_retention_bwd_dv_step_slice DHPrev q do_ k dv
            t s_qk_h s_vo_h B H DK DV BK BV scale) s).map (·.readMem dv outAddr)
        = some (if activeV s DV BV jv then
            bwdDvStepSpec s DHPrev q do_ k t s_qk_h s_vo_h H DK DV BK BV scale jv
          else s.readMem dv outAddr) := by
  intro jv
  simp [exec, fused_recurrent_retention_bwd_dv_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, stateOffset, outStepOffset, activeV, vIdx, kIdx,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV + idx.1.val + t * DV
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      ∑ jk : Fin BK,
        (s.readMem DHPrev
            (s.pids 2 * DK * DV + (s.pids 1 * BK + jk.val) * DV + (s.pids 0 * BV + idx.1.val))
          + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * DK) * scale *
              s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * DV)) *
        s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * DK)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < DV
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : outStepOffset s t s_vo_h B H DV BV a = outStepOffset s t s_vo_h B H DV BV b := by
      simpa [offsetFn, outStepOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dv (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem dv (offsetFn (jv, PUnit.unit)) =
    if activeV s DV BV jv then
      bwdDvStepSpec s DHPrev q do_ k t s_qk_h s_vo_h H DK DV BK BV scale jv
    else s.readMem dv (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < DV
  · simp only [P, hjv, if_true, activeV, vIdx]
    simp only [valueFn, bwdDvStepSpec, stateOffset, kValR, vValR, kIdx, vIdx]
  · simp only [P, activeV, vIdx, hjv, if_false, BlockState.setReg_readMem]

theorem fused_recurrent_retention_bwd_dv_step_slice_compute_correct
    (DHPrev q do_ k dv : RegionName)
    (t s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s t s_vo_h B H DV BV jv)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dv_step_slice DHPrev q do_ k dv
        t s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s DV BV jv)
        (fun jv => (dv, outStepOffset s t s_vo_h B H DV BV jv)))
      (expected := fun jv : Fin BV =>
        bwdDvStepSpec s DHPrev q do_ k t s_qk_h s_vo_h H DK DV BK BV scale jv) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_retention_bwd_dv_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jv hActive
  have h := fused_recurrent_retention_bwd_dv_step_slice_correct DHPrev q do_ k dv
    t s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj jv
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **`dv` step realizes the genuine closed form.** Under the reverse carry
invariant `DHPrev = b_b · dStateClosed(t+1)`, the row-`t` `dv` reduction
equals the genuine `dvClosed(t)`. -/
theorem bwdDvStepSpec_eq_dvClosed
    (s : BlockState) (DHPrev q do_ k : RegionName)
    (t s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (ht : t < T)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (t + 1) jk jv)
    (jv : Fin BV) :
    bwdDvStepSpec s DHPrev q do_ k t s_qk_h s_vo_h H DK DV BK BV scale jv
      = dvClosed s q do_ k s_qk_h s_vo_h H DK DV BK BV T scale t jv := by
  unfold bwdDvStepSpec dvClosed
  refine Finset.sum_congr rfl fun jk _ => ?_
  rw [hPrev jk jv, dStateClosed_recurrence s q do_ s_qk_h s_vo_h
    H DK DV BK BV T scale t ht jk jv]
  ring

/-! ## Genuine closed-form step realizations (the carry-folds)

The cross-step folds over `range(0, T)` threading `h` and `d_h` are the
trusted boundary; each *step face* is realized against the genuine closed
forms `stateClosed` / `outClosed` / `dqClosed` / `dStateClosed` / `dkClosed` /
`dvClosed` over the input regions `q, k, v, do, initial_state`. -/

/-- **Genuine state carry-fold step.** One state-update body, with the
materialized pre-update buffer `HPrev = stateClosed(m)`, realizes the genuine
`(m+1)`-step folded state `stateClosed(m+1)`. -/
theorem fused_recurrent_retention_state_step_closed_form
    (HPrev k v h0 HOut : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h H DK DV BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1))
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV m jk jv) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_state_step_slice HPrev k v HOut
        m s_qk_h s_vo_h H DK DV BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s DK DV BK BV idx.2.1 idx.1))
      (expected := fun idx =>
        stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          (m + 1) idx.2.1 idx.1) := by
  have h := fused_recurrent_retention_state_step_slice_compute_correct HPrev k v
    HOut m s_qk_h s_vo_h H DK DV BK BV s hOutInj
  have hcong : (fun idx : TileIndex [BV, BK] =>
      stateStepSpec s HPrev k v m s_qk_h s_vo_h H DK DV BK BV idx)
      = (fun idx : TileIndex [BV, BK] =>
        stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          (m + 1) idx.2.1 idx.1) := by
    funext idx
    exact stateStepSpec_eq_stateClosed_succ s HPrev k v h0 USE_INITIAL_STATE
      m s_qk_h s_vo_h H DK DV BK BV hPrev idx
  rwa [hcong] at h

/-- **Genuine forward output step.** One output body, with the materialized
pre-update buffer `HPrev = stateClosed(m)`, realizes the genuine post-update
output `outClosed(m)` into the masked `o` row. -/
theorem fused_recurrent_retention_output_step_closed_form
    (HPrev q k v h0 o : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s m s_vo_h B H DV BV jv))
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV m jk jv) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_output_step_slice HPrev q k v o
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s DV BV jv)
        (fun jv => (o, outStepOffset s m s_vo_h B H DV BV jv)))
      (expected := fun jv : Fin BV =>
        outClosed s q k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          scale m jv) := by
  have h := fused_recurrent_retention_output_step_slice_compute_correct HPrev
    q k v o m s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj
  have hcong : (fun jv : Fin BV =>
      outputStepSpec s HPrev q k v m s_qk_h s_vo_h H DK DV BK BV scale jv)
      = (fun jv : Fin BV =>
        outClosed s q k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          scale m jv) := by
    funext jv
    exact outputStepSpec_eq_outClosed s HPrev q k v h0 USE_INITIAL_STATE
      m s_qk_h s_vo_h H DK DV BK BV scale hPrev jv
  rwa [hcong] at h

/-- **Genuine backward `dq` step.** One phase-1 body, with the materialized
pre-update rebuilt-state buffer `HPrev = stateClosed(m)`, realizes the genuine
`dqClosed(m)` into the masked `dq` row. -/
theorem fused_recurrent_retention_bwd_dq_step_closed_form
    (HPrev k v do_ h0 dq : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h B H DK DV BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dqStepOffset s m s_qk_h B H DK BK jk))
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV m jk jv) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dq_step_slice HPrev k v do_ dq
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s DK BK jk)
        (fun jk => (dq, dqStepOffset s m s_qk_h B H DK BK jk)))
      (expected := fun jk : Fin BK =>
        dqClosed s k v do_ h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          scale m jk) := by
  have h := fused_recurrent_retention_bwd_dq_step_slice_compute_correct HPrev
    k v do_ dq m s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj
  have hcong : (fun jk : Fin BK =>
      bwdDqStepSpec s HPrev k v do_ m s_qk_h s_vo_h H DK DV BK BV scale jk)
      = (fun jk : Fin BK =>
        dqClosed s k v do_ h0 USE_INITIAL_STATE s_qk_h s_vo_h H DK DV BK BV
          scale m jk) := by
    funext jk
    exact bwdDqStepSpec_eq_dqClosed s HPrev k v do_ h0 USE_INITIAL_STATE
      m s_qk_h s_vo_h H DK DV BK BV scale hPrev jk
  rwa [hcong] at h

/-- **Genuine gradient-state carry step.** Under the reverse carry invariant
`DHPrev = b_b·dStateClosed(m+1)`, one carry body realizes
`b_b·dStateClosed(m)` — the invariant handed to time row `m−1`. -/
theorem fused_recurrent_retention_bwd_dstate_step_closed_form
    (DHPrev q do_ DHOut : RegionName)
    (m s_qk_h s_vo_h H DK DV BK BV T : Nat) (scale : ℝ) (s : BlockState)
    (hmT : m < T)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => stateOffset s DK DV BK BV idx.1 idx.2.1))
    (hDPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (m + 1) jk jv) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dstate_step_slice DHPrev q do_
        DHOut m s_qk_h s_vo_h H DK DV BK BV scale)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (DHOut, stateOffset s DK DV BK BV idx.1 idx.2.1))
      (expected := fun idx =>
        bbVal s H *
          dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale m
            idx.1 idx.2.1) := by
  have h := fused_recurrent_retention_bwd_dstate_step_slice_compute_correct
    DHPrev q do_ DHOut m s_qk_h s_vo_h H DK DV BK BV scale s hOutInj
  have hcong : (fun idx : TileIndex [BK, BV] =>
      bwdDStateStepSpec s DHPrev q do_ m s_qk_h s_vo_h H DK DV BK BV scale idx)
      = (fun idx : TileIndex [BK, BV] =>
        bbVal s H *
          dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale m
            idx.1 idx.2.1) := by
    funext idx
    exact bwdDStateStepSpec_eq s DHPrev q do_ m s_qk_h s_vo_h H DK DV BK BV T
      scale hmT hDPrev idx
  rwa [hcong] at h

/-- **Genuine backward `dk` step.** Under the reverse carry invariant, one
reverse body realizes `dkClosed(m)` into the masked `dk` row. -/
theorem fused_recurrent_retention_bwd_dk_step_closed_form
    (DHPrev q do_ v dk : RegionName)
    (m s_qk_h s_vo_h B H DK DV BK BV T : Nat) (scale : ℝ) (s : BlockState)
    (hmT : m < T)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dqStepOffset s m s_qk_h B H DK BK jk))
    (hDPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (m + 1) jk jv) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dk_step_slice DHPrev q do_ v dk
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s DK BK jk)
        (fun jk => (dk, dqStepOffset s m s_qk_h B H DK BK jk)))
      (expected := fun jk : Fin BK =>
        dkClosed s q do_ v s_qk_h s_vo_h H DK DV BK BV T scale m jk) := by
  have h := fused_recurrent_retention_bwd_dk_step_slice_compute_correct DHPrev
    q do_ v dk m s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj
  have hcong : (fun jk : Fin BK =>
      bwdDkStepSpec s DHPrev q do_ v m s_qk_h s_vo_h H DK DV BK BV scale jk)
      = (fun jk : Fin BK =>
        dkClosed s q do_ v s_qk_h s_vo_h H DK DV BK BV T scale m jk) := by
    funext jk
    exact bwdDkStepSpec_eq_dkClosed s DHPrev q do_ v m s_qk_h s_vo_h
      H DK DV BK BV T scale hmT hDPrev jk
  rwa [hcong] at h

/-- **Genuine backward `dv` step.** Under the reverse carry invariant, one
reverse body realizes `dvClosed(m)` into the masked `dv` row. -/
theorem fused_recurrent_retention_bwd_dv_step_closed_form
    (DHPrev q do_ k dv : RegionName)
    (m s_qk_h s_vo_h B H DK DV BK BV T : Nat) (scale : ℝ) (s : BlockState)
    (hmT : m < T)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s m s_vo_h B H DV BV jv))
    (hDPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem DHPrev (stateOffset s DK DV BK BV jk jv)
        = bbVal s H *
            dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale (m + 1) jk jv) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dv_step_slice DHPrev q do_ k dv
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s DV BV jv)
        (fun jv => (dv, outStepOffset s m s_vo_h B H DV BV jv)))
      (expected := fun jv : Fin BV =>
        dvClosed s q do_ k s_qk_h s_vo_h H DK DV BK BV T scale m jv) := by
  have h := fused_recurrent_retention_bwd_dv_step_slice_compute_correct DHPrev
    q do_ k dv m s_qk_h s_vo_h B H DK DV BK BV scale s hOutInj
  have hcong : (fun jv : Fin BV =>
      bwdDvStepSpec s DHPrev q do_ k m s_qk_h s_vo_h H DK DV BK BV scale jv)
      = (fun jv : Fin BV =>
        dvClosed s q do_ k s_qk_h s_vo_h H DK DV BK BV T scale m jv) := by
    funext jv
    exact bwdDvStepSpec_eq_dvClosed s DHPrev q do_ k m s_qk_h s_vo_h
      H DK DV BK BV T scale hmT hDPrev jv
  rwa [hcong] at h

/-! ## Offset-injectivity side conditions (dimension-general)

The flat state address `i_bh·DK·DV + (i_k·BK + j_k)·DV + (i_v·BV + j_v)` is
injective on both tile orientations whenever the tile fits the logical extents
(`BV ≤ DV`): `j_v` occupies the low `DV`-digit and `j_k·DV` the next, a
faithful mixed-radix encoding. The per-time output-row addresses are injective
in their lane whenever the tile is nonempty (`0 < BK` / `0 < BV`, contiguous
lanes in the low digit). Honest structural side conditions; the Python
regression (`BK = BV = DK = DV = 16`) satisfies them. -/

/-- **Dimension-general** state-address injectivity on the forward `[BV, BK]`
tile, given the tile fits (`BV ≤ DV`). -/
theorem fused_recurrent_retention_state_offset_injective_general
    (s : BlockState) (DK DV BK BV : Nat) (hBV : BV ≤ DV) :
    Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s DK DV BK BV idx.2.1 idx.1) := by
  rintro ⟨⟨av, hav⟩, ⟨ak, hak⟩, _⟩ ⟨⟨bv, hbv⟩, ⟨bk, hbk⟩, _⟩ h
  simp [stateOffset, vIdx, kIdx] at h
  have havV : av < DV := lt_of_lt_of_le hav hBV
  have hbvV : bv < DV := lt_of_lt_of_le hbv hBV
  have hVpos : 0 < DV := lt_of_le_of_lt (Nat.zero_le _) havV
  have hcore : (s.pids 1 * BK + ak) * DV + av = (s.pids 1 * BK + bk) * DV + bv := by
    omega
  have hv : av = bv := by
    have ha : ((s.pids 1 * BK + ak) * DV + av) % DV = av % DV := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hb : ((s.pids 1 * BK + bk) * DV + bv) % DV = bv % DV := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hmod : ((s.pids 1 * BK + ak) * DV + av) % DV
        = ((s.pids 1 * BK + bk) * DV + bv) % DV := by rw [hcore]
    rwa [ha, hb, Nat.mod_eq_of_lt havV, Nat.mod_eq_of_lt hbvV] at hmod
  have hk : ak = bk := by
    subst hv
    have hmul : (s.pids 1 * BK + ak) * DV = (s.pids 1 * BK + bk) * DV := by omega
    have := Nat.eq_of_mul_eq_mul_right hVpos hmul
    omega
  subst hv; subst hk; rfl

/-- **Dimension-general** state-address injectivity on the backward `[BK, BV]`
tile, given the tile fits (`BV ≤ DV`). -/
theorem fused_recurrent_retention_bwd_state_offset_injective_general
    (s : BlockState) (DK DV BK BV : Nat) (hBV : BV ≤ DV) :
    Function.Injective
      (fun idx : TileIndex [BK, BV] => stateOffset s DK DV BK BV idx.1 idx.2.1) := by
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  have h' : stateOffset s DK DV BK BV ⟨ak, hak⟩ ⟨av, hav⟩
      = stateOffset s DK DV BK BV ⟨bk, hbk⟩ ⟨bv, hbv⟩ := h
  have := fused_recurrent_retention_state_offset_injective_general s DK DV BK BV
    hBV (a₁ := (⟨av, hav⟩, ⟨ak, hak⟩, PUnit.unit))
    (a₂ := (⟨bv, hbv⟩, ⟨bk, hbk⟩, PUnit.unit)) h'
  simp only [Prod.mk.injEq] at this
  obtain ⟨h1, h2, -⟩ := this
  rw [h1, h2]

/-- **Dimension-general** per-time `o`/`dv` row-address injectivity, given
`0 < BV`. -/
theorem fused_recurrent_retention_out_step_offset_injective_general
    (s : BlockState) (t s_vo_h B H DV BV : Nat) (hBV : 0 < BV) :
    Function.Injective (fun jv : Fin BV => outStepOffset s t s_vo_h B H DV BV jv) := by
  intro a b h
  simp [outStepOffset] at h
  exact Fin.ext h

/-- **Dimension-general** per-time `dq`/`dk` row-address injectivity, given
`0 < BK`. -/
theorem fused_recurrent_retention_dq_step_offset_injective_general
    (s : BlockState) (t s_qk_h B H DK BK : Nat) (hBK : 0 < BK) :
    Function.Injective (fun jk : Fin BK => dqStepOffset s t s_qk_h B H DK BK jk) := by
  intro a b h
  simp [dqStepOffset] at h
  exact Fin.ext h

/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about six hand-cut single-step slices, not about the
launched kernels.** The launched forward and backward surfaces appear *only* in
clauses 1 and 2, which say nothing more than "they lower to the algorithm layer".
Two of the six faces (clauses 4 and 6) write **fiction regions** — `HOut` and
`DHOut` are the internal carry registers `h` and `d_h`, not Python tensors — so
they are step lemmas about a register update, not claims about kernel outputs.
The `STORE_FINAL_STATE` writeback has no face at all (see the module docstring).

**Genuine, shape-general fused recurrent retention step summary, forward and
backward.** Parameterized over the symbolic head strides `s_qk_h s_vo_h`,
batch/head/time `B H T`, head extents `DK DV`, tile sizes `BK BV`, the real
`scale`, the step index `m`, and both flags
`USE_INITIAL_STATE STORE_FINAL_STATE`. Each face is realized against the genuine
closed forms `stateClosed` / `outClosed` / `dqClosed` / `dStateClosed` /
`dkClosed` / `dvClosed` over the *input* regions
`q, k, v, do, initial_state` — never a read-back of the kernel's own output:

1. the full **forward** surface lowers to the algorithm layer;
2. the full **backward** surface (both loops, `tl.debug_barrier()`, pointer
   rebasing and decrements) lowers to the algorithm layer;
2a. the **`USE_INITIAL_STATE` seed prologue** realizes `stateClosed(0)` — the
   `m = 0` instance of the same closed form the step faces carry — into the
   internal carry register `HSeed`, for either flag setting;
2b. the reverse invariant's **top value** is zero: `b_b·dStateClosed(T) = 0`;
3. one forward **output** body realizes `outClosed(m)` — the reduction of the
   post-update state `stateClosed(m+1)` against `scale·q_m` — into the Python
   tensor `o`;
4. one **state-update** body realizes `stateClosed(m+1)` (the scalar-decay
   carry-fold `h = b_b·h + k_m ⊗ v_m`) into the *internal carry register*
   `HOut`, given the assumed invariant `HPrev = stateClosed(m)` — shared by the
   forward and backward-phase-1 loops;
5. one backward **`dq`** body realizes `dqClosed(m)` (post-update state
   reduced against `do_m`, then `·scale`) into the Python tensor `dq`;
6. one reverse **gradient-state carry** body realizes `b_b·dStateClosed(m)` into
   the *internal carry register* `DHOut`, given the assumed reverse invariant
   `DHPrev = b_b·dStateClosed(m+1)`;
7. one reverse **`dk`** body realizes `dkClosed(m)` into `dk`;
8. one reverse **`dv`** body realizes `dvClosed(m)` into `dv`.

Side conditions. Structural: `BV ≤ DV` (the tile fits the logical extents,
giving state-address injectivity), `0 < BK`, `0 < BV` (nonempty contiguous
output lanes), and `m < T` for the reverse-phase faces (the reverse loop visits
exactly the rows `T−1, …, 0`). **Load-bearing:** `hPrev` and `hDPrev` — the two
carry invariants — are *assumptions*, and they carry the entire recurrence.
Every clause that mentions a closed form does so only under them.

**They are not self-propagating.** Clause 4 writes `HOut`, while `hPrev`
constrains `HPrev`: a different region, and nothing chains one step's output
register into the next step's input register — that step-chaining is the one
piece of the induction still missing. What is **no longer** missing is either
base case: clause 2a is `stateClosed 0 = stateSeed` realized by the kernel's own
seed prologue, and clause 2b is `b_b·dStateClosed T = 0`. The flags flow through
verbatim and clauses 2a and 3–8 hold for every flag setting. -/
specification fused_recurrent_retention_output_summary_general
    (q k v o do_ dq dk dv initial_state final_state
      HSeed HPrev HOut DHPrev DHOut : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s_qk_h s_vo_h B H T DK DV BK BV m : Nat) (scale : ℝ) (s : BlockState)
    (hBV : BV ≤ DV) (hBKpos : 0 < BK) (hBVpos : 0 < BV)
    (hmT : m < T)
    (hPrev : ∀ (jk : Fin BK) (jv : Fin BV),
      s.readMem HPrev (stateOffset s DK DV BK BV jk jv)
        = stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
            H DK DV BK BV m jk jv)
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
    -- (2a) the seed prologue realizes `stateClosed 0` (the carry base case),
    --      into the INTERNAL carry register `HSeed` (a fiction region)
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_seed_slice initial_state HSeed
        DK DV BK BV USE_INITIAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => activeKV s DK DV BK BV idx)
        (fun idx : TileIndex [BV, BK] =>
          (HSeed, stateOffset s DK DV BK BV idx.2.1 idx.1)))
      (expected := fun idx : TileIndex [BV, BK] =>
        stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
          H DK DV BK BV 0 idx.2.1 idx.1)) ∧
    -- (2b) the reverse carry invariant's base case at the top of the scan
    (∀ (jk : Fin BK) (jv : Fin BV),
      bbVal s H *
        dStateClosed s q do_ s_qk_h s_vo_h H DK DV BK BV T scale T jk jv = 0) ∧
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
    --      into the INTERNAL carry register `HOut` (a fiction region)
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_state_step_slice HPrev k v HOut
        m s_qk_h s_vo_h H DK DV BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s DK DV BK BV idx.2.1 idx.1))
      (expected := fun idx =>
        stateClosed s k v initial_state USE_INITIAL_STATE s_qk_h s_vo_h
          H DK DV BK BV (m + 1) idx.2.1 idx.1)) ∧
    -- (5) the backward `dq` body realizes the genuine `dqClosed(m)`
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
    -- (6) the reverse gradient-state carry body realizes `b_b·dStateClosed(m)`
    --      into the INTERNAL carry register `DHOut` (a fiction region)
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
    -- (7) the reverse `dk` body realizes the genuine `dkClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dk_step_slice DHPrev q do_ v dk
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s DK BK jk)
        (fun jk => (dk, dqStepOffset s m s_qk_h B H DK BK jk)))
      (expected := fun jk : Fin BK =>
        dkClosed s q do_ v s_qk_h s_vo_h H DK DV BK BV T scale m jk)) ∧
    -- (8) the reverse `dv` body realizes the genuine `dvClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_retention_bwd_dv_step_slice DHPrev q do_ k dv
        m s_qk_h s_vo_h B H DK DV BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s DV BV jv)
        (fun jv => (dv, outStepOffset s m s_vo_h B H DV BV jv)))
      (expected := fun jv : Fin BV =>
        dvClosed s q do_ k s_qk_h s_vo_h H DK DV BK BV T scale m jv)) := by
  have hStateInj := fused_recurrent_retention_state_offset_injective_general
    s DK DV BK BV hBV
  have hBwdStateInj := fused_recurrent_retention_bwd_state_offset_injective_general
    s DK DV BK BV hBV
  have hOutInj := fused_recurrent_retention_out_step_offset_injective_general
    s m s_vo_h B H DV BV hBVpos
  have hDqInj := fused_recurrent_retention_dq_step_offset_injective_general
    s m s_qk_h B H DK BK hBKpos
  refine ⟨fused_recurrent_retention_fwd_surface_toAlgorithm_supported
      _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _,
    fused_recurrent_retention_bwd_surface_toAlgorithm_supported
      _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact fused_recurrent_retention_seed_slice_realizes_stateClosed_zero
      initial_state HSeed k v DK DV BK BV USE_INITIAL_STATE s_qk_h s_vo_h H s
      hStateInj
  · intro jk jv
    exact bb_mul_dStateClosed_top s q do_ s_qk_h s_vo_h H DK DV BK BV T scale
      jk jv
  · exact fused_recurrent_retention_output_step_closed_form HPrev q k v
      initial_state o USE_INITIAL_STATE m s_qk_h s_vo_h B H DK DV BK BV scale s
      hOutInj hPrev
  · exact fused_recurrent_retention_state_step_closed_form HPrev k v
      initial_state HOut USE_INITIAL_STATE m s_qk_h s_vo_h H DK DV BK BV s
      hStateInj hPrev
  · exact fused_recurrent_retention_bwd_dq_step_closed_form HPrev k v do_
      initial_state dq USE_INITIAL_STATE m s_qk_h s_vo_h B H DK DV BK BV scale s
      hDqInj hPrev
  · exact fused_recurrent_retention_bwd_dstate_step_closed_form DHPrev q do_
      DHOut m s_qk_h s_vo_h H DK DV BK BV T scale s hmT hBwdStateInj hDPrev
  · exact fused_recurrent_retention_bwd_dk_step_closed_form DHPrev q do_ v dk
      m s_qk_h s_vo_h B H DK DV BK BV T scale s hmT hDqInj hDPrev
  · exact fused_recurrent_retention_bwd_dv_step_closed_form DHPrev q do_ k dv
      m s_qk_h s_vo_h B H DK DV BK BV T scale s hmT hOutInj hDPrev

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention

