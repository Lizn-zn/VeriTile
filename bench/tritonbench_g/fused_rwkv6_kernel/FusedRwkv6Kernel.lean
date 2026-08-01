import VeriTile.Triton

/-!
# `fused_rwkv6_kernel` — genuine RWKV-6 closed-form correctness

`fused_recurrent_rwkv6_fwd_kernel` is the RWKV-6 forward recurrent state scan.
Program `(i_v, i_k, i_bh)` carries a `[BV, BK]` state tile `b_h` across the
`0..T` time loop. With `q_t = scale · q[t]`, per step `t`:

* `b_kv[j_v, j_k] = k_t[j_k] · v_t[j_v]`                          (outer product)
* `o_t[j_v]       = Σ_{j_k} (b_h[j_v,j_k] + b_kv[j_v,j_k] · u[j_k]) · q_t[j_k]`
                                                       (reduction over the *pre-update* state)
* `b_h[j_v,j_k]   = b_h[j_v,j_k] · exp(w_t[j_k]) + b_kv[j_v,j_k]`
                                                       (per-channel `k`-axis decay)

with an optional `h0` initial-state seed and an optional `ht` final-state store.

## Genuine closed form (NOT a self-reference)

Unrolling the recurrence, the state at step `m` (tile element `(j_v, j_k)`) is

```
b_h^(m)[j_v,j_k] = h0[j_v,j_k] · ∏_{j<m} exp(w_j[j_k])
                 + Σ_{t<m} (k_t[j_k]·v_t[j_v]) · ∏_{t<j<m} exp(w_j[j_k])
```

— `stateClosed` below, a standalone specification over the *input* regions
`k,v,w,h0` (the `q`/`u` channels enter only in `outputClosed`), never a
read-back of the kernel's own output. This is the
gated-recurrence carry-fold of `chunk_gate_recurrence` (PR #290), generalized
from a *scalar* gate to the per-channel decay gate `exp(w_t[j_k])` and an
outer-product source `k_t ⊗ v_t`. The per-step output is the reduction

```
o_m[j_v] = Σ_{j_k} ( b_h^(m)[j_v,j_k] + k_m[j_k]·v_m[j_v]·u[j_k] ) · scale·q_m[j_k]
```

(`outputClosed`), and the final state is `stateClosed` at `m = T`.

## Proof architecture

```
fused_recurrent_rwkv6_output_summary_general                  ← TOP THEOREM
  ├─ fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported   surface lowers
  ├─ fused_recurrent_rwkv6_output_step_closed_form             output reduction
  │      └─ outputStepSpec_eq_outputClosed   (Σ_{j_k} … = o_m)
  ├─ fused_recurrent_rwkv6_state_step_closed_form              state step
  │      (masked loads + `mask_kv` store ⇒ `writeIf (activeKV …)`)
  │      └─ stateStepSpec_eq_stateClosed_succ
  │           └─ stateClosed_succ  (b_h^(m+1) = b_h^(m)·exp(w_m) + k_m⊗v_m)
  └─ fused_recurrent_rwkv6_state_carry_fold                    cross-step fold
         └─ CarryFold.carryFold_execChain
              ├─ fused_recurrent_rwkv6_state_step_frame       (frame)
              ├─ stateStepSpec_transport                      (congruence)
              └─ finalStateOffset_congr / activeKV_congr
```

## Modeling boundary — read before trusting anything below

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. `REVERSE` is fixed to the exported default `false`.

The two correctness faces are proved over **hand-cut step slices**
(`fused_recurrent_rwkv6_output_step_slice` /
`fused_recurrent_rwkv6_state_step_slice`), *not* over the launched surface
`fused_recurrent_rwkv6_fwd_surface` — the surface appears only under
`toAlgorithm?` (clause 1 says it lowers, nothing more). Concretely, the
following are **outside** what any theorem in this file claims:

* **The cross-step fold — chained for the state, still open for the loop.**
  The individual step faces (clauses 2 and 3) each *assume* the carry invariant
  `BHPrev = stateClosed(m)` at their own `m`; that assumption is now a
  clause-local antecedent, not a hypothesis of the whole theorem.
  `fused_recurrent_rwkv6_state_carry_fold` (clause 4) discharges it for the
  **state** recurrence: `T` state slices run as a `CarryFold.execChain` whose
  carry-in and carry-out region are literally the same name `C`, so a `T`-step
  story needs *one* assumption about the initial buffer instead of `T` pinned
  ones. What that fold does **not** cover:
  - the launched `fused_recurrent_rwkv6_fwd_surface` itself. It keeps `b_h` in
    a *register* across its own `Stmt.forRange`; the chain models the
    materialized-state program, one launch per step. These are different
    objects and the register loop remains unmodeled here;
  - the per-step output row `o_t`, which writes `o` rather than `C` and so is
    not a stage of the chain. Clause 2 stays a single-step statement.
* **The load masks — no longer a gap.** Both step slices are now
  **mask-faithful**, so neither face carries a full-tile hypothesis. Python
  guards every load with `mask_bk` / `mask_bv` (`other=0`) and every store with
  `mask_kv` / `mask_bv`, and both slices reproduce all of them:
  - `fused_recurrent_rwkv6_state_step_slice` reproduces the four masks and the
    `mask_kv` store, and its face is a `writeIf (activeKV …)` statement: on an
    in-range `(j_v, j_k)` lane every operand of that lane is in range too, so
    the masked and live reads agree exactly where the face looks.
  - `fused_recurrent_rwkv6_output_step_slice` reproduces `mask_kv` on `prev`,
    `mask_bk` on `b_k`/`b_q`/`b_u`, `mask_bv` on `b_v`, and the `mask_bv` store.
    Its `expected` (`outputClosed`) is correspondingly **guarded**: the key-axis
    summand is `if activeK … then … else 0`, matching Python, where an
    out-of-range key column contributes `(0 + 0·v·0)·0 = 0` to
    `tl.sum(_, axis=1)`. What unblocks the guarded sum is
    `VeriTile/Triton/Semantics/MaskedReduction.lean`'s `ite_some_some` (which
    pushes each `other = 0.0` carrier's `some` outward so the existing `@[simp]`
    `WithBot` helpers can collapse the whole factor tower) together with the
    `TileShape.insertAxisIndex` `Fin`-literal normalizations; the missing one of
    those is what previously forced the full-tile scoping.
* **The `STORE_FINAL_STATE` writeback.** Not covered by any correctness face.
  The former `final_state_store_slice` face was a masked memcpy (load and store
  addresses character-identical) whose only content was the assumption
  `BHFinal = stateClosed(T)`; it has been deleted rather than presented as a
  result about `ht`.
* **The host-side `o.sum(0)`.** The Python launcher allocates
  `o : [NK, B, H, T, V]` and returns `o.sum(0)` — a reduction over the `NK`
  key-tile programs. `outputClosed` is the contribution of a *single* `i_k`
  program, so it is the value landing in `o[i_k]`, not the value the caller of
  `fused_recurrent_rwkv6` observes.

What *is* genuine: each step face's `expected` is a closed form
(`stateClosed` / `outputClosed`) over the **input** regions `q,k,v,w,u,h0`,
never a read-back of the kernel's own output; the `USE_INITIAL_STATE` branch is
modeled inside `stateSeed`, and the faces hold for either setting of it.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `fused_recurrent_rwkv6_output_summary_general` — shape-general.
Clauses 2 and 3 are single-step **slice** faces, both mask-faithful (they hold
for partial tiles with no full-tile hypothesis); clause 4 chains the state
slices across all `T` steps through a shared carry region. None of the four is
a statement about the launched kernel's register loop. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `fused_rwkv6_kernel.py`'s
`fused_recurrent_rwkv6_fwd_kernel` as used by the exported benchmark helper.

The exported helper always calls the autograd entry point with its default
`reverse = false`, so pointer movement is modeled in the forward direction. The
`REVERSE` parameter is retained to match the source signature. -/
def fused_recurrent_rwkv6_fwd_surface
    (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  p_q = q + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_v_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) +
    i_v * $(BV) + tl.arange(0, $(BV))
  p_w = w + i_bh * $(s_k_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_u = u + i_h * $(K) + tl.arange(0, $(BK)) + i_k * $(BK)
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  b_h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
    b_w = tl.exp(b_w)
    b_kv = b_k[None, :] * b_v[:, None]
    b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
    b_o = tl.sum(b_o, axis=1)
    b_h = b_h * b_w[None, :]
    b_h += b_kv
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_w += $(K)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}

/-- The full fused RWKV6 recurrent forward surface lowers to the algorithm
layer. -/
theorem fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported
    (q k v w u o h0 ht : RegionName)
    (s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE REVERSE : Bool) :
    ∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht s_k_h
      s_v_h B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      REVERSE).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_rwkv6_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Index / offset helpers

`s.pids 0 = i_v`, `s.pids 1 = i_k`, `s.pids 2 = i_bh`. A `[BV, BK]` state tile is
indexed by `idx : TileIndex [BV, BK]` with `idx.1` the value (`j_v`) axis and
`idx.2.1` the key (`j_k`) axis. -/

def vIndex (s : BlockState) (BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 0 * BV + jv.val

def kIndex (s : BlockState) (BK : Nat) (jk : Fin BK) : Nat :=
  s.pids 1 * BK + jk.val

/-- Flattened state-tile / `h0` address (row-major `i_bh·K·V + j_k·V + j_v`). -/
def finalStateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.2.1 * V + vIndex s BV idx.1

/-- Python's `mask_bv` lane predicate (`i_v·BV + j_v < V`). -/
def active (s : BlockState) (V BV : Nat) (jv : Fin BV) : Prop :=
  vIndex s BV jv < V

instance activeDecidable (s : BlockState) (V BV : Nat) (jv : Fin BV) :
    Decidable (active s V BV jv) := by
  unfold active; infer_instance

/-- Python's `mask_bk` lane predicate (`i_k·BK + j_k < K`). -/
def activeK (s : BlockState) (K BK : Nat) (jk : Fin BK) : Prop :=
  kIndex s BK jk < K

instance activeKDecidable (s : BlockState) (K BK : Nat) (jk : Fin BK) :
    Decidable (activeK s K BK jk) := by
  unfold activeK; infer_instance

/-- Python's `mask_kv = mask_bv[:, None] & mask_bk[None, :]` tile predicate. -/
def activeKV (s : BlockState) (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : Prop :=
  active s V BV idx.1 ∧ activeK s K BK idx.2.1

instance activeKVDecidable (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Decidable (activeKV s K V BK BV idx) := by
  unfold activeKV; infer_instance

/-! ## Genuine closed-form data (over the *input* regions)

`qVal/kVal/vVal/uVal/h0Val` read the kernel's exact block-pointer layouts at
time row `t`. The decay gate is the per-channel `decay s w t j_k = exp(w_t[j_k])`. -/

/-- Element `R[i_bh][t, j_k]` of the shared `[T, K]` **k-layout** at time row
`t`, key channel `j_k` (offset `i_bh·s_k_h + t·K + (i_k·BK + j_k)`). The `q`,
`k`, and `w` block pointers all use this layout — `qVal` and `decay` read
through it. -/
noncomputable def kVal (s : BlockState) (k : RegionName)
    (s_k_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + s.pids 1 * BK + jk.val + t * K)

/-- `q[t][j_k]·scale` — the kernel multiplies the loaded `q` row (k-layout)
by `scale`. -/
noncomputable def qVal (s : BlockState) (q : RegionName)
    (s_k_h K BK : Nat) (scale : ℝ) (t : Nat) (jk : Fin BK) : ℝ :=
  scale * kVal s q s_k_h K BK t jk

/-- Element `v[i_bh][t, j_v]` of the `[T, V]` **v-layout** at time row `t`,
value channel `j_v` (offset `i_bh·s_v_h + t·V + (i_v·BV + j_v)`). -/
noncomputable def vVal (s : BlockState) (v : RegionName)
    (s_v_h V BV : Nat) (t : Nat) (jv : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + s.pids 0 * BV + jv.val + t * V)

/-- Per-head bonus row element `u[i_bh % H][j_k]` of the `[H, K]` row-major
`u` matrix (time-independent). -/
noncomputable def uVal (s : BlockState) (u : RegionName)
    (H K BK : Nat) (jk : Fin BK) : ℝ :=
  s.readMem u ((s.pids 2 % H) * K + jk.val + s.pids 1 * BK)

/-- Per-channel decay gate at time `t`, key channel `j_k`: `exp(w_t[j_k])`
(the `w` row read through the shared k-layout). -/
noncomputable def decay (s : BlockState) (w : RegionName)
    (s_k_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  Real.exp (kVal s w s_k_h K BK t jk)

noncomputable def h0Val (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  s.readMem h0 (finalStateOffset s K V BK BV idx)

/-- Seeded initial state `b_h^(0)`: `h0` if `USE_INITIAL_STATE` else `0`. -/
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : ℝ :=
  if USE_INITIAL_STATE then h0Val s h0 K V BK BV idx else 0

/-- **Genuine closed form for the state after `m` steps**, tile element `idx`:
`seed · ∏_{j<m} exp(w_j) + Σ_{t<m} (k_t·v_t) · ∏_{t<j<m} exp(w_j)`. This is a
standalone specification over the input regions `k,v,w,h0` — never a read-back
of the kernel's own output. -/
noncomputable def stateClosed
    (s : BlockState) (k v w h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_k_h s_v_h K V BK BV m : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  stateSeed s h0 USE_INITIAL_STATE K V BK BV idx *
      (∏ j ∈ Finset.range m, decay s w s_k_h K BK j idx.2.1) +
    ∑ t ∈ Finset.range m,
      (kVal s k s_k_h K BK t idx.2.1 * vVal s v s_v_h V BV t idx.1) *
        (∏ j ∈ Finset.Ico (t + 1) m, decay s w s_k_h K BK j idx.2.1)

/-- **The state carry-fold recurrence.** Unrolling one step:
`b_h^(m+1) = b_h^(m) · exp(w_m) + k_m·v_m`. This is the exact closed-form
counterpart of the Python loop body `b_h = b_h * exp(w_t) + k_t ⊗ v_t`. -/
theorem stateClosed_succ
    (s : BlockState) (k v w h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_k_h s_v_h K V BK BV m : Nat) (idx : TileIndex [BV, BK]) :
    stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV (m + 1) idx
      = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx *
            decay s w s_k_h K BK m idx.2.1
        + kVal s k s_k_h K BK m idx.2.1 * vVal s v s_v_h V BV m idx.1 := by
  unfold stateClosed
  rw [Finset.prod_range_succ, Finset.sum_range_succ]
  rw [show Finset.Ico (m + 1) (m + 1) = (∅ : Finset Nat) from by simp,
      Finset.prod_empty, mul_one]
  have hsum :
      (∑ t ∈ Finset.range m,
          (kVal s k s_k_h K BK t idx.2.1 * vVal s v s_v_h V BV t idx.1) *
            ∏ j ∈ Finset.Ico (t + 1) (m + 1), decay s w s_k_h K BK j idx.2.1)
        = (∑ t ∈ Finset.range m,
            (kVal s k s_k_h K BK t idx.2.1 * vVal s v s_v_h V BV t idx.1) *
              ∏ j ∈ Finset.Ico (t + 1) m, decay s w s_k_h K BK j idx.2.1)
          * decay s w s_k_h K BK m idx.2.1 := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t ht
    simp only [Finset.mem_range] at ht
    rw [Finset.prod_Ico_succ_top (by omega : t + 1 ≤ m)]
    ring
  rw [hsum]; ring

/-- **Genuine closed form for output row `m`, lane `j_v`** — the per-step
reduction over the *in-range* key channels (`mask_bk`, i.e. `activeK`):
`o_m[j_v] = Σ_{j_k ∈ mask_bk} (b_h^(m)[j_v,j_k] + k_m[j_k]·v_m[j_v]·u[j_k]) · scale·q_m[j_k]`,
reading the *pre-update* state `stateClosed(m)`. The `activeK` guard mirrors
Python's `other=0` loads: an out-of-range key column contributes nothing to
`tl.sum(_, axis=1)`. -/
noncomputable def outputClosed
    (s : BlockState) (q k v w u h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_k_h s_v_h H K V BK BV : Nat) (scale : ℝ) (m : Nat) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    if activeK s K BK jk then
      (stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m
          (TileShape.insertAxisIndex [BV, BK] 1
            (TileShape.insertAxisIndex [BV] 0 PUnit.unit jv) jk)
        + (kVal s k s_k_h K BK m jk * vVal s v s_v_h V BV m jv) *
            uVal s u H K BK jk)
      * qVal s q s_k_h K BK scale m jk
    else 0

/-! ## State-update step slice (the per-channel decay carry-fold body)

This isolates the Python loop body's state update from the cross-step loop
induction. It loads the materialized previous-state tile `BHPrev`, the time-row
`k_t/v_t/w_t`, computes the outer product `b_kv = k ⊗ v`, the per-channel decay
`exp(w_t)`, and stores `b_h·exp(w_t) + b_kv` into a state buffer `BHOut` at the
canonical `[BV,BK]` layout. -/

def fused_recurrent_rwkv6_state_step_slice
    (BHPrev k v w BHOut : RegionName)
    (t s_k_h s_v_h K V BK BV : Nat) : ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_w = tl.load(w + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_w = tl.exp(b_w)
  b_kv = b_k[None, :] * b_v[:, None]
  acc = prev * b_w[None, :] + b_kv
  tl.store(BHOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(BHOut.dtype.element_ty), mask=mask_kv)
}

theorem fused_recurrent_rwkv6_state_step_slice_toAlgorithm_supported
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) :
    ∃ alg, (fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
      t s_k_h s_v_h K V BK BV).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_rwkv6_state_step_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- The arithmetic spec of one state-update body: `BHPrev·exp(w_t) + k_t·v_t`. -/
noncomputable def stateStepSpec
    (s : BlockState) (BHPrev k v w : RegionName)
    (t s_k_h s_v_h K V BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  s.readMem BHPrev (finalStateOffset s K V BK BV idx) *
      decay s w s_k_h K BK t idx.2.1
    + kVal s k s_k_h K BK t idx.2.1 * vVal s v s_v_h V BV t idx.1

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_rwkv6_state_step_slice_correct
    (BHPrev k v w BHOut : RegionName)
    (t s_k_h s_v_h K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => finalStateOffset s K V BK BV idx)) :
    ∀ idx : TileIndex [BV, BK],
      let outAddr := finalStateOffset s K V BK BV idx
      (exec (fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
            t s_k_h s_v_h K V BK BV) s).map (·.readMem BHOut outAddr)
        = some (if activeKV s K V BK BV idx then
            stateStepSpec s BHPrev k v w t s_k_h s_v_h K V BK BV idx
          else s.readMem BHOut outAddr) := by
  intro idx
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => s.pids 2 * K * V +
        (s.pids 1 * BK + idx.2.1.val) * V + (s.pids 0 * BV + idx.1.val)) := by
    simpa [finalStateOffset, kIndex, vIndex] using hOutInj
  simp [exec, fused_recurrent_rwkv6_state_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, finalStateOffset, kIndex, vIndex,
        TileShape.dropInsertedIndex]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
  by_cases hjv : s.pids 0 * BV + idx.1.val < V
  · by_cases hjk : s.pids 1 * BK + idx.2.1.val < K
    · simp [activeKV, active, activeK, vIndex, kIndex, hjv, hjk, stateStepSpec,
        finalStateOffset, decay, kVal, vVal, NumericDType.add, NumericDType.mul]
    · simp [activeKV, active, activeK, vIndex, kIndex, hjv, hjk, finalStateOffset]
  · simp [activeKV, active, activeK, vIndex, kIndex, hjv, finalStateOffset]

theorem fused_recurrent_rwkv6_state_step_slice_compute_correct
    (BHPrev k v w BHOut : RegionName)
    (t s_k_h s_v_h K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => finalStateOffset s K V BK BV idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
        t s_k_h s_v_h K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => activeKV s K V BK BV idx)
        (fun idx => (BHOut, finalStateOffset s K V BK BV idx)))
      (expected := fun idx =>
        stateStepSpec s BHPrev k v w t s_k_h s_v_h K V BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_rwkv6_state_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := fused_recurrent_rwkv6_state_step_slice_correct BHPrev k v w BHOut
    t s_k_h s_v_h K V BK BV s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **State carry-fold step (genuine).** If the materialized previous-state buffer
`BHPrev` holds the genuine `m`-step folded state `stateClosed(m)`, then one loop
body — `stateStepSpec`, i.e. `BHPrev·exp(w_m) + k_m·v_m` — produces exactly the
genuine `(m+1)`-step folded state `stateClosed(m+1)`. -/
theorem stateStepSpec_eq_stateClosed_succ
    (s : BlockState) (BHPrev k v w h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_k_h s_v_h K V BK BV : Nat)
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem BHPrev (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx)
    (idx : TileIndex [BV, BK]) :
    stateStepSpec s BHPrev k v w m s_k_h s_v_h K V BK BV idx
      = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV (m + 1) idx := by
  rw [stateClosed_succ]
  unfold stateStepSpec
  rw [hPrev idx]

/-! ## Cross-step carry fold — the `range(0, T)` loop, threaded through memory

Every step face above assumes its carry invariant (`hPrev`) at *its own* step.
This section discharges that assumption for all but the first step: the state
slices are run as a `CarryFold.execChain` whose carry-in and carry-out region
are literally the same name `C`, and what survives is a single hypothesis about
the **initial** buffer (`hSeed`) instead of one hypothesis per step.

Honesty limits, restated from `VeriTile.Triton.CarryFold` because they bound
what `fused_recurrent_rwkv6_state_carry_fold` below claims:

* the carry travels through a **memory region**, one launch per step. Python
  keeps `b_h` in a *register* across `range(0, T)`; that object is still not
  modeled anywhere in this file. What is proved here is the fold of the
  materialized-state program, not of the register loop;
* the fold covers the **state** recurrence only. The per-step output row `o_t`
  is not part of the chain — `fused_recurrent_rwkv6_output_step_slice` writes
  `o`, not `C`, so it is not a stage of this fold;
* `hSeed` is an assumption about the caller's buffer, not a theorem. It is the
  one place the seed enters, and it is stated only on write-active lanes,
  because that is the only place the chain can maintain it. -/

/-- One state step leaves everything outside the carry region alone: it writes
only `C`, and a `writeMem` scatter touches neither the program ids nor any other
region. This is obligation (1) of `CarryFold`'s docstring — a port's
`Realizes_without_Rounding` face pins only the *written* lanes, so the frame
needs its own walk. -/
theorem fused_recurrent_rwkv6_state_step_frame
    (C k v w : RegionName) (m s_k_h s_v_h K V BK BV : Nat) (u u' : BlockState)
    (hExec : exec (fused_recurrent_rwkv6_state_step_slice C k v w C
      m s_k_h s_v_h K V BK BV) u = some u') :
    u'.pids = u.pids ∧ ∀ r, r ≠ C → ∀ o, u'.mem r o = u.mem r o := by
  simp [exec, fused_recurrent_rwkv6_state_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  refine ⟨?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
    simp
  · intro r hr o
    rw [BlockState.foldl_writeMem_prop_masked_mem_preserve_other_region _ _ _ _ r hr o]
    simp

/-- Address congruence: `finalStateOffset` reads the state only through `pids`. -/
theorem finalStateOffset_congr (s u : BlockState) (K V BK BV : Nat)
    (h : u.pids = s.pids) (idx : TileIndex [BV, BK]) :
    finalStateOffset u K V BK BV idx = finalStateOffset s K V BK BV idx := by
  unfold finalStateOffset kIndex vIndex; rw [h]

/-- Mask congruence: `activeKV` reads the state only through `pids`. -/
theorem activeKV_congr (s u : BlockState) (K V BK BV : Nat)
    (h : u.pids = s.pids) (idx : TileIndex [BV, BK]) :
    activeKV u K V BK BV idx = activeKV s K V BK BV idx := by
  unfold activeKV active activeK kIndex vIndex; rw [h]

/-- `stateStepSpec` transported to a state `u` that agrees with the reference
state `s` on `pids` and on the three **input** regions it reads. The carry
region `C` is deliberately *not* transported: its content at `u` is exactly what
the fold's invariant supplies, and transporting it would beg the question.

This is obligation (2) of `CarryFold`'s docstring, and it is what forces the
`k ≠ C` / `v ≠ C` / `w ≠ C` side conditions on the fold below. -/
theorem stateStepSpec_transport
    (s u : BlockState) (C k v w : RegionName) (m s_k_h s_v_h K V BK BV : Nat)
    (hpids : u.pids = s.pids)
    (hk : ∀ o, u.readMem k o = s.readMem k o)
    (hv : ∀ o, u.readMem v o = s.readMem v o)
    (hw : ∀ o, u.readMem w o = s.readMem w o)
    (idx : TileIndex [BV, BK]) :
    stateStepSpec u C k v w m s_k_h s_v_h K V BK BV idx
      = u.readMem C (finalStateOffset s K V BK BV idx) *
            decay s w s_k_h K BK m idx.2.1
        + kVal s k s_k_h K BK m idx.2.1 * vVal s v s_v_h V BV m idx.1 := by
  unfold stateStepSpec decay kVal vVal finalStateOffset kIndex vIndex
  rw [hpids, hk, hv, hw]

/-- **Cross-step carry fold for the state recurrence (genuine).**

Running `T` state slices as a chain through the shared carry region `C`, the
final buffer holds the genuine closed form `stateClosed(T)` on every
write-active lane — and the chain leaves everything outside `C` untouched.

The point of the statement is what is *no longer* assumed: the step faces above
each assume `BHPrev = stateClosed(m)` at their own `m`, so a `T`-step story
needed `T` assumptions with nothing identifying one step's output buffer with
the next step's input buffer. Here there is **one** assumption, `hSeed`, about
the initial contents of `C`, and the identification is structural — the same
region name is both the slice's `BHPrev` and its `BHOut`.

Hypotheses, all necessary:

* `hkC` / `hvC` / `hwC` — the three input regions are distinct from the carry
  region. Without them a step could overwrite its own inputs and the closed
  form, which reads `s`, would stop describing what later steps compute.
* `hInj` — the state tile's addresses are injective, as in the step face.
* `hSeed` — the initial buffer holds the seed `b_h^(0)` on write-active lanes.
  On the remaining lanes nothing is claimed and nothing is needed: a masked step
  never writes them, so the invariant could not be maintained there anyway (this
  is obligation (3) of `CarryFold`'s docstring, and is why the fold is run at the
  subtype of write-active lanes).
* `hRun` — the chain runs to completion. Postcondition style, as everywhere in
  the correctness layer; nothing here proves that a stage terminates.

What this does **not** say: that the launched `fused_recurrent_rwkv6_fwd_surface`
computes this. That kernel keeps `b_h` in a register across its own
`Stmt.forRange`; the object folded here is the materialized-state program. -/
theorem fused_recurrent_rwkv6_state_carry_fold
    (C k v w h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (s_k_h s_v_h K V BK BV T : Nat) (s sFinal : BlockState)
    (hkC : k ≠ C) (hvC : v ≠ C) (hwC : w ≠ C)
    (hInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => finalStateOffset s K V BK BV idx))
    (hSeed : ∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
      s.readMem C (finalStateOffset s K V BK BV idx)
        = stateSeed s h0 USE_INITIAL_STATE K V BK BV idx)
    (hRun : execChain (foldStages
        (fun m => fused_recurrent_rwkv6_state_step_slice C k v w C
          m s_k_h s_v_h K V BK BV) T) s = some sFinal) :
    AgreeOutsideRegion C s sFinal ∧
    ∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
      sFinal.readMem C (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx := by
  have key := carryFold_execChain
    (ι := {idx : TileIndex [BV, BK] // activeKV s K V BK BV idx})
    (step := fun m => fused_recurrent_rwkv6_state_step_slice C k v w C
      m s_k_h s_v_h K V BK BV)
    (C := C)
    (addr := fun i => finalStateOffset s K V BK BV i.val)
    (val := fun m i =>
      stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m i.val)
    (n := T) (s := s) (sFinal := sFinal)
    (fun i => by
      -- base case: `stateClosed(0)` *is* the seed (empty product, empty sum)
      simpa [stateClosed] using hSeed i.val i.property)
    (fun m _hm t t' hAgree hInv hExec => by
      have hpidsR : t.resetRegs.pids = s.pids := by
        rw [BlockState.resetRegs_pids]; exact hAgree.pids
      obtain ⟨hp', hm'⟩ := fused_recurrent_rwkv6_state_step_frame C k v w
        m s_k_h s_v_h K V BK BV t.resetRegs t' hExec
      have hAgree' : AgreeOutsideRegion C s t' :=
        ⟨by rw [hp', BlockState.resetRegs_pids]; exact hAgree.pids,
         by
           intro r hr o
           rw [hm' r hr o, BlockState.resetRegs_mem]
           exact hAgree.mem r hr o⟩
      refine ⟨hAgree', ?_⟩
      intro i
      show t'.readMem C (finalStateOffset s K V BK BV i.val)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV (m + 1) i.val
      have hInjT : Function.Injective
          (fun idx : TileIndex [BV, BK] =>
            finalStateOffset t.resetRegs K V BK BV idx) := by
        intro a b hab
        exact hInj (by
          simpa [finalStateOffset_congr s t.resetRegs K V BK BV hpidsR] using hab)
      have hInvI : t.readMem C (finalStateOffset s K V BK BV i.val)
          = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m i.val :=
        hInv i
      have hcorr := fused_recurrent_rwkv6_state_step_slice_correct C k v w C
        m s_k_h s_v_h K V BK BV t.resetRegs hInjT i.val
      -- `simp only`, not `rw`: the mask sits under an `ite` whose `Decidable`
      -- instance mentions it, so a plain rewrite has no type-correct motive.
      simp only [hExec, Option.map_some, Option.some.injEq,
        finalStateOffset_congr s t.resetRegs K V BK BV hpidsR,
        activeKV_congr s t.resetRegs K V BK BV hpidsR, i.property, if_true] at hcorr
      rw [hcorr, stateStepSpec_transport s t.resetRegs C k v w
        m s_k_h s_v_h K V BK BV hpidsR
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem k hkC o)
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem v hvC o)
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem w hwC o)
        i.val]
      rw [BlockState.resetRegs_readMem, hInvI, stateClosed_succ])
    hRun
  exact ⟨key.1, fun idx hidx => key.2 ⟨idx, hidx⟩⟩

/-! ## Output-reduction step slice (the per-step output `o_t`)

This isolates the Python loop body's output computation. With the *pre-update*
state tile `BHPrev`, it loads `k_t/v_t/q_t/u`, forms
`(BHPrev + (k⊗v)·u)·(scale·q)`, reduces over the key axis (`tl.sum(_, axis=1)`),
and masked-stores the resulting `[BV]` row into `o` at time row `t`. -/

def fused_recurrent_rwkv6_output_step_slice
    (BHPrev q k v u o : RegionName)
    (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bv[:, None] & mask_bk[None, :]
  prev = tl.load(BHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0)
  b_v = tl.load(v + i_bh * $(s_v_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_k_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0) * $(scale)
  b_u = tl.load(u + i_h * $(K) + offs_k + i_k * $(BK), mask=mask_bk, other=0.0)
  b_kv = b_k[None, :] * b_v[:, None]
  b_o = (prev + b_kv * b_u[None, :]) * b_q[None, :]
  b_o = tl.sum(b_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_v_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (b_o).to(o.dtype.element_ty), mask=mask_bv)
}

theorem fused_recurrent_rwkv6_output_step_slice_toAlgorithm_supported
    (BHPrev q k v u o : RegionName)
    (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) :
    ∃ alg, (fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
      t s_k_h s_v_h B H T K V BK BV scale).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_rwkv6_output_step_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- The genuine per-lane output spec of one loop body, reading the *pre-update*
materialized state `BHPrev`:
`o_t[j_v] = Σ_{j_k ∈ mask_bk} (BHPrev[j_v,j_k] + k_t[j_k]·v_t[j_v]·u[j_k]) · scale·q_t[j_k]`.

The summand is **guarded by `mask_bk`** (`activeK`), exactly as Python is: on an
out-of-range key channel every operand of that column is loaded with `other=0`
(`prev` via `mask_kv`, `b_k`/`b_q`/`b_u` via `mask_bk`), so the column
contributes `(0 + 0·v·0)·0 = 0` to the `tl.sum(_, axis=1)`. Guarding the spec is
what makes this face true for **partial** key tiles, not only full ones. -/
noncomputable def outputStepSpec
    (s : BlockState) (BHPrev q k v u : RegionName)
    (t s_k_h s_v_h H K V BK BV : Nat) (scale : ℝ) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    if activeK s K BK jk then
      (s.readMem BHPrev (finalStateOffset s K V BK BV
          (TileShape.insertAxisIndex [BV, BK] 1
            (TileShape.insertAxisIndex [BV] 0 PUnit.unit jv) jk))
        + (kVal s k s_k_h K BK t jk * vVal s v s_v_h V BV t jv) *
            uVal s u H K BK jk)
      * qVal s q s_k_h K BK scale t jk
    else 0

/-- The masked output address at lane `j_v` for time row `t` — the kernel's exact
`o + (i_bh + i_k·B·H)·s_v_h + i_v·BV + j_v + t·V` layout. -/
def outStepOffset (s : BlockState) (t s_v_h B H V BV : Nat) (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * s_v_h + s.pids 0 * BV + jv.val + t * V

-- Raised over the pre-masking budget: the masked slice carries a guard on every
-- one of the five loads, so the executed register tower — and every defeq check
-- against it — is materially larger than the unmasked one used to be.
set_option maxHeartbeats 4000000 in
theorem fused_recurrent_rwkv6_output_step_slice_correct
    (BHPrev q k v u o : RegionName)
    (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s t s_v_h B H V BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := outStepOffset s t s_v_h B H V BV jv
      (exec (fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
            t s_k_h s_v_h B H T K V BK BV scale) s).map (·.readMem o outAddr)
        = some (if active s V BV jv then
            outputStepSpec s BHPrev q k v u t s_k_h s_v_h H K V BK BV scale jv
          else s.readMem o outAddr) := by
  intro jv
  -- The masked loads' `other = 0.0` default is an `OfScientific` literal; naming it
  -- as `(0 : ℝ)` once keeps the out-of-range branch inside `ring`'s reach.
  have hzero : (0.0 : ℝ) = 0 := by norm_num
  simp [exec, fused_recurrent_rwkv6_output_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, finalStateOffset, outStepOffset, active, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => (s.pids 2 + s.pids 1 * B * H) * s_v_h + s.pids 0 * BV + idx.1.val + t * V
  -- Collapse the guarded `WithBot ℝ` reduction the masked loads produce down to a
  -- plain real sum: `insertAxisIndex_one_rank2` normalizes the `tl.sum(_, axis=1)`
  -- lane projection, `ite_some_some` pushes each `other = 0.0` carrier's `some`
  -- outward, and the `@[simp]` `WithBot` helpers then finish the tower.
  simp only [TileShape.insertAxisIndex_one_rank2, MaskedReduction.ite_some_some,
    Option.map₂_some_some, WithBot.sum_someTerm_eq_some, WithBot.unbotD_some]
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      ∑ jk : Fin BK,
        ((if s.pids 0 * BV + idx.1.val < V ∧ s.pids 1 * BK + jk.val < K then
              s.readMem BHPrev
                (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V + (s.pids 0 * BV + idx.1.val))
            else 0.0)
          + (if s.pids 1 * BK + jk.val < K then
                s.readMem k (s.pids 2 * s_k_h + s.pids 1 * BK + jk.val + t * K)
              else 0.0) *
              (if s.pids 0 * BV + idx.1.val < V then
                  s.readMem v (s.pids 2 * s_v_h + s.pids 0 * BV + idx.1.val + t * V)
                else 0.0) *
            (if s.pids 1 * BK + jk.val < K then
                s.readMem u (s.pids 2 % H * K + jk.val + s.pids 1 * BK)
              else 0.0)) *
        ((if s.pids 1 * BK + jk.val < K then
              s.readMem q (s.pids 2 * s_k_h + s.pids 1 * BK + jk.val + t * K)
            else 0.0) * scale)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : outStepOffset s t s_v_h B H V BV a = outStepOffset s t s_v_h B H V BV b := by
      simpa [offsetFn, outStepOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem o (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem o (offsetFn (jv, PUnit.unit)) =
    if active s V BV jv then
      outputStepSpec s BHPrev q k v u t s_k_h s_v_h H K V BK BV scale jv
    else s.readMem o (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < V
  · simp only [P, hjv, if_true, active, vIndex]
    -- The two `insertAxisIndex` normalizations are what makes the specification's
    -- `tl.sum(_, axis = 1)` lane projection meet the executed one; without them the
    -- two sides differ by an unfolded `Fin`-literal projection `ring` cannot see
    -- through.
    simp only [valueFn, outputStepSpec, finalStateOffset, kVal, vVal, uVal, qVal,
      kIndex, vIndex, activeK, TileShape.insertAxisIndex_one_length,
      TileShape.insertAxisIndex_zero_length]
    apply Finset.sum_congr rfl
    intro jk _
    -- On an in-range key channel every guard is true and the two sides agree up to
    -- `ring`; on an out-of-range one Python's `other = 0` zeroes `prev`, `b_k`, `b_u`
    -- and `b_q`, so the column contributes `(0 + 0·v·0)·(0·scale) = 0` — which is
    -- exactly the guarded specification's `else 0`.
    by_cases hK : s.pids 1 * BK + jk.val < K
    · simp only [hK, hjv, and_self, if_true]
      ring
    · simp only [hK, if_false, and_false, hzero]
      ring
  · simp only [P, active, vIndex, hjv, if_false, BlockState.setReg_readMem]

theorem fused_recurrent_rwkv6_output_step_slice_compute_correct
    (BHPrev q k v u o : RegionName)
    (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s t s_v_h B H V BV jv)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
        t s_k_h s_v_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => active s V BV jv)
        (fun jv => (o, outStepOffset s t s_v_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        outputStepSpec s BHPrev q k v u t s_k_h s_v_h H K V BK BV scale jv) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_rwkv6_output_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jv hActive
  have h := fused_recurrent_rwkv6_output_step_slice_correct BHPrev q k v u o
    t s_k_h s_v_h B H T K V BK BV scale s hOutInj jv
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Output step realizes the genuine closed form.** When the materialized
pre-update state buffer `BHPrev` holds `stateClosed(m)`, the output step's
per-lane reduction `outputStepSpec` equals the genuine output `outputClosed(m)`
over the input regions. -/
theorem outputStepSpec_eq_outputClosed
    (s : BlockState) (BHPrev q k v w u h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_k_h s_v_h H K V BK BV : Nat) (scale : ℝ)
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem BHPrev (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx)
    (jv : Fin BV) :
    outputStepSpec s BHPrev q k v u m s_k_h s_v_h H K V BK BV scale jv
      = outputClosed s q k v w u h0 USE_INITIAL_STATE s_k_h s_v_h H K V BK BV scale m jv := by
  unfold outputStepSpec outputClosed
  apply Finset.sum_congr rfl
  intro jk _
  split_ifs with hK
  · rw [hPrev]
  · rfl

/-! ## Genuine closed-form step realizations (the carry-fold)

The cross-step fold over `range(0, T)` threading `b_h` is **not modeled**; each
*step face* is realized against the genuine closed form `stateClosed` /
`outputClosed` over the input regions `q,k,v,w,u,h0`, under an *assumed* carry
invariant on the materialized previous-state buffer.

* one state-update body realizes `stateClosed(m+1)` given `BHPrev = stateClosed(m)`;
* one output body realizes `outputClosed(m)` given `BHPrev = stateClosed(m)`. -/

/-- **Genuine state carry-fold step.** One loop body, with the materialized
pre-update state buffer `BHPrev = stateClosed(m)`, realizes the genuine
`(m+1)`-step folded state `stateClosed(m+1)` into the state buffer. -/
theorem fused_recurrent_rwkv6_state_step_closed_form
    (BHPrev k v w h0 BHOut : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_k_h s_v_h K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => finalStateOffset s K V BK BV idx))
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem BHPrev (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
        m s_k_h s_v_h K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BV, BK] => activeKV s K V BK BV idx)
        (fun idx => (BHOut, finalStateOffset s K V BK BV idx)))
      (expected := fun idx =>
        stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV (m + 1) idx) := by
  have h := fused_recurrent_rwkv6_state_step_slice_compute_correct BHPrev k v w
    BHOut m s_k_h s_v_h K V BK BV s hOutInj
  have hcong : (fun idx : TileIndex [BV, BK] =>
      stateStepSpec s BHPrev k v w m s_k_h s_v_h K V BK BV idx)
      = (fun idx : TileIndex [BV, BK] =>
        stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV (m + 1) idx) := by
    funext idx
    exact stateStepSpec_eq_stateClosed_succ s BHPrev k v w h0 USE_INITIAL_STATE
      m s_k_h s_v_h K V BK BV hPrev idx
  rwa [hcong] at h

/-- **Genuine output step.** One output body, with the materialized pre-update
state buffer `BHPrev = stateClosed(m)`, realizes the genuine output closed form
`outputClosed(m)` (the reduction over key channels) into the output buffer. -/
theorem fused_recurrent_rwkv6_output_step_closed_form
    (BHPrev q k v w u h0 o : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outStepOffset s m s_v_h B H V BV jv))
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem BHPrev (finalStateOffset s K V BK BV idx)
        = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
        m s_k_h s_v_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => active s V BV jv)
        (fun jv => (o, outStepOffset s m s_v_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        outputClosed s q k v w u h0 USE_INITIAL_STATE s_k_h s_v_h H K V BK BV scale m jv) := by
  have h := fused_recurrent_rwkv6_output_step_slice_compute_correct BHPrev q k v u o
    m s_k_h s_v_h B H T K V BK BV scale s hOutInj
  have hcong : (fun jv : Fin BV =>
      outputStepSpec s BHPrev q k v u m s_k_h s_v_h H K V BK BV scale jv)
      = (fun jv : Fin BV =>
        outputClosed s q k v w u h0 USE_INITIAL_STATE s_k_h s_v_h H K V BK BV scale m jv) := by
    funext jv
    exact outputStepSpec_eq_outputClosed s BHPrev q k v w u h0 USE_INITIAL_STATE
      m s_k_h s_v_h H K V BK BV scale hPrev jv
  rwa [hcong] at h

/-! ## Offset-injectivity side conditions (dimension-general + Python shape)

The flattened state address `i_bh·K·V + j_k·V + j_v` is injective on the
`[BV, BK]` tile whenever the tile fits the logical extents (`BV ≤ V`, `BK ≤ K`):
then `j_v` occupies the low `V`-digit and `j_k·V` the next, a faithful mixed-radix
encoding. The per-time output address `(i_bh + i_k·B·H)·s_v_h + i_v·BV + j_v + t·V`
is injective in `j_v` whenever `0 < BV` (lanes are contiguous). These are honest
structural side conditions; the Python regression (`K=V=BK=BV=8`) satisfies them. -/

/-- **Dimension-general** state-tile / `h0` address injectivity
(row-major `i_bh·K·V + j_k·V + j_v`), given the tile fits (`BV ≤ V`, `BK ≤ K`). -/
theorem fused_recurrent_rwkv6_final_state_offset_injective_general
    (s : BlockState) (K V BK BV : Nat) (hBV : BV ≤ V) (hBK : BK ≤ K) :
    Function.Injective (fun idx : TileIndex [BV, BK] => finalStateOffset s K V BK BV idx) := by
  rintro ⟨⟨av, hav⟩, ⟨ak, hak⟩, _⟩ ⟨⟨bv, hbv⟩, ⟨bk, hbk⟩, _⟩ h
  simp [finalStateOffset, vIndex, kIndex] at h
  have havV : av < V := lt_of_lt_of_le hav hBV
  have hbvV : bv < V := lt_of_lt_of_le hbv hBV
  have hVpos : 0 < V := lt_of_le_of_lt (Nat.zero_le _) havV
  -- Reduce to the clean mixed-radix equality `Aₖ·V + av = Bₖ·V + bv` (with the
  -- common `pids2·K·V` / `pids0·BV` digits cancelled) where `av, bv < V`.
  have hcore : (s.pids 1 * BK + ak) * V + av = (s.pids 1 * BK + bk) * V + bv := by
    omega
  -- Take the equation mod `V`: the low digit is forced equal.
  have hv : av = bv := by
    have ha : ((s.pids 1 * BK + ak) * V + av) % V = av % V := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hb : ((s.pids 1 * BK + bk) * V + bv) % V = bv % V := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hmod : ((s.pids 1 * BK + ak) * V + av) % V
        = ((s.pids 1 * BK + bk) * V + bv) % V := by rw [hcore]
    rwa [ha, hb, Nat.mod_eq_of_lt havV, Nat.mod_eq_of_lt hbvV] at hmod
  have hk : ak = bk := by
    subst hv
    have hmul : (s.pids 1 * BK + ak) * V = (s.pids 1 * BK + bk) * V := by omega
    have := Nat.eq_of_mul_eq_mul_right hVpos hmul
    omega
  subst bv; subst bk; rfl

/-- **Dimension-general** per-time output address injectivity, given `0 < BV`. -/
theorem fused_recurrent_rwkv6_out_step_offset_injective_general
    (s : BlockState) (t s_v_h B H V BV : Nat) (hBV : 0 < BV) :
    Function.Injective (fun jv : Fin BV => outStepOffset s t s_v_h B H V BV jv) := by
  intro a b h
  simp [outStepOffset] at h
  omega

/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about two hand-cut single-step slices, not about the
launched kernel.** Clauses 2 and 3 are `Realizes` facts about
`fused_recurrent_rwkv6_output_step_slice` and
`fused_recurrent_rwkv6_state_step_slice`; the launched surface
`fused_recurrent_rwkv6_fwd_surface` appears only in clause 1, which says nothing
more than "it lowers to the algorithm layer". The `range(0, T)` fold, the
`STORE_FINAL_STATE` writeback, and the host-side `o.sum(0)` over `NK` are all
outside this theorem (see the module docstring's modeling boundary).

Parameterized over the symbolic head strides `s_k_h s_v_h`, batch/head/time
`B H T`, key/value extents `K V`, tile sizes `BK BV`, the real `scale`, the step
index `m`, and both flags `USE_INITIAL_STATE STORE_FINAL_STATE`. The two step
faces are realized against the closed forms `stateClosed` / `outputClosed` over
the *input* regions (never a read-back of the kernel's own output):

1. the full RWKV6 forward surface lowers to the algorithm layer;
2. one **output** body realizes `outputClosed(m)` (the key-axis reduction),
   given the *assumed* carry invariant `BHPrev = stateClosed(m)` — **masked
   faithfully** (`mask_kv` on `prev`, `mask_bk` on `b_k`/`b_q`/`b_u`, `mask_bv`
   on `b_v` and on the store), with the key-axis summand correspondingly
   `activeK`-guarded, so this clause holds for partial tiles too;
3. one **state-update** body realizes `stateClosed(m+1)` (the per-channel decay
   step), given the same assumed carry invariant — **masked faithfully**
   (`mask_bk`/`mask_bv`/`mask_kv` loads and the `mask_kv` store), as a
   `writeIf (activeKV …)` statement about the in-range lanes;
4. the **cross-step carry fold**: chaining `T` state slices through one shared
   carry region `C` reaches `stateClosed(T)`, and leaves everything outside `C`
   untouched. This clause assumes *no* pinned carry — only that the initial
   buffer holds the seed on write-active lanes.

The pinned carry `BHPrev = stateClosed(m)` is a **clause-local antecedent** of
clauses 2 and 3 rather than a hypothesis of the theorem, precisely so that it
cannot weaken clause 4, whose whole point is not needing it.

Side conditions, all honest and all necessary:

* `BK ≤ K`, `BV ≤ V` — the tile fits the logical extents, giving offset
  injectivity for the state face and for the fold;
* `0 < BV` — contiguous output lanes, giving injectivity for the output face;
* inside clause 4 only: `k ≠ C`, `v ≠ C`, `w ≠ C` (a step must not overwrite its
  own inputs, or the closed form — which reads the *initial* state — would stop
  describing what later steps compute), the seed assumption on the initial
  buffer, and that the chain runs to completion (postcondition style: nothing
  here proves a stage terminates).

No clause carries a full-tile hypothesis: both slices mask exactly where Python
masks, and both `expected` closed forms are guarded to match. -/
specification fused_recurrent_rwkv6_output_summary_general
    (q k v w u o h0 ht BHPrev BHOut C : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s_k_h s_v_h B H T K V BK BV m : Nat) (scale : ℝ) (s sFinal : BlockState)
    (hBV : BV ≤ V) (hBK : BK ≤ K) (hBVpos : 0 < BV) :
    -- (1) the full surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_rwkv6_fwd_surface q k v w u o h0 ht
      s_k_h s_v_h B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    -- (2) the output body realizes the genuine `outputClosed(m)` — mask-faithful,
    --     so this holds for partial key/value tiles as well. The pinned carry is a
    --     clause-local antecedent, so it cannot weaken clause 4.
    ((∀ idx : TileIndex [BV, BK],
        s.readMem BHPrev (finalStateOffset s K V BK BV idx)
          = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o
          m s_k_h s_v_h B H T K V BK BV scale)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun jv : Fin BV => active s V BV jv)
          (fun jv => (o, outStepOffset s m s_v_h B H V BV jv)))
        (expected := fun jv : Fin BV =>
          outputClosed s q k v w u h0 USE_INITIAL_STATE s_k_h s_v_h H K V BK BV
            scale m jv)) ∧
    -- (3) the state-update body realizes the genuine `stateClosed(m+1)`, under the
    --     same clause-local pinned carry
    ((∀ idx : TileIndex [BV, BK],
        s.readMem BHPrev (finalStateOffset s K V BK BV idx)
          = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV m idx) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut
          m s_k_h s_v_h K V BK BV)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun idx : TileIndex [BV, BK] => activeKV s K V BK BV idx)
          (fun idx => (BHOut, finalStateOffset s K V BK BV idx)))
        (expected := fun idx =>
          stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV
            (m + 1) idx)) ∧
    -- (4) the **cross-step carry fold**: chaining `T` state slices through one
    --     shared carry region `C` reaches `stateClosed(T)` from a *single*
    --     assumption about the initial buffer — no per-step pinned carry at all.
    --     Its own antecedents keep clauses 2 and 3 free of them.
    (k ≠ C → v ≠ C → w ≠ C →
      (∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
        s.readMem C (finalStateOffset s K V BK BV idx)
          = stateSeed s h0 USE_INITIAL_STATE K V BK BV idx) →
      execChain (foldStages
        (fun j => fused_recurrent_rwkv6_state_step_slice C k v w C
          j s_k_h s_v_h K V BK BV) T) s = some sFinal →
      (AgreeOutsideRegion C s sFinal ∧
        ∀ idx : TileIndex [BV, BK], activeKV s K V BK BV idx →
          sFinal.readMem C (finalStateOffset s K V BK BV idx)
            = stateClosed s k v w h0 USE_INITIAL_STATE s_k_h s_v_h K V BK BV T idx)) := by
  have hStateInj := fused_recurrent_rwkv6_final_state_offset_injective_general
    s K V BK BV hBV hBK
  have hOutInj := fused_recurrent_rwkv6_out_step_offset_injective_general
    s m s_v_h B H V BV hBVpos
  refine ⟨fused_recurrent_rwkv6_fwd_surface_toAlgorithm_supported _ _ _ _ _ _ _ _
      _ _ _ _ _ _ _ _ _ _ _ _ _, ?_, ?_, ?_⟩
  · intro hPrev
    exact fused_recurrent_rwkv6_output_step_closed_form BHPrev q k v w u h0 o
      USE_INITIAL_STATE m s_k_h s_v_h B H T K V BK BV scale s hOutInj hPrev
  · intro hPrev
    exact fused_recurrent_rwkv6_state_step_closed_form BHPrev k v w h0 BHOut
      USE_INITIAL_STATE m s_k_h s_v_h K V BK BV s hStateInj hPrev
  · intro hkC hvC hwC hSeed hRun
    exact fused_recurrent_rwkv6_state_carry_fold C k v w h0 USE_INITIAL_STATE
      s_k_h s_v_h K V BK BV T s sFinal hkC hvC hwC hStateInj hSeed hRun

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel

