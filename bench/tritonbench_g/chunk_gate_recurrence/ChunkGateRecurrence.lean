import VeriTile.Triton

/-!
# `chunk_gate_recurrence` — strict per-kernel correctness

`chunk_gate_recurrence.py` implements a gated chunk recurrence for linear
attention. `_fwd_recurrence` carries a `[BLOCK_MODEL_K, BLOCK_MODEL_V]`
state `acc` across `NUM_BLOCK` chunks, seeded from `last_kv` (or zero), and at
each step updates `acc = acc * d_i + S_i`, storing `acc` into `O`.
`_bwd_recurrence` runs the matching reverse recurrence
`Dacc = Dacc * d_i + DS_i`, emitting per-chunk gradients `DI`, `DG`
(`sum(Dacc * S_i)`), and the boundary gradient `DL`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies of both `_fwd_recurrence` and `_bwd_recurrence`. The host
launch (`grid = (B*H, D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)`, the fixed
`BLOCK_MODEL_K = 64`, `BLOCK_MODEL_V = 16`, the divisibility assertions, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because the program ids are universally
quantified, the per-program statements cover every program of the grid.

## Proof architecture

The genuine forward closed-form spec is `fwdClosed`, a standalone
`seed · ∏ d + Σ S · ∏ d` over the *input* regions `S`, `D`, `last_kv` — never a
read-back of the kernel's own output `O`. The genuine **reverse** closed-form
spec is `bwdClosed a = Σ_{a ≤ u < NUM_BLOCK} DS_u · ∏_{a ≤ w < u} d_w`, a
standalone form over the *input* regions `DS` (the incoming `DO`) and `D` —
likewise never a read-back of `DI` / `DG` / `DL`.

```
chunk_gate_recurrence_output_summary_general                  ← TOP (bundled)
  ├─ chunk_gate_recurrence_fwd_surface_toAlgorithm_supported    fwd surface lowers
  ├─ chunk_gate_recurrence_bwd_surface_toAlgorithm_supported    bwd surface lowers
  ├─ chunk_gate_recurrence_fwd_initial_last_kv_closed_form_general  (fwdClosed 0 = last_kv)
  ├─ chunk_gate_recurrence_fwd_initial_zero_closed_form_general     (fwdClosed 0 = 0)
  ├─ chunk_gate_recurrence_forward_step_store_slice_closed_form (fwd carry-fold step)
  │    └─ forwardStepSpec_eq_fwdClosed_succ
  │         └─ fwdClosed_succ   (fwdClosed(m+1) = fwdClosed(m)·d_m + S_m)
  ├─ chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_closed_form
  │    │                          (DI chunk t = bwdClosed(t+1))
  │    └─ bwdDaccStepSpec_eq_bwdClosed
  │         └─ bwdCarryStep_eq_bwdClosed
  │              └─ bwdClosed_step  (bwdClosed(a) = bwdClosed(a+1)·d_a + DS_a)
  ├─ chunk_gate_recurrence_bwd_dg_step_store_slice_closed_form
  │                                (DG chunk t = Σ bwdClosed(t+1)·S_t)
  └─ chunk_gate_recurrence_bwd_DL_store_slice_closed_form
                                   (DL = bwdClosed(0), the post-loop step)

per-store slice lemmas (modeled exactly, fed materialized state buffers):
  forward:  forward_store_slice / initial_last_kv_store_slice /
            initial_zero_store_slice / forward_step_store_slice
  backward: bwd_dacc_step_DI_store_slice / bwd_dg_step_store_slice /
            bwd_DL_store_slice
each with a `*_correct` (algorithm-layer readback) and
`*_compute_correct` (ComputeCorrect) face (except `initial_last_kv_store_slice`,
which has only a `*_compute_correct` face).
```

**Two-index backward addressing.** `_bwd_recurrence` initialises `S`/`DI`/`DG` at
chunk `NUM_BLOCK - 2` but `DS`/`d` at `NUM_BLOCK - 1` (`chunk_gate_recurrence.py`
lines 60-69) and decrements all five in lockstep, so *every* iteration reads
`DS`/`d` one chunk **ahead** of the `S` it multiplies and the `DI`/`DG` it writes.
The backward step slices therefore carry both indices — `t_rel` for `S`/`DI`/`DG`
and `t_rel + 1` for `DS`/`D` — making that offset relation structural rather than
assumed. (An earlier revision of this file collapsed all five pointers onto one
shared index, which reproduced no real iteration at any value of `t_rel`.)

**The `DL` boundary-gradient store now has a face.** Python's post-loop
(lines 90-94) is `Dacc = Dacc * d_i + DS_i; tl.store(DL, Dacc)` at the pointer
positions the loop's `NUM_BLOCK - 1` decrements leave behind — `DS`/`d` at chunk
`0`. `chunk_gate_recurrence_bwd_DL_store_slice` models that multiply-add (an
earlier revision dropped it and stored a bare copy of the carry, i.e. a memcpy
compared against its own source over a fiction region).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The fixed block sizes
`BLOCK_MODEL_K = 64`, `BLOCK_MODEL_V = 16` are the Python defaults; the
`.to(tl.float32)` / `.to(_.dtype.element_ty)` casts erase to the identity at the
algorithm layer (post-erasure all dtypes unify to `ℝ`). Each single recurrence
step face — the gated update `acc * d_i + S_i` (forward) and
`Dacc * d_i + DS_i` plus the `sum(Dacc * S_i)` reduction (backward) — and each
masked block store are modeled exactly. The cross-chunk recurrence fold — the
forward `range(NUM_BLOCK-1)` loop threading `acc`, and the reverse
`range(NUM_BLOCK-1)` loop threading `Dacc` from the last chunk — is the trusted
boundary: the carried register tile is presented to each step slice as a
materialized previous-state buffer (`AccPrev` forward; `DaccPrev` for the reverse
loop body and `DaccTail` for the post-loop `DL` step), laid out as a *single*
scratch tile at `accOffset`, matching the fact that `acc`/`Dacc` are one register
tile reused across iterations rather than a chunk-indexed array.

Under the *assumed* carry invariant `AccPrev = fwdClosed(m)`, the forward loop
body realizes the genuine `(m+1)`-step folded closed form `fwdClosed(m+1)`
(theorem `forwardStepSpec_eq_fwdClosed_succ`, algebraic core `fwdClosed_succ`).
Symmetrically, under `DaccPrev = bwdClosed(t_rel+2)` the reverse loop body
realizes `bwdClosed(t_rel+1)` into `DI` chunk `t_rel` and
`Σ bwdClosed(t_rel+1)·S_{t_rel}` into `DG` chunk `t_rel` (theorem
`bwdDaccStepSpec_eq_bwdClosed`, algebraic core `bwdClosed_step`), and under
`DaccTail = bwdClosed(1)` the post-loop step realizes `bwdClosed(0)` into `DL`.
So both recurrences are closed per-statement against standalone specifications
over Python inputs; only the *pin* on the carry buffer is assumed. The carry pins
are satisfiable (hence the statements are not vacuous) precisely because the
carry regions are separate `RegionName`s from `S`/`D`/`DS`, so a state can always
be built by materializing the required tile after fixing the inputs; region
distinctness is therefore not needed as a hypothesis. The unconditional
arithmetic faces `bwdDaccStepSpec` (`Dacc·d_i + DS_i`) and `bwdDGStepSpec`
(`Σ Dacc·S_i`) remain available as the `*_compute_correct` lemmas, with no carry
hypothesis at all. There is no `@triton.autotune` on these kernels. Output offset
injectivity / non-collision is a hypothesis of the headline, not a lemma.

Region distinctness: no `≠` hypotheses are stated and none are needed — every
slice performs all of its loads before its single store and every `expected` is a
function of the *initial* state, so aliasing cannot falsify a face.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DTileKernelIO₁

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `chunk_gate_recurrence_output_summary_general` — one
bundled headline over both Python kernels: shape-general, but scoped to hand-cut
store/step **slices**, not to a launched kernel. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `chunk_gate_recurrence.py`'s `_fwd_recurrence`.

The optional `last_kv` argument is represented by `HAS_LAST_KV`. The backward
kernel walks the chunk dimension in reverse with pointer decrements, so it is
kept separate from this forward surface. -/
def chunk_gate_recurrence_fwd_surface
    (S D O last_kv : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (HAS_LAST_KV : Bool) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)

  S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
    offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
  O = O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
    offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
  if HAS_LAST_KV {
    last_kv = last_kv + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
      offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
    acc = tl.load(last_kv).to(tl.float32)
  } else {
    acc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
  }
  tl.store(O, (acc).to(O.dtype.element_ty))
  O += $(D_MODEL_K) * $(D_MODEL_V)
  D = D + offset_bh * $(NUM_BLOCK)
  for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
    d_i = tl.load(D)
    S_i = tl.load(S)
    acc = acc * d_i + S_i
    tl.store(O, (acc).to(O.dtype.element_ty))
    D += $(1)
    S += $(D_MODEL_K) * $(D_MODEL_V)
    O += $(D_MODEL_K) * $(D_MODEL_V)
  }
}

/-- The forward chunk-gate recurrence surface lowers to the algorithm layer,
including optional `last_kv`, recurrent decay/update, and output stores. -/
theorem chunk_gate_recurrence_fwd_surface_toAlgorithm_supported
    (S D O last_kv : RegionName)
    (NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (HAS_LAST_KV : Bool) :
    ∃ alg, (chunk_gate_recurrence_fwd_surface S D O last_kv NUM_HEAD NUM_BLOCK
      D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V HAS_LAST_KV).toAlgorithm?
        = Except.ok alg := by
  simp [chunk_gate_recurrence_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `chunk_gate_recurrence.py`'s `_bwd_recurrence`.

The Python backward kernel starts from the last/penultimate chunk positions,
walks the chunk axis backward by decrementing pointers inside a forward
`range(NUM_BLOCK - 1)` loop, writes `DG`/`DI` for intermediate chunks, and
finally writes `DL` from the accumulated state. -/
def chunk_gate_recurrence_bwd_surface
    (S D DI DG DL DS : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  NUM_K = $(D_MODEL_K) // $(BLOCK_MODEL_K)
  NUM_V = $(D_MODEL_V) // $(BLOCK_MODEL_V)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
  DI = DI + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
  DS = DS + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :] + ($(NUM_BLOCK) - $(1)) * $(D_MODEL_K) * $(D_MODEL_V)
  DG = DG + offset_bh * $(NUM_BLOCK) * NUM_K * NUM_V +
    offset_d * NUM_V + offset_s + ($(NUM_BLOCK) - $(2)) * NUM_K * NUM_V
  D = D + offset_bh * $(NUM_BLOCK) + ($(NUM_BLOCK) - $(1))
  Dacc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
  for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
    S_i = tl.load(S)
    DS_i = tl.load(DS)
    d_i = tl.load(D)
    Dacc = Dacc * d_i + DS_i
    DG_i = tl.sum(Dacc * (S_i).to(tl.float32))
    tl.store(DG, (DG_i).to(DG.dtype.element_ty))
    tl.store(DI, (Dacc).to(DI.dtype.element_ty))
    S -= $(D_MODEL_K) * $(D_MODEL_V)
    DI -= $(D_MODEL_K) * $(D_MODEL_V)
    DS -= $(D_MODEL_K) * $(D_MODEL_V)
    DG -= NUM_K * NUM_V
    D -= $(1)
  }
  DL = DL + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :]
  DS_i = tl.load(DS)
  d_i = tl.load(D)
  Dacc = Dacc * d_i + DS_i
  tl.store(DL, (Dacc).to(DL.dtype.element_ty))
}

/-- The backward chunk-gate surface lowers with the Python pointer-decrement
loop and final `DL` store preserved. -/
theorem chunk_gate_recurrence_bwd_surface_toAlgorithm_supported
    (S D DI DG DL DS : RegionName)
    (NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    (chunk_gate_recurrence_bwd_surface S D DI DG DL DS NUM_HEAD NUM_BLOCK
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V).toAlgorithm? =
      Except.ok
        (chunk_gate_recurrence_bwd_surface S D DI DG DL DS NUM_HEAD NUM_BLOCK
          D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V).toAlgKernel := by
  simp [chunk_gate_recurrence_bwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented forward recurrence tile-store slice of
`chunk_gate_recurrence.py`'s `_fwd_recurrence`.

The full kernel repeatedly advances `O` by one KV block and stores the running
`acc`. This slice models one such tile store from a precomputed `Acc` tile into
`O`, preserving the source offset decomposition over `(offset_bh, offset_d,
offset_s)`. -/
def chunk_gate_recurrence_forward_store_slice
    (Acc O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.load(Acc + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], (acc).to(O.dtype.element_ty))
}

def kIndex (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  idx.1.val

def vIndex (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  idx.2.1.val

def accOffset
    (s : BlockState) (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    kIndex idx * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + vIndex idx

def outOffset
    (s : BlockState) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    kIndex idx * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + vIndex idx

def timeTileOffset
    (s : BlockState)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
    t_rel * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    s.pids 2 * BLOCK_MODEL_V + kIndex idx * D_MODEL_V + vIndex idx

theorem chunk_gate_recurrence_forward_store_slice_correct
    (Acc O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      let outAddr := outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx
      (exec (chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK
            D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s).map
          (·.readMem O outAddr)
        = some (s.readMem Acc
            (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) := by
  intro idx
  simp [exec, chunk_gate_recurrence_forward_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, kIndex, vIndex,
        accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → Nat :=
    fun idx =>
      s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
        s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
        idx.1.val * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val
  let valueFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → ℝ :=
    fun idx =>
      s.readMem Acc
        (s.pids 0 * D_MODEL_K * D_MODEL_V +
          s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
          idx.1.val * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem O (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_MODEL_K, BLOCK_MODEL_V])).readMem O
        (offsetFn idx) =
    s.readMem Acc
      (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, accOffset, kIndex, vIndex]

theorem chunk_gate_recurrence_forward_store_slice_compute_correct
    (Acc O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        s.readMem Acc
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_forward_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gate_recurrence_forward_store_slice_correct Acc O NUM_BLOCK
    D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Initial forward output store for the `last_kv is not None` Python branch. -/
def chunk_gate_recurrence_initial_last_kv_store_slice
    (LastKv O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.load(LastKv + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :])
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], acc)
}

theorem chunk_gate_recurrence_initial_last_kv_store_slice_compute_correct
    (LastKv O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_last_kv_store_slice LastKv O
        NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx =>
        s.readMem LastKv
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) := by
  exact chunk_gate_recurrence_forward_store_slice_compute_correct LastKv O
    NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hOutInj

/-- Initial forward output store for the `last_kv is None` Python branch.

The source kernel initializes `acc` with a zero tile and immediately stores that
tile into the first output chunk before entering the recurrence loop. -/
def chunk_gate_recurrence_initial_zero_store_slice
    (O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], acc)
}

theorem chunk_gate_recurrence_initial_zero_store_slice_correct
    (O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      let outAddr := outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx
      (exec (chunk_gate_recurrence_initial_zero_store_slice O NUM_BLOCK
            D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s).map
          (·.readMem O outAddr)
        = some (0.0 : ℝ) := by
  intro idx
  simp [exec, chunk_gate_recurrence_initial_zero_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        outOffset, kIndex, vIndex, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
          s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
          idx.1.val * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val) := by
    simpa [outOffset, kIndex, vIndex] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj idx]
  norm_num

theorem chunk_gate_recurrence_initial_zero_store_slice_compute_correct
    (O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_zero_store_slice O NUM_BLOCK
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun _ : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        (0.0 : ℝ)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_initial_zero_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gate_recurrence_initial_zero_store_slice_correct O NUM_BLOCK
    D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- One forward recurrence step:
`acc = acc * d_i + S_i`, then store the updated accumulator into the next output
chunk. This isolates the Python loop body arithmetic from the full loop
induction. -/
def chunk_gate_recurrence_forward_step_store_slice
    (AccPrev S D O : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  prev = tl.load(AccPrev + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
  s_i = tl.load(S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  acc = prev * d_i + s_i
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      $(t_rel + 1) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], acc)
}

def dOffset (s : BlockState) (t_rel NUM_BLOCK : Nat) : Nat :=
  s.pids 0 * NUM_BLOCK + t_rel

def forwardStepTileOffset
    (s : BlockState)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
    t_rel * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    kIndex idx * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + vIndex idx

noncomputable def forwardStepSpec
    (s : BlockState) (AccPrev S D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  s.readMem AccPrev
      (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) *
    s.readMem D (dOffset s t_rel NUM_BLOCK) +
  s.readMem S
      (forwardStepTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx)

/-! ## Genuine forward closed form (the gated-recurrence fold)

`chunk_gate_recurrence.py`'s `_fwd_recurrence` seeds `acc` from `last_kv` (or
zero), stores it into output chunk `0`, and then for each chunk `t` runs the
gated update `acc = acc * d_t + S_t` and stores the new `acc` into output chunk
`t+1`. The scalar gate `d_t = D[offset_bh·NUM_BLOCK + t]` broadcasts over the
whole `[BLOCK_MODEL_K, BLOCK_MODEL_V]` state tile.

Unrolling the recurrence gives the **genuine closed form** for output chunk `m`
at tile element `idx`:

```
O[m][idx] = seed[idx] · ∏_{j<m} d_j  +  Σ_{t<m} S_t[idx] · ∏_{t<j<m} d_j
```

where `seed[idx] = last_kv[idx]` if `HAS_LAST_KV` else `0`. This is a standalone
spec over the *input* regions `S`, `D`, `last_kv` — never a read-back of the
kernel's own output `O`. (No docstring line may begin at column 0 with a
declaration keyword such as `theorem` or `specification`: `scripts/spec_sheet.py`
matches those at column 0 and would report a phantom headline.)

`fwdGate s D NUM_BLOCK j := D[offset_bh·NUM_BLOCK + j]` is the scalar gate at
chunk `j`; `fwdSeed` is the seeded initial state; `fwdClosed` is the closed form
above. -/

noncomputable def fwdGate
    (s : BlockState) (D : RegionName) (NUM_BLOCK j : Nat) : ℝ :=
  s.readMem D (dOffset s j NUM_BLOCK)

noncomputable def fwdSeed
    (s : BlockState) (LastKv : RegionName) (HAS_LAST_KV : Bool)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  if HAS_LAST_KV then
    s.readMem LastKv
      (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
  else 0

/-- Genuine closed form for forward output chunk `m`:
`seed · ∏_{j<m} d_j + Σ_{t<m} S_t · ∏_{t<j<m} d_j`. -/
noncomputable def fwdClosed
    (s : BlockState) (S D LastKv : RegionName) (HAS_LAST_KV : Bool)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V m : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  fwdSeed s LastKv HAS_LAST_KV D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx *
      (∏ j ∈ Finset.range m, fwdGate s D NUM_BLOCK j) +
    ∑ t ∈ Finset.range m,
      s.readMem S
          (forwardStepTileOffset s t NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V idx) *
        (∏ j ∈ Finset.Ico (t + 1) m, fwdGate s D NUM_BLOCK j)

/-- `fwdClosed` at `m = 0` is the seed (the initial store of `acc`). -/
theorem fwdClosed_zero
    (s : BlockState) (S D LastKv : RegionName) (HAS_LAST_KV : Bool)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) :
    fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V 0 idx
      = fwdSeed s LastKv HAS_LAST_KV D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx := by
  simp [fwdClosed]

/-- **The forward carry-fold recurrence.** Unrolling one chunk:
`fwdClosed(m+1) = fwdClosed(m) · d_m + S_m`. This is the exact closed-form
counterpart of the Python loop body `acc = acc * d_i + S_i`. -/
theorem fwdClosed_succ
    (s : BlockState) (S D LastKv : RegionName) (HAS_LAST_KV : Bool)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V m : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) :
    fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V (m + 1) idx
      = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V m idx *
          fwdGate s D NUM_BLOCK m
        + s.readMem S
            (forwardStepTileOffset s m NUM_BLOCK D_MODEL_K D_MODEL_V
              BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  unfold fwdClosed
  rw [Finset.prod_range_succ, Finset.sum_range_succ]
  -- the `t = m` summand has an empty `Ico (m+1) (m+1)` product = 1.
  rw [show Finset.Ico (m + 1) (m + 1) = (∅ : Finset Nat) from by simp,
      Finset.prod_empty, mul_one]
  -- split the last gate `d_m` out of each remaining `∏_{t<j<m+1} d_j` product.
  have hsum :
      (∑ t ∈ Finset.range m,
          s.readMem S
              (forwardStepTileOffset s t NUM_BLOCK D_MODEL_K D_MODEL_V
                BLOCK_MODEL_K BLOCK_MODEL_V idx) *
            ∏ j ∈ Finset.Ico (t + 1) (m + 1), fwdGate s D NUM_BLOCK j)
        = (∑ t ∈ Finset.range m,
            s.readMem S
                (forwardStepTileOffset s t NUM_BLOCK D_MODEL_K D_MODEL_V
                  BLOCK_MODEL_K BLOCK_MODEL_V idx) *
              ∏ j ∈ Finset.Ico (t + 1) m, fwdGate s D NUM_BLOCK j)
          * fwdGate s D NUM_BLOCK m := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t ht
    simp only [Finset.mem_range] at ht
    rw [Finset.prod_Ico_succ_top (by omega : t + 1 ≤ m)]
    ring
  rw [hsum]
  ring

/-- **Carry-fold step (genuine).** If the materialized previous-state buffer
`AccPrev` holds the genuine `m`-step folded state `fwdClosed(m)` (at the
canonical `accOffset` layout), then one loop body — `forwardStepSpec`, i.e.
`AccPrev · d_m + S_m` — produces exactly the genuine `(m+1)`-step folded state
`fwdClosed(m+1)`. This is the inductive step of the forward recurrence threaded
by `acc`. -/
theorem forwardStepSpec_eq_fwdClosed_succ
    (s : BlockState) (AccPrev S D LastKv : RegionName) (HAS_LAST_KV : Bool)
    (m NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V m idx)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) :
    forwardStepSpec s AccPrev S D m NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx
      = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V (m + 1) idx := by
  rw [fwdClosed_succ]
  unfold forwardStepSpec
  rw [hAcc idx]
  rfl

theorem chunk_gate_recurrence_forward_step_store_slice_correct
    (AccPrev S D O : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      let outAddr := forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K
        D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx
      (exec (chunk_gate_recurrence_forward_step_store_slice AccPrev S D O
            t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s).map
          (·.readMem O outAddr)
        = some (forwardStepSpec s AccPrev S D t_rel NUM_BLOCK D_MODEL_K
            D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  intro idx
  simp [exec, chunk_gate_recurrence_forward_step_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        accOffset, forwardStepTileOffset, dOffset, kIndex, vIndex,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
          (t_rel + 1) * D_MODEL_K * D_MODEL_V +
          s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
          idx.1.val * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val) := by
    simpa [forwardStepTileOffset, kIndex, vIndex] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj idx]
  simp [forwardStepSpec, accOffset, forwardStepTileOffset, dOffset, kIndex,
        vIndex, NumericDType.add, NumericDType.mul]

theorem chunk_gate_recurrence_forward_step_store_slice_compute_correct
    (AccPrev S D O : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_forward_step_store_slice AccPrev S D O
        t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx =>
        forwardStepSpec s AccPrev S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_forward_step_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gate_recurrence_forward_step_store_slice_correct AccPrev S
    D O t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s
    hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- **Genuine forward step (closed-form).** When the materialized previous-state
buffer `AccPrev` holds the genuine `m`-step folded state `fwdClosed(m)`, the
forward step slice realizes the genuine `(m+1)`-step folded state
`fwdClosed(m+1)` into output chunk `m+1`. The `expected` value is the standalone
closed form `seed · ∏ d + Σ S · ∏ d` over the *input* regions — not a read-back
of the kernel's own output. This is the inductive step of the carry-fold. -/
theorem chunk_gate_recurrence_forward_step_store_slice_closed_form
    (AccPrev S D O LastKv : RegionName) (HAS_LAST_KV : Bool)
    (m NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (m + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
    (hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V m idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_forward_step_store_slice AccPrev S D O
        m NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, forwardStepTileOffset s (m + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx =>
        fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V (m + 1) idx) := by
  have h := chunk_gate_recurrence_forward_step_store_slice_compute_correct
    AccPrev S D O m NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s
    hOutInj
  have hcong : (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
      forwardStepSpec s AccPrev S D m NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx)
      = (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V (m + 1) idx) := by
    funext idx
    exact forwardStepSpec_eq_fwdClosed_succ s AccPrev S D LastKv HAS_LAST_KV m
      NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V hAcc idx
  rwa [hcong] at h

/-! ## The reverse recurrence — faithful two-index addressing

`_bwd_recurrence` initialises five pointers and then decrements all of them in
lockstep inside `for i in range(NUM_BLOCK - 1)`:

| pointer | initial chunk (`.py` line) | chunk at loop iteration `i` |
|---------|---------------------------|------------------------------|
| `S`     | `NUM_BLOCK - 2`   (60)    | `NUM_BLOCK - 2 - i`          |
| `DI`    | `NUM_BLOCK - 2`   (62)    | `NUM_BLOCK - 2 - i`          |
| `DG`    | `NUM_BLOCK - 2`   (67)    | `NUM_BLOCK - 2 - i`          |
| `DS`    | `NUM_BLOCK - 1`   (65)    | `NUM_BLOCK - 1 - i`          |
| `d`     | `NUM_BLOCK - 1`   (69)    | `NUM_BLOCK - 1 - i`          |

Writing `t_rel := NUM_BLOCK - 2 - i` for the chunk being *written*, every
iteration satisfies

```
index(DS) = index(S) + 1 = t_rel + 1        index(d) = chunk(DI) + 1 = t_rel + 1
```

so the reverse loop body is `Dacc = Dacc * d_{t_rel+1} + DS_{t_rel+1}`, written to
`DI`/`DG` at chunk `t_rel` and multiplied against `S` at chunk `t_rel`. The step
slices below take `t_rel` and derive `t_rel + 1` internally, so the relation is
part of the modeled kernel rather than a hypothesis.

Unrolling that recurrence (seeded by `Dacc = tl.zeros` before the loop, i.e. at
`t_rel = NUM_BLOCK - 2` the incoming carry is `0`) gives the **genuine reverse
closed form** for the carry whose gate/`DS` index is `a`:

```
bwdClosed a = Σ_{a ≤ u < NUM_BLOCK} DS_u · ∏_{a ≤ w < u} d_w
```

with `bwdClosed NUM_BLOCK = 0` (the zero seed) and
`bwdClosed a = bwdClosed (a+1) · d_a + DS_a`. Chunk `t_rel`'s stored carry is
`bwdClosed (t_rel + 1)`; the post-loop `DL` value is `bwdClosed 0`. This is a
standalone spec over the *input* regions `DS` (the incoming `DO`) and `D` — never
a read-back of `DI`/`DG`/`DL`. -/

/-- Genuine closed form of the reverse carry with gate/`DS` index `a`:
`Σ_{a ≤ u < NUM_BLOCK} DS_u · ∏_{a ≤ w < u} d_w`. The gate factor reuses
`fwdGate` because the backward kernel reads the very same `cross_decay` row
(`d + offset_bh * NUM_BLOCK + ·`) as the forward kernel. -/
noncomputable def bwdClosed
    (s : BlockState) (DS D : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V a : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  ∑ u ∈ Finset.Ico a NUM_BLOCK,
    s.readMem DS
        (timeTileOffset s u NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx) *
      ∏ w ∈ Finset.Ico a u, fwdGate s D NUM_BLOCK w

/-- `bwdClosed` at `a = NUM_BLOCK` is `0` — exactly the `Dacc = tl.zeros(...)`
seed the reverse loop starts from (the first iteration writes chunk
`NUM_BLOCK - 2` and so has gate index `NUM_BLOCK - 1`, whose incoming carry has
index `NUM_BLOCK`). -/
theorem bwdClosed_top
    (s : BlockState) (DS D : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) :
    bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
        NUM_BLOCK idx = 0 := by
  simp [bwdClosed]

/-- **The reverse carry-fold recurrence.** Peeling the bottom index:
`bwdClosed(a) = bwdClosed(a+1) · d_a + DS_a`. This is the exact closed-form
counterpart of the Python loop body `Dacc = Dacc * d_i + DS_i`. -/
theorem bwdClosed_step
    (s : BlockState) (DS D : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V a : Nat)
    (ha : a < NUM_BLOCK)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) :
    bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
        a idx
      = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (a + 1) idx *
          fwdGate s D NUM_BLOCK a
        + s.readMem DS
            (timeTileOffset s a NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
              BLOCK_MODEL_V idx) := by
  unfold bwdClosed
  rw [Finset.sum_eq_sum_Ico_succ_bot ha, Finset.Ico_self, Finset.prod_empty,
    mul_one, Finset.sum_mul]
  -- split the bottom gate `d_a` out of each remaining `∏_{a ≤ w < u} d_w`.
  have hsum : ∀ u ∈ Finset.Ico (a + 1) NUM_BLOCK,
      s.readMem DS
            (timeTileOffset s u NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
              BLOCK_MODEL_V idx) *
          ∏ w ∈ Finset.Ico a u, fwdGate s D NUM_BLOCK w
        = (s.readMem DS
              (timeTileOffset s u NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
                BLOCK_MODEL_V idx) *
            ∏ w ∈ Finset.Ico (a + 1) u, fwdGate s D NUM_BLOCK w) *
          fwdGate s D NUM_BLOCK a := by
    intro u hu
    simp only [Finset.mem_Ico] at hu
    rw [Finset.prod_eq_prod_Ico_succ_bot (by omega : a < u)]
    ring
  rw [Finset.sum_congr rfl hsum]
  ring

/-- The one gated multiply-add the reverse recurrence performs, with the
gate/`DS` index `a` given explicitly: `Dacc · d_a + DS_a`. The carry `Dacc` is
read from the materialized single-tile scratch buffer at `accOffset` (Python's
`Dacc` is one register tile, not a chunk-indexed array). Both the reverse loop
body (`a := t_rel + 1`) and the post-loop `DL` step (`a := 0`) are instances. -/
noncomputable def bwdCarryStep
    (s : BlockState) (DaccPrev DS D : RegionName)
    (a NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  s.readMem DaccPrev
      (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) *
    s.readMem D (dOffset s a NUM_BLOCK) +
  s.readMem DS
      (timeTileOffset s a NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx)

/-- **Reverse carry-fold step (genuine).** If the materialized carry buffer holds
the genuine `(a+1)`-indexed reverse carry `bwdClosed(a+1)`, then one gated
multiply-add at gate index `a` produces exactly `bwdClosed(a)`. -/
theorem bwdCarryStep_eq_bwdClosed
    (s : BlockState) (DaccPrev DS D : RegionName)
    (a NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (ha : a < NUM_BLOCK)
    (hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (a + 1) idx)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) :
    bwdCarryStep s DaccPrev DS D a NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx
      = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V a idx := by
  rw [bwdClosed_step s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
    BLOCK_MODEL_V a ha idx]
  unfold bwdCarryStep
  rw [hDacc idx]
  rfl

/-- One backward recurrence step for the tile accumulator:
`Dacc = Dacc * d_i + DS_i`, then store the updated accumulator into `DI` at the
current reverse-loop chunk. This isolates the reverse loop body's accumulator
arithmetic from the full loop induction.

Faithful addressing (see the table above): `DS` and `D` are read at index
`t_rel + 1`, while `DI` is written at chunk `t_rel`. -/
def chunk_gate_recurrence_bwd_dacc_step_DI_store_slice
    (DaccPrev DS D DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  ds_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel + 1) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  out_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :])
  ds_i = tl.load(DS + ds_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel + 1))
  dacc = prev * d_i + ds_i
  tl.store(DI + out_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :],
    (dacc).to(DI.dtype.element_ty))
}

/-- The reverse loop body's arithmetic at written chunk `t_rel`: the gate and
`DS` are one chunk ahead. -/
noncomputable def bwdDaccStepSpec
    (s : BlockState) (DaccPrev DS D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  bwdCarryStep s DaccPrev DS D (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
    BLOCK_MODEL_K BLOCK_MODEL_V idx

/-- **Genuine reverse step for `DI`.** Under the carry pin
`DaccPrev = bwdClosed(t_rel+2)`, the loop body's stored value at chunk `t_rel` is
the genuine reverse closed form `bwdClosed(t_rel+1)`. -/
theorem bwdDaccStepSpec_eq_bwdClosed
    (s : BlockState) (DaccPrev DS D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (ht : t_rel + 1 < NUM_BLOCK)
    (hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 2) idx)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) :
    bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx
      = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V (t_rel + 1) idx :=
  bwdCarryStep_eq_bwdClosed s DaccPrev DS D (t_rel + 1) NUM_BLOCK D_MODEL_K
    D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V ht hDacc idx

theorem chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_correct
    (DaccPrev DS D DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      let outAddr := timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx
      (exec (chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev DS
            D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V) s).map
          (·.readMem DI outAddr)
        = some (bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K
            D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  intro idx
  simp [exec, chunk_gate_recurrence_bwd_dacc_step_DI_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        timeTileOffset, dOffset, kIndex, vIndex, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → Nat :=
    fun idx =>
      s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
        t_rel * D_MODEL_K * D_MODEL_V +
        s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
        s.pids 2 * BLOCK_MODEL_V + idx.1.val * D_MODEL_V + idx.2.1.val
  let valueFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → ℝ :=
    fun idx =>
      s.readMem DaccPrev
          (s.pids 0 * D_MODEL_K * D_MODEL_V +
            s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
            idx.1.val * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val) *
        s.readMem D (s.pids 0 * NUM_BLOCK + (t_rel + 1)) +
      s.readMem DS
          (s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
            (t_rel + 1) * D_MODEL_K * D_MODEL_V +
            s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
            s.pids 2 * BLOCK_MODEL_V + idx.1.val * D_MODEL_V + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, timeTileOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DI (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_MODEL_K, BLOCK_MODEL_V])).readMem DI
        (offsetFn idx) =
    bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
      BLOCK_MODEL_K BLOCK_MODEL_V idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, bwdDaccStepSpec, bwdCarryStep, accOffset, timeTileOffset,
        dOffset, kIndex, vIndex]

theorem chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_compute_correct
    (DaccPrev DS D DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev
        DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DI, timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx =>
        bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_dacc_step_DI_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_correct
    DaccPrev DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
    BLOCK_MODEL_V s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

def bwdDGOffset (s : BlockState) (t_rel NUM_BLOCK NUM_K NUM_V : Nat) : Nat :=
  s.pids 0 * NUM_BLOCK * NUM_K * NUM_V +
    t_rel * NUM_K * NUM_V + s.pids 1 * NUM_V + s.pids 2

/-- One backward recurrence step for the compact `DG` scalar:
compute `Dacc = Dacc * d_i + DS_i`, then `DG_i = tl.sum(Dacc * S_i)` and store
that scalar into the `[B*H, NUM_BLOCK, NUM_K, NUM_V]` gradient layout.

Faithful addressing: `DS` and `D` are read at index `t_rel + 1` while `S` is read
and `DG` written at chunk `t_rel`. -/
def chunk_gate_recurrence_bwd_dg_step_store_slice
    (DaccPrev DS S D DG : RegionName)
    (t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat) : ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  ds_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel + 1) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  s_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :])
  ds_i = tl.load(DS + ds_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  s_i = tl.load(S + s_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel + 1))
  dacc = prev * d_i + ds_i
  dg_i = tl.sum(dacc * s_i)
  tl.store(DG + offset_bh * $(NUM_BLOCK) * $(NUM_K) * $(NUM_V) +
    $(t_rel) * $(NUM_K) * $(NUM_V) + offset_d * $(NUM_V) + offset_s,
    (dg_i).to(DG.dtype.element_ty))
}

noncomputable def bwdDGStepSpec
    (s : BlockState) (DaccPrev DS S D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : ℝ :=
  ∑ i : Fin BLOCK_MODEL_K, ∑ j : Fin BLOCK_MODEL_V,
    let idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] :=
      TileShape.insertAxisIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] 1
        (TileShape.insertAxisIndex [BLOCK_MODEL_K] 0 PUnit.unit i) j
    bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx *
      s.readMem S
        (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)

theorem chunk_gate_recurrence_bwd_dg_step_store_slice_correct
    (DaccPrev DS S D DG : RegionName)
    (t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat)
    (s s' : BlockState)
    (hExec : exec (chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V) s = some s') :
    s'.readMem DG (bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V) =
      bwdDGStepSpec s DaccPrev DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V := by
  simp [exec, chunk_gate_recurrence_bwd_dg_step_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        NumericDType.add, NumericDType.mul, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  rw [← hExec]
  simp [bwdDGOffset, bwdDGStepSpec, bwdDaccStepSpec, bwdCarryStep, accOffset,
        timeTileOffset, dOffset, kIndex, vIndex, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        NumericDType.mul, NumericDType.add]
  congr

theorem chunk_gate_recurrence_bwd_dg_step_store_slice_compute_correct
    (DaccPrev DS S D DG : RegionName)
    (t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ =>
        bwdDGStepSpec s DaccPrev DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_dg_step_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact chunk_gate_recurrence_bwd_dg_step_store_slice_correct DaccPrev DS S D
    DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
    BLOCK_MODEL_V s s' hExec

/-- The post-loop `DL` boundary-gradient store (`chunk_gate_recurrence.py` lines
90-94). After the reverse loop's `NUM_BLOCK - 1` decrements the `DS` and `d`
pointers sit at chunk `0`, and Python performs **one more** gated multiply-add
`Dacc = Dacc * d_i + DS_i` before storing into `DL`. `DL` is a single
`[B, H, D_k, D_v]` tile with no chunk axis, hence the `accOffset` layout on the
store.

The tail index is a parameter `t_rel`; the headline instantiates it at `0`, which
is where the loop's `NUM_BLOCK - 1` decrements leave the pointers (the arrival
position is part of the trusted fold boundary). -/
def chunk_gate_recurrence_bwd_DL_store_slice
    (DaccTail DS D DL : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  ds_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccTail + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :])
  ds_i = tl.load(DS + ds_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
  dacc = prev * d_i + ds_i
  tl.store(DL + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :], (dacc).to(DL.dtype.element_ty))
}

theorem chunk_gate_recurrence_bwd_DL_store_slice_correct
    (DaccTail DS D DL : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      let outAddr := accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx
      (exec (chunk_gate_recurrence_bwd_DL_store_slice DaccTail DS D DL t_rel
            NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s).map
          (·.readMem DL outAddr)
        = some (bwdCarryStep s DaccTail DS D t_rel NUM_BLOCK D_MODEL_K
            D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  intro idx
  simp [exec, chunk_gate_recurrence_bwd_DL_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        accOffset, timeTileOffset, dOffset, kIndex, vIndex,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → Nat :=
    fun idx =>
      s.pids 0 * D_MODEL_K * D_MODEL_V +
        s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
        idx.1.val * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val
  let valueFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → ℝ :=
    fun idx =>
      s.readMem DaccTail
          (s.pids 0 * D_MODEL_K * D_MODEL_V +
            s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
            idx.1.val * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val) *
        s.readMem D (s.pids 0 * NUM_BLOCK + t_rel) +
      s.readMem DS
          (s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
            t_rel * D_MODEL_K * D_MODEL_V +
            s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
            s.pids 2 * BLOCK_MODEL_V + idx.1.val * D_MODEL_V + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, accOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DL (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_MODEL_K, BLOCK_MODEL_V])).readMem DL
        (offsetFn idx) =
    bwdCarryStep s DaccTail DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
      BLOCK_MODEL_K BLOCK_MODEL_V idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, bwdCarryStep, accOffset, timeTileOffset, dOffset, kIndex,
        vIndex]

theorem chunk_gate_recurrence_bwd_DL_store_slice_compute_correct
    (DaccTail DS D DL : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_DL_store_slice DaccTail DS D DL
        t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DL, accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx =>
        bwdCarryStep s DaccTail DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_DL_store_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gate_recurrence_bwd_DL_store_slice_correct DaccTail DS D DL
    t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Genuine dimension-general closed-form realizations

The per-statement step slices above are already dimension-parameterized
(symbolic `NUM_BLOCK`, `D_MODEL_*`, `BLOCK_MODEL_*`). These general closed-form
realizations thread the genuine specs through them with no pinned dimensions, on
honest side conditions only (output-offset injectivity per store). They feed the
dimension-general headline summaries below. -/

/-- **Genuine forward initial store (`last_kv` branch), dimension-general.** The
`last_kv`-seeded initial output store realizes the genuine closed form
`fwdClosed(0) = last_kv` over symbolic dimensions, on output-offset
injectivity. -/
theorem chunk_gate_recurrence_fwd_initial_last_kv_closed_form_general
    (LastKv O S D : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_last_kv_store_slice LastKv O
        NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.true NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx) := by
  have h := chunk_gate_recurrence_initial_last_kv_store_slice_compute_correct
    LastKv O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hOutInj
  have hcong : (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
      s.readMem LastKv
        (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
      = (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.true NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx) := by
    funext idx; rw [fwdClosed_zero]; simp [fwdSeed]
  rwa [hcong] at h

/-- **Genuine forward initial store (zero branch), dimension-general.** When
`last_kv` is absent the initial output store realizes `fwdClosed(0) = 0` over
symbolic dimensions, on output-offset injectivity. -/
theorem chunk_gate_recurrence_fwd_initial_zero_closed_form_general
    (O S D LastKv : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_zero_store_slice O NUM_BLOCK
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.false NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx) := by
  have h := chunk_gate_recurrence_initial_zero_store_slice_compute_correct
    O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hOutInj
  have hcong : (fun _ : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] => (0.0 : ℝ))
      = (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.false NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx) := by
    funext idx; rw [fwdClosed_zero]; simp [fwdSeed]; norm_num
  rwa [hcong] at h

/-- **Genuine reverse step for `DI`, dimension-general.** Under the reverse carry
pin `DaccPrev = bwdClosed(t_rel+2)`, the reverse loop body's `DI` store at chunk
`t_rel` realizes the genuine reverse closed form `bwdClosed(t_rel+1)` — the
standalone `Σ_{u} DS_u · ∏ d` over the *input* regions `DS`/`D`, not a read-back
of `DI`. -/
theorem chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_closed_form
    (DaccPrev DS D DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    (ht : t_rel + 1 < NUM_BLOCK)
    (hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 2) idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev
        DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DI, timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V (t_rel + 1) idx) := by
  have h := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_compute_correct
    DaccPrev DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
    BLOCK_MODEL_V s hOutInj
  have hcong : (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
      bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx)
      = (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V (t_rel + 1) idx) := by
    funext idx
    exact bwdDaccStepSpec_eq_bwdClosed s DaccPrev DS D t_rel NUM_BLOCK
      D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V ht hDacc idx
  rwa [hcong] at h

/-- Genuine closed form of the `DG` scalar written at chunk `t_rel`:
`Σ_{k,v} bwdClosed(t_rel+1)[k,v] · S_{t_rel}[k,v]` — the reverse carry contracted
against the `S` chunk *one behind* the carry's own index, exactly as the Python
loop pairs them. -/
noncomputable def bwdDGClosed
    (s : BlockState) (DS S D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : ℝ :=
  ∑ i : Fin BLOCK_MODEL_K, ∑ j : Fin BLOCK_MODEL_V,
    let idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] :=
      TileShape.insertAxisIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] 1
        (TileShape.insertAxisIndex [BLOCK_MODEL_K] 0 PUnit.unit i) j
    bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
        (t_rel + 1) idx *
      s.readMem S
        (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)

/-- Under the reverse carry pin, the `DG` reduction is the genuine
`Σ bwdClosed(t_rel+1) · S_{t_rel}`. -/
theorem bwdDGStepSpec_eq_bwdDGClosed
    (s : BlockState) (DaccPrev DS S D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (ht : t_rel + 1 < NUM_BLOCK)
    (hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 2) idx) :
    bwdDGStepSpec s DaccPrev DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V
      = bwdDGClosed s DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V := by
  simp only [bwdDGStepSpec, bwdDGClosed,
    bwdDaccStepSpec_eq_bwdClosed s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K
      D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V ht hDacc]

/-- **Genuine reverse step for `DG`, dimension-general.** Under the reverse carry
pin, the `DG` store at chunk `t_rel` realizes `Σ bwdClosed(t_rel+1) · S_{t_rel}`
over the *input* regions `DS`/`D`/`S`. -/
theorem chunk_gate_recurrence_bwd_dg_step_store_slice_closed_form
    (DaccPrev DS S D DG : RegionName)
    (t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (ht : t_rel + 1 < NUM_BLOCK)
    (hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 2) idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ =>
        bwdDGClosed s DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V) := by
  have h := chunk_gate_recurrence_bwd_dg_step_store_slice_compute_correct
    DaccPrev DS S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V
    BLOCK_MODEL_K BLOCK_MODEL_V s
  rwa [bwdDGStepSpec_eq_bwdDGClosed s DaccPrev DS S D t_rel NUM_BLOCK D_MODEL_K
    D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V ht hDacc] at h

/-- **Genuine post-loop `DL` step, dimension-general.** Under the tail carry pin
`DaccTail = bwdClosed(t_rel+1)`, the post-loop multiply-add stores the genuine
`bwdClosed(t_rel)` into `DL`. At the Python arrival position `t_rel = 0` this is
`bwdClosed 0 = Σ_{u < NUM_BLOCK} DS_u · ∏_{w < u} d_w`, the full reverse fold. -/
theorem chunk_gate_recurrence_bwd_DL_store_slice_closed_form
    (DaccTail DS D DL : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
    (ht : t_rel < NUM_BLOCK)
    (hDaccTail : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccTail
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 1) idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_DL_store_slice DaccTail DS D DL
        t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DL, accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V t_rel idx) := by
  have h := chunk_gate_recurrence_bwd_DL_store_slice_compute_correct DaccTail
    DS D DL t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s
    hOutInj
  have hcong : (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
      bwdCarryStep s DaccTail DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx)
      = (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V t_rel idx) := by
    funext idx
    exact bwdCarryStep_eq_bwdClosed s DaccTail DS D t_rel NUM_BLOCK D_MODEL_K
      D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V ht hDaccTail idx
  rwa [hcong] at h

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **SCOPE — this is a claim about six hand-cut store/step slices, not about the
launched kernels.** The launched surfaces appear only in the three lowering
clauses. Neither `range(NUM_BLOCK-1)` fold is modeled: the carry pins `hAcc`,
`hDacc`, `hDaccTail` are assumptions, and no clause chains one step's output into
the next step's carry buffer.

**Genuine shape-general output summary** for both Python kernels of
`chunk_gate_recurrence.py` — `_fwd_recurrence` and `_bwd_recurrence` — bundled
into one headline.

Every `expected` below is a standalone closed form over the *input* regions
(`S`, `D`, `last_kv` forward; `DS` = the incoming `DO`, `D`, `S` backward) — never
a read-back of the kernel's own `O`/`DI`/`DG`/`DL`:

* both forward surfaces (`last_kv` present/absent) and the backward surface lower
  to the algorithm layer at arbitrary symbolic dimensions;
* the forward initial store realizes `fwdClosed(0) = last_kv` (`last_kv` branch)
  and `fwdClosed(0) = 0` (zero branch);
* one forward loop body realizes `fwdClosed(t_rel+1)` from
  `AccPrev = fwdClosed(t_rel)`;
* one reverse loop body realizes `bwdClosed(t_rel+1)` into `DI` chunk `t_rel` and
  `Σ bwdClosed(t_rel+1) · S_{t_rel}` into `DG` chunk `t_rel`, from
  `DaccPrev = bwdClosed(t_rel+2)` — with `DS` and the gate `d` read one chunk
  **ahead** of `S`/`DI`/`DG`, as Python's split pointer initialisation dictates;
* the post-loop step realizes `bwdClosed(0)` into `DL`, restoring the
  `Dacc = Dacc * d_i + DS_i` that precedes that store.

Side conditions are honest: per-store output-offset injectivity, the single chunk
in-range bound, and the three carry pins. The cross-chunk folds over
`range(NUM_BLOCK-1)` (forward and reverse) are the trusted boundary, as is the
fact that `NUM_BLOCK - 1` decrements leave the backward pointers at chunk `0`. -/
specification chunk_gate_recurrence_output_summary_general
    (AccPrev S D O LastKv : RegionName) (HAS_LAST_KV : Bool)
    (DaccPrev DaccTail DS DI DG DL : RegionName) (t_rel : Nat)
    (NUM_HEAD NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat)
    (s : BlockState)
    -- forced by `BlockState.scatter_readback_nd`: the forward initial tile store
    -- must not have two lanes colliding on one `O` cell.
    (hOutInj0 : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    -- same, for the forward loop body's store into `O` chunk `t_rel + 1`.
    (hOutInjStep : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
    -- same, for the reverse loop body's store into `DI` chunk `t_rel`.
    (hDIInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    -- same, for the post-loop store into the chunk-free `DL` tile.
    (hDLInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
    -- forced by `bwdClosed_step`: the reverse loop body's gate/`DS` index is
    -- `t_rel + 1`, and peeling it off the fold needs it to be a real chunk.
    -- Python's loop runs `t_rel = NUM_BLOCK-2 … 0`, so `t_rel + 1 ≤ NUM_BLOCK-1`.
    -- (`bwdClosed_step` at the post-loop index `0` needs `0 < NUM_BLOCK`; that is
    -- implied by `hBwdIdx`, so it is not stated separately.)
    (hBwdIdx : t_rel + 1 < NUM_BLOCK)
    -- the forward carry: `acc` entering loop iteration `t_rel` is the `t_rel`-step
    -- fold. Assumed — the `range(NUM_BLOCK-1)` fold is the trusted boundary.
    (hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V t_rel idx)
    -- the reverse carry: `Dacc` entering the iteration that writes chunk `t_rel`
    -- is the reverse fold at gate index `t_rel + 2`. Assumed for the same reason;
    -- at the loop's first iteration (`t_rel = NUM_BLOCK - 2`) it is discharged by
    -- `bwdClosed_top`, matching Python's `Dacc = tl.zeros(...)` seed.
    (hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 2) idx)
    -- the reverse carry as the loop *exits* (gate index `1`), feeding the
    -- post-loop `DL` step. A separate region from `DaccPrev` on purpose: pinning
    -- one buffer to two different fold stages would silently constrain `t_rel`.
    (hDaccTail : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccTail
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V 1 idx) :
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    ((chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V).toAlgorithm? =
        Except.ok
          (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
            NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V).toAlgKernel) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_last_kv_store_slice LastKv O
        NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.true NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_zero_store_slice O NUM_BLOCK
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.false NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_forward_step_store_slice AccPrev S D O
        t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K
          D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V (t_rel + 1) idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev
        DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DI, timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V (t_rel + 1) idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ =>
        bwdDGClosed s DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_DL_store_slice DaccTail DS D DL
        0 NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DL, accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V 0 idx)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact chunk_gate_recurrence_fwd_surface_toAlgorithm_supported S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V Bool.true
  · exact chunk_gate_recurrence_fwd_surface_toAlgorithm_supported S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V Bool.false
  · exact chunk_gate_recurrence_bwd_surface_toAlgorithm_supported S D DI DG DL DS
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
  · exact chunk_gate_recurrence_fwd_initial_last_kv_closed_form_general
      LastKv O S D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s
      hOutInj0
  · exact chunk_gate_recurrence_fwd_initial_zero_closed_form_general
      O S D LastKv NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s
      hOutInj0
  · exact chunk_gate_recurrence_forward_step_store_slice_closed_form
      AccPrev S D O LastKv HAS_LAST_KV t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
      BLOCK_MODEL_K BLOCK_MODEL_V s hOutInjStep hAcc
  · exact chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_closed_form
      DaccPrev DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V s hDIInj hBwdIdx hDacc
  · exact chunk_gate_recurrence_bwd_dg_step_store_slice_closed_form
      DaccPrev DS S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V
      BLOCK_MODEL_K BLOCK_MODEL_V s hBwdIdx hDacc
  · exact chunk_gate_recurrence_bwd_DL_store_slice_closed_form DaccTail DS D DL
      0 NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hDLInj
      (by omega) hDaccTail

end Correct_without_Rounding


/-! ## ════════ `⊨` IO face for the forward output store ════════

The summary above is stated per *declared write map*. This section restates the
forward `Acc → O` chunk store on the audit-once IO surface
`Masked3DTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat memory**
placement.

Zero new library surface: the slice is an unmasked `[BLOCK_MODEL_K, BLOCK_MODEL_V]`
tile copy whose two addresses are built from **all three** program axes
(`offset_bh`, `offset_d`, `offset_s`) with different leading factors on the two
buffers — exactly what the three-axis tile skin's window functions take. -/

section IOFace

/-- Cell-level frame of an unmasked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame_unmasked {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl, BlockState.writeMem_mem, if_neg ?_]
      rintro ⟨h1, h2⟩
      rcases hc with h | h
      · exact h h1
      · exact h hd List.mem_cons_self h2.symm

theorem fwd_store_flattenOk (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ((chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [chunk_gate_recurrence_forward_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem fwd_store_terminates (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) (s : BlockState) :
    ∃ s1, exec (chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s = some s1 := by
  simp [exec, chunk_gate_recurrence_forward_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, TileShape.dropInsertedIndex]

theorem fwd_store_frame (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) (s s' : BlockState)
    (hExec : exec (chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ O ∨ ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
        o ≠ outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, chunk_gate_recurrence_forward_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame_unmasked (region := O)
    (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
      s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V
        + s.pids 1 * D_MODEL_V * BLOCK_MODEL_K + idx.1.val * D_MODEL_V
        + s.pids 2 * BLOCK_MODEL_V + idx.2.1.val)
    _ r o (TileShape.allIndices [BLOCK_MODEL_K, BLOCK_MODEL_V]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ => Ne.symm (h idx)

theorem fwd_store_traceSafe (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx
        < bounds Acc)
    (hout : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx < bounds O) :
    ((chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V).toAlgKernel).TraceSafe
      bounds s := by
  simp [Kernel.TraceSafe, chunk_gate_recurrence_forward_store_slice,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
    NumericDType.mul, TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b => hin (a, b, PUnit.unit)
  all_goals try exact fun a b => hout (a, b, PUnit.unit)
  all_goals try (simp [Op.SafeAt.eq_def]; done)

/-- Region-model run of the forward output store. -/
theorem fwd_store_region_run (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s₀ NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
    (xs : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s₀.readMem Acc
          (accOffset s₀ D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = xs idx) :
    ∃ s1, exec (chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s₀ = some s1
      ∧ (∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
          s1.readMem O (outOffset s₀ NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) = xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ O ∨ ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
            o ≠ outOffset s₀ NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := fwd_store_terminates Acc O NUM_BLOCK D_MODEL_K D_MODEL_V
    BLOCK_MODEL_K BLOCK_MODEL_V s₀
  refine ⟨s1, hexec, ?_, fwd_store_frame Acc O NUM_BLOCK D_MODEL_K D_MODEL_V
    BLOCK_MODEL_K BLOCK_MODEL_V s₀ s1 hexec⟩
  intro idx
  have h := chunk_gate_recurrence_forward_store_slice_correct Acc O NUM_BLOCK
    D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s₀ hOutInj idx
  have h' : s1.readMem O (outOffset s₀ NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
      = s₀.readMem Acc
        (accOffset s₀ D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
    simpa [hexec] using h
  rw [h', hx idx]

/-- IO signature of the forward output store on the three-axis tile surface. -/
def fwdStoreIO (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : Masked3DTileKernelIO₁ where
  kernel := chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
  inp := Acc
  out := O
  shape := [BLOCK_MODEL_K, BLOCK_MODEL_V]
  read := fun p₀ p₁ p₂ idx =>
    p₀ * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
      + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val
  write := fun p₀ p₁ p₂ idx =>
    p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
      + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val
  mask := fun _p₀ _p₁ _p₂ _ => True

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `chunk_gate_recurrence.py`'s forward
output chunk store: for every disjoint flat placement of `Acc` / `O`, every program
coordinate whose lanes are in bounds, and every launch state whose `Acc` chunk holds
`xs`, the translated pointer kernel terminates, every lane of the `O` chunk holds
`xs idx`, and every other memory cell is unchanged.

Both windows are built from all three program axes with different leading factors on
the two buffers. Dimension-general in `NUM_BLOCK`, `D_MODEL_K`, `D_MODEL_V` and the
two block sizes. Honest side-condition: output-address injectivity at every program
coordinate, the same hypothesis the per-write-map summary takes. -/
specification chunk_gate_recurrence_forward_store_io_correctness
    (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
          + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val)) :
    fwdStoreIO Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V ⊨ fun _p₀ _p₁ xs idx => xs idx := by
  refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact fwd_store_flattenOk Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V
  · intro bounds s h1 h2
    exact fwd_store_traceSafe Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V bounds s (fun idx => h1 idx trivial) (fun idx => h2 idx trivial)
  · intro s₀ xs hin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      fwd_store_region_run Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V s₀ (hOutInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) xs
        (fun idx => hin idx trivial)
    exact ⟨s1, hexec, fun idx _ => hval idx,
      fun r o hcond => hframe r o (by
        rcases hcond with h | h
        · exact Or.inl h
        · exact Or.inr fun idx => h idx trivial)⟩

end IOFace

end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence

