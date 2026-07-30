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
read-back of the kernel's own output `O`.

```
chunk_gate_recurrence_forward_output_summary_general          ← TOP (forward)
  ├─ chunk_gate_recurrence_fwd_surface_toAlgorithm_supported    full surface lowers
  ├─ chunk_gate_recurrence_fwd_initial_last_kv_closed_form_general  (fwdClosed 0 = last_kv)
  ├─ chunk_gate_recurrence_fwd_initial_zero_closed_form_general     (fwdClosed 0 = 0)
  └─ chunk_gate_recurrence_forward_step_store_slice_closed_form (carry-fold step)
       └─ forwardStepSpec_eq_fwdClosed_succ
            └─ fwdClosed_succ   (fwdClosed(m+1) = fwdClosed(m)·d_m + S_m)

chunk_gate_recurrence_backward_output_summary_general         ← TOP (backward)
  ├─ chunk_gate_recurrence_bwd_surface_toAlgorithm_supported   full surface lowers
  ├─ chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_compute_correct  (Dacc·d_i + DS_i)
  └─ chunk_gate_recurrence_bwd_dg_step_store_slice_compute_correct       (Σ Dacc·S_i)

per-store slice lemmas (modeled exactly, fed materialized state buffers):
  forward:  forward_store_slice / initial_last_kv_store_slice /
            initial_zero_store_slice / forward_step_store_slice
  backward: bwd_dacc_step_DI_store_slice / bwd_dg_step_store_slice
each with a `*_correct` (algorithm-layer readback) and
`*_compute_correct` (ComputeCorrect) face (except `initial_last_kv_store_slice`,
which has only a `*_compute_correct` face).
```

**The `DL` boundary-gradient store has no face.** Its former conjunct used a
slice whose load and store addresses were character-identical, with
`expected := bwdDLStoreSpec = s.readMem DaccPre ⟨that same address⟩` — a memcpy
compared against its own source, over a *fiction* region. It has been deleted
rather than presented as a result about `DL`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The fixed block sizes
`BLOCK_MODEL_K = 64`, `BLOCK_MODEL_V = 16` are the Python defaults; the
`.to(tl.float32)` / `.to(_.dtype.element_ty)` casts erase to the identity at the
algorithm layer (post-erasure all dtypes unify to `ℝ`). Each single recurrence
step face — the gated update `acc * d_i + S_i` (forward) and
`Dacc * d_i + DS_i` plus the `sum(Dacc * S_i)` reduction (backward) — and each
masked block store are modeled exactly. The cross-chunk recurrence fold — the
forward `range(NUM_BLOCK-1)` loop threading `acc`, and the reverse
`range(NUM_BLOCK-1)` loop threading `Dacc` from the last chunk plus the
post-loop `DL` step — is the trusted boundary: the carried state is presented to
each step slice as a materialized previous-state buffer
(`AccPrev` / `DaccPrev`). Under the *assumed* carry invariant
`AccPrev = fwdClosed(m)`, the forward loop body realizes the genuine
`(m+1)`-step folded closed form `fwdClosed(m+1)` (theorem
`forwardStepSpec_eq_fwdClosed_succ`, whose algebraic core is `fwdClosed_succ`),
closing the forward recurrence per-statement against a standalone specification.
The two backward step faces realize the arithmetic forms `bwdDaccStepSpec`
(`Dacc·d_i + DS_i`) and `bwdDGStepSpec` (`Σ Dacc·S_i`). Note these read the
carried gradient state from the **fiction** region `DaccPrev`, not from a Python
input: only the `D`/`DS`/`S` factors are genuine input reads, and the `DaccPrev`
factor is whatever the reverse fold is assumed to have put there. There is no
`@triton.autotune` on these kernels. Output offset injectivity / non-collision is
a hypothesis of each headline, not a lemma.

Region distinctness: no `≠` hypotheses are stated and none are needed — every
slice performs all of its loads before its single store and every `expected` is a
function of the *initial* state, so aliasing cannot falsify a face.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `chunk_gate_recurrence_forward_output_summary_general`
and `chunk_gate_recurrence_backward_output_summary_general` — shape-general, but
each scoped to hand-cut store/step **slices**, not to a launched kernel. -/

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

/-- One backward recurrence step for the tile accumulator:
`Dacc = Dacc * d_i + DS_i`, then store the updated accumulator into `DI` at the
current reverse-loop chunk. This isolates the reverse loop body's accumulator
arithmetic from the full loop induction. -/
def chunk_gate_recurrence_bwd_dacc_step_DI_store_slice
    (DaccPrev DS D DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  ds_i = tl.load(DS + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
  dacc = prev * d_i + ds_i
  tl.store(DI + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :],
    (dacc).to(DI.dtype.element_ty))
}

noncomputable def bwdDaccStepSpec
    (s : BlockState) (DaccPrev DS D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  s.readMem DaccPrev
      (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx) *
    s.readMem D (dOffset s t_rel NUM_BLOCK) +
  s.readMem DS
      (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx)

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
          (s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
            t_rel * D_MODEL_K * D_MODEL_V +
            s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
            s.pids 2 * BLOCK_MODEL_V + idx.1.val * D_MODEL_V + idx.2.1.val) *
        s.readMem D (s.pids 0 * NUM_BLOCK + t_rel) +
      s.readMem DS
          (s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
            t_rel * D_MODEL_K * D_MODEL_V +
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
  simp [valueFn, bwdDaccStepSpec, timeTileOffset, dOffset, kIndex, vIndex]

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
that scalar into the `[B*H, NUM_BLOCK, NUM_K, NUM_V]` gradient layout. -/
def chunk_gate_recurrence_bwd_dg_step_store_slice
    (DaccPrev DS S D DG : RegionName)
    (t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat) : ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  ds_i = tl.load(DS + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  s_i = tl.load(S + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
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
  simp [bwdDGOffset, bwdDGStepSpec, bwdDaccStepSpec, timeTileOffset, dOffset,
        kIndex, vIndex, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.mul,
        NumericDType.add]
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

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **SCOPE — this is a claim about three hand-cut store slices, not about the
launched kernel.** The launched surfaces appear only in the two lowering clauses.
The `range(NUM_BLOCK-1)` fold is not modeled: `hAcc` is an assumption, and no
clause chains one step's output into the next step's `AccPrev`.

**Genuine shape-general forward output summary** for
`chunk_gate_recurrence.py`'s `_fwd_recurrence`.

Both public forward surfaces (`last_kv` present/absent) lower to the algorithm
layer for arbitrary symbolic dimensions, and every observable forward writeback
realizes the genuine closed form `fwdClosed` (`seed · ∏ d + Σ S · ∏ d`) over the
*input* regions `S`, `D`, `last_kv` — never a read-back of the kernel's own
output `O`:

* the initial store (`last_kv` branch) realizes `fwdClosed(0) = last_kv`;
* the initial store (zero branch) realizes `fwdClosed(0) = 0`;
* one loop body realizes `fwdClosed(t_rel+1)` from the materialized previous
  state `AccPrev = fwdClosed(t_rel)`.

Side conditions are honest: per-store output-offset injectivity
(`hOutInj0`, `hOutInjStep`) and the carry invariant `hAcc*`. The cross-chunk
fold over `range(NUM_BLOCK-1)` is the trusted boundary. -/
specification chunk_gate_recurrence_forward_output_summary_general
    (AccPrev S D O LastKv : RegionName) (HAS_LAST_KV : Bool) (t_rel : Nat)
    (NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj0 : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    (hOutInjStep : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
    (hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V t_rel idx) :
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.false).toAlgorithm? = Except.ok alg) ∧
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
          BLOCK_MODEL_K BLOCK_MODEL_V (t_rel + 1) idx)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact chunk_gate_recurrence_fwd_surface_toAlgorithm_supported S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V Bool.true
  · exact chunk_gate_recurrence_fwd_surface_toAlgorithm_supported S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V Bool.false
  · exact chunk_gate_recurrence_fwd_initial_last_kv_closed_form_general
      LastKv O S D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s
      hOutInj0
  · exact chunk_gate_recurrence_fwd_initial_zero_closed_form_general
      O S D LastKv NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s
      hOutInj0
  · exact chunk_gate_recurrence_forward_step_store_slice_closed_form
      AccPrev S D O LastKv HAS_LAST_KV t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
      BLOCK_MODEL_K BLOCK_MODEL_V s hOutInjStep hAcc

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **SCOPE — this is a claim about two hand-cut step slices, not about the
launched kernel.** The launched surface appears only in the lowering clause; the
reverse `range(NUM_BLOCK-1)` fold is not modeled, the carried gradient state is
the fiction region `DaccPrev` with nothing constraining its contents, and the
`DL` boundary store has no face at all (see the module docstring).

**Shape-general backward step summary** for
`chunk_gate_recurrence.py`'s `_bwd_recurrence`.

The reverse-loop backward surface lowers to the algorithm layer for arbitrary
symbolic dimensions, and the two backward step faces realize the arithmetic forms
`DI = Dacc·d_i + DS_i` (`bwdDaccStepSpec`) and `DG = Σ Dacc·S_i`
(`bwdDGStepSpec`). Neither is a read-back of the kernel's own outputs; but note
that the `Dacc` factor in both is read from the **fiction** carry region
`DaccPrev`, so only the `D`/`DS`/`S` factors are genuine input reads.

Side conditions are honest: output-offset injectivity `hDIInj` for `DI` (the
scalar `DG` store needs none). The cross-chunk reverse fold over
`range(NUM_BLOCK-1)` is **not modeled**. -/
specification chunk_gate_recurrence_backward_output_summary_general
    (DaccPrev DS S D DI DG DL : RegionName) (t_rel : Nat)
    (NUM_HEAD NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hDIInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
 :
    ((chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V).toAlgorithm? =
        Except.ok
          (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
            NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V).toAlgKernel) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev
        DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DI, timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ =>
        bwdDGStepSpec s DaccPrev DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact chunk_gate_recurrence_bwd_surface_toAlgorithm_supported S D DI DG DL DS
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
  · exact chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_compute_correct
      DaccPrev DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V s hDIInj
  · exact chunk_gate_recurrence_bwd_dg_step_store_slice_compute_correct
      DaccPrev DS S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V
      BLOCK_MODEL_K BLOCK_MODEL_V s

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence

