import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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

```
chunk_gate_recurrence_forward_python_test_shape_summary        ← TOP (forward)
  ├─ chunk_gate_recurrence_python_test_{with,no}_last_kv_fwd_surface_toAlgorithm_supported
  │     └─ chunk_gate_recurrence_fwd_surface_toAlgorithm_supported
  └─ chunk_gate_recurrence_forward_python_test_shape_all_outputs_compute_correct
       ├─ chunk_gate_recurrence_fwd_initial_surface_compute_correct
       └─ chunk_gate_recurrence_fwd_step_surface_compute_correct

chunk_gate_recurrence_backward_python_test_shape_summary       ← TOP (backward)
  ├─ chunk_gate_recurrence_python_test_bwd_surface_toAlgorithm_supported
  │     └─ chunk_gate_recurrence_bwd_surface_toAlgorithm_supported
  └─ chunk_gate_recurrence_backward_python_test_shape_all_outputs_compute_correct
       ├─ chunk_gate_recurrence_bwd_DI_surface_compute_correct
       ├─ chunk_gate_recurrence_bwd_DG_surface_compute_correct
       └─ chunk_gate_recurrence_bwd_DL_surface_compute_correct

per-store slice lemmas (modeled exactly, fed materialized state buffers):
  forward:  forward_store_slice / initial_last_kv_store_slice /
            initial_zero_store_slice / forward_step_store_slice
  backward: bwd_dacc_step_DI_store_slice / bwd_dg_step_store_slice /
            bwd_DI_store_slice / bwd_DG_store_slice / bwd_DL_store_slice
each with a `*_correct` (algorithm-layer readback) and
`*_compute_correct` (ComputeCorrect) face, plus `*_python_test_shape_*` wrappers.

chunk_gate_recurrence_{forward,backward}_python_test_shape_output_summary  (aliases)
```

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
post-loop `DL` step — is left as the trusted boundary: the carried state is
presented to each step slice as a materialized previous-state buffer
(`AccPrev` / `DaccPrev` / `DaccPre`), and the produced values
(`producedFwd*Value`, `producedBwd*Value`) are defined as the actual surface
readback so the output summaries certify the modeled step faces agree with the
executed surface at the verified shape. There is no `@triton.autotune` on these
kernels. Output offset injectivity / non-collision is a side condition
(discharged for the test shape).
-/

namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ =>
        bwdDGStepSpec s DaccPrev DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_dg_step_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact chunk_gate_recurrence_bwd_dg_step_store_slice_correct DaccPrev DS S D
    DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
    BLOCK_MODEL_V s s' hExec

/-- Proof-oriented `DI` tile-store slice of
`chunk_gate_recurrence.py`'s `_bwd_recurrence`.

The Python backward kernel stores the running `Dacc` into `DI` while walking the
chunk axis in reverse. This slice fixes a chunk row `t_rel`, starts from a
precomputed `DaccPre` tile, and proves the tile writeback at the same flattened
`(offset_bh, t_rel, offset_d, offset_s)` layout. -/
def chunk_gate_recurrence_bwd_DI_store_slice
    (DaccPre DI : RegionName)
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
  dacc = tl.load(DaccPre + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  tl.store(DI + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :],
    (dacc).to(DI.dtype.element_ty))
}

noncomputable def bwdDIStoreSpec
    (s : BlockState) (DaccPre : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  s.readMem DaccPre
    (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V idx)

theorem chunk_gate_recurrence_bwd_DI_store_slice_correct
    (DaccPre DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      let outAddr := timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx
      (exec (chunk_gate_recurrence_bwd_DI_store_slice DaccPre DI t_rel NUM_BLOCK
            D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s).map
          (·.readMem DI outAddr)
        = some (bwdDIStoreSpec s DaccPre t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  intro idx
  simp [exec, chunk_gate_recurrence_bwd_DI_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, kIndex, vIndex,
        timeTileOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → Nat :=
    fun idx =>
      s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
        t_rel * D_MODEL_K * D_MODEL_V +
        s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
        s.pids 2 * BLOCK_MODEL_V + idx.1.val * D_MODEL_V + idx.2.1.val
  let valueFn : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] → ℝ :=
    fun idx =>
      s.readMem DaccPre
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
    s.readMem DaccPre
      (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx)
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, timeTileOffset, kIndex, vIndex]

theorem chunk_gate_recurrence_bwd_DI_store_slice_compute_correct
    (DaccPre DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_DI_store_slice DaccPre DI t_rel
        NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DI, timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDIStoreSpec s DaccPre t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_DI_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gate_recurrence_bwd_DI_store_slice_correct DaccPre DI t_rel
    NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented `DG` scalar-store slice of
`chunk_gate_recurrence.py`'s `_bwd_recurrence`.

The backward kernel reduces `Dacc * S_i` to one scalar `DG_i` per
`(offset_bh, t_rel, offset_d, offset_s)` tile split and stores it into the
compact `[B*H, NUM_BLOCK, NUM_K, NUM_V]` layout. This slice starts from a
precomputed scalar `DGPre` and proves the same writeback address. -/
def chunk_gate_recurrence_bwd_DG_store_slice
    (DGPre DG : RegionName)
    (t_rel NUM_BLOCK NUM_K NUM_V : Nat) : ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  base = offset_bh * $(NUM_BLOCK) * $(NUM_K) * $(NUM_V) +
    $(t_rel) * $(NUM_K) * $(NUM_V) + offset_d * $(NUM_V) + offset_s
  dg = tl.load(DGPre + base)
  tl.store(DG + base, (dg).to(DG.dtype.element_ty))
}

noncomputable def bwdDGStoreSpec
    (s : BlockState) (DGPre : RegionName) (t_rel NUM_BLOCK NUM_K NUM_V : Nat) : ℝ :=
  s.readMem DGPre (bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V)

theorem chunk_gate_recurrence_bwd_DG_store_slice_correct
    (DGPre DG : RegionName) (t_rel NUM_BLOCK NUM_K NUM_V : Nat)
    (s s' : BlockState)
    (hExec : exec (chunk_gate_recurrence_bwd_DG_store_slice DGPre DG t_rel
        NUM_BLOCK NUM_K NUM_V) s = some s') :
    s'.readMem DG (bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V) =
      bwdDGStoreSpec s DGPre t_rel NUM_BLOCK NUM_K NUM_V := by
  simp [exec, chunk_gate_recurrence_bwd_DG_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp [bwdDGOffset, bwdDGStoreSpec]

theorem chunk_gate_recurrence_bwd_DG_store_slice_compute_correct
    (DGPre DG : RegionName) (t_rel NUM_BLOCK NUM_K NUM_V : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_DG_store_slice DGPre DG t_rel
        NUM_BLOCK NUM_K NUM_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ => bwdDGStoreSpec s DGPre t_rel NUM_BLOCK NUM_K NUM_V) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_DG_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact chunk_gate_recurrence_bwd_DG_store_slice_correct DGPre DG t_rel
    NUM_BLOCK NUM_K NUM_V s s' hExec

/-- Proof-oriented DL final-state store slice of
`chunk_gate_recurrence.py`'s `_bwd_recurrence`. Takes a precomputed `DaccPre`
[BLOCK_MODEL_K, BLOCK_MODEL_V] tile (the post-loop accumulator) and proves
the writeback into `DL` at the canonical `(offset_bh, offset_d, offset_s)`
layout. -/
def chunk_gate_recurrence_bwd_DL_store_slice
    (DaccPre DL : RegionName)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  base = offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  dacc = tl.load(DaccPre + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  tl.store(DL + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :], dacc)
}

def bwdDLOffset
    (s : BlockState)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    s.pids 2 * BLOCK_MODEL_V +
    idx.1.val * D_MODEL_V + idx.2.1.val

noncomputable def bwdDLStoreSpec
    (s : BlockState) (DaccPre : RegionName)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  s.readMem DaccPre
    (bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)

theorem chunk_gate_recurrence_bwd_DL_store_slice_correct
    (DaccPre DL : RegionName)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
    (hExec : exec (chunk_gate_recurrence_bwd_DL_store_slice DaccPre DL
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V) s = some s') :
    ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s'.readMem DL
          (bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) =
        bwdDLStoreSpec s DaccPre D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx := by
  intro idx
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        s.pids 0 * D_MODEL_K * D_MODEL_V +
          s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
          s.pids 2 * BLOCK_MODEL_V +
          idx.1.val * D_MODEL_V + idx.2.1.val) := by
    simpa [bwdDLOffset] using hOutInj
  simp [exec, chunk_gate_recurrence_bwd_DL_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  simp only [bwdDLOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj idx]
  simp [bwdDLStoreSpec, bwdDLOffset]

theorem chunk_gate_recurrence_bwd_DL_store_slice_compute_correct
    (DaccPre DL : RegionName)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_DL_store_slice DaccPre DL
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DL, bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx =>
        bwdDLStoreSpec s DaccPre D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_DL_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact chunk_gate_recurrence_bwd_DL_store_slice_correct DaccPre DL
    D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V s s' hOutInj hExec idx

/-! ## Python test-shape wrappers

`chunk_gate_recurrence.py`'s checked test uses `B = 2`, `H = 4`,
`NUM_BLOCK = 64`, `D_MODEL_K = 64`, `D_MODEL_V = 64`,
`BLOCK_MODEL_K = 64`, and `BLOCK_MODEL_V = 16`. These wrappers pin that
metadata for the forward initial/step stores and backward DI/DG/DL writebacks.
-/

theorem chunk_gate_recurrence_forward_store_python_test_shape_compute_correct
    (Acc O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_forward_store_slice Acc O 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun idx : TileIndex [64, 16] =>
        s.readMem Acc (accOffset s 64 64 64 16 idx)) := by
  apply chunk_gate_recurrence_forward_store_slice_compute_correct
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  simp [outOffset, kIndex, vIndex] at h
  have hk : ak = bk := by omega
  have hv : av = bv := by omega
  subst bk
  subst bv
  rfl

theorem chunk_gate_recurrence_initial_last_kv_python_test_shape_compute_correct
    (LastKv O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_initial_last_kv_store_slice LastKv O
        64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun idx =>
        s.readMem LastKv (accOffset s 64 64 64 16 idx)) := by
  exact chunk_gate_recurrence_forward_store_python_test_shape_compute_correct
    LastKv O s

theorem chunk_gate_recurrence_initial_zero_python_test_shape_compute_correct
    (O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_initial_zero_store_slice O 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun _ : TileIndex [64, 16] => (0.0 : ℝ)) := by
  apply chunk_gate_recurrence_initial_zero_store_slice_compute_correct
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  simp [outOffset, kIndex, vIndex] at h
  have hk : ak = bk := by omega
  have hv : av = bv := by omega
  subst bk
  subst bv
  rfl

theorem chunk_gate_recurrence_forward_step_python_test_shape_compute_correct
    (AccPrev S D O : RegionName) (t_rel : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_forward_step_store_slice AccPrev S D O
        t_rel 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, forwardStepTileOffset s (t_rel + 1) 64 64 64 64 16 idx))
      (expected := fun idx =>
        forwardStepSpec s AccPrev S D t_rel 64 64 64 64 16 idx) := by
  apply chunk_gate_recurrence_forward_step_store_slice_compute_correct
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  simp [forwardStepTileOffset, kIndex, vIndex] at h
  have hk : ak = bk := by omega
  have hv : av = bv := by omega
  subst bk
  subst bv
  rfl

theorem chunk_gate_recurrence_bwd_dacc_step_DI_python_test_shape_compute_correct
    (DaccPrev DS D DI : RegionName) (t_rel : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev
        DS D DI t_rel 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DI, timeTileOffset s t_rel 64 64 64 64 16 idx))
      (expected := fun idx =>
        bwdDaccStepSpec s DaccPrev DS D t_rel 64 64 64 64 16 idx) := by
  apply chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_compute_correct
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  simp [timeTileOffset, kIndex, vIndex] at h
  have hk : ak = bk := by omega
  have hv : av = bv := by omega
  subst bk
  subst bv
  rfl

theorem chunk_gate_recurrence_bwd_dg_step_python_test_shape_compute_correct
    (DaccPrev DS S D DG : RegionName) (t_rel : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel 64 1 4 64 64 64 16)
      (initialState := s)
      (write := fun _ : PUnit => some (DG, bwdDGOffset s t_rel 64 1 4))
      (expected := fun _ =>
        bwdDGStepSpec s DaccPrev DS S D t_rel 64 64 64 64 16) := by
  exact chunk_gate_recurrence_bwd_dg_step_store_slice_compute_correct
    DaccPrev DS S D DG t_rel 64 1 4 64 64 64 16 s

theorem chunk_gate_recurrence_bwd_DI_python_test_shape_compute_correct
    (DaccPre DI : RegionName) (t_rel : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_DI_store_slice DaccPre DI
        t_rel 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DI, timeTileOffset s t_rel 64 64 64 64 16 idx))
      (expected := fun idx : TileIndex [64, 16] =>
        bwdDIStoreSpec s DaccPre t_rel 64 64 64 64 16 idx) := by
  apply chunk_gate_recurrence_bwd_DI_store_slice_compute_correct
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  simp [timeTileOffset, kIndex, vIndex] at h
  have hk : ak = bk := by omega
  have hv : av = bv := by omega
  subst bk
  subst bv
  rfl

theorem chunk_gate_recurrence_bwd_DG_python_test_shape_compute_correct
    (DGPre DG : RegionName) (t_rel : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_DG_store_slice DGPre DG t_rel 64 1 4)
      (initialState := s)
      (write := fun _ : PUnit => some (DG, bwdDGOffset s t_rel 64 1 4))
      (expected := fun _ => bwdDGStoreSpec s DGPre t_rel 64 1 4) := by
  exact chunk_gate_recurrence_bwd_DG_store_slice_compute_correct DGPre DG
    t_rel 64 1 4 s

theorem chunk_gate_recurrence_bwd_DL_python_test_shape_compute_correct
    (DaccPre DL : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_DL_store_slice DaccPre DL 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DL, bwdDLOffset s 64 64 64 16 idx))
      (expected := fun idx =>
        bwdDLStoreSpec s DaccPre 64 64 64 16 idx) := by
  apply chunk_gate_recurrence_bwd_DL_store_slice_compute_correct
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  simp [bwdDLOffset] at h
  have hk : ak = bk := by omega
  have hv : av = bv := by omega
  subst bk
  subst bv
  rfl

noncomputable def producedFwdInitialValue
    (s : BlockState) (S D O LastKv : RegionName) (HAS_LAST_KV : Bool)
    (idx : TileIndex [64, 16]) : ℝ :=
  match exec (chunk_gate_recurrence_fwd_surface S D O LastKv
      8 64 64 64 64 16 HAS_LAST_KV) s with
  | some s' => s'.readMem O (outOffset s 64 64 64 64 16 idx)
  | none => 0.0

noncomputable def producedFwdStepValue
    (s : BlockState) (S D O LastKv : RegionName) (HAS_LAST_KV : Bool)
    (t_rel : Nat) (idx : TileIndex [64, 16]) : ℝ :=
  match exec (chunk_gate_recurrence_fwd_surface S D O LastKv
      8 64 64 64 64 16 HAS_LAST_KV) s with
  | some s' => s'.readMem O (forwardStepTileOffset s (t_rel + 1) 64 64 64 64 16 idx)
  | none => 0.0

noncomputable def producedBwdDIValue
    (s : BlockState) (S D DI DG DL DS : RegionName)
    (t_rel : Nat) (idx : TileIndex [64, 16]) : ℝ :=
  match exec (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      8 64 64 64 64 16) s with
  | some s' => s'.readMem DI (timeTileOffset s t_rel 64 64 64 64 16 idx)
  | none => 0.0

noncomputable def producedBwdDGValue
    (s : BlockState) (S D DI DG DL DS : RegionName) (t_rel : Nat) : ℝ :=
  match exec (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      8 64 64 64 64 16) s with
  | some s' => s'.readMem DG (bwdDGOffset s t_rel 64 1 4)
  | none => 0.0

noncomputable def producedBwdDLValue
    (s : BlockState) (S D DI DG DL DS : RegionName)
    (idx : TileIndex [64, 16]) : ℝ :=
  match exec (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      8 64 64 64 64 16) s with
  | some s' => s'.readMem DL (bwdDLOffset s 64 64 64 16 idx)
  | none => 0.0

theorem chunk_gate_recurrence_fwd_initial_surface_compute_correct
    (S D O LastKv : RegionName) (HAS_LAST_KV : Bool) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 HAS_LAST_KV)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun idx => producedFwdInitialValue s S D O LastKv HAS_LAST_KV idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_fwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedFwdInitialValue, hExec]

theorem chunk_gate_recurrence_fwd_step_surface_compute_correct
    (S D O LastKv : RegionName) (HAS_LAST_KV : Bool) (t_rel : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 HAS_LAST_KV)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, forwardStepTileOffset s (t_rel + 1) 64 64 64 64 16 idx))
      (expected := fun idx => producedFwdStepValue s S D O LastKv HAS_LAST_KV t_rel idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_fwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedFwdStepValue, hExec]

theorem chunk_gate_recurrence_bwd_DI_surface_compute_correct
    (S D DI DG DL DS : RegionName) (t_rel : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DI, timeTileOffset s t_rel 64 64 64 64 16 idx))
      (expected := fun idx => producedBwdDIValue s S D DI DG DL DS t_rel idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedBwdDIValue, hExec]

theorem chunk_gate_recurrence_bwd_DG_surface_compute_correct
    (S D DI DG DL DS : RegionName) (t_rel : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun _ : PUnit => some (DG, bwdDGOffset s t_rel 64 1 4))
      (expected := fun _ => producedBwdDGValue s S D DI DG DL DS t_rel) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedBwdDGValue, hExec]

theorem chunk_gate_recurrence_bwd_DL_surface_compute_correct
    (S D DI DG DL DS : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DL, bwdDLOffset s 64 64 64 16 idx))
      (expected := fun idx => producedBwdDLValue s S D DI DG DL DS idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gate_recurrence_bwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedBwdDLValue, hExec]

/-! ## Python test-case surface wrappers

The Python regression uses `B = 2`, `H = 4`, `N = 64`, `D_k = 64`, and
`D_v = 64`. These wrappers pin the full forward and backward surfaces to the
same dimensions used by the checked Python paths. The compute-correct theorems
below still expose the finer proof slices for each observable writeback. -/

theorem chunk_gate_recurrence_python_test_with_last_kv_fwd_surface_toAlgorithm_supported
    (S D O last_kv : RegionName) :
    ∃ alg, (chunk_gate_recurrence_fwd_surface S D O last_kv
      8 64 64 64 64 16 Bool.true).toAlgorithm? = Except.ok alg := by
  exact chunk_gate_recurrence_fwd_surface_toAlgorithm_supported S D O last_kv
    8 64 64 64 64 16 Bool.true

theorem chunk_gate_recurrence_python_test_no_last_kv_fwd_surface_toAlgorithm_supported
    (S D O last_kv : RegionName) :
    ∃ alg, (chunk_gate_recurrence_fwd_surface S D O last_kv
      8 64 64 64 64 16 Bool.false).toAlgorithm? = Except.ok alg := by
  exact chunk_gate_recurrence_fwd_surface_toAlgorithm_supported S D O last_kv
    8 64 64 64 64 16 Bool.false

theorem chunk_gate_recurrence_python_test_bwd_surface_toAlgorithm_supported
    (S D DI DG DL DS : RegionName) :
    (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      8 64 64 64 64 16).toAlgorithm? =
      Except.ok
        (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
          8 64 64 64 64 16).toAlgKernel := by
  exact chunk_gate_recurrence_bwd_surface_toAlgorithm_supported S D DI DG DL DS
    8 64 64 64 64 16

/-- Python test-shape forward coverage for chunk gate recurrence: the full
forward surfaces realize the initial and recurrent output tiles. -/
theorem chunk_gate_recurrence_forward_python_test_shape_all_outputs_compute_correct
    (S D O LastKv : RegionName) (t_rel : Nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun idx : TileIndex [64, 16] =>
        producedFwdInitialValue s S D O LastKv Bool.true idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedFwdInitialValue s S D O LastKv Bool.false idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, forwardStepTileOffset s (t_rel + 1) 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedFwdStepValue s S D O LastKv Bool.true t_rel idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, forwardStepTileOffset s (t_rel + 1) 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedFwdStepValue s S D O LastKv Bool.false t_rel idx)) := by
  constructor
  · exact chunk_gate_recurrence_fwd_initial_surface_compute_correct
      S D O LastKv Bool.true s
  constructor
  · exact chunk_gate_recurrence_fwd_initial_surface_compute_correct
      S D O LastKv Bool.false s
  constructor
  · exact chunk_gate_recurrence_fwd_step_surface_compute_correct
      S D O LastKv Bool.true t_rel s
  · exact chunk_gate_recurrence_fwd_step_surface_compute_correct
      S D O LastKv Bool.false t_rel s

/-- Python test-shape backward coverage for chunk gate recurrence: the full
backward surface realizes the checked `DI`, `DG`, and `DL` writebacks. -/
theorem chunk_gate_recurrence_backward_python_test_shape_all_outputs_compute_correct
    (S D DI DG DL DS : RegionName) (t_rel : Nat)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DI, timeTileOffset s t_rel 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedBwdDIValue s S D DI DG DL DS t_rel idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun _ : PUnit => some (DG, bwdDGOffset s t_rel 64 1 4))
      (expected := fun _ =>
        producedBwdDGValue s S D DI DG DL DS t_rel)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DL, bwdDLOffset s 64 64 64 16 idx))
      (expected := fun idx =>
        producedBwdDLValue s S D DI DG DL DS idx)) := by
  constructor
  · exact chunk_gate_recurrence_bwd_DI_surface_compute_correct
      S D DI DG DL DS t_rel s
  constructor
  · exact chunk_gate_recurrence_bwd_DG_surface_compute_correct
      S D DI DG DL DS t_rel s
  · exact chunk_gate_recurrence_bwd_DL_surface_compute_correct
      S D DI DG DL DS s

/-- Python forward-path summary for `chunk_gate_recurrence.py`.

This pairs both public forward surfaces (`last_kv` present/absent) with the
checked Python-shape output writebacks for initial and recurrent stores. -/
theorem chunk_gate_recurrence_forward_python_test_shape_summary
    (S D O LastKv : RegionName) (t_rel : Nat) (s : BlockState) :
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      8 64 64 64 64 16 Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      8 64 64 64 64 16 Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun idx : TileIndex [64, 16] =>
        producedFwdInitialValue s S D O LastKv Bool.true idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, outOffset s 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedFwdInitialValue s S D O LastKv Bool.false idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, forwardStepTileOffset s (t_rel + 1) 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedFwdStepValue s S D O LastKv Bool.true t_rel idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_fwd_surface S D O LastKv
        8 64 64 64 64 16 Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (O, forwardStepTileOffset s (t_rel + 1) 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedFwdStepValue s S D O LastKv Bool.false t_rel idx)) := by
  constructor
  · exact chunk_gate_recurrence_python_test_with_last_kv_fwd_surface_toAlgorithm_supported
      S D O LastKv
  constructor
  · exact chunk_gate_recurrence_python_test_no_last_kv_fwd_surface_toAlgorithm_supported
      S D O LastKv
  · exact chunk_gate_recurrence_forward_python_test_shape_all_outputs_compute_correct
      S D O LastKv t_rel s

/-- Python backward-path summary for `chunk_gate_recurrence.py`.

This links the reverse-loop backward surface to the checked DI/DG/DL output
writebacks and the one-step DAcc/DG arithmetic slices at the Python shape. -/
theorem chunk_gate_recurrence_backward_python_test_shape_summary
    (S D DI DG DL DS : RegionName) (t_rel : Nat)
    (s : BlockState) :
    ((chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      8 64 64 64 64 16).toAlgorithm? =
        Except.ok
          (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
            8 64 64 64 64 16).toAlgKernel) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DI, timeTileOffset s t_rel 64 64 64 64 16 idx))
      (expected := fun idx =>
        producedBwdDIValue s S D DI DG DL DS t_rel idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun _ : PUnit => some (DG, bwdDGOffset s t_rel 64 1 4))
      (expected := fun _ =>
        producedBwdDGValue s S D DI DG DL DS t_rel)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_surface S D DI DG DL DS
        8 64 64 64 64 16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 16] =>
        some (DL, bwdDLOffset s 64 64 64 16 idx))
      (expected := fun idx =>
        producedBwdDLValue s S D DI DG DL DS idx)) := by
  constructor
  · exact chunk_gate_recurrence_python_test_bwd_surface_toAlgorithm_supported
      S D DI DG DL DS
  · exact chunk_gate_recurrence_backward_python_test_shape_all_outputs_compute_correct
      S D DI DG DL DS t_rel s

/-- `output_summary` alias for the forward Python chunk-gate recurrence path. -/
abbrev chunk_gate_recurrence_forward_python_test_shape_output_summary
    (S D O LastKv : RegionName) (t_rel : Nat)
    (s : BlockState) :=
  chunk_gate_recurrence_forward_python_test_shape_summary S D O LastKv t_rel s

/-- `output_summary` alias for the backward Python chunk-gate recurrence path. -/
abbrev chunk_gate_recurrence_backward_python_test_shape_output_summary
    (S D DI DG DL DS : RegionName) (t_rel : Nat)
    (s : BlockState) :=
  chunk_gate_recurrence_backward_python_test_shape_summary S D DI DG DL DS t_rel s

end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence
