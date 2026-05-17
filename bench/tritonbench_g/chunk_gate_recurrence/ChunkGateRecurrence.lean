import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
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
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
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

end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence
