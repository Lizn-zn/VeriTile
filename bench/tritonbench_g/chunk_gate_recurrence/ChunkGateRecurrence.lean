import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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
  offset_bh = tl.program_id(axis=0)
  offset_d = tl.program_id(axis=1)
  offset_s = tl.program_id(axis=2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.load(Acc + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], acc)
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

end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence
