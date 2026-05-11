import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented final `db` store slice of `bmm_chunk_bwd.py`'s
`_bmm_chunk_bwd_kernel`.

The full kernel computes `acc` with a chunk-size loop and optionally adds a
residual tile. This slice starts from a precomputed `Acc` tile and proves the
final masked writeback into `Db`, preserving the source program-id
decomposition and output stride shape. The caller supplies `chunk_size_limit`,
which in the source is `min(chunk_size, seqlen - pid_c * chunk_size)`. -/
def bmm_chunk_bwd_final_store_slice
    (Acc Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  pid_m = tl.program_id(axis=0) // $(num_pid_n)
  pid_n = tl.program_id(axis=0) % $(num_pid_n)
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  mask = (offs_m[:, None] < $(chunk_size_limit)) & (offs_n[None, :] < $(K))
  acc = tl.load(Acc + pid_b * $(stride_acc_batch) + pid_c * $(stride_acc_chunk) +
      pid_h * $(stride_acc_head) + offs_m[:, None] * $(stride_acc_m) +
      offs_n[None, :] * $(stride_acc_n), mask=mask, other=0.0)
  tl.store(Db + pid_b * $(stride_db_batch) + pid_c * $(chunk_size_limit) * $(stride_db_seqlen) +
      pid_h * $(stride_db_head) + offs_m[:, None] * $(stride_db_seqlen) +
      offs_n[None, :] * $(stride_db_k), acc, mask=mask)
}

def pidC (s : BlockState) (ngroups : Nat) : Nat :=
  s.pids 2 / ngroups

def pidH (s : BlockState) (ngroups : Nat) : Nat :=
  s.pids 2 - pidC s ngroups * ngroups

def pidM (s : BlockState) (num_pid_n : Nat) : Nat :=
  s.pids 0 / num_pid_n

def pidN (s : BlockState) (num_pid_n : Nat) : Nat :=
  s.pids 0 % num_pid_n

def mIndex (s : BlockState) (num_pid_n BLOCK_SIZE_M : Nat) (i : Fin BLOCK_SIZE_M) : Nat :=
  pidM s num_pid_n * BLOCK_SIZE_M + i.val

def nIndex (s : BlockState) (num_pid_n BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  pidN s num_pid_n * BLOCK_SIZE_N + j.val

def active
    (s : BlockState) (num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Prop :=
  mIndex s num_pid_n BLOCK_SIZE_M idx.1 < chunk_size_limit ∧
    nIndex s num_pid_n BLOCK_SIZE_N idx.2.1 < K

instance activeDecidable
    (s : BlockState) (num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) :
    Decidable (active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (num_pid_n ngroups stride_acc_batch stride_acc_chunk stride_acc_head
      stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  s.pids 1 * stride_acc_batch + pidC s ngroups * stride_acc_chunk +
    pidH s ngroups * stride_acc_head +
    mIndex s num_pid_n BLOCK_SIZE_M idx.1 * stride_acc_m +
    nIndex s num_pid_n BLOCK_SIZE_N idx.2.1 * stride_acc_n

def dbOffset
    (s : BlockState)
    (num_pid_n ngroups chunk_size_limit stride_db_batch stride_db_seqlen
      stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  s.pids 1 * stride_db_batch + pidC s ngroups * chunk_size_limit * stride_db_seqlen +
    pidH s ngroups * stride_db_head +
    mIndex s num_pid_n BLOCK_SIZE_M idx.1 * stride_db_seqlen +
    nIndex s num_pid_n BLOCK_SIZE_N idx.2.1 * stride_db_k

/-- Algorithm-layer correctness for the BMM chunk backward final `db` store
slice. -/
theorem bmm_chunk_bwd_final_store_slice_correct
    (Acc Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
          stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      let outAddr := dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
        stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N idx
      (exec (bmm_chunk_bwd_final_store_slice Acc Db num_pid_n ngroups
            chunk_size_limit K stride_acc_batch stride_acc_chunk stride_acc_head
            stride_acc_m stride_acc_n stride_db_batch stride_db_seqlen
            stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N) s).map
          (·.readMem Db outAddr)
        = some (if active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N idx then
            s.readMem Acc
              (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
                stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M
                BLOCK_SIZE_N idx)
          else s.readMem Db outAddr) := by
  intro idx
  simp [exec, bmm_chunk_bwd_final_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        pidC, pidH, pidM, pidN, mIndex, nIndex, active, accOffset, dbOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      s.pids 1 * stride_db_batch +
        (s.pids 2 / ngroups) * chunk_size_limit * stride_db_seqlen +
        (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_db_head +
        (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_db_seqlen +
        (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_db_k
  let valueFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
            s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K then
          some (s.readMem Acc
            (s.pids 1 * stride_acc_batch + (s.pids 2 / ngroups) * stride_acc_chunk +
              (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_acc_head +
              (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_acc_m +
              (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_acc_n))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
        s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Db (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE_M, BLOCK_SIZE_N])).readMem Db
        (offsetFn idx) =
    if P idx then
      s.readMem Acc
        (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
          stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx)
    else s.readMem Db (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
        s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K
  · simp [offsetFn, valueFn, P, active, accOffset, dbOffset, pidC, pidH, pidM,
      pidN, mIndex, nIndex, hActive]
  · simp [offsetFn, valueFn, P, active, accOffset, dbOffset, pidC, pidH, pidM,
      pidN, mIndex, nIndex, hActive]

/-- Compute-facing correctness for the BMM chunk backward final `db` store
slice. -/
theorem bmm_chunk_bwd_final_store_slice_compute_correct
    (Acc Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
          stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N idx)) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_final_store_slice Acc Db num_pid_n ngroups
        chunk_size_limit K stride_acc_batch stride_acc_chunk stride_acc_head
        stride_acc_m stride_acc_n stride_db_batch stride_db_seqlen stride_db_head
        stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
            stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
            stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_chunk_bwd_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := bmm_chunk_bwd_final_store_slice_correct Acc Db num_pid_n ngroups
    chunk_size_limit K stride_acc_batch stride_acc_chunk stride_acc_head
    stride_acc_m stride_acc_n stride_db_batch stride_db_seqlen stride_db_head
    stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.BmmChunkBwd
