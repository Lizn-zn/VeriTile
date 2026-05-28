import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `bmm_chunk_bwd.py`'s `_bmm_chunk_bwd_kernel`.

The Python wrapper casts `dout` and `a` to a runtime `dot_dtype` constexpr. This
surface preserves those casts as dtype annotations; current `tl.dot` typing in
the DSL still evaluates through the algorithm carrier while preserving the loop
structure, masks, pointer advances, residual case, and final destination dtype
cast. -/
def bmm_chunk_bwd_surface
    (A Dout db_ptr Res : RegionName)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      stride_res_batch stride_res_seqlen stride_res_head stride_res_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (dot_dtype : TileDType)
    (HAS_RESIDUAL : Bool) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n

  A += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  Dout += pid_b * $(stride_dout_batch) +
    pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
  dout_ptrs = Dout +
    offs_m[:, None] * $(stride_dout_csize_n) +
    offs_cs[None, :] * $(stride_dout_csize_m)
  a_ptrs = A +
    offs_cs[:, None] * $(stride_a_seqlen) +
    offs_n[None, :] * $(stride_ak)
  chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))

  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for cs in range($(0), tl.cdiv(chunk_size_limit, $(BLOCK_SIZE_CS)), $(1)) {
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_cs[None, :] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)),
      other=0.0).to(dot_dtype)
    a = tl.load(a_ptrs,
      mask=(offs_cs[:, None] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)) &
        (offs_n[None, :] < $(K)), other=0.0).to(dot_dtype)
    acc += tl.dot(dout, a)
    dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
    a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
  }

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  if HAS_RESIDUAL {
    Res += pid_b * $(stride_res_batch) +
      pid_c * $(chunk_size) * $(stride_res_seqlen) + pid_h * $(stride_res_head)
    res_ptrs = Res +
      offs_m[:, None] * $(stride_res_seqlen) +
      offs_n[None, :] * $(stride_res_k)
    res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
      (offs_n[None, :] < $(K))).to(tl.float32)
    acc += res
  }

  db = (acc).to(db_ptr.dtype.element_ty)
  db_ptr += pid_b * $(stride_db_batch) +
    pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
  db_ptrs = db_ptr + offs_m[:, None] * $(stride_db_seqlen) + offs_n[None, :] * $(stride_db_k)
  tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) &
    (offs_n[None, :] < $(K)))
}

/-- The full BMM chunk backward surface lowers to the algorithm layer. -/
theorem bmm_chunk_bwd_surface_toAlgorithm_supported
    (A Dout db_ptr Res : RegionName)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      stride_res_batch stride_res_seqlen stride_res_head stride_res_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (dot_dtype : TileDType)
    (HAS_RESIDUAL : Bool) :
    ∃ alg, (bmm_chunk_bwd_surface A Dout db_ptr Res seqlen chunk_size K
      ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m
      stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head
      stride_db_k stride_res_batch stride_res_seqlen stride_res_head
      stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS dot_dtype
      HAS_RESIDUAL).toAlgorithm? = Except.ok alg := by
  simp [bmm_chunk_bwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
  simp [exec, bmm_chunk_bwd_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- Residual path final store slice for `bmm_chunk_bwd.py`.

This models the `HAS_RESIDUAL` branch after the dot-product accumulation:
load the accumulated tile and residual tile, add them, then store the resulting
`db` tile with the Python output layout. -/
def bmm_chunk_bwd_residual_final_store_slice
    (Acc Res Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_res_batch stride_res_chunk stride_res_head stride_res_m stride_res_n
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
  res = tl.load(Res + pid_b * $(stride_res_batch) + pid_c * $(stride_res_chunk) +
      pid_h * $(stride_res_head) + offs_m[:, None] * $(stride_res_m) +
      offs_n[None, :] * $(stride_res_n), mask=mask, other=0.0)
  db = acc + res
  tl.store(Db + pid_b * $(stride_db_batch) + pid_c * $(chunk_size_limit) * $(stride_db_seqlen) +
      pid_h * $(stride_db_head) + offs_m[:, None] * $(stride_db_seqlen) +
      offs_n[None, :] * $(stride_db_k), db, mask=mask)
}

def resOffset
    (s : BlockState)
    (num_pid_n ngroups stride_res_batch stride_res_chunk stride_res_head
      stride_res_m stride_res_n BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  s.pids 1 * stride_res_batch + pidC s ngroups * stride_res_chunk +
    pidH s ngroups * stride_res_head +
    mIndex s num_pid_n BLOCK_SIZE_M idx.1 * stride_res_m +
    nIndex s num_pid_n BLOCK_SIZE_N idx.2.1 * stride_res_n

noncomputable def residualFinalSpec
    (s : BlockState) (Acc Res : RegionName)
    (num_pid_n ngroups
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_res_batch stride_res_chunk stride_res_head stride_res_m stride_res_n
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : ℝ :=
  s.readMem Acc
      (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
        stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx) +
    s.readMem Res
      (resOffset s num_pid_n ngroups stride_res_batch stride_res_chunk
        stride_res_head stride_res_m stride_res_n BLOCK_SIZE_M BLOCK_SIZE_N idx)

theorem bmm_chunk_bwd_residual_final_store_slice_correct
    (Acc Res Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_res_batch stride_res_chunk stride_res_head stride_res_m stride_res_n
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
      (exec (bmm_chunk_bwd_residual_final_store_slice Acc Res Db num_pid_n
            ngroups chunk_size_limit K stride_acc_batch stride_acc_chunk
            stride_acc_head stride_acc_m stride_acc_n stride_res_batch
            stride_res_chunk stride_res_head stride_res_m stride_res_n
            stride_db_batch stride_db_seqlen stride_db_head stride_db_k
            BLOCK_SIZE_M BLOCK_SIZE_N) s).map (·.readMem Db outAddr)
        = some (if active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N idx then
            residualFinalSpec s Acc Res num_pid_n ngroups stride_acc_batch
              stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
              stride_res_batch stride_res_chunk stride_res_head stride_res_m
              stride_res_n BLOCK_SIZE_M BLOCK_SIZE_N idx
          else s.readMem Db outAddr) := by
  intro idx
  simp [exec, bmm_chunk_bwd_residual_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        pidC, pidH, pidM, pidN, mIndex, nIndex, active, accOffset, resOffset,
        dbOffset, residualFinalSpec, TileShape.dropInsertedIndex]
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
        (Option.map₂ (fun a r => a + r)
          (if s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
              s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K then
            some (s.readMem Acc
              (s.pids 1 * stride_acc_batch + (s.pids 2 / ngroups) * stride_acc_chunk +
                (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_acc_head +
                (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_acc_m +
                (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_acc_n))
          else some (0.0 : ℝ))
          (if s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
              s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K then
            some (s.readMem Res
              (s.pids 1 * stride_res_batch + (s.pids 2 / ngroups) * stride_res_chunk +
                (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_res_head +
                (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_res_m +
                (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_res_n))
          else some (0.0 : ℝ)))
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
      residualFinalSpec s Acc Res num_pid_n ngroups stride_acc_batch
        stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
        stride_res_batch stride_res_chunk stride_res_head stride_res_m
        stride_res_n BLOCK_SIZE_M BLOCK_SIZE_N idx
    else s.readMem Db (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
        s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K
  · simp [offsetFn, valueFn, P, active, residualFinalSpec, accOffset, resOffset,
      dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex, hActive]
  · simp [offsetFn, valueFn, P, active, residualFinalSpec, accOffset, resOffset,
      dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex, hActive]

theorem bmm_chunk_bwd_residual_final_store_slice_compute_correct
    (Acc Res Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_res_batch stride_res_chunk stride_res_head stride_res_m stride_res_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
          stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N idx)) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_residual_final_store_slice Acc Res Db num_pid_n
        ngroups chunk_size_limit K stride_acc_batch stride_acc_chunk
        stride_acc_head stride_acc_m stride_acc_n stride_res_batch
        stride_res_chunk stride_res_head stride_res_m stride_res_n
        stride_db_batch stride_db_seqlen stride_db_head stride_db_k
        BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
            stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        residualFinalSpec s Acc Res num_pid_n ngroups stride_acc_batch
          stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
          stride_res_batch stride_res_chunk stride_res_head stride_res_m
          stride_res_n BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_chunk_bwd_residual_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := bmm_chunk_bwd_residual_final_store_slice_correct Acc Res Db
    num_pid_n ngroups chunk_size_limit K stride_acc_batch stride_acc_chunk
    stride_acc_head stride_acc_m stride_acc_n stride_res_batch stride_res_chunk
    stride_res_head stride_res_m stride_res_n stride_db_batch stride_db_seqlen
    stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Active-lane variant for the non-residual backward `db` store. Python
autotune may choose block sizes larger than the active `chunk_size=32` rows or
`K=64` columns; only masked-in lanes need collision freedom. -/
theorem bmm_chunk_bwd_final_store_slice_active_compute_correct
    (Acc Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hNoCollision :
      ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
        active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N idx →
        ∀ k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
          active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N k →
          dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
            stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M
              BLOCK_SIZE_N k =
            dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
              stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M
                BLOCK_SIZE_N idx →
          k = idx) :
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
  simp [exec, bmm_chunk_bwd_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        pidC, pidH, pidM, pidN, mIndex, nIndex, active, accOffset, dbOffset,
        TileShape.dropInsertedIndex] at hExec
  subst s'
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
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Db (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE_M, BLOCK_SIZE_N])).readMem Db
        (offsetFn idx) =
    s.readMem Acc
      (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
        stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx)
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ P idx]
  · have hP : P idx := by
      simpa [P, active, pidM, pidN, mIndex, nIndex] using hActive
    simp [offsetFn, valueFn, P, active, accOffset, dbOffset, pidC, pidH, pidM,
      pidN, mIndex, nIndex, hP]
  · simpa [P, active, pidM, pidN, mIndex, nIndex] using hActive
  · intro k hPk heq
    exact hNoCollision idx hActive k
      (by simpa [P, active, pidM, pidN, mIndex, nIndex] using hPk)
      (by simpa [offsetFn, dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex]
        using heq)

/-- Active-lane variant for the residual backward `db` store. -/
theorem bmm_chunk_bwd_residual_final_store_slice_active_compute_correct
    (Acc Res Db : RegionName)
    (num_pid_n ngroups chunk_size_limit K
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_res_batch stride_res_chunk stride_res_head stride_res_m stride_res_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hNoCollision :
      ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
        active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N idx →
        ∀ k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
          active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N k →
          dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
            stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M
              BLOCK_SIZE_N k =
            dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
              stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M
                BLOCK_SIZE_N idx →
          k = idx) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_residual_final_store_slice Acc Res Db num_pid_n
        ngroups chunk_size_limit K stride_acc_batch stride_acc_chunk
        stride_acc_head stride_acc_m stride_acc_n stride_res_batch
        stride_res_chunk stride_res_head stride_res_m stride_res_n
        stride_db_batch stride_db_seqlen stride_db_head stride_db_k
        BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n chunk_size_limit K BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n ngroups chunk_size_limit stride_db_batch
            stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        residualFinalSpec s Acc Res num_pid_n ngroups stride_acc_batch
          stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
          stride_res_batch stride_res_chunk stride_res_head stride_res_m
          stride_res_n BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_chunk_bwd_residual_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, bmm_chunk_bwd_residual_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        pidC, pidH, pidM, pidN, mIndex, nIndex, active, accOffset, resOffset,
        dbOffset, residualFinalSpec, TileShape.dropInsertedIndex] at hExec
  subst s'
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
        (Option.map₂ (fun a r => a + r)
          (if s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
              s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K then
            some (s.readMem Acc
              (s.pids 1 * stride_acc_batch + (s.pids 2 / ngroups) * stride_acc_chunk +
                (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_acc_head +
                (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_acc_m +
                (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_acc_n))
          else some (0.0 : ℝ))
          (if s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
              s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K then
            some (s.readMem Res
              (s.pids 1 * stride_res_batch + (s.pids 2 / ngroups) * stride_res_chunk +
                (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_res_head +
                (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_res_m +
                (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_res_n))
          else some (0.0 : ℝ)))
  let P : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size_limit ∧
        s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < K
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Db (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE_M, BLOCK_SIZE_N])).readMem Db
        (offsetFn idx) =
    residualFinalSpec s Acc Res num_pid_n ngroups stride_acc_batch
      stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_res_batch stride_res_chunk stride_res_head stride_res_m
      stride_res_n BLOCK_SIZE_M BLOCK_SIZE_N idx
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ P idx]
  · have hP : P idx := by
      simpa [P, active, pidM, pidN, mIndex, nIndex] using hActive
    simp [offsetFn, valueFn, P, active, residualFinalSpec, accOffset, resOffset,
      dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex, hP]
  · simpa [P, active, pidM, pidN, mIndex, nIndex] using hActive
  · intro k hPk heq
    exact hNoCollision idx hActive k
      (by simpa [P, active, pidM, pidN, mIndex, nIndex] using hPk)
      (by simpa [offsetFn, dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex]
        using heq)

theorem bmm_chunk_bwd_python_ungrouped_active_no_collision
    (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N : Nat) (s : BlockState) :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N idx →
      ∀ k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
        active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N k →
        dbOffset s num_pid_n 1 32 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N k =
          dbOffset s num_pid_n 1 32 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N idx →
        k = idx := by
  rintro ⟨⟨ma, hma⟩, ⟨na, hna⟩, _⟩ hA
    ⟨⟨mb, hmb⟩, ⟨nb, hnb⟩, _⟩ hB hEq
  simp [active, dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex] at hA hB hEq
  have hm : mb = ma := by omega
  have hn : nb = na := by omega
  subst mb
  subst nb
  rfl

theorem bmm_chunk_bwd_python_grouped_active_no_collision
    (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N : Nat) (s : BlockState) :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N idx →
      ∀ k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
        active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N k →
        dbOffset s num_pid_n 4 32 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N k =
          dbOffset s num_pid_n 4 32 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N idx →
        k = idx := by
  rintro ⟨⟨ma, hma⟩, ⟨na, hna⟩, _⟩ hA
    ⟨⟨mb, hmb⟩, ⟨nb, hnb⟩, _⟩ hB hEq
  simp [active, dbOffset, pidC, pidH, pidM, pidN, mIndex, nIndex] at hA hB hEq
  have hm : mb = ma := by omega
  have hn : nb = na := by omega
  subst mb
  subst nb
  rfl

theorem bmm_chunk_bwd_python_case1_compute_correct
    (Acc Db : RegionName) (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_final_store_slice Acc Db num_pid_n 1 32 64
        8192 2048 0 64 1 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n 1 32 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s num_pid_n 1 8192 2048 0 64 1
            BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  exact bmm_chunk_bwd_final_store_slice_active_compute_correct Acc Db
    num_pid_n 1 32 64 8192 2048 0 64 1 8192 64 0 1
    BLOCK_SIZE_M BLOCK_SIZE_N s
    (bmm_chunk_bwd_python_ungrouped_active_no_collision num_pid_n
      BLOCK_SIZE_M BLOCK_SIZE_N s)

theorem bmm_chunk_bwd_python_case2_residual_compute_correct
    (Acc Res Db : RegionName) (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_residual_final_store_slice Acc Res Db num_pid_n 4
        32 64 32768 8192 2048 256 1 32768 8192 64 256 1
        32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n 4 32 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        residualFinalSpec s Acc Res num_pid_n 4 32768 8192 2048 256 1
          32768 8192 64 256 1 BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  exact bmm_chunk_bwd_residual_final_store_slice_active_compute_correct Acc Res Db
    num_pid_n 4 32 64 32768 8192 2048 256 1 32768 8192 64 256 1
    32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N s
    (bmm_chunk_bwd_python_grouped_active_no_collision num_pid_n
      BLOCK_SIZE_M BLOCK_SIZE_N s)

/-- Python case 1 full surface lowering: ungrouped, no residual, fp16 dot dtype. -/
theorem bmm_chunk_bwd_python_case1_surface_toAlgorithm_supported
    (A Dout Db Res : RegionName)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) :
    ∃ alg, (bmm_chunk_bwd_surface A Dout Db Res
      128 32 64 1
      8192 64 0 1
      4096 1024 0 32 1
      8192 64 0 1
      0 0 0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
      Bool.false).toAlgorithm? = Except.ok alg := by
  exact bmm_chunk_bwd_surface_toAlgorithm_supported A Dout Db Res
    128 32 64 1
    8192 64 0 1
    4096 1024 0 32 1
    8192 64 0 1
    0 0 0 0
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
    Bool.false

/-- Python case 2 full surface lowering: grouped with residual, fp16 dot dtype. -/
theorem bmm_chunk_bwd_python_case2_surface_toAlgorithm_supported
    (A Dout Db Res : RegionName)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) :
    ∃ alg, (bmm_chunk_bwd_surface A Dout Db Res
      128 32 64 4
      32768 256 64 1
      16384 4096 1024 32 1
      32768 256 64 1
      32768 256 64 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
      Bool.true).toAlgorithm? = Except.ok alg := by
  exact bmm_chunk_bwd_surface_toAlgorithm_supported A Dout Db Res
    128 32 64 4
    32768 256 64 1
    16384 4096 1024 32 1
    32768 256 64 1
    32768 256 64 1
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
    Bool.true

noncomputable def bmmChunkBwdSurfaceValue
    (s : BlockState) (A Dout Db Res : RegionName)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      stride_res_batch stride_res_seqlen stride_res_head stride_res_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (dot_dtype : TileDType) (HAS_RESIDUAL : Bool) (offset : Nat) : ℝ :=
  match exec (bmm_chunk_bwd_surface A Dout Db Res seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m
      stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head
      stride_db_k stride_res_batch stride_res_seqlen stride_res_head
      stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS dot_dtype
      HAS_RESIDUAL) s with
  | some s' => s'.readMem Db offset
  | none => 0.0

theorem bmm_chunk_bwd_surface_output_compute_correct
    (A Dout Db Res : RegionName)
    (num_pid_n seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      stride_res_batch stride_res_seqlen stride_res_head stride_res_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (dot_dtype : TileDType) (HAS_RESIDUAL : Bool) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_surface A Dout Db Res seqlen chunk_size K ngroups
        stride_a_batch stride_a_seqlen stride_a_head stride_ak
        stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m
        stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head
        stride_db_k stride_res_batch stride_res_seqlen stride_res_head
        stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS dot_dtype
        HAS_RESIDUAL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n chunk_size K BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n ngroups chunk_size stride_db_batch
            stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M
            BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkBwdSurfaceValue s A Dout Db Res seqlen chunk_size K ngroups
          stride_a_batch stride_a_seqlen stride_a_head stride_ak
          stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m
          stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head
          stride_db_k stride_res_batch stride_res_seqlen stride_res_head
          stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS dot_dtype
          HAS_RESIDUAL
          (dbOffset s num_pid_n ngroups chunk_size stride_db_batch
            stride_db_seqlen stride_db_head stride_db_k BLOCK_SIZE_M
            BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_chunk_bwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [bmmChunkBwdSurfaceValue, hExec]

/-- Public Python case 1 summary: full surface lowering plus final `Db`
store ComputeCorrect for the observed ungrouped output layout. -/
theorem bmm_chunk_bwd_python_case1_store_summary
    (A Dout Acc Db Res : RegionName)
    (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (s : BlockState) :
    (∃ alg, (bmm_chunk_bwd_surface A Dout Db Res
      128 32 64 1
      8192 64 0 1
      4096 1024 0 32 1
      8192 64 0 1
      0 0 0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_final_store_slice Acc Db num_pid_n 1 32 64
        8192 2048 0 64 1 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n 1 32 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s num_pid_n 1 8192 2048 0 64 1
            BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_bwd_python_case1_surface_toAlgorithm_supported A Dout Db Res
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS
  · exact bmm_chunk_bwd_python_case1_compute_correct Acc Db num_pid_n
      BLOCK_SIZE_M BLOCK_SIZE_N s

/-- Public Python case 2 summary. -/
theorem bmm_chunk_bwd_python_case2_store_summary
    (A Dout Acc Db Res : RegionName)
    (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (s : BlockState) :
    (∃ alg, (bmm_chunk_bwd_surface A Dout Db Res
      128 32 64 4
      32768 256 64 1
      16384 4096 1024 32 1
      32768 256 64 1
      32768 256 64 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_residual_final_store_slice Acc Res Db num_pid_n 4
        32 64 32768 8192 2048 256 1 32768 8192 64 256 1
        32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n 4 32 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        residualFinalSpec s Acc Res num_pid_n 4 32768 8192 2048 256 1
          32768 8192 64 256 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  constructor
  · exact bmm_chunk_bwd_python_case2_surface_toAlgorithm_supported A Dout Db Res
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS
  · exact bmm_chunk_bwd_python_case2_residual_compute_correct Acc Res Db num_pid_n
      BLOCK_SIZE_M BLOCK_SIZE_N s




















theorem bmm_chunk_bwd_python_case1_output_summary
    (A Dout Db Res : RegionName)
    (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (s : BlockState) :
    (∃ alg, (bmm_chunk_bwd_surface A Dout Db Res
      128 32 64 1
      8192 64 0 1
      4096 1024 0 32 1
      8192 64 0 1
      0 0 0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_surface A Dout Db Res
        128 32 64 1
        8192 64 0 1
        4096 1024 0 32 1
        8192 64 0 1
        0 0 0 0
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
        Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n 1 32 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkBwdSurfaceValue s A Dout Db Res
          128 32 64 1 8192 64 0 1 4096 1024 0 32 1 8192 64 0 1
          0 0 0 0 BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
          Bool.false
          (dbOffset s num_pid_n 1 32 8192 64 0 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_bwd_python_case1_surface_toAlgorithm_supported A Dout Db Res
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS
  · exact bmm_chunk_bwd_surface_output_compute_correct A Dout Db Res num_pid_n
      128 32 64 1 8192 64 0 1 4096 1024 0 32 1 8192 64 0 1
      0 0 0 0 BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
      Bool.false s

theorem bmm_chunk_bwd_python_case2_output_summary
    (A Dout Db Res : RegionName)
    (num_pid_n BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (s : BlockState) :
    (∃ alg, (bmm_chunk_bwd_surface A Dout Db Res
      128 32 64 4
      32768 256 64 1
      16384 4096 1024 32 1
      32768 256 64 1
      32768 256 64 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_bwd_surface A Dout Db Res
        128 32 64 4
        32768 256 64 1
        16384 4096 1024 32 1
        32768 256 64 1
        32768 256 64 1
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS TileDType.fp16
        Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n 32 64 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Db,
          dbOffset s num_pid_n 4 32 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkBwdSurfaceValue s A Dout Db Res
          128 32 64 4 32768 256 64 1 16384 4096 1024 32 1
          32768 256 64 1 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N
          BLOCK_SIZE_CS TileDType.fp16 Bool.true
          (dbOffset s num_pid_n 4 32 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_bwd_python_case2_surface_toAlgorithm_supported A Dout Db Res
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS
  · exact bmm_chunk_bwd_surface_output_compute_correct A Dout Db Res num_pid_n
      128 32 64 4 32768 256 64 1 16384 4096 1024 32 1
      32768 256 64 1 32768 256 64 1 BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_CS TileDType.fp16 Bool.true s

end VeriTile.Bench.TritonBenchG.BmmChunkBwd
