import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of
`bmm_chunk_fwd.py`'s `_bmm_chunk_fwd_kernel`.

This covers both `HAS_SEQ_IDX=false` and `HAS_SEQ_IDX=true`; the sequence-index
loads use an `Int` region so the Python `other=-1` / `other=-2` sentinels are
represented directly. The causal early-return is represented as a guard around
the active body. The source kernel's runtime `dot_dtype` cast is preserved as a
surface dtype annotation on the `a`/`b` loads; current `tl.dot` typing in the DSL
still evaluates through the algorithm carrier while the loop shape, masks,
pointer advances, optional sequence filter, and final destination dtype cast are
preserved. -/
def bmm_chunk_fwd_surface
    (a_ptr b_ptr out_ptr : RegionName) (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (dot_dtype : TileDType)
    (IS_CAUSAL HAS_SEQ_IDX : Bool) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  active_block = true
  if IS_CAUSAL {
    active_block = pid_n * $(BLOCK_SIZE_N) < (pid_m + $(1)) * $(BLOCK_SIZE_M)
  }
  if active_block {

  a_ptr += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  b_ptr += pid_b * $(stride_b_batch) +
    pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
  if HAS_SEQ_IDX {
    seq_idx_ptr += pid_b * $(stride_seq_idx_batch) +
      pid_c * $(chunk_size) * $(stride_seq_idx_seqlen)
  }

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + offs_m[:, None] * $(stride_a_seqlen) +
    offs_k[None, :] * $(stride_ak)
  b_ptrs = b_ptr + offs_k[:, None] * $(stride_bk) +
    offs_n[None, :] * $(stride_b_seqlen)
  chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))

  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
      (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0).to(dot_dtype)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
      (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
    acc += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  if HAS_SEQ_IDX {
    chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
    seq_idx_m = tl.load(seq_idx_ptr + offs_m * $(stride_seq_idx_seqlen),
      mask=offs_m < chunk_size_limit, other=-1)
    seq_idx_n = tl.load(seq_idx_ptr + offs_n * $(stride_seq_idx_seqlen),
      mask=offs_n < chunk_size_limit, other=-2)
    acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
  }
  out = (acc).to(out_ptr.dtype.element_ty)
  out_ptr += pid_b * $(stride_out_batch) +
    pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
  out_ptrs = out_ptr + $(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn)
  tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) &
    (offs_n[None, :] < $(chunk_size)))
  }
}

/-- The full BMM chunk forward surface lowers to the algorithm layer, including
causal gating, optional sequence-index filtering, K-loop dot accumulation, and
the destination dtype cast. -/
theorem bmm_chunk_fwd_surface_toAlgorithm_supported
    (a_ptr b_ptr out_ptr : RegionName) (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (dot_dtype : TileDType)
    (IS_CAUSAL HAS_SEQ_IDX : Bool) :
    ∃ alg,
      (bmm_chunk_fwd_surface a_ptr b_ptr out_ptr seq_idx_ptr seqlen chunk_size
        K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
        stride_b_batch stride_b_seqlen stride_b_head stride_bk stride_out_batch
        stride_out_chunk stride_out_head stride_outm stride_outn
        stride_seq_idx_batch stride_seq_idx_seqlen BLOCK_SIZE_M BLOCK_SIZE_N
        BLOCK_SIZE_K dot_dtype IS_CAUSAL HAS_SEQ_IDX).toAlgorithm? =
        Except.ok alg := by
  simp [bmm_chunk_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final output-store slice of `bmm_chunk_fwd.py`'s
`_bmm_chunk_fwd_kernel`.

The full kernel computes `acc` with a K-looped dot product and optional
sequence-index filtering. This slice starts from a precomputed `Acc` tile and
proves the final masked writeback into `Out`, preserving the source program-id
decomposition and output stride shape. -/
def bmm_chunk_fwd_final_store_slice
    (Acc Out : RegionName)
    (num_pid_n ngroups chunk_size
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
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
  mask = (offs_m[:, None] < $(chunk_size)) & (offs_n[None, :] < $(chunk_size))
  acc = tl.load(Acc + pid_b * $(stride_acc_batch) + pid_c * $(stride_acc_chunk) +
      pid_h * $(stride_acc_head) + offs_m[:, None] * $(stride_acc_m) +
      offs_n[None, :] * $(stride_acc_n), mask=mask, other=0.0)
  tl.store(Out + pid_b * $(stride_out_batch) + pid_c * $(stride_out_chunk) +
      pid_h * $(stride_out_head) + offs_m[:, None] * $(stride_outm) +
      offs_n[None, :] * $(stride_outn), acc, mask=mask)
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
    (s : BlockState) (num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Prop :=
  mIndex s num_pid_n BLOCK_SIZE_M idx.1 < chunk_size ∧
    nIndex s num_pid_n BLOCK_SIZE_N idx.2.1 < chunk_size

instance activeDecidable
    (s : BlockState) (num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) :
    Decidable (active s num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
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

def outOffset
    (s : BlockState)
    (num_pid_n ngroups stride_out_batch stride_out_chunk stride_out_head
      stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  s.pids 1 * stride_out_batch + pidC s ngroups * stride_out_chunk +
    pidH s ngroups * stride_out_head +
    mIndex s num_pid_n BLOCK_SIZE_M idx.1 * stride_outm +
    nIndex s num_pid_n BLOCK_SIZE_N idx.2.1 * stride_outn

/-- Algorithm-layer correctness for the BMM chunk final output-store slice. -/
theorem bmm_chunk_fwd_final_store_slice_correct
    (Acc Out : RegionName)
    (num_pid_n ngroups chunk_size
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
          stride_out_head stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      let outAddr := outOffset s num_pid_n ngroups stride_out_batch
        stride_out_chunk stride_out_head stride_outm stride_outn BLOCK_SIZE_M
        BLOCK_SIZE_N idx
      (exec (bmm_chunk_fwd_final_store_slice Acc Out num_pid_n ngroups chunk_size
            stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m
            stride_acc_n stride_out_batch stride_out_chunk stride_out_head
            stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N) s).map
          (·.readMem Out outAddr)
        = some (if active s num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N idx then
            s.readMem Acc
              (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
                stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M
                BLOCK_SIZE_N idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, bmm_chunk_fwd_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        pidC, pidH, pidM, pidN, mIndex, nIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      s.pids 1 * stride_out_batch + (s.pids 2 / ngroups) * stride_out_chunk +
        (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_out_head +
        (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_outm +
        (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_outn
  let valueFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size ∧
            s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < chunk_size then
          some (s.readMem Acc
            (s.pids 1 * stride_acc_batch + (s.pids 2 / ngroups) * stride_acc_chunk +
              (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_acc_head +
              (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_acc_m +
              (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_acc_n))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size ∧
        s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < chunk_size
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, pidC, pidH, pidM, pidN, mIndex, nIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE_M, BLOCK_SIZE_N])).readMem Out
        (offsetFn idx) =
    if P idx then
      s.readMem Acc
        (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
          stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx)
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size ∧
        s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < chunk_size
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, pidC, pidH, pidM,
      pidN, mIndex, nIndex, hActive]
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, pidC, pidH, pidM,
      pidN, mIndex, nIndex, hActive]

/-- Compute-facing correctness for the BMM chunk final output-store slice. -/
theorem bmm_chunk_fwd_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (num_pid_n ngroups chunk_size
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] =>
        outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
          stride_out_head stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N idx)) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out num_pid_n ngroups
        chunk_size stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m
        stride_acc_n stride_out_batch stride_out_chunk stride_out_head stride_outm
        stride_outn BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
            stride_out_head stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
            stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_chunk_fwd_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := bmm_chunk_fwd_final_store_slice_correct Acc Out num_pid_n ngroups
    chunk_size stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m
    stride_acc_n stride_out_batch stride_out_chunk stride_out_head stride_outm
    stride_outn BLOCK_SIZE_M BLOCK_SIZE_N s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Active-lane variant for Python/autotuned configurations where the
boundary-checked inactive tile lanes may alias, but all lanes satisfying the
store mask have unique row-major output addresses. -/
theorem bmm_chunk_fwd_final_store_slice_active_compute_correct
    (Acc Out : RegionName)
    (num_pid_n ngroups chunk_size
      stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m stride_acc_n
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hNoCollision :
      ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
        active s num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N idx →
        ∀ k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
          active s num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N k →
          outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
            stride_out_head stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N k =
            outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
              stride_out_head stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N idx →
          k = idx) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out num_pid_n ngroups
        chunk_size stride_acc_batch stride_acc_chunk stride_acc_head stride_acc_m
        stride_acc_n stride_out_batch stride_out_chunk stride_out_head stride_outm
        stride_outn BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_pid_n chunk_size BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
            stride_out_head stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
            stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_chunk_fwd_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, bmm_chunk_fwd_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        pidC, pidH, pidM, pidN, mIndex, nIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex] at hExec
  subst s'
  let offsetFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      s.pids 1 * stride_out_batch + (s.pids 2 / ngroups) * stride_out_chunk +
        (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_out_head +
        (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_outm +
        (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_outn
  let valueFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size ∧
            s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < chunk_size then
          some (s.readMem Acc
            (s.pids 1 * stride_acc_batch + (s.pids 2 / ngroups) * stride_acc_chunk +
              (s.pids 2 - s.pids 2 / ngroups * ngroups) * stride_acc_head +
              (s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val) * stride_acc_m +
              (s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val) * stride_acc_n))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 / num_pid_n * BLOCK_SIZE_M + idx.1.val < chunk_size ∧
        s.pids 0 % num_pid_n * BLOCK_SIZE_N + idx.2.1.val < chunk_size
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE_M, BLOCK_SIZE_N])).readMem Out
        (offsetFn idx) =
    s.readMem Acc
      (accOffset s num_pid_n ngroups stride_acc_batch stride_acc_chunk
        stride_acc_head stride_acc_m stride_acc_n BLOCK_SIZE_M BLOCK_SIZE_N idx)
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ P idx]
  · have hP : P idx := by
      simpa [P, active, pidM, pidN, mIndex, nIndex] using hActive
    simp [offsetFn, valueFn, P, active, accOffset, outOffset, pidC, pidH, pidM,
      pidN, mIndex, nIndex, hP]
  · simpa [P, active, pidM, pidN, mIndex, nIndex] using hActive
  · intro k hPk heq
    exact hNoCollision idx hActive k
      (by simpa [P, active, pidM, pidN, mIndex, nIndex] using hPk)
      (by simpa [offsetFn, outOffset, pidC, pidH, pidM, pidN, mIndex, nIndex]
        using heq)

theorem bmm_chunk_fwd_python_chunk32_active_no_collision
    (num_pid_n ngroups stride_out_batch stride_out_chunk stride_out_head
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat) (s : BlockState) :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      active s num_pid_n 32 BLOCK_SIZE_M BLOCK_SIZE_N idx →
      ∀ k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
        active s num_pid_n 32 BLOCK_SIZE_M BLOCK_SIZE_N k →
        outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
          stride_out_head 32 1 BLOCK_SIZE_M BLOCK_SIZE_N k =
          outOffset s num_pid_n ngroups stride_out_batch stride_out_chunk
            stride_out_head 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx →
        k = idx := by
  rintro ⟨⟨ma, hma⟩, ⟨na, hna⟩, _⟩ hA
    ⟨⟨mb, hmb⟩, ⟨nb, hnb⟩, _⟩ hB hEq
  simp [active, outOffset, pidC, pidH, pidM, pidN, mIndex, nIndex] at hA hB hEq
  have hm : mb = ma := by omega
  have hn : nb = na := by omega
  subst mb
  subst nb
  rfl

theorem bmm_chunk_fwd_python_ungrouped_output_compute_correct
    (Acc Out : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out 1 1 32
        4096 1024 0 32 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  exact bmm_chunk_fwd_final_store_slice_active_compute_correct Acc Out
    1 1 32 4096 1024 0 32 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N s
    (bmm_chunk_fwd_python_chunk32_active_no_collision 1 1 4096 1024 0
      BLOCK_SIZE_M BLOCK_SIZE_N s)

theorem bmm_chunk_fwd_python_grouped_output_compute_correct
    (Acc Out : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out 1 4 32
        16384 4096 1024 32 1 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  exact bmm_chunk_fwd_final_store_slice_active_compute_correct Acc Out
    1 4 32 16384 4096 1024 32 1 16384 4096 1024 32 1
    BLOCK_SIZE_M BLOCK_SIZE_N s
    (bmm_chunk_fwd_python_chunk32_active_no_collision 1 4 16384 4096 1024
      BLOCK_SIZE_M BLOCK_SIZE_N s)

/-- Python case 1 full surface lowering: ungrouped, no `seq_idx`, non-causal,
with contiguous `(batch, seqlen, K) = (2, 128, 64)` inputs and fp16 dot dtype. -/
theorem bmm_chunk_fwd_python_case1_surface_toAlgorithm_supported
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 1
      8192 64 0 1
      8192 64 0 1
      4096 1024 0 32 1
      0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.false).toAlgorithm? = Except.ok alg := by
  exact bmm_chunk_fwd_surface_toAlgorithm_supported A B Out Seq
    128 32 64 1
    8192 64 0 1
    8192 64 0 1
    4096 1024 0 32 1
    0 0
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
    Bool.false Bool.false

/-- Python case 2 full surface lowering: grouped, `seq_idx`, causal. -/
theorem bmm_chunk_fwd_python_case2_surface_toAlgorithm_supported
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 4
      32768 256 64 1
      32768 256 64 1
      16384 4096 1024 32 1
      128 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.true Bool.true).toAlgorithm? = Except.ok alg := by
  exact bmm_chunk_fwd_surface_toAlgorithm_supported A B Out Seq
    128 32 64 4
    32768 256 64 1
    32768 256 64 1
    16384 4096 1024 32 1
    128 1
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
    Bool.true Bool.true

/-- Python case 3 full surface lowering: ungrouped with `seq_idx`, non-causal. -/
theorem bmm_chunk_fwd_python_case3_surface_toAlgorithm_supported
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 1
      8192 64 0 1
      8192 64 0 1
      4096 1024 0 32 1
      128 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.true).toAlgorithm? = Except.ok alg := by
  exact bmm_chunk_fwd_surface_toAlgorithm_supported A B Out Seq
    128 32 64 1
    8192 64 0 1
    8192 64 0 1
    4096 1024 0 32 1
    128 1
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
    Bool.false Bool.true

/-- Python case 4 full surface lowering: grouped, no `seq_idx`, non-causal. -/
theorem bmm_chunk_fwd_python_case4_surface_toAlgorithm_supported
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 4
      32768 256 64 1
      32768 256 64 1
      16384 4096 1024 32 1
      0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.false).toAlgorithm? = Except.ok alg := by
  exact bmm_chunk_fwd_surface_toAlgorithm_supported A B Out Seq
    128 32 64 4
    32768 256 64 1
    32768 256 64 1
    16384 4096 1024 32 1
    0 0
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
    Bool.false Bool.false

noncomputable def bmmChunkFwdSurfaceValue
    (s : BlockState) (A B Out : RegionName) (Seq : Region .int)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (dot_dtype : TileDType) (IS_CAUSAL HAS_SEQ_IDX : Bool)
    (offset : Nat) : ℝ :=
  match exec (bmm_chunk_fwd_surface A B Out Seq seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K dot_dtype IS_CAUSAL HAS_SEQ_IDX) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem bmm_chunk_fwd_surface_output_compute_correct
    (A B Out : RegionName) (Seq : Region .int)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (dot_dtype : TileDType) (IS_CAUSAL HAS_SEQ_IDX : Bool)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_surface A B Out Seq seqlen chunk_size K ngroups
        stride_a_batch stride_a_seqlen stride_a_head stride_ak
        stride_b_batch stride_b_seqlen stride_b_head stride_bk
        stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
        stride_seq_idx_batch stride_seq_idx_seqlen BLOCK_SIZE_M BLOCK_SIZE_N
        BLOCK_SIZE_K dot_dtype IS_CAUSAL HAS_SEQ_IDX)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 chunk_size BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 ngroups stride_out_batch stride_out_chunk stride_out_head
            stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkFwdSurfaceValue s A B Out Seq seqlen chunk_size K ngroups
          stride_a_batch stride_a_seqlen stride_a_head stride_ak
          stride_b_batch stride_b_seqlen stride_b_head stride_bk
          stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
          stride_seq_idx_batch stride_seq_idx_seqlen BLOCK_SIZE_M BLOCK_SIZE_N
          BLOCK_SIZE_K dot_dtype IS_CAUSAL HAS_SEQ_IDX
          (outOffset s 1 ngroups stride_out_batch stride_out_chunk stride_out_head
            stride_outm stride_outn BLOCK_SIZE_M BLOCK_SIZE_N idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bmm_chunk_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [bmmChunkFwdSurfaceValue, hExec]

/-- Public Python case 1 summary: full surface lowering plus final-store
ComputeCorrect for the observed ungrouped output layout. -/
theorem bmm_chunk_fwd_python_case1_store_summary
    (A B Acc Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 1
      8192 64 0 1
      8192 64 0 1
      4096 1024 0 32 1
      0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out 1 1 32
        4096 1024 0 32 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case1_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_python_ungrouped_output_compute_correct Acc Out
      BLOCK_SIZE_M BLOCK_SIZE_N s

/-- Public Python case 2 summary. -/
theorem bmm_chunk_fwd_python_case2_store_summary
    (A B Acc Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 4
      32768 256 64 1
      32768 256 64 1
      16384 4096 1024 32 1
      128 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.true Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out 1 4 32
        16384 4096 1024 32 1 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case2_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_python_grouped_output_compute_correct Acc Out
      BLOCK_SIZE_M BLOCK_SIZE_N s

/-- Public Python case 3 summary. -/
theorem bmm_chunk_fwd_python_case3_store_summary
    (A B Acc Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 1
      8192 64 0 1
      8192 64 0 1
      4096 1024 0 32 1
      128 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out 1 1 32
        4096 1024 0 32 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case3_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_python_ungrouped_output_compute_correct Acc Out
      BLOCK_SIZE_M BLOCK_SIZE_N s

/-- Public Python case 4 summary. -/
theorem bmm_chunk_fwd_python_case4_store_summary
    (A B Acc Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 4
      32768 256 64 1
      32768 256 64 1
      16384 4096 1024 32 1
      0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_final_store_slice Acc Out 1 4 32
        16384 4096 1024 32 1 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        s.readMem Acc
          (accOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case4_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_python_grouped_output_compute_correct Acc Out
      BLOCK_SIZE_M BLOCK_SIZE_N s




















theorem bmm_chunk_fwd_python_case1_output_summary
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 1
      8192 64 0 1
      8192 64 0 1
      4096 1024 0 32 1
      0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_surface A B Out Seq
        128 32 64 1
        8192 64 0 1
        8192 64 0 1
        4096 1024 0 32 1
        0 0
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
        Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkFwdSurfaceValue s A B Out Seq
          128 32 64 1 8192 64 0 1 8192 64 0 1 4096 1024 0 32 1
          0 0 BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
          Bool.false Bool.false
          (outOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case1_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_surface_output_compute_correct A B Out Seq
      128 32 64 1 8192 64 0 1 8192 64 0 1 4096 1024 0 32 1
      0 0 BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.false s

theorem bmm_chunk_fwd_python_case2_output_summary
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 4
      32768 256 64 1
      32768 256 64 1
      16384 4096 1024 32 1
      128 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.true Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_surface A B Out Seq
        128 32 64 4
        32768 256 64 1
        32768 256 64 1
        16384 4096 1024 32 1
        128 1
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
        Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkFwdSurfaceValue s A B Out Seq
          128 32 64 4 32768 256 64 1 32768 256 64 1
          16384 4096 1024 32 1 128 1 BLOCK_SIZE_M BLOCK_SIZE_N
          BLOCK_SIZE_K TileDType.fp16 Bool.true Bool.true
          (outOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case2_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_surface_output_compute_correct A B Out Seq
      128 32 64 4 32768 256 64 1 32768 256 64 1
      16384 4096 1024 32 1 128 1 BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K TileDType.fp16 Bool.true Bool.true s

theorem bmm_chunk_fwd_python_case3_output_summary
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 1
      8192 64 0 1
      8192 64 0 1
      4096 1024 0 32 1
      128 1
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_surface A B Out Seq
        128 32 64 1
        8192 64 0 1
        8192 64 0 1
        4096 1024 0 32 1
        128 1
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
        Bool.false Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkFwdSurfaceValue s A B Out Seq
          128 32 64 1 8192 64 0 1 8192 64 0 1 4096 1024 0 32 1
          128 1 BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
          Bool.false Bool.true
          (outOffset s 1 1 4096 1024 0 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case3_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_surface_output_compute_correct A B Out Seq
      128 32 64 1 8192 64 0 1 8192 64 0 1 4096 1024 0 32 1
      128 1 BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.true s

theorem bmm_chunk_fwd_python_case4_output_summary
    (A B Out : RegionName) (Seq : Region .int)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (s : BlockState) :
    (∃ alg, (bmm_chunk_fwd_surface A B Out Seq
      128 32 64 4
      32768 256 64 1
      32768 256 64 1
      16384 4096 1024 32 1
      0 0
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
      Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := bmm_chunk_fwd_surface A B Out Seq
        128 32 64 4
        32768 256 64 1
        32768 256 64 1
        16384 4096 1024 32 1
        0 0
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K TileDType.fp16
        Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 32 BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (Out,
          outOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        bmmChunkFwdSurfaceValue s A B Out Seq
          128 32 64 4 32768 256 64 1 32768 256 64 1
          16384 4096 1024 32 1 0 0 BLOCK_SIZE_M BLOCK_SIZE_N
          BLOCK_SIZE_K TileDType.fp16 Bool.false Bool.false
          (outOffset s 1 4 16384 4096 1024 32 1 BLOCK_SIZE_M BLOCK_SIZE_N idx))) := by
  constructor
  · exact bmm_chunk_fwd_python_case4_surface_toAlgorithm_supported A B Out Seq
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K
  · exact bmm_chunk_fwd_surface_output_compute_correct A B Out Seq
      128 32 64 4 32768 256 64 1 32768 256 64 1
      16384 4096 1024 32 1 0 0 BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K TileDType.fp16 Bool.false Bool.false s

end VeriTile.Bench.TritonBenchG.BmmChunkFwd
