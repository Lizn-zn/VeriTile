import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of the no-`seq_idx`, non-causal path of
`bmm_chunk_fwd.py`'s `_bmm_chunk_fwd_kernel`.

The public benchmark exercises this path for both grouped and non-grouped
inputs. The source kernel's runtime `dot_dtype` cast is not represented here:
current `tl.dot` typing in the DSL is real-valued, so the `a`/`b` loads remain
in the compute carrier while the loop shape, masks, pointer advances, and final
destination dtype cast are preserved. The `HAS_SEQ_IDX` path uses signed
sentinels (`other = -1` and `other = -2`), and the causal path uses an early return; those are
not collapsed into this surface. -/
def bmm_chunk_fwd_no_seq_surface
    (A B Out : RegionName)
    (seqlen chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n

  a_base = A + pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  b_base = B + pid_b * $(stride_b_batch) +
    pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_base + offs_m[:, None] * $(stride_a_seqlen) +
    offs_k[None, :] * $(stride_ak)
  b_ptrs = b_base + offs_k[:, None] * $(stride_bk) +
    offs_n[None, :] * $(stride_b_seqlen)
  chunk_size_limit = tl.minimum($(chunk_size), $(seqlen) - pid_c * $(chunk_size))

  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a_mask = (offs_m[:, None] < chunk_size_limit) &
      (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K))
    b_mask = (offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
      (offs_n[None, :] < chunk_size_limit)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }

  out = (acc).to(Out.dtype.element_ty)
  out_ptrs = Out + pid_b * $(stride_out_batch) +
    pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head) +
    $(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn)
  out_mask = (offs_m[:, None] < $(chunk_size)) & (offs_n[None, :] < $(chunk_size))
  tl.store(out_ptrs, out, mask=out_mask)
}

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
  simp [exec, bmm_chunk_fwd_final_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.BmmChunkFwd
