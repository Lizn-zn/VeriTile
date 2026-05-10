import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented first output-block store slice of
`block_sparse_attn.py`'s `block_sparse_attention_kernel`.

The full kernel walks a CSR sparse layout and accumulates one or two D blocks.
This slice starts from a precomputed first-block `Acc` tile and proves the final
masked writeback into `Out`, preserving the source `off_bh` decomposition and
`offs_m < total_seq_len` row mask. -/
def block_sparse_attn_output_store_slice
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(axis=0)
  off_bh = tl.program_id(axis=1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_D))
  mask = (offs_m[:, None] < $(total_seq_len)) & (offs_d[None, :] < $(BLOCK_D))
  acc = tl.load(Acc + off_b * $(stride_acc_b) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_b * $(stride_ob) + off_h * $(stride_oh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :], acc, mask=mask)
}

def offH (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 % num_heads

def offB (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 / num_heads

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (total_seq_len BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Prop :=
  mIndex s BLOCK_M idx.1 < total_seq_len

instance activeDecidable
    (s : BlockState) (total_seq_len BLOCK_M BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) :
    Decidable (active s total_seq_len BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (num_heads stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_acc_b + offH s num_heads * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState)
    (num_heads stride_ob stride_oh stride_om BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_ob + offH s num_heads * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx

noncomputable def accStoreValue
    (s : BlockState) (Acc : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m
      stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : ℝ :=
  WithBot.unbotD 0
    (if active s total_seq_len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s num_heads stride_acc_b stride_acc_h stride_acc_m
          stride_acc_d BLOCK_M idx))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the first block-sparse output store. -/
theorem block_sparse_attn_output_store_slice_correct
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      let outAddr := outOffset s num_heads stride_ob stride_oh stride_om
        BLOCK_M idx
      (exec (block_sparse_attn_output_store_slice Acc Out num_heads
            total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
            stride_ob stride_oh stride_om BLOCK_M BLOCK_D) s).map
          (·.readMem Out outAddr)
        = some (if active s total_seq_len BLOCK_M idx then
            accStoreValue s Acc num_heads total_seq_len stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, block_sparse_attn_output_store_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, offH, offB, mIndex, dIndex,
        active, accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_D] → Nat :=
    fun idx =>
      s.pids 1 / num_heads * stride_ob + s.pids 1 % num_heads * stride_oh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, BLOCK_D] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
          some (s.readMem Acc
            (s.pids 1 / num_heads * stride_acc_b +
              s.pids 1 % num_heads * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_D] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offH, offB, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_D])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc num_heads total_seq_len stride_acc_b stride_acc_h
        stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  · rfl
  · rfl

/-- Compute-facing correctness for the first block-sparse output store. -/
theorem block_sparse_attn_output_store_slice_compute_correct
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_slice Acc Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
          active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out,
          outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        accStoreValue s Acc num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [block_sparse_attn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := block_sparse_attn_output_store_slice_correct Acc Out num_heads
    total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_ob
    stride_oh stride_om BLOCK_M BLOCK_D s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.BlockSparseAttn
