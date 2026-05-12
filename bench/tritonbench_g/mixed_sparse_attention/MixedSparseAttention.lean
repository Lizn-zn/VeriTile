import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription/proof-oriented final output-store slice of
`mixed_sparse_attention.py`'s `_triton_mixed_sparse_attn_fwd_kernel`.

The full kernel combines block-sparse and column-sparse attention updates. This
slice starts from a precomputed normalized `Acc` tile and proves the final
`seqlens`-masked writeback into `Out`. The kernel-level early return for
`start_m * BLOCK_M >= seqlen` is represented at this surface by the same
all-false row mask; the sparse block/column softmax loops remain separate
modeling work, including their `tl.float32` accumulators. -/
def mixed_sparse_attention_output_store_slice
    (Acc Seqlens Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(axis=0)
  off_hz = tl.program_id(axis=1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  seqlen = tl.load(Seqlens + off_z, dtype=tl.uint64)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens

instance activeDecidable (s : BlockState) (H : Nat) (Seqlens : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s H Seqlens BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok

noncomputable def accStoreValue
    (s : BlockState) (Acc Seqlens : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H Seqlens BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))

theorem mixed_sparse_attention_output_store_slice_correct
    (Acc Seqlens Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_om stride_ok
        BLOCK_M idx
      (exec (mixed_sparse_attention_output_store_slice Acc Seqlens Out H
            stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz
            stride_qh stride_om stride_ok BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s H Seqlens BLOCK_M idx then
            accStoreValue s Acc Seqlens H stride_acc_z stride_acc_h
              stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, mixed_sparse_attention_output_store_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, BlockState.readMemValue, offZ,
        offH, seqLen, mIndex, dIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.pids 1 / H) * stride_qz + (s.pids 1 % H) * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om +
        idx.2.1.val * stride_ok
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val <
            s.readMemValue .nat Seqlens (s.pids 1 / H) then
          some (s.readMem Acc
            ((s.pids 1 / H) * stride_acc_z + (s.pids 1 % H) * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat Seqlens (s.pids 1 / H)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc Seqlens H stride_acc_z stride_acc_h stride_acc_m
        stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat Seqlens (s.pids 1 / H)
  · rfl
  · rfl

theorem mixed_sparse_attention_output_store_slice_compute_correct
    (Acc Seqlens Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out H
        stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz
        stride_qh stride_om stride_ok BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H Seqlens BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc Seqlens H stride_acc_z stride_acc_h stride_acc_m
          stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mixed_sparse_attention_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := mixed_sparse_attention_output_store_slice_correct Acc Seqlens Out H
    stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz stride_qh
    stride_om stride_ok BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.MixedSparseAttention
