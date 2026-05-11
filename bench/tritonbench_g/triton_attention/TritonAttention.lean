import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription/proof-oriented forward output-store slice of
`triton_attention.py`'s `_fwd_kernel`.

The Python kernel writes `acc` through a block pointer with
`boundary_check=(0, 1)`. This slice spells the same write as explicit pointer
arithmetic and an explicit two-axis boundary mask. -/
def triton_attention_forward_output_store_slice
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(axis=0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] + $(hzRowOffset) < $(D0)) &
    (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + offs_m[:, None] * $(BLOCK_DMODEL) + offs_d[None, :],
    mask=mask, other=0.0)
  tl.store(Out + (offs_m[:, None] + $(hzRowOffset)) * $(stride_om) +
      offs_d[None, :] * $(stride_on), acc, mask=mask)
}

def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active (s : BlockState) (hzRowOffset D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  rowIndex s BLOCK_M idx.1 + hzRowOffset < D0

instance activeDecidable (s : BlockState) (hzRowOffset D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s hzRowOffset D0 BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset (s : BlockState) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  rowIndex s BLOCK_M idx.1 * BLOCK_DMODEL + dIndex idx

def outOffset (s : BlockState) (hzRowOffset stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (rowIndex s BLOCK_M idx.1 + hzRowOffset) * stride_om + dIndex idx * stride_on

noncomputable def storeValue (s : BlockState) (Acc : RegionName)
    (hzRowOffset D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s hzRowOffset D0 BLOCK_M idx then
      some (s.readMem Acc (accOffset s BLOCK_M BLOCK_DMODEL idx))
    else some (0.0 : ℝ))

theorem triton_attention_forward_output_store_slice_correct
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s hzRowOffset stride_om stride_on BLOCK_M idx
      (exec (triton_attention_forward_output_store_slice Acc Out hzRowOffset D0
            stride_om stride_on BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s hzRowOffset D0 BLOCK_M idx then
            storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, triton_attention_forward_output_store_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, dIndex, active, accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset) * stride_om +
        idx.2.1.val * stride_on
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0 then
          some (s.readMem Acc
            ((s.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, rowIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0
  · rfl
  · rfl

theorem triton_attention_forward_output_store_slice_compute_correct
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out hzRowOffset D0
        stride_om stride_on BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s hzRowOffset D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := triton_attention_forward_output_store_slice_correct Acc Out
    hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.TritonAttention
