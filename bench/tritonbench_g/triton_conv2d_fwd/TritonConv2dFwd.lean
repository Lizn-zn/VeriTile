import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `triton_conv2d_fwd.py`'s `conv2d_forward_kernel`.

This preserves the grouped launch, flattened batch/height/width indexing,
group-local input/output feature offsets, nested kernel-height/kernel-width/
input-feature loops, masked input/weight loads, dot accumulation, and final
masked output store. -/
def conv2d_forward_surface
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_fp16 _tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ComputeKernel := triton {
  batch_height_width_pid = tl.program_id(0)
  out_feat_pid = tl.program_id(1)
  group_pid = tl.program_id(2)
  in_group_dim = $(in_feat_dim) // $(groups)
  out_group_dim = $(out_feat_dim) // $(groups)
  batch_height_width_offset =
    batch_height_width_pid * $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH) +
      tl.arange(0, $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
  batch_height_offset = batch_height_width_offset // $(out_width)
  batch_offset = batch_height_offset // $(out_height)
  output_feat_offset = out_feat_pid * $(BLOCK_SIZE_OUT_FEAT) +
    tl.arange(0, $(BLOCK_SIZE_OUT_FEAT))
  output_height_offset = batch_height_offset % $(out_height)
  output_width_offset = batch_height_width_offset % $(out_width)
    Input +=
      ($(input_batch_stride) * batch_offset +
        $(input_in_feat_stride) * group_pid * in_group_dim)[:, None]
    Weight +=
      ($(weight_out_feat_stride) * output_feat_offset +
        $(weight_out_feat_stride) * group_pid * out_group_dim)[None, :]
  accum = tl.zeros([$(BLOCK_SIZE_BATCH_HEIGHT_WIDTH), $(BLOCK_SIZE_OUT_FEAT)], dtype=tl.float32)
  for h in range($(0), $(kernel_height), $(1)) {
    for w in range($(0), $(kernel_width), $(1)) {
      for c in range($(0), in_group_dim, $(BLOCK_SIZE_IN_FEAT)) {
        input_feat_offset = c + tl.arange(0, $(BLOCK_SIZE_IN_FEAT))
        input_height_offset = h - $((padding_height : Int)) +
          $(stride_height) * output_height_offset
        input_width_offset = w - $((padding_width : Int)) +
          $(stride_width) * output_width_offset
          curr_input_pointer = Input +
            ($(input_in_feat_stride) * input_feat_offset)[None, :] +
            ($(input_height_stride) * input_height_offset)[:, None] +
            ($(input_width_stride) * input_width_offset)[:, None]
          curr_weight_pointer = Weight +
            ($(weight_in_feat_stride) * input_feat_offset)[:, None] +
            $(weight_height_stride) * h + $(weight_width_stride) * w
        input_mask = (batch_offset[:, None] < $(batch_dim)) &
          (input_feat_offset[None, :] < in_group_dim) &
          ($((0 : Int)) <= input_height_offset[:, None]) &
          (input_height_offset[:, None] < $(in_height)) &
          ($((0 : Int)) <= input_width_offset[:, None]) &
          (input_width_offset[:, None] < $(in_width))
        weight_mask = (input_feat_offset[:, None] < in_group_dim) &
          (output_feat_offset[None, :] < out_group_dim)
          input_block = tl.load(curr_input_pointer, mask=input_mask)
          weight_block = tl.load(curr_weight_pointer, mask=weight_mask)
        if _fp16 {
          input_block = (input_block).to(tl.float16)
          weight_block = (weight_block).to(tl.float16)
        }
        accum += tl.dot(input_block, weight_block, allow_tf32=_tf32)
      }
    }
  }
    Output += $(output_batch_stride) * batch_offset[:, None] +
      $(output_out_feat_stride) * (group_pid * out_group_dim + output_feat_offset)[None, :] +
      $(output_height_stride) * output_height_offset[:, None] +
      $(output_width_stride) * output_width_offset[:, None]
    output_mask = (batch_offset[:, None] < $(batch_dim)) &
      (output_feat_offset[None, :] < out_group_dim) &
      (output_height_offset[:, None] < $(out_height)) &
      (output_width_offset[:, None] < $(out_width))
    tl.store(Output, accum, mask=output_mask)
}

/-- The full conv2d forward surface lowers to the algorithm layer, including
grouped launch indexing, nested kernel/input-feature loops, optional fp16 cast,
dot accumulation, and the final masked output store. -/
theorem conv2d_forward_surface_toAlgorithm_supported
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_fp16 _tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ∃ alg,
      (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height
        in_width out_feat_dim out_height out_width input_batch_stride
        input_in_feat_stride input_height_stride input_width_stride
        weight_out_feat_stride weight_in_feat_stride weight_height_stride
        weight_width_stride output_batch_stride output_out_feat_stride
        output_height_stride output_width_stride kernel_height kernel_width
        stride_height stride_width padding_height padding_width groups _fp16 _tf32
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT
        BLOCK_SIZE_OUT_FEAT).toAlgorithm? = Except.ok alg := by
  simp [conv2d_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final output-store slice of `triton_conv2d_fwd.py`'s
`conv2d_forward_kernel`.

The full kernel computes `accum` with the nested convolution/dot loop. This
slice starts after accumulation, keeps the original flattened
batch/height/width indexing and grouped output-feature addressing, and proves
the final masked 2D writeback into `output_pointer`. -/
def conv2d_output_store_slice
    (Output Acc : RegionName)
    (batch_dim out_height out_width OUT_GROUP_DIM
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT : Nat) :
    ComputeKernel := triton {
  batch_height_width_pid = tl.program_id(0)
  out_feat_pid = tl.program_id(1)
  group_pid = tl.program_id(2)
  batch_height_width_offset =
    batch_height_width_pid * $(BLOCK_BHW) + tl.arange(0, $(BLOCK_BHW))
  batch_height_offset = batch_height_width_offset // $(out_width)
  batch_offset = batch_height_offset // $(out_height)
  output_height_offset = batch_height_offset % $(out_height)
  output_width_offset = batch_height_width_offset % $(out_width)
  output_feat_offset = out_feat_pid * $(BLOCK_OUT_FEAT) + tl.arange(0, $(BLOCK_OUT_FEAT))
  accum = tl.load(Acc + batch_height_width_offset[:, None] * $(acc_bhw_stride) +
    output_feat_offset[None, :] * $(acc_feat_stride))
  output_mask = (batch_offset[:, None] < $(batch_dim)) &
    (output_feat_offset[None, :] < $(OUT_GROUP_DIM)) &
    (output_height_offset[:, None] < $(out_height)) &
    (output_width_offset[:, None] < $(out_width))
  tl.store(Output +
      $(output_batch_stride) * batch_offset[:, None] +
      $(output_out_feat_stride) * (group_pid * $(OUT_GROUP_DIM) + output_feat_offset)[None, :] +
      $(output_height_stride) * output_height_offset[:, None] +
      $(output_width_stride) * output_width_offset[:, None],
    accum, mask=output_mask)
}

def bhwIndex (s : BlockState) (BLOCK_BHW : Nat) (i : Fin BLOCK_BHW) : Nat :=
  s.pids 0 * BLOCK_BHW + i.val

def featIndex (s : BlockState) (BLOCK_OUT_FEAT : Nat) (j : Fin BLOCK_OUT_FEAT) : Nat :=
  s.pids 1 * BLOCK_OUT_FEAT + j.val

def batchHeightIndex (s : BlockState) (out_width BLOCK_BHW : Nat) (i : Fin BLOCK_BHW) : Nat :=
  IntegralDType.floorDiv IntegralDType.nat (bhwIndex s BLOCK_BHW i) out_width

def batchIndex
    (s : BlockState) (out_height out_width BLOCK_BHW : Nat) (i : Fin BLOCK_BHW) : Nat :=
  IntegralDType.floorDiv IntegralDType.nat (batchHeightIndex s out_width BLOCK_BHW i) out_height

def heightIndex (s : BlockState) (out_height out_width BLOCK_BHW : Nat)
    (i : Fin BLOCK_BHW) : Nat :=
  IntegralDType.mod IntegralDType.nat (batchHeightIndex s out_width BLOCK_BHW i) out_height

def widthIndex (s : BlockState) (out_width BLOCK_BHW : Nat) (i : Fin BLOCK_BHW) : Nat :=
  IntegralDType.mod IntegralDType.nat (bhwIndex s BLOCK_BHW i) out_width

def active
    (s : BlockState) (batch_dim out_height out_width OUT_GROUP_DIM
      BLOCK_BHW BLOCK_OUT_FEAT : Nat)
    (idx : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT]) : Prop :=
  ((batchIndex s out_height out_width BLOCK_BHW idx.1 < batch_dim ∧
    featIndex s BLOCK_OUT_FEAT idx.2.1 < OUT_GROUP_DIM) ∧
    heightIndex s out_height out_width BLOCK_BHW idx.1 < out_height) ∧
    widthIndex s out_width BLOCK_BHW idx.1 < out_width

instance activeDecidable
    (s : BlockState) (batch_dim out_height out_width OUT_GROUP_DIM
      BLOCK_BHW BLOCK_OUT_FEAT : Nat)
    (idx : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT]) :
    Decidable (active s batch_dim out_height out_width OUT_GROUP_DIM
      BLOCK_BHW BLOCK_OUT_FEAT idx) := by
  unfold active
  infer_instance

def outputOffset
    (s : BlockState)
    (out_height out_width OUT_GROUP_DIM output_batch_stride output_out_feat_stride
      output_height_stride output_width_stride BLOCK_BHW BLOCK_OUT_FEAT : Nat)
    (idx : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT]) : Nat :=
  output_batch_stride * batchIndex s out_height out_width BLOCK_BHW idx.1 +
    output_out_feat_stride *
      (s.pids 2 * OUT_GROUP_DIM + featIndex s BLOCK_OUT_FEAT idx.2.1) +
    output_height_stride * heightIndex s out_height out_width BLOCK_BHW idx.1 +
    output_width_stride * widthIndex s out_width BLOCK_BHW idx.1

def accOffset
    (s : BlockState) (acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT : Nat)
    (idx : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT]) : Nat :=
  bhwIndex s BLOCK_BHW idx.1 * acc_bhw_stride +
    featIndex s BLOCK_OUT_FEAT idx.2.1 * acc_feat_stride

private noncomputable def preStoreState
    (s : BlockState) (Acc : RegionName)
    (batch_dim out_height out_width OUT_GROUP_DIM acc_bhw_stride acc_feat_stride
      BLOCK_BHW BLOCK_OUT_FEAT : Nat) : BlockState :=
  s.setReg "batch_height_width_pid" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "out_feat_pid" TileDType.nat [] (Tile.scalar (s.pids 1))
    |>.setReg "group_pid" TileDType.nat [] (Tile.scalar (s.pids 2))
    |>.setReg "batch_height_width_offset" TileDType.nat [BLOCK_BHW]
      (Tile.vec fun i => s.pids 0 * BLOCK_BHW + i.val)
    |>.setReg "batch_height_offset" TileDType.nat [BLOCK_BHW]
      (Tile.vec fun i => IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + i.val) out_width)
    |>.setReg "batch_offset" TileDType.nat [BLOCK_BHW]
      (Tile.vec fun i =>
        IntegralDType.floorDiv IntegralDType.nat
          (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + i.val) out_width)
          out_height)
    |>.setReg "output_height_offset" TileDType.nat [BLOCK_BHW]
      (Tile.vec fun i =>
        IntegralDType.mod IntegralDType.nat
          (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + i.val) out_width)
          out_height)
    |>.setReg "output_width_offset" TileDType.nat [BLOCK_BHW]
      (Tile.vec fun i => IntegralDType.mod IntegralDType.nat (s.pids 0 * BLOCK_BHW + i.val) out_width)
    |>.setReg "output_feat_offset" TileDType.nat [BLOCK_OUT_FEAT]
      (Tile.vec fun j => s.pids 1 * BLOCK_OUT_FEAT + j.val)
    |>.setReg "accum" TileDType.real [BLOCK_BHW, BLOCK_OUT_FEAT]
      { data := fun idx =>
        some (s.readMem Acc
          ((s.pids 0 * BLOCK_BHW + idx.1.val) * acc_bhw_stride +
            (s.pids 1 * BLOCK_OUT_FEAT + idx.2.1.val) * acc_feat_stride)) }
    |>.setReg "output_mask" TileDType.bool [BLOCK_BHW, BLOCK_OUT_FEAT]
      { data := fun idx =>
        decide
          (((IntegralDType.floorDiv IntegralDType.nat
              (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width)
              out_height < batch_dim ∧
            s.pids 1 * BLOCK_OUT_FEAT + idx.2.1.val < OUT_GROUP_DIM) ∧
            IntegralDType.mod IntegralDType.nat
              (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width)
              out_height < out_height) ∧
            IntegralDType.mod IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width < out_width) }

/-- Algorithm-layer correctness for the final masked conv2d output store. -/
theorem conv2d_output_store_slice_correct
    (Output Acc : RegionName)
    (batch_dim out_height out_width OUT_GROUP_DIM
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (outputOffset s out_height out_width OUT_GROUP_DIM output_batch_stride
        output_out_feat_stride output_height_stride output_width_stride
        BLOCK_BHW BLOCK_OUT_FEAT))
    (hExec : exec (conv2d_output_store_slice Output Acc batch_dim out_height out_width
        OUT_GROUP_DIM output_batch_stride output_out_feat_stride output_height_stride
        output_width_stride acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT)
        s = some s') :
    ∀ idx : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT],
      s'.readMem Output
          (outputOffset s out_height out_width OUT_GROUP_DIM output_batch_stride
            output_out_feat_stride output_height_stride output_width_stride
            BLOCK_BHW BLOCK_OUT_FEAT idx) =
        if active s batch_dim out_height out_width OUT_GROUP_DIM
            BLOCK_BHW BLOCK_OUT_FEAT idx then
          s.readMem Acc (accOffset s acc_bhw_stride acc_feat_stride
            BLOCK_BHW BLOCK_OUT_FEAT idx)
        else
          s.readMem Output
            (outputOffset s out_height out_width OUT_GROUP_DIM output_batch_stride
              output_out_feat_stride output_height_stride output_width_stride
              BLOCK_BHW BLOCK_OUT_FEAT idx) := by
  intro idx
  simp [exec, conv2d_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT] → Nat :=
    fun idx =>
      output_batch_stride *
          IntegralDType.floorDiv IntegralDType.nat
            (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width)
            out_height +
        output_out_feat_stride *
          (s.pids 2 * OUT_GROUP_DIM + (s.pids 1 * BLOCK_OUT_FEAT + idx.2.1.val)) +
        output_height_stride *
          IntegralDType.mod IntegralDType.nat
            (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width)
            out_height +
        output_width_stride *
          IntegralDType.mod IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width
  let valueFn : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT] → ℝ :=
    fun idx =>
      s.readMem Acc
        ((s.pids 0 * BLOCK_BHW + idx.1.val) * acc_bhw_stride +
          (s.pids 1 * BLOCK_OUT_FEAT + idx.2.1.val) * acc_feat_stride)
  let P : TileIndex [BLOCK_BHW, BLOCK_OUT_FEAT] → Prop :=
    fun idx =>
      ((IntegralDType.floorDiv IntegralDType.nat
          (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width)
          out_height < batch_dim ∧
        s.pids 1 * BLOCK_OUT_FEAT + idx.2.1.val < OUT_GROUP_DIM) ∧
        IntegralDType.mod IntegralDType.nat
          (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width)
          out_height < out_height) ∧
        IntegralDType.mod IntegralDType.nat (s.pids 0 * BLOCK_BHW + idx.1.val) out_width < out_width
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outputOffset, batchIndex, batchHeightIndex, heightIndex,
      widthIndex, featIndex, bhwIndex] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := Output)
    (s := preStoreState s Acc batch_dim out_height out_width OUT_GROUP_DIM
      acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P)
    hOffsetInj idx
  by_cases hActive : P idx
  · simpa [offsetFn, valueFn, P, active, outputOffset, accOffset, batchIndex,
      batchHeightIndex, heightIndex, widthIndex, featIndex, bhwIndex, preStoreState,
      Tile.vec, TileShape.dropInsertedIndex, hActive] using hscatter
  · simpa [offsetFn, valueFn, P, active, outputOffset, accOffset, batchIndex,
      batchHeightIndex, heightIndex, widthIndex, featIndex, bhwIndex, preStoreState,
      Tile.vec, TileShape.dropInsertedIndex, hActive] using hscatter

/-- Compute-facing correctness for the final masked conv2d output store. -/
theorem conv2d_output_store_slice_compute_correct
    (Output Acc : RegionName)
    (batch_dim out_height out_width OUT_GROUP_DIM
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (outputOffset s out_height out_width OUT_GROUP_DIM output_batch_stride
        output_out_feat_stride output_height_stride output_width_stride
        BLOCK_BHW BLOCK_OUT_FEAT)) :
    ComputeCorrect.Realizes
      (kernel := conv2d_output_store_slice Output Acc batch_dim out_height out_width
        OUT_GROUP_DIM output_batch_stride output_out_feat_stride output_height_stride
        output_width_stride acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s batch_dim out_height out_width OUT_GROUP_DIM BLOCK_BHW BLOCK_OUT_FEAT)
        (fun idx => (Output,
          outputOffset s out_height out_width OUT_GROUP_DIM output_batch_stride
            output_out_feat_stride output_height_stride output_width_stride
            BLOCK_BHW BLOCK_OUT_FEAT idx)))
      (expected := fun idx =>
        s.readMem Acc (accOffset s acc_bhw_stride acc_feat_stride
          BLOCK_BHW BLOCK_OUT_FEAT idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [conv2d_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := conv2d_output_store_slice_correct Output Acc batch_dim out_height out_width
    OUT_GROUP_DIM output_batch_stride output_out_feat_stride output_height_stride
    output_width_stride acc_bhw_stride acc_feat_stride BLOCK_BHW BLOCK_OUT_FEAT
    s s' hOutInj hExec idx
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.TritonConv2dFwd
