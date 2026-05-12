import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Real-valued surface of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, the `head_num/head_dim` mask,
`tl.abs`, per-head scale computation, value writeback, and scale writeback. The
Python kernel casts the scale to `OutScale.dtype.element_ty`; that cast is
represented explicitly. The final quotient cast to int8 is preserved as a
surface dtype annotation while the algorithm carrier records the real-valued
quotient. -/
def destindex_copy_quantize_kv_transform_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(axis=0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load($((DestLoc : Region .nat)) + cur_index)
  mask = (offs_h[:, None] < $(head_num)) and (offs_d[None, :] < $(head_dim))
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=mask, other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data, mask=mask)
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}

/-- Proof-oriented q-value writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes a per-head scale from `max(abs(src_data))`, casts the
quotient to int8, and stores both values and scales. This slice starts from a
precomputed per-head scale in `OutScale`, preserves the source
`head_num/head_dim` mask, and proves the destination-indexed value writeback in
VeriTile's real-tile arithmetic layer. -/
def destindex_copy_quantize_kv_transform_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(axis=0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load($((DestLoc : Region .nat)) + cur_index)
  mask = (offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim))
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=mask, other=0.0)
  data_scale = tl.load(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    mask=offs_h < $(head_num), other=1.0)
  q_src_data = src_data / data_scale[:, None]
  tl.store(Out + dest_index * $(stride_o_bs) +
      $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :],
    q_src_data, mask=mask)
}

def headIndex (_s : BlockState) (i : Fin BLOCK_HEAD) : Nat :=
  i.val

def dimIndex (_s : BlockState) (j : Fin BLOCK_DMODEL) : Nat :=
  j.val

def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)

def active
    (s : BlockState) (head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num ∧ dimIndex s idx.2.1 < head_dim

instance activeDecidable
    (s : BlockState) (head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    Decidable (active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL idx) := by
  unfold active
  infer_instance

def kOffset
    (s : BlockState) (stride_k_bs stride_k_h stride_k_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_k_bs + headIndex s idx.1 * stride_k_h +
    stride_k_d * dimIndex s idx.2.1

def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_o_bs + stride_o_h * headIndex s idx.1 +
    stride_o_d * dimIndex s idx.2.1

def scaleOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * headIndex s idx.1

noncomputable def quantizeKvTransformValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_d idx) /
    s.readMem OutScale (scaleOffset s DestLoc stride_os_bs stride_os_h idx)

/-- Algorithm-layer correctness for the destination-indexed quantized KV
transform value-store slice. -/
theorem destindex_copy_quantize_kv_transform_value_store_slice_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      let outAddr := outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx
      (exec (destindex_copy_quantize_kv_transform_value_store_slice K DestLoc Out
            OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h
            stride_o_d stride_os_bs stride_os_h head_num head_dim BLOCK_HEAD
            BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (if active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs
              stride_k_h stride_k_d stride_os_bs stride_os_h idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, destindex_copy_quantize_kv_transform_value_store_slice, stepStmts,
        stepStmt, evalOp, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, BlockState.readMemValue,
        headIndex, dimIndex, destIndex, kOffset, outOffset, scaleOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (match s.readMemTyped TileDType.nat DestLoc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_o_bs +
        stride_o_h * idx.1.val + stride_o_d * idx.2.1.val
  let valueFn : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun x1 x2 => x1 / x2)
          (if idx.1.val < head_num ∧ idx.2.1.val < head_dim then
            some (s.readMem K
              (s.pids 0 * stride_k_bs + idx.1.val * stride_k_h +
                stride_k_d * idx.2.1.val))
          else some (0.0 : ℝ))
          (if idx.1.val < head_num then
            some (s.readMem OutScale
              ((match s.readMemTyped TileDType.nat DestLoc (s.pids 0) with
                | some value => value
                | none => BlockState.defaultCarrier TileDType.nat) * stride_os_bs +
                stride_os_h * idx.1.val))
          else some (1.0 : ℝ)))
  let P : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] → Prop :=
    fun idx => idx.1.val < head_num ∧ idx.2.1.val < head_dim
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, destIndex, headIndex, dimIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_HEAD, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL idx then
      quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
        stride_k_d stride_os_bs stride_os_h idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : idx.1.val < head_num ∧ idx.2.1.val < head_dim
  · have hHead : idx.1.val < head_num := hActive.1
    simp [offsetFn, valueFn, P, active, quantizeKvTransformValueSpec, kOffset,
      outOffset, scaleOffset, destIndex, headIndex, dimIndex,
      BlockState.readMemValue, hActive, hHead]
    rfl
  · simp [offsetFn, valueFn, P, active, outOffset, scaleOffset, destIndex,
      headIndex, dimIndex, BlockState.readMemValue, hActive]

/-- Compute-facing correctness for the destination-indexed quantized KV
transform value-store slice. -/
theorem destindex_copy_quantize_kv_transform_value_store_slice_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K DestLoc Out
        OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h
        stride_o_d stride_os_bs stride_os_h head_num head_dim BLOCK_HEAD
        BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
          stride_k_d stride_os_bs stride_os_h idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_transform_value_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := destindex_copy_quantize_kv_transform_value_store_slice_correct K
    DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs
    stride_o_h stride_o_d stride_os_bs stride_os_h head_num head_dim BLOCK_HEAD
    BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.QuantizeKvTransform
