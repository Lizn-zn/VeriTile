import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.QuantizeKvCopy

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Real-valued surface of `quantize_kv_copy.py`'s grouped
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed grouped addressing, `tl.abs`, per-group
scale computation, value writeback, and scale writeback. The Python kernel casts
the scale to `OutScale.dtype.element_ty`; that cast is represented explicitly.
The final quotient cast to int8 is preserved as a surface dtype annotation while
the algorithm carrier records the real-valued quotient. -/
def destindex_copy_quantize_kv_group_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h _stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_g = tl.arange(0, $(BLOCK_GROUP_NUM))
  offs_d = tl.arange(0, $(BLOCK_GROUP_DIM))
  dest_index = tl.load($((DestLoc : Region .nat)) + cur_index)
  group_mask = offs_g < $(group_size)
  value_mask = group_mask[:, None] and (offs_d[None, :] < $(BLOCK_GROUP_DIM))
  src_data = tl.load(K + cur_index * $(stride_k_bs) + cur_head * $(stride_k_h) +
      offs_g[:, None] * $(stride_k_g) + offs_d[None, :] * $(stride_k_d),
    mask=value_mask, other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = (tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty)
  q_src_data = (src_data / data_scale[:, None]).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) + cur_head * $(stride_o_h) +
    offs_g[:, None] * $(stride_o_g) + offs_d[None, :] * $(stride_o_d)
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + cur_head * $(stride_os_h) +
    offs_g
  tl.store(o_ptrs, q_src_data, mask=value_mask)
  tl.store(os_ptrs, data_scale, mask=group_mask)
}

/-- Proof-oriented q-value writeback slice of `quantize_kv_copy.py`'s grouped
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel views the head dimension as `(group, group_dim)`, computes a
per-group scale with `max(abs(src_data), axis=1)`, casts to int8, and stores both
values and scales. This slice starts from a precomputed per-group scale in
`OutScale` and proves the destination-indexed grouped value writeback in
VeriTile's real-tile arithmetic layer. -/
def destindex_copy_quantize_kv_group_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h _stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_g = tl.arange(0, $(BLOCK_GROUP_NUM))
  offs_d = tl.arange(0, $(BLOCK_GROUP_DIM))
  dest_index = tl.load($((DestLoc : Region .nat)) + cur_index)
  mask = (offs_g[:, None] < $(group_size)) & (offs_d[None, :] < $(BLOCK_GROUP_DIM))
  src_data = tl.load(K + cur_index * $(stride_k_bs) + cur_head * $(stride_k_h) +
      offs_g[:, None] * $(stride_k_g) + offs_d[None, :] * $(stride_k_d),
    mask=mask, other=0.0)
  data_scale = tl.load(OutScale + dest_index * $(stride_os_bs) +
      cur_head * $(stride_os_h) + offs_g,
    mask=offs_g < $(group_size), other=1.0)
  q_src_data = src_data / data_scale[:, None]
  tl.store(Out + dest_index * $(stride_o_bs) + cur_head * $(stride_o_h) +
      offs_g[:, None] * $(stride_o_g) + offs_d[None, :] * $(stride_o_d),
    q_src_data, mask=mask)
}

def groupIndex (_s : BlockState) (i : Fin BLOCK_GROUP_NUM) : Nat :=
  i.val

def dimIndex (_s : BlockState) (j : Fin BLOCK_GROUP_DIM) : Nat :=
  j.val

def destIndex (s : BlockState) (DestLoc : RegionName) : Nat :=
  s.readMemValue .nat DestLoc (s.pids 0)

def active
    (s : BlockState) (group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Prop :=
  groupIndex s idx.1 < group_size

instance activeDecidable
    (s : BlockState) (group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) :
    Decidable (active s group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM idx) := by
  unfold active
  infer_instance

def kOffset
    (s : BlockState) (stride_k_bs stride_k_h stride_k_g stride_k_d : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Nat :=
  s.pids 0 * stride_k_bs + s.pids 1 * stride_k_h +
    groupIndex s idx.1 * stride_k_g + dimIndex s idx.2.1 * stride_k_d

def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_g stride_o_d : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Nat :=
  destIndex s DestLoc * stride_o_bs + s.pids 1 * stride_o_h +
    groupIndex s idx.1 * stride_o_g + dimIndex s idx.2.1 * stride_o_d

def scaleOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h _stride_os_g : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Nat :=
  destIndex s DestLoc * stride_os_bs + s.pids 1 * stride_os_h +
    groupIndex s idx.1

noncomputable def quantizeKvCopyGroupValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_os_bs stride_os_h stride_os_g : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_g stride_k_d idx) /
    s.readMem OutScale
      (scaleOffset s DestLoc stride_os_bs stride_os_h stride_os_g idx)

/-- Algorithm-layer correctness for the grouped destination-indexed quantized KV
value-store slice. -/
theorem destindex_copy_quantize_kv_group_value_store_slice_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM],
      let outAddr := outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx
      (exec (destindex_copy_quantize_kv_group_value_store_slice K DestLoc Out
            OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
            stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h
            stride_os_g group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM) s).map
          (·.readMem Out outAddr)
        = some (if active s group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM idx then
            quantizeKvCopyGroupValueSpec s K DestLoc OutScale stride_k_bs
              stride_k_h stride_k_g stride_k_d stride_os_bs stride_os_h
              stride_os_g idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, destindex_copy_quantize_kv_group_value_store_slice, stepStmts,
        stepStmt, evalOp, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, BlockState.readMemValue,
        groupIndex, dimIndex, destIndex, kOffset, outOffset, scaleOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] → Nat :=
    fun idx =>
      (match s.readMemTyped TileDType.nat DestLoc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_o_bs +
        s.pids 1 * stride_o_h + idx.1.val * stride_o_g +
        idx.2.1.val * stride_o_d
  let valueFn : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun x1 x2 => x1 / x2)
          (if idx.1.val < group_size then
            some (s.readMem K
              (s.pids 0 * stride_k_bs + s.pids 1 * stride_k_h +
                idx.1.val * stride_k_g + idx.2.1.val * stride_k_d))
          else some (0.0 : ℝ))
          (if idx.1.val < group_size then
            some (s.readMem OutScale
              ((match s.readMemTyped TileDType.nat DestLoc (s.pids 0) with
                | some value => value
                | none => BlockState.defaultCarrier TileDType.nat) * stride_os_bs +
                s.pids 1 * stride_os_h + idx.1.val))
          else some (1.0 : ℝ)))
  let P : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] → Prop :=
    fun idx => idx.1.val < group_size
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, destIndex, groupIndex, dimIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM])).readMem Out
        (offsetFn idx) =
    if active s group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM idx then
      quantizeKvCopyGroupValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
        stride_k_g stride_k_d stride_os_bs stride_os_h stride_os_g idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hGroup : idx.1.val < group_size
  · simp [offsetFn, valueFn, P, active, quantizeKvCopyGroupValueSpec, kOffset,
      outOffset, scaleOffset, destIndex, groupIndex, dimIndex,
      BlockState.readMemValue, hGroup]
    rfl
  · simp [offsetFn, valueFn, P, active, outOffset, scaleOffset, destIndex,
      groupIndex, dimIndex, BlockState.readMemValue, hGroup]

/-- Compute-facing correctness for the grouped destination-indexed quantized KV
value-store slice. -/
theorem destindex_copy_quantize_kv_group_value_store_slice_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_value_store_slice K DestLoc Out
        OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
        stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g
        group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM)
        (fun idx => (Out,
          outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)))
      (expected := fun idx =>
        quantizeKvCopyGroupValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
          stride_k_g stride_k_d stride_os_bs stride_os_h stride_os_g idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_group_value_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := destindex_copy_quantize_kv_group_value_store_slice_correct K
    DestLoc Out OutScale stride_k_bs stride_k_h stride_k_g stride_k_d
    stride_o_bs stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h
    stride_os_g group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.QuantizeKvCopy
