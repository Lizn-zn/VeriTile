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
surface dtype annotation and lowers through the DSL's fixed-width cast
placeholder. -/
def destindex_copy_quantize_kv_transform_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)),
    other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data,
    mask=(offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(head_dim)))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}

/-- The full real-valued quantize-kv-transform surface lowers through algorithm
erasure, including the final quotient `to(tl.int8)` cast placeholder. -/
theorem destindex_copy_quantize_kv_transform_real_surface_toAlgorithm_supported
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ∃ alg,
      (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL
        BLOCK_HEAD).toAlgorithm? = Except.ok alg := by
  simp [destindex_copy_quantize_kv_transform_real_surface,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
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
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
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

/-- Active-lane variant of the value-store correctness theorem. This is needed
for Python shapes with padded block dimensions: inactive padded lanes may alias
active lanes under the physical output strides, but the masked store only writes
active lanes. -/
theorem destindex_copy_quantize_kv_transform_value_store_slice_active_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num head_dim BLOCK_HEAD BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hNoCollision :
      ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
        active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL idx →
        ∀ k : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
          active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL k →
          outOffset s DestLoc stride_o_bs stride_o_h stride_o_d k =
            outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx →
          k = idx) :
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
  have hRead :
      (exec (destindex_copy_quantize_kv_transform_value_store_slice K DestLoc Out
            OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h
            stride_o_d stride_os_bs stride_os_h head_num head_dim BLOCK_HEAD
            BLOCK_DMODEL) s).map
          (·.readMem Out (outOffset s DestLoc stride_o_bs stride_o_h
            stride_o_d idx))
        = some (quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs
            stride_k_h stride_k_d stride_os_bs stride_os_h idx) := by
    simp [exec, destindex_copy_quantize_kv_transform_value_store_slice,
          stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
          Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          BlockState.readMemValue, headIndex, dimIndex, destIndex, kOffset,
          outOffset, scaleOffset, TileShape.dropInsertedIndex]
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
    change (List.foldl
        (fun (acc : BlockState) i =>
          if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
        _ (TileShape.allIndices [BLOCK_HEAD, BLOCK_DMODEL])).readMem Out
          (offsetFn idx) =
      quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
        stride_k_d stride_os_bs stride_os_h idx
    rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ P idx]
    · have hP : P idx := by
        simpa [P, active, headIndex, dimIndex] using hActive
      have hHead : idx.1.val < head_num := hP.1
      simp [offsetFn, valueFn, P, quantizeKvTransformValueSpec, kOffset,
        scaleOffset, destIndex, headIndex, dimIndex, BlockState.readMemValue,
        hP, hHead]
      rfl
    · simpa [P, active, headIndex, dimIndex] using hActive
    · intro k hPk heq
      exact hNoCollision idx hActive k
        (by simpa [P, active, headIndex, dimIndex] using hPk)
        (by simpa [offsetFn, outOffset, destIndex, headIndex, dimIndex,
          BlockState.readMemValue] using heq)
  rw [hExec] at hRead
  exact Option.some.inj hRead

/-- Named value writeback coverage for Python test case 1
(`H = 12`, `D = 96`, so `BLOCK_HEAD = 16`, `BLOCK_DMODEL = 128`). -/
theorem destindex_copy_quantize_kv_transform_test_h12_d96_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [16, 128] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs
        stride_o_h stride_o_d stride_os_bs stride_os_h 12 96 16 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 12 96 16 128)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h
          stride_o_d idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs
          stride_k_h stride_k_d stride_os_bs stride_os_h idx) := by
  exact destindex_copy_quantize_kv_transform_value_store_slice_compute_correct
    K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs
    stride_o_h stride_o_d stride_os_bs stride_os_h 12 96 16 128 s hOutInj

/-- Named value writeback coverage for Python test cases 2 and 3
(`H = 8`, `D = 64`). -/
theorem destindex_copy_quantize_kv_transform_test_h8_d64_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [8, 64] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs
        stride_o_h stride_o_d stride_os_bs stride_os_h 8 64 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 64 8 64)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h
          stride_o_d idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs
          stride_k_h stride_k_d stride_os_bs stride_os_h idx) := by
  exact destindex_copy_quantize_kv_transform_value_store_slice_compute_correct
    K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs
    stride_o_h stride_o_d stride_os_bs stride_os_h 8 64 8 64 s hOutInj

/-- Named value writeback coverage for Python test case 4 (`H = 1`, `D = 1`). -/
theorem destindex_copy_quantize_kv_transform_test_h1_d1_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [1, 1] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs
        stride_o_h stride_o_d stride_os_bs stride_os_h 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 1 1 1)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h
          stride_o_d idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale stride_k_bs
          stride_k_h stride_k_d stride_os_bs stride_os_h idx) := by
  exact destindex_copy_quantize_kv_transform_value_store_slice_compute_correct
    K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs
    stride_o_h stride_o_d stride_os_bs stride_os_h 1 1 1 1 s hOutInj

/-- Proof-oriented scale-writeback slice of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`. Companion to the value-store slice:
covers the 1D `Out_scale` writeback from a precomputed `Scale` tile under the
destination-indexed addressing. -/
def destindex_copy_quantize_kv_transform_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_h < $(head_num)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) + offs_h,
    mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      $(stride_os_h) * offs_h,
    data_scale, mask=mask)
}

def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num

instance scaleActiveDecidable (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) :
    Decidable (scaleActive head_num BLOCK_HEAD i) := by
  unfold scaleActive
  infer_instance

def scaleSourceOffset (s : BlockState) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : Nat :=
  s.pids 0 * stride_s_bs + i.val

def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * i.val

noncomputable def quantizeKvTransformScaleSpec
    (s : BlockState) (Scale : RegionName) (stride_s_bs : Nat)
    (i : Fin BLOCK_HEAD) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs i)

theorem destindex_copy_quantize_kv_transform_scale_store_slice_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HEAD =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ∀ i : Fin BLOCK_HEAD,
      let outAddr := scaleOutOffset s DestLoc stride_os_bs stride_os_h i
      (exec (destindex_copy_quantize_kv_transform_scale_store_slice Scale DestLoc
            OutScale stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD)
          s).map (·.readMem OutScale outAddr)
        = some (if scaleActive head_num BLOCK_HEAD i then
            quantizeKvTransformScaleSpec s Scale stride_s_bs i
          else s.readMem OutScale outAddr) := by
  intro i
  simp [exec, destindex_copy_quantize_kv_transform_scale_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, destIndex, scaleSourceOffset, scaleOutOffset]
  let offsetFn : TileIndex [BLOCK_HEAD] → Nat :=
    fun j =>
      (match s.readMemTyped TileDType.nat DestLoc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_os_bs +
        stride_os_h * j.1.val
  let valueFn : TileIndex [BLOCK_HEAD] → ℝ :=
    fun j =>
      WithBot.unbotD 0
        (if j.1.val < head_num then
          some (s.readMem Scale (s.pids 0 * stride_s_bs + j.1.val))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_HEAD] → Prop :=
    fun j => j.1.val < head_num
  have hOffsetInj : Function.Injective offsetFn := by
    intro a b hab
    have hInner :
        scaleOutOffset s DestLoc stride_os_bs stride_os_h a.1 =
          scaleOutOffset s DestLoc stride_os_bs stride_os_h b.1 := by
      simpa [offsetFn, scaleOutOffset, destIndex, BlockState.readMemValue] using hab
    have : a.1 = b.1 := hOutInj hInner
    cases a; cases b
    simp only at this
    cases this; rfl
  change (List.foldl
      (fun (acc : BlockState) j =>
        if P j then acc.writeMem OutScale (offsetFn j) (valueFn j) else acc)
      _ (TileShape.allIndices [BLOCK_HEAD])).readMem OutScale
        (offsetFn (i, PUnit.unit)) =
    if scaleActive head_num BLOCK_HEAD i then
      quantizeKvTransformScaleSpec s Scale stride_s_bs i
    else s.readMem OutScale (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < head_num
  · simp [offsetFn, valueFn, P, scaleActive, quantizeKvTransformScaleSpec,
      scaleSourceOffset, scaleOutOffset, destIndex,
      BlockState.readMemValue, hi]
  · simp [offsetFn, valueFn, P, scaleActive, scaleOutOffset, destIndex,
      BlockState.readMemValue, hi]

theorem destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HEAD =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive head_num BLOCK_HEAD)
        (fun i => (OutScale,
          scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale stride_s_bs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_transform_scale_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := destindex_copy_quantize_kv_transform_scale_store_slice_correct Scale
    DestLoc OutScale stride_s_bs stride_os_bs stride_os_h head_num BLOCK_HEAD s
    hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Named scale writeback coverage for Python test case 1
(`H = 12`, `BLOCK_HEAD = 16`). -/
theorem destindex_copy_quantize_kv_transform_test_h12_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 16 =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale stride_s_bs stride_os_bs stride_os_h 12 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 12 16)
        (fun i => (OutScale,
          scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale stride_s_bs i) := by
  exact destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct
    Scale DestLoc OutScale stride_s_bs stride_os_bs stride_os_h 12 16
    s hOutInj

/-- Named scale writeback coverage for Python test cases 2 and 3
(`H = 8`, `BLOCK_HEAD = 8`). -/
theorem destindex_copy_quantize_kv_transform_test_h8_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 8 =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale stride_s_bs stride_os_bs stride_os_h 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale,
          scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale stride_s_bs i) := by
  exact destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct
    Scale DestLoc OutScale stride_s_bs stride_os_bs stride_os_h 8 8
    s hOutInj

/-- Named scale writeback coverage for Python test case 4
(`H = 1`, `BLOCK_HEAD = 1`). -/
theorem destindex_copy_quantize_kv_transform_test_h1_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 1 =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale stride_s_bs stride_os_bs stride_os_h 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 1 1)
        (fun i => (OutScale,
          scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale stride_s_bs i) := by
  exact destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct
    Scale DestLoc OutScale stride_s_bs stride_os_bs stride_os_h 1 1
    s hOutInj

/-! ## Python test-shape wrappers

Python cases 2 and 3 use `B * N_CTX = 8192`, `H = 8`, `D = 64`; `K/Out`
strides are `(512, 64, 1)` and `Out_scale` strides are `(8, 1, 1)`. Case 4
uses the minimal `H = 1`, `D = 1` layout with all relevant strides equal to
`1`.

Python case 1 has `H = 12`, `D = 96`, `BLOCK_HEAD = 16`,
`BLOCK_DMODEL = 128`. Its inactive padded columns make the full tile address
map non-injective with strides `(1152, 96, 1)`, so it needs an active-lane
no-collision store theorem rather than the stronger full-tile injectivity
premise used by the generic theorem above. -/

theorem destindex_copy_quantize_kv_transform_python_h12_d96_active_no_collision
    (s : BlockState) (DestLoc : RegionName) :
    ∀ idx : TileIndex [16, 128],
      active s 12 96 16 128 idx →
      ∀ k : TileIndex [16, 128],
        active s 12 96 16 128 k →
        outOffset s DestLoc 1152 96 1 k =
          outOffset s DestLoc 1152 96 1 idx →
        k = idx := by
  rintro ⟨⟨ha, hha⟩, ⟨da, hda⟩, _⟩ hA
    ⟨⟨hb, hhb⟩, ⟨db, hdb⟩, _⟩ hB hEq
  simp [active, outOffset, destIndex, headIndex, dimIndex] at hA hB hEq
  have hh : hb = ha := by omega
  have hd : db = da := by omega
  subst hb
  subst db
  rfl

theorem destindex_copy_quantize_kv_transform_python_h8_d64_value_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [8, 64] => outOffset s DestLoc 512 64 1 idx) := by
  rintro ⟨⟨ha, hha⟩, ⟨da, hda⟩, _⟩ ⟨⟨hb, hhb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, destIndex, headIndex, dimIndex] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem destindex_copy_quantize_kv_transform_python_h1_d1_value_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [1, 1] => outOffset s DestLoc 1 1 1 idx) := by
  rintro ⟨⟨ha, hha⟩, ⟨da, hda⟩, _⟩ ⟨⟨hb, hhb⟩, ⟨db, hdb⟩, _⟩ _h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem destindex_copy_quantize_kv_transform_python_h8_scale_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun i : Fin 8 => scaleOutOffset s DestLoc 8 1 i) := by
  intro a b h
  simp [scaleOutOffset, destIndex] at h
  exact Fin.ext (by omega)

theorem destindex_copy_quantize_kv_transform_python_h1_scale_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun i : Fin 1 => scaleOutOffset s DestLoc 1 1 i) := by
  intro a b _h
  exact Fin.ext (by omega)

theorem destindex_copy_quantize_kv_transform_python_h12_scale_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun i : Fin 16 => scaleOutOffset s DestLoc 12 1 i) := by
  intro a b h
  simp [scaleOutOffset, destIndex] at h
  exact Fin.ext (by omega)

theorem destindex_copy_quantize_kv_transform_python_h12_d96_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1152 96 1 1152 96 1 12 1 12 96 16 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 12 96 16 128)
        (fun idx => (Out, outOffset s DestLoc 1152 96 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1152 96 1 12 1 idx) := by
  exact destindex_copy_quantize_kv_transform_value_store_slice_active_compute_correct
    K DestLoc Out OutScale 1152 96 1 1152 96 1 12 1 12 96 16 128 s
    (destindex_copy_quantize_kv_transform_python_h12_d96_active_no_collision
      s DestLoc)

theorem destindex_copy_quantize_kv_transform_python_h8_d64_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 512 64 1 512 64 1 8 1 8 64 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 64 8 64)
        (fun idx => (Out, outOffset s DestLoc 512 64 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 512 64 1 8 1 idx) := by
  exact destindex_copy_quantize_kv_transform_value_store_slice_compute_correct
    K DestLoc Out OutScale 512 64 1 512 64 1 8 1 8 64 8 64 s
    (destindex_copy_quantize_kv_transform_python_h8_d64_value_offset_injective
      s DestLoc)

theorem destindex_copy_quantize_kv_transform_python_h1_d1_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1 1 1 1 1 1 1 1 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 1 1 1)
        (fun idx => (Out, outOffset s DestLoc 1 1 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1 1 1 1 1 idx) := by
  exact destindex_copy_quantize_kv_transform_value_store_slice_compute_correct
    K DestLoc Out OutScale 1 1 1 1 1 1 1 1 1 1 1 1 s
    (destindex_copy_quantize_kv_transform_python_h1_d1_value_offset_injective
      s DestLoc)

theorem destindex_copy_quantize_kv_transform_python_h12_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 12 12 1 12 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 12 16)
        (fun i => (OutScale, scaleOutOffset s DestLoc 12 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 12 i) := by
  exact destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct
    Scale DestLoc OutScale 12 12 1 12 16 s
    (destindex_copy_quantize_kv_transform_python_h12_scale_offset_injective
      s DestLoc)

theorem destindex_copy_quantize_kv_transform_python_h8_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 8 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset s DestLoc 8 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 8 i) := by
  exact destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct
    Scale DestLoc OutScale 8 8 1 8 8 s
    (destindex_copy_quantize_kv_transform_python_h8_scale_offset_injective
      s DestLoc)

theorem destindex_copy_quantize_kv_transform_python_h1_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 1 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 1 1)
        (fun i => (OutScale, scaleOutOffset s DestLoc 1 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 1 i) := by
  exact destindex_copy_quantize_kv_transform_scale_store_slice_compute_correct
    Scale DestLoc OutScale 1 1 1 1 1 s
    (destindex_copy_quantize_kv_transform_python_h1_scale_offset_injective
      s DestLoc)

/-- Python `H = 12, D = 96` case: both the active-lane destination value
writeback and the per-head scale writeback are compute-correct. -/
theorem destindex_copy_quantize_kv_transform_python_h12_d96_all_outputs_compute_correct
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1152 96 1 1152 96 1 12 1 12 96 16 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 12 96 16 128)
        (fun idx => (Out, outOffset s DestLoc 1152 96 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1152 96 1 12 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 12 12 1 12 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 12 16)
        (fun i => (OutScale, scaleOutOffset s DestLoc 12 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 12 i)) := by
  constructor
  · exact destindex_copy_quantize_kv_transform_python_h12_d96_value_store_compute_correct
      K DestLoc Out OutScale s
  · exact destindex_copy_quantize_kv_transform_python_h12_scale_store_compute_correct
      Scale DestLoc OutScale s

/-- Python `H = 8, D = 64` case: both destination value writeback and per-head
scale writeback are compute-correct. -/
theorem destindex_copy_quantize_kv_transform_python_h8_d64_all_outputs_compute_correct
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 512 64 1 512 64 1 8 1 8 64 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 64 8 64)
        (fun idx => (Out, outOffset s DestLoc 512 64 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 512 64 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 8 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset s DestLoc 8 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 8 i)) := by
  constructor
  · exact destindex_copy_quantize_kv_transform_python_h8_d64_value_store_compute_correct
      K DestLoc Out OutScale s
  · exact destindex_copy_quantize_kv_transform_python_h8_scale_store_compute_correct
      Scale DestLoc OutScale s

/-- Python `H = 1, D = 1` case: both destination value writeback and the scalar
scale writeback are compute-correct. -/
theorem destindex_copy_quantize_kv_transform_python_h1_d1_all_outputs_compute_correct
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_value_store_slice K
        DestLoc Out OutScale 1 1 1 1 1 1 1 1 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 1 1 1 1)
        (fun idx => (Out, outOffset s DestLoc 1 1 1 idx)))
      (expected := fun idx =>
        quantizeKvTransformValueSpec s K DestLoc OutScale 1 1 1 1 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_transform_scale_store_slice Scale
        DestLoc OutScale 1 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 1 1)
        (fun i => (OutScale, scaleOutOffset s DestLoc 1 1 i)))
      (expected := fun i =>
        quantizeKvTransformScaleSpec s Scale 1 i)) := by
  constructor
  · exact destindex_copy_quantize_kv_transform_python_h1_d1_value_store_compute_correct
      K DestLoc Out OutScale s
  · exact destindex_copy_quantize_kv_transform_python_h1_scale_store_compute_correct
      Scale DestLoc OutScale s

end VeriTile.Bench.TritonBenchG.QuantizeKvTransform
