import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Real-valued surface of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, `tl.abs`, the per-head
`tl.max(..., axis=1)` scale computation, value writeback, and scale writeback.
The Python kernel casts the scale to fp16 before broadcasting it and casts the
quotient to int8; both casts are preserved as surface dtype annotations and
lower through the DSL's fixed-width cast surfaces. -/
def destindex_copy_quantize_kv_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h _stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) +
      offs_h[:, None] * $(stride_k_h) + $(stride_k_d) * offs_d[None, :],
    mask=offs_h[:, None] < $(head_num), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(tl.float16))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + $(stride_os_h) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < $(head_num))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(head_num))
}

/-- The quantize-copy-kv surface lowers through algorithm erasure, including the
final quotient `to(tl.int8)` cast surface. -/
theorem destindex_copy_quantize_kv_real_surface_toAlgorithm_supported
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ∃ alg,
      (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL
        BLOCK_HEAD).toAlgorithm? = Except.ok alg := by
  simp [destindex_copy_quantize_kv_real_surface,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented q-value writeback slice of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes a per-head `data_scale` with `max(abs(src_data))`,
casts the quotient to int8, and stores both the quantized values and the scale.
VeriTile's current arithmetic layer models real tiles, so this slice starts from
a precomputed per-head scale in `OutScale` and proves the masked destination
indexed value writeback before the fixed-width int8 cast surface. -/
def destindex_copy_quantize_kv_value_store_slice
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num BLOCK_HEAD BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(DestLoc + cur_index)
  mask = (offs_h[:, None] < $(head_num)) & (offs_d[None, :] < $(BLOCK_DMODEL))
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
    (s : BlockState) (head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex s idx.1 < head_num

instance activeDecidable
    (s : BlockState) (head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    Decidable (active s head_num BLOCK_HEAD BLOCK_DMODEL idx) := by
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

noncomputable def quantizeCopyKvValueSpec
    (s : BlockState) (K DestLoc OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_os_bs stride_os_h : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : ℝ :=
  s.readMem K (kOffset s stride_k_bs stride_k_h stride_k_d idx) /
    s.readMem OutScale (scaleOffset s DestLoc stride_os_bs stride_os_h idx)

/-- Algorithm-layer correctness for the destination-indexed quantized KV value
store slice. -/
theorem destindex_copy_quantize_kv_value_store_slice_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      let outAddr := outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx
      (exec (destindex_copy_quantize_kv_value_store_slice K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h head_num BLOCK_HEAD BLOCK_DMODEL)
          s).map (·.readMem Out outAddr)
        = some (if active s head_num BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeCopyKvValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
              stride_k_d stride_os_bs stride_os_h idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, destindex_copy_quantize_kv_value_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, BlockState.readMemValue, headIndex, dimIndex,
        destIndex, kOffset, outOffset, scaleOffset, TileShape.dropInsertedIndex]
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
          (if idx.1.val < head_num then
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
    fun idx => idx.1.val < head_num
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, destIndex, headIndex, dimIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_HEAD, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if active s head_num BLOCK_HEAD BLOCK_DMODEL idx then
      quantizeCopyKvValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
        stride_k_d stride_os_bs stride_os_h idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hHead : idx.1.val < head_num
  · simp [offsetFn, valueFn, P, active, quantizeCopyKvValueSpec, kOffset,
      outOffset, scaleOffset, destIndex, headIndex, dimIndex,
      BlockState.readMemValue, hHead]
    rfl
  · simp [offsetFn, valueFn, P, active, outOffset, scaleOffset, destIndex,
      headIndex, dimIndex, BlockState.readMemValue, hHead]

/-- Compute-facing correctness for the destination-indexed quantized KV value
store slice. -/
theorem destindex_copy_quantize_kv_value_store_slice_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h
      head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h head_num BLOCK_HEAD BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s head_num BLOCK_HEAD BLOCK_DMODEL)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
          stride_k_d stride_os_bs stride_os_h idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_value_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := destindex_copy_quantize_kv_value_store_slice_correct K DestLoc Out
    OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    stride_os_bs stride_os_h head_num BLOCK_HEAD BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Named value writeback coverage for Python test cases 1 and 2
(`H = 8`, `D = 64`, `BLOCK_HEAD = 8`). -/
theorem destindex_copy_quantize_kv_test_h8_d64_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [8, 64] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h
        stride_o_d stride_os_bs stride_os_h 8 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 64)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h
          stride_o_d idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
          stride_k_d stride_os_bs stride_os_h idx) := by
  exact destindex_copy_quantize_kv_value_store_slice_compute_correct K DestLoc
    Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h
    stride_o_d stride_os_bs stride_os_h 8 8 64 s hOutInj

/-- Named value writeback coverage for Python test case 3
(`H = 8`, `D = 256`, `BLOCK_HEAD = 8`). -/
theorem destindex_copy_quantize_kv_test_h8_d256_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [8, 256] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h
        stride_o_d stride_os_bs stride_os_h 8 8 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 256)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h
          stride_o_d idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale stride_k_bs stride_k_h
          stride_k_d stride_os_bs stride_os_h idx) := by
  exact destindex_copy_quantize_kv_value_store_slice_compute_correct K DestLoc
    Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h
    stride_o_d stride_os_bs stride_os_h 8 8 256 s hOutInj

/-- Proof-oriented scale writeback slice of `quantize_copy_kv.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

The full kernel computes `data_scale = max(abs(src_data), axis=1) / 127` and
stores it to `Out_scale` with destination-indexed addressing. This slice starts
from a precomputed per-head `Scale` region and proves the observed scale store
surface independently of the integer value-store cast surface. -/
def destindex_copy_quantize_kv_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_s_h stride_os_bs stride_os_h
      head_num BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_h < $(head_num)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) +
      offs_h * $(stride_s_h), mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      offs_h * $(stride_os_h), data_scale, mask=mask)
}

def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num

instance scaleActiveDecidable (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) :
    Decidable (scaleActive head_num BLOCK_HEAD i) := by
  unfold scaleActive
  infer_instance

def scaleSourceOffset (s : BlockState) (stride_s_bs stride_s_h : Nat)
    (i : Fin BLOCK_HEAD) : Nat :=
  s.pids 0 * stride_s_bs + i.val * stride_s_h

def scaleOutOffset1
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + i.val * stride_os_h

noncomputable def quantizeCopyKvScaleSpec
    (s : BlockState) (Scale : RegionName)
    (stride_s_bs stride_s_h : Nat) (i : Fin BLOCK_HEAD) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs stride_s_h i)

theorem destindex_copy_quantize_kv_scale_store_slice_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_s_h stride_os_bs stride_os_h
      head_num BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HEAD =>
        scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)) :
    ∀ i : Fin BLOCK_HEAD,
      let outAddr := scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i
      (exec (destindex_copy_quantize_kv_scale_store_slice Scale DestLoc OutScale
            stride_s_bs stride_s_h stride_os_bs stride_os_h head_num
            BLOCK_HEAD) s).map (·.readMem OutScale outAddr)
        = some (if scaleActive head_num BLOCK_HEAD i then
            quantizeCopyKvScaleSpec s Scale stride_s_bs stride_s_h i
          else s.readMem OutScale outAddr) := by
  intro i
  simp [exec, destindex_copy_quantize_kv_scale_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, destIndex, scaleSourceOffset, scaleOutOffset1]
  let offsetFn : TileIndex [BLOCK_HEAD] → Nat :=
    fun i =>
      (match s.readMemTyped TileDType.nat DestLoc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_os_bs +
        i.1.val * stride_os_h
  let valueFn : TileIndex [BLOCK_HEAD] → ℝ :=
    fun i =>
      WithBot.unbotD 0
        (if i.1.val < head_num then
          some (s.readMem Scale (s.pids 0 * stride_s_bs + i.1.val * stride_s_h))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_HEAD] → Prop :=
    fun i => i.1.val < head_num
  have hOffsetInj : Function.Injective offsetFn := by
    intro a b hab
    have hInner :
        scaleOutOffset1 s DestLoc stride_os_bs stride_os_h a.1 =
          scaleOutOffset1 s DestLoc stride_os_bs stride_os_h b.1 := by
      simpa [offsetFn, scaleOutOffset1, destIndex, BlockState.readMemValue] using hab
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
      quantizeCopyKvScaleSpec s Scale stride_s_bs stride_s_h i
    else s.readMem OutScale (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj
    (i, PUnit.unit)]
  by_cases hi : i.val < head_num
  · simp [offsetFn, valueFn, P, scaleActive, quantizeCopyKvScaleSpec,
      scaleSourceOffset, scaleOutOffset1, destIndex, BlockState.readMemValue,
      hi]
  · simp [offsetFn, valueFn, P, scaleActive, scaleOutOffset1, destIndex,
      BlockState.readMemValue, hi]

theorem destindex_copy_quantize_kv_scale_store_slice_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_s_h stride_os_bs stride_os_h
      head_num BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HEAD =>
        scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h head_num
        BLOCK_HEAD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive head_num BLOCK_HEAD)
        (fun i => (OutScale,
          scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeCopyKvScaleSpec s Scale stride_s_bs stride_s_h i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_scale_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := destindex_copy_quantize_kv_scale_store_slice_correct Scale DestLoc
    OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h head_num
    BLOCK_HEAD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Named scale writeback coverage for all Python tests
(`H = 8`, `BLOCK_HEAD = 8`).  The `D=64` and `D=256` cases share this
per-head scale writeback shape. -/
theorem destindex_copy_quantize_kv_test_h8_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_s_h stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 8 =>
        scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale,
          scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeCopyKvScaleSpec s Scale stride_s_bs stride_s_h i) := by
  exact destindex_copy_quantize_kv_scale_store_slice_compute_correct Scale
    DestLoc OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h 8 8
    s hOutInj

/-! ## Python test-shape wrappers

The checked Python tests use `B * N_CTX = 8192` and `H = 8`. Cases 1 and 2
use `D = 64`, so `K/Out` strides are `(512, 64, 1)` and `Out_scale` strides
are `(8, 1, 1)`. Case 3 uses `D = 256`, so `K/Out` strides are
`(2048, 256, 1)` with the same per-head scale layout `(8, 1, 1)`. -/

theorem destindex_copy_quantize_kv_python_d64_value_offset_injective
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

theorem destindex_copy_quantize_kv_python_d256_value_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [8, 256] => outOffset s DestLoc 2048 256 1 idx) := by
  rintro ⟨⟨ha, hha⟩, ⟨da, hda⟩, _⟩ ⟨⟨hb, hhb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, destIndex, headIndex, dimIndex] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem destindex_copy_quantize_kv_python_scale_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun i : Fin 8 => scaleOutOffset1 s DestLoc 8 1 i) := by
  intro a b h
  simp [scaleOutOffset1, destIndex] at h
  exact Fin.ext (by omega)

theorem destindex_copy_quantize_kv_python_d64_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 512 64 1 512 64 1 8 1 8 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 64)
        (fun idx => (Out, outOffset s DestLoc 512 64 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 512 64 1 8 1 idx) := by
  exact destindex_copy_quantize_kv_value_store_slice_compute_correct K DestLoc
    Out OutScale 512 64 1 512 64 1 8 1 8 8 64 s
    (destindex_copy_quantize_kv_python_d64_value_offset_injective s DestLoc)

theorem destindex_copy_quantize_kv_python_d256_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 2048 256 1 2048 256 1 8 1 8 8 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 256)
        (fun idx => (Out, outOffset s DestLoc 2048 256 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 2048 256 1 8 1 idx) := by
  exact destindex_copy_quantize_kv_value_store_slice_compute_correct K DestLoc
    Out OutScale 2048 256 1 2048 256 1 8 1 8 8 256 s
    (destindex_copy_quantize_kv_python_d256_value_offset_injective s DestLoc)

theorem destindex_copy_quantize_kv_python_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale 8 1 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset1 s DestLoc 8 1 i)))
      (expected := fun i => quantizeCopyKvScaleSpec s Scale 8 1 i) := by
  exact destindex_copy_quantize_kv_scale_store_slice_compute_correct Scale
    DestLoc OutScale 8 1 8 1 8 8 s
    (destindex_copy_quantize_kv_python_scale_offset_injective s DestLoc)

/-- Python cases 1 and 2 (`D = 64`) expose both destination-indexed value
writeback and per-head scale writeback. -/
theorem destindex_copy_quantize_kv_python_d64_all_outputs_compute_correct
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 512 64 1 512 64 1 8 1 8 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 64)
        (fun idx => (Out, outOffset s DestLoc 512 64 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 512 64 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale 8 1 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset1 s DestLoc 8 1 i)))
      (expected := fun i => quantizeCopyKvScaleSpec s Scale 8 1 i)) := by
  constructor
  · exact destindex_copy_quantize_kv_python_d64_value_store_compute_correct
      K DestLoc Out OutScale s
  · exact destindex_copy_quantize_kv_python_scale_store_compute_correct
      Scale DestLoc OutScale s

/-- Python case 3 (`D = 256`) exposes both destination-indexed value writeback
and per-head scale writeback. -/
theorem destindex_copy_quantize_kv_python_d256_all_outputs_compute_correct
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 2048 256 1 2048 256 1 8 1 8 8 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 256)
        (fun idx => (Out, outOffset s DestLoc 2048 256 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 2048 256 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale 8 1 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset1 s DestLoc 8 1 i)))
      (expected := fun i => quantizeCopyKvScaleSpec s Scale 8 1 i)) := by
  constructor
  · exact destindex_copy_quantize_kv_python_d256_value_store_compute_correct
      K DestLoc Out OutScale s
  · exact destindex_copy_quantize_kv_python_scale_store_compute_correct
      Scale DestLoc OutScale s

/-- Public Python `D = 64` summary: the checked Python shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
theorem destindex_copy_quantize_kv_python_d64_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        512 64 1 512 64 1 8 1 1 8 64 8).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 512 64 1 512 64 1 8 1 8 8 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 64)
        (fun idx => (Out, outOffset s DestLoc 512 64 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 512 64 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale 8 1 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset1 s DestLoc 8 1 i)))
      (expected := fun i => quantizeCopyKvScaleSpec s Scale 8 1 i))) := by
  constructor
  · exact destindex_copy_quantize_kv_real_surface_toAlgorithm_supported K
      DestLoc Out OutScale 512 64 1 512 64 1 8 1 1 8 64 8
  · exact destindex_copy_quantize_kv_python_d64_all_outputs_compute_correct
      K Scale DestLoc Out OutScale s

/-- Public Python `D = 256` summary: the checked Python shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
theorem destindex_copy_quantize_kv_python_d256_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        2048 256 1 2048 256 1 8 1 1 8 256 8).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_value_store_slice K DestLoc Out
        OutScale 2048 256 1 2048 256 1 8 1 8 8 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 8 8 256)
        (fun idx => (Out, outOffset s DestLoc 2048 256 1 idx)))
      (expected := fun idx =>
        quantizeCopyKvValueSpec s K DestLoc OutScale 2048 256 1 8 1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_scale_store_slice Scale DestLoc
        OutScale 8 1 8 1 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 8 8)
        (fun i => (OutScale, scaleOutOffset1 s DestLoc 8 1 i)))
      (expected := fun i => quantizeCopyKvScaleSpec s Scale 8 1 i))) := by
  constructor
  · exact destindex_copy_quantize_kv_real_surface_toAlgorithm_supported K
      DestLoc Out OutScale 2048 256 1 2048 256 1 8 1 1 8 256 8
  · exact destindex_copy_quantize_kv_python_d256_all_outputs_compute_correct
      K Scale DestLoc Out OutScale s

/-- `output_summary` alias for the Python `D = 64` quantized KV copy case. -/
abbrev destindex_copy_quantize_kv_python_d64_output_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :=
  destindex_copy_quantize_kv_python_d64_summary K Scale DestLoc Out OutScale s

/-- `output_summary` alias for the Python `D = 256` quantized KV copy case. -/
abbrev destindex_copy_quantize_kv_python_d256_output_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :=
  destindex_copy_quantize_kv_python_d256_summary K Scale DestLoc Out OutScale s

/-- Complete checked-shape target for `quantize_copy_kv.py`.

The Python tests exercise both `D = 64` and `D = 256`; each case summary keeps
the faithful max/scale/int8-cast surface and proves the externally visible
value and scale outputs for the corresponding layout. -/
abbrev destindex_copy_quantize_kv_python_test_shape_complete_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :=
  And.intro
    (destindex_copy_quantize_kv_python_d64_summary K Scale DestLoc Out
      OutScale s)
    (destindex_copy_quantize_kv_python_d256_summary K Scale DestLoc Out
      OutScale s)

end VeriTile.Bench.TritonBenchG.QuantizeCopyKv
