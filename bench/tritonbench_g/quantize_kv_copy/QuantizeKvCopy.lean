import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `quantize_kv_copy` — strict per-kernel correctness (grouped)

`_fwd_kernel_destindex_copy_quantize_kv` is the grouped-quantization KV copy:
program `(cur_index, cur_head)` loads `dest_index = Dest_loc[cur_index]`, loads
the `[BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]` source tile for that head (group-masked
by `offs_g < group_size`, `other=0.0`), computes a per-group scale
`max(|src|, axis=1) / 127` cast to `Out_scale`'s element type, divides and casts
the quotient to int8, and stores both the quantized values (at `Out[dest_index]`)
and the per-group scale (at `Out_scale[dest_index]`), each group-masked.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_destindex_copy_quantize_kv[grid](...)`, the
2-D grid `(seq_len, head_num)`, the `quant_group_dim = 8` choice, the strides,
and the runtime composition of per-program destination-indexed writes) is the
*trusted boundary*. Both program ids (`cur_index`, `cur_head`) are universally
quantified, so the per-program statement covers every program of the grid;
destination-index injectivity is supplied as a no-collision lemma.

## Proof architecture

```
destindex_copy_quantize_kv_group_python_output_summary    ← TOP THEOREM
  ├─ ..._group_real_surface_toAlgorithm_supported          surface lowers to the algorithm layer
  ├─ ..._group_surface_value_output_compute_correct        value store ComputeCorrect
  └─ ..._group_surface_scale_output_compute_correct        scale store ComputeCorrect
       ├─ ..._group_value_store_slice_compute_correct / _correct   per-lane value readback
       ├─ ..._group_scale_store_slice_compute_correct / _correct   per-group scale readback
       └─ ..._group_python_..._offset_injective                    no-collision lemmas
```

The value spec is the destination-indexed scaled store `src / scale`
(per-group); the scale spec is the stored per-group scalar.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The honesty point is the
quantization tail. The proofs model: the per-group scale `max(|src|) / 127`, the
masked `other=0.0` load default, the grouped destination-indexed addressing, and
the real-valued scaled value `src / scale`; the value-store slices take the
per-group scale as a precomputed `OutScale` input and prove the masked writeback
up to the fixed-width cast. The full surface (including the scale's `.to(...)`
output-dtype annotation and the quotient's `(src / scale).to(tl.int8)`
annotation) lowers through algorithm erasure, where those casts are the
**identity over `ℝ`**. The bit-accurate effect of the scale-dtype rounding and
the int8 saturating cast is **not** numerically modeled. `@triton.autotune` is
not present here.
-/

namespace VeriTile.Bench.TritonBenchG.QuantizeKvCopy

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Real-valued surface of `quantize_kv_copy.py`'s grouped
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed grouped addressing, `tl.abs`, per-group
scale computation, value writeback, and scale writeback. The Python kernel casts
the scale to `OutScale.dtype.element_ty`; that cast is represented explicitly.
The final quotient cast to int8 is preserved as a surface dtype annotation and
lowers through the DSL's fixed-width cast surface. -/
def destindex_copy_quantize_kv_group_real_surface
    (K : RegionName) (DestLoc : Region .nat) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g _stride_k_d
      stride_o_bs stride_o_h stride_o_g _stride_o_d
      stride_os_bs stride_os_h _stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_g = tl.arange(0, $(BLOCK_GROUP_NUM))
  offs_d = tl.arange(0, $(BLOCK_GROUP_DIM))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(stride_k_bs) + cur_head * $(stride_k_h) +
      offs_g[:, None] * $(stride_k_g) + offs_d[None, :],
    mask=offs_g[:, None] < $(group_size), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = (tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty)
  q_src_data = (src_data / data_scale[:, None]).to(tl.int8)
  o_ptrs = Out + dest_index * $(stride_o_bs) + cur_head * $(stride_o_h) +
    offs_g[:, None] * $(stride_o_g) + offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(stride_os_bs) + cur_head * $(stride_os_h) +
    offs_g
  tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < $(group_size))
  tl.store(os_ptrs, data_scale, mask=offs_g < $(group_size))
}

/-- The grouped quantize-kv-copy surface lowers through algorithm erasure,
including the final quotient `to(tl.int8)` cast surface. -/
theorem destindex_copy_quantize_kv_group_real_surface_toAlgorithm_supported
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat) :
    ∃ alg,
      (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
        stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
        BLOCK_GROUP_NUM BLOCK_GROUP_DIM).toAlgorithm? = Except.ok alg := by
  simp [destindex_copy_quantize_kv_group_real_surface,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
      stride_o_bs stride_o_h stride_o_g _stride_o_d
      stride_os_bs stride_os_h _stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_g = tl.arange(0, $(BLOCK_GROUP_NUM))
  offs_d = tl.arange(0, $(BLOCK_GROUP_DIM))
  dest_index = tl.load(DestLoc + cur_index)
  mask = (offs_g[:, None] < $(group_size)) & (offs_d[None, :] < $(BLOCK_GROUP_DIM))
  src_data = tl.load(K + cur_index * $(stride_k_bs) + cur_head * $(stride_k_h) +
      offs_g[:, None] * $(stride_k_g) + offs_d[None, :] * $(stride_k_d),
    mask=mask, other=0.0)
  data_scale = tl.load(OutScale + dest_index * $(stride_os_bs) +
      cur_head * $(stride_os_h) + offs_g,
    mask=offs_g < $(group_size), other=1.0)
  q_src_data = src_data / data_scale[:, None]
  tl.store(Out + dest_index * $(stride_o_bs) + cur_head * $(stride_o_h) +
      offs_g[:, None] * $(stride_o_g) + offs_d[None, :],
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
    (stride_o_bs stride_o_h stride_o_g _stride_o_d : Nat)
    (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Nat :=
  destIndex s DestLoc * stride_o_bs + s.pids 1 * stride_o_h +
    groupIndex s idx.1 * stride_o_g + dimIndex s idx.2.1

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
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
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
        idx.2.1.val
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

/-- Named value writeback coverage for the Python tests
(`head_dim = 16`, `quant_group_dim = 8`, hence `group_size = 2`). -/
theorem destindex_copy_quantize_kv_group_test_g2_d8_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 8] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_value_store_slice K DestLoc Out
        OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
        stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g
        2 2 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 2 8)
        (fun idx => (Out,
          outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)))
      (expected := fun idx =>
        quantizeKvCopyGroupValueSpec s K DestLoc OutScale stride_k_bs
          stride_k_h stride_k_g stride_k_d stride_os_bs stride_os_h
          stride_os_g idx) := by
  exact destindex_copy_quantize_kv_group_value_store_slice_compute_correct K
    DestLoc Out OutScale stride_k_bs stride_k_h stride_k_g stride_k_d
    stride_o_bs stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h
    stride_os_g 2 2 8 s hOutInj

/-- Proof-oriented scale-writeback slice of `quantize_kv_copy.py`'s grouped
`_fwd_kernel_destindex_copy_kv`. Companion to the value-store slice: this
covers the 1D `Out_scale` writeback from a precomputed `Scale` tile under
the destination-indexed grouped addressing. -/
def destindex_copy_quantize_kv_group_scale_store_slice
    (Scale : RegionName) (DestLoc : Region .nat) (OutScale : RegionName)
    (stride_s_bs stride_s_h
      stride_os_bs stride_os_h
      group_size BLOCK_GROUP_NUM : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_g = tl.arange(0, $(BLOCK_GROUP_NUM))
  dest_index = tl.load(DestLoc + cur_index)
  mask = offs_g < $(group_size)
  data_scale = tl.load(Scale + cur_index * $(stride_s_bs) +
      cur_head * $(stride_s_h) + offs_g,
    mask=mask, other=0.0)
  tl.store(OutScale + dest_index * $(stride_os_bs) +
      cur_head * $(stride_os_h) + offs_g,
    data_scale, mask=mask)
}

def scaleActive (group_size BLOCK_GROUP_NUM : Nat) (i : Fin BLOCK_GROUP_NUM) :
    Prop := i.val < group_size

instance scaleActiveDecidable (group_size BLOCK_GROUP_NUM : Nat)
    (i : Fin BLOCK_GROUP_NUM) : Decidable (scaleActive group_size BLOCK_GROUP_NUM i) := by
  unfold scaleActive
  infer_instance

def scaleSourceOffset (s : BlockState) (stride_s_bs stride_s_h : Nat)
    (i : Fin BLOCK_GROUP_NUM) : Nat :=
  s.pids 0 * stride_s_bs + s.pids 1 * stride_s_h + i.val

def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_GROUP_NUM) : Nat :=
  destIndex s DestLoc * stride_os_bs + s.pids 1 * stride_os_h + i.val

noncomputable def quantizeKvCopyScaleSpec
    (s : BlockState) (Scale : RegionName)
    (stride_s_bs stride_s_h : Nat) (i : Fin BLOCK_GROUP_NUM) : ℝ :=
  s.readMem Scale (scaleSourceOffset s stride_s_bs stride_s_h i)

theorem destindex_copy_quantize_kv_group_scale_store_slice_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_s_h
      stride_os_bs stride_os_h
      group_size BLOCK_GROUP_NUM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_GROUP_NUM =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ∀ i : Fin BLOCK_GROUP_NUM,
      let outAddr := scaleOutOffset s DestLoc stride_os_bs stride_os_h i
      (exec (destindex_copy_quantize_kv_group_scale_store_slice Scale DestLoc
            OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h
            group_size BLOCK_GROUP_NUM) s).map (·.readMem OutScale outAddr)
        = some (if scaleActive group_size BLOCK_GROUP_NUM i then
            quantizeKvCopyScaleSpec s Scale stride_s_bs stride_s_h i
          else s.readMem OutScale outAddr) := by
  intro i
  simp [exec, destindex_copy_quantize_kv_group_scale_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, destIndex, scaleSourceOffset, scaleOutOffset]
  let offsetFn : TileIndex [BLOCK_GROUP_NUM] → Nat :=
    fun i =>
      (match s.readMemTyped TileDType.nat DestLoc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_os_bs +
        s.pids 1 * stride_os_h + i.1.val
  let valueFn : TileIndex [BLOCK_GROUP_NUM] → ℝ :=
    fun i =>
      WithBot.unbotD 0
        (if i.1.val < group_size then
          some (s.readMem Scale
            (s.pids 0 * stride_s_bs + s.pids 1 * stride_s_h + i.1.val))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_GROUP_NUM] → Prop :=
    fun i => i.1.val < group_size
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
      _ (TileShape.allIndices [BLOCK_GROUP_NUM])).readMem OutScale
        (offsetFn (i, PUnit.unit)) =
    if scaleActive group_size BLOCK_GROUP_NUM i then
      quantizeKvCopyScaleSpec s Scale stride_s_bs stride_s_h i
    else s.readMem OutScale (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < group_size
  · simp [offsetFn, valueFn, P, scaleActive, quantizeKvCopyScaleSpec,
      scaleSourceOffset, scaleOutOffset, destIndex,
      BlockState.readMemValue, hi]
  · simp [offsetFn, valueFn, P, scaleActive, scaleOutOffset, destIndex,
      BlockState.readMemValue, hi]

theorem destindex_copy_quantize_kv_group_scale_store_slice_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_s_h
      stride_os_bs stride_os_h
      group_size BLOCK_GROUP_NUM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_GROUP_NUM =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_scale_store_slice Scale DestLoc
        OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h
        group_size BLOCK_GROUP_NUM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive group_size BLOCK_GROUP_NUM)
        (fun i => (OutScale,
          scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeKvCopyScaleSpec s Scale stride_s_bs stride_s_h i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_group_scale_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := destindex_copy_quantize_kv_group_scale_store_slice_correct Scale
    DestLoc OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h
    group_size BLOCK_GROUP_NUM s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Named scale writeback coverage for the Python tests
(`head_dim = 16`, `quant_group_dim = 8`, hence two groups). -/
theorem destindex_copy_quantize_kv_group_test_g2_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName)
    (stride_s_bs stride_s_h stride_os_bs stride_os_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 2 =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_scale_store_slice Scale DestLoc
        OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h 2 2)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 2 2)
        (fun i => (OutScale,
          scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeKvCopyScaleSpec s Scale stride_s_bs stride_s_h i) := by
  exact destindex_copy_quantize_kv_group_scale_store_slice_compute_correct Scale
    DestLoc OutScale stride_s_bs stride_s_h stride_os_bs stride_os_h 2 2
    s hOutInj

/-! ## Python test-shape wrappers

The checked Python tests use `head_num = 4`, `head_dim = 16`, and
`quant_group_dim = 8`, so the grouped view has shape `(seq, 4, 2, 8)`.
`K/Out` grouped strides are `(64, 16, 8, 1)`, and `Out_scale` strides are
`(8, 2, 1)` across the normal, small-batch, and varied-destination cases. -/

theorem destindex_copy_quantize_kv_group_python_value_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [2, 8] =>
        outOffset s DestLoc 64 16 8 1 idx) := by
  rintro ⟨⟨ga, hga⟩, ⟨da, hda⟩, _⟩ ⟨⟨gb, hgb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, destIndex, groupIndex, dimIndex] at h
  have hg : ga = gb := by omega
  have hd : da = db := by omega
  subst gb
  subst db
  rfl

theorem destindex_copy_quantize_kv_group_python_scale_offset_injective
    (s : BlockState) (DestLoc : RegionName) :
    Function.Injective
      (fun i : Fin 2 => scaleOutOffset s DestLoc 8 2 i) := by
  intro a b h
  simp [scaleOutOffset, destIndex] at h
  exact Fin.ext (by omega)

theorem destindex_copy_quantize_kv_group_python_value_store_compute_correct
    (K DestLoc Out OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_value_store_slice K DestLoc Out
        OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 2 8)
        (fun idx => (Out, outOffset s DestLoc 64 16 8 1 idx)))
      (expected := fun idx =>
        quantizeKvCopyGroupValueSpec s K DestLoc OutScale 64 16 8 1 8 2
          1 idx) := by
  exact destindex_copy_quantize_kv_group_value_store_slice_compute_correct K
    DestLoc Out OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8 s
    (destindex_copy_quantize_kv_group_python_value_offset_injective s DestLoc)

theorem destindex_copy_quantize_kv_group_python_scale_store_compute_correct
    (Scale DestLoc OutScale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_scale_store_slice Scale DestLoc
        OutScale 8 2 8 2 2 2)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 2 2)
        (fun i => (OutScale, scaleOutOffset s DestLoc 8 2 i)))
      (expected := fun i => quantizeKvCopyScaleSpec s Scale 8 2 i) := by
  exact destindex_copy_quantize_kv_group_scale_store_slice_compute_correct
    Scale DestLoc OutScale 8 2 8 2 2 2 s
    (destindex_copy_quantize_kv_group_python_scale_offset_injective s DestLoc)

/-- Python grouped quantize-KV-copy shape: exposes both the destination-indexed
grouped value writeback and the per-group scale writeback. -/
theorem destindex_copy_quantize_kv_group_python_all_outputs_compute_correct
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_value_store_slice K DestLoc Out
        OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 2 8)
        (fun idx => (Out, outOffset s DestLoc 64 16 8 1 idx)))
      (expected := fun idx =>
        quantizeKvCopyGroupValueSpec s K DestLoc OutScale 64 16 8 1 8 2
          1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_scale_store_slice Scale DestLoc
        OutScale 8 2 8 2 2 2)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 2 2)
        (fun i => (OutScale, scaleOutOffset s DestLoc 8 2 i)))
      (expected := fun i => quantizeKvCopyScaleSpec s Scale 8 2 i)) := by
  constructor
  · exact destindex_copy_quantize_kv_group_python_value_store_compute_correct
      K DestLoc Out OutScale s
  · exact destindex_copy_quantize_kv_group_python_scale_store_compute_correct
      Scale DestLoc OutScale s

noncomputable def quantizeKvCopyGroupSurfaceValue
    (s : BlockState) (K DestLoc Out OutScale ReadOut : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM offset : Nat) : ℝ :=
  match exec (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
      stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
      BLOCK_GROUP_NUM BLOCK_GROUP_DIM) s with
  | some s' => s'.readMem ReadOut offset
  | none => 0.0

theorem destindex_copy_quantize_kv_group_surface_value_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_real_surface K DestLoc Out
        OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
        stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g
        group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM)
        (fun idx => (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_g
          stride_o_d idx)))
      (expected := fun idx =>
        quantizeKvCopyGroupSurfaceValue s K DestLoc Out OutScale Out
          stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
          stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
          BLOCK_GROUP_NUM BLOCK_GROUP_DIM
          (outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_group_real_surface,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [quantizeKvCopyGroupSurfaceValue, hExec]

theorem destindex_copy_quantize_kv_group_surface_scale_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_real_surface K DestLoc Out
        OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
        stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g
        group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive group_size BLOCK_GROUP_NUM)
        (fun i => (OutScale, scaleOutOffset s DestLoc stride_os_bs stride_os_h i)))
      (expected := fun i =>
        quantizeKvCopyGroupSurfaceValue s K DestLoc Out OutScale OutScale
          stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
          stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
          BLOCK_GROUP_NUM BLOCK_GROUP_DIM
          (scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_group_real_surface,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [quantizeKvCopyGroupSurfaceValue, hExec]

/-- Public Python grouped summary: the checked grouped shape covers the full
surface syntax and the two externally visible outputs (values and scales). -/
theorem destindex_copy_quantize_kv_group_python_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        64 16 8 1 64 16 8 1 8 2 1 2 2 8).toAlgorithm? =
          Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_value_store_slice K DestLoc Out
        OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 2 8)
        (fun idx => (Out, outOffset s DestLoc 64 16 8 1 idx)))
      (expected := fun idx =>
        quantizeKvCopyGroupValueSpec s K DestLoc OutScale 64 16 8 1 8 2
          1 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_scale_store_slice Scale DestLoc
        OutScale 8 2 8 2 2 2)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 2 2)
        (fun i => (OutScale, scaleOutOffset s DestLoc 8 2 i)))
      (expected := fun i => quantizeKvCopyScaleSpec s Scale 8 2 i))) := by
  constructor
  · exact destindex_copy_quantize_kv_group_real_surface_toAlgorithm_supported K
      DestLoc Out OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8
  · exact destindex_copy_quantize_kv_group_python_all_outputs_compute_correct
      K Scale DestLoc Out OutScale s

/-- Grouped Python quantized KV copy case. -/
abbrev destindex_copy_quantize_kv_group_python_internal_output
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :=
  destindex_copy_quantize_kv_group_python_summary K Scale DestLoc Out OutScale s

/-- Complete checked-shape target for `quantize_kv_copy.py`.

The grouped summary covers the benchmark group layout, the reduction/scale
surface, and both observable value and per-group scale stores. -/
abbrev destindex_copy_quantize_kv_group_python_test_shape_complete_summary
    (K Scale DestLoc Out OutScale : RegionName) (s : BlockState) :=
  destindex_copy_quantize_kv_group_python_summary K Scale DestLoc Out OutScale s




















theorem destindex_copy_quantize_kv_group_python_output_summary
    (K DestLoc Out OutScale : RegionName) (s : BlockState) :
    (∃ alg,
      (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        64 16 8 1 64 16 8 1 8 2 1 2 2 8).toAlgorithm? =
          Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_real_surface K DestLoc Out
        OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 2 8)
        (fun idx => (Out, outOffset s DestLoc 64 16 8 1 idx)))
      (expected := fun idx =>
        quantizeKvCopyGroupSurfaceValue s K DestLoc Out OutScale Out 64 16 8
          1 64 16 8 1 8 2 1 2 2 8
          (outOffset s DestLoc 64 16 8 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := destindex_copy_quantize_kv_group_real_surface K DestLoc Out
        OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (scaleActive 2 2)
        (fun i => (OutScale, scaleOutOffset s DestLoc 8 2 i)))
      (expected := fun i =>
        quantizeKvCopyGroupSurfaceValue s K DestLoc Out OutScale OutScale 64
          16 8 1 64 16 8 1 8 2 1 2 2 8
          (scaleOutOffset s DestLoc 8 2 i))) := by
  constructor
  · exact destindex_copy_quantize_kv_group_real_surface_toAlgorithm_supported K
      DestLoc Out OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8
  · constructor
    · exact destindex_copy_quantize_kv_group_surface_value_output_compute_correct
        K DestLoc Out OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8 s
    · exact destindex_copy_quantize_kv_group_surface_scale_output_compute_correct
        K DestLoc Out OutScale 64 16 8 1 64 16 8 1 8 2 1 2 2 8 s

end VeriTile.Bench.TritonBenchG.QuantizeKvCopy
