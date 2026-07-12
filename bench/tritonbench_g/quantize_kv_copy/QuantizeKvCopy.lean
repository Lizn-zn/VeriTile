import VeriTile.Triton
import VeriTile.Examples.AttentionForwardClosedForm

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
destindex_copy_quantize_kv_group_output_summary_general       ← ★ MAIN THEOREM (ComputeCorrect.Realizes_without_Rounding conjunction, symbolic dims/strides)
  ├─ ..._group_real_surface_toAlgorithm_supported             surface lowers to the algorithm layer
  ├─ ..._group_real_surface_value_output_compute_correct      genuine int8 value readback
  │    ├─ scatter_readback_int_prop_masked_nd                 per-lane int store readback
  │    ├─ foldl_writeMem_prop_masked_readMemValue_int_other    peel trailing scale store
  │    └─ (value destination-offset injectivity — theorem hypothesis)
  └─ ..._group_real_surface_scale_output_compute_correct      genuine scale readback
       ├─ BlockState.scatter_readback_prop_masked_nd          per-group scale store readback
       ├─ foldl_writeMemTyped_int_prop_masked_readMem_other   peel prior int store
       └─ (scale destination-offset injectivity — theorem hypothesis)
```

The genuine value spec (`quantizeKvCopyGroupSurfaceIntValue`) is the int8 cast of
`src / fp16(max(|src|, axis=1)/127)` read back through `readMemValue .int Out`;
the genuine scale spec (`quantizeKvCopyGroupScaleCell`) is the stored real cell
of `max(|src|, axis=1)/127` read back through `readMem OutScale` (the
`.to(Out_scale.dtype.element_ty)` cast lowers to the real identity). Both are
computed from the kernel **inputs**, so the top summary is not self-referential.
The slice-based `quantizeKvCopyGroupValueSpec` / `quantizeKvCopyScaleSpec` (with
precomputed scale) remain as supporting coverage.

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

/-! ## Integer memory infra

The genuine value store writes an `.int` channel (the `.to(tl.int8)` quotient)
while the scale store writes the `.real` channel. The following lemmas are the
`.int` analogues of the `.nat`/`.real` scatter-readback and cross-channel
preservation lemmas in `VeriTile.Triton.Semantics.State`. -/

/-- `.int` masked-foldl preservation: writes whose (masked) offsets all miss `o`
leave `readMemValue .int region o` unchanged. -/
private theorem foldl_writeMemTyped_int_masked_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → Int) (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
      BlockState.readMemValue ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
          .int region o
      = s.readMemValue .int region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    rw [List.foldl_cons]
    have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
      fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
    by_cases hmaskhd : mask hd = Bool.true
    · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
      simp only [hmaskhd, if_true]
      rw [ih _ htl]
      rw [BlockState.writeMemTyped_int_readMemValue_int]
      show (if region = region ∧ o = offsetFn hd then valueFn hd
          else s.readMemValue TileDType.int region o) =
        s.readMemValue TileDType.int region o
      rw [if_neg]
      rintro ⟨_, h_eq⟩
      exact hhd h_eq.symm
    · have hmaskhd' : mask hd = Bool.false := by
        rcases hmaskFalse : mask hd
        · rfl
        · exact absurd hmaskFalse hmaskhd
      simp only [hmaskhd', if_false, Bool.false_eq_true]
      exact ih _ htl

/-- `.int` scatter readback: reading `readMemValue .int` of the masked
`writeMemTyped .int` foldl at lane `i` returns the stored value if `P i`, else
the prior value. -/
theorem scatter_readback_int_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → Int) (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    BlockState.readMemValue ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s)
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change BlockState.readMemValue ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_int_masked_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    rw [BlockState.writeMemTyped_int_readMemValue_int]
    show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _) = valueFn i
    exact if_pos ⟨rfl, rfl⟩
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_int_masked_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

/-- Prop-masked real `writeMem` store foldl leaves `readMemValue .int` on a
different region unchanged. The genuine value readback reads `.int Out`; the
trailing real scale store to `OutScale` (the `.to(OutScale.dtype.element_ty)`
cast lowers to a real write) is peeled off with this. -/
theorem foldl_writeMem_prop_masked_readMemValue_int_other {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (R : RegionName) (off : Nat) (hRR : R ≠ region) :
    BlockState.readMemValue ((l.foldl
        (fun acc k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s))
      .int R off
      = s.readMemValue .int R off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hhd : P hd
      · simp only [hhd, if_true]
        rw [ih]
        unfold BlockState.writeMem BlockState.readMemValue BlockState.readMemTyped
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hhd, if_false]
        exact ih _

/-- `.int` store foldl leaves `readMem` (real) on a different region unchanged.
The scale readback reads the real cell of `OutScale`; the prior int value store
to `Out` is peeled off with this. -/
theorem foldl_writeMemTyped_int_prop_masked_readMem_other {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → Int)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (R : RegionName) (off : Nat) (hRR : R ≠ region) :
    ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s)).readMem R off
      = s.readMem R off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · simp only [hP, if_true]
        rw [ih]
        unfold BlockState.writeMemTyped BlockState.readMem
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hP, if_false]
        exact ih _

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
    ComputeCorrect.Realizes_without_Rounding
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

/-- Proof-oriented scale-writeback slice of `quantize_kv_copy.py`'s grouped
`_fwd_kernel_destindex_copy_quantize_kv`. Companion to the value-store slice: this
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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## Genuine value/scale specs for the full grouped real surface

The specs below are computed entirely from the kernel **inputs** (`K`,
`DestLoc`, the strides, `group_size`); they do not reference the executed state,
so the resulting `..._real_surface_..._compute_correct` theorems are not
self-referential. The scale spec is the per-group `max(|src|, axis=1) / 127`
(the `.to(Out_scale.dtype.element_ty)` cast lowers to the real identity in the
DSL, so the kernel stores this real value directly to `Out_scale`); the value
spec is the int8 cast of `src / scale`. Both `max`/`abs` and the masked
`other=0.0` load are modeled exactly; the output-dtype scale rounding and the
int8 saturation are the trusted fixed-width cast surfaces (see the modeling
boundary note above). -/

/-- The masked source lane value (`tl.load(..., other=0.0)`): the group-masked
`K[cur_index, cur_head, g, d]` as a `WithBot ℝ`. -/
noncomputable def maskedSrc
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size : Nat)
    (g d : Nat) : WithBot ℝ :=
  if g < group_size then
    some (s.readMem K
      (s.pids 0 * stride_k_bs + s.pids 1 * stride_k_h + g * stride_k_g + d))
  else some (0.0 : ℝ)

/-- The per-group `data_scale` *value* (`max(|src|, axis=1) / 127`): the row
reduce-max of `|maskedSrc|` divided by `127`, as a `WithBot ℝ`. This is the value
the kernel both divides by (for the int8 value) and stores to `Out_scale`. -/
noncomputable def quantizeKvCopyGroupScaleValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size BLOCK_GROUP_DIM : Nat)
    (hD : 0 < BLOCK_GROUP_DIM) (g : Nat) : WithBot ℝ :=
  Option.map (· / 127.0)
    ((Finset.univ.sup'
        (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_GROUP_DIM)).Nonempty)
        (fun x : Fin BLOCK_GROUP_DIM =>
          if maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size g x.val
              < (some 0 : WithBot ℝ) then
            NumericDType.real.sub (some 0)
              (maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size g x.val)
          else maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size g x.val) :
      WithBot ℝ))

/-- Genuine quantized value spec (`(src / data_scale).to(int8)`): the int8 cast
of the masked source lane divided by the per-group scale. -/
noncomputable def quantizeKvCopyGroupSurfaceIntValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size BLOCK_GROUP_DIM : Nat)
    (hD : 0 < BLOCK_GROUP_DIM) (idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM]) : Int :=
  WithBot.realToInt8
    (FloatDType.real.cast FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (maskedSrc s K stride_k_bs stride_k_h stride_k_g group_size idx.1.val idx.2.1.val)
        (quantizeKvCopyGroupScaleValue s K stride_k_bs stride_k_h stride_k_g
          group_size BLOCK_GROUP_DIM hD idx.1.val)))

/-- Genuine scale store cell (`data_scale`): the real value the kernel writes to
`Out_scale`, observed through `readMem`. -/
noncomputable def quantizeKvCopyGroupScaleCell
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_g group_size BLOCK_GROUP_DIM : Nat)
    (hD : 0 < BLOCK_GROUP_DIM) (g : Nat) : ℝ :=
  WithBot.unbotD 0
    (quantizeKvCopyGroupScaleValue s K stride_k_bs stride_k_h stride_k_g
      group_size BLOCK_GROUP_DIM hD g)

/-- **Genuine value-output correctness for the full grouped real surface.** The
destination-indexed int8 value store realizes `quantizeKvCopyGroupSurfaceIntValue`
(read back via `readMemValue .int Out`). Not self-referential: the expected value
is computed from the inputs. -/
theorem destindex_copy_quantize_kv_group_real_surface_value_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState) (hD : 0 < BLOCK_GROUP_DIM) (hOut : Out ≠ OutScale)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM],
      let outAddr := outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx
      (exec (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
            stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
            BLOCK_GROUP_NUM BLOCK_GROUP_DIM) s).map
          (·.readMemValue .int Out outAddr)
        = some (if active s group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM idx then
            quantizeKvCopyGroupSurfaceIntValue s K stride_k_bs stride_k_h
              stride_k_g group_size BLOCK_GROUP_DIM hD idx
          else s.readMemValue .int Out outAddr) := by
  intro idx
  simp only [outOffset, destIndex, groupIndex, dimIndex]
  simp [exec, destindex_copy_quantize_kv_group_real_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, dif_pos hD]
  simp only [Tile.remap, Broadcast.leftIndex, Broadcast.consL, Broadcast.consSame,
    Broadcast.consR, Broadcast.nil, decide_eq_true_eq]
  rw [foldl_writeMem_prop_masked_readMemValue_int_other OutScale _ _
    (fun i : TileIndex [BLOCK_GROUP_NUM] => i.1.val < group_size) _ _ Out _ hOut]
  rw [scatter_readback_int_prop_masked_nd (region := Out)
    (shape := [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM])
    _ (fun i => s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
        s.pids 1 * stride_o_h + i.1.val * stride_o_g + i.2.1.val)
    _ (fun i => i.1.val < group_size) ?_ idx]
  · simp only [active, groupIndex]
    by_cases h : idx.1.val < group_size
    · rw [if_pos h, if_pos h]
      simp only [quantizeKvCopyGroupSurfaceIntValue, quantizeKvCopyGroupScaleValue,
        maskedSrc, if_pos h, Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      simp only [BlockState.setReg_readMemValue]
  · simpa [outOffset, destIndex, groupIndex, dimIndex, BlockState.readMemValue]
      using hOutInj

/-- **Genuine scale-output correctness for the full grouped real surface.** The
destination-indexed scale store realizes `quantizeKvCopyGroupScaleCell` (read
back via `readMem OutScale`; the `.to(Out_scale.dtype.element_ty)` cast lowers to
the real identity, so the kernel writes the real `max(|src|)/127` cell). Not
self-referential. -/
theorem destindex_copy_quantize_kv_group_real_surface_scale_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState) (hD : 0 < BLOCK_GROUP_DIM) (hOut : OutScale ≠ Out)
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_GROUP_NUM =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ∀ i : Fin BLOCK_GROUP_NUM,
      let outAddr := scaleOutOffset s DestLoc stride_os_bs stride_os_h i
      (exec (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
            stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
            BLOCK_GROUP_NUM BLOCK_GROUP_DIM) s).map
          (·.readMem OutScale outAddr)
        = some (if scaleActive group_size BLOCK_GROUP_NUM i then
            quantizeKvCopyGroupScaleCell s K stride_k_bs stride_k_h stride_k_g
              group_size BLOCK_GROUP_DIM hD i.val
          else s.readMem OutScale outAddr) := by
  intro i
  simp only [scaleOutOffset, destIndex]
  simp [exec, destindex_copy_quantize_kv_group_real_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, dif_pos hD]
  simp only [Tile.remap, Broadcast.leftIndex, Broadcast.consL, Broadcast.consSame,
    Broadcast.consR, Broadcast.nil, decide_eq_true_eq]
  rw [BlockState.scatter_readback_prop_masked_nd (region := OutScale) (shape := [BLOCK_GROUP_NUM])
    _ (fun j => s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs +
        s.pids 1 * stride_os_h + j.1.val)
    _ (fun j => j.1.val < group_size) ?_ (i, PUnit.unit)]
  · simp only [scaleActive]
    show (if i.val < group_size then _ else _) = (if i.val < group_size then _ else _)
    by_cases h : i.val < group_size
    · rw [if_pos h, if_pos h]
      simp only [quantizeKvCopyGroupScaleCell, quantizeKvCopyGroupScaleValue, maskedSrc,
        if_pos h, Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      rw [foldl_writeMemTyped_int_prop_masked_readMem_other (region := Out)
        (P := fun k : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] => k.1.val < group_size)
        (hRR := hOut)]
      simp only [BlockState.setReg_readMem]
  · intro a b hab
    have h1 : a.1 = b.1 := by
      apply hScaleInj
      simpa [scaleOutOffset, destIndex, BlockState.readMemValue] using hab
    obtain ⟨a1, a2⟩ := a
    obtain ⟨b1, b2⟩ := b
    simp only at h1
    subst h1
    rfl

/-- **Dimension-general grouped summary (genuine, gap-free).** The symbolic
grouped summary: all strides, `group_size`, `BLOCK_GROUP_NUM`, and
`BLOCK_GROUP_DIM` are arbitrary `Nat` parameters. Stated on the standard
`ComputeCorrect.Realizes_without_Rounding` trust surface: the lowering conjunct plus one
`Realizes_without_Rounding` per stored output (the int8 value store and the real scale store),
each with a **total** write map so the inactive-lane guarantee lives inside
`expected` (active lanes get the genuine input-derived spec, inactive lanes keep
their prior memory). Honest hypotheses: `hD : 0 < BLOCK_GROUP_DIM` (nonempty
reduce axis), `hOut : Out ≠ OutScale` (no aliasing), and the value/scale
destination-offset injectivity (no-collision) hypotheses the genuine readbacks
need. Both expected values are computed from the kernel **inputs**, so this
summary is not self-referential. -/
specification destindex_copy_quantize_kv_group_output_summary_general
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_g stride_k_d
      stride_o_bs stride_o_h stride_o_g stride_o_d
      stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM : Nat)
    (s : BlockState) (hD : 0 < BLOCK_GROUP_DIM) (hOut : Out ≠ OutScale)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_GROUP_NUM =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    (∃ alg,
      (destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
        stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
        BLOCK_GROUP_NUM BLOCK_GROUP_DIM).toAlgorithm? =
          Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
        stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
        BLOCK_GROUP_NUM BLOCK_GROUP_DIM)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        some (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx))
      (expected := fun idx : TileIndex [BLOCK_GROUP_NUM, BLOCK_GROUP_DIM] =>
        if active s group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM idx then
          quantizeKvCopyGroupSurfaceIntValue s K stride_k_bs stride_k_h
            stride_k_g group_size BLOCK_GROUP_DIM hD idx
        else s.readMemValue .int Out
          (outOffset s DestLoc stride_o_bs stride_o_h stride_o_g stride_o_d idx)) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := destindex_copy_quantize_kv_group_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs stride_o_h
        stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g group_size
        BLOCK_GROUP_NUM BLOCK_GROUP_DIM)
      (initialState := s)
      (write := fun i : Fin BLOCK_GROUP_NUM =>
        some (OutScale, scaleOutOffset s DestLoc stride_os_bs stride_os_h i))
      (expected := fun i : Fin BLOCK_GROUP_NUM =>
        if scaleActive group_size BLOCK_GROUP_NUM i then
          quantizeKvCopyGroupScaleCell s K stride_k_bs stride_k_h stride_k_g
            group_size BLOCK_GROUP_DIM hD i.val
        else s.readMem OutScale
          (scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) := by
  refine ⟨destindex_copy_quantize_kv_group_real_surface_toAlgorithm_supported K
      DestLoc Out OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
      stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM, ?_, ?_⟩
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [destindex_copy_quantize_kv_group_real_surface,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro idx
    simp only [ComputeCorrect.OutputReadable.read_int]
    have h := destindex_copy_quantize_kv_group_real_surface_value_output_compute_correct K
      DestLoc Out OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
      stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM s hD hOut hOutInj idx
    rw [hExec] at h
    simpa using Option.some.inj h
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [destindex_copy_quantize_kv_group_real_surface,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro i
    simp only [ComputeCorrect.OutputReadable.read_real]
    have h := destindex_copy_quantize_kv_group_real_surface_scale_output_compute_correct K
      DestLoc Out OutScale stride_k_bs stride_k_h stride_k_g stride_k_d stride_o_bs
      stride_o_h stride_o_g stride_o_d stride_os_bs stride_os_h stride_os_g
      group_size BLOCK_GROUP_NUM BLOCK_GROUP_DIM s hD (fun h => hOut h.symm) hScaleInj i
    rw [hExec] at h
    simpa using Option.some.inj h

end VeriTile.Bench.TritonBenchG.QuantizeKvCopy
