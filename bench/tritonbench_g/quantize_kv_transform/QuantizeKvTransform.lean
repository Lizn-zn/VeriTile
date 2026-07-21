import VeriTile.Triton

/-!
# `quantize_kv_transform` — strict per-kernel correctness

`_fwd_kernel_destindex_copy_quantize_kv` copies and int8-quantizes a KV cache
entry to a destination-indexed slot, with both head and dim masking: program
`cur_index` loads `dest_index = Dest_loc[cur_index]`, loads the
`[BLOCK_HEAD, BLOCK_DMODEL]` source tile masked by
`(offs_h < head_num) & (offs_d < head_dim)` (`other=0.0`), computes a per-head
scale `max(|src|, axis=1) / 127` cast to `Out_scale`'s element type, divides and
casts the quotient to int8, and stores the quantized values (at `Out[dest_index]`,
head-and-dim masked) and the per-head scale (at `Out_scale[dest_index]`, head
masked).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_destindex_copy_quantize_kv[grid](...)`, the
grid `(seq_len,)`, `BLOCK_HEAD = next_power_of_2(head_num)` and `BLOCK_DMODEL =
next_power_of_2(head_dim)`, the strides, and the runtime composition of
per-program destination-indexed writes) is the *trusted boundary*. `cur_index`
is universally quantified, so the per-program statement covers every program of
the grid; destination-index injectivity is supplied as a no-collision lemma.

## Proof architecture

```
destindex_copy_quantize_kv_transform_output_summary_general             ← TOP THEOREM (dimension-general, Realizes_without_Rounding)
  ├─ ..._transform_real_surface_toAlgorithm_supported                    surface lowers to the algorithm layer
  ├─ ..._transform_real_surface_value_output_compute_correct            genuine int8 value readback (engine lemma)
  │    ├─ scatter_readback_int_prop_masked_nd                            per-lane int store readback (no-collision)
  │    └─ foldl_writeMem_prop_masked_readMemValue_int_other              peel trailing real scale store
  └─ ..._transform_real_surface_scale_output_compute_correct             genuine real scale readback (engine lemma)
       ├─ BlockState.scatter_readback_prop_masked_nd                     per-head real store readback
       └─ foldl_writeMemTyped_int_prop_masked_readMem_other              peel prior int value store
```

The top theorem is a single bundled headline: the lowering witness
(`∃ alg, … = Except.ok alg`) plus two `ComputeCorrect.Realizes_without_Rounding` conjuncts (one per
stored output), each with a total write map whose inactive-lane guarantee is
carried inside `expected`. The genuine value spec
(`quantizeKvTransformSurfaceIntValue`) is the int8 cast of
`src / (max(|src|, axis=1)/127)` read back through `readMemValue .int Out`; the
genuine scale spec (`quantizeKvTransformScaleCell`) is the stored real value of
`max(|src|, axis=1)/127` read back through `readMem OutScale`. Both are computed
from the kernel inputs, so the summary is not self-referential. The two
`..._compute_correct` engine lemmas prove the raw `(exec …).map = some (if …)`
readback that each `Realizes_without_Rounding` conjunct is discharged by. There are no pinned
per-shape summaries; the theorem is universally quantified over all strides and
block dimensions.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The honesty point is the
quantization tail. The proofs model: the per-head scale `max(|src|) / 127`, the
combined head-and-dim mask with `other=0.0` load default, the
destination-indexed addressing, and the real-valued scaled value `src / scale`.
Unlike `quantize_copy_kv`, the scale cast `.to(OutScale.dtype.element_ty)` lowers
as the **identity over `ℝ`** (the DSL erases `element_ty` casts), so the scale
store writes the real channel; the quotient `(src / scale).to(tl.int8)` writes
the int channel through the fixed-width int8 cast surface. The bit-accurate
effect of the scale-dtype rounding and the int8 saturating cast is **not**
numerically modeled. `@triton.autotune` is not present here.
-/

namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! ## Integer scatter / cross-channel memory infra

The genuine value store writes an `.int` channel (the `.to(tl.int8)` quotient).
The scale store writes the `.real` channel (the `.to(OutScale.dtype.element_ty)`
cast is the identity over `ℝ`). The lemmas below are the `.int` analogue of the
`.real` scatter-readback in `VeriTile.Triton.Semantics.State`, plus the
cross-channel preservation lemmas that peel one store off the other's readback
by region distinctness. -/

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

/-- `.int` typed masked-foldl preserves the register file (Prop mask). Needed so
the trailing scale store's `offs_h` register read reduces past the value store. -/
@[simp] theorem foldl_writeMemTyped_int_prop_masked_regs {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → Int)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (dtype' : TileDType) (shape : TileShape) (name : RegName) :
    ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s).regs
        dtype' shape name)
      = s.regs dtype' shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hP : P hd
      · simp [hP]
      · simp [hP]

/-- A masked `.real` store foldl to `OutScale` leaves `readMemValue .int` on a
different region `Out` unchanged. The value readback reads `.int Out`; the
trailing real scale store to `OutScale` is peeled off with this. -/
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

/-- A masked `.int` store foldl to `Out` leaves `readMem` (real) on a different
region `OutScale` unchanged. The scale readback reads `.real OutScale`; the
prior int value store to `Out` is peeled off with this. -/
theorem foldl_writeMemTyped_int_prop_masked_readMem_other {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → Int)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (R : RegionName) (off : Nat) (hRR : R ≠ region) :
    BlockState.readMem ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      R off
      = s.readMem R off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hhd : P hd
      · simp only [hhd, if_true]
        rw [ih]
        unfold BlockState.writeMemTyped BlockState.readMem
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hhd, if_false]
        exact ih _

/-- Real-valued surface of `quantize_kv_transform.py`'s
`_fwd_kernel_destindex_copy_quantize_kv`.

This preserves destination-indexed addressing, the `head_num/head_dim` mask,
`tl.abs`, per-head scale computation, value writeback, and scale writeback. The
Python kernel casts the scale to `OutScale.dtype.element_ty`; that cast is
represented explicitly. The final quotient cast to int8 is preserved as a
surface dtype annotation and lowers through the DSL's fixed-width cast
surface. -/
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
erasure, including the final quotient `to(tl.int8)` cast surface. -/
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

def outOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destIndex s DestLoc * stride_o_bs + stride_o_h * headIndex s idx.1 +
    stride_o_d * dimIndex s idx.2.1

def scaleActive (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) : Prop :=
  i.val < head_num

instance scaleActiveDecidable (head_num BLOCK_HEAD : Nat) (i : Fin BLOCK_HEAD) :
    Decidable (scaleActive head_num BLOCK_HEAD i) := by
  unfold scaleActive
  infer_instance

def scaleOutOffset
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + stride_os_h * i.val

/-! ## Genuine value/scale specs for the full real surface

The specs below are computed entirely from the kernel **inputs** (`K`,
`DestLoc`, the strides, `head_num`, `head_dim`); they do not reference the
executed state, so the resulting `..._real_surface_..._compute_correct` theorems
are not self-referential. The combined head-and-dim load mask
(`(offs_h < head_num) & (offs_d < head_dim)`, `other=0.0`) and the per-head
`max(|src|, axis=1) / 127` reduction are modeled exactly. Unlike
`quantize_copy_kv`, the scale cast `.to(OutScale.dtype.element_ty)` lowers as the
identity over `ℝ` (the DSL erases `element_ty` casts), so the scale store writes
the real channel and the genuine scale spec is the real value of
`max(|src|, axis=1) / 127`. The quotient `(src / scale).to(tl.int8)` writes the
int channel; the int8 saturation is the trusted fixed-width cast surface. -/

/-- The masked source lane value (`tl.load(..., other=0.0)`): the head- and
dim-masked `K[cur_index, h, d]` as a `WithBot ℝ`. -/
noncomputable def maskedSrc
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim : Nat)
    (h d : Nat) : WithBot ℝ :=
  if h < head_num ∧ d < head_dim then
    some (s.readMem K (s.pids 0 * stride_k_bs + h * stride_k_h + stride_k_d * d))
  else some (0.0 : ℝ)

/-- The per-head `data_scale` *value* (`max(|src|, axis=1) / 127`): the row
reduce-max of `|maskedSrc|` divided by `127`, as a real value. The
`.to(OutScale.dtype.element_ty)` cast in the surface is the identity over `ℝ`. -/
noncomputable def quantizeKvTransformScaleValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : WithBot ℝ :=
  Option.map (· / 127.0)
    ((Finset.univ.sup'
        (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
        (fun x : Fin BLOCK_DMODEL =>
          if maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim h x.val
              < (some 0 : WithBot ℝ) then
            NumericDType.real.sub (some 0)
              (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim h x.val)
          else maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim h x.val) :
      WithBot ℝ))

/-- Genuine quantized value spec (`(src / data_scale).to(int8)`): the int8 cast
of the masked source lane divided by the per-head scale. -/
noncomputable def quantizeKvTransformSurfaceIntValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Int :=
  WithBot.realToInt8
    (FloatDType.real.cast FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num head_dim
          idx.1.val idx.2.1.val)
        (quantizeKvTransformScaleValue s K stride_k_bs stride_k_h stride_k_d
          head_num head_dim BLOCK_DMODEL hD idx.1.val)))

/-- Genuine scale store cell (`data_scale`): the real value the kernel writes to
`Out_scale`, observed through `readMem`. -/
noncomputable def quantizeKvTransformScaleCell
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : ℝ :=
  WithBot.unbotD 0
    (quantizeKvTransformScaleValue s K stride_k_bs stride_k_h stride_k_d
      head_num head_dim BLOCK_DMODEL hD h)

/-- **Genuine value-output correctness for the full real surface.** The
destination-indexed int8 value store realizes `quantizeKvTransformSurfaceIntValue`
(read back via `readMemValue .int Out`). Not self-referential: the expected
value is computed from the inputs. -/
theorem destindex_copy_quantize_kv_transform_real_surface_value_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      let outAddr := outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx
      (exec (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL
            BLOCK_HEAD) s).map (·.readMemValue .int Out outAddr)
        = some (if active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeKvTransformSurfaceIntValue s K stride_k_bs stride_k_h stride_k_d
              head_num head_dim BLOCK_DMODEL hD idx
          else s.readMemValue .int Out outAddr) := by
  intro idx
  simp only [outOffset, destIndex, headIndex, dimIndex]
  simp [exec, destindex_copy_quantize_kv_transform_real_surface, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, dif_pos hD]
  rw [foldl_writeMem_prop_masked_readMemValue_int_other OutScale _ _
    (fun i : TileIndex [BLOCK_HEAD, 1] => i.1.val < head_num) _ _ Out _ hOut]
  rw [scatter_readback_int_prop_masked_nd (region := Out) (shape := [BLOCK_HEAD, BLOCK_DMODEL])
    _ (fun i => s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
        stride_o_h * i.1.val + stride_o_d * i.2.1.val)
    _ (fun i => i.1.val < head_num ∧ i.2.1.val < head_dim) ?_ idx]
  · simp only [active, headIndex, dimIndex]
    by_cases h : idx.1.val < head_num ∧ idx.2.1.val < head_dim
    · rw [if_pos h, if_pos h]
      simp only [quantizeKvTransformSurfaceIntValue, quantizeKvTransformScaleValue,
        maskedSrc, if_pos h, if_pos h.1, Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      simp only [BlockState.setReg_readMemValue]
  · simpa [outOffset, destIndex, headIndex, dimIndex, BlockState.readMemValue] using hOutInj

/-- **Genuine scale-output correctness for the full real surface.** The
destination-indexed scale store realizes `quantizeKvTransformScaleCell` (read
back via `readMem OutScale`). Not self-referential. -/
theorem destindex_copy_quantize_kv_transform_real_surface_scale_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : OutScale ≠ Out)
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD =>
        scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    ∀ i : Fin BLOCK_HEAD,
      let outAddr := scaleOutOffset s DestLoc stride_os_bs stride_os_h i
      (exec (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL
            BLOCK_HEAD) s).map (·.readMem OutScale outAddr)
        = some (if scaleActive head_num BLOCK_HEAD i then
            quantizeKvTransformScaleCell s K stride_k_bs stride_k_h stride_k_d
              head_num head_dim BLOCK_DMODEL hD i.val
          else s.readMem OutScale outAddr) := by
  intro i
  simp only [scaleOutOffset, destIndex]
  simp [exec, destindex_copy_quantize_kv_transform_real_surface, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, dif_pos hD]
  rw [BlockState.scatter_readback_prop_masked_nd (region := OutScale) (shape := [BLOCK_HEAD, 1])
    _ (fun j => s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs + stride_os_h * j.1.val)
    _ (fun j => j.1.val < head_num) ?_ (i, ⟨0, by omega⟩, PUnit.unit)]
  · simp only [scaleActive]
    show (if i.val < head_num then _ else _) = (if i.val < head_num then _ else _)
    by_cases h : i.val < head_num
    · rw [if_pos h, if_pos h]
      simp only [quantizeKvTransformScaleCell, quantizeKvTransformScaleValue, maskedSrc,
        if_pos h, Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      rw [foldl_writeMemTyped_int_prop_masked_readMem_other (region := Out)
        (P := fun k : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          k.1.val < head_num ∧ k.2.1.val < head_dim)
        (hRR := hOut)]
      simp only [BlockState.setReg_readMem]
  · intro a b hab
    have h1 : a.1 = b.1 := by
      apply hScaleInj
      simpa [scaleOutOffset, destIndex, BlockState.readMemValue] using hab
    obtain ⟨a1, a2, a3⟩ := a
    obtain ⟨b1, b2, b3⟩ := b
    simp only at h1
    subst h1
    have : a2 = b2 := Subsingleton.elim _ _
    subst this
    rfl

/-- **Dimension-general output summary (`ComputeCorrect.Realizes_without_Rounding` form).** For
arbitrary strides / `head_num` / `head_dim` / `BLOCK_DMODEL` / `BLOCK_HEAD` (and
any program ids in `s`), the destindex quantize-KV-transform surface lowers
(`∃ alg, … = Except.ok alg`), and its two stored outputs each `Realizes_without_Rounding` a
genuine input-memory closed form:

* the int8 value store to `Out` realizes the genuine per-cell int value
  `quantizeKvTransformSurfaceIntValue` on active lanes (`active`), and preserves
  the prior `Out` contents on inactive lanes — read back via `readMemValue .int`;
* the real per-row scale store to `OutScale` realizes the genuine
  `quantizeKvTransformScaleCell` on active rows (`scaleActive`), and preserves the
  prior `OutScale` contents on inactive rows — read back via `readMem`.

Both write maps are total (`fun _ => some …`); the active/inactive split is
carried inside `expected`. Honest side conditions: offset injectivity for the
value tile (`hValInj`) and the per-head scale (`hScaleInj`), and region
distinctness `hOut`. The general version assumes full tile injectivity, which
holds whenever `BLOCK_DMODEL = head_dim` and `BLOCK_HEAD = head_num`. -/
specification destindex_copy_quantize_kv_transform_output_summary_general
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => scaleOutOffset s DestLoc stride_os_bs stride_os_h i)) :
    (∃ alg, (destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        some (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
      (expected := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        (if active s head_num head_dim BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeKvTransformSurfaceIntValue s K stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL hD idx
          else s.readMemValue .int Out (outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_HEAD =>
        some (OutScale, scaleOutOffset s DestLoc stride_os_bs stride_os_h i))
      (expected := fun i : Fin BLOCK_HEAD =>
        (if scaleActive head_num BLOCK_HEAD i then
            quantizeKvTransformScaleCell s K stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL hD i.val
          else s.readMem OutScale (scaleOutOffset s DestLoc stride_os_bs stride_os_h i))) := by
  refine ⟨destindex_copy_quantize_kv_transform_real_surface_toAlgorithm_supported K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD, ?_, ?_⟩
  · -- value store to `Out` realizes `quantizeKvTransformSurfaceIntValue`
    unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [destindex_copy_quantize_kv_transform_real_surface,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro idx
    simp only [ComputeCorrect.OutputReadable.read_int]
    have h := destindex_copy_quantize_kv_transform_real_surface_value_output_compute_correct
      K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD s hD hOut hValInj idx
    rw [hExec] at h
    simpa using Option.some.inj h
  · -- scale store to `OutScale` realizes `quantizeKvTransformScaleCell`
    unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [destindex_copy_quantize_kv_transform_real_surface,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro i
    simp only [ComputeCorrect.OutputReadable.read_real]
    have h := destindex_copy_quantize_kv_transform_real_surface_scale_output_compute_correct
      K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD s hD (fun h => hOut h.symm) hScaleInj i
    rw [hExec] at h
    simpa using Option.some.inj h

/-! ## The `⊨` (KernelIO) surface

The two `..._compute_correct` engine lemmas above prove the two stored outputs'
readbacks in the raw exec form. This section re-packages the **full faithful
surface** on the typed two-output metadata combinator
`MetaMasked2DKernelIO₁ₓ₂ ⊨ (value, scale)`: the `.int` quantized tile via
`oty1 := .int` (read back through `readMemValue .int`) and the real
`max |·| / 127` scale column via `oty2 := .float` (the
`.to(Out_scale.dtype.element_ty)` cast is the ℝ identity, so `readMem` is the
faithful readback). Both spec halves are expressed over the **loaded** tile
`xs` (the `⊨` combinator's input view), so the headline is not
self-referential. The masked `K` load carries `other=0.0`, so masked-off lanes
default to `0` and the reduce-max is a genuine function of the loaded inputs —
plain `Implements.intro` suffices (no `intro_undef`). -/

namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform

open scoped VeriTile.Triton.MetaMasked2DKernelIO₁ₓ₂

/-- `.int` masked-store `foldl` cell frame: a masked `writeMemTyped .int` fold
leaves every cell it does not actively hit unchanged. -/
private theorem foldl_writeMemTyped_int_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → Int) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP, ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))]
        show (if r = region ∧ o = offsetFn hd then _ else s.mem r o) = s.mem r o
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- A masked real `writeMem` store `foldl` cell frame. -/
private theorem foldl_writeMem_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP, ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-! ### Pure `xs`-view specs

The two output halves as functions of the **loaded** tile `xs : Fin (BLOCK_HEAD *
BLOCK_DMODEL) → ℝ` (lane `j` is tile cell `(j / BLOCK_DMODEL, j % BLOCK_DMODEL)`
via the shared `Lane2D` bridge). Masked-off lanes take the `other=0.0` default. -/

/-- The masked source lane over the loaded tile. -/
noncomputable def srcXs (BLOCK_HEAD BLOCK_DMODEL head_num head_dim : Nat)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (h : Fin BLOCK_HEAD) (d : Fin BLOCK_DMODEL) : WithBot ℝ :=
  if h.val < head_num ∧ d.val < head_dim then
    some (xs (Lane2D.encode (h, d, PUnit.unit)))
  else some (0.0 : ℝ)

/-- Per-head scale value over `xs` (`max |·| / 127`), matching
`quantizeKvTransformScaleValue` with `maskedSrc` replaced by `srcXs`. -/
noncomputable def scaleValXs (BLOCK_HEAD BLOCK_DMODEL head_num head_dim : Nat)
    (hD : 0 < BLOCK_DMODEL) (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (h : Fin BLOCK_HEAD) : WithBot ℝ :=
  Option.map (· / 127.0)
    ((Finset.univ.sup'
        (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
        (fun d : Fin BLOCK_DMODEL =>
          if srcXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim xs h d < (some 0 : WithBot ℝ) then
            NumericDType.real.sub (some 0)
              (srcXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim xs h d)
          else srcXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim xs h d) :
      WithBot ℝ))

/-- The `.int` value at lane `j` over `xs`. -/
noncomputable def valueXs (BLOCK_HEAD BLOCK_DMODEL head_num head_dim : Nat)
    (hD : 0 < BLOCK_DMODEL) (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (j : Fin (BLOCK_HEAD * BLOCK_DMODEL)) : Int :=
  WithBot.realToInt8
    (FloatDType.real.cast FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (srcXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim xs
          (Lane2D.decode j).1 (Lane2D.decode j).2.1)
        (scaleValXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim hD xs (Lane2D.decode j).1)))

/-- The real scale cell at head `h` over `xs`. -/
noncomputable def scaleCellXs (BLOCK_HEAD BLOCK_DMODEL head_num head_dim : Nat)
    (hD : 0 < BLOCK_DMODEL) (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (h : Fin BLOCK_HEAD) : ℝ :=
  WithBot.unbotD 0 (scaleValXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim hD xs h)

/-! ### The metadata IO signature -/

/-- The full quantize-KV-transform surface's typed two-output metadata **IO
signature**: the `.nat` dest-index slot `DestLoc[pid₀]`, the masked `K` data
tile, the `.int` quantized tile `out1 = Out` (`oty1 := .int`), and the real
scale column `out2 = OutScale` (`oty2 := .float`). Lane `j` of the data/`out1`
tile is `(j / BLOCK_DMODEL, j % BLOCK_DMODEL)`; `out2` has one column per head. -/
def quantizeKvTransformIO
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    MetaMasked2DKernelIO₁ₓ₂ where
  kernel := destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD
  mbuf1 := DestLoc
  inp := K
  out1 := Out
  out2 := OutScale
  B := BLOCK_HEAD * BLOCK_DMODEL
  C := BLOCK_HEAD
  oty1 := .int
  oty2 := .float
  mwin1 := fun pid₀ _ => pid₀
  read := fun pid₀ _ _ j =>
    pid₀ * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
      stride_k_d * (j.val % BLOCK_DMODEL)
  write1 := fun _ _ m1 j =>
    m1 * stride_o_bs + stride_o_h * (j.val / BLOCK_DMODEL) +
      stride_o_d * (j.val % BLOCK_DMODEL)
  write2 := fun _ _ m1 i => m1 * stride_os_bs + stride_os_h * i.val
  mask := fun _ _ _ j =>
    j.val / BLOCK_DMODEL < head_num ∧ j.val % BLOCK_DMODEL < head_dim
  writeMask2 := fun _ _ _ i => i.val < head_num

/-! ### Bridging the `maskedSrc`-view specs to the `xs`-view specs -/

/-- Under the loaded-tile pin `hx`, the per-head scale value matches its
`xs`-view. -/
private theorem scaleValXs_eq
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s₀ : BlockState) (hD : 0 < BLOCK_DMODEL)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (hx : ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL),
      (j.val / BLOCK_DMODEL < head_num ∧ j.val % BLOCK_DMODEL < head_dim) →
      s₀.readMem K (s₀.pids 0 * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
        stride_k_d * (j.val % BLOCK_DMODEL)) = xs j)
    (h : Fin BLOCK_HEAD) :
    quantizeKvTransformScaleValue s₀ K stride_k_bs stride_k_h stride_k_d
        head_num head_dim BLOCK_DMODEL hD h.val
      = scaleValXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim hD xs h := by
  have hsrc : ∀ d : Fin BLOCK_DMODEL,
      maskedSrc s₀ K stride_k_bs stride_k_h stride_k_d head_num head_dim h.val d.val
        = srcXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim xs h d := by
    intro d
    unfold maskedSrc srcXs
    by_cases hact : h.val < head_num ∧ d.val < head_dim
    · rw [if_pos hact, if_pos hact]
      have hj := hx (Lane2D.encode (h, d, PUnit.unit))
        (by rw [Lane2D.encode_div, Lane2D.encode_mod]; exact hact)
      rw [Lane2D.encode_div, Lane2D.encode_mod] at hj
      rw [hj]
    · rw [if_neg hact, if_neg hact]
  unfold quantizeKvTransformScaleValue scaleValXs
  simp only [hsrc]

/-- Under `hx`, the `.int` value at active lane `j` matches its `xs`-view. -/
private theorem valueXs_eq
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s₀ : BlockState) (hD : 0 < BLOCK_DMODEL)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (hx : ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL),
      (j.val / BLOCK_DMODEL < head_num ∧ j.val % BLOCK_DMODEL < head_dim) →
      s₀.readMem K (s₀.pids 0 * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
        stride_k_d * (j.val % BLOCK_DMODEL)) = xs j)
    (j : Fin (BLOCK_HEAD * BLOCK_DMODEL)) :
    quantizeKvTransformSurfaceIntValue s₀ K stride_k_bs stride_k_h stride_k_d
        head_num head_dim BLOCK_DMODEL hD (Lane2D.decode j)
      = valueXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim hD xs j := by
  unfold quantizeKvTransformSurfaceIntValue valueXs
  rw [scaleValXs_eq K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD s₀ hD xs hx (Lane2D.decode j).1]
  have hsrc :
      maskedSrc s₀ K stride_k_bs stride_k_h stride_k_d head_num head_dim
          (Lane2D.decode j).1.val (Lane2D.decode j).2.1.val
        = srcXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim xs
            (Lane2D.decode j).1 (Lane2D.decode j).2.1 := by
    unfold maskedSrc srcXs
    by_cases hact : (Lane2D.decode j).1.val < head_num ∧ (Lane2D.decode j).2.1.val < head_dim
    · rw [if_pos hact, if_pos hact]
      have hkey : Lane2D.encode
          (((Lane2D.decode j).1, (Lane2D.decode j).2.1, PUnit.unit)
            : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) = j := by
        have h : (((Lane2D.decode j).1, (Lane2D.decode j).2.1, PUnit.unit)
            : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) = Lane2D.decode j := rfl
        rw [h, Lane2D.encode_decode]
      simp only [Lane2D.decode_row, Lane2D.decode_col]
      rw [hkey]
      have hj := hx j (by
        rw [Lane2D.decode_row, Lane2D.decode_col] at hact; exact hact)
      rw [hj]
    · rw [if_neg hact, if_neg hact]
  rw [hsrc]

/-! ### Termination, flat-bridge membership, safety, frame -/

/-- Termination: the surface executes to completion from any state. -/
private theorem quantize_kv_transform_exec_isSome
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) (hD : 0 < BLOCK_DMODEL)
    (s : BlockState) :
    ∃ s1, exec ((destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel) s = some s1 := by
  simp [exec, destindex_copy_quantize_kv_transform_real_surface,
        ComputeKernel.toAlgKernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop, Tile.expandDim,
        Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, dif_pos hD]

/-- The surface sits inside the flat-memory bridge's covered fragment. -/
theorem quantize_kv_transform_flattenOk
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ((destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [destindex_copy_quantize_kv_transform_real_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk: the scalar `.nat` `DestLoc` load, the masked `K`
load, and the two masked stores reduce to the cell/lane-wise bounds. -/
theorem quantize_kv_transform_traceSafe
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) (hD : 0 < BLOCK_DMODEL)
    (bounds : RegionBounds) (s : BlockState)
    (hd : s.pids 0 < bounds DestLoc)
    (hk : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      idx.1.val < head_num → idx.2.1.val < head_dim →
      s.pids 0 * stride_k_bs + idx.1.val * stride_k_h + stride_k_d * idx.2.1.val
        < bounds K)
    (ho : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      idx.1.val < head_num → idx.2.1.val < head_dim →
      s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
        stride_o_h * idx.1.val + stride_o_d * idx.2.1.val < bounds Out)
    (hos : ∀ i : Fin BLOCK_HEAD, i.val < head_num →
      s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs + stride_os_h * i.val
        < bounds OutScale) :
    Kernel.TraceSafe bounds
      ((destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [destindex_copy_quantize_kv_transform_real_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.remap, Tile.reduceMax,
    Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex,
    Option.bind, Option.map, TileShape.insertAxis, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, NumericDType.div, ComparableDType.lt,
    dif_pos hD]
  refine ⟨hd, fun a a_1 h1 h2 => hk (a, a_1, PUnit.unit) h1 h2,
    fun a a_1 h1 h2 => ho (a, a_1, PUnit.unit) h1 h2, fun a h1 => hos a h1⟩

/-- Frame: every cell off both written windows (the active `Out` value cells and
the active `OutScale` scale cells) is preserved. -/
private theorem quantize_kv_transform_frame
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) (hD : 0 < BLOCK_DMODEL)
    (s s1 : BlockState)
    (hExec : exec ((destindex_copy_quantize_kv_transform_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissVal : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      idx.1.val < head_num → idx.2.1.val < head_dim →
      ¬(Out = r ∧ s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
          stride_o_h * idx.1.val + stride_o_d * idx.2.1.val = o))
    (hmissScale : ∀ i : Fin BLOCK_HEAD, i.val < head_num →
      ¬(OutScale = r ∧ s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs +
          stride_os_h * i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, destindex_copy_quantize_kv_transform_real_surface,
        ComputeKernel.toAlgKernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop, Tile.expandDim,
        Tile.ptrAdd, Tile.remap, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, dif_pos hD] at hExec
  subst hExec
  refine Eq.trans (foldl_writeMem_preserve_cell _ _ _ r o _ _ ?_) ?_
  · intro k _ hP hc
    exact hmissScale k.1 hP hc
  · refine Eq.trans (foldl_writeMemTyped_int_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hP hc
    exact hmissVal k hP.1 hP.2 hc

/-- Under `hx`, the scale cell at head `i` matches its `xs`-view. -/
private theorem scaleCellXs_eq
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s₀ : BlockState) (hD : 0 < BLOCK_DMODEL)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (hx : ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL),
      (j.val / BLOCK_DMODEL < head_num ∧ j.val % BLOCK_DMODEL < head_dim) →
      s₀.readMem K (s₀.pids 0 * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
        stride_k_d * (j.val % BLOCK_DMODEL)) = xs j)
    (i : Fin BLOCK_HEAD) :
    quantizeKvTransformScaleCell s₀ K stride_k_bs stride_k_h stride_k_d
        head_num head_dim BLOCK_DMODEL hD i.val
      = scaleCellXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim hD xs i := by
  unfold quantizeKvTransformScaleCell scaleCellXs
  rw [scaleValXs_eq K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD s₀ hD xs hx i]

/-- **The `⊨` headline (typed two-output metadata surface).** The full faithful
`quantize_kv_transform` surface implements, on its `MetaMasked2DKernelIO₁ₓ₂`
signature, the pair `(int8 quantized tile, real max|·|/127 scale column)` over
the loaded tile `xs` and the loaded dest-index slot `m1`.

Honest side conditions, each required for truth: `hD : 0 < BLOCK_DMODEL` (the
`tl.max(·, axis=1)` reduce axis must be nonempty — this is also what makes the
kernel terminate), `hOut : Out ≠ OutScale` (the two stores must not alias), and
the value-tile / scale-column destination-offset injectivity `hValInj` /
`hScaleInj` (a masked scatter with colliding lanes is last-writer-wins, so the
per-lane readbacks are false without them; both are `Dest_loc`-independent since
the loaded row only shifts every address by a constant). The masked `K` load
carries `other=0.0`, so masked-off lanes default to `0` and the scale is a
genuine function of the loaded inputs — plain `Implements.intro` (no
`intro_undef`). -/
specification quantize_kv_transform_io_correctness
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)
    (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        stride_o_h * idx.1.val + stride_o_d * idx.2.1.val))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => stride_os_h * i.val)) :
    quantizeKvTransformIO K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD
      ⊨ fun _ _ _ xs =>
          (fun j => valueXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim hD xs j,
           fun i => scaleCellXs BLOCK_HEAD BLOCK_DMODEL head_num head_dim hD xs i) := by
  refine MetaMasked2DKernelIO₁ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact quantize_kv_transform_flattenOk K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD
  · -- safety
    intro bounds s m1 hm1 hb1 hbr hbw1 hbw2
    simp only [quantizeKvTransformIO] at hm1 hb1 hbr hbw1 hbw2
    refine quantize_kv_transform_traceSafe K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD
      hD bounds s hb1 (fun idx h1 h2 => ?_) (fun idx h1 h2 => ?_) (fun i h1 => ?_)
    · simpa using hbr (Lane2D.encode idx) (by simpa using ⟨h1, h2⟩)
    · rw [hm1]
      simpa using hbw1 (Lane2D.encode idx) (by simpa using ⟨h1, h2⟩)
    · rw [hm1]
      simpa using hbw2 i (by simpa using h1)
  · -- run
    intro s₀ m1 xs hm1 hx
    simp only [quantizeKvTransformIO] at hm1 hx ⊢
    have hOutInj : Function.Injective
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          outOffset s₀ DestLoc stride_o_bs stride_o_h stride_o_d idx) := by
      intro a b hab
      apply hValInj
      simp only [outOffset, headIndex, dimIndex] at hab
      rw [Nat.add_assoc, Nat.add_assoc] at hab
      exact Nat.add_left_cancel hab
    have hScInj : Function.Injective
        (fun i : Fin BLOCK_HEAD => scaleOutOffset s₀ DestLoc stride_os_bs stride_os_h i) := by
      intro a b hab
      apply hScaleInj
      simp only [scaleOutOffset] at hab
      exact Nat.add_left_cancel hab
    obtain ⟨s1, hs1⟩ := quantize_kv_transform_exec_isSome K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD hD s₀
    refine ⟨s1, hs1, ?_, ?_, ?_⟩
    · -- value output
      intro j hmask
      have hv := destindex_copy_quantize_kv_transform_real_surface_value_output_compute_correct
        K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD s₀
        hD hOut hOutInj (Lane2D.decode j)
      have haddr : outOffset s₀ DestLoc stride_o_bs stride_o_h stride_o_d (Lane2D.decode j)
          = m1 * stride_o_bs + stride_o_h * (j.val / BLOCK_DMODEL) +
            stride_o_d * (j.val % BLOCK_DMODEL) := by
        simp only [outOffset, destIndex, headIndex, dimIndex, Lane2D.decode_row,
          Lane2D.decode_col, hm1]
      have hact : active s₀ head_num head_dim BLOCK_HEAD BLOCK_DMODEL (Lane2D.decode j) := by
        simp only [active, headIndex, dimIndex, Lane2D.decode_row, Lane2D.decode_col]
        exact hmask
      rw [hs1, haddr] at hv
      simp only [if_pos hact] at hv
      have hkey : s1.readMemValue .int Out (m1 * stride_o_bs +
          stride_o_h * (j.val / BLOCK_DMODEL) + stride_o_d * (j.val % BLOCK_DMODEL))
        = quantizeKvTransformSurfaceIntValue s₀ K stride_k_bs stride_k_h stride_k_d
            head_num head_dim BLOCK_DMODEL hD (Lane2D.decode j) := Option.some.inj hv
      show s1.readMemValue .int Out (m1 * stride_o_bs +
        stride_o_h * (j.val / BLOCK_DMODEL) + stride_o_d * (j.val % BLOCK_DMODEL)) = _
      rw [hkey,
        valueXs_eq K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d
          head_num head_dim BLOCK_DMODEL BLOCK_HEAD s₀ hD xs hx j]
    · -- scale output
      intro i hmask
      have hv := destindex_copy_quantize_kv_transform_real_surface_scale_output_compute_correct
        K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD s₀
        hD (fun h => hOut h.symm) hScInj i
      have haddr : scaleOutOffset s₀ DestLoc stride_os_bs stride_os_h i
          = m1 * stride_os_bs + stride_os_h * i.val := by
        simp only [scaleOutOffset, destIndex, hm1]
      have hact : scaleActive head_num BLOCK_HEAD i := hmask
      rw [hs1, haddr] at hv
      simp only [if_pos hact] at hv
      have hkey : s1.readMem OutScale (m1 * stride_os_bs + stride_os_h * i.val)
        = quantizeKvTransformScaleCell s₀ K stride_k_bs stride_k_h stride_k_d
            head_num head_dim BLOCK_DMODEL hD i.val := Option.some.inj hv
      show s1.readMem OutScale (m1 * stride_os_bs + stride_os_h * i.val) = _
      rw [hkey,
        scaleCellXs_eq K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d
          head_num head_dim BLOCK_DMODEL BLOCK_HEAD s₀ hD xs hx i]
    · -- frame
      intro r o hc1 hc2
      refine quantize_kv_transform_frame K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num head_dim BLOCK_DMODEL BLOCK_HEAD
        hD s₀ s1 hs1 r o (fun idx h1 h2 ⟨hr, ho⟩ => ?_) (fun i h1 ⟨hr, ho⟩ => ?_)
      · rcases hc1 with hne | hno
        · exact hne hr.symm
        · refine hno (Lane2D.encode idx) (by simpa using ⟨h1, h2⟩) ?_
          simp only [Lane2D.encode_div, Lane2D.encode_mod]
          rw [← ho]
          simp only [destIndex, hm1]
      · rcases hc2 with hne | hno
        · exact hne hr.symm
        · refine hno i h1 ?_
          rw [← ho]
          simp only [destIndex, hm1]

end VeriTile.Bench.TritonBenchG.QuantizeKvTransform
