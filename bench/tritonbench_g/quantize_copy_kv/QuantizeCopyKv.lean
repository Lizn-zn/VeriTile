import VeriTile.Triton
import VeriTile.Examples.AttentionForwardClosedForm

/-!
# `quantize_copy_kv` — strict per-kernel correctness

`_fwd_kernel_destindex_copy_quantize_kv` copies and int8-quantizes a KV cache
entry to a destination-indexed slot: program `cur_index` loads `dest_index =
Dest_loc[cur_index]`, loads the `[BLOCK_HEAD, BLOCK_DMODEL]` source tile (head-
masked by `offs_h < head_num`, `other=0.0`), computes a per-head scale
`max(|src|, axis=1) / 127` cast to fp16, divides and casts the quotient to int8,
and stores both the quantized values (at `Out[dest_index]`) and the per-head
scale (at `Out_scale[dest_index]`), each head-masked.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_destindex_copy_quantize_kv[grid](...)`, the
grid `(seq_len,)`, `BLOCK_HEAD = next_power_of_2(head_num)`, the strides, and the
runtime composition of per-program destination-indexed writes) is the *trusted
boundary*. `cur_index` is universally quantified, so the per-program statement
covers every program of the grid. Destination-index injectivity (no two programs
write the same slot) is supplied as a hypothesis / no-collision lemma, not
assumed of the host.

## Proof architecture

```
destindex_copy_quantize_kv_output_summary_general             ← TOP THEOREM
  ├─ ..._real_surface_toAlgorithm_supported                    surface lowers to the algorithm layer
  ├─ ..._real_surface_value_output_compute_correct             genuine int8 value readback
  │    ├─ scatter_readback_int_prop_masked_nd                  per-lane int store readback
  │    ├─ foldl_writeMemTyped_fp16_prop_masked_readMemValue_int_other   peel trailing fp16 store
  │    └─ hValInj                                              no-collision offset-injectivity hyp
  └─ ..._real_surface_scale_output_compute_correct             genuine fp16 scale readback
       ├─ scatter_readback_fp16_prop_masked_nd                 per-head fp16 store readback
       ├─ foldl_writeMemTyped_int_prop_masked_readMemValue_fp16_other   peel prior int store
       └─ hScaleInj                                            no-collision offset-injectivity hyp
```

The genuine value spec (`quantizeCopyKvSurfaceIntValue`) is the int8 cast of
`src / fp16(max(|src|, axis=1)/127)` read back through `readMemValue .int Out`;
the genuine scale spec (`quantizeCopyKvScaleCell`) is the stored fp16 cell of
`max(|src|, axis=1)/127` read back through `readMemValue .fp16 OutScale`. Both
are computed from the kernel inputs, so the top summary is not self-referential.
The top summary is dimension-general (arbitrary strides / `head_num` /
`BLOCK_DMODEL` / `BLOCK_HEAD`); its int8 value conjunct is stated as
`ComputeCorrect.Realizes_without_Rounding`, while the fp16 scale conjunct stays in raw exec-
readback form because `TileCarrier .fp16` has no `OutputReadable` carrier (see
the summary's docstring for the honest note).

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The honesty point is the
quantization tail. The proofs model: the per-head scale `max(|src|) / 127`, the
masked `other=0.0` load default, the destination-indexed addressing, and the
real-valued scaled value `src / scale`; the value-store slices take the per-head
scale as a precomputed `OutScale` input and prove the masked writeback up to the
fixed-width cast. The full surface (including the scale's `.to(tl.float16)` and
the quotient's `(src / scale).to(tl.int8)` annotations) lowers through algorithm
erasure, where those casts are the **identity over `ℝ`**. The bit-accurate effect
of the fp16 scale rounding and the int8 saturating cast is **not** numerically
modeled. `@triton.autotune` is not present here.

## The `⊨[R]` (rounding KernelIO) surface — the fp16 scale, resolved

The two LightLLM siblings `quantize_kv_copy` and `quantize_kv_transform` migrate
onto the exact typed two-output combinator
`MetaMasked2DKernelIO₁ₓ₂ ⊨ (int8 tile, real scale)`. This kernel's int8 value
tile IS equally expressible there (`oty1 := .int`, read back through
`readMemValue .int`) — but its **scale store is `.to(tl.float16)`**, which lands
on the **rounding** axis, *not* the `ChanTy` axis (`float | bool | nat | int`)
that `oty2` ranges over: a reduced-precision float cell reads back through
`readMemAs .fp16` at a `TileCarrier .fp16`, which is not a `ChanTy` carrier. So
the exact `⊨` cannot type it.

The **rounding** relation `MetaMasked2DKernelIO₁ₓ₂.ImplementsR` (`⊨[R]`, added
in `VeriTile.Triton.Memory.KernelSpec`) closes exactly that gap and is this
kernel's genuine headline (`quantize_copy_kv_io_correctness` below). `out1`
keeps its `.int` `ChanTy` readback (the quantized tile); `out2` is an fp-typed
rounding channel at the declared grid `out2DType := .fp16`, read back through
`readMemAs .fp16` as the typed cell `fp16.ofReal (R.round .fp16 (max|·|/127))`.
Under `execR R` the scale is rounded at the `.to(tl.float16)` cast (site 1) and
again at the fp16-typed store (site 2), and `R.round_idem` collapses the pair to
one boundary round — the honest crux (`quantizeCopyKvScaleCellR_eq_round`). The
int8 value tile divides `src` by the **already-fp16-rounded** scale, stated
exactly (the int store is exact even under `execR R`; the `R`-dependence lives
honestly inside the divisor). Nothing is faked, and the exact
`destindex_copy_quantize_kv_output_summary_general` (the `.triv`-shadow
`Realizes_without_Rounding` value + raw fp16-cell scale) is retained unchanged
below.
-/

namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! ## Integer / fp16 memory infra

The genuine value store writes an `.int` channel (the `.to(tl.int8)` quotient)
and the scale store writes an `.fp16` channel. The following lemmas are the
`.int`/`.fp16` analogues of the `.nat`/`.real` scatter-readback and
cross-channel preservation lemmas in `VeriTile.Triton.Semantics.State`. -/

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

/-- Prop-masked `.fp16` store foldl leaves `readMemValue .int` on a different
region unchanged. The genuine value readback reads `.int Out`; the trailing
fp16 scale store to `OutScale` is peeled off with this. -/
theorem foldl_writeMemTyped_fp16_prop_masked_readMemValue_int_other {α : Type}
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier TileDType.fp16)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (R : RegionName) (off : Nat) (hRR : R ≠ region) :
    BlockState.readMemValue ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc) s))
      .int R off
      = s.readMemValue .int R off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hhd : P hd
      · simp only [hhd, if_true]
        rw [ih]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
          BlockState.readMemValue BlockState.readMemTyped
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hhd, if_false]
        exact ih _

/-- `.int` store foldl leaves `readMemValue .fp16` on a different region
unchanged. The scale readback reads `.fp16 OutScale`; the prior int value store
to `Out` is peeled off with this. -/
theorem foldl_writeMemTyped_int_prop_masked_readMemValue_fp16_other {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → Int)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (R : RegionName) (off : Nat) (hRR : R ≠ region) :
    BlockState.readMemValue ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      .fp16 R off
      = s.readMemValue .fp16 R off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · simp only [hP, if_true]
        rw [ih]
        unfold BlockState.writeMemTyped BlockState.readMemValue BlockState.readMemAs
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hP, if_false]
        exact ih _

/-- `.fp16` masked-foldl preservation (Prop mask): writes whose active offsets
all miss `o` leave `readMemValue .fp16` unchanged. -/
theorem foldl_writeMemTyped_fp16_prop_masked_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier .fp16)
    (P : α → Prop) [DecidablePred P] (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, P k → offsetFn k ≠ o) →
      BlockState.readMemValue ((l.foldl
          (fun acc k =>
            if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc) s))
          .fp16 region o
      = s.readMemValue .fp16 region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s hpre
    rw [List.foldl_cons]
    have htl : ∀ k ∈ tl, P k → offsetFn k ≠ o :=
      fun k hk => hpre k (List.mem_cons_of_mem hd hk)
    by_cases hP : P hd
    · have hhd : offsetFn hd ≠ o := hpre hd List.mem_cons_self hP
      simp only [hP, if_true]
      rw [ih _ htl]
      unfold BlockState.writeMemTyped BlockState.writeMemAs
        BlockState.readMemValue BlockState.readMemAs
      simp only
      rw [if_neg]
      rintro ⟨_, h_eq⟩
      exact hhd h_eq.symm
    · simp only [hP, if_false]
      exact ih _ htl

/-- `.fp16` scatter readback: reading `readMemValue .fp16` of the masked
`writeMemTyped .fp16` foldl at lane `i` returns the fp16 round-tripped stored
value if `P i`, else the prior value. -/
theorem scatter_readback_fp16_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier .fp16) (P : TileIndex shape → Prop)
    [DecidablePred P] (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    BlockState.readMemValue ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc) s)
      .fp16 region (offsetFn i)
    = if P i then FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i))
      else s.readMemValue .fp16 region (offsetFn i) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [show TileShape.allIndices shape = l₁ ++ i :: l₂ from hl, List.foldl_append,
    List.foldl_cons]
  have h_l1 : ∀ k ∈ l₁, P k → offsetFn k ≠ offsetFn i := by
    intro k hk _ heq; have := h_inj heq; rw [this] at hk
    exact (hl1_disj i hk i List.mem_cons_self) rfl
  have h_l2 : ∀ k ∈ l₂, P k → offsetFn k ≠ offsetFn i := by
    intro k hk _ heq; have := h_inj heq; subst this; exact hi_notin_l2 hk
  rw [foldl_writeMemTyped_fp16_prop_masked_preserves offsetFn valueFn P (offsetFn i) l₂ _ h_l2]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
      BlockState.readMemValue BlockState.readMemAs
    simp only [if_pos (And.intro rfl rfl)]
    rfl
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_fp16_prop_masked_preserves offsetFn valueFn P (offsetFn i) l₁ _ h_l1]

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

def scaleOutOffset1
    (s : BlockState) (DestLoc : RegionName)
    (stride_os_bs stride_os_h : Nat) (i : Fin BLOCK_HEAD) : Nat :=
  destIndex s DestLoc * stride_os_bs + i.val * stride_os_h

/-! ## Genuine value/scale specs for the full real surface

The two specs below are computed entirely from the kernel **inputs** (`K`,
`DestLoc`, the strides, `head_num`); they do not reference the executed state, so
the resulting `..._real_surface_..._compute_correct` theorems are not
self-referential. The value spec is the int8 cast of `src / fp16(scale)`; the
scale spec is the fp16 cell of `max(|src|, axis=1) / 127`. Both `max`/`abs` and
the masked `other=0.0` load are modeled exactly; the fp16 rounding and int8
saturation are the trusted fixed-width cast surfaces (see the modeling boundary
note above). -/

/-- The masked source lane value (`tl.load(..., other=0.0)`): the head-masked
`K[cur_index, h, d]` as a `WithBot ℝ`. -/
noncomputable def maskedSrc
    (s : BlockState) (K : RegionName) (stride_k_bs stride_k_h stride_k_d head_num : Nat)
    (h d : Nat) : WithBot ℝ :=
  if h < head_num then
    some (s.readMem K (s.pids 0 * stride_k_bs + h * stride_k_h + stride_k_d * d))
  else some (0.0 : ℝ)

/-- The per-head `data_scale` *value* (`(max(|src|, axis=1) / 127).to(fp16)`):
the fp16 cast of the row reduce-max of `|maskedSrc|` divided by `127`. This is
the value computed by the kernel before the fp16 store rounding. -/
noncomputable def quantizeCopyKvScaleValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : TileCarrier .fp16 :=
  FloatDType.real.cast FloatDType.fp16
    (Option.map (· / 127.0)
      ((Finset.univ.sup'
          (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
          (fun x : Fin BLOCK_DMODEL =>
            if maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val
                < (some 0 : WithBot ℝ) then
              NumericDType.real.sub (some 0)
                (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val)
            else maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val) :
        WithBot ℝ)))

/-- Genuine quantized value spec (`(src / data_scale).to(int8)`): the int8 cast
of the masked source lane divided by the fp16-cast per-head scale. -/
noncomputable def quantizeCopyKvSurfaceIntValue
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Int :=
  WithBot.realToInt8
    (FloatDType.real.cast FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num idx.1.val idx.2.1.val)
        (FloatDType.fp16.cast FloatDType.real
          (quantizeCopyKvScaleValue s K stride_k_bs stride_k_h stride_k_d head_num
            BLOCK_DMODEL hD idx.1.val))))

/-- Genuine fp16 scale store cell (`data_scale` after the fp16 store rounding):
the value the kernel writes to `Out_scale`, observed through `readMemValue
.fp16`. -/
noncomputable def quantizeCopyKvScaleCell
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : TileCarrier .fp16 :=
  FloatDType.fp16.ofReal (FloatDType.fp16.storeValue
    (quantizeCopyKvScaleValue s K stride_k_bs stride_k_h stride_k_d head_num
      BLOCK_DMODEL hD h))

/-- **Genuine value-output correctness for the full real surface.** The
destination-indexed int8 value store realizes `quantizeCopyKvSurfaceIntValue`
(read back via `readMemValue .int Out`). Not self-referential: the expected value
is computed from the inputs. -/
theorem destindex_copy_quantize_kv_real_surface_value_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      let outAddr := outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx
      (exec (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD)
          s).map (·.readMemValue .int Out outAddr)
        = some (if active s head_num BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeCopyKvSurfaceIntValue s K stride_k_bs stride_k_h stride_k_d
              head_num BLOCK_DMODEL hD idx
          else s.readMemValue .int Out outAddr) := by
  intro idx
  simp only [outOffset, destIndex, headIndex, dimIndex]
  simp [exec, destindex_copy_quantize_kv_real_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, dif_pos hD]
  rw [foldl_writeMemTyped_fp16_prop_masked_readMemValue_int_other OutScale _ _
    (fun i : TileIndex [BLOCK_HEAD, 1] => i.1.val < head_num) _ _ Out _ hOut]
  simp only [Tile.remap, Broadcast.leftIndex, Broadcast.consL, Broadcast.consSame,
    Broadcast.consR, Broadcast.nil, decide_eq_true_eq]
  rw [scatter_readback_int_prop_masked_nd (region := Out) (shape := [BLOCK_HEAD, BLOCK_DMODEL])
    _ (fun i => s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
        stride_o_h * i.1.val + stride_o_d * i.2.1.val)
    _ (fun i => i.1.val < head_num) ?_ idx]
  · simp only [active, headIndex]
    by_cases h : idx.1.val < head_num
    · rw [if_pos h, if_pos h]
      simp only [quantizeCopyKvSurfaceIntValue, quantizeCopyKvScaleValue, maskedSrc,
        if_pos h, Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      simp only [BlockState.setReg_readMemValue]
  · simpa [outOffset, destIndex, headIndex, dimIndex, BlockState.readMemValue] using hOutInj

/-- **Genuine scale-output correctness for the full real surface.** The
destination-indexed fp16 scale store realizes `quantizeCopyKvScaleCell` (read back
via `readMemValue .fp16 OutScale`). Not self-referential. -/
theorem destindex_copy_quantize_kv_real_surface_scale_output_compute_correct
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : OutScale ≠ Out)
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD =>
        scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)) :
    ∀ i : Fin BLOCK_HEAD,
      let outAddr := scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i
      (exec (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD)
          s).map (·.readMemValue .fp16 OutScale outAddr)
        = some (if scaleActive head_num BLOCK_HEAD i then
            quantizeCopyKvScaleCell s K stride_k_bs stride_k_h stride_k_d head_num
              BLOCK_DMODEL hD i.val
          else s.readMemValue .fp16 OutScale outAddr) := by
  intro i
  simp only [scaleOutOffset1, destIndex, Nat.mul_comm i.val stride_os_h]
  simp [exec, destindex_copy_quantize_kv_real_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, dif_pos hD]
  simp only [Tile.remap, Broadcast.leftIndex, Broadcast.consL, Broadcast.consSame,
    Broadcast.consR, Broadcast.nil, decide_eq_true_eq]
  rw [scatter_readback_fp16_prop_masked_nd (region := OutScale) (shape := [BLOCK_HEAD, 1])
    _ (fun j => s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs + stride_os_h * j.1.val)
    _ (fun j => j.1.val < head_num) ?_ (i, ⟨0, by omega⟩, PUnit.unit)]
  · simp only [scaleActive]
    show (if i.val < head_num then _ else _) = (if i.val < head_num then _ else _)
    by_cases h : i.val < head_num
    · rw [if_pos h, if_pos h]
      simp only [quantizeCopyKvScaleCell, quantizeCopyKvScaleValue, maskedSrc, if_pos h,
        Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      rw [foldl_writeMemTyped_int_prop_masked_readMemValue_fp16_other (region := Out)
        (P := fun k : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] => k.1.val < head_num)
        (hRR := hOut)]
      simp only [BlockState.setReg_readMemValue]
  · intro a b hab
    have h1 : a.1 = b.1 := by
      apply hScaleInj
      simpa [scaleOutOffset1, destIndex, BlockState.readMemValue] using hab
    obtain ⟨a1, a2, a3⟩ := a
    obtain ⟨b1, b2, b3⟩ := b
    simp only at h1
    subst h1
    have : a2 = b2 := Subsingleton.elim _ _
    subst this
    rfl

/-- **Dimension-general output summary.** For arbitrary strides / `head_num` /
`BLOCK_DMODEL` / `BLOCK_HEAD` (and any program ids in `s`), the destindex
quantize-KV-copy surface lowers, writes the genuine per-cell int value
`quantizeCopyKvSurfaceIntValue` to `Out` (masked by `active`), and the genuine
fp16 per-row scale `quantizeCopyKvScaleCell` to `OutScale` (masked by
`scaleActive`) — under honest offset-injectivity side conditions.

The three headline conjuncts are: (1) the surface lowers through algorithm
erasure; (2) the int8 value output stated as `ComputeCorrect.Realizes_without_Rounding` over a
total per-lane write map to `Out` (the `expected` carries the inactive-lane
`readMemValue .int Out` guarantee, so the map is total and the mask lives in
`expected`); (3) the fp16 per-row scale output.

**Honest carrier note for conjunct (3).** Unlike conjunct (2), the fp16 scale
output is NOT phrased as `ComputeCorrect.Realizes_without_Rounding`: it stays in the raw
`(exec …).map (·.readMemValue .fp16 OutScale …) = some (if …)` form. This is a
framework *carrier* limitation, not a proof gap. `ComputeCorrect.Realizes_without_Rounding`
requires an `OutputReadable` instance for the readback type, and the only
instances in `VeriTile.Triton.Correctness` are for `MemCell`, `ℝ`, `Nat`,
and `Int`. The scale reads back at `TileCarrier .fp16` (the *decoded* fp16
value), which has no `OutputReadable` carrier, so it cannot be wrapped in
`Realizes_without_Rounding`. The conjunct is nonetheless genuine and non-self-referential: it
reads INPUT memory and `quantizeCopyKvScaleCell` is computed from the kernel
inputs (see `destindex_copy_quantize_kv_real_surface_scale_output_compute_correct`).
The genuine *rounding-axis* home for this fp16 output is the `⊨[R]` headline
`quantize_copy_kv_io_correctness` (its `readMemAs .fp16` scale conjunct); this
exact summary is retained as its `.triv`-shadow.

Concrete literal-dimension instantiations of this general summary are not kept as
separate declarations. -/
specification destindex_copy_quantize_kv_output_summary_general
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)) :
    (∃ alg, (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        some (Out, outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx))
      (expected := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        (if active s head_num BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeCopyKvSurfaceIntValue s K stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL hD idx
          else s.readMemValue .int Out (outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx) : Int))) ∧
    -- Conjunct (3): raw `.map … = some (if …)` form, NOT `Realizes_without_Rounding` — see the
    -- honest carrier note in the docstring. `TileCarrier .fp16` (the decoded fp16
    -- value read back here) has no `OutputReadable` instance in the framework, so
    -- this genuine, non-self-referential output cannot be wrapped in `Realizes_without_Rounding`.
    (∀ i : Fin BLOCK_HEAD,
      (exec (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD) s).map
          (·.readMemValue .fp16 OutScale (scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i))
        = some (if scaleActive head_num BLOCK_HEAD i then
            quantizeCopyKvScaleCell s K stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL hD i.val
          else s.readMemValue .fp16 OutScale (scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i))) := by
  refine ⟨destindex_copy_quantize_kv_real_surface_toAlgorithm_supported K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD,
    ?_,
    destindex_copy_quantize_kv_real_surface_scale_output_compute_correct K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD s hD (fun h => hOut h.symm) hScaleInj⟩
  -- Middle conjunct: the int8 value output as `ComputeCorrect.Realizes_without_Rounding` over a
  -- total write map. `readMemValue .int : Int` has an `OutputReadable Int`
  -- carrier, so this converts from the raw exec-readback engine lemma.
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [destindex_copy_quantize_kv_real_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp only [ComputeCorrect.OutputReadable.read_int]
  have h := destindex_copy_quantize_kv_real_surface_value_output_compute_correct
    K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD s hD hOut hValInj idx
  simp only [outOffset, destIndex, headIndex, dimIndex] at h ⊢
  rw [hExec] at h
  simpa using Option.some.inj h

/-! ## The `⊨[R]` (rounding KernelIO) surface — resolving the fp16 scale

The exact `⊨` surface (`MetaMasked2DKernelIO₁ₓ₂ ⊨ …`) cannot type this kernel's
`.to(tl.float16)` scale store: an fp16 cell reads back at `TileCarrier .fp16`,
off the `ChanTy` value axis. The **rounding** surface
`MetaMasked2DKernelIO₁ₓ₂.ImplementsR` (`⊨[R]`) closes exactly that gap: `out1`
keeps its `.int` `ChanTy` readback, while `out2` is an fp-typed rounding
channel at `out2DType := .fp16`, read back through `readMemAs .fp16` as the
typed cell `fp16.ofReal (R.round .fp16 (scale))`.

Under `execR R` the scale's `.to(tl.float16)` cast rounds the ideal
`max(|src|)/127` at the fp16 grid (rounding site 1, `R.cast .real .fp16`); the
fp16-typed store rounds it again (site 2, `writeMemAsR R .fp16`), and
`R.round_idem` collapses the pair to one boundary round — this is what
`readMemAs .fp16 OutScale` observes. The int8 value tile divides `src` by the
**already-fp16-rounded** scale (`R.cast .real .fp16`) and casts to int8; the
int store is exact even under `execR R`, so its `readMemValue .int` readback is
that value stated exactly (its `R`-dependence lives honestly inside the divisor).
-/

/-- `.int` typed masked-foldl preserves the register file (Prop mask). Needed so
the trailing scale store's `offs_h`/`dest_index` register reads reduce past the
value store in the per-execution safety walk. -/
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

/-! ### execR memory infra: peeling a rounded fp16 store off an `.int` readback -/

/-- Prop-masked `writeMemAsR R .fp16` store foldl leaves `readMemValue .int` on a
different region unchanged (the `execR` sibling of
`foldl_writeMemTyped_fp16_prop_masked_readMemValue_int_other`). The genuine value
readback reads `.int Out`; the trailing rounded fp16 scale store to `OutScale` is
peeled off with this. -/
theorem foldl_writeMemAsR_fp16_prop_masked_readMemValue_int_other {α : Type}
    (R : RoundingModel) (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier TileDType.fp16)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (Rn : RegionName) (off : Nat) (hRR : Rn ≠ region) :
    BlockState.readMemValue ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemAsR R .fp16 region (offsetFn k) (valueFn k) else acc) s))
      .int Rn off
      = s.readMemValue .int Rn off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hhd : P hd
      · simp only [hhd, if_true]
        rw [ih]
        unfold BlockState.writeMemAsR BlockState.readMemValue BlockState.readMemTyped
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hhd, if_false]
        exact ih _

/-- Readback of a `P`-masked `writeMemAsR R .fp16` scatter through
`readMemAs .fp16` at lane `i`: the doubly-rounded stored cell if `P i`, else the
prior contents. `scatter_memcell_R_prop_masked_nd` gives the stored `MemCell`;
`readMemAs`'s own `storeValue ∘ ofReal` round-trip is the identity, so the cell's
`R.storeValue` survives verbatim. -/
theorem scatter_readMemAs_R_fp16_prop_masked_nd (R : RoundingModel)
    {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier .fp16) (P : TileIndex shape → Prop)
    [DecidablePred P] (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    BlockState.readMemAs ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemAsR R .fp16 region (offsetFn k) (valueFn k) else acc) s)
      .fp16 region (offsetFn i)
    = if P i then FloatDType.fp16.ofReal (R.storeValue FloatDType.fp16 (valueFn i))
      else s.readMemAs .fp16 region (offsetFn i) := by
  unfold BlockState.readMemAs
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .fp16 s offsetFn valueFn P h_inj i]
  by_cases hPi : P i
  · simp only [hPi, if_true, MemCell.readAs_of_same]
    simp [FloatDType.storeValue, FloatDType.ofReal]
  · simp only [hPi, if_false]

/-- An `.int` store foldl leaves `readMemAs .fp16` on a different region
unchanged — the `readMemAs` sibling of
`foldl_writeMemTyped_int_prop_masked_readMemValue_fp16_other`, used to peel the
prior int value store off the fp16 scale readback in the inactive scale row. -/
theorem foldl_writeMemTyped_int_prop_masked_readMemAs_fp16_other {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → Int)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (Rn : RegionName) (off : Nat) (hRR : Rn ≠ region) :
    BlockState.readMemAs ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      .fp16 Rn off
      = s.readMemAs .fp16 Rn off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · simp only [hP, if_true]
        rw [ih]
        unfold BlockState.writeMemTyped BlockState.readMemAs
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hP, if_false]
        exact ih _

/-! ### R-parametrized value/scale specs (the `execR R` closed forms)

Identical to the exact `quantizeCopyKv*` specs, with the fp16 scale cast on the
rounding channel: `FloatDType.real.cast FloatDType.fp16` becomes
`RoundingModel.cast R FloatDType.real FloatDType.fp16`. Every other operation
(the `abs`/`max(·,axis=1)`/`÷127` reduction, the masked `other=0.0` load, the
`÷` and int8 cast) is unchanged — `evalOpR R` rounds only `Op.castFloat`. -/

/-- The per-head `data_scale` fp16 carrier under `execR R`
(`(max(|src|, axis=1)/127).to(tl.float16)`): the fp16 **rounding** cast of the
row reduce-max of `|maskedSrc|` divided by `127`. -/
noncomputable def quantizeCopyKvScaleValueR (R : RoundingModel)
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : TileCarrier .fp16 :=
  RoundingModel.cast R FloatDType.real FloatDType.fp16
    (Option.map (· / 127.0)
      ((Finset.univ.sup'
          (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
          (fun x : Fin BLOCK_DMODEL =>
            if maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val
                < (some 0 : WithBot ℝ) then
              NumericDType.real.sub (some 0)
                (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val)
            else maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val) :
        WithBot ℝ)))

/-- Genuine quantized value spec under `execR R` (`(src / data_scale).to(int8)`):
the int8 cast of the masked source lane divided by the **fp16-rounded** per-head
scale. -/
noncomputable def quantizeCopyKvSurfaceIntValueR (R : RoundingModel)
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Int :=
  WithBot.realToInt8
    (RoundingModel.cast R FloatDType.real FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num idx.1.val idx.2.1.val)
        (RoundingModel.cast R FloatDType.fp16 FloatDType.real
          (quantizeCopyKvScaleValueR R s K stride_k_bs stride_k_h stride_k_d head_num
            BLOCK_DMODEL hD idx.1.val))))

/-- Genuine fp16 scale store cell under `execR R` (`data_scale` after the fp16
store rounding): the value the kernel writes to `Out_scale`, observed through
`readMemAs .fp16`. `R.storeValue` is the store-boundary round (site 2). -/
noncomputable def quantizeCopyKvScaleCellR (R : RoundingModel)
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : TileCarrier .fp16 :=
  FloatDType.fp16.ofReal (R.storeValue FloatDType.fp16
    (quantizeCopyKvScaleValueR R s K stride_k_bs stride_k_h stride_k_d head_num
      BLOCK_DMODEL hD h))

/-- **Genuine value-output correctness under `execR R`.** The destination-indexed
int8 value store realizes `quantizeCopyKvSurfaceIntValueR` (read back via
`readMemValue .int Out`) — the honest `R`-dependent value (`src` divided by the
fp16-rounded scale). Not self-referential. -/
theorem destindex_copy_quantize_kv_real_surface_value_output_compute_correctR
    (R : RoundingModel)
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      let outAddr := outOffset s DestLoc stride_o_bs stride_o_h stride_o_d idx
      (execR R (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD)
          s).map (·.readMemValue .int Out outAddr)
        = some (if active s head_num BLOCK_HEAD BLOCK_DMODEL idx then
            quantizeCopyKvSurfaceIntValueR R s K stride_k_bs stride_k_h stride_k_d
              head_num BLOCK_DMODEL hD idx
          else s.readMemValue .int Out outAddr) := by
  intro idx
  simp only [outOffset, destIndex, headIndex, dimIndex]
  simp [execR, destindex_copy_quantize_kv_real_surface, stepStmtsR, stepStmtR,
        evalOpR, evalOpR.eq_def, BlockState.writeMemTypedR, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.ptrAdd, Tile.reduceMax,
        Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, TileShape.dropInsertedIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt, dif_pos hD]
  rw [foldl_writeMemAsR_fp16_prop_masked_readMemValue_int_other R OutScale _ _
    (fun i : TileIndex [BLOCK_HEAD, 1] => i.1.val < head_num) _ _ Out _ hOut]
  simp only [Tile.remap, Broadcast.leftIndex, Broadcast.consL, Broadcast.consSame,
    Broadcast.consR, Broadcast.nil, decide_eq_true_eq]
  rw [scatter_readback_int_prop_masked_nd (region := Out) (shape := [BLOCK_HEAD, BLOCK_DMODEL])
    _ (fun i => s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
        stride_o_h * i.1.val + stride_o_d * i.2.1.val)
    _ (fun i => i.1.val < head_num) ?_ idx]
  · simp only [active, headIndex]
    by_cases h : idx.1.val < head_num
    · rw [if_pos h, if_pos h]
      simp only [quantizeCopyKvSurfaceIntValueR, quantizeCopyKvScaleValueR, maskedSrc,
        if_pos h, Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      simp only [BlockState.setReg_readMemValue]
  · simpa [outOffset, destIndex, headIndex, dimIndex, BlockState.readMemValue] using hOutInj

/-- **Genuine scale-output correctness under `execR R`.** The destination-indexed
fp16 scale store realizes `quantizeCopyKvScaleCellR` (read back via
`readMemAs .fp16 OutScale`). Not self-referential. -/
theorem destindex_copy_quantize_kv_real_surface_scale_output_compute_correctR
    (R : RoundingModel)
    (K DestLoc Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d
      stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) (hD : 0 < BLOCK_DMODEL) (hOut : OutScale ≠ Out)
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD =>
        scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i)) :
    ∀ i : Fin BLOCK_HEAD,
      let outAddr := scaleOutOffset1 s DestLoc stride_os_bs stride_os_h i
      (execR R (destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
            stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
            stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD)
          s).map (·.readMemAs .fp16 OutScale outAddr)
        = some (if scaleActive head_num BLOCK_HEAD i then
            quantizeCopyKvScaleCellR R s K stride_k_bs stride_k_h stride_k_d head_num
              BLOCK_DMODEL hD i.val
          else s.readMemAs .fp16 OutScale outAddr) := by
  intro i
  simp only [scaleOutOffset1, destIndex, Nat.mul_comm i.val stride_os_h]
  simp [execR, destindex_copy_quantize_kv_real_surface, stepStmtsR, stepStmtR,
        evalOpR, evalOpR.eq_def, BlockState.writeMemTypedR, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.ptrAdd, Tile.reduceMax,
        Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, TileShape.dropInsertedIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt, dif_pos hD]
  simp only [Tile.remap, Broadcast.leftIndex, Broadcast.consL, Broadcast.consSame,
    Broadcast.consR, Broadcast.nil, decide_eq_true_eq]
  rw [scatter_readMemAs_R_fp16_prop_masked_nd R (region := OutScale) (shape := [BLOCK_HEAD, 1])
    _ (fun j => s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs + stride_os_h * j.1.val)
    _ (fun j => j.1.val < head_num) ?_ (i, ⟨0, by omega⟩, PUnit.unit)]
  · simp only [scaleActive]
    show (if i.val < head_num then _ else _) = (if i.val < head_num then _ else _)
    by_cases h : i.val < head_num
    · rw [if_pos h, if_pos h]
      simp only [quantizeCopyKvScaleCellR, quantizeCopyKvScaleValueR, maskedSrc, if_pos h,
        Option.map]
      rfl
    · rw [if_neg h, if_neg h]
      rw [foldl_writeMemTyped_int_prop_masked_readMemAs_fp16_other (region := Out)
        (P := fun k : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] => k.1.val < head_num)
        (hRR := hOut)]
      simp only [BlockState.setReg_readMemAs]
  · intro a b hab
    have h1 : a.1 = b.1 := by
      apply hScaleInj
      simpa [scaleOutOffset1, destIndex, BlockState.readMemValue] using hab
    obtain ⟨a1, a2, a3⟩ := a
    obtain ⟨b1, b2, b3⟩ := b
    simp only at h1
    subst h1
    have : a2 = b2 := Subsingleton.elim _ _
    subst this
    rfl

/-! ### The store-boundary round collapse

The scale readback quantizes twice — the `.to(tl.float16)` cast (`R.cast .real
.fp16`, site 1) then the fp16 store (`R.storeValue .fp16`, site 2). `round_idem`
collapses the pair to a single `R.round .fp16` of the exact real ideal. -/

/-- The exact (`R`-independent) real scale ideal `max(|src|, axis=1) / 127`. The
`⊨[R]` headline's `out2` value is this real quantized once at the fp16 grid. -/
noncomputable def quantizeCopyKvScaleReal
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (· / 127.0)
      ((Finset.univ.sup'
          (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
          (fun x : Fin BLOCK_DMODEL =>
            if maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val
                < (some 0 : WithBot ℝ) then
              NumericDType.real.sub (some 0)
                (maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val)
            else maskedSrc s K stride_k_bs stride_k_h stride_k_d head_num h x.val) :
        WithBot ℝ)))

/-- Store-boundary round collapse: `R.round dtype` of the `roundW`-then-`unbotD`
value is `R.round dtype` of the plain `unbotD` value (double rounding = one). -/
private theorem round_roundW_unbotD (R : RoundingModel) (dtype : FloatDType)
    (w : WithBot ℝ) :
    R.round dtype ((R.roundW dtype w).unbotD 0) = R.round dtype (w.unbotD 0) := by
  cases w with
  | bot => simp [RoundingModel.roundW]
  | coe x => simp [RoundingModel.roundW, R.round_idem]

/-- The fp16 scale cell under `execR R` is the exact real ideal rounded **once**
at the fp16 grid: `quantizeCopyKvScaleCellR = fp16.ofReal (R.round .fp16 (scale))`.
This is the honest crux — what `readMemAs .fp16 OutScale` observes. -/
theorem quantizeCopyKvScaleCellR_eq_round (R : RoundingModel)
    (s : BlockState) (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL : Nat)
    (hD : 0 < BLOCK_DMODEL) (h : Nat) :
    quantizeCopyKvScaleCellR R s K stride_k_bs stride_k_h stride_k_d head_num
        BLOCK_DMODEL hD h
      = FloatDType.fp16.ofReal (R.round FloatDType.fp16
          (quantizeCopyKvScaleReal s K stride_k_bs stride_k_h stride_k_d head_num
            BLOCK_DMODEL hD h)) := by
  unfold quantizeCopyKvScaleCellR quantizeCopyKvScaleValueR quantizeCopyKvScaleReal
  rw [RoundingModel.storeValue, RoundingModel.cast]
  simp only [FloatDType.real_toWithBot, FloatDType.storeValue, FloatDType.fp16_toWithBot,
    FloatDType.fp16_ofWithBot]
  rw [round_roundW_unbotD]

/-! ### `xs`-view specs (over the loaded tile) and their bridges

The two output halves as functions of the **loaded** tile
`xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ` (lane `j` is tile cell
`(j / BLOCK_DMODEL, j % BLOCK_DMODEL)` via the shared `Lane2D` bridge). Masked-off
lanes (`offs_h ≥ head_num`) take the `other=0.0` default. -/

/-- The masked source lane over the loaded tile (head-only mask). -/
noncomputable def srcXs (BLOCK_HEAD BLOCK_DMODEL head_num : Nat)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (h : Fin BLOCK_HEAD) (d : Fin BLOCK_DMODEL) : WithBot ℝ :=
  if h.val < head_num then
    some (xs (Lane2D.encode (h, d, PUnit.unit)))
  else some (0.0 : ℝ)

/-- Per-head scale fp16 carrier over `xs` under `execR R`. -/
noncomputable def scaleValXsR (R : RoundingModel) (BLOCK_HEAD BLOCK_DMODEL head_num : Nat)
    (hD : 0 < BLOCK_DMODEL) (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (h : Fin BLOCK_HEAD) : TileCarrier .fp16 :=
  RoundingModel.cast R FloatDType.real FloatDType.fp16
    (Option.map (· / 127.0)
      ((Finset.univ.sup'
          (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
          (fun d : Fin BLOCK_DMODEL =>
            if srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d < (some 0 : WithBot ℝ) then
              NumericDType.real.sub (some 0)
                (srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d)
            else srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d) :
        WithBot ℝ)))

/-- The exact real scale ideal over `xs` (`max |·| / 127`). -/
noncomputable def scaleRealXs (BLOCK_HEAD BLOCK_DMODEL head_num : Nat)
    (hD : 0 < BLOCK_DMODEL) (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (h : Fin BLOCK_HEAD) : ℝ :=
  WithBot.unbotD 0
    (Option.map (· / 127.0)
      ((Finset.univ.sup'
          (⟨⟨0, hD⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLOCK_DMODEL)).Nonempty)
          (fun d : Fin BLOCK_DMODEL =>
            if srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d < (some 0 : WithBot ℝ) then
              NumericDType.real.sub (some 0)
                (srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d)
            else srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d) :
        WithBot ℝ)))

/-- The `.int` value at lane `j` over `xs` under `execR R`. -/
noncomputable def valueXsR (R : RoundingModel) (BLOCK_HEAD BLOCK_DMODEL head_num : Nat)
    (hD : 0 < BLOCK_DMODEL) (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (j : Fin (BLOCK_HEAD * BLOCK_DMODEL)) : Int :=
  WithBot.realToInt8
    (RoundingModel.cast R FloatDType.real FloatDType.real
      (Option.map₂ (fun x1 x2 => x1 / x2)
        (srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs
          (Lane2D.decode j).1 (Lane2D.decode j).2.1)
        (RoundingModel.cast R FloatDType.fp16 FloatDType.real
          (scaleValXsR R BLOCK_HEAD BLOCK_DMODEL head_num hD xs (Lane2D.decode j).1))))

/-- Under the loaded-tile pin `hx`, the per-head scale fp16 value matches its
`xs`-view. -/
private theorem scaleValXsR_eq (R : RoundingModel)
    (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s₀ : BlockState) (hD : 0 < BLOCK_DMODEL)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (hx : ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL), j.val / BLOCK_DMODEL < head_num →
      s₀.readMem K (s₀.pids 0 * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
        stride_k_d * (j.val % BLOCK_DMODEL)) = xs j)
    (h : Fin BLOCK_HEAD) :
    quantizeCopyKvScaleValueR R s₀ K stride_k_bs stride_k_h stride_k_d
        head_num BLOCK_DMODEL hD h.val
      = scaleValXsR R BLOCK_HEAD BLOCK_DMODEL head_num hD xs h := by
  have hsrc : ∀ d : Fin BLOCK_DMODEL,
      maskedSrc s₀ K stride_k_bs stride_k_h stride_k_d head_num h.val d.val
        = srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d := by
    intro d
    unfold maskedSrc srcXs
    by_cases hact : h.val < head_num
    · rw [if_pos hact, if_pos hact]
      have hj := hx (Lane2D.encode (h, d, PUnit.unit))
        (by rw [Lane2D.encode_div]; exact hact)
      rw [Lane2D.encode_div, Lane2D.encode_mod] at hj
      rw [hj]
    · rw [if_neg hact, if_neg hact]
  unfold quantizeCopyKvScaleValueR scaleValXsR
  simp only [hsrc]

/-- Under `hx`, the exact real scale ideal matches its `xs`-view. -/
private theorem scaleRealXs_eq
    (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s₀ : BlockState) (hD : 0 < BLOCK_DMODEL)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (hx : ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL), j.val / BLOCK_DMODEL < head_num →
      s₀.readMem K (s₀.pids 0 * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
        stride_k_d * (j.val % BLOCK_DMODEL)) = xs j)
    (h : Fin BLOCK_HEAD) :
    quantizeCopyKvScaleReal s₀ K stride_k_bs stride_k_h stride_k_d
        head_num BLOCK_DMODEL hD h.val
      = scaleRealXs BLOCK_HEAD BLOCK_DMODEL head_num hD xs h := by
  have hsrc : ∀ d : Fin BLOCK_DMODEL,
      maskedSrc s₀ K stride_k_bs stride_k_h stride_k_d head_num h.val d.val
        = srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs h d := by
    intro d
    unfold maskedSrc srcXs
    by_cases hact : h.val < head_num
    · rw [if_pos hact, if_pos hact]
      have hj := hx (Lane2D.encode (h, d, PUnit.unit))
        (by rw [Lane2D.encode_div]; exact hact)
      rw [Lane2D.encode_div, Lane2D.encode_mod] at hj
      rw [hj]
    · rw [if_neg hact, if_neg hact]
  unfold quantizeCopyKvScaleReal scaleRealXs
  simp only [hsrc]

/-- Under `hx`, the `.int` value at lane `j` matches its `xs`-view. -/
private theorem valueXsR_eq (R : RoundingModel)
    (K : RegionName)
    (stride_k_bs stride_k_h stride_k_d head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s₀ : BlockState) (hD : 0 < BLOCK_DMODEL)
    (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (hx : ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL), j.val / BLOCK_DMODEL < head_num →
      s₀.readMem K (s₀.pids 0 * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
        stride_k_d * (j.val % BLOCK_DMODEL)) = xs j)
    (j : Fin (BLOCK_HEAD * BLOCK_DMODEL)) :
    quantizeCopyKvSurfaceIntValueR R s₀ K stride_k_bs stride_k_h stride_k_d
        head_num BLOCK_DMODEL hD (Lane2D.decode j)
      = valueXsR R BLOCK_HEAD BLOCK_DMODEL head_num hD xs j := by
  unfold quantizeCopyKvSurfaceIntValueR valueXsR
  rw [scaleValXsR_eq R K stride_k_bs stride_k_h stride_k_d
    head_num BLOCK_DMODEL BLOCK_HEAD s₀ hD xs hx (Lane2D.decode j).1]
  have hsrc :
      maskedSrc s₀ K stride_k_bs stride_k_h stride_k_d head_num
          (Lane2D.decode j).1.val (Lane2D.decode j).2.1.val
        = srcXs BLOCK_HEAD BLOCK_DMODEL head_num xs
            (Lane2D.decode j).1 (Lane2D.decode j).2.1 := by
    unfold maskedSrc srcXs
    by_cases hact : (Lane2D.decode j).1.val < head_num
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
        rw [Lane2D.decode_row] at hact; exact hact)
      rw [hj]
    · rw [if_neg hact, if_neg hact]
  rw [hsrc]

/-! ### IO signature, termination, safety, frame -/

/-- The full quantize-copy-kv surface's typed two-output metadata **IO
signature** on the rounding axis: the `.nat` dest-index slot `DestLoc[pid₀]`, the
head-masked `K` data tile, the `.int` quantized tile `out1 = Out`
(`oty1 := .int`), and the **fp16** scale column `out2 = OutScale`
(`oty2 := .float` for the exact axis is unused; `out2DType := .fp16` puts it on
the rounding axis). Lane `j` of the data/`out1` tile is
`(j / BLOCK_DMODEL, j % BLOCK_DMODEL)`; `out2` has one column per head. -/
def quantizeCopyKvIO
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    MetaMasked2DKernelIO₁ₓ₂ where
  kernel := destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD
  mbuf1 := DestLoc
  inp := K
  out1 := Out
  out2 := OutScale
  B := BLOCK_HEAD * BLOCK_DMODEL
  C := BLOCK_HEAD
  oty1 := .int
  oty2 := .float
  out2DType := .fp16
  mwin1 := fun pid₀ _ => pid₀
  read := fun pid₀ _ _ j =>
    pid₀ * stride_k_bs + (j.val / BLOCK_DMODEL) * stride_k_h +
      stride_k_d * (j.val % BLOCK_DMODEL)
  write1 := fun _ _ m1 j =>
    m1 * stride_o_bs + stride_o_h * (j.val / BLOCK_DMODEL) +
      stride_o_d * (j.val % BLOCK_DMODEL)
  write2 := fun _ _ m1 i => m1 * stride_os_bs + stride_os_h * i.val
  mask := fun _ _ _ j => j.val / BLOCK_DMODEL < head_num
  writeMask2 := fun _ _ _ i => i.val < head_num

/-- Termination under `execR R`: the surface executes to completion from any
state (the `tl.max(·, axis=1)` reduce axis is nonempty). -/
private theorem quantize_copy_kv_exec_isSomeR (R : RoundingModel)
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) (hD : 0 < BLOCK_DMODEL)
    (s : BlockState) :
    ∃ s1, execR R ((destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel) s = some s1 := by
  simp [execR, destindex_copy_quantize_kv_real_surface, ComputeKernel.toAlgKernel,
        stepStmtsR, stepStmtR, evalOpR, evalOpR.eq_def, BlockState.writeMemTypedR,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop, Tile.expandDim,
        Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, dif_pos hD]

/-- The surface sits inside the flat-memory bridge's covered fragment
(`R`-independent). -/
theorem quantize_copy_kv_flattenOk
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ((destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [destindex_copy_quantize_kv_real_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk under `R`: the scalar `.nat` `DestLoc` load, the
head-masked `K` load, and the two masked stores reduce to the cell/lane-wise
bounds. -/
theorem quantize_copy_kv_traceSafeR (R : RoundingModel)
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) (hD : 0 < BLOCK_DMODEL)
    (bounds : RegionBounds) (s : BlockState)
    (hd : s.pids 0 < bounds DestLoc)
    (hk : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL], idx.1.val < head_num →
      s.pids 0 * stride_k_bs + idx.1.val * stride_k_h + stride_k_d * idx.2.1.val
        < bounds K)
    (ho : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL], idx.1.val < head_num →
      s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
        stride_o_h * idx.1.val + stride_o_d * idx.2.1.val < bounds Out)
    (hos : ∀ i : Fin BLOCK_HEAD, i.val < head_num →
      s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs + stride_os_h * i.val
        < bounds OutScale) :
    Kernel.TraceSafeR R bounds
      ((destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp [destindex_copy_quantize_kv_real_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    BlockState.writeMemTypedR, foldl_writeMemTyped_int_prop_masked_regs,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    MaskOpt.ActiveR,
    Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.remap, Tile.reduceMax,
    Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex,
    Option.bind, Option.map, TileShape.insertAxis, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, NumericDType.div, ComparableDType.lt,
    dif_pos hD]
  refine ⟨hd, fun a a_1 h1 => hk (a, a_1, PUnit.unit) h1,
    fun a a_1 h1 => ho (a, a_1, PUnit.unit) h1, fun a h1 => hos a h1⟩

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

/-- Masked `writeMemAsR R .fp16` store `foldl` cell frame: a masked rounded fp16
fold leaves every cell it does not actively hit unchanged. -/
private theorem foldl_writeMemAsR_fp16_preserve_cell {α : Type} (R : RoundingModel)
    {region : RegionName} (offsetFn : α → Nat) (valueFn : α → TileCarrier .fp16)
    (P : α → Prop) [DecidablePred P] (r : RegionName) (o : Nat) (l : List α)
    (s : BlockState) (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMemAsR R .fp16 region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP, ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMemAsR_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Frame under `execR R`: every cell off both written windows (the active `Out`
value cells and the active `OutScale` scale cells) is preserved. -/
private theorem quantize_copy_kv_frameR (R : RoundingModel)
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) (hD : 0 < BLOCK_DMODEL)
    (s s1 : BlockState)
    (hExec : execR R ((destindex_copy_quantize_kv_real_surface K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL
        BLOCK_HEAD).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissVal : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL], idx.1.val < head_num →
      ¬(Out = r ∧ s.readMemValue .nat DestLoc (s.pids 0) * stride_o_bs +
          stride_o_h * idx.1.val + stride_o_d * idx.2.1.val = o))
    (hmissScale : ∀ i : Fin BLOCK_HEAD, i.val < head_num →
      ¬(OutScale = r ∧ s.readMemValue .nat DestLoc (s.pids 0) * stride_os_bs +
          stride_os_h * i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [execR, destindex_copy_quantize_kv_real_surface,
        ComputeKernel.toAlgKernel, stepStmtsR, stepStmtR, evalOpR, evalOpR.eq_def,
        BlockState.writeMemTypedR, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.uop, Tile.expandDim, Tile.ptrAdd, Tile.remap, Tile.reduceMax,
        Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, TileShape.dropInsertedIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, dif_pos hD] at hExec
  subst hExec
  refine Eq.trans (foldl_writeMemAsR_fp16_preserve_cell R _ _ _ r o _ _ ?_) ?_
  · intro k _ hP hc
    exact hmissScale k.1 hP hc
  · refine Eq.trans (foldl_writeMemTyped_int_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hP hc
    exact hmissVal k hP hc

/-! ### The headline -/

open scoped VeriTile.Triton.MetaMasked2DKernelIO₁ₓ₂

/-- **The `⊨[R]` headline (mixed-dtype rounding metadata surface).** For every
rounding model `R`, the full faithful `quantize_copy_kv` surface implements, on
its `MetaMasked2DKernelIO₁ₓ₂` signature, the pair `(int8 quantized tile, fp16
`max|·|/127` scale column)` over the loaded tile `xs` and the loaded dest-index
slot `m1` — with the fp16 scale on the **rounding axis** (`out2DType := .fp16`):
`readMemAs .fp16 OutScale` holds `fp16.ofReal (R.round .fp16 (max|·|/127))`, and
the int8 value tile divides by the fp16-rounded scale.

Honest side conditions, each required for truth: `hD : 0 < BLOCK_DMODEL` (the
`tl.max(·, axis=1)` reduce axis must be nonempty — this is also what makes the
kernel terminate), `hOut : Out ≠ OutScale` (the two stores must not alias), and
the value-tile / scale-column destination-offset injectivity `hValInj` /
`hScaleInj` (a masked scatter with colliding lanes is last-writer-wins, so the
per-lane readbacks are false without them; both are `Dest_loc`-independent since
the loaded row only shifts every address by a constant). The masked `K` load
carries `other=0.0`, so masked-off lanes default to `0`. -/
specification quantize_copy_kv_io_correctness (R : RoundingModel)
    (K : RegionName) (DestLoc : RegionName) (Out OutScale : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (hD : 0 < BLOCK_DMODEL) (hOut : Out ≠ OutScale)
    (hValInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        stride_o_h * idx.1.val + stride_o_d * idx.2.1.val))
    (hScaleInj : Function.Injective
      (fun i : Fin BLOCK_HEAD => stride_os_h * i.val)) :
    quantizeCopyKvIO K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD
      ⊨[R] fun _ _ _ xs =>
          (fun j => valueXsR R BLOCK_HEAD BLOCK_DMODEL head_num hD xs j,
           fun i => scaleRealXs BLOCK_HEAD BLOCK_DMODEL head_num hD xs i) := by
  refine MetaMasked2DKernelIO₁ₓ₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact quantize_copy_kv_flattenOk K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD
  · -- safety
    intro bounds s m1 hm1 hb1 hbr hbw1 hbw2
    simp only [quantizeCopyKvIO] at hm1 hb1 hbr hbw1 hbw2
    refine quantize_copy_kv_traceSafeR R K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD
      hD bounds s hb1 (fun idx h1 => ?_) (fun idx h1 => ?_) (fun i h1 => ?_)
    · simpa [Lane2D.encode_div, Lane2D.encode_mod] using
        hbr (Lane2D.encode idx) (by simpa [Lane2D.encode_div] using h1)
    · rw [hm1]
      simpa [Lane2D.encode_div, Lane2D.encode_mod] using
        hbw1 (Lane2D.encode idx) (by simpa [Lane2D.encode_div] using h1)
    · rw [hm1]
      simpa using hbw2 i (by simpa using h1)
  · -- run
    intro s₀ m1 xs _hundef hm1 hx
    simp only [quantizeCopyKvIO] at hm1 hx ⊢
    have hOutInj : Function.Injective
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          outOffset s₀ DestLoc stride_o_bs stride_o_h stride_o_d idx) := by
      intro a b hab
      apply hValInj
      simp only [outOffset, headIndex, dimIndex] at hab
      rw [Nat.add_assoc, Nat.add_assoc] at hab
      exact Nat.add_left_cancel hab
    have hScInj : Function.Injective
        (fun i : Fin BLOCK_HEAD => scaleOutOffset1 s₀ DestLoc stride_os_bs stride_os_h i) := by
      intro a b hab
      apply hScaleInj
      simp only [scaleOutOffset1] at hab
      show stride_os_h * a.val = stride_os_h * b.val
      rw [Nat.mul_comm stride_os_h a.val, Nat.mul_comm stride_os_h b.val]
      exact Nat.add_left_cancel hab
    obtain ⟨s1, hs1⟩ := quantize_copy_kv_exec_isSomeR R K DestLoc Out OutScale
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD hD s₀
    refine ⟨s1, hs1, ?_, ?_, ?_⟩
    · -- value output
      intro j hmask
      have hv := destindex_copy_quantize_kv_real_surface_value_output_compute_correctR
        R K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD s₀
        hD hOut hOutInj (Lane2D.decode j)
      have haddr : outOffset s₀ DestLoc stride_o_bs stride_o_h stride_o_d (Lane2D.decode j)
          = m1 * stride_o_bs + stride_o_h * (j.val / BLOCK_DMODEL) +
            stride_o_d * (j.val % BLOCK_DMODEL) := by
        simp only [outOffset, destIndex, headIndex, dimIndex, Lane2D.decode_row,
          Lane2D.decode_col, hm1]
      have hact : active s₀ head_num BLOCK_HEAD BLOCK_DMODEL (Lane2D.decode j) := by
        simp only [active, headIndex, Lane2D.decode_row]
        exact hmask
      rw [hs1, haddr] at hv
      simp only [if_pos hact] at hv
      have hkey : s1.readMemValue .int Out (m1 * stride_o_bs +
          stride_o_h * (j.val / BLOCK_DMODEL) + stride_o_d * (j.val % BLOCK_DMODEL))
        = quantizeCopyKvSurfaceIntValueR R s₀ K stride_k_bs stride_k_h stride_k_d
            head_num BLOCK_DMODEL hD (Lane2D.decode j) := Option.some.inj hv
      show s1.readMemValue .int Out (m1 * stride_o_bs +
        stride_o_h * (j.val / BLOCK_DMODEL) + stride_o_d * (j.val % BLOCK_DMODEL)) = _
      rw [hkey,
        valueXsR_eq R K stride_k_bs stride_k_h stride_k_d
          head_num BLOCK_DMODEL BLOCK_HEAD s₀ hD xs hx j]
    · -- scale output (fp16 rounding channel)
      intro i hmask
      have hv := destindex_copy_quantize_kv_real_surface_scale_output_compute_correctR
        R K DestLoc Out OutScale stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD s₀
        hD (fun h => hOut h.symm) hScInj i
      have haddr : scaleOutOffset1 s₀ DestLoc stride_os_bs stride_os_h i
          = m1 * stride_os_bs + stride_os_h * i.val := by
        simp only [scaleOutOffset1, destIndex, hm1, Nat.mul_comm i.val stride_os_h]
      have hact : scaleActive head_num BLOCK_HEAD i := hmask
      rw [hs1, haddr] at hv
      simp only [if_pos hact] at hv
      have hkey : s1.readMemAs .fp16 OutScale (m1 * stride_os_bs + stride_os_h * i.val)
        = quantizeCopyKvScaleCellR R s₀ K stride_k_bs stride_k_h stride_k_d
            head_num BLOCK_DMODEL hD i.val := Option.some.inj hv
      show s1.readMemAs .fp16 OutScale (m1 * stride_os_bs + stride_os_h * i.val) = _
      rw [hkey, quantizeCopyKvScaleCellR_eq_round R s₀ K stride_k_bs stride_k_h stride_k_d
        head_num BLOCK_DMODEL hD i.val,
        scaleRealXs_eq K stride_k_bs stride_k_h stride_k_d
          head_num BLOCK_DMODEL BLOCK_HEAD s₀ hD xs hx i]
    · -- frame
      intro r o hc1 hc2
      refine quantize_copy_kv_frameR R K DestLoc Out OutScale
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        stride_os_bs stride_os_h stride_os_d head_num BLOCK_DMODEL BLOCK_HEAD
        hD s₀ s1 hs1 r o (fun idx h1 ⟨hr, ho⟩ => ?_) (fun i h1 ⟨hr, ho⟩ => ?_)
      · rcases hc1 with hne | hno
        · exact hne hr.symm
        · refine hno (Lane2D.encode idx) (by simpa [Lane2D.encode_div] using h1) ?_
          simp only [Lane2D.encode_div, Lane2D.encode_mod]
          rw [← ho]
          simp only [destIndex, hm1]
      · rcases hc2 with hne | hno
        · exact hne hr.symm
        · refine hno i h1 ?_
          rw [← ho]
          simp only [destIndex, hm1]

end VeriTile.Bench.TritonBenchG.QuantizeCopyKv
