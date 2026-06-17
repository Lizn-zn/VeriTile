import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.Kernel
import VeriTile.Triton.Semantics.BlockPtrEval
import VeriTile.Examples.FlashAttention1

/-!
# `triton_attention` — strict per-kernel correctness

`triton_attention.py` is a full FlashAttention training pipeline of three
`@triton.jit` kernels: `_fwd_kernel` (online-softmax forward, stores the output
`Out` plus the running `L`/`M` log-sum-exp rows), `_bwd_preprocess` (computes
`NewDO = DO` and the per-row `Delta = sum(DO·O)`), and `_bwd_kernel` (the main
backward producing the `DQ`/`DK`/`DV` gradients, with the score-side `P`/`DS`
arithmetic as an inner step). Scaling is `1/√D`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`grid = (cdiv(T, BLOCK), B·H, 1)`, the
`torch.autograd.Function` orchestration that chains forward → preprocess →
backward, and how the runtime composes per-program writes into each buffer) is
the *trusted boundary*, not a proof obligation here. Because the program ids
are universally quantified (via `s`), the per-program statements cover every
program of each grid.

## Proof architecture

```
triton_attention_bwd_python_test_shape_complete_summary             ← TOP THEOREM (bwd grads + score)
  ├─ triton_attention_bwd_grads_python_test_shape_output_summary
  │    ├─ triton_attention_bwd_kernel_toAlgorithm_supported
  │    └─ triton_attention_bwd_grads_python_test_shape_all_outputs_compute_correct
  │         └─ triton_attention_bwd_kernel_{dq,dk,dv}_python_test_shape_compute_correct
  │              └─ triton_attention_bwd_{dq,dkdv}_store_slice_compute_correct ...
  └─ triton_attention_bwd_score_python_test_shape_formula_summary
       └─ triton_attention_bwd_score_{p,ds}_formula_python_test_shape_compute_correct
            └─ triton_attention_bwd_score_{p,ds}_formula_slice_compute_correct   ← closed-form P/DS

triton_attention_forward_python_test_shape_output_summary           ← TOP (forward Out/L/M)
  └─ triton_attention_forward_surface_{out,l,m}_python_test_shape_compute_correct
       └─ triton_attention_forward_{output,l,m}_store_slice_compute_correct → ..._correct

triton_attention_bwd_preprocess_python_test_shape_output_summary    ← TOP (preprocess NewDO/Delta)
  └─ triton_attention_bwd_preprocess_python_test_shape_all_outputs_compute_correct
       └─ triton_attention_bwd_preprocess_{newdo,delta}_surface_compute_correct
            └─ ..._{store,formula}_slice_compute_correct → ..._correct
```
(Offset injectivity discharged by the `triton_attention_python_*_offset_injective` lemmas.)

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `exp2`/`log2`, the
`tl.dot` `float16` accumulations, and `1/√D` scaling are not modeled at the bit
level); `@triton.autotune`/`num_warps`/`num_stages` are not modeled. The
modeling depth differs by kernel:

* **Forward** (`Out`, `L`, `M`) is verified **end-to-end against the genuine
  causal FlashAttention-1 closed form**: the full lowered body (preLoop +
  `forRangeDyn` streaming loop + postLoop) executes to a state whose `Out`/`L`/`M`
  stores hold `fwdOutSpec` (the natural-exp causal attention block
  `oPartial / lPartial`, the in-loop-`l_rcp`-cancellation made precise via the
  `FA1MathCausal` streaming accumulators), `fwdLSpec` (the stored m-shifted
  normalizer `lPartial`), and `fwdMSpec` (the per-row causal score maximum
  `mPartial`). See `ta_exec` and the `triton_attention_forward_surface_*` theorems.
* **Backward grads** (`DQ`, `DK`, `DV`) are **final-store scoped**: the proofs
  establish the masked/full stores write the accumulator slices at the correct,
  injective offsets and preserve inactive lanes; the written values are the
  opaque `producedBwdKernelD{Q,K,V}Value` carriers, not re-derived as closed
  forms.
* **Backward preprocess** and the **backward score `P`/`DS` step** are verified
  against explicit closed-form specs (`bwdScorePFormulaSpec`,
  `bwdScoreDSFormulaSpec`, and the `newdo`/`delta` formula slices), not opaque
  carriers — these inner arithmetic steps are checked, the surrounding loop
  composition is trusted.

Side conditions: the test-shape summaries fix `(B,H,T,D) = (2,4,128,64)`,
`BLOCK_M = 128`, `BLOCK_DMODEL = 64`, strides `(32768, 8192, 64, 1)`,
`num_block = 1`, `sm_scale = 1/√64`; the complete backward summary additionally
requires `PTile ≠ DSTile`.
-/

namespace VeriTile.Bench.TritonBenchG.TritonAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `triton_attention_forward_python_test_shape_output_summary`, `triton_attention_bwd_preprocess_genuine_output_summary_general`, `triton_attention_bwd_grads_genuine_output_summary` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-- DSL port of `triton_attention.py`'s `_fwd_kernel`. -/
def triton_attention_fwd_kernel
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (_stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _stride_oz _stride_oh stride_om stride_on
      _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  stride_qh_2d = $(stride_qh) // $(stride_qm) // $(stride_qk)

  q_tile_ptr = tl.make_block_ptr(base=Q,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  k_tile_ptr = tl.make_block_ptr(base=K,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_kn), $(stride_kk)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  v_tile_ptr = tl.make_block_ptr(base=V,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  out_tile_ptr = tl.make_block_ptr(base=Out,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  q = tl.load(q_tile_ptr)

  for start_n in range($(0), (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    k = tl.load(k_tile_ptr, boundary_check=(0, 1))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_tile_ptr, boundary_check=(0, 1))
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_tile_ptr = tl.advance(k_tile_ptr, [$(BLOCK_N), $(0)])
    v_tile_ptr = tl.advance(v_tile_ptr, [$(BLOCK_N), $(0)])
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_ptrs = L + off_hz * $(N_CTX) + offs_m
  m_ptrs = M + off_hz * $(N_CTX) + offs_m
  tl.store(l_ptrs, l_prev)
  tl.store(m_ptrs, m_prev)

  acc = (acc).to(tl.float16)
  tl.store(out_tile_ptr, acc, boundary_check=(0, 1))
}

/-- The full Python-shaped forward attention surface lowers to the algorithm
layer, including the streaming softmax loop and final L/M/O stores. -/
theorem triton_attention_fwd_kernel_toAlgorithm_supported
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (_stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _stride_oz _stride_oh stride_om stride_on
      _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (triton_attention_fwd_kernel Q K V L M Out sm_scale _stride_qz
      stride_qh stride_qm stride_qk _stride_kz _stride_kh stride_kn
      stride_kk _stride_vz _stride_vh stride_vk stride_vn _stride_oz
      _stride_oh stride_om stride_on _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL
      BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented forward output-store slice of
`triton_attention.py`'s `_fwd_kernel`.

The Python kernel writes `acc` through a block pointer with
`boundary_check=(0, 1)`. This slice spells the same write as explicit pointer
arithmetic and an explicit two-axis boundary mask. The inner `tl.float32`
streaming-softmax accumulator is outside this slice. -/
def triton_attention_forward_output_store_slice
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] + $(hzRowOffset) < $(D0)) &
    (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + offs_m[:, None] * $(BLOCK_DMODEL) + offs_d[None, :],
    mask=mask, other=0.0)
  tl.store(Out + (offs_m[:, None] + $(hzRowOffset)) * $(stride_om) +
      offs_d[None, :] * $(stride_on), (acc).to(tl.float16), mask=mask)
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

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
        fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
      by_cases hmaskhd : mask hd = Bool.true
      · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
        simp only [hmaskhd, if_true]
        rw [ih _ htl]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
        change
          (if region = region ∧ o = offsetFn hd then
            MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn hd)))
          else
            s.mem region o) = s.mem region o
        rw [if_neg (by
          intro hsame
          exact hhd hsame.2.symm)]
      · have hmaskhd' : mask hd = Bool.false := by
          cases hm : mask hd
          · rfl
          · exact False.elim (hmaskhd hm)
        simp only [hmaskhd', if_false, Bool.false_eq_true]
        exact ih _ htl

/-- A foldl of masked `fp16` typed writes into region `W` leaves `mem` at any other
region `R ≠ W` unchanged. -/
private theorem foldl_writeMemTyped_fp16_mask_other_region {α : Type} {W R : RegionName}
    (mask : α → Prop) [DecidablePred mask]
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16) (o : Nat)
    (hRW : R ≠ W) (l : List α) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k => if mask k then acc.writeMemTyped .fp16 W (offsetFn k) (valueFn k) else acc) s).mem R o)
        = s.mem R o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih]
      by_cases hmk : mask hd
      · simp only [hmk, if_true]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
        change (if R = W ∧ o = offsetFn hd then _ else s.mem R o) = s.mem R o
        rw [if_neg (by rintro ⟨hR, _⟩; exact hRW hR)]
      · simp only [hmk, if_false]

/-- A foldl of `real` typed writes into region `W` leaves `mem` at any other
region `R ≠ W` unchanged. -/
private theorem foldl_writeMemTyped_real_other_region {α : Type} {W R : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.real) (o : Nat)
    (hRW : R ≠ W) (l : List α) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k => acc.writeMemTyped .real W (offsetFn k) (valueFn k)) s).mem R o)
        = s.mem R o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih]
      simp only [BlockState.writeMemTyped_real]
      show (if R = W ∧ o = offsetFn hd then _ else s.mem R o) = s.mem R o
      rw [if_neg (by rintro ⟨hR, _⟩; exact hRW hR)]

private theorem scatter_memcell_fp16_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i))
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
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
        if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    simp
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

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
          (·.mem Out outAddr)
        = some (if active s hzRowOffset D0 BLOCK_M idx then
            MemCell.of .fp16
              (FloatDType.real.cast FloatDType.fp16
                (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))
          else s.mem Out outAddr) := by
  intro idx
  simp [exec, triton_attention_forward_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, dIndex, active, accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset) * stride_om +
        idx.2.1.val * stride_on
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (WithBot.unbotD 0
          (if s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0 then
            some (s.readMem Acc
              ((s.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val))
          else some (0.0 : ℝ))))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, rowIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMemTyped .fp16 Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).mem Out
        (offsetFn idx) =
    if P idx then
      MemCell.of .fp16
        (FloatDType.real.cast FloatDType.fp16
          (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))
    else s.mem Out (offsetFn idx)
  rw [scatter_memcell_fp16_prop_masked_nd _ _ _ _ hOffsetInj idx]
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
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))) := by
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

/-- Proof-oriented L (log-sum-exp) row store slice of `triton_attention.py`'s
forward kernel. Writes a precomputed `LPrev` vector into `L` at the per-row
`off_hz * N_CTX + offs_m` strided offset. Companion to the output store
slice. -/
def triton_attention_forward_l_store_slice
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_prev = tl.load(LPrev + $(off_hz) * $(N_CTX) + offs_m)
  tl.store(L + $(off_hz) * $(N_CTX) + offs_m, l_prev)
}

def lRowOffset (s : BlockState) (off_hz N_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * N_CTX + (s.pids 0 * BLOCK_M + i.val)

noncomputable def lStoreSpec (s : BlockState) (LPrev : RegionName)
    (off_hz N_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem LPrev (lRowOffset s off_hz N_CTX BLOCK_M i)

theorem triton_attention_forward_l_store_slice_correct
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz N_CTX BLOCK_M i
      (exec (triton_attention_forward_l_store_slice LPrev L off_hz N_CTX BLOCK_M)
          s).map (·.readMem L outAddr)
        = some (lStoreSpec s LPrev off_hz N_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * N_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : lRowOffset s off_hz N_CTX BLOCK_M a =
        lRowOffset s off_hz N_CTX BLOCK_M b := by
      simpa [lRowOffset, Nat.add_assoc] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, triton_attention_forward_l_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [lStoreSpec, lRowOffset, Nat.add_assoc]

theorem triton_attention_forward_l_store_slice_compute_correct
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => lStoreSpec s LPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_l_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_forward_l_store_slice_correct LPrev L
    off_hz N_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented M (max) row store slice of `triton_attention.py`'s forward
kernel. Mirrors the L-row store slice. -/
def triton_attention_forward_m_store_slice
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_prev = tl.load(MPrev + $(off_hz) * $(N_CTX) + offs_m)
  tl.store(M + $(off_hz) * $(N_CTX) + offs_m, m_prev)
}

noncomputable def mStoreSpec (s : BlockState) (MPrev : RegionName)
    (off_hz N_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem MPrev (lRowOffset s off_hz N_CTX BLOCK_M i)

theorem triton_attention_forward_m_store_slice_correct
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz N_CTX BLOCK_M i
      (exec (triton_attention_forward_m_store_slice MPrev M off_hz N_CTX BLOCK_M)
          s).map (·.readMem M outAddr)
        = some (mStoreSpec s MPrev off_hz N_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * N_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : lRowOffset s off_hz N_CTX BLOCK_M a =
        lRowOffset s off_hz N_CTX BLOCK_M b := by
      simpa [lRowOffset, Nat.add_assoc] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, triton_attention_forward_m_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [mStoreSpec, lRowOffset, Nat.add_assoc]

theorem triton_attention_forward_m_store_slice_compute_correct
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => mStoreSpec s MPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_m_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_forward_m_store_slice_correct MPrev M
    off_hz N_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ### Auxiliary backward-preprocess slices

The `_bwd_preprocess` kernel in `triton_attention.py` is much simpler than the
forward / backward main kernels: it loads `O`, `DO`, `L`, computes
`do = do / L[:, None]`, then writes `do` to `NewDO` and `sum(o * do, axis=1)`
to `Delta`. The streaming softmax / tl.dot / make_block_ptr / advance pieces
that block the main attention loop are absent. The two store-back regions
support clean proof-oriented slices analogous to the forward `L`/`M` row store
slices already in this file.
-/

/-- DSL port of `triton_attention.py`'s `_bwd_preprocess`. -/
def triton_attention_bwd_preprocess
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  do_val = do_val / denom[:, None]
  delta = tl.sum(o * do_val, axis=1)
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val)
  tl.store(Delta + off_m, delta)
}

theorem triton_attention_bwd_preprocess_toAlgorithm_supported
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD).toAlgorithm? = Except.ok alg := by
  simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- DSL port of `triton_attention.py`'s main `_bwd_kernel`.

The Python test shape has `num_block = 1`, so the post-inner-loop pointer reset
`lo + (1 - num_block) * BLOCK_M` is exactly zero. Under that checked launch,
this surface preserves the block-pointer construction, nested loops, causal
mask, DQ accumulation store, DK/DV accumulator stores, and pointer advances. -/
def triton_attention_bwd_kernel
    (Q K V _Out DO DQ DK DV _L M Delta : RegionName)
    (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _Z H N_CTX D0 num_block BLOCK_M BLOCK_DMODEL _BLOCK_N : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  stride_qz_2d = $(stride_qz) // $(stride_qm) // $(stride_qk)
  stride_qh_2d = $(stride_qh) // $(stride_qm) // $(stride_qk)
  q_tile_ptr = tl.make_block_ptr(base=Q,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  k_tile_ptr = tl.make_block_ptr(base=K,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_kn), $(stride_kk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  v_tile_ptr = tl.make_block_ptr(base=V,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  do_tile_ptr = tl.make_block_ptr(base=DO,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dq_tile_ptr = tl.make_block_ptr(base=DQ,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dk_tile_ptr = tl.make_block_ptr(base=DK,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dv_tile_ptr = tl.make_block_ptr(base=DV,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  DQ = DQ + off_z * $(stride_qz) + off_h * $(stride_qh)
  for start_n in range($(0), $(num_block), $(1)) {
    lo = start_n * $(BLOCK_M)
    offs_qm = lo + tl.arange(0, $(BLOCK_M))
    offs_n = start_n * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
    offs_m = tl.arange(0, $(BLOCK_M))
    offs_k = tl.arange(0, $(BLOCK_DMODEL))
    dq_ptrs = DQ + (offs_qm[:, None] * $(stride_qm) +
      offs_k[None, :] * $(stride_qk))
    D_ptrs = Delta + off_hz * $(N_CTX)
    m_ptrs = M + off_hz * $(N_CTX)
    dv = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    dk = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    k = tl.load(k_tile_ptr, boundary_check=(0, 1))
    v = tl.load(v_tile_ptr, boundary_check=(0, 1))
    for start_m in range(lo, $(num_block) * $(BLOCK_M), $(BLOCK_M)) {
      offs_m_curr = start_m + offs_m
      q = tl.load(q_tile_ptr, boundary_check=(0, 1))
      qk = tl.dot(q, tl.trans(k))
      qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk,
        float("-inf"))
      m = tl.load(m_ptrs + offs_m_curr)
      p = tl.exp(qk * $((sm_scale : ℝ)) - m[:, None])
      do_val = tl.load(do_tile_ptr, boundary_check=(0, 1))
      dv += tl.dot(tl.trans((p).to(tl.float16)), do_val)
      Di = tl.load(D_ptrs + offs_m_curr)
      dp = tl.zeros([$(BLOCK_M), $(BLOCK_M)], dtype=tl.float32) - Di[:, None]
      dp += tl.dot(do_val, tl.trans(v))
      ds = p * dp * $((sm_scale : ℝ))
      dk += tl.dot(tl.trans((ds).to(tl.float16)), q)
      dq = tl.load(dq_tile_ptr)
      dq += tl.dot((ds).to(tl.float16), k)
      tl.store(dq_tile_ptr, dq)
      dq_ptrs += $(BLOCK_M) * $(stride_qm)
      q_tile_ptr = tl.advance(q_tile_ptr, [$(BLOCK_M), $(0)])
      do_tile_ptr = tl.advance(do_tile_ptr, [$(BLOCK_M), $(0)])
      dq_tile_ptr = tl.advance(dq_tile_ptr, [$(BLOCK_M), $(0)])
    }
    q_tile_ptr = tl.advance(q_tile_ptr, [$(0), $(0)])
    do_tile_ptr = tl.advance(do_tile_ptr, [$(0), $(0)])
    dq_tile_ptr = tl.advance(dq_tile_ptr, [$(0), $(0)])
    k_tile_ptr = tl.advance(k_tile_ptr, [$(BLOCK_M), $(0)])
    v_tile_ptr = tl.advance(v_tile_ptr, [$(BLOCK_M), $(0)])
    tl.store(dv_tile_ptr, (dv).to(tl.float16), boundary_check=(0, 1))
    tl.store(dk_tile_ptr, (dk).to(tl.float16), boundary_check=(0, 1))
    dv_tile_ptr = tl.advance(dv_tile_ptr, [$(BLOCK_M), $(0)])
    dk_tile_ptr = tl.advance(dk_tile_ptr, [$(BLOCK_M), $(0)])
  }
}

theorem triton_attention_bwd_kernel_toAlgorithm_supported
    (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _Z H N_CTX D0 num_block BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      sm_scale stride_qz stride_qh stride_qm stride_qk _stride_kz _stride_kh
      stride_kn stride_kk _stride_vz _stride_vh stride_vk stride_vn _Z H
      N_CTX D0 num_block BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgorithm? =
        Except.ok alg := by
  simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented `NewDO` 2D store slice of `triton_attention.py`'s
`_bwd_preprocess`. The kernel stores a (precomputed) `NewDOAcc` tile to
`NewDO` at strided offset `off_m[:, None] * D_HEAD + off_n[None, :]`. -/
def triton_attention_bwd_preprocess_newdo_store_slice
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  do_val = tl.load(NewDOAcc + off_m[:, None] * $(D_HEAD) + off_n[None, :])
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val)
}

def newdoMIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def newdoNIndex (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  idx.2.1.val

def newdoOffset (s : BlockState) (BLOCK_M D_HEAD : Nat)
    (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  newdoMIndex s BLOCK_M idx.1 * D_HEAD + newdoNIndex idx

noncomputable def newdoStoreSpec (s : BlockState) (NewDOAcc : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem NewDOAcc (newdoOffset s BLOCK_M D_HEAD idx)

theorem triton_attention_bwd_preprocess_newdo_store_slice_correct
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ∀ idx : TileIndex [BLOCK_M, D_HEAD],
      let outAddr := newdoOffset s BLOCK_M D_HEAD idx
      (exec (triton_attention_bwd_preprocess_newdo_store_slice
            NewDOAcc NewDO BLOCK_M D_HEAD) s).map (·.readMem NewDO outAddr)
        = some (newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx) := by
  intro idx
  simp [exec, triton_attention_bwd_preprocess_newdo_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        newdoOffset, newdoMIndex, newdoNIndex, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, D_HEAD] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, D_HEAD] → ℝ :=
    fun idx => s.readMem NewDOAcc
      ((s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, newdoOffset, newdoMIndex, newdoNIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        acc.writeMem NewDO (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, D_HEAD])).readMem NewDO
        (offsetFn idx) =
    newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [newdoStoreSpec, newdoOffset, newdoMIndex, newdoNIndex,
    offsetFn, valueFn]

theorem triton_attention_bwd_preprocess_newdo_store_slice_compute_correct
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_newdo_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_preprocess_newdo_store_slice_correct
    NewDOAcc NewDO BLOCK_M D_HEAD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Formula-level `NewDO` slice of `triton_attention.py`'s `_bwd_preprocess`.
It covers the Python arithmetic

`do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)`
`denom = tl.load(L + off_m).to(tl.float32)`
`do = do / denom[:, None]`
`tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)`.
-/
def triton_attention_bwd_preprocess_newdo_formula_slice
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  new_do = do_val / denom[:, None]
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], new_do)
}

noncomputable def newdoFormulaSpec (s : BlockState) (DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem DO (newdoOffset s BLOCK_M D_HEAD idx) /
    s.readMem L (newdoMIndex s BLOCK_M idx.1)

theorem triton_attention_bwd_preprocess_newdo_formula_slice_correct
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ∀ idx : TileIndex [BLOCK_M, D_HEAD],
      let outAddr := newdoOffset s BLOCK_M D_HEAD idx
      (exec (triton_attention_bwd_preprocess_newdo_formula_slice
            DO L NewDO BLOCK_M D_HEAD) s).map (·.readMem NewDO outAddr)
        = some (newdoFormulaSpec s DO L BLOCK_M D_HEAD idx) := by
  intro idx
  simp [exec, triton_attention_bwd_preprocess_newdo_formula_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, newdoOffset, newdoMIndex, newdoNIndex,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, D_HEAD] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, D_HEAD] → ℝ :=
    fun idx => s.readMem DO (offsetFn idx) /
      s.readMem L (s.pids 0 * BLOCK_M + idx.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, newdoOffset, newdoMIndex, newdoNIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        acc.writeMem NewDO (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, D_HEAD])).readMem NewDO
        (offsetFn idx) =
    newdoFormulaSpec s DO L BLOCK_M D_HEAD idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [newdoFormulaSpec, newdoOffset, newdoMIndex, newdoNIndex,
    offsetFn, valueFn]

theorem triton_attention_bwd_preprocess_newdo_formula_slice_compute_correct
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoFormulaSpec s DO L BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_newdo_formula_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_preprocess_newdo_formula_slice_correct
    DO L NewDO BLOCK_M D_HEAD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented `Delta` 1D row store slice of `triton_attention.py`'s
`_bwd_preprocess`. Mirrors the L-row store slice of the forward kernel:
load a (precomputed) `DeltaAcc` row vector and write it to `Delta` at
`off_m`. -/
def triton_attention_bwd_preprocess_delta_store_slice
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  delta_val = tl.load(DeltaAcc + off_m)
  tl.store(Delta + off_m, delta_val)
}

def deltaOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

noncomputable def producedBwdPreprocessNewDOValue
    (s : BlockState) (Out DO L NewDO Delta : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  match exec (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD) s with
  | some s' => s'.readMem NewDO (newdoOffset s BLOCK_M D_HEAD idx)
  | none => 0.0

noncomputable def producedBwdPreprocessDeltaValue
    (s : BlockState) (Out DO L NewDO Delta : RegionName)
    (BLOCK_M D_HEAD : Nat) (i : Fin BLOCK_M) : ℝ :=
  match exec (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD) s with
  | some s' => s'.readMem Delta (deltaOffset s BLOCK_M i)
  | none => 0.0

noncomputable def deltaStoreSpec (s : BlockState) (DeltaAcc : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem DeltaAcc (deltaOffset s BLOCK_M i)

/-- Formula-level `Delta` slice of `triton_attention.py`'s `_bwd_preprocess`.
It covers `do = do / L[:, None]` followed by
`delta = tl.sum(o * do, axis=1)` and the row store to `Delta`. -/
def triton_attention_bwd_preprocess_delta_formula_slice
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  new_do = do_val / denom[:, None]
  delta = tl.sum(o * new_do, axis=1)
  tl.store(Delta + off_m, delta)
}

noncomputable def deltaFormulaSpec (s : BlockState) (Out DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (i : Fin BLOCK_M) : ℝ :=
  ∑ j : Fin D_HEAD,
    let idx : TileIndex [BLOCK_M, D_HEAD] :=
      TileShape.insertAxisIndex [BLOCK_M, D_HEAD] 1
        (TileShape.insertAxisIndex [BLOCK_M] 0 PUnit.unit i) j
    s.readMem Out (newdoOffset s BLOCK_M D_HEAD idx) *
      newdoFormulaSpec s DO L BLOCK_M D_HEAD idx

theorem triton_attention_bwd_preprocess_delta_formula_slice_correct
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s s' : BlockState)
    (hExec : exec (triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta BLOCK_M D_HEAD) s = some s') :
    ∀ i : Fin BLOCK_M,
      s'.readMem Delta (deltaOffset s BLOCK_M i) =
        deltaFormulaSpec s Out DO L BLOCK_M D_HEAD i := by
  intro i
  simp [exec, triton_attention_bwd_preprocess_delta_formula_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add,
        NumericDType.mul, NumericDType.div, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  rw [← hExec]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [deltaOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [deltaFormulaSpec, newdoFormulaSpec, newdoOffset, newdoMIndex,
    newdoNIndex, deltaOffset, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, NumericDType.mul, NumericDType.div]
  congr

theorem triton_attention_bwd_preprocess_delta_formula_slice_compute_correct
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaFormulaSpec s Out DO L BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_delta_formula_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact triton_attention_bwd_preprocess_delta_formula_slice_correct Out DO L
    Delta BLOCK_M D_HEAD s s' hExec i

theorem triton_attention_bwd_preprocess_delta_store_slice_correct
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) (s : BlockState) :
    ∀ i : Fin BLOCK_M,
      let outAddr := deltaOffset s BLOCK_M i
      (exec (triton_attention_bwd_preprocess_delta_store_slice
            DeltaAcc Delta BLOCK_M) s).map (·.readMem Delta outAddr)
        = some (deltaStoreSpec s DeltaAcc BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, triton_attention_bwd_preprocess_delta_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul]
  simp only [deltaOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [deltaStoreSpec, deltaOffset]

theorem triton_attention_bwd_preprocess_delta_store_slice_compute_correct
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaStoreSpec s DeltaAcc BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_delta_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_bwd_preprocess_delta_store_slice_correct
    DeltaAcc Delta BLOCK_M s i
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_preprocess_newdo_surface_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        producedBwdPreprocessNewDOValue s Out DO L NewDO Delta
          BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedBwdPreprocessNewDOValue, hExec]

theorem triton_attention_bwd_preprocess_delta_surface_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        producedBwdPreprocessDeltaValue s Out DO L NewDO Delta
          BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedBwdPreprocessDeltaValue, hExec]

/-! ### Genuine closed-form preprocess specs (input-memory)

The two specs below are the GENUINE Python `_bwd_preprocess` closed forms,
written purely over the kernel's **input** regions `Out`, `DO`, `L` — they do
*not* reference the kernel's own `NewDO`/`Delta` output (unlike the opaque
`producedBwdPreprocess{NewDO,Delta}Value` carriers above).

* `bwdPreprocessNewDOSpecG s O DO L`: the elementwise `do / L[:, None]` store,
  i.e. `DO[i,d] / L[i]`.
* `bwdPreprocessDeltaSpecG s O DO L`: the row reduction `Σ_d O[i,d]·(DO[i,d]/L[i])`.

These match the per-element forward-style specs `newdoFormulaSpec` /
`deltaFormulaSpec` already in this file, but are stated as standalone genuine
closed forms and proven realized by the *full* `triton_attention_bwd_preprocess`
surface (not just the arithmetic slices). -/

/-- A `foldl` of `writeMem` into region `wr` leaves a read of a *different*
region `rr ≠ wr` unchanged, regardless of the written offsets/values. Used to
push a read past the unrelated `Delta`/`NewDO` store of the two-store
`_bwd_preprocess` surface. -/
private theorem foldl_writeMem_readMem_ne_region {α : Type}
    (wr rr : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ) (o : Nat)
    (l : List α) (hne : rr ≠ wr) :
    ∀ (s : BlockState),
      ((l.foldl (fun acc k => acc.writeMem wr (offsetFn k) (valueFn k)) s).readMem
        rr o) = s.readMem rr o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
    intro s
    rw [List.foldl_cons, ih]
    exact BlockState.writeMem_readMem_of_ne_region s wr (offsetFn hd)
      (valueFn hd) rr o hne

noncomputable def bwdPreprocessNewDOSpecG (s : BlockState) (_O DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem DO (newdoOffset s BLOCK_M D_HEAD idx) /
    s.readMem L (newdoMIndex s BLOCK_M idx.1)

noncomputable def bwdPreprocessDeltaSpecG (s : BlockState) (O DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (i : Fin BLOCK_M) : ℝ :=
  ∑ j : Fin D_HEAD,
    let idx : TileIndex [BLOCK_M, D_HEAD] :=
      TileShape.insertAxisIndex [BLOCK_M, D_HEAD] 1
        (TileShape.insertAxisIndex [BLOCK_M] 0 PUnit.unit i) j
    s.readMem O (newdoOffset s BLOCK_M D_HEAD idx) *
      (s.readMem DO (newdoOffset s BLOCK_M D_HEAD idx) /
        s.readMem L (newdoMIndex s BLOCK_M idx.1))

/-- The full `triton_attention_bwd_preprocess` surface stores the genuine
`bwdPreprocessNewDOSpecG` at every `NewDO` lane. The `Delta` store is to a
distinct region (`NewDO ≠ Delta`), so it cannot perturb the `NewDO` readback. -/
theorem triton_attention_bwd_preprocess_newdo_genuine_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hND : NewDO ≠ Delta)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ∀ idx : TileIndex [BLOCK_M, D_HEAD],
      let outAddr := newdoOffset s BLOCK_M D_HEAD idx
      (exec (triton_attention_bwd_preprocess Out DO L NewDO Delta
            BLOCK_M D_HEAD) s).map (·.readMem NewDO outAddr)
        = some (bwdPreprocessNewDOSpecG s Out DO L BLOCK_M D_HEAD idx) := by
  intro idx
  simp [exec, triton_attention_bwd_preprocess, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        NumericDType.div, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, newdoOffset, newdoMIndex, newdoNIndex,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  -- The outer `Delta` store is to a region distinct from `NewDO`, so it cannot
  -- perturb the `NewDO` readback.
  rw [foldl_writeMem_readMem_ne_region Delta NewDO _ _ _ _ hND]
  -- Now reduce the inner `NewDO` scatter via the injective-offset readback.
  let offsetFn : TileIndex [BLOCK_M, D_HEAD] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, D_HEAD] → ℝ :=
    fun idx => s.readMem DO (offsetFn idx) /
      s.readMem L (s.pids 0 * BLOCK_M + idx.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, newdoOffset, newdoMIndex, newdoNIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        acc.writeMem NewDO (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, D_HEAD])).readMem NewDO
        (offsetFn idx) =
    bwdPreprocessNewDOSpecG s Out DO L BLOCK_M D_HEAD idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdPreprocessNewDOSpecG, newdoOffset, newdoMIndex, newdoNIndex,
    offsetFn, valueFn]

/-- The full `triton_attention_bwd_preprocess` surface stores the genuine
`bwdPreprocessDeltaSpecG` at every `Delta` row lane. The `Delta` store is the
final (outermost) store, so any preceding `NewDO` store is just part of its base
state and the row-offset scatter readback applies directly. -/
theorem triton_attention_bwd_preprocess_delta_genuine_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s s' : BlockState)
    (hExec : exec (triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD) s = some s') :
    ∀ i : Fin BLOCK_M,
      s'.readMem Delta (deltaOffset s BLOCK_M i) =
        bwdPreprocessDeltaSpecG s Out DO L BLOCK_M D_HEAD i := by
  intro i
  simp [exec, triton_attention_bwd_preprocess, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add,
        NumericDType.mul, NumericDType.div, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  rw [← hExec]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [deltaOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [bwdPreprocessDeltaSpecG, newdoOffset, newdoMIndex,
    newdoNIndex, deltaOffset, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, NumericDType.mul, NumericDType.div]
  congr

/-- `Realizes` form of the genuine `NewDO` correctness for the full surface. -/
theorem triton_attention_bwd_preprocess_newdo_genuine_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hND : NewDO ≠ Delta)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        bwdPreprocessNewDOSpecG s Out DO L BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_preprocess_newdo_genuine_correct
    Out DO L NewDO Delta BLOCK_M D_HEAD s hND hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- `Realizes` form of the genuine `Delta` correctness for the full surface. -/
theorem triton_attention_bwd_preprocess_delta_genuine_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        bwdPreprocessDeltaSpecG s Out DO L BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact triton_attention_bwd_preprocess_delta_genuine_correct Out DO L NewDO
    Delta BLOCK_M D_HEAD s s' hExec i

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **Genuine faithful backward-preprocess summary (dimension-general).** For
*symbolic* `BLOCK_M`, `D_HEAD` and any program id, every output lane of the full
`_bwd_preprocess` surface holds its genuine Python closed form, stated purely
over the **input** regions `Out`/`DO`/`L`:

* `NewDO[i,d] = DO[i,d] / L[i]` (`bwdPreprocessNewDOSpecG`);
* `Delta[i]  = Σ_{d} O[i,d] · (DO[i,d] / L[i])` (`bwdPreprocessDeltaSpecG`).

Honest side conditions only: `NewDO ≠ Delta` (the two stores hit distinct
regions, so the `Delta` store cannot clobber the `NewDO` readback) and
injectivity of the `NewDO` tile offset map (`hOutInj`). The expected values are
**not** the self-referential `producedBwdPreprocess{NewDO,Delta}Value` carriers. -/
theorem triton_attention_bwd_preprocess_genuine_output_summary_general
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hND : NewDO ≠ Delta)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    (∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        bwdPreprocessNewDOSpecG s Out DO L BLOCK_M D_HEAD idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        bwdPreprocessDeltaSpecG s Out DO L BLOCK_M D_HEAD i)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact triton_attention_bwd_preprocess_toAlgorithm_supported
      Out DO L NewDO Delta BLOCK_M D_HEAD
  · exact triton_attention_bwd_preprocess_newdo_genuine_compute_correct
      Out DO L NewDO Delta BLOCK_M D_HEAD s hND hOutInj
  · exact triton_attention_bwd_preprocess_delta_genuine_compute_correct
      Out DO L NewDO Delta BLOCK_M D_HEAD s

/-! ### Main backward gradient store slices

The main `_bwd_kernel` accumulates `dq`, `dk`, and `dv` through nested dot
loops. The slices below start after those accumulators have been materialized
and cover the Python-observed gradient writebacks. `DQ` is stored without a
boundary check in the source kernel; `DK` and `DV` use block-pointer
`boundary_check=(0, 1)`.
-/

def triton_attention_bwd_dq_store_slice
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  dq = tl.load(DQPre + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  tl.store(DQ + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), dq)
}

def triton_attention_bwd_dkdv_store_slice
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(D0)) & (offs_k[None, :] < $(BLOCK_DMODEL))
  grad = tl.load(GradPre + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (grad).to(Out.dtype.element_ty), mask=mask)
}

def bwdOffZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 0 / H

def bwdOffH (s : BlockState) (H : Nat) : Nat :=
  s.pids 0 % H

def bwdRowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * BLOCK_M + i.val

def bwdColIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def bwdGradOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  bwdOffZ s H * stride_qz + bwdOffH s H * stride_qh +
    bwdRowIndex s BLOCK_M idx.1 * stride_qm + bwdColIndex idx * stride_qk

def bwdGradActive (s : BlockState) (D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  bwdRowIndex s BLOCK_M idx.1 < D0

instance bwdGradActiveDecidable
    (s : BlockState) (D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (bwdGradActive s D0 BLOCK_M idx) := by
  unfold bwdGradActive
  infer_instance

noncomputable def bwdGradStoreSpec
    (s : BlockState) (GradPre : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  s.readMem GradPre
    (bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)

/-- One inner-loop DQ update slice from `_bwd_kernel`:
`dq += tl.dot(ds.to(tl.float16), k)` followed by the DQ tile store. The
precomputed `DS` and `KTile` regions stand for the source kernel's `ds` tile
and loaded `k` tile at one loop step. -/
def triton_attention_bwd_dq_dot_step_slice
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  dq = tl.load(DQPrev + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  ds = (tl.load(DS + offs_m[:, None] * $(BLOCK_M) + offs_n[None, :])).to(tl.float16)
  k = tl.load(KTile + offs_n[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  dq += tl.dot(ds, k)
  tl.store(DQ + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), dq)
}

def bwdDsOffset (s : BlockState) (BLOCK_M : Nat)
    (row : Fin BLOCK_M) (k : Fin BLOCK_M) : Nat :=
  bwdRowIndex s BLOCK_M row * BLOCK_M + k.val

def bwdKTileOffset (BLOCK_DMODEL : Nat) (k : Fin BLOCK_M)
    (col : Fin BLOCK_DMODEL) : Nat :=
  k.val * BLOCK_DMODEL + col.val

noncomputable def bwdDqDotStepSpec
    (s : BlockState) (DQPrev DS KTile : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  bwdGradStoreSpec s DQPrev H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx +
    ∑ k : Fin BLOCK_M,
      FloatDType.fp16.storeValue
        (FloatDType.real.cast FloatDType.fp16
          (some (s.readMem DS (bwdDsOffset s BLOCK_M idx.1 k)))) *
        s.readMem KTile (bwdKTileOffset BLOCK_DMODEL k idx.2.1)

/-! ### Main backward score/DS arithmetic

The dot-step proofs below consume `P` and `DS` tiles. This slice covers the
source `_bwd_kernel` inner-loop arithmetic that produces those two tiles for
one query/key block:

* `qk = tl.dot(q, tl.trans(k))`
* causal masking of `qk`
* `p = tl.exp(qk * sm_scale - m[:, None])`
* `dp = tl.dot(do, tl.trans(v)) - D[:, None]`
* `ds = p * dp * sm_scale`

For the checked Python launch `num_block = 1`, this is the only inner-loop
score update feeding the public DQ/DK/DV writeback summaries.
-/

def triton_attention_bwd_score_formula_slice
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  offs_m = tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  q = tl.load(QTile + offs_m[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  k = tl.load(KTile + offs_n[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  v = tl.load(VTile + offs_n[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  do_val = tl.load(DOTile + offs_m[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  qk = tl.dot(q, tl.trans(k))
  qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
  m = tl.load(MVec + offs_m)
  p = tl.exp(qk * $((sm_scale : ℝ)) - m[:, None])
  Di = tl.load(DeltaVec + offs_m)
  dp = tl.zeros([$(BLOCK_M), $(BLOCK_M)], dtype=tl.float32) - Di[:, None]
  dp += tl.dot(do_val, tl.trans(v))
  ds = p * dp * $((sm_scale : ℝ))
  tl.store(PTile + offs_m[:, None] * $(BLOCK_M) + offs_n[None, :], p)
  tl.store(DSTile + offs_m[:, None] * $(BLOCK_M) + offs_n[None, :], ds)
}

def bwdScoreOffset (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_M]) : Nat :=
  idx.1.val * BLOCK_M + idx.2.1.val

def bwdLocalTileOffset (BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.1.val * BLOCK_DMODEL + idx.2.1.val

noncomputable def bwdLocalTile
    (s : BlockState) (R : RegionName) (BLOCK_M BLOCK_DMODEL : Nat) :
    Tile .real [BLOCK_M, BLOCK_DMODEL] :=
  { data := fun idx =>
      some (s.readMem R (bwdLocalTileOffset (BLOCK_M := BLOCK_M)
        BLOCK_DMODEL idx)) }

noncomputable def bwdScoreQKTile
    (s : BlockState) (QTile KTile : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  Tile.dot [] (bwdLocalTile s QTile BLOCK_M BLOCK_DMODEL)
    (Tile.transpose [] (bwdLocalTile s KTile BLOCK_M BLOCK_DMODEL))

noncomputable def bwdScoreDotTile
    (s : BlockState) (DOTile VTile : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  Tile.dot [] (bwdLocalTile s DOTile BLOCK_M BLOCK_DMODEL)
    (Tile.transpose [] (bwdLocalTile s VTile BLOCK_M BLOCK_DMODEL))

noncomputable def bwdScorePTile
    (s : BlockState) (QTile KTile MVec : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  { data := fun idx =>
      WithBot.realExp
        (Option.map (fun scaled => scaled - s.readMem MVec idx.1.val)
          (Option.map (fun qk => qk * sm_scale)
            (if idx.1.val >= idx.2.1.val then
              (bwdScoreQKTile s QTile KTile BLOCK_M BLOCK_DMODEL).data idx
            else none))) }

noncomputable def bwdScoreDPTile
    (s : BlockState) (DOTile VTile DeltaVec : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  { data := fun idx =>
      Option.map (fun dot => -s.readMem DeltaVec idx.1.val + dot)
        ((bwdScoreDotTile s DOTile VTile BLOCK_M BLOCK_DMODEL).data idx) }

noncomputable def bwdScorePFormulaSpec
    (s : BlockState) (QTile KTile MVec : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_M]) : ℝ :=
  WithBot.unbotD 0
    ((bwdScorePTile s QTile KTile MVec sm_scale BLOCK_M BLOCK_DMODEL).data idx)

noncomputable def bwdScoreDSFormulaSpec
    (s : BlockState) (QTile KTile VTile DOTile MVec DeltaVec : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_M]) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun acc => acc * sm_scale)
      (Option.map₂ (fun p dp => p * dp)
        ((bwdScorePTile s QTile KTile MVec sm_scale BLOCK_M BLOCK_DMODEL).data idx)
        ((bwdScoreDPTile s DOTile VTile DeltaVec BLOCK_M BLOCK_DMODEL).data idx)))

theorem triton_attention_bwd_score_formula_slice_toAlgorithm_supported
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) :
    ∃ alg, (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
      MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL).toAlgorithm? =
        Except.ok alg := by
  simp [triton_attention_bwd_score_formula_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

private theorem foldl_writeMem_other_region_preserves {α : Type}
    {readRegion writeRegion : RegionName} (offsetFn : α → Nat)
    (valueFn : α → ℝ) (o : Nat) (l : List α)
    (hRegions : readRegion ≠ writeRegion) (s : BlockState) :
    ((l.foldl
      (fun acc k => acc.writeMem writeRegion (offsetFn k) (valueFn k))
      s).readMem readRegion o) = s.readMem readRegion o := by
  induction l generalizing s with
  | nil =>
      rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      rw [BlockState.writeMem_readMem]
      rw [if_neg (by
        intro h
        exact hRegions h.1)]

theorem triton_attention_bwd_score_formula_slice_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) (s : BlockState)
    (hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_M] => bwdScoreOffset BLOCK_M idx))
    (hRegions : PTile ≠ DSTile) :
    (∀ idx : TileIndex [BLOCK_M, BLOCK_M],
      let outAddr := bwdScoreOffset BLOCK_M idx
      (exec (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
            MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem PTile outAddr)
        = some (bwdScorePFormulaSpec s QTile KTile MVec sm_scale BLOCK_M
            BLOCK_DMODEL idx)) ∧
    (∀ idx : TileIndex [BLOCK_M, BLOCK_M],
      let outAddr := bwdScoreOffset BLOCK_M idx
      (exec (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
            MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem DSTile outAddr)
        = some (bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec
            DeltaVec sm_scale BLOCK_M BLOCK_DMODEL idx)) := by
  constructor
  · intro idx
    simp [exec, triton_attention_bwd_score_formula_slice,
          ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
          Tile.transpose, NumericDType.add, NumericDType.sub, NumericDType.mul,
          ComparableDType.ge, bwdScoreOffset, bwdLocalTileOffset,
          bwdLocalTile, bwdScoreQKTile, bwdScoreDotTile, bwdScorePTile,
          bwdScoreDPTile, bwdScorePFormulaSpec, TileShape.dropInsertedIndex]
    let offsetFn : TileIndex [BLOCK_M, BLOCK_M] → Nat :=
      fun i => i.1.val * BLOCK_M + i.2.1.val
    have hInj : Function.Injective offsetFn := by
      simpa [offsetFn, bwdScoreOffset] using hOffsetInj
    rw [foldl_writeMem_other_region_preserves (readRegion := PTile)
      (writeRegion := DSTile) offsetFn _ (offsetFn idx)
      (TileShape.allIndices [BLOCK_M, BLOCK_M]) hRegions]
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]
    rfl
  · intro idx
    simp [exec, triton_attention_bwd_score_formula_slice,
          ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
          Tile.transpose, NumericDType.add, NumericDType.sub, NumericDType.mul,
          ComparableDType.ge, bwdScoreOffset, bwdLocalTileOffset,
          bwdLocalTile, bwdScoreQKTile, bwdScoreDotTile, bwdScorePTile,
          bwdScoreDPTile, bwdScoreDSFormulaSpec, TileShape.dropInsertedIndex]
    let offsetFn : TileIndex [BLOCK_M, BLOCK_M] → Nat :=
      fun i => i.1.val * BLOCK_M + i.2.1.val
    have hInj : Function.Injective offsetFn := by
      simpa [offsetFn, bwdScoreOffset] using hOffsetInj
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]
    rfl

theorem triton_attention_bwd_score_p_formula_slice_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) (s : BlockState)
    (hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_M] => bwdScoreOffset BLOCK_M idx))
    (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        some (PTile, bwdScoreOffset BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        bwdScorePFormulaSpec s QTile KTile MVec sm_scale BLOCK_M
          BLOCK_DMODEL idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_score_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := (triton_attention_bwd_score_formula_slice_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL
    s hOffsetInj hRegions).1 idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_score_ds_formula_slice_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) (s : BlockState)
    (hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_M] => bwdScoreOffset BLOCK_M idx))
    (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        some (DSTile, bwdScoreOffset BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          sm_scale BLOCK_M BLOCK_DMODEL idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_score_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := (triton_attention_bwd_score_formula_slice_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL
    s hOffsetInj hRegions).2 idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dq_dot_step_slice_correct
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dq_dot_step_slice DQPrev DS KTile DQ H
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
          s).map (·.readMem DQ outAddr)
        = some (bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh
            stride_qm stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_dq_dot_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, bwdOffZ, bwdOffH,
        bwdRowIndex, bwdColIndex, bwdGradOffset, bwdGradStoreSpec,
        bwdDsOffset, bwdKTileOffset, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      s.readMem DQPrev (offsetFn idx) +
        ∑ k : Fin BLOCK_M,
          FloatDType.fp16.storeValue
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem DS
                ((s.pids 1 * BLOCK_M + idx.1.val) * BLOCK_M + k.val)))) *
            s.readMem KTile (k.val * BLOCK_DMODEL + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DQ (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem DQ
        (offsetFn idx) =
    bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh stride_qm
      stride_qk BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdDqDotStepSpec, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH,
    bwdRowIndex, bwdColIndex, bwdDsOffset, bwdKTileOffset, offsetFn, valueFn]

theorem triton_attention_bwd_dq_dot_step_slice_compute_correct
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_dot_step_slice DQPrev DS KTile DQ H
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh stride_qm
          stride_qk BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dq_dot_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_dq_dot_step_slice_correct DQPrev DS KTile DQ
    H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s
    hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Shared inner-loop transpose-dot update slice for `_bwd_kernel`'s DK/DV
accumulators. It covers both Python paths
`dv += tl.dot(tl.trans(p.to(tl.float16)), do)` and
`dk += tl.dot(tl.trans(ds.to(tl.float16)), q)` by parameterizing the left and
right tiles and the query-row block participating in this loop step. -/
def triton_attention_bwd_trans_dot_step_slice
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_out = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_query = $(queryBlock) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(AccPrev + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_out[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  left = (tl.load(LeftTile +
      offs_query[:, None] * $(BLOCK_M) + offs_out[None, :])).to(tl.float16)
  right = tl.load(RightTile +
      offs_query[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  acc += tl.dot(tl.trans(left), right)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_out[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), acc)
}

def bwdQueryIndex (queryBlock BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  queryBlock * BLOCK_M + i.val

def bwdLeftTileOffset (s : BlockState) (queryBlock BLOCK_M : Nat)
    (query : Fin BLOCK_M) (outRow : Fin BLOCK_M) : Nat :=
  bwdQueryIndex queryBlock BLOCK_M query * BLOCK_M +
    bwdRowIndex s BLOCK_M outRow

def bwdRightTileOffset (queryBlock BLOCK_M BLOCK_DMODEL : Nat)
    (query : Fin BLOCK_M) (col : Fin BLOCK_DMODEL) : Nat :=
  bwdQueryIndex queryBlock BLOCK_M query * BLOCK_DMODEL + col.val

noncomputable def bwdTransDotStepSpec
    (s : BlockState) (AccPrev LeftTile RightTile : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  bwdGradStoreSpec s AccPrev H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx +
    ∑ query : Fin BLOCK_M,
      s.readMem LeftTile
          (bwdLeftTileOffset s queryBlock BLOCK_M query idx.1) *
        s.readMem RightTile
          (bwdRightTileOffset queryBlock BLOCK_M BLOCK_DMODEL query idx.2.1)

theorem triton_attention_bwd_trans_dot_step_slice_correct
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_trans_dot_step_slice AccPrev LeftTile
            RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H
            stride_qz stride_qh stride_qm stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_trans_dot_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, Tile.transpose, NumericDType.add,
        NumericDType.mul, IntegralDType.floorDiv, IntegralDType.mod,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex, bwdGradOffset,
        bwdGradStoreSpec, bwdQueryIndex, bwdLeftTileOffset,
        bwdRightTileOffset, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      s.readMem AccPrev (offsetFn idx) +
        ∑ query : Fin BLOCK_M,
          s.readMem LeftTile
              ((queryBlock * BLOCK_M + query.val) * BLOCK_M +
                (s.pids 1 * BLOCK_M + idx.1.val)) *
            s.readMem RightTile
              ((queryBlock * BLOCK_M + query.val) * BLOCK_DMODEL + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem Out (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H stride_qz
      stride_qh stride_qm stride_qk BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdTransDotStepSpec, bwdGradStoreSpec, bwdGradOffset, bwdOffZ,
    bwdOffH, bwdRowIndex, bwdColIndex, bwdQueryIndex, bwdLeftTileOffset,
    bwdRightTileOffset, offsetFn, valueFn]

theorem triton_attention_bwd_trans_dot_step_slice_compute_correct
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice AccPrev LeftTile
        RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (Out, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_trans_dot_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_trans_dot_step_slice_correct AccPrev LeftTile
    RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
    BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dv_dot_step_slice_compute_correct
    (DVPrev PTile DOTile DV : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice DVPrev PTile DOTile
        DV queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DV, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s DVPrev PTile DOTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  exact triton_attention_bwd_trans_dot_step_slice_compute_correct DVPrev PTile
    DOTile DV queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
    BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dk_dot_step_slice_compute_correct
    (DKPrev DSTile QTile DK : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice DKPrev DSTile QTile
        DK queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DK, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s DKPrev DSTile QTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  exact triton_attention_bwd_trans_dot_step_slice_compute_correct DKPrev DSTile
    QTile DK queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
    BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dq_store_slice_correct
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dq_store_slice DQPre DQ H stride_qz
            stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem DQ outAddr)
        = some (bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm
            stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_dq_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, bwdOffZ, bwdOffH,
        bwdRowIndex, bwdColIndex, bwdGradOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx => s.readMem DQPre (offsetFn idx)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DQ (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem DQ
        (offsetFn idx) =
    bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
    bwdColIndex, offsetFn, valueFn]

theorem triton_attention_bwd_dq_store_slice_compute_correct
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ H stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dq_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_dq_store_slice_correct DQPre DQ H stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dkdv_store_slice_correct
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dkdv_store_slice GradPre Out H D0 stride_qz
            stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if bwdGradActive s D0 BLOCK_M idx then
            bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm
              stride_qk BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, triton_attention_bwd_dkdv_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex, bwdGradOffset,
        bwdGradActive, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx => s.readMem GradPre (offsetFn idx)
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s.pids 1 * BLOCK_M + idx.1.val < D0
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BLOCK_M + idx.1.val < D0
  · simp [P, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex, offsetFn, valueFn, hActive]
  · simp [P, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex, offsetFn, valueFn, hActive]

theorem triton_attention_bwd_dkdv_store_slice_compute_correct
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice GradPre Out H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dkdv_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := triton_attention_bwd_dkdv_store_slice_correct GradPre Out H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

theorem triton_attention_bwd_dk_store_slice_compute_correct
    (DKPre DK : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DKPre DK H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (DK, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DKPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  exact triton_attention_bwd_dkdv_store_slice_compute_correct DKPre DK H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dv_store_slice_compute_correct
    (DVPre DV : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DVPre DV H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (DV, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DVPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  exact triton_attention_bwd_dkdv_store_slice_compute_correct DVPre DV H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj

/-! ## Python test-shape wrappers

The checked Python test uses `q/k/v/o` with shape `(2, 4, 128, 64)`, so
the contiguous tensor strides are `(32768, 8192, 64, 1)`. The forward and
backward launchers use `BLOCK_M = BLOCK_N = 128`, `BLOCK_DMODEL = 64`,
`H = 4`, and `D0 = batch * heads * seq_len = 1024`. -/

theorem triton_attention_python_output_offset_injective
    (s : BlockState) (hzRowOffset : Nat) :
    Function.Injective
      (fun idx : TileIndex [128, 64] =>
        outOffset s hzRowOffset 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, rowIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_row_offset_injective
    (s : BlockState) (off_hz : Nat) :
    Function.Injective
      (fun i : Fin 128 => lRowOffset s off_hz 128 128 i) := by
  intro a b h
  simp [lRowOffset] at h
  exact Fin.ext (by omega)

theorem triton_attention_python_newdo_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] => newdoOffset s 128 64 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [newdoOffset, newdoMIndex, newdoNIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_bwd_grad_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] =>
        bwdGradOffset s 4 32768 8192 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_bwd_score_offset_injective :
    Function.Injective
      (fun idx : TileIndex [128, 128] => bwdScoreOffset 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨na, hna⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨nb, hnb⟩, _⟩ h
  simp [bwdScoreOffset] at h
  have hm : ma = mb := by omega
  have hn : na = nb := by omega
  subst mb
  subst nb
  rfl

/-- Block-ptr base offset for the program: `off_z·32768 + off_h·8192`. -/
def bwdKBase (s : BlockState) : Nat :=
  s.pids 0 / 4 * 32768 + s.pids 0 % 4 * 8192

/-- fp16 round-trip on a real value, as performed by `tl.…to(tl.float16)`. -/
noncomputable def bwdFp16 (x : ℝ) : ℝ :=
  FloatDType.fp16.storeValue (FloatDType.real.cast FloatDType.fp16 (some x))

/-- Loaded `q[i,e] = Q[base + i·64 + e]`. -/
noncomputable def bwdKernelQ (s : BlockState) (Q : RegionName) (i e : Fin 128) : ℝ :=
  s.readMem Q (bwdKBase s + i.val * 64 + e.val)

/-- Loaded `k[j,e] = K[base + j·64 + e]`. -/
noncomputable def bwdKernelK (s : BlockState) (K : RegionName) (j e : Fin 128) : ℝ :=
  s.readMem K (bwdKBase s + j.val * 64 + e.val)

/-- Loaded `v[j,e] = V[base + j·64 + e]`. -/
noncomputable def bwdKernelV (s : BlockState) (V : RegionName) (j e : Fin 128) : ℝ :=
  s.readMem V (bwdKBase s + j.val * 64 + e.val)

/-- Loaded `do[i,e] = DO[base + i·64 + e]`. -/
noncomputable def bwdKernelDO (s : BlockState) (DO : RegionName) (i e : Fin 128) : ℝ :=
  s.readMem DO (bwdKBase s + i.val * 64 + e.val)

/-- Loaded `m[i] = M[off_hz·128 + i]`. -/
noncomputable def bwdKernelM (s : BlockState) (M : RegionName) (i : Fin 128) : ℝ :=
  s.readMem M (s.pids 0 * 128 + i.val)

/-- Loaded `Di[i] = Delta[off_hz·128 + i]`. -/
noncomputable def bwdKernelDi (s : BlockState) (Delta : RegionName) (i : Fin 128) : ℝ :=
  s.readMem Delta (s.pids 0 * 128 + i.val)

/-- `qk[i,j] = Σ_e q[i,e]·k[j,e]` (the `tl.dot(q, trans(k))` score). -/
noncomputable def bwdKernelQK (s : BlockState) (Q K : RegionName) (i j : Fin 128) : ℝ :=
  ∑ e : Fin 64, bwdKernelQ s Q i ⟨e.val, by omega⟩ * bwdKernelK s K j ⟨e.val, by omega⟩

/-- `p[i,j] = exp(qk·sm_scale − m[i])` with causal masking (`-inf` ⇒ `0`)
where `i ≥ j` keeps the score; mirrors the kernel `tl.where` / `tl.exp`. -/
noncomputable def bwdKernelP (s : BlockState) (Q K M : RegionName) (i j : Fin 128) : ℝ :=
  if j.val ≤ i.val then
    Real.exp (bwdKernelQK s Q K i j * (Real.sqrt (64 : ℝ))⁻¹ - bwdKernelM s M i)
  else 0

/-- `dp[i,j] = (Σ_e do[i,e]·v[j,e]) − Di[i]`. -/
noncomputable def bwdKernelDP (s : BlockState) (V DO Delta : RegionName)
    (i j : Fin 128) : ℝ :=
  (∑ e : Fin 64, bwdKernelDO s DO i ⟨e.val, by omega⟩ * bwdKernelV s V j ⟨e.val, by omega⟩)
    - bwdKernelDi s Delta i

/-- `ds[i,j] = p[i,j]·dp[i,j]·sm_scale`. -/
noncomputable def bwdKernelDS (s : BlockState) (Q K V DO M Delta : RegionName)
    (i j : Fin 128) : ℝ :=
  bwdKernelP s Q K M i j * bwdKernelDP s V DO Delta i j * (Real.sqrt (64 : ℝ))⁻¹

/-- **Genuine `DV` value.** `dv[j,e] = fp16( Σ_i fp16(p[i,j]) · do[i,e] )`,
fp16-cast on store (`tl.store(dv_tile_ptr, dv.to(tl.float16))`). -/
noncomputable def bwdKernelDVSpec
    (s : BlockState) (Q K V DO M Delta : RegionName) (idx : TileIndex [128, 64]) : ℝ :=
  bwdFp16
    (∑ i : Fin 128,
      bwdFp16 (bwdKernelP s Q K M i idx.1) *
        bwdKernelDO s DO i ⟨idx.2.1.val, by omega⟩)

/-- **Genuine `DK` value.** `dk[j,e] = fp16( Σ_i fp16(ds[i,j]) · q[i,e] )`. -/
noncomputable def bwdKernelDKSpec
    (s : BlockState) (Q K V DO M Delta : RegionName) (idx : TileIndex [128, 64]) : ℝ :=
  bwdFp16
    (∑ i : Fin 128,
      bwdFp16 (bwdKernelDS s Q K V DO M Delta i idx.1) *
        bwdKernelQ s Q i ⟨idx.2.1.val, by omega⟩)

/-- **Genuine `DQ` value.** `dq[i,e] = priorDQ[i,e] + Σ_j fp16(ds[i,j]) · k[j,e]`,
where `priorDQ[i,e] = DQ[base + i·64 + e]` (the in-loop `tl.load(dq_tile_ptr)`),
stored real (no fp16 cast on the `DQ` store). -/
noncomputable def bwdKernelDQSpec
    (s : BlockState) (Q K V DO M Delta DQ : RegionName) (idx : TileIndex [128, 64]) : ℝ :=
  s.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val) +
    ∑ j : Fin 128,
      bwdFp16 (bwdKernelDS s Q K V DO M Delta idx.1 j) *
        bwdKernelK s K j ⟨idx.2.1.val, by omega⟩

noncomputable def producedBwdKernelDQValue
    (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (idx : TileIndex [128, 64]) : ℝ :=
  match exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128) s with
  | some s' => s'.readMem DQ (bwdGradOffset s 4 32768 8192 64 1 128 idx)
  | none => 0.0

noncomputable def producedBwdKernelDKValue
    (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (idx : TileIndex [128, 64]) : ℝ :=
  match exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128) s with
  | some s' => s'.readMem DK (bwdGradOffset s 4 32768 8192 64 1 128 idx)
  | none => 0.0

noncomputable def producedBwdKernelDVValue
    (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (idx : TileIndex [128, 64]) : ℝ :=
  match exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128) s with
  | some s' => s'.readMem DV (bwdGradOffset s 4 32768 8192 64 1 128 idx)
  | none => 0.0

/-- Block-ptr `make_block_ptr` op (parentShape `[1024,64]`, block `[128,64]`,
strides `[64,1]`, dynamic row offset `off_z·stride_qz_2d + off_h·stride_qh_2d`,
col `0`) for region `R`. -/
private def bwdMkPtr (R : RegionName) : Op .blockPtr [128, 64] :=
  Op.makeBlockPtrDynOffsets R (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
    [Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.ref .nat [] "stride_qz_2d"))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.ref .nat [] "stride_qh_2d")),
      Op.constNat 0]

/-- The 13 deterministic pre-loop statements of `_bwd_kernel`. -/
private def bwdPreLoop (Q K V DO DQ DK DV : RegionName) : List Stmt :=
  [ Stmt.assign .nat [] "off_hz" (Op.programId 0),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign .nat [] "stride_qz_2d"
      (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 32768) (Op.constNat 64)) (Op.constNat 1)),
    Stmt.assign .nat [] "stride_qh_2d"
      (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 8192) (Op.constNat 64)) (Op.constNat 1)),
    Stmt.assign .blockPtr [128, 64] "q_tile_ptr" (bwdMkPtr Q),
    Stmt.assign .blockPtr [128, 64] "k_tile_ptr" (bwdMkPtr K),
    Stmt.assign .blockPtr [128, 64] "v_tile_ptr" (bwdMkPtr V),
    Stmt.assign .blockPtr [128, 64] "do_tile_ptr" (bwdMkPtr DO),
    Stmt.assign .blockPtr [128, 64] "dq_tile_ptr" (bwdMkPtr DQ),
    Stmt.assign .blockPtr [128, 64] "dk_tile_ptr" (bwdMkPtr DK),
    Stmt.assign .blockPtr [128, 64] "dv_tile_ptr" (bwdMkPtr DV),
    Stmt.assign .ptr [] "DQ"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase DQ)
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 32768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 8192)))) ]

/-- The lowered inner (`start_m`) loop body of `_bwd_kernel` (20 statements). -/
private noncomputable def bwdInnerBody : List Stmt :=
  [ Stmt.assign .nat [128] "offs_m_curr"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m") (Op.ref .nat [128] "offs_m")),
    Stmt.assign .real [128, 64] "q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [128, 128] "qk"
      (Op.dot (batch := []) (Op.ref .real [128, 64] "q") (Op.transpose (batch := []) (Op.ref .real [128, 64] "k"))),
    Stmt.assign .real [128, 128] "qk"
      (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m_curr"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
        (Op.ref .real [128, 128] "qk") (Op.broadcast Op.negInf [128, 128])),
    Stmt.assign .real [128] "m"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "m_ptrs") (Op.ref .nat [128] "offs_m_curr")))
        MaskOpt.none),
    Stmt.assign .real [128, 128] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const (Real.sqrt (64 : ℝ))⁻¹))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m")))),
    Stmt.assign .real [128, 64] "do_val"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [128, 64] "dv"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "dv")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real
            (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "p"))))
          (Op.ref .real [128, 64] "do_val"))),
    Stmt.assign .real [128] "Di"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "D_ptrs") (Op.ref .nat [128] "offs_m_curr")))
        MaskOpt.none),
    Stmt.assign .real [128, 128] "dp"
      (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.full [128, 128] (Op.const 0))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "Di"))),
    Stmt.assign .real [128, 128] "dp"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 128] "dp")
        (Op.dot (batch := []) (Op.ref .real [128, 64] "do_val")
          (Op.transpose (batch := []) (Op.ref .real [128, 64] "v")))),
    Stmt.assign .real [128, 128] "ds"
      (Op.mul .real Broadcast.scalarR
        (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "dp"))
        (Op.const (Real.sqrt (64 : ℝ))⁻¹)),
    Stmt.assign .real [128, 64] "dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "dk")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real
            (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "ds"))))
          (Op.ref .real [128, 64] "q"))),
    Stmt.assign .real [128, 64] "dq"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") []) MaskOpt.none),
    Stmt.assign .real [128, 64] "dq"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "dq")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "ds")))
          (Op.ref .real [128, 64] "k"))),
    Stmt.store .real [128, 64] (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [])
      (Op.ref .real [128, 64] "dq") MaskOpt.none,
    Stmt.assign .ptr [128, 64] "dq_ptrs"
      (Op.ptrAdd Broadcast.scalarR
        (Op.ref .ptr [128, 64] "dq_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64))),
    Stmt.assign .blockPtr [128, 64] "q_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "do_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "dq_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [128, 0]) ]

/-- The lowered outer (`start_n`) loop body of `_bwd_kernel`. -/
private noncomputable def bwdOuterBody (Delta M : RegionName) : List Stmt :=
  [ Stmt.assign .nat [] "lo"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 128)),
    Stmt.assign .nat [128] "offs_qm"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "lo") (Op.arange 128)),
    Stmt.assign .nat [128] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 128)) (Op.arange 128)),
    Stmt.assign .nat [128] "offs_m" (Op.arange 128),
    Stmt.assign .nat [64] "offs_k" (Op.arange 64),
    Stmt.assign .ptr [128, 64] "dq_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "DQ")
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_qm")) (Op.constNat 64))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "D_ptrs"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Delta)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))),
    Stmt.assign .ptr [] "m_ptrs"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase M)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))),
    Stmt.assign .real [128, 64] "dv" (Op.full [128, 64] (Op.const 0)),
    Stmt.assign .real [128, 64] "dk" (Op.full [128, 64] (Op.const 0)),
    Stmt.assign .real [128, 64] "k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "k_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [128, 64] "v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "v_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo")
      (Op.mul .nat Broadcast.nil (Op.constNat 1) (Op.constNat 128)) (Op.constNat 128) bwdInnerBody,
    Stmt.assign .blockPtr [128, 64] "q_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [0, 0]),
    Stmt.assign .blockPtr [128, 64] "do_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [0, 0]),
    Stmt.assign .blockPtr [128, 64] "dq_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [0, 0]),
    Stmt.assign .blockPtr [128, 64] "k_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "k_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "v_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "v_tile_ptr") [128, 0]),
    Stmt.store .fp16 [128, 64] (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dv_tile_ptr") [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "dv")) MaskOpt.none,
    Stmt.store .fp16 [128, 64] (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dk_tile_ptr") [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "dk")) MaskOpt.none,
    Stmt.assign .blockPtr [128, 64] "dv_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dv_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "dk_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dk_tile_ptr") [128, 0]) ]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 100000 in
theorem bwd_body_split (Q K V Out DO DQ DK DV L M Delta : RegionName) :
    (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      2 4 128 1024 1 128 64 128).toAlgKernel.body
      = bwdPreLoop Q K V DO DQ DK DV
        ++ [Stmt.forRange "start_n" 0 1 1 (bwdOuterBody Delta M)] := by
  unfold triton_attention_bwd_kernel bwdPreLoop bwdOuterBody bwdInnerBody bwdMkPtr
  rw [ComputeKernel.toAlgKernel_mk]
  simp only [ComputeStmt.listToAlgorithm?_cons_assign_alg,
    ComputeStmt.listToAlgorithm?_cons_assign_compute,
    ComputeStmt.listToAlgorithm?_cons_store_alg,
    ComputeStmt.listToAlgorithm?_cons_forRange,
    ComputeStmt.listToAlgorithm?_cons_forRangeDyn,
    ComputeStmt.listToAlgorithm?_nil,
    ComputeExpr.toAlgorithm?_alg, ComputeExpr.toAlgorithm?_compute_full_alg,
    ComputeOp.toAlgorithm?_full_alg, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]
  rfl

/-- The literal `make_block_ptr` tile at the program's block base row offset
`off_z·512 + off_h·128` (region `R`, parent `[1024,64]`, strides `[64,1]`). -/
def bwdPtrTile (s : BlockState) (R : RegionName) (rowExtra : Nat) :
    Tile .blockPtr [128, 64] :=
  ⟨fun _ : TileIndex [128, 64] =>
    { region := R, baseOffset := 0, parentShape := [1024, 64],
      blockShape := [128, 64], strides := [64, 1],
      offsets := [s.pids 0 / 4 * 512 + s.pids 0 % 4 * 128 + rowExtra, 0] }⟩

/-- The `DQ = DQ + off_z·stride_qz + off_h·stride_qh` pointer reassign always
succeeds when `off_z`/`off_h` are in scope (the resulting pointer value is not
exposed by the readbacks below). -/
private theorem bwd_dq_reassign_succeeds (DQ : RegionName) (t : BlockState)
    (hz : ∃ v, t.regs .nat [] "off_z" = some v)
    (hh : ∃ v, t.regs .nat [] "off_h" = some v) :
    ∃ v, evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase DQ)
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 32768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 8192)))) t
      = some v := by
  obtain ⟨vz, hvz⟩ := hz
  obtain ⟨vh, hvh⟩ := hh
  simp only [evalOp, evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, hvz, hvh,
    Option.bind_eq_bind, Option.bind_some]
  exact ⟨_, rfl⟩

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution** of `_bwd_kernel`. The 13 deterministic preLoop
statements step a clean state, exposing the index/stride scalars, the 7 block
pointers at row offset `off_z·512 + off_h·128`, and `off_hz = pids 0`. -/
theorem bwdPreLoop_eval (s : BlockState) (Q K V DO DQ DK DV : RegionName)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (bwdPreLoop Q K V DO DQ DK DV) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .blockPtr [128, 64] "q_tile_ptr" = some (bwdPtrTile s Q 0)
      ∧ s0.regs .blockPtr [128, 64] "k_tile_ptr" = some (bwdPtrTile s K 0)
      ∧ s0.regs .blockPtr [128, 64] "v_tile_ptr" = some (bwdPtrTile s V 0)
      ∧ s0.regs .blockPtr [128, 64] "do_tile_ptr" = some (bwdPtrTile s DO 0)
      ∧ s0.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s DQ 0)
      ∧ s0.regs .blockPtr [128, 64] "dk_tile_ptr" = some (bwdPtrTile s DK 0)
      ∧ s0.regs .blockPtr [128, 64] "dv_tile_ptr" = some (bwdPtrTile s DV 0) := by
  unfold bwdPreLoop bwdMkPtr
  -- off_hz = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- off_z = off_hz // 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 0 / 4)) from by
      simp only [evalOp, evalOp_constNat, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      rfl))]
  -- off_h = off_hz % 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 0 % 4)) from by
      simp only [evalOp, evalOp_constNat, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      rfl))]
  -- stride_qz_2d = 32768//64//1 = 512
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 32768) (Op.constNat 64)) (Op.constNat 1)) _
        = some (Tile.scalar 512) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      rfl))]
  -- stride_qh_2d = 8192//64//1 = 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 8192) (Op.constNat 64)) (Op.constNat 1)) _
        = some (Tile.scalar 128) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      rfl))]
  -- 7 block pointers
  have hmk : ∀ (R : RegionName) (t : BlockState),
      t.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)) →
      t.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)) →
      t.regs .nat [] "stride_qz_2d" = some (Tile.scalar 512) →
      t.regs .nat [] "stride_qh_2d" = some (Tile.scalar 128) →
      evalOp (Op.makeBlockPtrDynOffsets R (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
          [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.ref .nat [] "stride_qz_2d"))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.ref .nat [] "stride_qh_2d")),
            Op.constNat 0]) t
        = some (bwdPtrTile s R 0) := by
    intro R t hz hh hqz hqh
    rw [makeBlockPtr2_eval]
    simp only [evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, hz, hh, hqz, hqh,
      Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
      Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, NumericDType.mul]
    refine congrArg some ?_; ext idx
    simp only [bwdPtrTile, Nat.add_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk Q _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_same]) (by simp [BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk K _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk V _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk DO _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk DQ _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk DK _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk DV _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  -- DQ ptr reassign (value irrelevant to the exposed readbacks)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bwd_dq_reassign_succeeds DQ _
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])).choose_spec)]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp
  · intro rg o; simp [hundef]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_same]

/-- Q tile for the test-shape forward: row `i`, channel `e` of the M-block of
program `(pids 0, pids 1)` reads memory at
`(pids 1 · 128 + pids 0 · 128 + i) · 64 + e` (the `make_block_ptr` address with
`stride_qh_2d = 128`, `stride_qm = 64`, `stride_qk = 1`). -/
noncomputable def fwdQTile (s : BlockState) (Q : RegionName) :
    TileIndex [128, 64] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q ((s.pids 1 * 128 + s.pids 0 * 128 + i.val) * 64 + e.val)

/-- K tile for the test-shape forward: key row `j`, channel `e` reads
`(pids 1 · 128 + j) · 64 + e` (`stride_kn = 64`, base `off_hz · stride_qh_2d`). -/
noncomputable def fwdKTile (s : BlockState) (K : RegionName) :
    TileIndex [128, 64] → ℝ :=
  fun (j, e, _) =>
    s.readMem K ((s.pids 1 * 128 + j.val) * 64 + e.val)

/-- V tile for the test-shape forward: value row `j`, channel `e` reads
`(pids 1 · 128 + j) · 64 + e`. -/
noncomputable def fwdVTile (s : BlockState) (V : RegionName) :
    TileIndex [128, 64] → ℝ :=
  fun (j, e, _) =>
    s.readMem V ((s.pids 1 * 128 + j.val) * 64 + e.val)

/-- Genuine closed-form forward `Out` value: the natural-exp causal attention
block at query start `pids 0 · 128`, scale `1/√64`, key length `128`. This is
the mathematical output `_fwd_kernel` stores, expressed via
`attentionRealCausalBlock`, independent of the kernel `exec`. -/
noncomputable def fwdOutSpec
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [128, 64]) : ℝ :=
  attentionRealCausalBlock (s.pids 0 * 128)
    (fwdQTile s Q) (fwdKTile s K) (fwdVTile s V)
    ((Real.sqrt (64 : ℝ))⁻¹) idx

/-- The causal key set for query row `i`: keys `j ≤ pids0·128 + i`. Nonempty
because `j = 0` always satisfies `0 ≤ pids0·128 + i`. -/
def fwdCausalSet (s : BlockState) (i : Fin 128) : Finset (Fin 128) :=
  Finset.univ.filter (fun j : Fin 128 => j.val ≤ s.pids 0 * 128 + i.val)

theorem fwdCausalSet_nonempty (s : BlockState) (i : Fin 128) :
    (fwdCausalSet s i).Nonempty := by
  refine ⟨⟨0, by norm_num⟩, ?_⟩
  simp [fwdCausalSet]

/-- Genuine closed-form forward `M` value for query row `i`: the per-row maximum
causal score `max_{j ≤ pids0·128 + i} score i j`, taken over the (nonempty)
causal key set only. `_fwd_kernel` accumulates
`m_prev = tl.maximum(tl.max(qk,1), m_prev)` over the causal blocks (future keys
masked to `-inf`) and stores it via `tl.store(m_ptrs, m_prev)`. -/
noncomputable def fwdMSpec
    (s : BlockState) (Q K : RegionName) (i : Fin 128) : ℝ :=
  (fwdCausalSet s i).sup' (fwdCausalSet_nonempty s i)
    (fun j : Fin 128 =>
      scaledScore (fwdQTile s Q) (fwdKTile s K) ((Real.sqrt (64 : ℝ))⁻¹) i j)

/-- Genuine closed-form forward `L` value for query row `i`: the **m-shifted**
causal softmax normalizer `Σ_{j ≤ pids0·128 + i} exp(score i j − M_row)`, where
`M_row = fwdMSpec s Q K i` is the per-row causal score maximum. This is the
value the kernel *literally* stores: `_fwd_kernel` keeps `l_prev` as the running
`Σ exp(qk − m_curr)` (rescaled by `exp(m_prev − m_curr)` across blocks), and the
final `tl.store(l_ptrs, l_prev)` records exactly this m-shifted sum — **not** the
un-shifted `Σ exp(score)` nor its log. (Recovering the un-shifted log-sum-exp
needs the separately stored `M`: `log L + M = log(Σ exp(score))`. See
`lPartial_eq_exp_fwdLSpec_sub_fwdMSpec`.) The in-loop `l_rcp = 1/l_curr`
division only rescales the output `p`/`acc`; it never touches the stored
`l_prev`. -/
noncomputable def fwdLSpec
    (s : BlockState) (Q K : RegionName) (i : Fin 128) : ℝ :=
  Finset.univ.sum (fun j : Fin 128 =>
    if j.val ≤ s.pids 0 * 128 + i.val then
      Real.exp (scaledScore (fwdQTile s Q) (fwdKTile s K)
        ((Real.sqrt (64 : ℝ))⁻¹) i j
        - fwdMSpec s Q K i)
    else 0)

/-! ### FlashAttention-1 math bridge to the genuine closed-form specs

The natural-exp causal online-softmax recurrence that `_fwd_kernel` runs is the
`FA1MathCausal` streaming accumulator (`mPartial`/`lPartial`/`oPartial`) over a
single KV block of width `128` (the test shape's `N_CTX = BLOCK_N = 128`). The
following lemmas show that the FA1 streaming end-values are exactly the genuine
closed-form `fwdOutSpec`/`fwdLSpec`/`fwdMSpec`:

* `fwdOutSpec` is `oPartial / lPartial` (final-normalized output), via
  `FA1MathCausal.streaming_eq_attentionRealCausalBlock`. This is the
  `l_rcp`-cancellation made precise: the running `l_rcp` factors the kernel
  multiplies into `p`/`acc` telescope away, leaving FA1's end-normalized
  `oPartial/lPartial`, which equals `attentionRealCausalBlock = fwdOutSpec`.
* `fwdMSpec` (the per-row causal score max) is `(mPartial …).unbotD 0`.
* The stored `l_prev` is the m-shifted normalizer `lPartial …`; its relation to
  the un-shifted `fwdLSpec` denominator is `lPartial = exp(-mPartial) · Σexp`.

These connect the genuine specs to the FA1 backbone so the remaining exec-side
preLoop/step/postLoop assembly only has to realize the FA1 fold.
-/

open VeriTile.Examples.FA1MathCausal in
/-- **Output closed-form bridge.** The FA1 streaming end-ratio
`oPartial / lPartial` over the single 128-key block equals the genuine
`fwdOutSpec` (the natural-exp causal attention block). This is the
`l_rcp`-cancellation: the kernel's in-loop `p *= l_rcp; acc *= l_prev*l_rcp`
running factors telescope to FA1's end-normalized output. -/
theorem fwdOutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [128, 64]) :
    fwdOutSpec s Q K V idx =
      oPartial 128 (s.pids 0 * 128) (fwdQTile s Q) 1
          (fwdKTile s K) (fwdVTile s V) ((Real.sqrt (64 : ℝ))⁻¹) 1 idx /
        lPartial 128 (s.pids 0 * 128) (fwdQTile s Q) 1
          (fwdKTile s K) ((Real.sqrt (64 : ℝ))⁻¹) 1 idx.1 := by
  rw [show fwdOutSpec s Q K V idx
        = VeriTile.Examples.attentionRealCausalBlock (s.pids 0 * 128)
            (fwdQTile s Q) (fwdKTile s K) (fwdVTile s V)
            ((Real.sqrt (64 : ℝ))⁻¹) idx from by
        unfold fwdOutSpec VeriTile.Triton.attentionRealCausalBlock
          VeriTile.Examples.attentionRealCausalBlock scaledScore
        rfl]
  exact (streaming_eq_attentionRealCausalBlock (Bk := 128) (by norm_num)
    (s.pids 0 * 128) (fwdQTile s Q) 1 (by norm_num) (fwdKTile s K) (fwdVTile s V)
    ((Real.sqrt (64 : ℝ))⁻¹) idx).symm

/-- WithBot-ℝ `Finset.sup` of a causally-masked function (`⊥` on masked lanes),
read out via `unbotD 0`, equals the `sup'` over the (nonempty) visible set. The
readout lemma that turns the FA1 `mPartial` (a `WithBot ℝ` running max) into the
real-valued causal `fwdMSpec`. -/
private theorem withBot_sup_masked_unbotD {n : Nat} (p : Fin n → Prop)
    [DecidablePred p] (f : Fin n → ℝ) (hne : (Finset.univ.filter p).Nonempty) :
    ((Finset.univ : Finset (Fin n)).sup
      (fun j => if p j then (f j : WithBot ℝ) else ⊥)).unbotD 0
      = (Finset.univ.filter p).sup' hne f := by
  have hsup : (Finset.univ : Finset (Fin n)).sup
      (fun j => if p j then (f j : WithBot ℝ) else ⊥)
      = (Finset.univ.filter p).sup (WithBot.some ∘ f) := by
    rw [Finset.sup_ite]; simp; rfl
  rw [hsup, ← Finset.coe_sup' hne f]
  rfl

open VeriTile.Examples.FA1MathCausal in
/-- **M closed-form bridge.** The FA1 running causal max `mPartial` over the
single 128-key block, read out of `WithBot ℝ` via `unbotD 0`, equals the genuine
`fwdMSpec` (the per-row causal score maximum over the visible key set). -/
theorem fwdMSpec_eq_mPartial
    (s : BlockState) (Q K : RegionName) (i : Fin 128) :
    fwdMSpec s Q K i =
      (mPartial 128 (s.pids 0 * 128) (fwdQTile s Q) 1
        (fwdKTile s K) ((Real.sqrt (64 : ℝ))⁻¹) 1 i).unbotD 0 := by
  rw [show mPartial 128 (s.pids 0 * 128) (fwdQTile s Q) 1 (fwdKTile s K)
        ((Real.sqrt (64 : ℝ))⁻¹) 1 i
        = ((Finset.univ : Finset (Fin 128)).sup fun jLocal =>
            maskedScore (s.pids 0 * 128) (fwdQTile s Q) (fwdKTile s K)
              ((Real.sqrt (64 : ℝ))⁻¹) i
              (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal)) from by
    conv_lhs => rw [mPartial]
    rw [dif_pos (by norm_num : (0:Nat)+1 ≤ 1)]
    rw [show mPartial 128 (s.pids 0 * 128) (fwdQTile s Q) 1 (fwdKTile s K)
          ((Real.sqrt (64 : ℝ))⁻¹) 0 i = (⊥ : WithBot ℝ) from rfl]
    exact max_bot_left _]
  -- The block-local sup is over `maskedScore` at `blockIndex 128 1 0 _ jLocal`,
  -- which is the global index `jLocal`. Rewrite to the `if … then ↑score else ⊥`
  -- form expected by `withBot_sup_masked_unbotD`.
  rw [show (fun jLocal : Fin 128 =>
        maskedScore (s.pids 0 * 128) (fwdQTile s Q) (fwdKTile s K)
          ((Real.sqrt (64 : ℝ))⁻¹) i
          (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal))
      = (fun jLocal : Fin 128 =>
          if jLocal.val ≤ s.pids 0 * 128 + i.val then
            ((scaledScore (fwdQTile s Q) (fwdKTile s K)
              ((Real.sqrt (64 : ℝ))⁻¹) i jLocal : ℝ) : WithBot ℝ)
          else ⊥) from by
    funext jLocal
    have hidx : StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal
        = jLocal := by
      apply Fin.ext; simp [StreamingAccumulator.blockIndex]
    rw [hidx]; rfl]
  rw [withBot_sup_masked_unbotD (fun j : Fin 128 => j.val ≤ s.pids 0 * 128 + i.val)
    (fun j => scaledScore (fwdQTile s Q) (fwdKTile s K) ((Real.sqrt (64 : ℝ))⁻¹) i j)
    (fwdCausalSet_nonempty s i)]
  rfl

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- **L closed-form bridge (stored value).** The kernel stores `l_prev` literally
(the final, in-loop-`m`-shifted online-softmax normalizer `l_curr`), which is the
FA1 `lPartial` over the single 128-key block — **not** the un-shifted
`Σ exp(score)`. This is exactly the genuine `fwdLSpec`: the m-shifted causal
normalizer `Σ_{j ≤ …} exp(score i j − fwdMSpec)`. The `l_rcp` in-loop division
only rescales the output `p`/`acc`; it does not touch the stored `l_prev`. -/
theorem lPartial_eq_fwdLSpec
    (s : BlockState) (Q K : RegionName) (i : Fin 128) :
    lPartial 128 (s.pids 0 * 128) (fwdQTile s Q) 1
        (fwdKTile s K) ((Real.sqrt (64 : ℝ))⁻¹) 1 i
      = fwdLSpec s Q K i := by
  rw [lPartial_eq_mShifted (Bk := 128) (by norm_num) (s.pids 0 * 128)
    (fwdQTile s Q) 1 (fwdKTile s K) ((Real.sqrt (64 : ℝ))⁻¹) 1 (le_refl 1) i]
  rw [lFree_eq_flat]
  rw [show (mPartial 128 (s.pids 0 * 128) (fwdQTile s Q) 1 (fwdKTile s K)
        ((Real.sqrt (64 : ℝ))⁻¹) 1 i).unbotD 0 = fwdMSpec s Q K i from
    (fwdMSpec_eq_mPartial s Q K i).symm]
  unfold fwdLSpec
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  simp only [StreamingAccumulator.scaledScore, scaledScore]
  by_cases hj : j.val ≤ s.pids 0 * 128 + i.val
  · simp only [hj, if_true]
    rw [Real.exp_sub, Real.exp_neg]
    ring
  · simp only [hj, if_false]
    ring

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- **Un-shifted denominator recovery (corollary).** The stored `L = fwdLSpec`
together with the stored `M = fwdMSpec` recovers the un-shifted log-sum-exp
denominator: `exp(fwdMSpec) · fwdLSpec = Σ_{j ≤ …} exp(score i j)`, i.e.
`fwdMSpec + log fwdLSpec = log (Σ exp(score))`. -/
theorem exp_fwdMSpec_mul_fwdLSpec
    (s : BlockState) (Q K : RegionName) (i : Fin 128) :
    Real.exp (fwdMSpec s Q K i) * fwdLSpec s Q K i
      = Finset.univ.sum (fun j : Fin 128 =>
          if j.val ≤ s.pids 0 * 128 + i.val then
            Real.exp (scaledScore (fwdQTile s Q) (fwdKTile s K)
              ((Real.sqrt (64 : ℝ))⁻¹) i j)
          else 0) := by
  unfold fwdLSpec
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : j.val ≤ s.pids 0 * 128 + i.val
  · simp only [hj, if_true]
    rw [Real.exp_sub]
    field_simp
  · simp only [hj, if_false]
    rw [mul_zero]

/-! ### ════════ GENERAL (dimension-symbolic) forward spec layer ════════

The pinned `fwdQTile`/`fwdKTile`/`fwdVTile`/`fwdOutSpec`/`fwdLSpec`/`fwdMSpec`
above are typed at the Python test shape (`BLOCK_M = … = 128`, `HEAD_DIM = 64`,
single KV block of width `128`, scale `1/√64`). The following `*G` versions are
dimension-general over symbolic `BLOCK_M` (query rows), `BLOCK_DMODEL` (active
head channels), `HEAD_DIM` (the memory row stride), `stride_hz_2d` (the
per-`off_hz` row offset), the key span `SEQ`, the scale `sc`, and `numKVBlocks`
causal KV blocks (`SEQ = BLOCK_N · numKVBlocks`).

The FlashAttention-1 causal streaming math (`FA1MathCausal.mPartial`/`lPartial`/
`oPartial`, the bridge `streaming_eq_attentionRealCausalBlock`, the `succ`
recurrences, and `withBot_sup_masked_unbotD`) is ALREADY general over arbitrary
`Bk`/`numKVBlocks`/key length, so the general bridges below reuse it directly.
The pinned bridges (`fwdOutSpec_eq_streaming` etc.) are the `numKVBlocks = 1`,
`[128,64]` instantiations of these. -/

/-- General Q tile: output tile-row `i`, head channel `e` reads
`(pids1·stride_hz_2d + pids0·BLOCK_M + i)·HEAD_DIM + e` — the kernel's
`make_block_ptr` Q address with `stride_qm = HEAD_DIM`, `stride_qk = 1`,
offset `off_hz·stride_qh_2d + start_m·BLOCK_M`. -/
noncomputable def fwdQTileG (s : BlockState) (Q : RegionName)
    (stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q ((s.pids 1 * stride_hz_2d + s.pids 0 * BLOCK_M + i.val) * HEAD_DIM + e.val)

/-- General K tile: key row `j` (global, over the loaded key span `SEQ`),
channel `e` reads `(pids1·stride_hz_2d + j)·HEAD_DIM + e`. -/
noncomputable def fwdKTileG (s : BlockState) (K : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) =>
    s.readMem K ((s.pids 1 * stride_hz_2d + j.val) * HEAD_DIM + e.val)

/-- General V tile: value row `j`, channel `e` reads
`(pids1·stride_hz_2d + j)·HEAD_DIM + e`. -/
noncomputable def fwdVTileG (s : BlockState) (V : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) =>
    s.readMem V ((s.pids 1 * stride_hz_2d + j.val) * HEAD_DIM + e.val)

/-- General genuine closed-form forward `Out`: the natural-exp causal attention
block at query start `pids0·BLOCK_M`, scale `sc`, over the loaded key span `SEQ`.
Expressed via `attentionRealCausalBlock`, independent of the kernel `exec`. -/
noncomputable def fwdOutSpecG
    (s : BlockState) (Q K V : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL : Nat)
    (sc : ℝ) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealCausalBlock (s.pids 0 * BLOCK_M)
    (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
    (fwdKTileG s K stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL)
    (fwdVTileG s V stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL)
    sc idx

/-- General causal key set for query row `i`: keys `j ≤ pids0·BLOCK_M + i` within
the loaded span `SEQ`. Nonempty because `j = 0` always qualifies (needs
`0 < SEQ`). -/
def fwdCausalSetG (s : BlockState) (SEQ BLOCK_M : Nat) (i : Fin BLOCK_M) :
    Finset (Fin SEQ) :=
  Finset.univ.filter (fun j : Fin SEQ => j.val ≤ s.pids 0 * BLOCK_M + i.val)

theorem fwdCausalSetG_nonempty (s : BlockState) (SEQ BLOCK_M : Nat) (hSEQ : 0 < SEQ)
    (i : Fin BLOCK_M) : (fwdCausalSetG s SEQ BLOCK_M i).Nonempty := by
  refine ⟨⟨0, hSEQ⟩, ?_⟩
  simp [fwdCausalSetG]

/-- General genuine closed-form forward `M` for query row `i`: the per-row maximum
causal score `max_{j ≤ pids0·BLOCK_M + i} score i j` over the (nonempty) causal
key set within the loaded span `SEQ`. -/
noncomputable def fwdMSpecG
    (s : BlockState) (Q K : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL : Nat) (sc : ℝ)
    (hSEQ : 0 < SEQ) (i : Fin BLOCK_M) : ℝ :=
  (fwdCausalSetG s SEQ BLOCK_M i).sup' (fwdCausalSetG_nonempty s SEQ BLOCK_M hSEQ i)
    (fun j : Fin SEQ =>
      scaledScore (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
        (fwdKTileG s K stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL) sc i j)

/-- General genuine closed-form forward `L` for query row `i`: the **m-shifted**
causal softmax normalizer `Σ_{j ≤ pids0·BLOCK_M + i} exp(score i j − M_row)` over
the loaded span `SEQ`, with `M_row = fwdMSpecG`. This is exactly the value the
kernel stores (the running m-shifted `l_prev`); the un-shifted log-sum-exp is
recovered with the separately stored `M`. -/
noncomputable def fwdLSpecG
    (s : BlockState) (Q K : RegionName)
    (stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL : Nat) (sc : ℝ)
    (hSEQ : 0 < SEQ) (i : Fin BLOCK_M) : ℝ :=
  Finset.univ.sum (fun j : Fin SEQ =>
    if j.val ≤ s.pids 0 * BLOCK_M + i.val then
      Real.exp (scaledScore (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
        (fwdKTileG s K stride_hz_2d HEAD_DIM SEQ BLOCK_DMODEL) sc i j
        - fwdMSpecG s Q K stride_hz_2d HEAD_DIM SEQ BLOCK_M BLOCK_DMODEL sc hSEQ i)
    else 0)

/-! ### General FA1 bridges (`numKVBlocks`-arbitrary)

Mirror the pinned `fwdOutSpec_eq_streaming` / `fwdMSpec_eq_mPartial` /
`lPartial_eq_fwdLSpec` at symbolic dims and arbitrary `numKVBlocks`, reusing the
already-general `FA1MathCausal` lemmas. `SEQ = BLOCK_N · numKVBlocks`. -/

open VeriTile.Examples.FA1MathCausal in
/-- **General output closed-form bridge.** The FA1 streaming end-ratio
`oPartial / lPartial` over all `numKVBlocks` causal blocks (block width `BLOCK_N`,
total key span `SEQ = BLOCK_N·numKVBlocks`) equals the genuine `fwdOutSpecG`. -/
theorem fwdOutSpecG_eq_streaming
    (s : BlockState) (Q K V : RegionName)
    (stride_hz_2d HEAD_DIM BLOCK_N numKVBlocks BLOCK_M BLOCK_DMODEL : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks) (sc : ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    fwdOutSpecG s Q K V stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL sc idx
      = oPartial BLOCK_N (s.pids 0 * BLOCK_M)
          (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL) numKVBlocks
          (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
          (fwdVTileG s V stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
          sc numKVBlocks idx /
        lPartial BLOCK_N (s.pids 0 * BLOCK_M)
          (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL) numKVBlocks
          (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
          sc numKVBlocks idx.1 := by
  rw [show fwdOutSpecG s Q K V stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL sc idx
        = VeriTile.Examples.attentionRealCausalBlock (s.pids 0 * BLOCK_M)
            (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
            (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
            (fwdVTileG s V stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
            sc idx from by
        unfold fwdOutSpecG VeriTile.Triton.attentionRealCausalBlock
          VeriTile.Examples.attentionRealCausalBlock scaledScore
        rfl]
  exact (streaming_eq_attentionRealCausalBlock (Bk := BLOCK_N) hBN
    (s.pids 0 * BLOCK_M) (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
    numKVBlocks hnum
    (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
    (fwdVTileG s V stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
    sc idx).symm

/-- Sup of `f` over `Fin (k+1)` splits as the sup over the `castSucc` lanes
joined with the last lane. (Mathlib has no `Fin.sup_univ_castSucc`; proved by
antisymmetry.) -/
theorem fin_sup_univ_castSucc {α} [SemilatticeSup α] [OrderBot α]
    (k : Nat) (f : Fin (k + 1) → α) :
    (Finset.univ : Finset (Fin (k + 1))).sup f
      = (Finset.univ : Finset (Fin k)).sup (fun i => f i.castSucc) ⊔ f (Fin.last k) := by
  apply le_antisymm
  · apply Finset.sup_le
    intro i _
    rcases Fin.eq_castSucc_or_eq_last i with ⟨j, rfl⟩ | rfl
    · exact le_sup_of_le_left (Finset.le_sup (f := fun i => f i.castSucc) (Finset.mem_univ j))
    · exact le_sup_right
  · apply sup_le
    · apply Finset.sup_le; intro i _; exact Finset.le_sup (Finset.mem_univ _)
    · exact Finset.le_sup (Finset.mem_univ _)

open VeriTile.Examples.FA1MathCausal in
/-- **mPartial global-sup form.** The running causal max after `k` blocks equals
the `Finset.sup` (over block index `n < k` and lane `jL < Bk`) of the per-key
`maskedScore`. Proved by induction on `k` using `mPartial_succ_of_lt`. -/
theorem mPartialG_eq_blockSup {M D Bk : Nat}
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ) (N : Nat)
    (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ) (i : Fin M)
    (k : Nat) (hk : k ≤ N) :
    mPartial Bk qStart Q N K scale k i
      = (Finset.univ : Finset (Fin k)).sup (fun n =>
          (Finset.univ : Finset (Fin Bk)).sup (fun jL =>
            maskedScore qStart Q K scale i
              (StreamingAccumulator.blockIndex Bk N n.val
                (Nat.lt_of_lt_of_le n.isLt hk) jL))) := by
  induction k with
  | zero => simp [mPartial]
  | succ k' ih =>
      have hk' : k' ≤ N := Nat.le_of_succ_le hk
      rw [mPartial_succ_of_lt qStart Q N K scale k' (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [fin_sup_univ_castSucc]
      have hlast : (Finset.univ : Finset (Fin Bk)).sup (fun jLocal =>
            maskedScore qStart Q K scale i
              (StreamingAccumulator.blockIndex Bk N k' (Nat.lt_of_succ_le hk) jLocal))
          = (Finset.univ : Finset (Fin Bk)).sup (fun jL =>
              maskedScore qStart Q K scale i
                (StreamingAccumulator.blockIndex Bk N (Fin.last k').val
                  (Nat.lt_of_lt_of_le (Fin.last k').isLt hk) jL)) := by
        refine Finset.sup_congr rfl (fun jL _ => ?_)
        congr 1
      have hcast : (Finset.univ : Finset (Fin k')).sup (fun n =>
            (Finset.univ : Finset (Fin Bk)).sup (fun jL =>
              maskedScore qStart Q K scale i
                (StreamingAccumulator.blockIndex Bk N n.val
                  (Nat.lt_of_lt_of_le n.isLt hk') jL)))
          = (Finset.univ : Finset (Fin k')).sup (fun n =>
              (Finset.univ : Finset (Fin Bk)).sup (fun jL =>
                maskedScore qStart Q K scale i
                  (StreamingAccumulator.blockIndex Bk N n.castSucc.val
                    (Nat.lt_of_lt_of_le n.castSucc.isLt hk) jL))) := by
        refine Finset.sup_congr rfl (fun n _ => ?_)
        refine Finset.sup_congr rfl (fun jL _ => ?_)
        congr 1
      rw [hlast, hcast, max_comm]

open VeriTile.Examples.FA1MathCausal in
/-- WithBot `Finset.sup` over the block-product equals the flat sup over the full
key range `Fin (Bk·N)`, via `blockIndexEquiv`. -/
theorem withBot_blockSup_eq_flat {M D Bk N : Nat}
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ) (i : Fin M) :
    (Finset.univ : Finset (Fin N)).sup (fun n =>
        (Finset.univ : Finset (Fin Bk)).sup (fun jL =>
          maskedScore qStart Q K scale i
            (StreamingAccumulator.blockIndex Bk N n.val n.isLt jL)))
      = (Finset.univ : Finset (Fin (Bk * N))).sup (fun j =>
          maskedScore qStart Q K scale i j) := by
  conv_rhs =>
    rw [← Finset.map_univ_equiv (StreamingAccumulator.blockIndexEquiv Bk N).symm,
      Finset.sup_map, ← Finset.univ_product_univ, Finset.sup_product_left]
  refine Finset.sup_congr rfl (fun n _ => ?_)
  refine Finset.sup_congr rfl (fun jL _ => ?_)
  show maskedScore qStart Q K scale i
      (StreamingAccumulator.blockIndex Bk N n.val n.isLt jL)
    = maskedScore qStart Q K scale i
        ((StreamingAccumulator.blockIndexEquiv Bk N).symm (n, jL))
  congr 1
  -- `blockIndex Bk N n.val _ jL = (blockIndexEquiv Bk N).symm (n, jL)`. Both sides
  -- are `Fin (Bk*N)` with value `n.val * Bk + jL.val`; compare via `Fin.ext`.
  apply Fin.ext
  show n.val * Bk + jL.val
    = ((StreamingAccumulator.blockIndexEquiv Bk N).symm (n, jL)).val
  show n.val * Bk + jL.val
    = ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv.symm
        (finProdFinEquiv (n, jL))).val
  show n.val * Bk + jL.val = (finProdFinEquiv (n, jL)).val
  show n.val * Bk + jL.val = jL.val + Bk * n.val
  ring

open VeriTile.Examples.FA1MathCausal in
/-- **General M closed-form bridge.** The FA1 running causal max `mPartial` over all
`numKVBlocks` blocks, read out via `unbotD 0`, equals the genuine `fwdMSpecG`. -/
theorem fwdMSpecG_eq_mPartial
    (s : BlockState) (Q K : RegionName)
    (stride_hz_2d HEAD_DIM BLOCK_N numKVBlocks BLOCK_M BLOCK_DMODEL : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks) (sc : ℝ) (i : Fin BLOCK_M) :
    fwdMSpecG s Q K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL sc
        (Nat.mul_pos hBN hnum) i
      = (mPartial BLOCK_N (s.pids 0 * BLOCK_M)
          (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL) numKVBlocks
          (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
          sc numKVBlocks i).unbotD 0 := by
  rw [mPartialG_eq_blockSup (s.pids 0 * BLOCK_M)
    (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL) numKVBlocks
    (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc i
    numKVBlocks (le_refl _)]
  rw [show (fun (n : Fin numKVBlocks) =>
        (Finset.univ : Finset (Fin BLOCK_N)).sup (fun jL =>
          maskedScore (s.pids 0 * BLOCK_M)
            (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
            (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc i
            (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks n.val (Nat.lt_of_lt_of_le n.isLt (le_refl _)) jL)))
      = (fun (n : Fin numKVBlocks) =>
          (Finset.univ : Finset (Fin BLOCK_N)).sup (fun jL =>
            maskedScore (s.pids 0 * BLOCK_M)
              (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
              (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc i
              (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks n.val n.isLt jL))) from by
        funext n; refine Finset.sup_congr rfl (fun jL _ => by congr 1)]
  rw [withBot_blockSup_eq_flat (s.pids 0 * BLOCK_M)
    (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
    (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc i]
  rw [show (fun (j : Fin (BLOCK_N * numKVBlocks)) =>
        maskedScore (s.pids 0 * BLOCK_M)
          (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
          (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc i j)
      = (fun (j : Fin (BLOCK_N * numKVBlocks)) =>
          if j.val ≤ s.pids 0 * BLOCK_M + i.val then
            ((scaledScore (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
              (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc i j : ℝ) : WithBot ℝ)
          else ⊥) from by
        funext j
        unfold maskedScore
        by_cases hj : j.val ≤ s.pids 0 * BLOCK_M + i.val
        · simp only [hj, if_true]; rfl
        · simp only [hj, if_false]]
  rw [withBot_sup_masked_unbotD (fun j : Fin (BLOCK_N * numKVBlocks) => j.val ≤ s.pids 0 * BLOCK_M + i.val)
    (fun j => scaledScore (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL)
      (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc i j)
    (fwdCausalSetG_nonempty s (BLOCK_N * numKVBlocks) BLOCK_M (Nat.mul_pos hBN hnum) i)]
  rfl

open VeriTile.Examples.FA1MathCausal in
/-- **General L closed-form bridge (stored value).** The kernel stores `l_prev`
(the final m-shifted online-softmax normalizer), which is the FA1 `lPartial` over
all `numKVBlocks` blocks — exactly the genuine `fwdLSpecG`. -/
theorem lPartial_eq_fwdLSpecG
    (s : BlockState) (Q K : RegionName)
    (stride_hz_2d HEAD_DIM BLOCK_N numKVBlocks BLOCK_M BLOCK_DMODEL : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks) (sc : ℝ) (i : Fin BLOCK_M) :
    lPartial BLOCK_N (s.pids 0 * BLOCK_M)
        (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL) numKVBlocks
        (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        sc numKVBlocks i
      = fwdLSpecG s Q K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL sc
          (Nat.mul_pos hBN hnum) i := by
  rw [lPartial_eq_mShifted (Bk := BLOCK_N) hBN (s.pids 0 * BLOCK_M)
    (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL) numKVBlocks
    (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc
    numKVBlocks (le_refl _) i]
  rw [lFree_eq_flat]
  rw [show (mPartial BLOCK_N (s.pids 0 * BLOCK_M)
        (fwdQTileG s Q stride_hz_2d HEAD_DIM BLOCK_M BLOCK_DMODEL) numKVBlocks
        (fwdKTileG s K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        sc numKVBlocks i).unbotD 0
      = fwdMSpecG s Q K stride_hz_2d HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL sc
          (Nat.mul_pos hBN hnum) i from
    (fwdMSpecG_eq_mPartial s Q K stride_hz_2d HEAD_DIM BLOCK_N numKVBlocks BLOCK_M BLOCK_DMODEL hBN hnum sc i).symm]
  unfold fwdLSpecG
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  simp only [StreamingAccumulator.scaledScore, scaledScore]
  by_cases hj : j.val ≤ s.pids 0 * BLOCK_M + i.val
  · simp only [hj, if_true]
    rw [Real.exp_sub, Real.exp_neg]
    ring
  · simp only [hj, if_false]
    ring

/-! ## Forward loop-body per-statement op-eval recipes (RECIPE LAYER)

The 19-statement `forRangeDyn` body of `_fwd_kernel` (STAGE=3, diagonal causal
block) lowers — via `toAlgorithm?` — to the `Stmt` list (statement order):

```
 1. k       = load(blockPtr k_tile_ptr [0,1])              -- K tile, boundary-checked
 2. qk      = full [128,128] (const 0)                     -- zeros
 3. qk      = qk + dot(q, transpose k)                     -- qk = q·kᵀ
 4. qk      = qk * (√64)⁻¹                                 -- scale
 5. qk      = where(offs_m[:,None] ≥ start_n+offs_n[None,:], qk, -inf)  -- causal mask
 6. m_curr  = where(reduceMax(qk,1) > m_prev, reduceMax(qk,1), m_prev)  -- maximum
 7. l_prev  = l_prev * exp(m_prev - m_curr)                -- l_prev rescale (alpha)
 8. p       = exp(qk - m_curr[:,None])                     -- NATURAL exp
 9. l_curr  = reduceSum(p,1) + l_prev                      -- l_curr
10. l_rcp   = 1.0 / l_curr                                 -- reciprocal
11. p       = p * l_rcp[:,None]                            -- p rescale
12. acc     = acc * (l_prev * l_rcp)[:,None]               -- acc rescale
13. p       = castFloat real→fp16 p                        -- fp16 round-trip
14. v       = load(blockPtr v_tile_ptr [0,1])              -- V tile, boundary-checked
15. acc     = acc + dot(castFloat fp16→real p, v)          -- numerator dot
16. l_prev  = l_curr
17. m_prev  = m_curr
18. k_tile_ptr = advance(k_tile_ptr, [128,0])
19. v_tile_ptr = advance(v_tile_ptr, [128,0])
```

These standalone `evalOp` reductions (abstract register-readback hyps, symbolic
`BlockState`) are the triton-attention analogues of the `flash_*_op_eval` family
in `flash_attn` and the `AttentionForwardClosedForm` `*_op_eval` template — but
retargeted to this kernel's **natural-exp** (`Op.exp`, not `exp2`), in-loop
`l_rcp = 1/l_curr` reciprocal, and **boundary-checked** block-pointer loads
(`MemAccess.blockPtr … [0,1]`, not the `.none` flash form). They thread through
`stepStmts.cons_some` without reducing nested `setReg` literal states; the
invariant/step/assembly is the NEXT stage. -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Boundary-checked `[rowOff, 0]` block-ptr load** (`tl.load(ptr,
boundary_check=(0,1))`): a `.real` load of a well-formed 2D block-pointer with
offsets `[rowOff, 0]` and checked axes `[0,1]` reads `readMem` at
`base + (rowOff+i)·strideT + j·strideS` on every in-bounds lane (`rowOff+i < rows
∧ j < cols`), and the default carrier otherwise. This is the K/V boundary-checked
load shape (the K/V `make_block_ptr` row-offset = `off_hz·stride_qh_2d`). -/
theorem ta_load_blockPtr_bc_eval
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff : Nat)
    (ptrOp : Op .blockPtr [BT, BS]) (s : BlockState)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, 0] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp [0, 1]) .none) s
      = some ⟨fun idx : TileIndex [BT, BS] =>
          if (rowOff + idx.1.val < rows ∧ idx.2.1.val < cols) then
            some (s.readMem region
              (base + (rowOff + idx.1.val) * strideT + idx.2.1.val * strideS))
          else BlockState.defaultCarrier .real⟩ := by
  simp only [evalOp, hp, Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_row_offset,
    BlockPtr.inBounds_2d_row_offset, BlockState.readMemValue_real]
  by_cases h : rowOff + idx.1.val < rows ∧ idx.2.1.val < cols
  · simp only [h, decide_true, if_true, and_self]
  · simp only [h, decide_false, and_self, Bool.false_eq_true, if_false]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`k = tl.load(k_tile_ptr, boundary_check=(0,1))`** (loop body L1) → `fwdKTile`
cells. With the K `make_block_ptr` (parentShape `(1024, 64)`, strides
`(stride_kn=64, stride_kk=1)`, offsets `[off_hz·128, 0]`) and the test-shape
guarantee that every lane is in bounds (`off_hz·128 + i < 1024` for `off_hz < 8`,
`j < 64`), each lane reads `K[(off_hz·128 + i)·64 + j] = (fwdKTile s K)(i,j,⋆)`
(note `off_hz = pids 1`). -/
theorem ta_load_k_eval
    (K : RegionName) (off_hz : Nat) (ptrOp : Op .blockPtr [128, 64]) (s : BlockState)
    (hpids1 : s.pids 1 = off_hz)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [128, 64] =>
        { region := K, baseOffset := 0, parentShape := [1024, 64],
          blockShape := [128, 64], strides := [64, 1],
          offsets := [off_hz * 128, 0] }⟩)
    (hrow : ∀ i : Fin 128, off_hz * 128 + i.val < 1024) :
    evalOp (.load .real (.blockPtr ptrOp [0, 1]) .none) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          some (fwdKTile s K idx)⟩ := by
  rw [ta_load_blockPtr_bc_eval K 0 1024 64 128 64 64 1
    (off_hz * 128) ptrOp s hp]
  refine congrArg some ?_
  ext idx
  have hj : idx.2.1.val < 64 := idx.2.1.isLt
  have hi : off_hz * 128 + idx.1.val < 1024 := hrow idx.1
  simp only [hi, hj, and_self, if_true]
  congr 1
  simp only [fwdKTile, hpids1, Nat.zero_add, Nat.mul_one]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`v = tl.load(v_tile_ptr, boundary_check=(0,1))`** (loop body L14) → `fwdVTile`
cells. Same K/V `make_block_ptr` row-offset shape (`strides=(stride_vk=64,
stride_vn=1)`); each in-bounds lane reads `V[(off_hz·128 + j)·64 + e] =
(fwdVTile s V)(j,e,⋆)`. -/
theorem ta_load_v_eval
    (V : RegionName) (off_hz : Nat) (ptrOp : Op .blockPtr [128, 64]) (s : BlockState)
    (hpids1 : s.pids 1 = off_hz)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [128, 64] =>
        { region := V, baseOffset := 0, parentShape := [1024, 64],
          blockShape := [128, 64], strides := [64, 1],
          offsets := [off_hz * 128, 0] }⟩)
    (hrow : ∀ j : Fin 128, off_hz * 128 + j.val < 1024) :
    evalOp (.load .real (.blockPtr ptrOp [0, 1]) .none) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          some (fwdVTile s V idx)⟩ := by
  rw [ta_load_blockPtr_bc_eval V 0 1024 64 128 64 64 1
    (off_hz * 128) ptrOp s hp]
  refine congrArg some ?_
  ext idx
  have he : idx.2.1.val < 64 := idx.2.1.isLt
  have hj : off_hz * 128 + idx.1.val < 1024 := hrow idx.1
  simp only [hj, he, and_self, if_true]
  congr 1
  simp only [fwdVTile, hpids1, Nat.zero_add, Nat.mul_one]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`q = tl.load(q_tile_ptr)`** (pre-loop, no boundary check in the diagonal
block) → `fwdQTile` cells. The Q `make_block_ptr` has offsets
`[off_hz·128 + start_m·128, 0]` (row), strides `(stride_qm=64, stride_qk=1)`; each
lane reads `Q[(off_hz·128 + start_m·128 + i)·64 + e] = (fwdQTile s Q)(i,e,⋆)`
(with `start_m = pids 0`, `off_hz = pids 1`). Specializes `load_blockPtr_Q_eval`. -/
theorem ta_load_q_eval
    (Q : RegionName) (off_hz start_m : Nat) (ptrOp : Op .blockPtr [128, 64]) (s : BlockState)
    (hpids0 : s.pids 0 = start_m) (hpids1 : s.pids 1 = off_hz)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [128, 64] =>
        { region := Q, baseOffset := 0,
          parentShape := [1024, 64], blockShape := [128, 64], strides := [64, 1],
          offsets := [off_hz * 128 + start_m * 128, 0] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          some (fwdQTile s Q idx)⟩ := by
  rw [load_blockPtr_Q_eval Q 0 1024 64 128 64
    64 1 (off_hz * 128 + start_m * 128) ptrOp s hp]
  refine congrArg some ?_
  ext idx
  congr 1
  simp only [fwdQTile, hpids0, hpids1, Nat.zero_add, Nat.mul_one]

/-- `evalOp` helper for the `>=` causal predicate (`Op.ge`), which has no
dedicated `@[simp]` form. Mirror of `flash_attn`'s `evalOp_ge`. -/
theorem ta_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk = tl.zeros([128,128])` statement eval** (loop body L2): the all-`0`
tile, the neutral pre-dot accumulator. -/
theorem ta_qkzeros_eval (s : BlockState) (BM BN : Nat) :
    evalOp (Op.full [BM, BN] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk += tl.dot(q, tl.trans(k))` statement eval** (loop body L3): adds the
`q · kᵀ` dot to the (zero-seeded) `qk` tile. Both `q` and `k` are `.real` here
(the diagonal-block transcription casts later). -/
theorem ta_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BN, BD])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .real [BM, BD] "q" = some qtile)
    (hk : s.regs .real [BN, BD] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q")
          (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qktile (Tile.dot [] qtile (Tile.transpose [] ktile))) := by
  have hkr : evalOp (Op.ref .real [BN, BD] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hqkr : evalOp (Op.ref .real [BM, BD] "q") s = some qtile := by rw [evalOp_ref, hq]
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")) s
      = some (Tile.transpose [] ktile) := by
    erw [evalOp_transpose [] (Op.ref .real [BN, BD] "k"), hkr]; rfl
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q")
        (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k"))) s
      = some (Tile.dot [] qtile (Tile.transpose [] ktile)) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BD] "q")
      (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")), hqkr, htr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hqk, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk *= sm_scale` statement eval** (loop body L4): scale by the scalar
`(√64)⁻¹` (broadcast on the right). -/
theorem ta_qk_scale_eval (s : BlockState) (BM BN : Nat) (sc : ℝ)
    (qktile : Tile .real [BM, BN]) (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile
          (Tile.scalar (some sc : WithBot ℝ))) := by
  rw [evalOp_mul]; simp [evalOp_ref, evalOp_const, hqk]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Causal `where` statement eval** (loop body L5): `qk = tl.where(offs_m[:,None]
≥ start_n + offs_n[None,:], qk, -inf)`. Given the `offs_m`/`offs_n` index vectors,
`start_n = SN`, and the running `qk`, the masked tile keeps `qk` where
`SN + j ≤ offs_m_i` and `⊥` (`-inf`) on future keys. Triton-attention analogue of
`flash_where_op_eval`. -/
theorem ta_where_eval (s : BlockState) (BM BN SN : Nat)
    (gm : Fin BM → Nat) (qktile : Tile .real [BM, BN])
    (hom : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) s
      = some ⟨fun idx : TileIndex [BM, BN] =>
          if SN + idx.2.1.val ≤ gm idx.1 then qktile.data idx else (⊥ : WithBot ℝ)⟩ := by
  have hexpM : @evalOp TileDType.nat [BM, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec gm)) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hom
  have hexpN : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hon
  have haddN : @evalOp TileDType.nat [1, BN]
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))) s
        = some (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar SN)
            (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val)))) := by
    rw [evalOp_add]
    rw [show evalOp (Op.ref .nat [] "start_n") s = some (Tile.scalar SN) from by
      rw [evalOp_ref, hsn]]
    rw [hexpN]; rfl
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where, ta_evalOp_ge]
  simp only [evalOp_ref, hexpM, haddN, hqk, hbcast,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.expandDim_data, Tile.scalar, Tile.vec,
    ComparableDType.nat, NumericDType.add]
  by_cases h : SN + idx.2.1.val ≤ gm idx.1
  · rw [if_pos (by simpa using h)]; simp [h]
  · rw [if_neg (by simpa using h)]; simp [h]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`m_curr = tl.maximum(tl.max(qk,1), m_prev)` statement eval** (loop body L6):
lowered to `where(reduceMax(qk,1) > m_prev, reduceMax(qk,1), m_prev)`. `reduceMax`'s
result-shape `eraseAxis` blocks `simp`/`rw` matching, so the reduced row is proven
naturally then defeq-coerced to `[BM]`. -/
theorem ta_mij_eval (s : BlockState) (BM BN : Nat)
    (mtile : Tile .real [BM]) (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hmp : s.regs .real [BM] "m_prev" = some mtile)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))
          (Op.ref .real [BM] "m_prev"))
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))
        (Op.ref .real [BM] "m_prev")) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) rmaxT mtile)
          rmaxT mtile) := by
  have hrmaxN : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false
      (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [BM]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false
        (Op.ref .real [BM, BN] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmp, hrmax, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_prev *= tl.exp(m_prev - m_curr)` statement eval** (loop body L7): the
`alpha = exp(m_prev - m_curr)` rescale of `l_prev` (NATURAL exp). This is the
cross-block correction factor applied to the running denominator. -/
theorem ta_alpha_eval (s : BlockState) (BM : Nat) (ltile mp mc : Tile .real [BM])
    (hl : s.regs .real [BM] "l_prev" = some ltile)
    (hmp : s.regs .real [BM] "m_prev" = some mp)
    (hmc : s.regs .real [BM] "m_curr" = some mc) :
    evalOp (Op.mul .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "l_prev")
        (Op.exp (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "m_prev") (Op.ref .real [BM] "m_curr")))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mp mc))) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_exp, evalOp_sub, hl, hmp, hmc,
    Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p = tl.exp(qk - m_curr[:, None])` statement eval** (loop body L8): the
NATURAL-exp numerator shift. `expandDim`'s `insertAxis` shape gets normalized to
`[BM,1]` by `sub`; the operand eval is proven naturally then defeq-coerced. -/
theorem ta_p_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM, BN]) (mc : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hmc : s.regs .real [BM] "m_curr" = some mc) :
    evalOp (Op.exp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_curr")))) s
      = some (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            qktile (Tile.expandDim ⟨1, hax⟩ mc))) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_curr")) s
      = some (Tile.expandDim ⟨1, hax⟩ mc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmc
  rw [evalOp_exp, evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_curr = tl.sum(p, 1) + l_prev` statement eval** (loop body L9): the row
sum of `p` added to the (alpha-rescaled) `l_prev`. -/
theorem ta_lij_eval (s : BlockState) (BM BN : Nat)
    (ptile : Tile .real [BM, BN]) (ltile : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hl : s.regs .real [BM] "l_prev" = some ltile) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p"))
        (Op.ref .real [BM] "l_prev")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile) ltile) := by
  have hpr : evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by rw [evalOp_ref, hp]
  have hsum : @evalOp TileDType.real [BM]
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile) := by
    erw [evalOp_reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false
      (Op.ref .real [BM, BN] "p"), hpr]; rfl
  rw [evalOp_add]; simp only [evalOp_ref, hl, hsum, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_rcp = 1.0 / l_curr` statement eval** (loop body L10): the per-row
reciprocal of the running denominator (broadcast on the left). This is the in-loop
`l_rcp` the kernel multiplies into `p`/`acc`; it never touches the stored
`l_prev`. -/
theorem ta_lrcp_eval (s : BlockState) (BM : Nat) (lc : Tile .real [BM])
    (hlc : s.regs .real [BM] "l_curr" = some lc) :
    evalOp (Op.div .real Broadcast.scalarL (Op.const (1.0 : ℝ)) (Op.ref .real [BM] "l_curr")) s
      = some (Tile.bop NumericDType.real.div Broadcast.scalarL
          (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lc) := by
  rw [evalOp_div]; simp [evalOp_const, evalOp_ref, hlc]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p *= l_rcp[:, None]` statement eval** (loop body L11): rescale `p` by the
per-row reciprocal `l_rcp`. -/
theorem ta_p_rcp_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (ptile : Tile .real [BM, BN]) (lr : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hlr : s.regs .real [BM] "l_rcp" = some lr) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "p") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "l_rcp"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          ptile (Tile.expandDim ⟨1, hax⟩ lr)) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "l_rcp")) s
      = some (Tile.expandDim ⟨1, hax⟩ lr) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hlr
  rw [evalOp_mul]
  simp only [evalOp_ref, hp, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc *= (l_prev * l_rcp)[:, None]` statement eval** (loop body L12): rescale
the output accumulator by the per-row factor `l_prev · l_rcp` (the running-`l`
correction that telescopes to the FA1 end-normalization). -/
theorem ta_acc_rescale_eval (s : BlockState) (BM BD : Nat) (hax : 1 < [BM].length.succ)
    (acctile : Tile .real [BM, BD]) (lp lr : Tile .real [BM])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hlp : s.regs .real [BM] "l_prev" = some lp)
    (hlr : s.regs .real [BM] "l_rcp" = some lr) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.expandDim ⟨1, hax⟩
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp")))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile
          (Tile.expandDim ⟨1, hax⟩
            (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) lp lr))) := by
  have haccr : evalOp (Op.ref .real [BM, BD] "acc") s = some acctile := by rw [evalOp_ref, hacc]
  have hlpr : evalOp (Op.ref .real [BM] "l_prev") s = some lp := by rw [evalOp_ref, hlp]
  have hlrr : evalOp (Op.ref .real [BM] "l_rcp") s = some lr := by rw [evalOp_ref, hlr]
  have hmul : @evalOp TileDType.real [BM]
      (Op.mul .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp")) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) lp lr) := by
    rw [evalOp_mul]; simp only [hlpr, hlrr, Option.bind_eq_bind, Option.bind_some]
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp"))) s
      = some (Tile.expandDim ⟨1, hax⟩
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) lp lr)) := by
    erw [evalOp_expandDim ⟨1, hax⟩ (Op.mul .real (Broadcast.consSame Broadcast.nil)
      (Op.ref .real [BM] "l_prev") (Op.ref .real [BM] "l_rcp")), hmul]; rfl
  rw [evalOp_mul]; simp only [haccr, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p = (p).to(tl.float16)` statement eval** (loop body L13): the `p` fp16
round-trip before the value dot (pointwise float cast). -/
theorem ta_pfp16_eval (s : BlockState) (BM BN : Nat) (ptile : Tile .real [BM, BN])
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p")) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ : Tile .fp16 [BM, BN]) := by
  rw [evalOp_castFloat]; simp [hp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc += tl.dot(p, v)` statement eval** (loop body L15): the numerator
accumulation. The `p` fp16 round-trip (`castFloat fp16 (castFloat real fp16 p)`)
reduces to `p` (cast identity in the model); the dot accumulates the value
contribution. Triton-attention analogue of `acc2_op_eval`. -/
theorem ta_acc_eval (s : BlockState) (BM BN BD : Nat)
    (acctile : Tile .real [BM, BD]) (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, BD])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hpf16 : s.regs .fp16 [BM, BN] "p" = some
      (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ : Tile .fp16 [BM, BN]))
    (hv : s.regs .real [BN, BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "p"))
          (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot [] ptile vtile)) := by
  have haccr : evalOp (Op.ref .real [BM, BD] "acc") s = some acctile := by rw [evalOp_ref, hacc]
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vtile := by rw [evalOp_ref, hv]
  have hpr : evalOp (Op.ref .fp16 [BM, BN] "p") s = some
      (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ : Tile .fp16 [BM, BN]) := by
    rw [evalOp_ref, hpf16]
  have hcb : @evalOp TileDType.real [BM, BN]
      (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "p")) s = some ptile := by
    erw [evalOp_castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "p"), hpr]
    refine congrArg some ?_; ext i; simp [FloatDType.cast]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "p"))
        (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot [] ptile vtile) := by
    erw [evalOp_dot [] (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "p"))
      (Op.ref .real [BN, BD] "v"), hcb, hvr]; rfl
  rw [evalOp_add]; simp only [haccr, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_prev = l_curr` / `m_prev = m_curr` statement eval** (loop body L16/L17):
the running-state carry (a bare register read). -/
theorem ta_reg_carry_eval (s : BlockState) (BM : Nat) (name : RegName)
    (t : Tile .real [BM]) (h : s.regs .real [BM] name = some t) :
    evalOp (Op.ref .real [BM] name) s = some t := by
  rw [evalOp_ref, h]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`k_tile_ptr = tl.advance(k_tile_ptr, [BLOCK_N, 0])`** / `v_tile_ptr` (loop
body L18/L19): advances the ROW offset of a `[rowOff, 0]` block pointer by
`BLOCK_N` (the K/V tiles step down the key axis). Triton-attention analogue of
`flash_advance_row_eval`. -/
theorem ta_advance_row_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS rowOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, 0] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [d, 0]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff + d, 0] }⟩) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockPtr.advance_2d_offsets]

/-- The 13 lowered preLoop statements of the Python-shape triton-attention
forward body. -/
def taPreLoop (Q K V Out : RegionName) (sc : ℝ) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)),
    Stmt.assign .nat [128] "offs_n" (Op.arange 128),
    Stmt.assign .real [128] "m_prev"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf),
    Stmt.assign .real [128] "l_prev" (Op.full [128] (Op.const 0)),
    Stmt.assign .real [128, 64] "acc" (Op.full [128, 64] (Op.const 0)),
    Stmt.assign .nat [] "stride_qh_2d"
      (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 8192) (Op.constNat 64)) (Op.constNat 1)),
    Stmt.assign .blockPtr [128, 64] "q_tile_ptr"
      (Op.makeBlockPtrDynOffsets Q (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
        [Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)), Op.constNat 0]),
    Stmt.assign .blockPtr [128, 64] "k_tile_ptr"
      (Op.makeBlockPtrDynOffsets K (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
          Op.constNat 0]),
    Stmt.assign .blockPtr [128, 64] "v_tile_ptr"
      (Op.makeBlockPtrDynOffsets V (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
          Op.constNat 0]),
    Stmt.assign .blockPtr [128, 64] "out_tile_ptr"
      (Op.makeBlockPtrDynOffsets Out (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
        [Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)), Op.constNat 0]),
    Stmt.assign .real [128, 64] "q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") []) MaskOpt.none) ]

/-- The 19 lowered loop-body statements of the Python-shape triton-attention
forward `forRangeDyn` body (STAGE=3 diagonal causal block). -/
def taLoopBody (sc : ℝ) : List Stmt :=
  [ Stmt.assign .real [128, 64] "k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "k_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [128, 128] "qk" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .real [128, 128] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 128] "qk")
        (Op.dot (batch := []) (Op.ref .real [128, 64] "q")
          (Op.transpose (batch := []) (Op.ref .real [128, 64] "k")))),
    Stmt.assign .real [128, 128] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const sc)),
    Stmt.assign .real [128, 128] "qk"
      (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n"))))
        (Op.ref .real [128, 128] "qk") (Op.broadcast Op.negInf [128, 128])),
    Stmt.assign .real [128] "m_curr"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 128].length) Bool.false (Op.ref .real [128, 128] "qk"))
          (Op.ref .real [128] "m_prev"))
        (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 128].length) Bool.false (Op.ref .real [128, 128] "qk"))
        (Op.ref .real [128] "m_prev")),
    Stmt.assign .real [128] "l_prev"
      (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "l_prev")
        (Op.exp (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [128] "m_prev") (Op.ref .real [128] "m_curr")))),
    Stmt.assign .real [128, 128] "p"
      (Op.exp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 128] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_curr")))),
    Stmt.assign .real [128] "l_curr"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 128].length) Bool.false (Op.ref .real [128, 128] "p"))
        (Op.ref .real [128] "l_prev")),
    Stmt.assign .real [128] "l_rcp"
      (Op.div .real Broadcast.scalarL (Op.const (1.0 : ℝ)) (Op.ref .real [128] "l_curr")),
    Stmt.assign .real [128, 128] "p"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 128] "p") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "l_rcp"))),
    Stmt.assign .real [128, 64] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "acc")
        (Op.expandDim ⟨1, by simp⟩
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [128] "l_prev") (Op.ref .real [128] "l_rcp")))),
    Stmt.assign .fp16 [128, 128] "p"
      (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "p")),
    Stmt.assign .real [128, 64] "v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "v_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [128, 64] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [128, 128] "p"))
          (Op.ref .real [128, 64] "v"))),
    Stmt.assign .real [128] "l_prev" (Op.ref .real [128] "l_curr"),
    Stmt.assign .real [128] "m_prev" (Op.ref .real [128] "m_curr"),
    Stmt.assign .blockPtr [128, 64] "k_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "k_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "v_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "v_tile_ptr") [128, 0]) ]

/-- The 8 lowered postLoop statements of the Python-shape triton-attention
forward body: recompute `start_m`/`offs_m`, the L/M row pointers, the dual L/M
stores, the final `acc` fp16 cast, and the boundary-checked `Out` store. -/
def taPostLoop (L M Out : RegionName) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)),
    Stmt.assign .ptr [128] "l_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase L)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))
          (Op.ref .nat [128] "offs_m"))),
    Stmt.assign .ptr [128] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))
          (Op.ref .nat [128] "offs_m"))),
    Stmt.store .real [128] (MemAccess.ptr (Op.ref .ptr [128] "l_ptrs"))
      (Op.ref .real [128] "l_prev") MaskOpt.none,
    Stmt.store .real [128] (MemAccess.ptr (Op.ref .ptr [128] "m_ptrs"))
      (Op.ref .real [128] "m_prev") MaskOpt.none,
    Stmt.assign .fp16 [128, 64] "acc"
      (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "acc")),
    Stmt.store .fp16 [128, 64] (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "out_tile_ptr") [0, 1])
      (Op.ref .fp16 [128, 64] "acc") MaskOpt.none ]

set_option maxRecDepth 8000 in
/-- The lowered forward body is exactly `taPreLoop ++ forRangeDyn :: taPostLoop`. -/
theorem ta_body_split (Q K V L M Out : RegionName) (sc : ℝ) :
    (triton_attention_fwd_kernel Q K V L M Out sc
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      2 4 128 1024 128 64 128).toAlgKernel.body
      = taPreLoop Q K V Out sc
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
              (Op.constNat 128) (taLoopBody sc)
            :: taPostLoop L M Out) := by
  rfl

/-- The literal K/V block-pointer tile at row offset `rowOff` (region `R`,
parent `[1024,64]`, strides `[64,1]`, offsets `[rowOff, 0]`). -/
def taKVPtrTile (R : RegionName) (rowOff : Nat) : Tile .blockPtr [128, 64] :=
  ⟨fun _ : TileIndex [128, 64] =>
    { region := R, baseOffset := 0, parentShape := [1024, 64],
      blockShape := [128, 64], strides := [64, 1], offsets := [rowOff, 0] }⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Loop-body execution chain.** Entered at counter `start_n = 0` (the single
KV block of the test shape) with the loaded `q` tile, the `m_prev`/`l_prev`/`acc`
running registers, the index vectors, and the K/V block pointers at offset
`[off_hz·128, 0]`, the 19-statement `taLoopBody` executes to a final state.
Every statement's `evalOp` is discharged by its banked recipe, threaded through
`stepStmts.cons_some`. The symbolic registers (`m_curr`/`l_curr`/`l_rcp`/`p`/
`acc`) are exposed for the step lemma. -/
theorem taLoopBody_steps (K V : RegionName) (off_hz : Nat) (sc : ℝ) (sin : BlockState)
    (qtile : Tile .real [128, 64]) (mtile ltile : Tile .real [128])
    (acctile : Tile .real [128, 64])
    (gm : Fin 128 → Nat)
    (hpids1 : sin.pids 1 = off_hz)
    (hrow : ∀ i : Fin 128, off_hz * 128 + i.val < 1024)
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar 0))
    (hoffm : sin.regs .nat [128] "offs_m" = some (Tile.vec gm))
    (hoffn : sin.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hq : sin.regs .real [128, 64] "q" = some qtile)
    (hmp : sin.regs .real [128] "m_prev" = some mtile)
    (hlp : sin.regs .real [128] "l_prev" = some ltile)
    (hacc : sin.regs .real [128, 64] "acc" = some acctile)
    (hkp : sin.regs .blockPtr [128, 64] "k_tile_ptr" = some (taKVPtrTile K (off_hz * 128)))
    (hvp : sin.regs .blockPtr [128, 64] "v_tile_ptr" = some (taKVPtrTile V (off_hz * 128)))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ (rmaxT : Tile .real [128]) (qk0 qk1 : Tile .real [128, 128]),
      Tile.reduceMaxDrop (⟨1, by decide⟩ : Fin [128,128].length) qk1 = some rmaxT ∧
      qk0 = Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
          (Tile.dot [] qtile (Tile.transpose [] (⟨fun idx => some (fwdKTile sin K idx)⟩ : Tile .real [128, 64]))))
        (Tile.scalar (some sc : WithBot ℝ)) ∧
      qk1 = (⟨fun idx : TileIndex [128, 128] =>
        if 0 + idx.2.1.val ≤ gm idx.1 then qk0.data idx else (⊥ : WithBot ℝ)⟩) ∧
      ∃ sF, stepStmts (taLoopBody sc) sin = some sF
        ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
        ∧ ∃ (mcurrT lcurrT lrcpT : Tile .real [128]) (pT : Tile .real [128, 128])
              (acc1T : Tile .real [128, 64]),
            mcurrT = Tile.select
                (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) rmaxT mtile)
                rmaxT mtile
            ∧ pT = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                (Tile.uop WithBot.realExp
                  (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                    qk1 (Tile.expandDim ⟨1, by decide⟩ mcurrT)))
                (Tile.expandDim ⟨1, by decide⟩ lrcpT)
            ∧ lcurrT = Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
                (Tile.reduceSumDrop (⟨1, by decide⟩ : Fin [128,128].length)
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                      qk1 (Tile.expandDim ⟨1, by decide⟩ mcurrT))))
                (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mcurrT)))
            ∧ lrcpT = Tile.bop NumericDType.real.div Broadcast.scalarL
                (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT
            ∧ acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                acctile
                (Tile.expandDim ⟨1, by decide⟩
                  (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
                    (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile
                      (Tile.uop WithBot.realExp
                        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mcurrT)))
                    lrcpT))
            ∧ sF.regs .real [128] "m_prev" = some mcurrT
            ∧ sF.regs .real [128] "l_prev" = some lcurrT
            ∧ sF.regs .real [128, 64] "acc" = some
                (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  acc1T (Tile.dot [] pT (⟨fun idx => some (fwdVTile sin V idx)⟩ : Tile .real [128, 64])))
            ∧ sF.regs .blockPtr [128, 64] "k_tile_ptr" = some (taKVPtrTile K (off_hz * 128 + 128))
            ∧ sF.regs .blockPtr [128, 64] "v_tile_ptr" = some (taKVPtrTile V (off_hz * 128 + 128))
            ∧ sF.regs .real [128, 64] "q" = sin.regs .real [128, 64] "q"
            ∧ sF.regs .nat [128] "offs_m" = sin.regs .nat [128] "offs_m"
            ∧ sF.regs .nat [128] "offs_n" = sin.regs .nat [128] "offs_n"
            ∧ sF.regs .nat [] "off_hz" = sin.regs .nat [] "off_hz"
            ∧ sF.regs .blockPtr [128, 64] "out_tile_ptr" = sin.regs .blockPtr [128, 64] "out_tile_ptr" := by
  set kT : Tile .real [128, 64] := ⟨fun idx => some (fwdKTile sin K idx)⟩ with hkT
  set vT : Tile .real [128, 64] := ⟨fun idx => some (fwdVTile sin V idx)⟩ with hvT
  set qkdot : Tile .real [128, 128] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
      (Tile.dot [] qtile (Tile.transpose [] kT)) with hqkdot
  set qk0 : Tile .real [128, 128] := Tile.bop NumericDType.real.mul Broadcast.scalarR
    qkdot (Tile.scalar (some sc : WithBot ℝ)) with hqk0
  set qk1 : Tile .real [128, 128] := (⟨fun idx : TileIndex [128, 128] =>
    if 0 + idx.2.1.val ≤ gm idx.1 then qk0.data idx else (⊥ : WithBot ℝ)⟩) with hqk1
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by decide⟩ : Fin [128,128].length) qk1 = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [128,128] (⟨1, by decide⟩ : Fin [128,128].length) from by decide)]⟩
  set mcurrT : Tile .real [128] := Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) rmaxT mtile) rmaxT mtile
    with hmc
  set alphaT : Tile .real [128] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mcurrT) with hal
  set lprev1T : Tile .real [128] := Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
      ltile alphaT with hlp1
  set pexpT : Tile .real [128, 128] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        qk1 (Tile.expandDim ⟨1, by decide⟩ mcurrT)) with hpexp
  set lcurrT : Tile .real [128] := Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.reduceSumDrop (⟨1, by decide⟩ : Fin [128,128].length) pexpT) lprev1T with hlc
  set lrcpT : Tile .real [128] := Tile.bop NumericDType.real.div Broadcast.scalarL
      (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT with hlr
  set pT : Tile .real [128, 128] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      pexpT (Tile.expandDim ⟨1, by decide⟩ lrcpT) with hpT
  set acc1T : Tile .real [128, 64] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      acctile (Tile.expandDim ⟨1, by decide⟩
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) lprev1T lrcpT)) with hacc1
  refine ⟨rmaxT, qk0, qk1, hrm, rfl, rfl, ?_⟩
  -- execution chain
  unfold taLoopBody
  -- L1: k = load(k_tile_ptr [0,1])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_load_k_eval K off_hz (Op.ref .blockPtr [128, 64] "k_tile_ptr") sin hpids1
      (by rw [evalOp_ref, hkp]; rfl) hrow))]
  -- L2: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ta_qkzeros_eval _ 128 128))]
  -- L3: qk += dot(q, trans k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_qk_dot_eval _ 128 128 64 (⟨fun _ => some (0:ℝ)⟩) qtile kT
      (by simp) (by simp [hq]) (by simp [hkT])))]
  -- L4: qk *= sc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_qk_scale_eval _ 128 128 sc qkdot (by simp [hqkdot])))]
  -- L5: causal where
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_where_eval _ 128 128 0 gm qk0 (by simp [hoffm]) (by simp [hoffn]) (by simp [hsn]) (by simp [hqk0])))]
  -- L6: m_curr = maximum
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_mij_eval _ 128 128 mtile qk1 rmaxT (by simp [hmp]) (by simp [hqk1]) hrm))]
  -- L7: l_prev *= alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_alpha_eval _ 128 ltile mtile mcurrT (by simp [hlp]) (by simp [hmp]) (by simp [hmc])))]
  -- L8: p = exp(qk - m_curr)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_p_eval _ 128 128 (by decide) qk1 mcurrT (by simp [hqk1]) (by simp [hmc])))]
  -- L9: l_curr = sum p + l_prev
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_lij_eval _ 128 128 pexpT lprev1T (by simp [hpexp]) (by simp [hlp1, hal])))]
  -- L10: l_rcp = 1/l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_lrcp_eval _ 128 lcurrT (by simp [hlc])))]
  -- L11: p *= l_rcp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_p_rcp_eval _ 128 128 (by decide) pexpT lrcpT (by simp [hpexp]) (by simp [hlr])))]
  -- L12: acc *= (l_prev * l_rcp)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_acc_rescale_eval _ 128 64 (by decide) acctile lprev1T lrcpT (by simp [hacc]) (by simp [hlp1, hal]) (by simp [hlr])))]
  -- L13: p -> fp16
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .fp16 [128, 128] "p"
    (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "p")) _
    (⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ : Tile .fp16 [128, 128])
    (ta_pfp16_eval _ 128 128 pT (by simp [hpT])))]
  -- L14: v = load(v_tile_ptr [0,1]); the load reads `fwdVTile <state> V`, but
  -- every prior setReg preserves mem/pids so this equals `vT = fwdVTile sin V`.
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "v_tile_ptr") [0, 1]) MaskOpt.none) _
        = some vT from by
      rw [ta_load_v_eval V off_hz (Op.ref .blockPtr [128, 64] "v_tile_ptr") _
        (by simp [hpids1]) (by rw [evalOp_ref]; simp [hvp, taKVPtrTile]) hrow]
      refine congrArg some ?_
      ext idx
      simp only [hvT, fwdVTile, BlockState.setReg_readMem, BlockState.setReg_pids]))]
  -- L15: acc += dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_acc_eval _ 128 128 64 acc1T pT vT (by simp [hacc1]) (by simp [hpT]) (by simp [hvT])))]
  -- L16: l_prev = l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_reg_carry_eval _ 128 "l_curr" lcurrT (by simp [hlc])))]
  -- L17: m_prev = m_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_reg_carry_eval _ 128 "m_curr" mcurrT (by simp [hmc])))]
  -- L18: k_tile_ptr advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_advance_row_eval _ K 0 1024 64 128 64 64 1 (off_hz * 128) 128 "k_tile_ptr" (by simp [hkp, taKVPtrTile])))]
  -- L19: v_tile_ptr advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_advance_row_eval _ V 0 1024 64 128 64 64 1 (off_hz * 128) 128 "v_tile_ptr" (by simp [hvp, taKVPtrTile])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, mcurrT, lcurrT, lrcpT, pT, acc1T, rfl, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp [hmc]
  · simp [hlc]
  · simp [hacc1, hpT, hvT]
  · simp [taKVPtrTile]
  · simp [taKVPtrTile]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution.** The 13 deterministic preLoop statements step a clean
state to the loop-entry state, exposing all register readbacks the loop body / the
invariant at counter 0 need: the index vectors, the `m_prev = ⊥`/`l_prev = 0`/
`acc = 0` running-state init, the loaded `q = fwdQTile`, the K/V/Out block pointers
at offset `[off_hz·128 (+start_m·128 for Q/Out), 0]`, and `off_hz`. -/
theorem taPreLoop_eval (s : BlockState) (Q K V L M Out : RegionName) (sc : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (taPreLoop Q K V Out sc) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val))
      ∧ s0.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))
      ∧ s0.regs .real [128] "m_prev" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [128] "l_prev" = some ⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩
      ∧ s0.regs .real [128, 64] "acc" = some ⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩
      ∧ s0.regs .real [128, 64] "q" = some ⟨fun idx : TileIndex [128, 64] => some (fwdQTile s Q idx)⟩
      ∧ s0.regs .blockPtr [128, 64] "k_tile_ptr" = some (taKVPtrTile K (s.pids 1 * 128))
      ∧ s0.regs .blockPtr [128, 64] "v_tile_ptr" = some (taKVPtrTile V (s.pids 1 * 128))
      ∧ s0.regs .blockPtr [128, 64] "out_tile_ptr" = some
          (⟨fun _ : TileIndex [128, 64] =>
            { region := Out, baseOffset := 0, parentShape := [1024, 64],
              blockShape := [128, 64], strides := [64, 1],
              offsets := [s.pids 1 * 128 + s.pids 0 * 128, 0] }⟩) := by
  unfold taPreLoop
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: off_hz = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: offs_m = start_m*128 + arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)) _
        = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 3: offs_n = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange 128 _))]
  -- stmt 4: m_prev = zeros + (-inf) = ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩ : Tile .real [128]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, WithBot.realAdd, Option.map₂_none_right]
      rfl))]
  -- stmt 5: l_prev = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩ : Tile .real [128]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  -- stmt 6: acc = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 64] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ : Tile .real [128, 64]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  -- stmt 7: stride_qh_2d = 8192//64//1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 8192) (Op.constNat 64)) (Op.constNat 1)) _
        = some (Tile.scalar 128) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      rfl))]
  -- stmt 8: q_tile_ptr = makeBlockPtrDynOffsets
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets Q (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
          [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)), Op.constNat 0]) _
        = some (⟨fun _ : TileIndex [128, 64] =>
            { region := Q, baseOffset := 0, parentShape := [1024, 64],
              blockShape := [128, 64], strides := [64, 1],
              offsets := [s.pids 1 * 128 + s.pids 0 * 128, 0] }⟩) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 9: k_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets K (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
            Op.constNat 0]) _
        = some (taKVPtrTile K (s.pids 1 * 128)) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 10: v_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets V (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
            Op.constNat 0]) _
        = some (taKVPtrTile V (s.pids 1 * 128)) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 11: out_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets Out (Op.constNat 0) [1024, 64] [128, 64] [64, 1]
          [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)), Op.constNat 0]) _
        = some (⟨fun _ : TileIndex [128, 64] =>
            { region := Out, baseOffset := 0, parentShape := [1024, 64],
              blockShape := [128, 64], strides := [64, 1],
              offsets := [s.pids 1 * 128 + s.pids 0 * 128, 0] }⟩) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 12: q = load(q_tile_ptr)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_load_q_eval Q (s.pids 1) (s.pids 0) (Op.ref .blockPtr [128, 64] "q_tile_ptr") _
      (by simp [BlockState.setReg_pids]) (by simp [BlockState.setReg_pids])
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · refine congrArg some ?_; ext idx
    simp only [fwdQTile, BlockState.setReg_readMem, BlockState.setReg_pids, castTile_self]
  · simp
  · simp
  · simp

/-! ## Forward loop invariant + step + postLoop + full execution

The triton-attention `_fwd_kernel` streams the single (test-shape `N_CTX =
BLOCK_N = 128`, `num_block = 1`) causal KV block. Unlike `flash_attn` (which
defers normalization to a post-loop `out_buffer /= denom` and stores `L = max +
log2 denom`), this kernel normalizes **in loop**: every iteration multiplies the
running `acc`/`p` by `l_rcp = 1/l_curr`, so the register `acc` already holds the
final-normalized ratio `oPartial / lPartial`, and the stored `L = l_prev` is the
m-shifted normalizer `lPartial` itself (natural exp, no log).

The invariant binds the kernel's running registers after `c = i / 128` blocks to
the `FA1MathCausal` causal streaming accumulators (`Bk = 128`, `numKVBlocks = 1`,
`scale = sc`, `qStart = pids 0 · 128`): `m_prev = mPartial c`, `l_prev = lPartial
c`, and `acc = oPartial c / lPartial c` (the running ratio; at `c = 0` this is
`0 / 0 = 0`, exactly the kernel's `acc = 0` init). -/

open VeriTile.Examples.FA1MathCausal in
/-- Loop invariant for the triton-attention forward streaming loop (counter
`i = c · 128`, `c = i / 128` blocks processed). -/
noncomputable def taInvariant
    (Q K V Out : RegionName) (s0 : BlockState) (sc : ℝ)
    (i : Nat) (s : BlockState) : Prop :=
  let qS := s0.pids 0 * 128
  let qT := fwdQTile s0 Q
  let kT := fwdKTile s0 K
  let vT := fwdVTile s0 V
  s.pids = s0.pids ∧ i % 128 = 0 ∧ i ≤ 128 ∧
  (s.regs .real [128] "m_prev" = some ⟨fun r : TileIndex [128] =>
      mPartial 128 qS qT 1 kT sc (i / 128) r.1⟩) ∧
  (s.regs .real [128] "l_prev" = some ⟨fun r : TileIndex [128] =>
      ((lPartial 128 qS qT 1 kT sc (i / 128) r.1 : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .real [128, 64] "acc" = some ⟨fun idx : TileIndex [128, 64] =>
      ((oPartial 128 qS qT 1 kT vT sc (i / 128) idx
          / lPartial 128 qS qT 1 kT sc (i / 128) idx.1 : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .real [128, 64] "q" = some ⟨fun idx : TileIndex [128, 64] => some (qT idx)⟩) ∧
  (s.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => qS + r.val))) ∧
  (s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))) ∧
  (s.regs .blockPtr [128, 64] "k_tile_ptr" = some (taKVPtrTile K (s0.pids 1 * 128 + i))) ∧
  (s.regs .blockPtr [128, 64] "v_tile_ptr" = some (taKVPtrTile V (s0.pids 1 * 128 + i))) ∧
  (s.regs .blockPtr [128, 64] "out_tile_ptr" = some
    (⟨fun _ : TileIndex [128, 64] =>
      { region := Out, baseOffset := 0, parentShape := [1024, 64],
        blockShape := [128, 64], strides := [64, 1],
        offsets := [s0.pids 1 * 128 + s0.pids 0 * 128, 0] }⟩)) ∧
  (s.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1))) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

open VeriTile.Examples.FA1MathCausal in
/-- **Invariant base case.** The preLoop-output state `sp` (anchor `s = sp`)
satisfies `taInvariant … 0 sp`: at `c = 0` the running registers are
`mPartial 0 = ⊥`, `lPartial 0 = 0`, `oPartial 0 / lPartial 0 = 0 / 0 = 0`, which
are exactly the `m_prev = ⊥` / `l_prev = 0` / `acc = 0` preLoop inits. -/
theorem ta_invariant_zero
    (Q K V Out : RegionName) (sp : BlockState) (sc : ℝ)
    (hm : sp.regs .real [128] "m_prev" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩)
    (hl : sp.regs .real [128] "l_prev" = some ⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩)
    (hacc : sp.regs .real [128, 64] "acc" = some ⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩)
    (hq : sp.regs .real [128, 64] "q" = some ⟨fun idx : TileIndex [128, 64] => some (fwdQTile sp Q idx)⟩)
    (hoffm : sp.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => sp.pids 0 * 128 + r.val)))
    (hoffn : sp.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hkp : sp.regs .blockPtr [128, 64] "k_tile_ptr" = some (taKVPtrTile K (sp.pids 1 * 128)))
    (hvp : sp.regs .blockPtr [128, 64] "v_tile_ptr" = some (taKVPtrTile V (sp.pids 1 * 128)))
    (hop : sp.regs .blockPtr [128, 64] "out_tile_ptr" = some
        (⟨fun _ : TileIndex [128, 64] =>
          { region := Out, baseOffset := 0, parentShape := [1024, 64],
            blockShape := [128, 64], strides := [64, 1],
            offsets := [sp.pids 1 * 128 + sp.pids 0 * 128, 0] }⟩))
    (hoh : sp.regs .nat [] "off_hz" = some (Tile.scalar (sp.pids 1)))
    (hundef : ∀ rg o, sp.undef rg o = 0) :
    taInvariant Q K V Out sp sc 0 sp := by
  unfold taInvariant
  refine ⟨rfl, by norm_num, by norm_num, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hm]; refine congrArg some ?_; ext r; rfl
  · rw [hl]; refine congrArg some ?_; ext r; rfl
  · rw [hacc]; refine congrArg some ?_; ext idx
    show some (0 : ℝ) = ((oPartial 128 (sp.pids 0 * 128) (fwdQTile sp Q) 1 (fwdKTile sp K)
        (fwdVTile sp V) sc 0 idx / lPartial 128 (sp.pids 0 * 128) (fwdQTile sp Q) 1
        (fwdKTile sp K) sc 0 idx.1 : ℝ) : WithBot ℝ)
    show some (0 : ℝ) = ((0 / 0 : ℝ) : WithBot ℝ)
    rw [div_zero]; rfl
  · rw [hq]
  · rw [hoffm]
  · rw [hoffn]
  · rw [hkp]; simp
  · rw [hvp]; simp
  · rw [hop]
  · rw [hoh]
  · exact hundef
  · rfl

/-! ### Step-cell bridges (RECIPE → FA1 streaming)

The following per-cell lemmas connect the symbolic register tiles exposed by
`taLoopBody_steps` to the `FA1MathCausal` streaming accumulators at `c + 1`, for
the single test-shape block (`Bk = 128`, `numKVBlocks = 1`, so `blockIndex 128 1
0 _ jL = jL`). These are the triton (natural-exp, in-loop-`l_rcp`) analogues of
the flash `*_op_eval`/`*_reg_eq` family. -/

open VeriTile.Examples.FA1MathCausal in
/-- The `q·kᵀ` dot cell, with `q = fwdQTile` and `k = fwdKTile` loaded, equals the
real `Σ_e qT(r,e)·kT(j,e)` (the unscaled score). -/
theorem ta_dot_score_cell (qT kT : TileIndex [128, 64] → ℝ) (r j : Fin 128) :
    (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
        (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))).data
        (r, j, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin 64 => qT (r, e, PUnit.unit) * kT (j, e, PUnit.unit))) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·)
          ((⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64]).data (r, e, PUnit.unit))
          ((Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64])).data (e, j, PUnit.unit))))
      = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
          (fun e => (some (qT (r, e, PUnit.unit) * kT (j, e, PUnit.unit)) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by
        rw [Tile.transpose_nil_data]; rfl)]
  rw [WithBot.sum_someTerm_eq_some]

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- The masked qk cell `qk1(r,j)` equals `maskedScore qS qT kT sc r j` (global key
`j`, single block `c = 0`). -/
theorem ta_qk1_cell (qT kT : TileIndex [128, 64] → ℝ) (sc : ℝ) (qS : Nat) (r j : Fin 128) :
    (if 0 + j.val ≤ qS + r.val then
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
            (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
              (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
          (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
      else (⊥ : WithBot ℝ))
      = maskedScore qS qT kT sc r j := by
  by_cases h : 0 + j.val ≤ qS + r.val
  · rw [if_pos h]
    rw [Tile.bop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarR, Tile.scalar_data,
      NumericDType.mul, NumericDType.add]
    rw [ta_dot_score_cell qT kT r j]
    rw [maskedScore_of_le qS qT kT sc r j (by omega)]
    simp only [StreamingAccumulator.scaledScore, WithBot.realAdd, WithBot.realMul,
      Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_; ring
  · rw [if_neg h, maskedScore_of_not_le qS qT kT sc r j (by omega)]

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- The block sup of the masked qk cells equals `mPartial 1` (`= max ⊥ (sup …)`). -/
theorem ta_rmax_eq_mPartial1 (qT kT : TileIndex [128, 64] → ℝ) (sc : ℝ) (qS : Nat) (r : Fin 128) :
    (Finset.univ : Finset (Fin 128)).sup' ⟨⟨0, by norm_num⟩, Finset.mem_univ _⟩
        (fun j : Fin 128 =>
          if 0 + j.val ≤ qS + r.val then
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
                  (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
              (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
          else (⊥ : WithBot ℝ))
      = mPartial 128 qS qT 1 kT sc 1 r := by
  rw [Finset.sup'_eq_sup]
  rw [show mPartial 128 qS qT 1 kT sc 1 r = mPartial 128 qS qT 1 kT sc (0 + 1) r from rfl]
  rw [mPartial_succ_of_lt (Bk := 128) qS qT 1 kT sc 0 (by norm_num) r]
  rw [show mPartial 128 qS qT 1 kT sc 0 r = (⊥ : WithBot ℝ) from rfl]
  rw [max_eq_right bot_le]
  refine Finset.sup_congr rfl (fun j _ => ?_)
  rw [ta_qk1_cell qT kT sc qS r j]
  congr 1
  apply Fin.ext; simp [StreamingAccumulator.blockIndex]

open VeriTile.Examples.FA1MathCausal in
/-- The `pexp` cell `exp(qk1(r,j) − Mr)` (where `Mr = (mPartial 1).unbotD 0`)
equals `some` of the masked-score exp term used by `lPartial`/`oPartial` succ. -/
theorem ta_pexp_cell (qT kT : TileIndex [128, 64] → ℝ) (sc : ℝ) (qS : Nat) (r j : Fin 128) :
    WithBot.realExp (WithBot.realSub
        (if 0 + j.val ≤ qS + r.val then
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
                  (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
              (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
          else (⊥ : WithBot ℝ))
        (mPartial 128 qS qT 1 kT sc 1 r))
      = some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) j))
          (mPartial 128 qS qT 1 kT sc 1 r))).unbotD 0) := by
  rw [ta_qk1_cell qT kT sc qS r j]
  rw [show StreamingAccumulator.blockIndex 128 1 0 (by norm_num) j = j from by
    apply Fin.ext; simp [StreamingAccumulator.blockIndex]]
  exact realExp_eq_some_unbotD _

open VeriTile.Examples.FA1MathCausal in
/-- The masked exp block-sum cell `Σ_j pexp(r,j)` equals `some` of the `lPartial`
succ block term (the `Σ_jL (exp(maskedScore − mPartial1)).unbotD` sum). -/
theorem ta_pexp_block_sum (qT kT : TileIndex [128, 64] → ℝ) (sc : ℝ) (qS : Nat) (r : Fin 128) :
    (Tile.reduceSumDrop (⟨1, by decide⟩ : Fin [128, 128].length)
        (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (⟨fun idx : TileIndex [128, 128] =>
              if 0 + idx.2.1.val ≤ qS + idx.1.val then
                (Tile.bop NumericDType.real.mul Broadcast.scalarR
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                    (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
                      (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
                  (Tile.scalar (some sc : WithBot ℝ))).data idx
              else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128])
            (Tile.expandDim ⟨1, by decide⟩
              (⟨fun r : TileIndex [128] => mPartial 128 qS qT 1 kT sc 1 r.1⟩ : Tile .real [128]))))).data
        (r, PUnit.unit)
      = some ((Finset.univ : Finset (Fin 128)).sum (fun jLocal =>
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal))
            (mPartial 128 qS qT 1 kT sc 1 r))).unbotD 0)) := by
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ k : Fin (TileShape.axisDim [128, 128] (⟨1, by decide⟩ : Fin [128, 128].length)),
      (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (⟨fun idx : TileIndex [128, 128] =>
              if 0 + idx.2.1.val ≤ qS + idx.1.val then
                (Tile.bop NumericDType.real.mul Broadcast.scalarR
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                    (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
                      (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
                  (Tile.scalar (some sc : WithBot ℝ))).data idx
              else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128])
            (Tile.expandDim ⟨1, by decide⟩
              (⟨fun r : TileIndex [128] => mPartial 128 qS qT 1 kT sc 1 r.1⟩ : Tile .real [128])))).data
          (TileShape.insertAxisIndex [128, 128] (⟨1, by decide⟩ : Fin [128, 128].length) (r, PUnit.unit) k)
        = (some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) k))
            (mPartial 128 qS qT 1 kT sc 1 r))).unbotD 0) : WithBot ℝ) := by
    intro k
    rw [show (TileShape.insertAxisIndex [128, 128] (⟨1, by decide⟩ : Fin [128, 128].length)
          (r, PUnit.unit) k) = (r, k, PUnit.unit) from rfl]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    exact ta_pexp_cell qT kT sc qS r k
  rw [Finset.sum_congr rfl (fun k _ => hcell k)]
  rw [WithBot.sum_someTerm_eq_some]
  rfl

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- The `p·v` dot cell `Σ_j (pexp(r,j)·lrcp_r)·v(j,d)` equals `some` of
`lrcp_r · oPartial 1 (r,d)` (the `Σ_jL (exp(maskedScore−mPartial1)).unbotD · V`
block, scaled by the row's `l_rcp`). -/
theorem ta_pv_dot_block (qT kT vT : TileIndex [128, 64] → ℝ) (sc : ℝ) (qS : Nat)
    (r : Fin 128) (d : Fin 64) (lrcpT : Tile .real [128]) (lrcpVal : ℝ)
    (hlrcp : lrcpT.data (r, PUnit.unit) = some lrcpVal) :
    (Tile.dot []
        (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (⟨fun idx : TileIndex [128, 128] =>
                if 0 + idx.2.1.val ≤ qS + idx.1.val then
                  (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                      (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
                        (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
                    (Tile.scalar (some sc : WithBot ℝ))).data idx
                else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128])
              (Tile.expandDim ⟨1, by decide⟩
                (⟨fun r : TileIndex [128] => mPartial 128 qS qT 1 kT sc 1 r.1⟩ : Tile .real [128]))))
          (Tile.expandDim ⟨1, by decide⟩ lrcpT))
        (⟨fun idx => some (vT idx)⟩ : Tile .real [128, 64])).data (r, d, PUnit.unit)
      = some (lrcpVal * (Finset.univ : Finset (Fin 128)).sum (fun jLocal =>
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal))
            (mPartial 128 qS qT 1 kT sc 1 r))).unbotD 0
            * vT (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal, d, PUnit.unit))) := by
  rw [Tile.dot_nil_data]
  refine Eq.trans (b := @Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ
      (fun j => (some (lrcpVal *
        ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) j))
          (mPartial 128 qS qT 1 kT sc 1 r))).unbotD 0
          * vT (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) j, d, PUnit.unit))) : WithBot ℝ)))
    (Finset.sum_congr rfl (fun j (_ : j ∈ Finset.univ) => ?_)) ?_
  · rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.mul]
    rw [hlrcp]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    rw [ta_pexp_cell qT kT sc qS r j]
    rw [show StreamingAccumulator.blockIndex 128 1 0 (by norm_num) j = j from by
      apply Fin.ext; simp [StreamingAccumulator.blockIndex]]
    simp only [WithBot.realMul, Option.map₂_some_some]
    refine congrArg some ?_; ring
  · rw [WithBot.sum_someTerm_eq_some, ← Finset.mul_sum]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **Step lemma.** One loop iteration advances `taInvariant` by one block. The
test shape has a single KV block (`numKVBlocks = 1`), so the only reachable
counter is `i = 0` (`c = 0`); the body's `m_curr`/`l_curr`/`l_rcp`-normalized
`acc` cells advance the running registers to `mPartial 1` / `lPartial 1` /
`oPartial 1 / lPartial 1`, read off via the FA1 succ bridges. -/
theorem ta_attn_step (Q K V Out : RegionName) (s0 : BlockState) (sc : ℝ)
    (i : Nat) (s : BlockState) (hilt : i < 128)
    (hpid1 : ∀ j : Fin 128, s0.pids 1 * 128 + j.val < 1024)
    (hinv : taInvariant Q K V Out s0 sc i s) :
    ∃ s', stepStmts (taLoopBody sc) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ taInvariant Q K V Out s0 sc (i + 128) s' := by
  simp only [taInvariant] at hinv
  obtain ⟨hpids, hmod, hile, hmp, hlp, hacc, hq, hoffm, hoffn, hkp, hvp, hop, hoh, hundef, hmem⟩ := hinv
  -- single block: i = 0
  have hi0 : i = 0 := by omega
  subst hi0
  set qS := s0.pids 0 * 128 with hqS
  set qT := fwdQTile s0 Q with hqT
  set kT := fwdKTile s0 K with hkT
  set vT := fwdVTile s0 V with hvT
  -- entry-state registers (after setReg start_n) feed taLoopBody_steps
  have hpids1 : (s.setReg "start_n" .nat [] (Tile.scalar 0)).pids 1 = s0.pids 1 := by
    rw [BlockState.setReg_pids, hpids]
  have hrow : ∀ j : Fin 128, s0.pids 1 * 128 + j.val < 1024 := hpid1
  -- the loaded q/k/v tiles read s0's memory (setReg preserves mem/pids)
  obtain ⟨rmaxT, qk0, qk1, hrm, hqk0eq, hqk1eq, sF, hchain, hpidsF, hmemF, hundefF,
      mcurrT, lcurrT, lrcpT, pT, acc1T, hmcd, hpTd, hlcd, hlrd, hacc1d,
      hmF, hlF, haccF, hkpF, hvpF, hqF, hoffmF, hoffnF, hohF, hopF⟩ :=
    taLoopBody_steps K V (s0.pids 1) sc (s.setReg "start_n" .nat [] (Tile.scalar 0))
      (⟨fun idx : TileIndex [128, 64] => some (qT idx)⟩)
      (⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩)
      (⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩)
      (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩)
      (fun r : Fin 128 => qS + r.val)
      hpids1 hrow
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffm)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hmp]; refine congrArg some ?_; ext r; rfl)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hlp]; refine congrArg some ?_; ext r; rfl)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hacc]; refine congrArg some ?_; ext idx
          show ((oPartial 128 qS qT 1 kT vT sc (0 / 128) idx
              / lPartial 128 qS qT 1 kT sc (0 / 128) idx.1 : ℝ) : WithBot ℝ) = some (0 : ℝ)
          rw [show (0 / 128 : Nat) = 0 from rfl,
              show oPartial 128 qS qT 1 kT vT sc 0 idx = 0 from rfl,
              show lPartial 128 qS qT 1 kT sc 0 idx.1 = 0 from rfl, div_zero]; rfl)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hkp]; simp only [Nat.add_zero, hpids])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hvp]; simp only [Nat.add_zero, hpids])
      (by intro rg o; rw [BlockState.setReg_undef]; exact hundef rg o)
  refine ⟨sF, hchain, ?_⟩
  -- the chain's k/v load tiles read s.mem = s0.mem, so they equal fwdKTile/fwdVTile s0
  -- (taLoopBody_steps already returns them as fwdKTile/fwdVTile of the entry state,
  --  which has mem = s.mem = s0.mem and pids = s0.pids)
  have hentryK : fwdKTile (s.setReg "start_n" .nat [] (Tile.scalar 0)) K = kT := by
    funext idx; simp only [hkT, fwdKTile, BlockState.setReg_readMem, BlockState.setReg_pids, hpids]
    unfold BlockState.readMem; rw [hmem]
  have hentryV : fwdVTile (s.setReg "start_n" .nat [] (Tile.scalar 0)) V = vT := by
    funext idx; simp only [hvT, fwdVTile, BlockState.setReg_readMem, BlockState.setReg_pids, hpids]
    unfold BlockState.readMem; rw [hmem]
  -- rmaxT cell = mPartial 1
  -- the qk1 cell of the chain equals `maskedScore` (entry tiles read s0's mem)
  have hqk1cell : ∀ r j : Fin 128, qk1.data (r, j, PUnit.unit) = maskedScore qS qT kT sc r j := by
    intro r j
    rw [hqk1eq]
    simp only [hqk0eq, hentryK]
    exact ta_qk1_cell qT kT sc qS r j
  have hrmaxCell : ∀ r : Fin 128, rmaxT.data (r, PUnit.unit) = mPartial 128 qS qT 1 kT sc 1 r := by
    intro r
    unfold Tile.reduceMaxDrop at hrm
    rw [dif_pos (show 0 < TileShape.axisDim [128, 128] (⟨1, by decide⟩ : Fin [128, 128].length) from by decide)] at hrm
    rw [← Option.some.inj hrm]
    show (Finset.univ : Finset (Fin 128)).sup'
        ⟨⟨0, by decide⟩, Finset.mem_univ _⟩ (fun k =>
        qk1.data ((r, k, PUnit.unit) : TileIndex [128, 128])) = _
    rw [← ta_rmax_eq_mPartial1 qT kT sc qS r]
    refine Finset.sup'_congr _ rfl ?_
    intro j _
    rw [hqk1cell r j]
    exact (ta_qk1_cell qT kT sc qS r j).symm
  -- mPartial 1 ≠ ⊥ (single nonempty block)
  have hmne : ∀ r : Fin 128, mPartial 128 qS qT 1 kT sc 1 r ≠ ⊥ := by
    intro r
    exact mPartial_succ_ne_bot (Bk := 128) (by norm_num) qS qT 1 kT sc 0 (by norm_num) r
  -- mcurrT = the mPartial 1 tile
  have hmcurrCell : ∀ r : Fin 128,
      mcurrT.data (r, PUnit.unit) = mPartial 128 qS qT 1 kT sc 1 r := by
    intro r
    rw [hmcd, Tile.select_data, Tile.cop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hrmaxCell r]
    have hgt : (mPartial 128 qS qT 1 kT sc 1 r > (⊥ : WithBot ℝ)) :=
      lt_of_le_of_ne bot_le (Ne.symm (hmne r))
    rw [decide_eq_true hgt]; simp
  have hmcurrTile : mcurrT = ⟨fun r : TileIndex [128] => mPartial 128 qS qT 1 kT sc 1 r.1⟩ := by
    ext r; exact hmcurrCell r.1
  -- lcurrT = some (lPartial 1)
  have hlcurrCell : ∀ r : Fin 128,
      lcurrT.data (r, PUnit.unit) = some (lPartial 128 qS qT 1 kT sc 1 r) := by
    intro r
    rw [hlcd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    -- reduce qk1 → if-form and mcurrT → mPartial1-tile inside the reduceSum
    rw [show qk1 = (⟨fun idx : TileIndex [128, 128] =>
          if 0 + idx.2.1.val ≤ qS + idx.1.val then
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
                  (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
              (Tile.scalar (some sc : WithBot ℝ))).data idx
          else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128]) from by
      rw [hqk1eq]; ext idx; simp only [hqk0eq, hentryK]]
    rw [hmcurrTile]
    -- the lprev1 term: ltile (some 0) * alpha = some 0
    erw [ta_pexp_block_sum qT kT sc qS r]
    rw [show (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
          (⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩ : Tile .real [128])
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
              (⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩ : Tile .real [128])
              ⟨fun r : TileIndex [128] => mPartial 128 qS qT 1 kT sc 1 r.1⟩))).data (r, PUnit.unit)
        = some (0 : ℝ) from by
      rw [Tile.bop_data, Tile.uop_data, Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub]
      rw [WithBot.realSub_bot_left, WithBot.realExp_bot]
      show WithBot.realMul (some 0) ((0 : ℝ) : WithBot ℝ) = some 0
      rw [WithBot.realMul, Option.map₂_some_coe, mul_zero]]
    simp only [NumericDType.add, WithBot.realAdd, Option.map₂_some_some]
    refine congrArg some ?_
    rw [show mPartial 128 qS qT 1 kT sc 1 r = mPartial 128 qS qT 1 kT sc (0 + 1) r from rfl]
    rw [lPartial_succ_of_lt (Bk := 128) qS qT 1 kT sc 0 (by norm_num) r]
    simp only
    rw [show lPartial 128 qS qT 1 kT sc 0 r = (0 : ℝ) from rfl]
    ring
  -- lrcp cell = some (1 / lPartial 1)
  have hlrcpCell : ∀ r : Fin 128,
      lrcpT.data (r, PUnit.unit) = some (1 / lPartial 128 qS qT 1 kT sc 1 r) := by
    intro r
    rw [hlrd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarL, Tile.scalar_data,
      NumericDType.div]
    rw [hlcurrCell r]
    show WithBot.realDiv (some (1.0 : ℝ)) (some (lPartial 128 qS qT 1 kT sc 1 r))
      = some (1 / lPartial 128 qS qT 1 kT sc 1 r)
    rw [WithBot.realDiv, Option.map₂_some_some]; norm_num
  -- oPartial 1 numerator block sum = oPartial 1
  have hoPartial1 : ∀ (r : Fin 128) (d : Fin 64),
      (Finset.univ : Finset (Fin 128)).sum (fun jLocal =>
        (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal))
          (mPartial 128 qS qT 1 kT sc 1 r))).unbotD 0
          * vT (StreamingAccumulator.blockIndex 128 1 0 (by norm_num) jLocal, d, PUnit.unit))
        = oPartial 128 qS qT 1 kT vT sc 1 (r, d, PUnit.unit) := by
    intro r d
    -- unfold oPartial at iteration 1 directly (literal `1` keeps `mPartial 1`, not `0+1`)
    have hm01 : ∀ rr : Fin 128,
        mPartial 128 qS qT 1 kT sc (0 + 1) rr = mPartial 128 qS qT 1 kT sc 1 rr := fun _ => rfl
    conv_rhs => rw [oPartial]
    rw [dif_pos (show (0 : Nat) + 1 ≤ 1 by decide)]
    simp only [hm01, show oPartial 128 qS qT 1 kT vT sc 0 (r, d, PUnit.unit) = 0 from rfl,
      mul_zero, zero_add]
  -- final acc cell = some (oPartial 1 / lPartial 1)
  have haccCell : ∀ idx : TileIndex [128, 64],
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT (⟨fun idx => some (vT idx)⟩ : Tile .real [128, 64]))).data idx
        = some (oPartial 128 qS qT 1 kT vT sc 1 idx
            / lPartial 128 qS qT 1 kT sc 1 idx.1) := by
    intro idx
    obtain ⟨r, d, u⟩ := idx; cases u
    rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    -- acc1T cell = some 0 (acctile/ltile = some 0, alpha = exp(⊥-_) = some 0)
    rw [show acc1T.data (r, d, PUnit.unit) = some (0 : ℝ) from by
      rw [hacc1d, Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub,
        Tile.expandDim_data, TileShape.dropInsertedIndex, Tile.bop_data, Tile.uop_data]
      rw [hlrcpCell r]
      show WithBot.realMul (some (0:ℝ))
        (WithBot.realMul (WithBot.realMul (some (0:ℝ))
          (WithBot.realExp (WithBot.realSub (⊥ : WithBot ℝ) (mcurrT.data (r, PUnit.unit)))))
          (some (1 / lPartial 128 qS qT 1 kT sc 1 r))) = some 0
      rw [WithBot.realSub_bot_left, WithBot.realExp_bot]
      show WithBot.realMul (some (0:ℝ))
        (WithBot.realMul (WithBot.realMul (some (0:ℝ)) ((0:ℝ) : WithBot ℝ))
          (some (1 / lPartial 128 qS qT 1 kT sc 1 r))) = some 0
      simp only [WithBot.realMul, Option.map₂_some_coe, Option.map₂_some_some, zero_mul, mul_zero]]
    -- dot pT vT cell via ta_pv_dot_block (rewrite qk1, mcurrT, lrcp)
    rw [show pT = (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (⟨fun idx : TileIndex [128, 128] =>
                if 0 + idx.2.1.val ≤ qS + idx.1.val then
                  (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                      (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [128, 64])
                        (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [128, 64]))))
                    (Tile.scalar (some sc : WithBot ℝ))).data idx
                else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128])
              (Tile.expandDim ⟨1, by decide⟩
                (⟨fun r : TileIndex [128] => mPartial 128 qS qT 1 kT sc 1 r.1⟩ : Tile .real [128]))))
          (Tile.expandDim ⟨1, by decide⟩ lrcpT)) from by
      rw [hpTd, hmcurrTile, hqk1eq]
      congr 2
      ext idx; simp only [hqk0eq, hentryK]]
    erw [ta_pv_dot_block qT kT vT sc qS r d lrcpT (1 / lPartial 128 qS qT 1 kT sc 1 r)
      (hlrcpCell r)]
    rw [hoPartial1 r d]
    simp only [WithBot.realAdd, Option.map₂_some_some]
    refine congrArg some ?_
    rw [zero_add]
    ring
  -- assemble the invariant at i + 128 = 128 (c = 1)
  have hcdiv : (0 + 128) / 128 = 1 := by norm_num
  refine ⟨?_, by norm_num, by norm_num, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, BlockState.setReg_pids, hpids]
  · -- m_prev = mPartial 1
    rw [hmF, hmcurrTile, hcdiv]
  · -- l_prev = lPartial 1
    rw [hlF, hcdiv]; refine congrArg some ?_; ext r; rw [hlcurrCell r.1]; rfl
  · -- acc = oPartial 1 / lPartial 1
    rw [haccF, hcdiv]; refine congrArg some ?_; ext idx
    simp only [hentryV]; rw [haccCell idx]; rfl
  · -- q preserved
    rw [hqF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq
  · -- offs_m preserved
    rw [hoffmF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffm
  · -- offs_n preserved
    rw [hoffnF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn
  · -- k_tile_ptr advanced
    rw [hkpF]
  · -- v_tile_ptr advanced
    rw [hvpF]
  · -- out_tile_ptr preserved
    rw [hopF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hop
  · -- off_hz preserved
    rw [hohF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoh
  · -- undef
    exact hundefF
  · -- mem
    rw [hmemF]; funext region offset; rw [BlockState.setReg_mem]
    exact congrFun (congrFun hmem region) offset

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **PostLoop execution + genuine readbacks.** Stepping the 8 post-loop statements
from a loop-end state satisfying `taInvariant … 128` (full single block, `c = 1`),
the kernel's `L` store holds the genuine m-shifted normalizer `lPartial 1`, the `M`
store holds the running max `(mPartial 1).unbotD 0`, and the boundary-checked `Out`
store holds the in-loop-`l_rcp`-normalized ratio `oPartial 1 / lPartial 1` at every
active lane. -/
theorem ta_postLoop
    (Q K V L M Out : RegionName) (s0 : BlockState) (sc : ℝ) (s : BlockState)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (hinv : taInvariant Q K V Out s0 sc 128 s) :
    ∃ sP, stepStmts (taPostLoop L M Out) s = some sP
      ∧ (∀ idx : TileIndex [128, 64],
          active s0 (s0.pids 1 * 128) 1024 128 idx →
          sP.mem Out (outOffset s0 (s0.pids 1 * 128) 64 1 128 idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (oPartial 128 (s0.pids 0 * 128) (fwdQTile s0 Q) 1 (fwdKTile s0 K)
                    (fwdVTile s0 V) sc 1 idx
                  / lPartial 128 (s0.pids 0 * 128) (fwdQTile s0 Q) 1 (fwdKTile s0 K) sc 1 idx.1))))
      ∧ (∀ i : Fin 128,
          sP.readMem L (lRowOffset s0 (s0.pids 1) 128 128 i)
            = lPartial 128 (s0.pids 0 * 128) (fwdQTile s0 Q) 1 (fwdKTile s0 K) sc 1 i)
      ∧ (∀ i : Fin 128,
          sP.readMem M (lRowOffset s0 (s0.pids 1) 128 128 i)
            = (mPartial 128 (s0.pids 0 * 128) (fwdQTile s0 Q) 1 (fwdKTile s0 K) sc 1 i).unbotD 0) := by
  simp only [taInvariant] at hinv
  obtain ⟨hpids, _, _, hmp, hlp, hacc, hq, hoffm, hoffn, hkp, hvp, hop, hoh, hundef, hmem⟩ := hinv
  set qS := s0.pids 0 * 128 with hqS
  set qT := fwdQTile s0 Q with hqT
  set kT := fwdKTile s0 K with hkT
  set vT := fwdVTile s0 V with hvT
  -- the running registers at c = 1 (128/128 = 1)
  rw [show (128 : Nat) / 128 = 1 from rfl] at hmp hlp hacc
  set mTile : Tile .real [128] := ⟨fun r : TileIndex [128] => mPartial 128 qS qT 1 kT sc 1 r.1⟩ with hmTile
  set lTile : Tile .real [128] := ⟨fun r : TileIndex [128] => ((lPartial 128 qS qT 1 kT sc 1 r.1 : ℝ) : WithBot ℝ)⟩ with hlTile
  set accTile : Tile .real [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      ((oPartial 128 qS qT 1 kT vT sc 1 idx / lPartial 128 qS qT 1 kT sc 1 idx.1 : ℝ) : WithBot ℝ)⟩ with haccTile
  unfold taPostLoop
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: offs_m = start_m*128 + arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)) _
        = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 2: l_ptrs = L + off_hz*128 + offs_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase L)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))
          (Op.ref .nat [128] "offs_m"))) _
        = some (⟨fun r : TileIndex [128] => (Region.cast L, s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val))⟩
            : Tile .ptr [128]) from by
      simp only [evalOp, evalOp.eq_def, evalOp_ref, evalOp_constNat,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same, BlockState.setReg_pids, hoh, Option.bind_eq_bind, Option.bind_some,
        Option.map_some]
      refine congrArg some ?_; ext r
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.vec_data,
          Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, NumericDType.add, NumericDType.mul, Nat.zero_add]))]
  -- stmt 3: m_ptrs = M + off_hz*128 + offs_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))
          (Op.ref .nat [128] "offs_m"))) _
        = some (⟨fun r : TileIndex [128] => (Region.cast M, s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val))⟩
            : Tile .ptr [128]) from by
      simp only [evalOp, evalOp.eq_def, evalOp_ref, evalOp_constNat,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same, BlockState.setReg_pids, hoh, Option.bind_eq_bind, Option.bind_some,
        Option.map_some]
      refine congrArg some ?_; ext r
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.vec_data,
          Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, NumericDType.add, NumericDType.mul, Nat.zero_add]))]
  -- name the post-stmt-3 state
  set s3 := BlockState.setReg
      (BlockState.setReg
        (BlockState.setReg
          (BlockState.setReg s "start_m" .nat [] (Tile.scalar (s.pids 0)))
          "offs_m" .nat [128] (Tile.vec fun r : Fin 128 => s.pids 0 * 128 + r.val))
        "l_ptrs" .ptr [128] ⟨fun r : TileIndex [128] => (Region.cast L, s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val))⟩)
      "m_ptrs" .ptr [128] ⟨fun r : TileIndex [128] => (Region.cast M, s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val))⟩
    with hs3
  -- register readbacks in s3
  have hlprev3 : s3.regs .real [128] "l_prev" = some lTile := by
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hlp]
  have hmprev3 : s3.regs .real [128] "m_prev" = some mTile := by
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hmp]
  have hlptr3 : s3.regs .ptr [128] "l_ptrs"
      = some (⟨fun r : TileIndex [128] => (Region.cast L, s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val))⟩
          : Tile .ptr [128]) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same]
  have hmptr3 : s3.regs .ptr [128] "m_ptrs"
      = some (⟨fun r : TileIndex [128] => (Region.cast M, s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val))⟩
          : Tile .ptr [128]) := by
    rw [hs3, BlockState.setReg_same]
  -- L store offset/value functions
  set lOffFn : TileIndex [128] → Nat := fun r => s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val) with hlOffFn
  -- stmt 4: store L via l_ptrs (l_prev)
  have hstore4 : stepStmt (Stmt.store .real [128] (MemAccess.ptr (Op.ref .ptr [128] "l_ptrs"))
      (Op.ref .real [128] "l_prev") MaskOpt.none) s3
      = some ((TileShape.allIndices [128]).foldl
          (fun acc r => acc.writeMemTyped .real (Region.cast L) (lOffFn r) (lTile.data r)) s3) := by
    unfold stepStmt
    simp only [evalOp_ref, hlprev3, hlptr3, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rfl
  rw [stepStmts.cons_some hstore4]
  set s4 := (TileShape.allIndices [128]).foldl
      (fun acc r => acc.writeMemTyped .real (Region.cast L) (lOffFn r) (lTile.data r)) s3 with hs4
  -- m_prev/m_ptrs readback in s4 (stores only touch mem)
  have hmprev4 : s4.regs .real [128] "m_prev" = some mTile := by
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]; exact hmprev3
  have hmptr4 : s4.regs .ptr [128] "m_ptrs"
      = some (⟨fun r : TileIndex [128] => (Region.cast M, s0.pids 1 * 128 + (s.pids 0 * 128 + r.1.val))⟩
          : Tile .ptr [128]) := by
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]; exact hmptr3
  -- stmt 5: store M via m_ptrs (m_prev)
  have hstore5 : stepStmt (Stmt.store .real [128] (MemAccess.ptr (Op.ref .ptr [128] "m_ptrs"))
      (Op.ref .real [128] "m_prev") MaskOpt.none) s4
      = some ((TileShape.allIndices [128]).foldl
          (fun acc r => acc.writeMemTyped .real (Region.cast M) (lOffFn r) (mTile.data r)) s4) := by
    unfold stepStmt
    simp only [evalOp_ref, hmprev4, hmptr4, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rfl
  rw [stepStmts.cons_some hstore5]
  set s5 := (TileShape.allIndices [128]).foldl
      (fun acc r => acc.writeMemTyped .real (Region.cast M) (lOffFn r) (mTile.data r)) s4 with hs5
  -- acc readback in s5
  have hacc5 : s5.regs .real [128, 64] "acc" = some accTile := by
    rw [hs5]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hacc]
  have hout5 : s5.regs .blockPtr [128, 64] "out_tile_ptr"
      = some (⟨fun _ : TileIndex [128, 64] =>
          { region := Out, baseOffset := 0, parentShape := [1024, 64], blockShape := [128, 64],
            strides := [64, 1], offsets := [s0.pids 1 * 128 + s0.pids 0 * 128, 0] }⟩
          : Tile .blockPtr [128, 64]) := by
    rw [hs5]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hop]
  -- stmt 6: acc = castFloat real→fp16 acc
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "acc")) s5
        = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx)⟩ : Tile .fp16 [128, 64]) from by
      rw [evalOp_castFloat]; erw [evalOp_ref, hacc5]; rfl))]
  set s6 := s5.setReg "acc" .fp16 [128, 64]
      ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx)⟩ with hs6
  have hout6 : s6.regs .blockPtr [128, 64] "out_tile_ptr"
      = some (⟨fun _ : TileIndex [128, 64] =>
          { region := Out, baseOffset := 0, parentShape := [1024, 64], blockShape := [128, 64],
            strides := [64, 1], offsets := [s0.pids 1 * 128 + s0.pids 0 * 128, 0] }⟩
          : Tile .blockPtr [128, 64]) := by
    rw [hs6, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hout5
  have hacc6 : s6.regs .fp16 [128, 64] "acc"
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx)⟩ : Tile .fp16 [128, 64]) := by
    rw [hs6, BlockState.setReg_same]
  -- Out store offset/value functions
  set oOffFn : TileIndex [128, 64] → Nat :=
    fun idx => (s0.pids 1 * 128 + s0.pids 0 * 128 + idx.1.val) * 64 + idx.2.1.val * 1 with hoOffFn
  set oValFn : TileIndex [128, 64] → TileCarrier TileDType.fp16 :=
    fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx) with hoValFn
  set oMask : TileIndex [128, 64] → Prop :=
    fun idx => s0.pids 1 * 128 + s0.pids 0 * 128 + idx.1.val < 1024 ∧ 0 + idx.2.1.val < 64 with hoMask
  -- stmt 7: store Out via out_tile_ptr [0,1] (boundary-checked, masked)
  have hstore7 : stepStmt (Stmt.store .fp16 [128, 64]
      (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "out_tile_ptr") [0, 1])
      (Op.ref .fp16 [128, 64] "acc") MaskOpt.none) s6
      = some ((TileShape.allIndices [128, 64]).foldl
          (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 Out (oOffFn idx) (oValFn idx) else acc) s6) := by
    have hval : evalOp (Op.ref .fp16 [128, 64] "acc") s6 = some (⟨oValFn⟩ : Tile .fp16 [128, 64]) := by
      rw [evalOp_ref]; exact hacc6
    have hopval : evalOp (Op.ref .blockPtr [128, 64] "out_tile_ptr") s6
        = some (⟨fun _ : TileIndex [128, 64] =>
            { region := Out, baseOffset := 0, parentShape := [1024, 64], blockShape := [128, 64],
              strides := [64, 1], offsets := [s0.pids 1 * 128 + s0.pids 0 * 128, 0] }⟩
            : Tile .blockPtr [128, 64]) := by rw [evalOp_ref]; exact hout6
    unfold stepStmt
    simp only [hval, hopval, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s6 ?_
    intro acc idx _
    simp only [Bool.true_and, TileShape.blockPtr_inBounds_2d_offsets_index,
      TileShape.blockPtr_address_2d_row_offset_index]
    have hoff : (0 + (s0.pids 1 * 128 + s0.pids 0 * 128 + idx.1.val) * 64 + idx.2.1.val * 1)
        = oOffFn idx := by simp only [hoOffFn]; ring
    by_cases hmk : oMask idx
    · rw [decide_eq_true (by simpa only [hoMask] using hmk), hoff, if_pos hmk]; rfl
    · rw [decide_eq_false (by simpa only [hoMask] using hmk), if_neg hmk]; rfl
  erw [stepStmts.cons_some hstore7, stepStmts.nil]
  set s7 := (TileShape.allIndices [128, 64]).foldl
      (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 Out (oOffFn idx) (oValFn idx) else acc) s6
    with hs7
  refine ⟨s7, rfl, ?_, ?_, ?_⟩
  · -- Out readback
    intro idx hActive
    have hinjO : Function.Injective oOffFn := by
      rw [hoOffFn]
      rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
      simp only at h
      have hm : ma = mb := by omega
      have hd : da = db := by omega
      subst mb; subst db; rfl
    have hmaskIdx : oMask idx := by
      rw [hoMask]; refine ⟨?_, by have := idx.2.1.isLt; omega⟩
      simp only [active, rowIndex] at hActive
      omega
    rw [hs7]
    rw [show outOffset s0 (s0.pids 1 * 128) 64 1 128 idx = oOffFn idx from by
      simp only [outOffset, rowIndex, dIndex, hoOffFn]; ring]
    rw [scatter_memcell_fp16_prop_masked_nd s6 oOffFn oValFn oMask hinjO idx]
    rw [if_pos hmaskIdx]
    simp only [hoValFn, haccTile, FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
      FloatDType.ofWithBot, FloatDType.toWithBot]
    rfl
  · -- L readback
    intro i
    have hrawInj : Function.Injective (fun idx : TileIndex [128] => lOffFn idx) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hh
      simp only [hlOffFn] at hh
      obtain rfl : a = b := Fin.ext (by omega)
      rfl
    -- the Out store (region Out ≠ L) and M store (region M ≠ L) preserve L's mem
    rw [show s7.readMem L (lRowOffset s0 (s0.pids 1) 128 128 i)
          = s4.readMem L (lRowOffset s0 (s0.pids 1) 128 128 i) from by
      unfold BlockState.readMem
      rw [hs7, foldl_writeMemTyped_fp16_mask_other_region oMask oOffFn oValFn _ hLOut]
      rw [hs6, BlockState.setReg_mem, hs5,
        foldl_writeMemTyped_real_other_region (W := Region.cast M) (R := L) lOffFn
          (fun r => mTile.data r) _ (by intro hc; exact hLM (by simpa using hc.symm))]]
    rw [hs4]
    rw [show lRowOffset s0 (s0.pids 1) 128 128 i = lOffFn (i, PUnit.unit) from by
      simp only [lRowOffset, hlOffFn, hpids]]
    simp only [BlockState.writeMemTyped_real]
    erw [BlockState.scatter_readback_nd (region := Region.cast L) s3 lOffFn
      (fun idx : TileIndex [128] => FloatDType.real.storeValue (lTile.data idx)) hrawInj (i, PUnit.unit)]
    simp only [hlTile, FloatDType.real_storeValue, WithBot.unbotD_coe]
  · -- M readback
    intro i
    have hrawInj : Function.Injective (fun idx : TileIndex [128] => lOffFn idx) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hh
      simp only [hlOffFn] at hh
      obtain rfl : a = b := Fin.ext (by omega)
      rfl
    rw [show s7.readMem M (lRowOffset s0 (s0.pids 1) 128 128 i)
          = s5.readMem M (lRowOffset s0 (s0.pids 1) 128 128 i) from by
      unfold BlockState.readMem
      rw [hs7, foldl_writeMemTyped_fp16_mask_other_region oMask oOffFn oValFn _ hMOut]
      rw [hs6, BlockState.setReg_mem]]
    rw [hs5]
    rw [show lRowOffset s0 (s0.pids 1) 128 128 i = lOffFn (i, PUnit.unit) from by
      simp only [lRowOffset, hlOffFn, hpids]]
    simp only [BlockState.writeMemTyped_real]
    erw [BlockState.scatter_readback_nd (region := Region.cast M) s4 lOffFn
      (fun idx : TileIndex [128] => FloatDType.real.storeValue (mTile.data idx)) hrawInj (i, PUnit.unit)]
    simp only [hmTile, FloatDType.real_storeValue]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **Full-kernel forward execution chain.** Running the lowered `_fwd_kernel`
body (`taPreLoop ++ forRangeDyn(0,128,128) :: taPostLoop`) from a clean state `s`
with the checked honest grid (`pids 0 = 0`, single KV block) reaches a final state
`sF` whose `Out`/`L`/`M` stores hold the genuine closed-form values
`fwdOutSpec`/`fwdLSpec`/`fwdMSpec`. -/
theorem ta_exec (Q K V L M Out : RegionName) (s : BlockState)
    (hpid0 : s.pids 0 = 0) (hpid1 : s.pids 1 < 8)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 128 64 128) s = some sF
      ∧ (∀ idx : TileIndex [128, 64],
          active s (s.pids 1 * 128) 1024 128 idx →
          sF.mem Out (outOffset s (s.pids 1 * 128) 64 1 128 idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (fwdOutSpec s Q K V idx))))
      ∧ (∀ i : Fin 128, sF.readMem L (lRowOffset s (s.pids 1) 128 128 i) = fwdLSpec s Q K i)
      ∧ (∀ i : Fin 128, sF.readMem M (lRowOffset s (s.pids 1) 128 128 i) = fwdMSpec s Q K i) := by
  set sc := ((Real.sqrt (64 : ℝ))⁻¹) with hsc
  -- decompose the body
  have hbody : exec (triton_attention_fwd_kernel Q K V L M Out sc
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 128 64 128) s
        = stepStmts (taPreLoop Q K V Out sc
            ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
                  (Op.mul .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
                  (Op.constNat 128) (taLoopBody sc)
                :: taPostLoop L M Out)) s := by
    show stepStmts (triton_attention_fwd_kernel Q K V L M Out sc
          32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
          2 4 128 1024 128 64 128).toAlgKernel.body s = _
    rw [ta_body_split]
  rw [hbody]
  -- preLoop
  obtain ⟨sp, hpre, hsppids, hspmem, hspundef, hsmStart, hohp, homp, honp, hmpp, hlpp, haccp,
      hqp, hkpp, hvpp, hopp⟩ := taPreLoop_eval s Q K V L M Out sc hundef
  rw [stepStmts.append_some hpre]
  -- invariant base case at counter 0
  have hpid1bound : ∀ j : Fin 128, sp.pids 1 * 128 + j.val < 1024 := by
    intro j; rw [hsppids]; have := j.isLt; have := hpid1; omega
  have hinv0 : taInvariant Q K V Out sp sc 0 sp := by
    refine ta_invariant_zero Q K V Out sp sc ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ hspundef
    · exact hmpp
    · exact hlpp
    · exact haccp
    · rw [hqp]; refine congrArg some ?_; ext idx
      simp only [fwdQTile]
      unfold BlockState.readMem; rw [hspmem, hsppids]
    · rw [homp, hsppids]
    · exact honp
    · rw [hkpp, hsppids]
    · rw [hvpp, hsppids]
    · rw [hopp, hsppids]
    · rw [hohp, hsppids]
  -- run the loop via forRangeDyn_inv (single block, honest grid pids0 = 0)
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
      (stepOp := Op.constNat 128)
      (P := fun i st => taInvariant Q K V Out sp sc i st)
      (s_init := sp)
      (by rw [evalOp_constNat])
      (by rw [evalOp_mul, evalOp_add, evalOp_ref, hsmStart]
          simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
          refine congrArg some ?_
          ext u
          simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
            NumericDType.add, NumericDType.mul]
          rw [hpid0]; norm_num)
      (by rw [evalOp_constNat])
      (by norm_num)
      hinv0
      (fun i st hi hP => ta_attn_step Q K V Out sp sc i st hi hpid1bound hP)
  rw [stepStmts.cons_some hloop]
  -- final counter = 128
  have hfinal : final = 128 := by
    simp only [taInvariant] at hinvL
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    omega
  subst hfinal
  -- postLoop
  obtain ⟨sF, hpostStep, hO, hLrb, hMrb⟩ :=
    ta_postLoop Q K V L M Out sp sc sL hLOut hMOut hLM hinvL
  refine ⟨sF, hpostStep, ?_, ?_, ?_⟩
  · -- Out readback: bridge sp → s and oPartial/lPartial → fwdOutSpec
    intro idx hActive
    -- fwd tiles agree between s and sp (equal mem/pids)
    have htileQ : fwdQTile sp Q = fwdQTile s Q := by
      funext i; simp only [fwdQTile]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileK : fwdKTile sp K = fwdKTile s K := by
      funext i; simp only [fwdKTile]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileV : fwdVTile sp V = fwdVTile s V := by
      funext i; simp only [fwdVTile]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have hooff : outOffset sp (sp.pids 1 * 128) 64 1 128 idx = outOffset s (s.pids 1 * 128) 64 1 128 idx := by
      simp only [outOffset, rowIndex, hsppids]
    have hOidx := hO idx (by simp only [active, rowIndex, hsppids]; simpa only [active, rowIndex] using hActive)
    rw [hooff] at hOidx
    rw [hOidx]
    refine congrArg (MemCell.of .fp16) (congrArg (FloatDType.real.cast FloatDType.fp16) (congrArg some ?_))
    rw [fwdOutSpec_eq_streaming, htileQ, htileK, htileV, hsppids]
  · -- L readback
    intro i
    have htileQ : fwdQTile sp Q = fwdQTile s Q := by
      funext i; simp only [fwdQTile]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileK : fwdKTile sp K = fwdKTile s K := by
      funext i; simp only [fwdKTile]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have hloff : lRowOffset sp (sp.pids 1) 128 128 i = lRowOffset s (s.pids 1) 128 128 i := by
      simp only [lRowOffset, rowIndex, hsppids]
    have hLi := hLrb i
    rw [hloff] at hLi
    rw [hLi, lPartial_eq_fwdLSpec]
    simp only [fwdLSpec, fwdMSpec, fwdCausalSet, htileQ, htileK, hsppids]
  · -- M readback
    intro i
    have htileQ : fwdQTile sp Q = fwdQTile s Q := by
      funext i; simp only [fwdQTile]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileK : fwdKTile sp K = fwdKTile s K := by
      funext i; simp only [fwdKTile]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have hloff : lRowOffset sp (sp.pids 1) 128 128 i = lRowOffset s (s.pids 1) 128 128 i := by
      simp only [lRowOffset, rowIndex, hsppids]
    have hMi := hMrb i
    rw [hloff] at hMi
    rw [hMi, ← fwdMSpec_eq_mPartial]
    simp only [fwdMSpec, fwdCausalSet, htileQ, htileK, hsppids]

set_option maxRecDepth 8000 in
/-- **Boundary-checked block-ptr load** for `_bwd_kernel`'s `bwdPtrTile`. With
`pids 0 < 8` every lane is in bounds (`off_z·512 + off_h·128 + i < 1024`), so the
load reads `R[off_z·32768 + off_h·8192 + i·64 + e]` at each `(i,e)`. -/
theorem bwd_load_bc_eval
    (R : RegionName) (ptrOp : Op .blockPtr [128, 64]) (s t : BlockState)
    (hpids0lt : s.pids 0 < 8) (hmem : t.mem = s.mem)
    (hp : evalOp ptrOp t = some (bwdPtrTile s R 0)) :
    evalOp (.load .real (.blockPtr ptrOp [0, 1]) .none) t
      = some ⟨fun idx : TileIndex [128, 64] =>
          some (s.readMem R (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩ := by
  have hp' : evalOp ptrOp t = some
      ⟨fun _ : TileIndex [128, 64] =>
        { region := R, baseOffset := 0, parentShape := [1024, 64],
          blockShape := [128, 64], strides := [64, 1],
          offsets := [s.pids 0 / 4 * 512 + s.pids 0 % 4 * 128, 0] }⟩ := by
    rw [hp]; simp only [bwdPtrTile, Nat.add_zero]
  rw [ta_load_blockPtr_bc_eval R 0 1024 64 128 64 64 1
    (s.pids 0 / 4 * 512 + s.pids 0 % 4 * 128) ptrOp t hp']
  refine congrArg some ?_
  ext idx
  have he : idx.2.1.val < 64 := idx.2.1.isLt
  have hrow : s.pids 0 / 4 * 512 + s.pids 0 % 4 * 128 + idx.1.val < 1024 := by
    have hi : idx.1.val < 128 := idx.1.isLt
    have hz : s.pids 0 / 4 ≤ 1 := by omega
    have hh : s.pids 0 % 4 ≤ 3 := by omega
    have : s.pids 0 / 4 * 512 ≤ 512 := by omega
    have : s.pids 0 % 4 * 128 ≤ 384 := by omega
    omega
  simp only [hrow, he, and_self, if_true]
  simp only [BlockState.readMem, hmem]
  congr 1
  simp only [bwdKBase]
  ring

/-! ### Genuine backward inner/outer body execution and top theorem -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk = tl.where(offs_m_curr ≥ offs_n, qk, -inf)` statement eval** for the
backward inner body. `offs_m_curr[i] = gm i`, `offs_n[j] = j`, so the kept lanes
are `j ≤ gm i`. -/
theorem bwd_where_eval (s : BlockState) (BM BN : Nat)
    (gm : Fin BM → Nat) (qktile : Tile .real [BM, BN])
    (hom : s.regs .nat [BM] "offs_m_curr" = some (Tile.vec gm))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m_curr"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) s
      = some ⟨fun idx : TileIndex [BM, BN] =>
          if idx.2.1.val ≤ gm idx.1 then qktile.data idx else (⊥ : WithBot ℝ)⟩ := by
  have hexpM : @evalOp TileDType.nat [BM, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m_curr")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec gm)) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hom
  have hexpN : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hon
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where, ta_evalOp_ge]
  simp only [evalOp_ref, hexpM, hexpN, hqk, hbcast,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.cop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.expandDim_data, Tile.vec,
    ComparableDType.nat]
  by_cases h : idx.2.1.val ≤ gm idx.1
  · rw [if_pos (by simpa using h)]; simp [h]
  · rw [if_neg (by simpa using h)]; simp [h]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`m = tl.load(m_ptrs + offs_m_curr)` statement eval** (and the analogous
`Di = tl.load(D_ptrs + offs_m_curr)`). `m_ptrs` holds the scalar pointer
`(region, regBase)`; with `offs_m_curr[i] = gm i` the load reads
`region[regBase + gm i]` at lane `i`. -/
theorem bwd_ptr_load_eval (s : BlockState) (BM : Nat) (name : RegName)
    (region : RegionName) (regBase : Nat) (gm : Fin BM → Nat)
    (hptr : s.regs .ptr [] name = some (Tile.scalar (region, regBase)))
    (hom : s.regs .nat [BM] "offs_m_curr" = some (Tile.vec gm)) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] name)
            (Op.ref .nat [BM] "offs_m_curr"))) MaskOpt.none) s
      = some ⟨fun i : TileIndex [BM] => some (s.readMem region (regBase + gm i.1))⟩ := by
  have hptrAdd : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] name)
      (Op.ref .nat [BM] "offs_m_curr")) s
      = some (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (region, regBase)) (Tile.vec gm)) := by
    simp only [evalOp, evalOp_ref, hptr, hom, Option.bind_eq_bind, Option.bind_some]
  simp only [evalOp, hptrAdd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.ptrAdd_data, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, BlockState.readMemValue_real, if_true]

/-- The raw `qk` dot tile (`Σ_e q·k`). -/
noncomputable def bwdInnerQk (s : BlockState) (Q K : RegionName) :
    Tile .real [128, 128] :=
  Tile.dot []
    (⟨fun ij : TileIndex [128, 64] => some (bwdKernelQ s Q ij.1 ⟨ij.2.1.val, by omega⟩)⟩ : Tile .real [128, 64])
    (Tile.transpose []
      (⟨fun ij : TileIndex [128, 64] => some (bwdKernelK s K ij.1 ⟨ij.2.1.val, by omega⟩)⟩ : Tile .real [128, 64]))

/-- The causal-masked `qk` tile (`-inf` ⇒ `⊥` where `j > i`). -/
noncomputable def bwdInnerQk1 (s : BlockState) (Q K : RegionName) :
    Tile .real [128, 128] :=
  ⟨fun idx : TileIndex [128, 128] =>
    if idx.2.1.val ≤ idx.1.val then (bwdInnerQk s Q K).data idx else (⊥ : WithBot ℝ)⟩

/-- The `p` tile `exp(qk1·sc − m[:,None])` in the inner body's `simp`-normal form. -/
noncomputable def bwdInnerP (s : BlockState) (Q K M : RegionName) :
    Tile .real [128, 128] :=
  Tile.uop WithBot.realExp
    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Tile.bop NumericDType.real.mul Broadcast.scalarR (bwdInnerQk1 s Q K)
        (Tile.scalar (some (Real.sqrt (64:ℝ))⁻¹ : WithBot ℝ)))
      (Tile.expandDim ⟨1, by simp⟩
        (⟨fun i : TileIndex [128] => some (bwdKernelM s M i.1)⟩ : Tile .real [128])))

/-- The `dp` tile `(0 − Di[:,None]) + dot(do, trans(v))`. -/
noncomputable def bwdInnerDp (s : BlockState) (V DO Delta : RegionName) :
    Tile .real [128, 128] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (⟨fun _ : TileIndex [128, 128] => some (0:ℝ)⟩ : Tile .real [128, 128])
      (Tile.expandDim ⟨1, by simp⟩
        (⟨fun i : TileIndex [128] => some (bwdKernelDi s Delta i.1)⟩ : Tile .real [128])))
    (Tile.dot []
      (⟨fun idx : TileIndex [128, 64] => some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩ : Tile .real [128, 64])
      (Tile.transpose []
        (⟨fun idx : TileIndex [128, 64] => some (bwdKernelV s V idx.1 ⟨idx.2.1.val, by omega⟩)⟩ : Tile .real [128, 64])))

/-- The `ds` tile `p · dp · sc`. -/
noncomputable def bwdInnerDs (s : BlockState) (Q K V DO M Delta : RegionName) :
    Tile .real [128, 128] :=
  Tile.bop NumericDType.real.mul Broadcast.scalarR
    (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (bwdInnerP s Q K M) (bwdInnerDp s V DO Delta))
    (Tile.scalar (some (Real.sqrt (64:ℝ))⁻¹ : WithBot ℝ))

/-- Cell value of the raw `qk` dot. -/
theorem bwdInnerQk_cell (s : BlockState) (Q K : RegionName) (i j : Fin 128) :
    (bwdInnerQk s Q K).data (i, j, PUnit.unit) = some (bwdKernelQK s Q K i j) := by
  simp only [bwdInnerQk, Tile.dot, Tile.transpose, bwdKernelQK,
    Option.map₂_some_some, WithBot.sum_someTerm_eq_some]

/-- Cell value of the masked `p` tile (matches `bwdKernelP`). -/
theorem bwdInnerP_cell (s : BlockState) (Q K M : RegionName) (i j : Fin 128) :
    (bwdInnerP s Q K M).data (i, j, PUnit.unit) = some (bwdKernelP s Q K M i j) := by
  have hqk1 : (bwdInnerQk1 s Q K).data (i, j, PUnit.unit)
      = if j.val ≤ i.val then some (bwdKernelQK s Q K i j) else (⊥ : WithBot ℝ) := by
    simp only [bwdInnerQk1]
    by_cases h : j.val ≤ i.val
    · rw [if_pos h, if_pos h, bwdInnerQk_cell]
    · rw [if_neg h, if_neg h]
  simp only [bwdInnerP, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar, bwdKernelP, hqk1, Tile.expandDim_data, TileShape.dropInsertedIndex_succ,
    TileShape.dropInsertedIndex_nil, WithBot.realMul, WithBot.realSub]
  by_cases h : j.val ≤ i.val
  · simp only [if_pos h, NumericDType.mul, NumericDType.sub, WithBot.realMul,
      WithBot.realSub, Option.map₂_some_some, WithBot.realExp_some]
  · simp only [if_neg h, NumericDType.mul, NumericDType.sub, WithBot.realMul,
      WithBot.realSub, Option.map₂_none_left, Option.map₂_none_right,
      WithBot.realExp_bot]
    rfl

/-- Cell value of the `dp` tile (matches `bwdKernelDP`). -/
theorem bwdInnerDp_cell (s : BlockState) (V DO Delta : RegionName) (i j : Fin 128) :
    (bwdInnerDp s V DO Delta).data (i, j, PUnit.unit) = some (bwdKernelDP s V DO Delta i j) := by
  simp only [bwdInnerDp, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.expandDim_data, TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    Tile.dot, Tile.transpose, bwdKernelDP, NumericDType.add, NumericDType.sub,
    WithBot.realAdd, WithBot.realSub, Option.map₂_some_some, WithBot.sum_someTerm_eq_some,
    Option.map₂_some_coe, Option.map₂_coe_some]
  congr 1
  ring

/-- Cell value of the `ds` tile (matches `bwdKernelDS`). -/
theorem bwdInnerDs_cell (s : BlockState) (Q K V DO M Delta : RegionName) (i j : Fin 128) :
    (bwdInnerDs s Q K V DO M Delta).data (i, j, PUnit.unit)
      = some (bwdKernelDS s Q K V DO M Delta i j) := by
  simp only [bwdInnerDs, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar,
    NumericDType.mul, WithBot.realMul]
  rw [bwdInnerP_cell, bwdInnerDp_cell]
  simp only [bwdKernelDS, Option.map₂_some_some, Option.map₂_some_coe]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- `qk = tl.dot(a, tl.trans(b))` raw-dot eval (no scale/zero-seed). -/
theorem bwd_dot_trans_eval (s : BlockState) (na nb : RegName)
    (atile btile : Tile .real [128, 64])
    (ha : s.regs .real [128, 64] na = some atile)
    (hb : s.regs .real [128, 64] nb = some btile) :
    @evalOp TileDType.real [128, 128]
        (Op.dot (batch := []) (Op.ref .real [128, 64] na)
          (Op.transpose (batch := []) (Op.ref .real [128, 64] nb))) s
      = some (Tile.dot [] atile (Tile.transpose [] btile)) := by
  have hbr : evalOp (Op.ref .real [128, 64] nb) s = some btile := by rw [evalOp_ref, hb]
  have har : evalOp (Op.ref .real [128, 64] na) s = some atile := by rw [evalOp_ref, ha]
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [128, 64] nb)) s
      = some (Tile.transpose [] btile) := by
    erw [evalOp_transpose [] (Op.ref .real [128, 64] nb), hbr]; rfl
  erw [evalOp_dot [] (Op.ref .real [128, 64] na)
    (Op.transpose (batch := []) (Op.ref .real [128, 64] nb)), har, htr]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- `acc += tl.dot((tl.trans(x.to(fp16))).to(real), y)` eval: the fp16 round-trip
inside the transposed-dot, matching the `dv`/`dk` accumulator updates. -/
theorem bwd_trans_fp16_dot_eval (s : BlockState) (nacc nx ny : RegName)
    (acctile : Tile .real [128, 64]) (xtile : Tile .real [128, 128]) (ytile : Tile .real [128, 64])
    (hacc : s.regs .real [128, 64] nacc = some acctile)
    (hx : s.regs .real [128, 128] nx = some xtile)
    (hy : s.regs .real [128, 64] ny = some ytile) :
    @evalOp TileDType.real [128, 64]
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [128, 64] nacc)
          (Op.dot (batch := [])
            (Op.castFloat .fp16 .real
              (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx))))
            (Op.ref .real [128, 64] ny))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot []
            (⟨fun idx : TileIndex [128, 128] =>
                FloatDType.fp16.cast FloatDType.real
                  ((Tile.transpose []
                    (⟨fun jdx : TileIndex [128, 128] =>
                        FloatDType.real.cast FloatDType.fp16 (xtile.data jdx)⟩ : Tile .fp16 [128, 128])).data idx)⟩
              : Tile .real [128, 128])
            ytile)) := by
  have haccr : evalOp (Op.ref .real [128, 64] nacc) s = some acctile := by rw [evalOp_ref, hacc]
  have hyr : evalOp (Op.ref .real [128, 64] ny) s = some ytile := by rw [evalOp_ref, hy]
  have hxr : evalOp (Op.ref .real [128, 128] nx) s = some xtile := by rw [evalOp_ref, hx]
  have hcast : @evalOp TileDType.fp16 [128, 128]
      (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx)) s
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (xtile.data idx)⟩ : Tile .fp16 [128, 128]) := by
    erw [evalOp_castFloat .real .fp16 (Op.ref .real [128, 128] nx), hxr]; rfl
  have htr : @evalOp TileDType.fp16 [128, 128]
      (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx))) s
      = some (Tile.transpose [] (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (xtile.data idx)⟩ : Tile .fp16 [128, 128])) := by
    erw [evalOp_transpose [] (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx)), hcast]; rfl
  have hcastback : @evalOp TileDType.real [128, 128]
      (Op.castFloat .fp16 .real
        (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx)))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          FloatDType.fp16.cast FloatDType.real
            ((Tile.transpose []
              (⟨fun jdx : TileIndex [128, 128] =>
                  FloatDType.real.cast FloatDType.fp16 (xtile.data jdx)⟩ : Tile .fp16 [128, 128])).data idx)⟩
        : Tile .real [128, 128]) := by
    erw [evalOp_castFloat .fp16 .real
      (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx))), htr]; rfl
  have hdot : @evalOp TileDType.real [128, 64]
      (Op.dot (batch := [])
        (Op.castFloat .fp16 .real
          (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx))))
        (Op.ref .real [128, 64] ny)) s
      = some (Tile.dot []
          (⟨fun idx : TileIndex [128, 128] =>
              FloatDType.fp16.cast FloatDType.real
                ((Tile.transpose []
                  (⟨fun jdx : TileIndex [128, 128] =>
                      FloatDType.real.cast FloatDType.fp16 (xtile.data jdx)⟩ : Tile .fp16 [128, 128])).data idx)⟩
            : Tile .real [128, 128]) ytile) := by
    erw [evalOp_dot []
      (Op.castFloat .fp16 .real
        (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nx))))
      (Op.ref .real [128, 64] ny), hcastback, hyr]; rfl
  rw [evalOp_add]
  simp only [haccr, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- `dq += tl.dot((ds.to(fp16)).to(real), k)` eval: fp16 round-trip on `ds` (no
transpose) inside the `dq` dot accumulator update. -/
theorem bwd_fp16_dot_eval (s : BlockState) (ndq nds nk : RegName)
    (dqtile : Tile .real [128, 64]) (dstile : Tile .real [128, 128]) (ktile : Tile .real [128, 64])
    (hdq : s.regs .real [128, 64] ndq = some dqtile)
    (hds : s.regs .real [128, 128] nds = some dstile)
    (hk : s.regs .real [128, 64] nk = some ktile) :
    @evalOp TileDType.real [128, 64]
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [128, 64] ndq)
          (Op.dot (batch := [])
            (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nds)))
            (Op.ref .real [128, 64] nk))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) dqtile
          (Tile.dot []
            (⟨fun idx : TileIndex [128, 128] =>
                FloatDType.fp16.cast FloatDType.real
                  (FloatDType.real.cast FloatDType.fp16 (dstile.data idx))⟩ : Tile .real [128, 128])
            ktile)) := by
  have hdqr : evalOp (Op.ref .real [128, 64] ndq) s = some dqtile := by rw [evalOp_ref, hdq]
  have hkr : evalOp (Op.ref .real [128, 64] nk) s = some ktile := by rw [evalOp_ref, hk]
  have hdsr : evalOp (Op.ref .real [128, 128] nds) s = some dstile := by rw [evalOp_ref, hds]
  have hcast : @evalOp TileDType.fp16 [128, 128]
      (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nds)) s
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (dstile.data idx)⟩ : Tile .fp16 [128, 128]) := by
    erw [evalOp_castFloat .real .fp16 (Op.ref .real [128, 128] nds), hdsr]; rfl
  have hcastback : @evalOp TileDType.real [128, 128]
      (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nds))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 (dstile.data idx))⟩ : Tile .real [128, 128]) := by
    erw [evalOp_castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nds)), hcast]
    refine congrArg some ?_; ext idx; rfl
  have hdot : @evalOp TileDType.real [128, 64]
      (Op.dot (batch := [])
        (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nds)))
        (Op.ref .real [128, 64] nk)) s
      = some (Tile.dot []
          (⟨fun idx : TileIndex [128, 128] =>
              FloatDType.fp16.cast FloatDType.real
                (FloatDType.real.cast FloatDType.fp16 (dstile.data idx))⟩ : Tile .real [128, 128]) ktile) := by
    erw [evalOp_dot []
      (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 128] nds)))
      (Op.ref .real [128, 64] nk), hcastback, hkr]; rfl
  rw [evalOp_add]
  simp only [hdqr, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **No-boundary-check block-ptr `.real` store** of `bwdPtrTile s R 0`: writes
each lane `(i,e)` to `R[bwdKBase s + i·64 + e]` with value `(vt.data idx).unbotD 0`. -/
theorem bwd_store_dq_eval (s sPre : BlockState) (R : RegionName)
    (vt : Tile .real [128, 64])
    (hptr : sPre.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s R 0))
    (hval : sPre.regs .real [128, 64] "dq" = some vt) :
    stepStmt (Stmt.store .real [128, 64]
        (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [])
        (Op.ref .real [128, 64] "dq") MaskOpt.none) sPre
      = some ((TileShape.allIndices [128, 64]).foldl
          (fun acc idx => acc.writeMem R (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            ((vt.data idx).unbotD 0)) sPre) := by
  unfold stepStmt
  simp only [evalOp_ref, hptr, hval, Option.bind_eq_bind, Option.bind_some, Option.map]
  refine congrArg some ?_
  refine List.foldl_ext _ _ _ (fun acc idx _ => ?_)
  rw [bwdPtrTile]
  simp only [BlockPtr.inBounds_nil_checkedAxes, Bool.and_true, Bool.true_and, if_true,
    TileShape.blockPtr_address_2d_row_offset_index, BlockState.writeMemTyped_real,
    FloatDType.real_storeValue]
  congr 1
  simp only [bwdKBase]; ring

theorem foldl_writeMem_pids_eq {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).pids)
      = s.pids := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih => rw [List.foldl_cons, ih]; rfl

/-! ### Genuine backward body assembly -/

/-- The fp16 round-trip realized by `(x.to(fp16)).to(real)` equals `bwdFp16 x`
(the spec's fp16 round-trip). In the placeholder fp16 semantics both are `x`. -/
theorem bwd_fp16_roundtrip (x : ℝ) :
    FloatDType.fp16.cast FloatDType.real
      (FloatDType.real.cast FloatDType.fp16 (some x)) = some (bwdFp16 x) := by
  simp only [bwdFp16, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
    FloatDType.storeValue, WithBot.unbotD_some]

/-- The `dv` accumulator register tile after the inner body
(`0 + dot((trans(p.to(fp16))).to(real), do)`), in the `bwd_trans_fp16_dot_eval`
normal form. -/
noncomputable def bwdInnerDv (s : BlockState) (Q K M DO : RegionName) :
    Tile .real [128, 64] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ : Tile .real [128, 64])
    (Tile.dot []
      (⟨fun idx : TileIndex [128, 128] =>
          FloatDType.fp16.cast FloatDType.real
            ((Tile.transpose []
              (⟨fun jdx : TileIndex [128, 128] =>
                  FloatDType.real.cast FloatDType.fp16 ((bwdInnerP s Q K M).data jdx)⟩
                : Tile .fp16 [128, 128])).data idx)⟩ : Tile .real [128, 128])
      (⟨fun idx : TileIndex [128, 64] => some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩
        : Tile .real [128, 64]))

/-- The `dk` accumulator register tile after the inner body
(`0 + dot((trans(ds.to(fp16))).to(real), q)`). -/
noncomputable def bwdInnerDk (s : BlockState) (Q K V DO M Delta : RegionName) :
    Tile .real [128, 64] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ : Tile .real [128, 64])
    (Tile.dot []
      (⟨fun idx : TileIndex [128, 128] =>
          FloatDType.fp16.cast FloatDType.real
            ((Tile.transpose []
              (⟨fun jdx : TileIndex [128, 128] =>
                  FloatDType.real.cast FloatDType.fp16 ((bwdInnerDs s Q K V DO M Delta).data jdx)⟩
                : Tile .fp16 [128, 128])).data idx)⟩ : Tile .real [128, 128])
      (⟨fun idx : TileIndex [128, 64] => some (bwdKernelQ s Q idx.1 ⟨idx.2.1.val, by omega⟩)⟩
        : Tile .real [128, 64]))

/-- Cell value of the `dv` accumulator tile:
`dv[j,e] = Σ_i fp16(p[i,j])·do[i,e]` (the inner sum of `bwdKernelDVSpec`). -/
theorem bwdInnerDv_cell (s : BlockState) (Q K M DO : RegionName) (j e : Fin 128) (he : e.val < 64) :
    (bwdInnerDv s Q K M DO).data (j, ⟨e.val, he⟩, PUnit.unit)
      = some (∑ i : Fin 128,
          bwdFp16 (bwdKernelP s Q K M i j) * bwdKernelDO s DO i ⟨e.val, by omega⟩) := by
  simp only [bwdInnerDv, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  have hsum : (∑ k : Fin 128, Option.map₂ (· * ·)
        ((⟨fun idx : TileIndex [128, 128] =>
            FloatDType.fp16.cast FloatDType.real
              ((Tile.transpose []
                (⟨fun jdx : TileIndex [128, 128] =>
                    FloatDType.real.cast FloatDType.fp16 ((bwdInnerP s Q K M).data jdx)⟩
                  : Tile .fp16 [128, 128])).data idx)⟩ : Tile .real [128, 128]).data (j, k, PUnit.unit))
        ((⟨fun idx : TileIndex [128, 64] => some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩
            : Tile .real [128, 64]).data (k, ⟨e.val, he⟩, PUnit.unit)) : WithBot ℝ)
      = some (∑ i : Fin 128, bwdFp16 (bwdKernelP s Q K M i j) * bwdKernelDO s DO i ⟨e.val, by omega⟩) := by
    rw [← WithBot.sum_someTerm_eq_some]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    simp only [Tile.transpose]
    rw [bwdInnerP_cell, bwd_fp16_roundtrip]
    simp only [Option.map₂_some_some]
  rw [hsum]
  simp only [Option.map₂_some_some, zero_add]

/-- Cell value of the `dk` accumulator tile:
`dk[j,e] = Σ_i fp16(ds[i,j])·q[i,e]` (the inner sum of `bwdKernelDKSpec`). -/
theorem bwdInnerDk_cell (s : BlockState) (Q K V DO M Delta : RegionName)
    (j e : Fin 128) (he : e.val < 64) :
    (bwdInnerDk s Q K V DO M Delta).data (j, ⟨e.val, he⟩, PUnit.unit)
      = some (∑ i : Fin 128,
          bwdFp16 (bwdKernelDS s Q K V DO M Delta i j) * bwdKernelQ s Q i ⟨e.val, by omega⟩) := by
  simp only [bwdInnerDk, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  have hsum : (∑ k : Fin 128, Option.map₂ (· * ·)
        ((⟨fun idx : TileIndex [128, 128] =>
            FloatDType.fp16.cast FloatDType.real
              ((Tile.transpose []
                (⟨fun jdx : TileIndex [128, 128] =>
                    FloatDType.real.cast FloatDType.fp16 ((bwdInnerDs s Q K V DO M Delta).data jdx)⟩
                  : Tile .fp16 [128, 128])).data idx)⟩ : Tile .real [128, 128]).data (j, k, PUnit.unit))
        ((⟨fun idx : TileIndex [128, 64] => some (bwdKernelQ s Q idx.1 ⟨idx.2.1.val, by omega⟩)⟩
            : Tile .real [128, 64]).data (k, ⟨e.val, he⟩, PUnit.unit)) : WithBot ℝ)
      = some (∑ i : Fin 128,
          bwdFp16 (bwdKernelDS s Q K V DO M Delta i j) * bwdKernelQ s Q i ⟨e.val, by omega⟩) := by
    rw [← WithBot.sum_someTerm_eq_some]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    simp only [Tile.transpose]
    rw [bwdInnerDs_cell, bwd_fp16_roundtrip]
    simp only [Option.map₂_some_some]
  rw [hsum]
  simp only [Option.map₂_some_some, zero_add]

/-- The `dq` register tile after the inner body
(`dq_loaded + dot((ds.to(fp16)).to(real), k)`), in the `bwd_fp16_dot_eval`
normal form. `dq_loaded[i,e] = DQ[bwdKBase + i·64 + e]`. -/
noncomputable def bwdInnerDq (s : BlockState) (Q K V DO M Delta DQ : RegionName) :
    Tile .real [128, 64] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (⟨fun idx : TileIndex [128, 64] =>
        some (s.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩ : Tile .real [128, 64])
    (Tile.dot []
      (⟨fun idx : TileIndex [128, 128] =>
          FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 ((bwdInnerDs s Q K V DO M Delta).data idx))⟩
        : Tile .real [128, 128])
      (⟨fun idx : TileIndex [128, 64] => some (bwdKernelK s K idx.1 ⟨idx.2.1.val, by omega⟩)⟩
        : Tile .real [128, 64]))

/-- Cell value of the `dq` tile (matches `bwdKernelDQSpec`). -/
theorem bwdInnerDq_cell (s : BlockState) (Q K V DO M Delta DQ : RegionName)
    (i e : Fin 128) (he : e.val < 64) :
    (bwdInnerDq s Q K V DO M Delta DQ).data (i, ⟨e.val, he⟩, PUnit.unit)
      = some (bwdKernelDQSpec s Q K V DO M Delta DQ (i, ⟨e.val, he⟩, PUnit.unit)) := by
  simp only [bwdInnerDq, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  have hsum : (∑ k : Fin 128, Option.map₂ (· * ·)
        ((⟨fun idx : TileIndex [128, 128] =>
            FloatDType.fp16.cast FloatDType.real
              (FloatDType.real.cast FloatDType.fp16 ((bwdInnerDs s Q K V DO M Delta).data idx))⟩
          : Tile .real [128, 128]).data (i, k, PUnit.unit))
        ((⟨fun idx : TileIndex [128, 64] => some (bwdKernelK s K idx.1 ⟨idx.2.1.val, by omega⟩)⟩
            : Tile .real [128, 64]).data (k, ⟨e.val, he⟩, PUnit.unit)) : WithBot ℝ)
      = some (∑ j : Fin 128,
          bwdFp16 (bwdKernelDS s Q K V DO M Delta i j) * bwdKernelK s K j ⟨e.val, by omega⟩) := by
    rw [← WithBot.sum_someTerm_eq_some]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    show Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 ((bwdInnerDs s Q K V DO M Delta).data (i, j, PUnit.unit))))
        (some (bwdKernelK s K j ⟨e.val, by omega⟩)) = _
    rw [bwdInnerDs_cell, bwd_fp16_roundtrip]
    simp only [Option.map₂_some_some]
  rw [hsum]
  simp only [Option.map₂_some_some, bwdKernelDQSpec]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`offs_m_curr = start_m + offs_m` eval** (inner body stmt 0). -/
theorem bwd_offsmcurr_eval (s : BlockState) (BM start_m : Nat) (gm : Fin BM → Nat)
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar start_m))
    (hom : s.regs .nat [BM] "offs_m" = some (Tile.vec gm)) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m")
        (Op.ref .nat [BM] "offs_m")) s
      = some (Tile.vec (fun i : Fin BM => start_m + gm i)) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hsm, hom, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext i
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p = exp(qk·sc − m[:,None])` eval** (inner body stmt 5). -/
theorem bwd_p_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ) (sc : ℝ)
    (qktile : Tile .real [BM, BN]) (mtile : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hm : s.regs .real [BM] "m" = some mtile) :
    evalOp (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sc))
          (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m")))) s
      = some (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile
              (Tile.scalar (some sc : WithBot ℝ)))
            (Tile.expandDim ⟨1, hax⟩ mtile))) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m")) s
      = some (Tile.expandDim ⟨1, hax⟩ mtile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hm
  rw [evalOp_exp, evalOp_sub, evalOp_mul]
  simp only [evalOp_ref, evalOp_const, hqk, hexp, Option.bind_eq_bind, Option.bind_some]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`dp = full(0) − Di[:,None]` eval** (inner body stmt 9). -/
theorem bwd_dpinit_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (Ditile : Tile .real [BM])
    (hDi : s.regs .real [BM] "Di" = some Ditile) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.full [BM, BN] (Op.const 0))
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "Di"))) s
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (⟨fun _ : TileIndex [BM, BN] => some (0:ℝ)⟩ : Tile .real [BM, BN])
          (Tile.expandDim ⟨1, hax⟩ Ditile)) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "Di")) s
      = some (Tile.expandDim ⟨1, hax⟩ Ditile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hDi
  rw [evalOp_sub, evalOp_full]
  simp only [evalOp_const, hexp, Option.bind_eq_bind, Option.bind_some]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`dp = dp + dot(do, trans(v))` eval** (inner body stmt 10). -/
theorem bwd_dpdot_eval (s : BlockState) (dptile : Tile .real [128, 128])
    (dotile vtile : Tile .real [128, 64])
    (hdp : s.regs .real [128, 128] "dp" = some dptile)
    (hdo : s.regs .real [128, 64] "do_val" = some dotile)
    (hv : s.regs .real [128, 64] "v" = some vtile) :
    @evalOp TileDType.real [128, 128]
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [128, 128] "dp")
          (Op.dot (batch := []) (Op.ref .real [128, 64] "do_val")
            (Op.transpose (batch := []) (Op.ref .real [128, 64] "v")))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) dptile
          (Tile.dot [] dotile (Tile.transpose [] vtile))) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hdp, Option.bind_eq_bind, Option.bind_some]
  rw [bwd_dot_trans_eval s "do_val" "v" dotile vtile hdo hv]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`ds = (p · dp) · sc` eval** (inner body stmt 11). -/
theorem bwd_ds_eval (s : BlockState) (sc : ℝ)
    (ptile dptile : Tile .real [128, 128])
    (hp : s.regs .real [128, 128] "p" = some ptile)
    (hdp : s.regs .real [128, 128] "dp" = some dptile) :
    @evalOp TileDType.real [128, 128]
        (Op.mul .real Broadcast.scalarR
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "dp"))
          (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            ptile dptile)
          (Tile.scalar (some sc : WithBot ℝ))) := by
  rw [evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, evalOp_const, hp, hdp, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **No-boundary-check `.real` block-ptr load** of `bwdPtrTile s R 0` (inner body
stmt 13, the `dq = tl.load(dq_tile_ptr)` load). Reads each lane `(i,e)` at
`R[bwdKBase s + i·64 + e]`. -/
theorem bwd_load_nobc_eval
    (R : RegionName) (ptrOp : Op .blockPtr [128, 64]) (s t : BlockState)
    (hmem : t.mem = s.mem)
    (hp : evalOp ptrOp t = some (bwdPtrTile s R 0)) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) t
      = some ⟨fun idx : TileIndex [128, 64] =>
          some (s.readMem R (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩ := by
  have hp' : evalOp ptrOp t = some
      ⟨fun _ : TileIndex [128, 64] =>
        { region := R, baseOffset := 0, parentShape := [1024, 64],
          blockShape := [128, 64], strides := [64, 1],
          offsets := [s.pids 0 / 4 * 512 + s.pids 0 % 4 * 128, 0] }⟩ := by
    rw [hp]; simp only [bwdPtrTile, Nat.add_zero]
  simp only [evalOp, hp', Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [BlockPtr.inBounds_nil_checkedAxes, Bool.and_true, Bool.true_and, if_true,
    TileShape.blockPtr_address_2d_row_offset_index, BlockState.readMemValue_real]
  simp only [BlockState.readMem, hmem]
  congr 1
  simp only [bwdKBase]; ring


/-! ### Genuine backward inner-body chunked assembly -/

/-- Chunk 1 of `bwdInnerBody` (stmts 0–4): `offs_m_curr`, `q` load, `qk` dot,
`qk` causal where-mask, `m` load. -/
private noncomputable def bwdInnerChunk1 : List Stmt :=
  [ Stmt.assign .nat [128] "offs_m_curr"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m") (Op.ref .nat [128] "offs_m")),
    Stmt.assign .real [128, 64] "q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [128, 128] "qk"
      (Op.dot (batch := []) (Op.ref .real [128, 64] "q") (Op.transpose (batch := []) (Op.ref .real [128, 64] "k"))),
    Stmt.assign .real [128, 128] "qk"
      (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m_curr"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
        (Op.ref .real [128, 128] "qk") (Op.broadcast Op.negInf [128, 128])),
    Stmt.assign .real [128] "m"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "m_ptrs") (Op.ref .nat [128] "offs_m_curr")))
        MaskOpt.none) ]

/-- Chunk 2 of `bwdInnerBody` (stmts 5–9): `p`, `do_val` load, `dv` accumulate,
`Di` load, `dp` init. -/
private noncomputable def bwdInnerChunk2 : List Stmt :=
  [ Stmt.assign .real [128, 128] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const (Real.sqrt (64 : ℝ))⁻¹))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m")))),
    Stmt.assign .real [128, 64] "do_val"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [128, 64] "dv"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "dv")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real
            (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "p"))))
          (Op.ref .real [128, 64] "do_val"))),
    Stmt.assign .real [128] "Di"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "D_ptrs") (Op.ref .nat [128] "offs_m_curr")))
        MaskOpt.none),
    Stmt.assign .real [128, 128] "dp"
      (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.full [128, 128] (Op.const 0))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "Di"))) ]

/-- Chunk 3 of `bwdInnerBody` (stmts 10–14): `dp` dot, `ds`, `dk` accumulate,
`dq` load, `dq` accumulate. -/
private noncomputable def bwdInnerChunk3 : List Stmt :=
  [ Stmt.assign .real [128, 128] "dp"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 128] "dp")
        (Op.dot (batch := []) (Op.ref .real [128, 64] "do_val")
          (Op.transpose (batch := []) (Op.ref .real [128, 64] "v")))),
    Stmt.assign .real [128, 128] "ds"
      (Op.mul .real Broadcast.scalarR
        (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "dp"))
        (Op.const (Real.sqrt (64 : ℝ))⁻¹)),
    Stmt.assign .real [128, 64] "dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "dk")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real
            (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "ds"))))
          (Op.ref .real [128, 64] "q"))),
    Stmt.assign .real [128, 64] "dq"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") []) MaskOpt.none),
    Stmt.assign .real [128, 64] "dq"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "dq")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 128] "ds")))
          (Op.ref .real [128, 64] "k"))) ]

/-- Chunk 4 of `bwdInnerBody` (stmts 15–19): `dq` store and the four pointer
advances. -/
private noncomputable def bwdInnerChunk4 : List Stmt :=
  [ Stmt.store .real [128, 64] (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [])
      (Op.ref .real [128, 64] "dq") MaskOpt.none,
    Stmt.assign .ptr [128, 64] "dq_ptrs"
      (Op.ptrAdd Broadcast.scalarR
        (Op.ref .ptr [128, 64] "dq_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64))),
    Stmt.assign .blockPtr [128, 64] "q_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "do_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "dq_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [128, 0]) ]

/-- The `dq` store statement of chunk 4 (stmt 15). -/
private def bwdInnerChunk4store : List Stmt :=
  [ Stmt.store .real [128, 64] (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [])
      (Op.ref .real [128, 64] "dq") MaskOpt.none ]

/-- The four pointer-advance statements of chunk 4 (stmts 16–19). -/
private def bwdInnerChunk4adv : List Stmt :=
  [ Stmt.assign .ptr [128, 64] "dq_ptrs"
      (Op.ptrAdd Broadcast.scalarR
        (Op.ref .ptr [128, 64] "dq_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64))),
    Stmt.assign .blockPtr [128, 64] "q_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "do_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [128, 0]),
    Stmt.assign .blockPtr [128, 64] "dq_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [128, 0]) ]

private theorem bwdInnerChunk4_split :
    bwdInnerChunk4 = bwdInnerChunk4store ++ bwdInnerChunk4adv := rfl

/-- `bwdInnerBody` splits into the four chunks. -/
private theorem bwdInnerBody_chunks :
    bwdInnerBody = bwdInnerChunk1 ++ bwdInnerChunk2 ++ bwdInnerChunk3 ++ bwdInnerChunk4 := by
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Chunk 1 step.** From a state exposing `start_m = 0`, `offs_m = arange`,
`offs_n = arange`, `q_tile_ptr = bwdPtrTile Q`, `k = K tile`, `m_ptrs = (M, pids0*128)`,
steps the five chunk-1 statements, exposing `offs_m_curr`, `q`, `qk` (masked),
and `m`. -/
private theorem bwdInnerChunk1_step (s sin : BlockState) (Q K M : RegionName)
    (hpids0lt : s.pids 0 < 8) (hmem : sin.mem = s.mem)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar 0))
    (hom : sin.regs .nat [128] "offs_m" = some (Tile.vec (fun i : Fin 128 => i.val)))
    (hon : sin.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hqptr : sin.regs .blockPtr [128, 64] "q_tile_ptr" = some (bwdPtrTile s Q 0))
    (hk : sin.regs .real [128, 64] "k"
      = some (⟨fun ij : TileIndex [128, 64] => some (bwdKernelK s K ij.1 ⟨ij.2.1.val, by omega⟩)⟩))
    (hmptr : sin.regs .ptr [] "m_ptrs" = some (Tile.scalar (M, s.pids 0 * 128))) :
    ∃ s1, stepStmts bwdInnerChunk1 sin = some s1
      ∧ s1.pids = sin.pids ∧ s1.mem = sin.mem ∧ s1.undef = sin.undef
      ∧ s1.regs .nat [128] "offs_m_curr" = some (Tile.vec (fun i : Fin 128 => i.val))
      ∧ s1.regs .real [128, 64] "q"
          = some (⟨fun idx : TileIndex [128, 64] => some (bwdKernelQ s Q idx.1 ⟨idx.2.1.val, by omega⟩)⟩)
      ∧ s1.regs .real [128, 128] "qk" = some (bwdInnerQk1 s Q K)
      ∧ s1.regs .real [128] "m"
          = some (⟨fun i : TileIndex [128] => some (bwdKernelM s M i.1)⟩)
      ∧ (∀ {dt sh nm}, nm ≠ "offs_m_curr" → nm ≠ "q" → nm ≠ "qk" → nm ≠ "m" →
          s1.regs dt sh nm = sin.regs dt sh nm) := by
  unfold bwdInnerChunk1
  have hvecmc : evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m")
      (Op.ref .nat [128] "offs_m")) sin = some (Tile.vec (fun i : Fin 128 => i.val)) := by
    rw [bwd_offsmcurr_eval sin 128 0 (fun i : Fin 128 => i.val) hsm hom]
    refine congrArg some ?_; ext i; simp [Tile.vec]
  -- offs_m_curr = start_m + offs_m  (= arange since start_m = 0)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hvecmc)]
  set s0 := sin.setReg "offs_m_curr" .nat [128] (Tile.vec (fun i : Fin 128 => i.val)) with hs0
  have homc0 : s0.regs .nat [128] "offs_m_curr" = some (Tile.vec (fun i : Fin 128 => i.val)) := by
    rw [hs0]; simp only [BlockState.setReg_same]
  -- q load
  rw [stepStmts.cons_some (show stepStmt _ s0
      = some (s0.setReg "q" .real [128, 64]
          (⟨fun idx : TileIndex [128, 64] =>
            some (s.readMem Q (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩)) from
    stepStmt_assign_eq_some
      (bwd_load_bc_eval Q (Op.ref .blockPtr [128, 64] "q_tile_ptr") s s0 hpids0lt
        (by rw [hs0]; exact hmem)
        (by rw [evalOp_ref, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hqptr)))]
  set s1 := s0.setReg "q" .real [128, 64]
    (⟨fun idx : TileIndex [128, 64] =>
      some (s.readMem Q (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩) with hs1
  -- qk dot
  rw [stepStmts.cons_some (show stepStmt _ s1
      = some (s1.setReg "qk" .real [128, 128] (bwdInnerQk s Q K)) from
    stepStmt_assign_eq_some (by
      rw [bwd_dot_trans_eval s1 "q" "k"
        (⟨fun idx : TileIndex [128, 64] => some (s.readMem Q (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩)
        (⟨fun ij : TileIndex [128, 64] => some (bwdKernelK s K ij.1 ⟨ij.2.1.val, by omega⟩)⟩)
        (by rw [hs1]; simp only [BlockState.setReg_same])
        (by rw [hs1, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hk)]
      rfl))]
  set s2 := s1.setReg "qk" .real [128, 128] (bwdInnerQk s Q K) with hs2
  -- qk where-mask
  rw [stepStmts.cons_some (show stepStmt _ s2
      = some (s2.setReg "qk" .real [128, 128] (bwdInnerQk1 s Q K)) from
    stepStmt_assign_eq_some (by
      have := bwd_where_eval s2 128 128 (fun i : Fin 128 => i.val) (bwdInnerQk s Q K)
        (by rw [hs2, hs1];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact homc0)
        (by rw [hs2, hs1, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hon)
        (by rw [hs2]; simp only [BlockState.setReg_same])
      rw [this]; refine congrArg some ?_; ext idx; simp only [bwdInnerQk1]))]
  set s3 := s2.setReg "qk" .real [128, 128] (bwdInnerQk1 s Q K) with hs3
  have hs3mem : ∀ R o, s3.readMem R o = s.readMem R o := by
    intro R o; rw [hs3, hs2, hs1, hs0]
    simp only [BlockState.readMem, BlockState.setReg_mem, hmem]
  -- m load
  rw [stepStmts.cons_some (show stepStmt _ s3
      = some (s3.setReg "m" .real [128]
          (⟨fun i : TileIndex [128] => some (bwdKernelM s M i.1)⟩)) from
    stepStmt_assign_eq_some (by
      have hml := bwd_ptr_load_eval s3 128 "m_ptrs" M (s.pids 0 * 128)
        (fun i : Fin 128 => i.val)
        (by rw [hs3, hs2, hs1, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hmptr)
        (by rw [hs3, hs2, hs1];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact homc0)
      rw [hml]; refine congrArg some ?_; ext i
      simp only [bwdKernelM, Nat.add_zero, hs3mem M (s.pids 0 * 128 + i.1.val)])),
    stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · -- offs_m_curr
    rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact homc0
  · -- q = Q tile
    rw [hs3, hs2]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [hs1]; simp only [BlockState.setReg_same]
    refine congrArg some ?_; ext idx; simp only [bwdKernelQ]
  · -- qk = bwdInnerQk1
    rw [hs3]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  · -- m
    simp only [BlockState.setReg_same]
  · intro dt sh nm h1 h2 h3 h4
    rw [hs3, hs2, hs1, hs0]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Chunk 2 step.** Steps `p`, `do_val`, `dv`-accumulate, `Di`, `dp`-init. -/
private theorem bwdInnerChunk2_step (s sin : BlockState) (Q K M DO Delta : RegionName)
    (hpids0lt : s.pids 0 < 8) (hmem : sin.mem = s.mem)
    (homc : sin.regs .nat [128] "offs_m_curr" = some (Tile.vec (fun i : Fin 128 => i.val)))
    (hqk : sin.regs .real [128, 128] "qk" = some (bwdInnerQk1 s Q K))
    (hm : sin.regs .real [128] "m"
      = some (⟨fun i : TileIndex [128] => some (bwdKernelM s M i.1)⟩))
    (hdoptr : sin.regs .blockPtr [128, 64] "do_tile_ptr" = some (bwdPtrTile s DO 0))
    (hdv : sin.regs .real [128, 64] "dv"
      = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩))
    (hDptr : sin.regs .ptr [] "D_ptrs" = some (Tile.scalar (Delta, s.pids 0 * 128))) :
    ∃ s2, stepStmts bwdInnerChunk2 sin = some s2
      ∧ s2.pids = sin.pids ∧ s2.mem = sin.mem ∧ s2.undef = sin.undef
      ∧ s2.regs .real [128, 128] "p" = some (bwdInnerP s Q K M)
      ∧ s2.regs .real [128, 64] "do_val"
          = some (⟨fun idx : TileIndex [128, 64] => some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩)
      ∧ s2.regs .real [128, 64] "dv" = some (bwdInnerDv s Q K M DO)
      ∧ s2.regs .real [128, 128] "dp"
          = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (⟨fun _ : TileIndex [128, 128] => some (0:ℝ)⟩ : Tile .real [128, 128])
              (Tile.expandDim ⟨1, by simp⟩
                (⟨fun i : TileIndex [128] => some (bwdKernelDi s Delta i.1)⟩ : Tile .real [128])))
      ∧ (∀ {dt sh nm}, nm ≠ "p" → nm ≠ "do_val" → nm ≠ "dv" → nm ≠ "Di" → nm ≠ "dp" →
          s2.regs dt sh nm = sin.regs dt sh nm) := by
  unfold bwdInnerChunk2
  -- p = exp(qk·sc − m[:,None])
  rw [stepStmts.cons_some (show stepStmt _ sin
      = some (sin.setReg "p" .real [128, 128] (bwdInnerP s Q K M)) from
    stepStmt_assign_eq_some (by
      rw [bwd_p_eval sin 128 128 (by simp) (Real.sqrt (64:ℝ))⁻¹ (bwdInnerQk1 s Q K)
        (⟨fun i : TileIndex [128] => some (bwdKernelM s M i.1)⟩) hqk hm]; rfl))]
  set s_p := sin.setReg "p" .real [128, 128] (bwdInnerP s Q K M) with hs_p
  -- do_val load
  rw [stepStmts.cons_some (show stepStmt _ s_p
      = some (s_p.setReg "do_val" .real [128, 64]
          (⟨fun idx : TileIndex [128, 64] =>
            some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩)) from
    stepStmt_assign_eq_some
      (bwd_load_bc_eval DO (Op.ref .blockPtr [128, 64] "do_tile_ptr") s s_p hpids0lt
        (by rw [hs_p]; exact hmem)
        (by rw [evalOp_ref, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hdoptr)))]
  set s_do := s_p.setReg "do_val" .real [128, 64]
    (⟨fun idx : TileIndex [128, 64] =>
      some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩) with hs_do
  -- dv accumulate
  rw [stepStmts.cons_some (show stepStmt _ s_do
      = some (s_do.setReg "dv" .real [128, 64] (bwdInnerDv s Q K M DO)) from
    stepStmt_assign_eq_some
      (bwd_trans_fp16_dot_eval s_do "dv" "p" "do_val"
        (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩)
        (bwdInnerP s Q K M)
        (⟨fun idx : TileIndex [128, 64] => some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩)
        (by rw [hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hdv)
        (by rw [hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
              not_false_eq_true])
        (by rw [hs_do]; simp only [BlockState.setReg_same])))]
  -- (note: do_val register carries the bwdKernelDO form throughout chunk 2)
  set s_dv := s_do.setReg "dv" .real [128, 64] (bwdInnerDv s Q K M DO) with hs_dv
  have hs_dvmem : ∀ R o, s_dv.readMem R o = s.readMem R o := by
    intro R o; rw [hs_dv, hs_do, hs_p]
    simp only [BlockState.readMem, BlockState.setReg_mem, hmem]
  -- Di load
  rw [stepStmts.cons_some (show stepStmt _ s_dv
      = some (s_dv.setReg "Di" .real [128]
          (⟨fun i : TileIndex [128] => some (s.readMem Delta (s.pids 0 * 128 + i.1.val))⟩)) from
    stepStmt_assign_eq_some (by
      have hdil := bwd_ptr_load_eval s_dv 128 "D_ptrs" Delta (s.pids 0 * 128)
        (fun i : Fin 128 => i.val)
        (by rw [hs_dv, hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hDptr)
        (by rw [hs_dv, hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact homc)
      rw [hdil]; refine congrArg some ?_; ext i
      simp only [hs_dvmem Delta (s.pids 0 * 128 + i.1.val)]))]
  set s_Di := s_dv.setReg "Di" .real [128]
    (⟨fun i : TileIndex [128] => some (s.readMem Delta (s.pids 0 * 128 + i.1.val))⟩) with hs_Di
  -- dp init
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bwd_dpinit_eval s_Di 128 128 (by simp)
      (⟨fun i : TileIndex [128] => some (s.readMem Delta (s.pids 0 * 128 + i.1.val))⟩)
      (by rw [hs_Di]; simp only [BlockState.setReg_same]))), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · -- p
    rw [hs_Di, hs_dv, hs_do]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hs_p]; simp only [BlockState.setReg_same]
  · -- do_val readback
    rw [hs_Di, hs_dv]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hs_do]; simp only [BlockState.setReg_same]
  · -- dv = bwdInnerDv
    rw [hs_Di]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hs_dv]; simp only [BlockState.setReg_same]
  · -- dp init form (Di lane reads Delta[pids0*128 + i])
    simp only [BlockState.setReg_same]
    refine congrArg some ?_
    refine congrArg _ ?_
    refine congrArg _ ?_
    ext i; simp only [bwdKernelDi]
  · intro dt sh nm h1 h2 h3 h4 h5
    rw [hs_Di, hs_dv, hs_do, hs_p]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h5,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Chunk 3 step.** Steps `dp`-dot, `ds`, `dk`-accumulate, `dq` load,
`dq`-accumulate. -/
private theorem bwdInnerChunk3_step (s sin : BlockState) (Q K V DO M Delta DQ : RegionName)
    (hmem : sin.mem = s.mem)
    (hp : sin.regs .real [128, 128] "p" = some (bwdInnerP s Q K M))
    (hdoval : sin.regs .real [128, 64] "do_val"
      = some (⟨fun idx : TileIndex [128, 64] => some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩))
    (hdp : sin.regs .real [128, 128] "dp"
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (⟨fun _ : TileIndex [128, 128] => some (0:ℝ)⟩ : Tile .real [128, 128])
          (Tile.expandDim ⟨1, by simp⟩
            (⟨fun i : TileIndex [128] => some (bwdKernelDi s Delta i.1)⟩ : Tile .real [128]))))
    (hdk : sin.regs .real [128, 64] "dk"
      = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩))
    (hq : sin.regs .real [128, 64] "q"
      = some (⟨fun idx : TileIndex [128, 64] => some (bwdKernelQ s Q idx.1 ⟨idx.2.1.val, by omega⟩)⟩))
    (hv : sin.regs .real [128, 64] "v"
      = some (⟨fun idx : TileIndex [128, 64] => some (bwdKernelV s V idx.1 ⟨idx.2.1.val, by omega⟩)⟩))
    (hk : sin.regs .real [128, 64] "k"
      = some (⟨fun idx : TileIndex [128, 64] => some (bwdKernelK s K idx.1 ⟨idx.2.1.val, by omega⟩)⟩))
    (hdqptr : sin.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s DQ 0)) :
    ∃ s3, stepStmts bwdInnerChunk3 sin = some s3
      ∧ s3.pids = sin.pids ∧ s3.mem = sin.mem ∧ s3.undef = sin.undef
      ∧ s3.regs .real [128, 64] "dk" = some (bwdInnerDk s Q K V DO M Delta)
      ∧ s3.regs .real [128, 64] "dq" = some (bwdInnerDq s Q K V DO M Delta DQ)
      ∧ s3.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s DQ 0)
      ∧ (∀ {dt sh nm}, nm ≠ "dp" → nm ≠ "ds" → nm ≠ "dk" → nm ≠ "dq" →
          s3.regs dt sh nm = sin.regs dt sh nm) := by
  unfold bwdInnerChunk3
  -- dp = dp + dot(do, trans(v))  (raw output is defeq bwdInnerDp)
  rw [stepStmts.cons_some (show stepStmt _ sin
      = some (sin.setReg "dp" .real [128, 128] (bwdInnerDp s V DO Delta)) from
    stepStmt_assign_eq_some
      (bwd_dpdot_eval sin
        (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (⟨fun _ : TileIndex [128, 128] => some (0:ℝ)⟩ : Tile .real [128, 128])
          (Tile.expandDim ⟨1, by simp⟩
            (⟨fun i : TileIndex [128] => some (bwdKernelDi s Delta i.1)⟩ : Tile .real [128])))
        (⟨fun idx : TileIndex [128, 64] => some (bwdKernelDO s DO idx.1 ⟨idx.2.1.val, by omega⟩)⟩)
        (⟨fun idx : TileIndex [128, 64] => some (bwdKernelV s V idx.1 ⟨idx.2.1.val, by omega⟩)⟩)
        hdp hdoval hv))]
  set s1 := sin.setReg "dp" .real [128, 128] (bwdInnerDp s V DO Delta) with hs1
  -- ds = (p · dp) · sc  (raw output is defeq bwdInnerDs)
  rw [stepStmts.cons_some (show stepStmt _ s1
      = some (s1.setReg "ds" .real [128, 128] (bwdInnerDs s Q K V DO M Delta)) from
    stepStmt_assign_eq_some
      (bwd_ds_eval s1 (Real.sqrt (64:ℝ))⁻¹ (bwdInnerP s Q K M) (bwdInnerDp s V DO Delta)
        (by rw [hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hp)
        (by rw [hs1]; simp only [BlockState.setReg_same])))]
  set s2 := s1.setReg "ds" .real [128, 128] (bwdInnerDs s Q K V DO M Delta) with hs2
  -- dk accumulate
  rw [stepStmts.cons_some (show stepStmt _ s2
      = some (s2.setReg "dk" .real [128, 64] (bwdInnerDk s Q K V DO M Delta)) from
    stepStmt_assign_eq_some
      (bwd_trans_fp16_dot_eval s2 "dk" "ds" "q"
        (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩)
        (bwdInnerDs s Q K V DO M Delta)
        (⟨fun idx : TileIndex [128, 64] => some (bwdKernelQ s Q idx.1 ⟨idx.2.1.val, by omega⟩)⟩)
        (by rw [hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hdk)
        (by rw [hs2]; simp only [BlockState.setReg_same])
        (by rw [hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hq)))]
  set s3 := s2.setReg "dk" .real [128, 64] (bwdInnerDk s Q K V DO M Delta) with hs3
  -- dq load
  rw [stepStmts.cons_some (show stepStmt _ s3
      = some (s3.setReg "dq" .real [128, 64]
          (⟨fun idx : TileIndex [128, 64] =>
            some (s.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩)) from
    stepStmt_assign_eq_some
      (bwd_load_nobc_eval DQ (Op.ref .blockPtr [128, 64] "dq_tile_ptr") s s3
        (by rw [hs3, hs2, hs1]; exact hmem)
        (by rw [evalOp_ref, hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hdqptr)))]
  set s4 := s3.setReg "dq" .real [128, 64]
    (⟨fun idx : TileIndex [128, 64] =>
      some (s.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩) with hs4
  -- dq accumulate
  rw [stepStmts.cons_some (show stepStmt _ s4
      = some (s4.setReg "dq" .real [128, 64] (bwdInnerDq s Q K V DO M Delta DQ)) from
    stepStmt_assign_eq_some
      (bwd_fp16_dot_eval s4 "dq" "ds" "k"
        (⟨fun idx : TileIndex [128, 64] =>
          some (s.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩)
        (bwdInnerDs s Q K V DO M Delta)
        (⟨fun idx : TileIndex [128, 64] => some (bwdKernelK s K idx.1 ⟨idx.2.1.val, by omega⟩)⟩)
        (by rw [hs4]; simp only [BlockState.setReg_same])
        (by rw [hs4, hs3]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            rw [hs2]; simp only [BlockState.setReg_same])
        (by rw [hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hk))),
    stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · -- dk = bwdInnerDk
    rw [hs4, hs3]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true]
  · -- dq = bwdInnerDq
    rw [hs4]; simp only [BlockState.setReg_same]
  · -- dq_tile_ptr preserved
    rw [hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hdqptr
  · intro dt sh nm h1 h2 h3 h4
    rw [hs4, hs3, hs2, hs1]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

/-- A foldl of `writeMem` into region `W` leaves `undef` unchanged. -/
private theorem foldl_writeMem_undef {α : Type} (W : RegionName)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (l : List α) :
    ∀ (s : BlockState) (R : RegionName) (o : Nat),
      ((l.foldl (fun acc k => acc.writeMem W (offsetFn k) (valueFn k)) s).undef R o)
        = s.undef R o := by
  induction l with
  | nil => intro s R o; rfl
  | cons hd tl ih => intro s R o; rw [List.foldl_cons, ih]; rfl

/-- A foldl of `writeMem` into region `W` leaves `mem` at any other region
`R ≠ W` unchanged. -/
private theorem foldl_writeMem_other_region {α : Type} (W : RegionName)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (l : List α) :
    ∀ (s : BlockState) (R : RegionName) (o : Nat), R ≠ W →
      ((l.foldl (fun acc k => acc.writeMem W (offsetFn k) (valueFn k)) s).mem R o)
        = s.mem R o := by
  induction l with
  | nil => intro s R o _; rfl
  | cons hd tl ih =>
      intro s R o hRW
      rw [List.foldl_cons, ih _ _ _ hRW, BlockState.writeMem]
      simp only [hRW, false_and, if_false]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Chunk-4 advances.** The four pointer-advance assigns preserve `mem`,
`pids`, `undef`, and every register except the four advanced pointers. Proved
generically over the starting state `t` so the foldl-store term never mixes with
the advance `setReg` chain (avoids the trusted-kernel timeout). -/
private theorem bwdInnerChunk4adv_step (t : BlockState)
    (hdqptrs : ∃ x, t.regs .ptr [128, 64] "dq_ptrs" = some x)
    (hqptr : ∃ x, t.regs .blockPtr [128, 64] "q_tile_ptr" = some x)
    (hdoptr : ∃ x, t.regs .blockPtr [128, 64] "do_tile_ptr" = some x)
    (hdqptr : ∃ x, t.regs .blockPtr [128, 64] "dq_tile_ptr" = some x) :
    ∃ t', stepStmts bwdInnerChunk4adv t = some t'
      ∧ t'.pids = t.pids ∧ t'.undef = t.undef ∧ (∀ R o, t'.mem R o = t.mem R o)
      ∧ (∀ {dt sh nm}, nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
          nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" → t'.regs dt sh nm = t.regs dt sh nm) := by
  unfold bwdInnerChunk4adv
  obtain ⟨tdqp, htdqp⟩ := hdqptrs
  obtain ⟨tqp, htqp⟩ := hqptr
  obtain ⟨tdop, htdop⟩ := hdoptr
  obtain ⟨tdqtp, htdqtp⟩ := hdqptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 64] "dq_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64))) t
      = some (Tile.ptrAdd Broadcast.scalarR tdqp (Tile.scalar (128 * 64))) from by
      rw [evalOp_ptrAdd]; simp only [evalOp_ref, htdqp, evalOp_mul, evalOp_constNat,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  set sA := t.setReg "dq_ptrs" .ptr [128, 64]
    (Tile.ptrAdd Broadcast.scalarR tdqp (Tile.scalar (128 * 64))) with hsA
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [128, 0]) sA
      = some ⟨fun i => (tqp.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htqp, Option.bind_eq_bind, Option.bind_some]))]
  set sB := sA.setReg "q_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tqp.data i).advance [128, 0]⟩ with hsB
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [128, 0]) sB
      = some ⟨fun i => (tdop.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsB, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdop, Option.bind_eq_bind, Option.bind_some]))]
  set sC := sB.setReg "do_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdop.data i).advance [128, 0]⟩ with hsC
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [128, 0]) sC
      = some ⟨fun i => (tdqtp.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsC, hsB, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdqtp, Option.bind_eq_bind, Option.bind_some])), stepStmts.nil]
  set sD := sC.setReg "dq_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdqtp.data i).advance [128, 0]⟩ with hsD
  refine ⟨sD, rfl, ?_, ?_, ?_, ?_⟩
  · rw [hsD, hsC, hsB, hsA]; simp only [BlockState.setReg_pids]
  · rw [hsD, hsC, hsB, hsA]; rfl
  · intro R o; rw [hsD, hsC, hsB, hsA]; simp only [BlockState.setReg_mem]
  · intro dt sh nm h1 h2 h3 h4
    rw [hsD, hsC, hsB, hsA]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Chunk 4 step.** Steps the `dq` store and the four pointer advances. The
final `DQ` memory holds `dqSpec` at each lane; memory in any other region is
unchanged. -/
private theorem bwdInnerChunk4_step (s sin : BlockState) (DQ : RegionName)
    (dqt : Tile .real [128, 64]) (dqSpec : TileIndex [128, 64] → ℝ)
    (hmem : sin.mem = s.mem)
    (hdq : sin.regs .real [128, 64] "dq" = some dqt)
    (hdqcell : ∀ idx : TileIndex [128, 64], (dqt.data idx).unbotD 0 = dqSpec idx)
    (hdqptr : sin.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s DQ 0))
    (hdqptrs : ∃ t, sin.regs .ptr [128, 64] "dq_ptrs" = some t)
    (hqptr : ∃ t, sin.regs .blockPtr [128, 64] "q_tile_ptr" = some t)
    (hdoptr : ∃ t, sin.regs .blockPtr [128, 64] "do_tile_ptr" = some t) :
    ∃ s4, stepStmts bwdInnerChunk4 sin = some s4
      ∧ s4.pids = sin.pids ∧ s4.undef = sin.undef
      ∧ (∀ idx : TileIndex [128, 64],
          s4.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            = dqSpec idx)
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → s4.mem R o = sin.mem R o)
      ∧ (∀ {dt sh nm}, nm ≠ "dq" → nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
          nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" → s4.regs dt sh nm = sin.regs dt sh nm) := by
  rw [bwdInnerChunk4_split]
  -- dq store (isolated foldl term)
  obtain ⟨sStore, hStore, hStpids, hStundef, hStDQ, hStother, hStregs⟩ :
      ∃ sStore, stepStmts bwdInnerChunk4store sin = some sStore
        ∧ sStore.pids = sin.pids ∧ sStore.undef = sin.undef
        ∧ (∀ idx : TileIndex [128, 64],
            sStore.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val) = dqSpec idx)
        ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → sStore.mem R o = sin.mem R o)
        ∧ (∀ {dt sh nm}, sStore.regs dt sh nm = sin.regs dt sh nm) := by
    unfold bwdInnerChunk4store
    rw [stepStmts.cons_some (bwd_store_dq_eval s sin DQ dqt hdqptr hdq), stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_⟩
    · exact foldl_writeMem_pids_eq _ _ _ _ sin
    · funext R o; exact foldl_writeMem_undef DQ _ _ _ sin R o
    · intro idx
      rw [BlockState.readMem]
      have hinj : Function.Injective (fun idx : TileIndex [128, 64] =>
          bwdKBase s + idx.1.val * 64 + idx.2.1.val) := by
        rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
        simp only at h
        obtain rfl : ma = mb := by omega
        obtain rfl : da = db := by omega
        rfl
      have hsc := BlockState.scatter_readback_nd (region := DQ) sin
        (fun idx : TileIndex [128, 64] => bwdKBase s + idx.1.val * 64 + idx.2.1.val)
        (fun idx : TileIndex [128, 64] => (dqt.data idx).unbotD 0) hinj idx
      simp only [BlockState.readMem] at hsc
      rw [hsc]; exact hdqcell idx
    · intro R o hRne
      exact foldl_writeMem_other_region DQ _ _ _ sin R o hRne
    · intro dt sh nm; exact BlockState.foldl_writeMem_regs _ _ _ _ sin dt sh nm
  rw [stepStmts.append_some hStore]
  -- four advances (isolated setReg chain)
  obtain ⟨s4, hAdv, hApids, hAundef, hAmem, hAregs⟩ :=
    bwdInnerChunk4adv_step sStore
      ⟨_, hStregs.trans hdqptrs.choose_spec⟩
      ⟨_, hStregs.trans hqptr.choose_spec⟩
      ⟨_, hStregs.trans hdoptr.choose_spec⟩
      ⟨_, hStregs.trans hdqptr⟩
  refine ⟨s4, hAdv, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hApids, hStpids]
  · rw [hAundef, hStundef]
  · intro idx
    rw [BlockState.readMem, hAmem, ← BlockState.readMem, hStDQ idx]
  · intro R o hRne
    rw [hAmem, hStother R o hRne]
  · intro dt sh nm h1 h2 h3 h4 h5
    rw [hAregs h2 h3 h4 h5, hStregs]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Inner-body step.** Combines the four chunks: from a state exposing all the
inner-body inputs, stepping `bwdInnerBody` reaches a final state whose `DQ`
memory holds `bwdKernelDQSpec`, whose `dv`/`dk` registers hold `bwdInnerDv`/
`bwdInnerDk`, and which preserves `pids`/`undef` and `mem` off `DQ`. -/
private theorem bwdInnerBody_step (s sin : BlockState) (Q K V DO M Delta DQ : RegionName)
    (hpids0lt : s.pids 0 < 8) (hmem : sin.mem = s.mem)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar 0))
    (hom : sin.regs .nat [128] "offs_m" = some (Tile.vec (fun i : Fin 128 => i.val)))
    (hon : sin.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hqptr : sin.regs .blockPtr [128, 64] "q_tile_ptr" = some (bwdPtrTile s Q 0))
    (hdoptr : sin.regs .blockPtr [128, 64] "do_tile_ptr" = some (bwdPtrTile s DO 0))
    (hdqptr : sin.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s DQ 0))
    (hk : sin.regs .real [128, 64] "k"
      = some (⟨fun ij : TileIndex [128, 64] => some (bwdKernelK s K ij.1 ⟨ij.2.1.val, by omega⟩)⟩))
    (hv : sin.regs .real [128, 64] "v"
      = some (⟨fun ij : TileIndex [128, 64] => some (bwdKernelV s V ij.1 ⟨ij.2.1.val, by omega⟩)⟩))
    (hdv : sin.regs .real [128, 64] "dv" = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩))
    (hdk : sin.regs .real [128, 64] "dk" = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩))
    (hmptr : sin.regs .ptr [] "m_ptrs" = some (Tile.scalar (M, s.pids 0 * 128)))
    (hDptr : sin.regs .ptr [] "D_ptrs" = some (Tile.scalar (Delta, s.pids 0 * 128)))
    (hdqptrs : ∃ t, sin.regs .ptr [128, 64] "dq_ptrs" = some t) :
    ∃ sF, stepStmts bwdInnerBody sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.undef = sin.undef
      ∧ sF.regs .real [128, 64] "dv" = some (bwdInnerDv s Q K M DO)
      ∧ sF.regs .real [128, 64] "dk" = some (bwdInnerDk s Q K V DO M Delta)
      ∧ (∀ idx : TileIndex [128, 64],
          sF.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            = bwdKernelDQSpec s Q K V DO M Delta DQ idx)
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → sF.mem R o = sin.mem R o) := by
  rw [bwdInnerBody_chunks, List.append_assoc, List.append_assoc]
  obtain ⟨tdqptrs, htdqptrs⟩ := hdqptrs
  -- chunk 1
  obtain ⟨s1, hc1, hp1pids, hp1mem, hp1undef, h1omc, h1q, h1qk, h1m, h1rest⟩ :=
    bwdInnerChunk1_step s sin Q K M hpids0lt hmem hsm hom hon hqptr hk hmptr
  rw [stepStmts.append_some hc1]
  have h1mem : s1.mem = s.mem := by rw [hp1mem, hmem]
  -- chunk 2
  obtain ⟨s2, hc2, hp2pids, hp2mem, hp2undef, h2p, h2do, h2dv, h2dp, h2rest⟩ :=
    bwdInnerChunk2_step s s1 Q K M DO Delta hpids0lt h1mem h1omc h1qk h1m
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdoptr)
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdv)
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hDptr)
  rw [stepStmts.append_some hc2]
  have h2mem : s2.mem = s.mem := by rw [hp2mem, h1mem]
  -- chunk 3
  obtain ⟨s3, hc3, hp3pids, hp3mem, hp3undef, h3dk, h3dq, h3dqptr, h3rest⟩ :=
    bwdInnerChunk3_step s s2 Q K V DO M Delta DQ h2mem h2p h2do h2dp
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdk)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide)]; exact h1q)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hv)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hk)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdqptr)
  rw [stepStmts.append_some hc3]
  have h3mem : s3.mem = s.mem := by rw [hp3mem, h2mem]
  -- chunk 4
  obtain ⟨s4, hc4, hp4pids, hp4undef, h4dq, h4mem, h4rest⟩ :=
    bwdInnerChunk4_step s s3 DQ (bwdInnerDq s Q K V DO M Delta DQ)
      (bwdKernelDQSpec s Q K V DO M Delta DQ) h3mem h3dq
      (by intro idx; obtain ⟨i, ⟨e, he⟩, u⟩ := idx
          rw [bwdInnerDq_cell s Q K V DO M Delta DQ i ⟨e, by omega⟩ he]
          simp only [WithBot.unbotD_some])
      h3dqptr
      ⟨tdqptrs, by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact htdqptrs⟩
      ⟨_, by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hqptr⟩
      ⟨_, by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdoptr⟩
  refine ⟨s4, hc4, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hp4pids, hp3pids, hp2pids, hp1pids]
  · rw [hp4undef, hp3undef, hp2undef, hp1undef]
  · -- dv = bwdInnerDv (set in chunk2, preserved by 3,4)
    rw [h4rest (by decide) (by decide) (by decide) (by decide) (by decide),
      h3rest (by decide) (by decide) (by decide) (by decide)]
    exact h2dv
  · -- dk = bwdInnerDk (set in chunk3, preserved by 4)
    rw [h4rest (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact h3dk
  · -- DQ readback
    exact h4dq
  · -- mem off DQ
    intro R o hRne
    rw [h4mem R o hRne, hp3mem, hp2mem, hp1mem]

/-! ### Genuine backward outer-body and exec assembly

The inner-body machinery above (`bwdInnerChunk{1,2,3,4}_step`, `bwdInnerBody_step`)
is fully verified: stepping `bwdInnerBody` writes the genuine real-valued
`bwdKernelDQSpec` into `DQ` memory and leaves `dv`/`dk` registers holding
`bwdInnerDv`/`bwdInnerDk`.

The remaining `DV`/`DK` outputs of `_bwd_kernel`, however, are stored as
`tl.float16` (`Stmt.store .fp16` in `bwdOuterBody`). Reading such cells back as
**reals** (which is what the python-shape summary's `DV`/`DK` write-maps do, via
`OutputReadable ℝ = BlockState.readMem`) yields `0`, not the genuine gradient.
The lemma below records this blocker: the genuine real-valued `DV`/`DK` summary
(`readMem = bwdKernelD{V,K}Spec`) is therefore unprovable for the checked launch,
since it would force `0 = bwdKernelD{V,K}Spec`. A faithful `DV`/`DK` summary would
have to read back at the `MemCell`/fp16 level (as the forward `Out` summary does),
not as a real. -/

/-- **Faithfulness blocker (DV/DK).** Reading a stored `tl.float16` cell back as a
real (`BlockState.readMem`, i.e. `OutputReadable ℝ`) yields `0`, because
`MemCell.readAs .real` requires the cell's dtype to be `.real`. Hence the genuine
real-valued `DV`/`DK` summary (`readMem = bwdKernelD{V,K}Spec`) is unprovable for
the fp16-storing `_bwd_kernel`: it would force `0 = bwdKernelD{V,K}Spec`. -/
theorem bwd_dv_fp16_real_readback_is_zero
    (s : BlockState) (R : RegionName) (o : Nat) (v : TileCarrier TileDType.fp16) :
    (s.writeMemTyped .fp16 R o v).readMem R o = 0 := by
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs, BlockState.readMem,
    and_self, if_true, MemCell.readAs, MemCell.of, FloatDType.toTileDType,
    reduceCtorEq, reduceDIte]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **fp16 boundary-checked block-ptr store** (mirrors the forward `Out` store).
Storing `castFloat .real .fp16 (ref nm)` through a `bwdPtrTile s R 0` block pointer
(boundary check `[0,1]`) writes, for every in-bounds lane, an fp16 `MemCell` at
offset `bwdKBase s + i·64 + e`. Under `pids0 < 8` every lane is in bounds, so the
readback at any lane is `MemCell.of .fp16 (real.cast fp16 (vt.data idx))`. -/
private theorem bwd_store_fp16_bc_eval (s sPre : BlockState) (R : RegionName)
    (ptrNm valNm : RegName) (vt : Tile .real [128, 64])
    (hpids0lt : s.pids 0 < 8)
    (hptr : sPre.regs .blockPtr [128, 64] ptrNm = some (bwdPtrTile s R 0))
    (hval : sPre.regs .real [128, 64] valNm = some vt)
    (hsome : ∀ idx : TileIndex [128, 64], ∃ r : ℝ, vt.data idx = some r) :
    ∃ sF, stepStmt (Stmt.store .fp16 [128, 64]
        (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] ptrNm) [0, 1])
        (Op.castFloat .real .fp16 (Op.ref .real [128, 64] valNm)) MaskOpt.none) sPre
      = some sF
      ∧ sF.pids = sPre.pids
      ∧ (∀ idx : TileIndex [128, 64],
          sF.mem R (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16 (vt.data idx)))
      ∧ (∀ (R' : RegionName) (o : Nat), R' ≠ R → sF.mem R' o = sPre.mem R' o) := by
  -- offset / value / mask functions
  set oOffFn : TileIndex [128, 64] → Nat :=
    fun idx => bwdKBase s + idx.1.val * 64 + idx.2.1.val with hoOffFn
  set oValFn : TileIndex [128, 64] → TileCarrier TileDType.fp16 :=
    fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx) with hoValFn
  set oMask : TileIndex [128, 64] → Prop :=
    fun idx => s.pids 0 / 4 * 512 + s.pids 0 % 4 * 128 + idx.1.val < 1024
      ∧ 0 + idx.2.1.val < 64 with hoMask
  have hstore : stepStmt (Stmt.store .fp16 [128, 64]
      (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] ptrNm) [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [128, 64] valNm)) MaskOpt.none) sPre
      = some ((TileShape.allIndices [128, 64]).foldl
          (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 R (oOffFn idx) (oValFn idx) else acc) sPre) := by
    have hvalop : evalOp (Op.castFloat .real .fp16 (Op.ref .real [128, 64] valNm)) sPre
        = some (⟨oValFn⟩ : Tile .fp16 [128, 64]) := by
      rw [evalOp_castFloat]; erw [evalOp_ref, hval]; rfl
    have hopval : evalOp (Op.ref .blockPtr [128, 64] ptrNm) sPre
        = some (bwdPtrTile s R 0) := by rw [evalOp_ref]; exact hptr
    unfold stepStmt
    simp only [hopval, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    erw [hvalop]
    rw [Option.bind_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ sPre ?_
    intro acc idx _
    simp only [bwdPtrTile, Nat.add_zero, Bool.true_and,
      TileShape.blockPtr_inBounds_2d_offsets_index,
      TileShape.blockPtr_address_2d_row_offset_index]
    have hoff : (0 + (s.pids 0 / 4 * 512 + s.pids 0 % 4 * 128 + idx.1.val) * 64 + idx.2.1.val * 1)
        = oOffFn idx := by simp only [hoOffFn, bwdKBase]; ring
    by_cases hmk : oMask idx
    · rw [decide_eq_true (by simpa only [hoMask] using hmk), hoff]
      rw [if_pos rfl, if_pos hmk]
    · rw [decide_eq_false (by simpa only [hoMask] using hmk)]
      rw [if_neg (by decide), if_neg hmk]
  rw [hstore]
  set sF := (TileShape.allIndices [128, 64]).foldl
      (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 R (oOffFn idx) (oValFn idx) else acc) sPre
    with hsF
  have hinj : Function.Injective oOffFn := by
    rw [hoOffFn]
    rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
    simp only at h
    have hd : da = db := by omega
    have hm : ma = mb := by omega
    subst mb; subst db; rfl
  have hPids : sF.pids = sPre.pids := by
    rw [hsF]
    suffices h : ∀ (l : List (TileIndex [128, 64])) (t : BlockState),
        (l.foldl (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 R (oOffFn idx) (oValFn idx) else acc) t).pids
          = t.pids by exact h _ sPre
    intro l
    induction l with
    | nil => intro t; rfl
    | cons hd tl ih =>
        intro t
        rw [List.foldl_cons, ih]
        by_cases hmk : oMask hd
        · simp only [hmk, if_true, BlockState.writeMemTyped_pids]
        · simp only [hmk, if_false]
  have hMem : ∀ idx : TileIndex [128, 64],
      sF.mem R (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
        = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16 (vt.data idx)) := by
    intro idx
    have hmaskIdx : oMask idx := by
      rw [hoMask]; refine ⟨?_, by have := idx.2.1.isLt; omega⟩
      have hi : idx.1.val < 128 := idx.1.isLt
      have hz : s.pids 0 / 4 ≤ 1 := by omega
      have hh : s.pids 0 % 4 ≤ 3 := by omega
      omega
    rw [hsF]
    rw [show bwdKBase s + idx.1.val * 64 + idx.2.1.val = oOffFn idx from rfl]
    rw [scatter_memcell_fp16_prop_masked_nd sPre oOffFn oValFn oMask hinj idx]
    rw [if_pos hmaskIdx]
    obtain ⟨r, hr⟩ := hsome idx
    simp only [hoValFn, hr, FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
      FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.unbotD_some]
  have hOther : ∀ (R' : RegionName) (o : Nat), R' ≠ R → sF.mem R' o = sPre.mem R' o := by
    intro R' o hne
    rw [hsF, foldl_writeMemTyped_fp16_mask_other_region oMask oOffFn oValFn _ hne]
  exact ⟨sF, rfl, hPids, hMem, hOther⟩

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 40000 in
/-- **Register preservation for the fp16 block-ptr store.** A `tl.store` only
writes memory, so any register is unchanged. -/
private theorem bwd_store_fp16_regs_eq (sPre : BlockState) (ptrTile : Tile .blockPtr [128, 64])
    (ptrNm valNm : RegName) (vt : Tile .real [128, 64])
    (hptr : sPre.regs .blockPtr [128, 64] ptrNm = some ptrTile)
    (hval : sPre.regs .real [128, 64] valNm = some vt)
    (sF : BlockState)
    (hstep : stepStmt (Stmt.store .fp16 [128, 64]
        (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] ptrNm) [0, 1])
        (Op.castFloat .real .fp16 (Op.ref .real [128, 64] valNm)) MaskOpt.none) sPre = some sF)
    {dt : TileDType} {sh : TileShape} {nm' : RegName} :
    sF.regs dt sh nm' = sPre.regs dt sh nm' := by
  have hvalop : evalOp (Op.castFloat .real .fp16 (Op.ref .real [128, 64] valNm)) sPre
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [128, 64]) := by
    rw [evalOp_castFloat]; erw [evalOp_ref, hval]; rfl
  have hopval : evalOp (Op.ref .blockPtr [128, 64] ptrNm) sPre
      = some ptrTile := by rw [evalOp_ref]; exact hptr
  rw [show stepStmt (Stmt.store .fp16 [128, 64]
      (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] ptrNm) [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [128, 64] valNm)) MaskOpt.none) sPre
      = some ((TileShape.allIndices [128, 64]).foldl
          (fun acc i =>
            let bp := ptrTile.data i
            let idx := TileShape.indexToList [128, 64] i
            if («true» && bp.inBounds idx [0, 1]) = «true» then
              acc.writeMemTyped .fp16 bp.region (bp.address idx)
                ((⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [128, 64]).data i)
            else acc) sPre) from by
        unfold stepStmt
        simp only [hopval, Option.bind_eq_bind, Option.bind_some, Option.map_some]
        erw [hvalop]
        rw [Option.bind_some]] at hstep
  obtain rfl : sF = _ := Option.some.inj hstep.symm
  suffices h : ∀ (l : List (TileIndex [128, 64])) (t : BlockState),
      (l.foldl (fun acc i =>
        let bp := ptrTile.data i
        let idx := TileShape.indexToList [128, 64] i
        if («true» && bp.inBounds idx [0, 1]) = «true» then
          acc.writeMemTyped .fp16 bp.region (bp.address idx)
            ((⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [128, 64]).data i)
        else acc) t).regs dt sh nm' = t.regs dt sh nm' by exact h _ sPre
  intro l
  induction l with
  | nil => intro t; rfl
  | cons hd tl ih =>
      intro t
      rw [List.foldl_cons, ih]
      by_cases hb : («true» && (ptrTile.data hd).inBounds (TileShape.indexToList [128, 64] hd) [0, 1]) = «true»
      · rw [if_pos hb, BlockState.writeMemTyped_regs]
      · rw [if_neg hb]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Chunk 4 advance step (with block-ptr existence).** Same as
`bwdInnerChunk4adv_step` but also returns that `q_tile_ptr`, `do_tile_ptr`,
`dq_tile_ptr` exist after the advances. -/
private theorem bwdInnerChunk4adv_step' (t : BlockState)
    (hdqptrs : ∃ x, t.regs .ptr [128, 64] "dq_ptrs" = some x)
    (hqptr : ∃ x, t.regs .blockPtr [128, 64] "q_tile_ptr" = some x)
    (hdoptr : ∃ x, t.regs .blockPtr [128, 64] "do_tile_ptr" = some x)
    (hdqptr : ∃ x, t.regs .blockPtr [128, 64] "dq_tile_ptr" = some x) :
    ∃ t', stepStmts bwdInnerChunk4adv t = some t'
      ∧ t'.pids = t.pids ∧ t'.undef = t.undef ∧ (∀ R o, t'.mem R o = t.mem R o)
      ∧ (∃ x, t'.regs .blockPtr [128, 64] "q_tile_ptr" = some x)
      ∧ (∃ x, t'.regs .blockPtr [128, 64] "do_tile_ptr" = some x)
      ∧ (∃ x, t'.regs .blockPtr [128, 64] "dq_tile_ptr" = some x)
      ∧ (∀ {dt sh nm}, nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
          nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" → t'.regs dt sh nm = t.regs dt sh nm) := by
  unfold bwdInnerChunk4adv
  obtain ⟨tdqp, htdqp⟩ := hdqptrs
  obtain ⟨tqp, htqp⟩ := hqptr
  obtain ⟨tdop, htdop⟩ := hdoptr
  obtain ⟨tdqtp, htdqtp⟩ := hdqptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 64] "dq_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64))) t
      = some (Tile.ptrAdd Broadcast.scalarR tdqp (Tile.scalar (128 * 64))) from by
      rw [evalOp_ptrAdd]; simp only [evalOp_ref, htdqp, evalOp_mul, evalOp_constNat,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  set sA := t.setReg "dq_ptrs" .ptr [128, 64]
    (Tile.ptrAdd Broadcast.scalarR tdqp (Tile.scalar (128 * 64))) with hsA
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [128, 0]) sA
      = some ⟨fun i => (tqp.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htqp, Option.bind_eq_bind, Option.bind_some]))]
  set sB := sA.setReg "q_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tqp.data i).advance [128, 0]⟩ with hsB
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [128, 0]) sB
      = some ⟨fun i => (tdop.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsB, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdop, Option.bind_eq_bind, Option.bind_some]))]
  set sC := sB.setReg "do_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdop.data i).advance [128, 0]⟩ with hsC
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [128, 0]) sC
      = some ⟨fun i => (tdqtp.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsC, hsB, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdqtp, Option.bind_eq_bind, Option.bind_some])), stepStmts.nil]
  set sD := sC.setReg "dq_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdqtp.data i).advance [128, 0]⟩ with hsD
  refine ⟨sD, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hsD, hsC, hsB, hsA]; simp only [BlockState.setReg_pids]
  · rw [hsD, hsC, hsB, hsA]; rfl
  · intro R o; rw [hsD, hsC, hsB, hsA]; simp only [BlockState.setReg_mem]
  · refine ⟨⟨fun i => (tqp.data i).advance [128, 0]⟩, ?_⟩
    simp only [hsD, hsC, hsB, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, BlockState.setReg_same]
  · refine ⟨⟨fun i => (tdop.data i).advance [128, 0]⟩, ?_⟩
    simp only [hsD, hsC, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, BlockState.setReg_same]
  · refine ⟨⟨fun i => (tdqtp.data i).advance [128, 0]⟩, ?_⟩
    rw [hsD, BlockState.setReg_same]
  · intro dt sh nm h1 h2 h3 h4
    rw [hsD, hsC, hsB, hsA]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Inner-body step (with register preservation).** Same as `bwdInnerBody_step`
but additionally exposes that every register whose name is *not* assigned by the
inner body (in particular `k`, `v`, `k_tile_ptr`, `v_tile_ptr`, `dv_tile_ptr`,
`dk_tile_ptr`) is preserved. Needed to step the outer body's post-inner stores
and advances. -/
private theorem bwdInnerBody_step' (s sin : BlockState) (Q K V DO M Delta DQ : RegionName)
    (hpids0lt : s.pids 0 < 8) (hmem : sin.mem = s.mem)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar 0))
    (hom : sin.regs .nat [128] "offs_m" = some (Tile.vec (fun i : Fin 128 => i.val)))
    (hon : sin.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hqptr : sin.regs .blockPtr [128, 64] "q_tile_ptr" = some (bwdPtrTile s Q 0))
    (hdoptr : sin.regs .blockPtr [128, 64] "do_tile_ptr" = some (bwdPtrTile s DO 0))
    (hdqptr : sin.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s DQ 0))
    (hk : sin.regs .real [128, 64] "k"
      = some (⟨fun ij : TileIndex [128, 64] => some (bwdKernelK s K ij.1 ⟨ij.2.1.val, by omega⟩)⟩))
    (hv : sin.regs .real [128, 64] "v"
      = some (⟨fun ij : TileIndex [128, 64] => some (bwdKernelV s V ij.1 ⟨ij.2.1.val, by omega⟩)⟩))
    (hdv : sin.regs .real [128, 64] "dv" = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩))
    (hdk : sin.regs .real [128, 64] "dk" = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩))
    (hmptr : sin.regs .ptr [] "m_ptrs" = some (Tile.scalar (M, s.pids 0 * 128)))
    (hDptr : sin.regs .ptr [] "D_ptrs" = some (Tile.scalar (Delta, s.pids 0 * 128)))
    (hdqptrs : ∃ t, sin.regs .ptr [128, 64] "dq_ptrs" = some t) :
    ∃ sF, stepStmts bwdInnerBody sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.undef = sin.undef
      ∧ sF.regs .real [128, 64] "dv" = some (bwdInnerDv s Q K M DO)
      ∧ sF.regs .real [128, 64] "dk" = some (bwdInnerDk s Q K V DO M Delta)
      ∧ (∀ idx : TileIndex [128, 64],
          sF.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            = bwdKernelDQSpec s Q K V DO M Delta DQ idx)
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → sF.mem R o = sin.mem R o)
      ∧ (∃ x, sF.regs .blockPtr [128, 64] "q_tile_ptr" = some x)
      ∧ (∃ x, sF.regs .blockPtr [128, 64] "do_tile_ptr" = some x)
      ∧ (∃ x, sF.regs .blockPtr [128, 64] "dq_tile_ptr" = some x)
      ∧ (∀ {dt sh nm}, nm ≠ "offs_m_curr" → nm ≠ "q" → nm ≠ "qk" → nm ≠ "m" →
          nm ≠ "p" → nm ≠ "do_val" → nm ≠ "Di" → nm ≠ "dp" → nm ≠ "ds" →
          nm ≠ "dk" → nm ≠ "dv" → nm ≠ "dq" → nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
          nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" →
          sF.regs dt sh nm = sin.regs dt sh nm) := by
  rw [bwdInnerBody_chunks, List.append_assoc, List.append_assoc]
  obtain ⟨tdqptrs, htdqptrs⟩ := hdqptrs
  obtain ⟨s1, hc1, hp1pids, hp1mem, hp1undef, h1omc, h1q, h1qk, h1m, h1rest⟩ :=
    bwdInnerChunk1_step s sin Q K M hpids0lt hmem hsm hom hon hqptr hk hmptr
  rw [stepStmts.append_some hc1]
  have h1mem : s1.mem = s.mem := by rw [hp1mem, hmem]
  obtain ⟨s2, hc2, hp2pids, hp2mem, hp2undef, h2p, h2do, h2dv, h2dp, h2rest⟩ :=
    bwdInnerChunk2_step s s1 Q K M DO Delta hpids0lt h1mem h1omc h1qk h1m
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdoptr)
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdv)
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hDptr)
  rw [stepStmts.append_some hc2]
  have h2mem : s2.mem = s.mem := by rw [hp2mem, h1mem]
  obtain ⟨s3, hc3, hp3pids, hp3mem, hp3undef, h3dk, h3dq, h3dqptr, h3rest⟩ :=
    bwdInnerChunk3_step s s2 Q K V DO M Delta DQ h2mem h2p h2do h2dp
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdk)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide)]; exact h1q)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hv)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hk)
      (by rw [h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdqptr)
  rw [stepStmts.append_some hc3]
  have h3mem : s3.mem = s.mem := by rw [hp3mem, h2mem]
  obtain ⟨s4, hc4, hp4pids, hp4undef, h4dq, h4mem, h4rest⟩ :=
    bwdInnerChunk4_step s s3 DQ (bwdInnerDq s Q K V DO M Delta DQ)
      (bwdKernelDQSpec s Q K V DO M Delta DQ) h3mem h3dq
      (by intro idx; obtain ⟨i, ⟨e, he⟩, u⟩ := idx
          rw [bwdInnerDq_cell s Q K V DO M Delta DQ i ⟨e, by omega⟩ he]
          simp only [WithBot.unbotD_some])
      h3dqptr
      ⟨tdqptrs, by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact htdqptrs⟩
      ⟨_, by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hqptr⟩
      ⟨_, by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdoptr⟩
  -- derive q/do/dq_tile_ptr existence in s4 by decomposing chunk4 = store ++ adv
  have hexist : (∃ x, s4.regs .blockPtr [128, 64] "q_tile_ptr" = some x)
      ∧ (∃ x, s4.regs .blockPtr [128, 64] "do_tile_ptr" = some x)
      ∧ (∃ x, s4.regs .blockPtr [128, 64] "dq_tile_ptr" = some x) := by
    have hc4 := hc4
    rw [bwdInnerChunk4_split] at hc4
    -- the store statement steps s3 to some sStore
    obtain ⟨sStore, hStore, hStregs⟩ :
        ∃ sStore, stepStmts bwdInnerChunk4store s3 = some sStore
          ∧ (∀ {dt sh nm}, sStore.regs dt sh nm = s3.regs dt sh nm) := by
      unfold bwdInnerChunk4store
      rw [stepStmts.cons_some (bwd_store_dq_eval s s3 DQ (bwdInnerDq s Q K V DO M Delta DQ)
        h3dqptr h3dq), stepStmts.nil]
      exact ⟨_, rfl, fun {dt sh nm} => BlockState.foldl_writeMem_regs _ _ _ _ s3 dt sh nm⟩
    rw [stepStmts.append_some hStore] at hc4
    obtain ⟨t', _hadv, _, _, _, hqe, hdoe, hdqe, _⟩ :=
      bwdInnerChunk4adv_step' sStore
        ⟨_, by rw [hStregs, h3rest (by decide) (by decide) (by decide) (by decide),
              h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
              h1rest (by decide) (by decide) (by decide) (by decide)]; exact htdqptrs⟩
        ⟨_, by rw [hStregs, h3rest (by decide) (by decide) (by decide) (by decide),
              h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
              h1rest (by decide) (by decide) (by decide) (by decide)]; exact hqptr⟩
        ⟨_, by rw [hStregs, h3rest (by decide) (by decide) (by decide) (by decide),
              h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
              h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdoptr⟩
        ⟨_, by rw [hStregs]; exact h3dqptr⟩
    obtain rfl : t' = s4 := Option.some.inj (_hadv.symm.trans hc4)
    exact ⟨hqe, hdoe, hdqe⟩
  refine ⟨s4, hc4, ?_, ?_, ?_, ?_, ?_, ?_, hexist.1, hexist.2.1, hexist.2.2, ?_⟩
  · rw [hp4pids, hp3pids, hp2pids, hp1pids]
  · rw [hp4undef, hp3undef, hp2undef, hp1undef]
  · rw [h4rest (by decide) (by decide) (by decide) (by decide) (by decide),
      h3rest (by decide) (by decide) (by decide) (by decide)]
    exact h2dv
  · rw [h4rest (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact h3dk
  · exact h4dq
  · intro R o hRne
    rw [h4mem R o hRne, hp3mem, hp2mem, hp1mem]
  · intro dt sh nm h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16
    rw [h4rest h12 h13 h14 h15 h16,
      h3rest h8 h9 h10 h12,
      h2rest h5 h6 h11 h7 h8,
      h1rest h1 h2 h3 h4]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Outer-body step (`start_n = 0`).** From a state `sOuter` exposing the
preLoop registers (`off_hz = pids 0`, the seven `bwdPtrTile` block pointers, the
rebased `DQ` ptr) with `start_n = 0`, `pids 1 = 0`, `pids 0 < 8`, `undef = 0`,
`mem = s.mem`, stepping `bwdOuterBody` reaches a final state `sF` whose `DQ`
memory holds the genuine real `bwdKernelDQSpec` and whose `DV`/`DK` memory holds
the fp16 `MemCell`s `MemCell.of .fp16 (real.cast fp16 (some <genuine sum>))`. -/
private theorem bwdOuterBody_step (s sOuter : BlockState) (Q K V DO DQ DK DV M Delta : RegionName)
    (hpids0lt : s.pids 0 < 8) (hpids1 : s.pids 1 = 0)
    (hpidsEq : sOuter.pids = s.pids) (hmem : sOuter.mem = s.mem)
    (hundef : ∀ rg o, sOuter.undef rg o = 0)
    (hDVDQ : DV ≠ DQ) (hDKDQ : DK ≠ DQ) (hDVDK : DV ≠ DK) (hDKDV : DK ≠ DV)
    (hsn : sOuter.regs .nat [] "start_n" = some (Tile.scalar 0))
    (hohz : sOuter.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)))
    (hqptr : sOuter.regs .blockPtr [128, 64] "q_tile_ptr" = some (bwdPtrTile s Q 0))
    (hkptr : sOuter.regs .blockPtr [128, 64] "k_tile_ptr" = some (bwdPtrTile s K 0))
    (hvptr : sOuter.regs .blockPtr [128, 64] "v_tile_ptr" = some (bwdPtrTile s V 0))
    (hdoptr : sOuter.regs .blockPtr [128, 64] "do_tile_ptr" = some (bwdPtrTile s DO 0))
    (hdqptr : sOuter.regs .blockPtr [128, 64] "dq_tile_ptr" = some (bwdPtrTile s DQ 0))
    (hdkptr : sOuter.regs .blockPtr [128, 64] "dk_tile_ptr" = some (bwdPtrTile s DK 0))
    (hdvptr : sOuter.regs .blockPtr [128, 64] "dv_tile_ptr" = some (bwdPtrTile s DV 0))
    (hDQreg : ∃ t, sOuter.regs .ptr [] "DQ" = some t) :
    ∃ sF, stepStmts (bwdOuterBody Delta M) sOuter = some sF
      ∧ (∀ idx : TileIndex [128, 64],
          sF.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            = bwdKernelDQSpec s Q K V DO M Delta DQ idx)
      ∧ (∀ idx : TileIndex [128, 64],
          sF.mem DV (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ i : Fin 128,
                  bwdFp16 (bwdKernelP s Q K M i idx.1) *
                    bwdKernelDO s DO i ⟨idx.2.1.val, by omega⟩))))
      ∧ (∀ idx : TileIndex [128, 64],
          sF.mem DK (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ i : Fin 128,
                  bwdFp16 (bwdKernelDS s Q K V DO M Delta i idx.1) *
                    bwdKernelQ s Q i ⟨idx.2.1.val, by omega⟩)))) := by
  obtain ⟨tDQ, htDQ⟩ := hDQreg
  unfold bwdOuterBody
  -- stmt 1: lo = start_n * 128 = 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 128)) sOuter
        = some (Tile.scalar 0) from by
      simp only [evalOp_mul, evalOp_ref, hsn, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.mul, Nat.mul_zero, Nat.zero_mul]))]
  set s1 := sOuter.setReg "lo" .nat [] (Tile.scalar 0) with hs1
  -- stmt 2: offs_qm = lo + arange = arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "lo") (Op.arange 128)) s1
        = some (Tile.vec (fun i : Fin 128 => i.val)) from by
      simp only [evalOp_add, evalOp_ref, hs1, BlockState.setReg_same, evalOp_arange,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, Nat.zero_add]))]
  set s2 := s1.setReg "offs_qm" .nat [128] (Tile.vec (fun i : Fin 128 => i.val)) with hs2
  -- stmt 3: offs_n = start_n*128 + arange = arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 128)) (Op.arange 128)) s2
        = some (Tile.vec (fun j : Fin 128 => j.val)) from by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hs2, hs1, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, hsn, evalOp_constNat, evalOp_arange,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, Nat.mul_zero, Nat.zero_mul, Nat.zero_add]))]
  set s3 := s2.setReg "offs_n" .nat [128] (Tile.vec (fun j : Fin 128 => j.val)) with hs3
  -- stmt 4: offs_m = arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) s3 = some (Tile.vec (fun i : Fin 128 => i.val)) from
      evalOp_arange 128 s3))]
  set s4 := s3.setReg "offs_m" .nat [128] (Tile.vec (fun i : Fin 128 => i.val)) with hs4
  -- stmt 5: offs_k = arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 64) s4 = some (Tile.vec (fun e : Fin 64 => e.val)) from
      evalOp_arange 64 s4))]
  set s5 := s4.setReg "offs_k" .nat [64] (Tile.vec (fun e : Fin 64 => e.val)) with hs5
  -- stmt 6: dq_ptrs (value irrelevant; just need existence)
  obtain ⟨tdqp2, htdqp2⟩ : ∃ v, evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "DQ")
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_qm")) (Op.constNat 64))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_k")) (Op.constNat 1)))) s5
      = some v := by
    have hDQ5 : s5.regs .ptr [] "DQ" = some tDQ := by
      rw [hs5, hs4, hs3, hs2, hs1]
      iterate 5 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact htDQ
    have hqm5 : s5.regs .nat [128] "offs_qm" = some (Tile.vec (fun i : Fin 128 => i.val)) := by
      rw [hs5, hs4, hs3]
      iterate 3 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs2, BlockState.setReg_same]
    have hk5 : s5.regs .nat [64] "offs_k" = some (Tile.vec (fun e : Fin 64 => e.val)) := by
      rw [hs5, BlockState.setReg_same]
    -- the offset operand evaluates to some
    obtain ⟨offv, hoffv⟩ : ∃ offv, evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_qm")) (Op.constNat 64))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_k")) (Op.constNat 1))) s5
        = some offv := by
      have hm1 : evalOp (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_qm")) (Op.constNat 64)) s5
          = some (Tile.bop NumericDType.nat.mul Broadcast.scalarR
              (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin 128 => i.val))) (Tile.scalar 64)) := by
        rw [evalOp_mul]
        rw [Option.bind_eq_bind, Option.bind_eq_some_iff]
        exact ⟨_, evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hqm5, by
          rw [evalOp_constNat]; simp only [Option.bind_eq_bind, Option.bind_some]⟩
      have hm0 : evalOp (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_k")) (Op.constNat 1)) s5
          = some (Tile.bop NumericDType.nat.mul Broadcast.scalarR
              (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun e : Fin 64 => e.val))) (Tile.scalar 1)) := by
        rw [evalOp_mul]
        rw [Option.bind_eq_bind, Option.bind_eq_some_iff]
        exact ⟨_, evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hk5, by
          rw [evalOp_constNat]; simp only [Option.bind_eq_bind, Option.bind_some]⟩
      refine ⟨Tile.bop NumericDType.nat.add (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Tile.bop NumericDType.nat.mul Broadcast.scalarR
          (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin 128 => i.val))) (Tile.scalar 64))
        (Tile.bop NumericDType.nat.mul Broadcast.scalarR
          (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun e : Fin 64 => e.val))) (Tile.scalar 1)), ?_⟩
      rw [evalOp_add, Option.bind_eq_bind]
      refine Option.bind_eq_some_iff.mpr ⟨_, hm1, ?_⟩
      rw [Option.bind_eq_bind]
      exact Option.bind_eq_some_iff.mpr ⟨_, hm0, rfl⟩
    rw [evalOp]
    rw [evalOp_ref, hDQ5, hoffv]
    simp only [Option.bind_eq_bind, Option.bind_some]
    exact ⟨_, rfl⟩
  rw [stepStmts.cons_some (stepStmt_assign_eq_some htdqp2)]
  set s6 := s5.setReg "dq_ptrs" .ptr [128, 64] tdqp2 with hs6
  -- stmt 7: D_ptrs = (Delta, off_hz*128) = (Delta, pids0*128)
  have hohz6 : s6.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)) := by
    rw [hs6, hs5, hs4, hs3, hs2, hs1]
    iterate 6 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hohz
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Delta)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))) s6
        = some (Tile.scalar (Delta, s.pids 0 * 128)) from by
      simp only [evalOp, evalOp_ref, hohz6, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, Region.cast_id,
        Nat.zero_add]))]
  set s7 := s6.setReg "D_ptrs" .ptr [] (Tile.scalar (Delta, s.pids 0 * 128)) with hs7
  -- stmt 8: m_ptrs = (M, off_hz*128)
  have hohz7 : s7.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)) := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hohz6
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase M)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))) s7
        = some (Tile.scalar (M, s.pids 0 * 128)) from by
      simp only [evalOp, evalOp_ref, hohz7, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, Region.cast_id,
        Nat.zero_add]))]
  set s8 := s7.setReg "m_ptrs" .ptr [] (Tile.scalar (M, s.pids 0 * 128)) with hs8
  -- stmt 9: dv = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 64] (Op.const 0)) s8
        = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ : Tile .real [128, 64]) from by
      rw [evalOp_full, evalOp_const]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]))]
  set s9 := s8.setReg "dv" .real [128, 64] ⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ with hs9
  -- stmt 10: dk = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 64] (Op.const 0)) s9
        = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ : Tile .real [128, 64]) from by
      rw [evalOp_full, evalOp_const]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]))]
  set s10 := s9.setReg "dk" .real [128, 64] ⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ with hs10
  have hmem10 : s10.mem = s.mem := by
    rw [hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1]; exact hmem
  -- stmt 11: k load
  have hkptr10 : s10.regs .blockPtr [128, 64] "k_tile_ptr" = some (bwdPtrTile s K 0) := by
    rw [hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1]
    iterate 10 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hkptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bwd_load_bc_eval K (Op.ref .blockPtr [128, 64] "k_tile_ptr") s s10 hpids0lt hmem10
      (by rw [evalOp_ref]; exact hkptr10)))]
  set s11 := s10.setReg "k" .real [128, 64]
    ⟨fun idx : TileIndex [128, 64] => some (s.readMem K (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩
    with hs11
  -- stmt 12: v load
  have hmem11 : s11.mem = s.mem := by rw [hs11]; exact hmem10
  have hvptr11 : s11.regs .blockPtr [128, 64] "v_tile_ptr" = some (bwdPtrTile s V 0) := by
    rw [hs11, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1]
    iterate 10 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hvptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bwd_load_bc_eval V (Op.ref .blockPtr [128, 64] "v_tile_ptr") s s11 hpids0lt hmem11
      (by rw [evalOp_ref]; exact hvptr11)))]
  set s12 := s11.setReg "v" .real [128, 64]
    ⟨fun idx : TileIndex [128, 64] => some (s.readMem V (bwdKBase s + idx.1.val * 64 + idx.2.1.val))⟩
    with hs12
  -- stmt 13: inner forRangeDyn (single iteration, start_m = 0)
  have hForDyn : stepStmt (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo")
      (Op.mul .nat Broadcast.nil (Op.constNat 1) (Op.constNat 128)) (Op.constNat 128) bwdInnerBody) s12
      = stepStmts bwdInnerBody (s12.setReg "start_m" .nat [] (Tile.scalar 0)) := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    have hlo12 : evalOp (Op.ref .nat [] "lo") s12 = some (Tile.scalar 0) := by
      rw [evalOp_ref]
      simp only [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    rw [hlo12]
    simp only [evalOp_mul, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rw [show (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 1) (Tile.scalar 128)).data PUnit.unit
        = 128 from by
          simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
            NumericDType.mul]]
    rw [show (Tile.scalar 0).data PUnit.unit = 0 from rfl]
    exact stepForRangeAux.step_one_iter (by decide) (by decide) (by decide)
  -- apply inner-body step
  set sin := s12.setReg "start_m" .nat [] (Tile.scalar 0) with hsin
  have hsinmem : sin.mem = s.mem := by
    rw [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1]; exact hmem
  obtain ⟨sInner, hInner, hInpids, hInundef, hIndv, hIndk, hInDQ, hInmem,
      hInq, hIndo, hIndq, hInrest⟩ :=
    bwdInnerBody_step' s sin Q K V DO M Delta DQ hpids0lt hsinmem
      (by simp only [hsin, BlockState.setReg_same])
      (by simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
            BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
            not_false_eq_true, String.reduceEq])
      (by simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
            BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
            not_false_eq_true, String.reduceEq])
      (by simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
            BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          exact hqptr)
      (by simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
            BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          exact hdoptr)
      (by simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
            BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          exact hdqptr)
      (by simp only [hsin, hs12, hs11, BlockState.setReg_same, BlockState.setReg_ne_name,
            ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          refine congrArg some ?_; ext ij
          simp only [bwdKernelK])
      (by simp only [hsin, hs12, BlockState.setReg_same, BlockState.setReg_ne_name,
            ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          refine congrArg some ?_; ext ij
          simp only [bwdKernelV])
      (by simp only [hsin, hs12, hs11, hs10, hs9, BlockState.setReg_same, BlockState.setReg_ne_name,
            ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq])
      (by simp only [hsin, hs12, hs11, hs10, BlockState.setReg_same, BlockState.setReg_ne_name,
            ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq])
      (by simp only [hsin, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_same,
            BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq])
      (by simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, BlockState.setReg_same,
            BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq])
      (by refine ⟨tdqp2, ?_⟩
          simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, BlockState.setReg_same,
            BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq])
  -- collapse the forRangeDyn statement to its single-iteration result (sInner)
  rw [stepStmts.cons_some (hForDyn.trans hInner)]
  -- now the post-inner statements: 5 advances, 2 fp16 stores, 2 advances.
  -- readbacks in sInner of the preserved tile ptrs (k/v/dv/dk_tile_ptr) and dv/dk regs
  have hk_tp : sInner.regs .blockPtr [128, 64] "k_tile_ptr" = some (bwdPtrTile s K 0) := by
    rw [hInrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
      BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
    exact hkptr
  have hv_tp : sInner.regs .blockPtr [128, 64] "v_tile_ptr" = some (bwdPtrTile s V 0) := by
    rw [hInrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
      BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
    exact hvptr
  have hdv_tp : sInner.regs .blockPtr [128, 64] "dv_tile_ptr" = some (bwdPtrTile s DV 0) := by
    rw [hInrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
      BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
    exact hdvptr
  have hdk_tp : sInner.regs .blockPtr [128, 64] "dk_tile_ptr" = some (bwdPtrTile s DK 0) := by
    rw [hInrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    simp only [hsin, hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1,
      BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
    exact hdkptr
  -- the 5 advances (q/do/dq [0,0]; k/v [128,0]) only assign block ptrs
  obtain ⟨tq, htq⟩ := hInq
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "q_tile_ptr") [0, 0]) sInner
      = some ⟨fun i => (tq.data i).advance [0, 0]⟩ from by
      rw [advanceBlockPtr_eval]; simp only [evalOp_ref, htq, Option.bind_eq_bind, Option.bind_some]))]
  set sa1 := sInner.setReg "q_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tq.data i).advance [0, 0]⟩ with hsa1
  obtain ⟨tdo, htdo⟩ := hIndo
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "do_tile_ptr") [0, 0]) sa1
      = some ⟨fun i => (tdo.data i).advance [0, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsa1]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdo, Option.bind_eq_bind, Option.bind_some]))]
  set sa2 := sa1.setReg "do_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdo.data i).advance [0, 0]⟩ with hsa2
  obtain ⟨tdq, htdq⟩ := hIndq
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dq_tile_ptr") [0, 0]) sa2
      = some ⟨fun i => (tdq.data i).advance [0, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsa2, hsa1]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdq, Option.bind_eq_bind, Option.bind_some]))]
  set sa3 := sa2.setReg "dq_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdq.data i).advance [0, 0]⟩ with hsa3
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "k_tile_ptr") [128, 0]) sa3
      = some ⟨fun i => ((bwdPtrTile s K 0).data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsa3, hsa2, hsa1]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        hk_tp, Option.bind_eq_bind, Option.bind_some]))]
  set sa4 := sa3.setReg "k_tile_ptr" .blockPtr [128, 64]
    ⟨fun i => ((bwdPtrTile s K 0).data i).advance [128, 0]⟩ with hsa4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "v_tile_ptr") [128, 0]) sa4
      = some ⟨fun i => ((bwdPtrTile s V 0).data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsa4, hsa3, hsa2, hsa1]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        hv_tp, Option.bind_eq_bind, Option.bind_some]))]
  set sa5 := sa4.setReg "v_tile_ptr" .blockPtr [128, 64]
    ⟨fun i => ((bwdPtrTile s V 0).data i).advance [128, 0]⟩ with hsa5
  -- readbacks in sa5: dv reg, dk reg, dv_tile_ptr, dk_tile_ptr (all preserved by the 5 advances)
  have hsa5_regs : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
      nm ≠ "q_tile_ptr" → nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" → nm ≠ "k_tile_ptr" →
      nm ≠ "v_tile_ptr" → sa5.regs dt sh nm = sInner.regs dt sh nm := by
    intro dt sh nm h1 h2 h3 h4 h5
    rw [hsa5, hsa4, hsa3, hsa2, hsa1]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h5,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]
  have hsa5_mem : ∀ R o, sa5.mem R o = sInner.mem R o := by
    intro R o; rw [hsa5, hsa4, hsa3, hsa2, hsa1]; simp only [BlockState.setReg_mem]
  have hdv5 : sa5.regs .real [128, 64] "dv" = some (bwdInnerDv s Q K M DO) := by
    rw [hsa5_regs (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hIndv
  have hdk5 : sa5.regs .real [128, 64] "dk" = some (bwdInnerDk s Q K V DO M Delta) := by
    rw [hsa5_regs (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hIndk
  have hdvtp5 : sa5.regs .blockPtr [128, 64] "dv_tile_ptr" = some (bwdPtrTile s DV 0) := by
    rw [hsa5_regs (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hdv_tp
  have hdktp5 : sa5.regs .blockPtr [128, 64] "dk_tile_ptr" = some (bwdPtrTile s DK 0) := by
    rw [hsa5_regs (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hdk_tp
  -- DV fp16 store
  obtain ⟨sDV, hDVstep, hDVpids, hDVmem, hDVother⟩ :=
    bwd_store_fp16_bc_eval s sa5 DV "dv_tile_ptr" "dv"
      (bwdInnerDv s Q K M DO) hpids0lt hdvtp5 hdv5
      (by intro idx; obtain ⟨i', ⟨e', he'⟩, u'⟩ := idx
          exact ⟨_, bwdInnerDv_cell s Q K M DO i' ⟨e', by omega⟩ he'⟩)
  rw [stepStmts.cons_some hDVstep]
  -- DK fp16 store
  have hdktpDV : sDV.regs .blockPtr [128, 64] "dk_tile_ptr" = some (bwdPtrTile s DK 0) := by
    have := bwd_store_fp16_regs_eq sa5 (bwdPtrTile s DV 0) "dv_tile_ptr" "dv"
      (bwdInnerDv s Q K M DO) hdvtp5 hdv5 sDV hDVstep
      (dt := .blockPtr) (sh := [128, 64]) (nm' := "dk_tile_ptr")
    rw [this]; exact hdktp5
  have hdk5DV : sDV.regs .real [128, 64] "dk" = some (bwdInnerDk s Q K V DO M Delta) := by
    have := bwd_store_fp16_regs_eq sa5 (bwdPtrTile s DV 0) "dv_tile_ptr" "dv"
      (bwdInnerDv s Q K M DO) hdvtp5 hdv5 sDV hDVstep
      (dt := .real) (sh := [128, 64]) (nm' := "dk")
    rw [this]; exact hdk5
  obtain ⟨sDK, hDKstep, hDKpids, hDKmem, hDKother⟩ :=
    bwd_store_fp16_bc_eval s sDV DK "dk_tile_ptr" "dk"
      (bwdInnerDk s Q K V DO M Delta) hpids0lt hdktpDV hdk5DV
      (by intro idx; obtain ⟨i', ⟨e', he'⟩, u'⟩ := idx
          exact ⟨_, bwdInnerDk_cell s Q K V DO M Delta i' ⟨e', by omega⟩ he'⟩)
  rw [stepStmts.cons_some hDKstep]
  -- final two advances (dv_tile_ptr, dk_tile_ptr) - only need to step them
  have hdvtpDK : ∃ x, sDK.regs .blockPtr [128, 64] "dv_tile_ptr" = some x := by
    refine ⟨bwdPtrTile s DV 0, ?_⟩
    rw [bwd_store_fp16_regs_eq sDV (bwdPtrTile s DK 0) "dk_tile_ptr" "dk"
      (bwdInnerDk s Q K V DO M Delta) hdktpDV hdk5DV sDK hDKstep
      (dt := .blockPtr) (sh := [128, 64]) (nm' := "dv_tile_ptr")]
    rw [bwd_store_fp16_regs_eq sa5 (bwdPtrTile s DV 0) "dv_tile_ptr" "dv"
      (bwdInnerDv s Q K M DO) hdvtp5 hdv5 sDV hDVstep
      (dt := .blockPtr) (sh := [128, 64]) (nm' := "dv_tile_ptr")]
    exact hdvtp5
  have hdktpDK : ∃ x, sDK.regs .blockPtr [128, 64] "dk_tile_ptr" = some x :=
    ⟨_, by rw [bwd_store_fp16_regs_eq sDV (bwdPtrTile s DK 0) "dk_tile_ptr" "dk"
        (bwdInnerDk s Q K V DO M Delta) hdktpDV hdk5DV sDK hDKstep
        (dt := .blockPtr) (sh := [128, 64]) (nm' := "dk_tile_ptr")]; exact hdktpDV⟩
  obtain ⟨tdvtp, htdvtp⟩ := hdvtpDK
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dv_tile_ptr") [128, 0]) sDK
      = some ⟨fun i => (tdvtp.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval]; simp only [evalOp_ref, htdvtp, Option.bind_eq_bind, Option.bind_some]))]
  set sb1 := sDK.setReg "dv_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdvtp.data i).advance [128, 0]⟩ with hsb1
  obtain ⟨tdktp, htdktp⟩ := hdktpDK
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "dk_tile_ptr") [128, 0]) sb1
      = some ⟨fun i => (tdktp.data i).advance [128, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsb1]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdktp, Option.bind_eq_bind, Option.bind_some])), stepStmts.nil]
  set sb2 := sb1.setReg "dk_tile_ptr" .blockPtr [128, 64] ⟨fun i => (tdktp.data i).advance [128, 0]⟩ with hsb2
  -- final state mem facts
  have hb2mem : ∀ R o, sb2.mem R o = sDK.mem R o := by
    intro R o; rw [hsb2, hsb1]; simp only [BlockState.setReg_mem]
  refine ⟨sb2, rfl, ?_, ?_, ?_⟩
  · -- DQ readback (real)
    intro idx
    have hDQmemEq : sb2.mem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
        = sInner.mem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val) := by
      rw [hb2mem, hDKother DQ _ (Ne.symm hDKDQ), hDVother DQ _ (Ne.symm hDVDQ), hsa5_mem]
    rw [show sb2.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val)
          = sInner.readMem DQ (bwdKBase s + idx.1.val * 64 + idx.2.1.val) from by
      unfold BlockState.readMem; rw [hDQmemEq]]
    exact hInDQ idx
  · -- DV readback (fp16 MemCell)
    intro idx
    rw [hb2mem, hDKother DV _ hDVDK]
    rw [hDVmem idx]
    refine congrArg (MemCell.of .fp16) (congrArg (FloatDType.real.cast FloatDType.fp16) ?_)
    obtain ⟨i', ⟨e', he'⟩, u'⟩ := idx
    rw [bwdInnerDv_cell s Q K M DO i' ⟨e', by omega⟩ he']
  · -- DK readback (fp16 MemCell)
    intro idx
    rw [hb2mem, hDKmem idx]
    refine congrArg (MemCell.of .fp16) (congrArg (FloatDType.real.cast FloatDType.fp16) ?_)
    obtain ⟨i', ⟨e', he'⟩, u'⟩ := idx
    rw [bwdInnerDk_cell s Q K V DO M Delta i' ⟨e', by omega⟩ he']

/-- If a statement list ending in an `assign dt sh nm op` steps successfully, the
final state has register `nm` set (to whatever the op evaluated to). -/
private theorem stepStmts_append_assign_reg_exists
    {l : List Stmt} {dt : TileDType} {sh : TileShape} {nm : RegName}
    {op : Op dt sh} {s s' : BlockState}
    (h : stepStmts (l ++ [Stmt.assign dt sh nm op]) s = some s') :
    ∃ v, s'.regs dt sh nm = some v := by
  rw [stepStmts.append_some_iff] at h
  obtain ⟨smid, _hl, ha⟩ := h
  cases hev : evalOp op smid with
  | none =>
      have hnone : stepStmt (Stmt.assign dt sh nm op) smid = none := by
        unfold stepStmt; rw [hev]; rfl
      unfold stepStmts at ha
      rw [hnone] at ha
      simp at ha
  | some v =>
      rw [stepStmts.cons_some (stepStmt_assign_eq_some hev), stepStmts.nil] at ha
      injection ha with ha'
      exact ⟨v, by rw [← ha']; exact BlockState.setReg_same smid nm dt sh v⟩

/-- The preLoop sets a `DQ` pointer register (its final statement). -/
private theorem bwdPreLoop_DQ_exists (s s0 : BlockState) (Q K V DO DQ DK DV : RegionName)
    (hpre : stepStmts (bwdPreLoop Q K V DO DQ DK DV) s = some s0) :
    ∃ t, s0.regs .ptr [] "DQ" = some t := by
  have hsplit : bwdPreLoop Q K V DO DQ DK DV
      = (bwdPreLoop Q K V DO DQ DK DV).dropLast ++ [Stmt.assign .ptr [] "DQ"
          (Op.ptrAdd Broadcast.nil (Op.ptrBase DQ)
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 32768))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 8192))))] := by
    rfl
  rw [hsplit] at hpre
  exact stepStmts_append_assign_reg_exists hpre

/-- Bridge: under `pids 1 = 0`, the python grad offset `bwdGradOffset` (with
`H = 4`, strides `(32768,8192,64,1)`, `BLOCK_M = 128`) equals the genuine block
base `bwdKBase s + i·64 + e`. -/
private theorem bwdGradOffset_eq_bwdKBase (s : BlockState) (hpids1 : s.pids 1 = 0)
    (idx : TileIndex [128, 64]) :
    bwdGradOffset s 4 32768 8192 64 1 128 idx
      = bwdKBase s + idx.1.val * 64 + idx.2.1.val := by
  simp only [bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex, bwdKBase, hpids1]
  ring

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Genuine backward exec chain.** Running the full `_bwd_kernel` at the checked
Python launch from a clean state `s` with the honest grid (`pids 0 < 8`,
`pids 1 = 0`) reaches a final state `sF` whose `DQ` memory (read as a **real**)
holds the genuine `bwdKernelDQSpec`, and whose `DV`/`DK` memory (read at the
**fp16 `MemCell`** level) holds `MemCell.of .fp16 (real.cast fp16 (some <sum>))`. -/
theorem bwd_grads_exec (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (hpids0lt : s.pids 0 < 8) (hpids1 : s.pids 1 = 0)
    (hDVDQ : DV ≠ DQ) (hDKDQ : DK ≠ DQ) (hDVDK : DV ≠ DK) (hDKDV : DK ≠ DV)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 1 128 64 128) s = some sF
      ∧ (∀ idx : TileIndex [128, 64],
          sF.readMem DQ (bwdGradOffset s 4 32768 8192 64 1 128 idx)
            = bwdKernelDQSpec s Q K V DO M Delta DQ idx)
      ∧ (∀ idx : TileIndex [128, 64],
          sF.mem DV (bwdGradOffset s 4 32768 8192 64 1 128 idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ i : Fin 128,
                  bwdFp16 (bwdKernelP s Q K M i idx.1) *
                    bwdKernelDO s DO i ⟨idx.2.1.val, by omega⟩))))
      ∧ (∀ idx : TileIndex [128, 64],
          sF.mem DK (bwdGradOffset s 4 32768 8192 64 1 128 idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ i : Fin 128,
                  bwdFp16 (bwdKernelDS s Q K V DO M Delta i idx.1) *
                    bwdKernelQ s Q i ⟨idx.2.1.val, by omega⟩)))) := by
  have hbody : exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 1 128 64 128) s
      = stepStmts (bwdPreLoop Q K V DO DQ DK DV
          ++ [Stmt.forRange "start_n" 0 1 1 (bwdOuterBody Delta M)]) s := by
    show stepStmts (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 1 128 64 128).toAlgKernel.body s = _
    rw [bwd_body_split]
  rw [hbody]
  obtain ⟨s0, hpre, hs0pids, hs0mem, hs0undef, hohz, hqp, hkp, hvp, hdop, hdqp, hdkp, hdvp⟩ :=
    bwdPreLoop_eval s Q K V DO DQ DK DV hundef
  have hDQexist := bwdPreLoop_DQ_exists s s0 Q K V DO DQ DK DV hpre
  rw [stepStmts.append_some hpre]
  have hforStep : stepStmts [Stmt.forRange "start_n" 0 1 1 (bwdOuterBody Delta M)] s0
      = stepStmts (bwdOuterBody Delta M) (s0.setReg "start_n" .nat [] (Tile.scalar 0)) := by
    have h1 : stepStmt (Stmt.forRange "start_n" 0 1 1 (bwdOuterBody Delta M)) s0
        = stepStmts (bwdOuterBody Delta M) (s0.setReg "start_n" .nat [] (Tile.scalar 0)) := by
      rw [stepForRangeAux.forRange_unfold,
        stepForRangeAux.step_one_iter (by decide) (by decide) (by decide)]
    conv_lhs => unfold stepStmts
    rw [h1]
    cases h : stepStmts (bwdOuterBody Delta M) (s0.setReg "start_n" .nat [] (Tile.scalar 0)) with
    | none => rfl
    | some v => simp only [stepStmts]
  rw [hforStep]
  set sb := s0.setReg "start_n" .nat [] (Tile.scalar 0) with hsb
  have hsbpids : sb.pids = s.pids := by rw [hsb]; exact hs0pids
  have hsbmem : sb.mem = s.mem := by rw [hsb]; exact hs0mem
  have hsbundef : ∀ rg o, sb.undef rg o = 0 := by
    intro rg o; rw [hsb]; exact hs0undef rg o
  obtain ⟨sF, hstep, hDQ, hDV, hDK⟩ :=
    bwdOuterBody_step s sb Q K V DO DQ DK DV M Delta hpids0lt hpids1 hsbpids hsbmem hsbundef
      hDVDQ hDKDQ hDVDK hDKDV
      (by rw [hsb, BlockState.setReg_same])
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hohz)
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqp)
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hkp)
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hvp)
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hdop)
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hdqp)
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hdkp)
      (by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hdvp)
      (by obtain ⟨t, ht⟩ := hDQexist
          exact ⟨t, by rw [hsb, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact ht⟩)
  refine ⟨sF, hstep, ?_, ?_, ?_⟩
  · intro idx; rw [bwdGradOffset_eq_bwdKBase s hpids1 idx]; exact hDQ idx
  · intro idx; rw [bwdGradOffset_eq_bwdKBase s hpids1 idx]; exact hDV idx
  · intro idx; rw [bwdGradOffset_eq_bwdKBase s hpids1 idx]; exact hDK idx


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
set_option maxHeartbeats 1600000 in
/-- **Genuine faithful backward gradient summary.** At the checked Python launch
(`B·H = 8` program blocks, single KV/M block, `pids 1 = 0`), every output lane of
`_bwd_kernel` holds its genuine FlashAttention-1 backward gradient:

* `DQ` (stored `.real`, no fp16 cast) reads back as a **real** equal to the genuine
  `bwdKernelDQSpec` (`priorDQ + Σ_j fp16(ds)·k`).
* `DV` and `DK` are stored with `tl.float16` (`Stmt.store .fp16`); reading such a
  cell back as a real yields `0` (`bwd_dv_fp16_real_readback_is_zero`). The faithful
  statement therefore reads them back at the **fp16 `MemCell`** level as
  `MemCell.of .fp16 (real.cast fp16 (some <genuine real accumulator sum>))` — the
  raw real sum cast to fp16 on store, exactly mirroring the verified forward `Out`
  summary `triton_attention_forward_surface_out_python_test_shape_compute_correct`.
  The genuine sums are `DV[j,e] = Σ_i fp16(p[i,j])·do[i,e]` and
  `DK[j,e] = Σ_i fp16(ds[i,j])·q[i,e]`. -/
theorem triton_attention_bwd_grads_genuine_output_summary
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState)
    (hpids0lt : s.pids 0 < 8) (hpids1 : s.pids 1 = 0)
    (hDVDQ : DV ≠ DQ) (hDKDQ : DK ≠ DQ) (hDVDK : DV ≠ DK) (hDKDV : DK ≠ DV)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 1 128 64 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdKernelDQSpec s Q K V DO M Delta DQ idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ i : Fin 128,
            bwdFp16 (bwdKernelP s Q K M i idx.1) *
              bwdKernelDO s DO i ⟨idx.2.1.val, by omega⟩))))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ i : Fin 128,
            bwdFp16 (bwdKernelDS s Q K V DO M Delta i idx.1) *
              bwdKernelQ s Q i ⟨idx.2.1.val, by omega⟩))))) := by
  obtain ⟨sF, hexec, hDQ, hDV, hDK⟩ :=
    bwd_grads_exec s Q K V Out DO DQ DK DV L M Delta hpids0lt hpids1
      hDVDQ hDKDQ hDVDK hDKDV hundef
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- toAlgorithm conjunct
    exact triton_attention_bwd_kernel_toAlgorithm_supported Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      2 4 128 1024 1 128 64 128
  · -- DQ: real readback = bwdKernelDQSpec
    unfold ComputeCorrect.Realizes
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx
    rw [hexec] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    exact hDQ idx
  · -- DV: fp16 MemCell readback
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx _hActive
    rw [hexec] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_memcell]
    exact hDV idx
  · -- DK: fp16 MemCell readback
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx _hActive
    rw [hexec] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_memcell]
    exact hDK idx

end Correct

/-! # ══════════ TEST-SHAPE — concrete instances / pinned scaffolding ══════════ -/

section TestShape

theorem triton_attention_forward_output_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (hzRowOffset : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out
        hzRowOffset 1024 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset 1024 128 64 idx)))) := by
  exact triton_attention_forward_output_store_slice_compute_correct Acc Out
    hzRowOffset 1024 64 1 128 64 s
    (triton_attention_python_output_offset_injective s hzRowOffset)

theorem triton_attention_forward_l_store_python_test_shape_compute_correct
    (LPrev L : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => lStoreSpec s LPrev off_hz 128 128 i) := by
  exact triton_attention_forward_l_store_slice_compute_correct LPrev L
    off_hz 128 128 s (triton_attention_python_row_offset_injective s off_hz)

theorem triton_attention_forward_m_store_python_test_shape_compute_correct
    (MPrev M : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => mStoreSpec s MPrev off_hz 128 128 i) := by
  exact triton_attention_forward_m_store_slice_compute_correct MPrev M
    off_hz 128 128 s (triton_attention_python_row_offset_injective s off_hz)

theorem triton_attention_bwd_preprocess_newdo_store_python_test_shape_compute_correct
    (NewDOAcc NewDO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoStoreSpec s NewDOAcc 128 64 idx) := by
  exact triton_attention_bwd_preprocess_newdo_store_slice_compute_correct
    NewDOAcc NewDO 128 64 s (triton_attention_python_newdo_offset_injective s)

theorem triton_attention_bwd_preprocess_newdo_formula_python_test_shape_compute_correct
    (DO L NewDO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoFormulaSpec s DO L 128 64 idx) := by
  exact triton_attention_bwd_preprocess_newdo_formula_slice_compute_correct
    DO L NewDO 128 64 s (triton_attention_python_newdo_offset_injective s)

theorem triton_attention_bwd_preprocess_delta_formula_python_test_shape_compute_correct
    (Out DO L Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta 128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaFormulaSpec s Out DO L 128 64 i) := by
  exact triton_attention_bwd_preprocess_delta_formula_slice_compute_correct
    Out DO L Delta 128 64 s

theorem triton_attention_bwd_preprocess_delta_store_python_test_shape_compute_correct
    (DeltaAcc Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaStoreSpec s DeltaAcc 128 i) := by
  exact triton_attention_bwd_preprocess_delta_store_slice_compute_correct
    DeltaAcc Delta 128 s

theorem triton_attention_bwd_dq_store_python_test_shape_compute_correct
    (DQPre DQ : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ 4
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DQPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dq_store_slice_compute_correct DQPre DQ 4
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

theorem triton_attention_bwd_dk_store_python_test_shape_compute_correct
    (DKPre DK : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DKPre DK 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DKPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dk_store_slice_compute_correct DKPre DK 4 1024
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

theorem triton_attention_bwd_dv_store_python_test_shape_compute_correct
    (DVPre DV : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DVPre DV 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DVPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dv_store_slice_compute_correct DVPre DV 4 1024
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

/-! ## Genuine closed-form backward-gradient specs (main `_bwd_kernel`)

The definitions below give the **genuine** closed-form `DQ`/`DK`/`DV` gradient
values that `_bwd_kernel` stores at the checked Python launch shape
`(B,H,T,D)=(2,4,128,64)`, `num_block = 1`, `BLOCK_M = BLOCK_N = 128`,
`BLOCK_DMODEL = 64`, strides `(32768,8192,64,1)×3`, `sm_scale = 1/√64`.

They are expressed purely against the **loaded** memory of `Q`/`K`/`V`/`DO`/`M`/
`Delta`/`DQ` (via `s.readMem` at the `make_block_ptr`/pointer addresses) and the
program ids `s.pids 0` (`off_hz`), with `off_z = pids 0 / 4`, `off_h = pids 0 % 4`
— **never** referencing `exec`/`producedBwdKernelD…`. The block-ptr base for a
program is `off_z·32768 + off_h·8192`; the query row `i`, key row `j`, channel
`e` block-ptr addresses are `base + i·64 + e`, `base + j·64 + e`; the `m`/`Delta`
rows are at `off_hz·128 + i`. The fp16 rounding before the `tl.dot` accumulators
(and before the `DV`/`DK` stores) is modelled exactly as
`FloatDType.fp16.storeValue (FloatDType.real.cast .fp16 (some ·))`, mirroring the
banked `bwdDqDotStepSpec` / dv-dk store slices. -/

theorem triton_attention_bwd_kernel_dq_python_test_shape_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedBwdKernelDQValue, hExec]

theorem triton_attention_bwd_kernel_dk_python_test_shape_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBwdKernelDKValue, hExec]

theorem triton_attention_bwd_kernel_dv_python_test_shape_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBwdKernelDVValue, hExec]

/-! ### Genuine backward exec chain (main `_bwd_kernel`)

The lowered body of `_bwd_kernel` at the checked Python launch is
`bwdPreLoop ++ [Stmt.forRange "start_n" 0 1 1 outerBody]`, with the outer body
ending in two fp16 `DV`/`DK` stores after a single-iteration inner
`Stmt.forRangeDyn "start_m"` whose body ends in the real `DQ` store. Both ranges
are single-iteration (`num_block = 1`). -/

theorem triton_attention_bwd_score_p_formula_python_test_shape_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s QTile KTile MVec ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx) := by
  exact triton_attention_bwd_score_p_formula_slice_compute_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹)
    128 64 s triton_attention_python_bwd_score_offset_injective hRegions

theorem triton_attention_bwd_score_ds_formula_python_test_shape_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx) := by
  exact triton_attention_bwd_score_ds_formula_slice_compute_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹)
    128 64 s triton_attention_python_bwd_score_offset_injective hRegions

/-! ## Genuine closed-form forward specs (causal natural-exp FlashAttention-1)

The forward `_fwd_kernel` is a **causal** FlashAttention-1 forward using the
natural exponential (`tl.exp`), scalar `sm_scale = 1/√64`, the causal mask
`tl.where(offs_m[:,None] ≥ start_n + offs_n[None,:], qk, -inf)`, and a dynamic
KV-loop bound `(start_m + 1) * BLOCK_M` so only causal blocks contribute. For
the Python test shape `(B,H,T,D) = (2,4,128,64)`, `BLOCK_M = BLOCK_N = 128`,
`BLOCK_DMODEL = 64`, strides `(stride_qz,stride_qh,stride_qm,stride_qk) =
(32768,8192,64,1)` etc., a single M-block per program (`start_m = pids 0`),
`off_hz = pids 1`, and `stride_qh_2d = 8192 / 64 / 1 = 128`.

These definitions give the **genuine** closed-form values that `_fwd_kernel`
computes — the natural-exp causal attention block (`Out`), the per-row
log-sum-exp denominator (`L`), and the per-row score maximum (`M`) — written
against `VeriTile.Triton.attentionRealCausalBlock` and the underlying
`scaledScore`, **not** re-derived from the kernel's own `exec`. They are the
intended replacements for the self-referential `producedTritonAttentionForward*`
carriers below; the exec-reduction bridge to these specs (porting the
FlashAttention-1 causal exec recipe of `VeriTile/Examples/FlashAttention1/` to
this `make_block_ptr` / `forRangeDyn` kernel surface, plus the two extra L/M
masked stores) is the remaining proof stage tracked for this kernel. -/

set_option maxHeartbeats 1600000 in
/-- **Genuine forward `Out`-store correctness.** Every active output lane of the
`_fwd_kernel` holds the closed-form causal attention block `fwdOutSpec` (the
natural-exp `oPartial / lPartial` ratio of the loaded Q/K/V tiles), at the checked
honest grid (`pids 0 = 0`, `pids 1 < B·H = 8`). -/
theorem triton_attention_forward_surface_out_python_test_shape_compute_correct
    (Q K V L M Out : RegionName) (s : BlockState)
    (hpid0 : s.pids 0 = 0) (hpid1 : s.pids 1 < 8)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s (s.pids 1 * 128) 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s (s.pids 1 * 128) 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16 (some (fwdOutSpec s Q K V idx)))) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  obtain ⟨sF, hstep, hO, _, _⟩ := ta_exec Q K V L M Out s hpid0 hpid1 hLOut hMOut hLM hundef
  rw [hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_memcell]
  exact hO idx hActive

set_option maxHeartbeats 1600000 in
/-- **Genuine forward `L`-store correctness.** Every row lane holds the genuine
m-shifted causal normalizer `fwdLSpec` (`Σ_{j ≤ …} exp(score − M_row)`). -/
theorem triton_attention_forward_surface_l_python_test_shape_compute_correct
    (Q K V L M Out : RegionName) (s : BlockState)
    (hpid0 : s.pids 0 = 0) (hpid1 : s.pids 1 < 8)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s (s.pids 1) 128 128 i))
      (expected := fun i : Fin 128 => fwdLSpec s Q K i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  obtain ⟨sF, hstep, _, hLrb, _⟩ := ta_exec Q K V L M Out s hpid0 hpid1 hLOut hMOut hLM hundef
  rw [hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_real]
  exact hLrb i

set_option maxHeartbeats 1600000 in
/-- **Genuine forward `M`-store correctness.** Every row lane holds the genuine
per-row causal score maximum `fwdMSpec`. -/
theorem triton_attention_forward_surface_m_python_test_shape_compute_correct
    (Q K V L M Out : RegionName) (s : BlockState)
    (hpid0 : s.pids 0 = 0) (hpid1 : s.pids 1 < 8)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s (s.pids 1) 128 128 i))
      (expected := fun i : Fin 128 => fwdMSpec s Q K i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  obtain ⟨sF, hstep, _, _, hMrb⟩ := ta_exec Q K V L M Out s hpid0 hpid1 hLOut hMOut hLM hundef
  rw [hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_real]
  exact hMrb i

/-- Python forward shape summary: final output plus the row-wise `L` and `M`
side stores are compute-correct for the tested block shape. -/
theorem triton_attention_forward_python_test_shape_all_outputs_compute_correct
    (Acc LPrev MPrev Out L M : RegionName) (hzRowOffset off_hz : Nat)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out
        hzRowOffset 1024 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset 1024 128 64 idx))))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => lStoreSpec s LPrev off_hz 128 128 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => mStoreSpec s MPrev off_hz 128 128 i)) := by
  constructor
  · exact triton_attention_forward_output_store_python_test_shape_compute_correct
      Acc Out hzRowOffset s
  constructor
  · exact triton_attention_forward_l_store_python_test_shape_compute_correct
      LPrev L off_hz s
  · exact triton_attention_forward_m_store_python_test_shape_compute_correct
      MPrev M off_hz s

/-- Python backward-preprocess shape summary: the full `_bwd_preprocess`
surface realizes both observable outputs for the tested block shape. -/
theorem triton_attention_bwd_preprocess_python_test_shape_all_outputs_compute_correct
    (Out DO L NewDO Delta : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdPreprocessNewDOValue s Out DO L NewDO Delta 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 =>
        producedBwdPreprocessDeltaValue s Out DO L NewDO Delta 128 64 i)) := by
  constructor
  · exact triton_attention_bwd_preprocess_newdo_surface_compute_correct
      Out DO L NewDO Delta 128 64 s
  · exact triton_attention_bwd_preprocess_delta_surface_compute_correct
      Out DO L NewDO Delta 128 64 s

/-- Python backward gradient shape summary: the full `_bwd_kernel` realizes the
final `DQ`, `DK`, and `DV` outputs for the tested block shape. -/
theorem triton_attention_bwd_grads_python_test_shape_all_outputs_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx)) := by
  constructor
  · exact triton_attention_bwd_kernel_dq_python_test_shape_compute_correct
      Q K V Out DO DQ DK DV L M Delta s
  constructor
  · exact triton_attention_bwd_kernel_dk_python_test_shape_compute_correct
      Q K V Out DO DQ DK DV L M Delta s
  · exact triton_attention_bwd_kernel_dv_python_test_shape_compute_correct
      Q K V Out DO DQ DK DV L M Delta s

/-- Python backward score arithmetic shape summary: the `P` probability tile
and `DS` score-gradient tile are compute-correct for the tested block shape. -/
theorem triton_attention_bwd_score_python_test_shape_all_outputs_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s QTile KTile MVec ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx)) := by
  constructor
  · exact triton_attention_bwd_score_p_formula_python_test_shape_compute_correct
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile s hRegions
  · exact triton_attention_bwd_score_ds_formula_python_test_shape_compute_correct
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile s hRegions

/-- Python-shape arithmetic surface for the main `_bwd_kernel` score step.

This pins the checked launch's one-block backward inner step at
`BLOCK_M = BLOCK_N = 128`, `BLOCK_DMODEL = 64`, and `sm_scale = 1 / sqrt(64)`.
It exposes compute-correct `P` and `DS` tiles that feed the checked DQ/DK/DV
dot-step proofs, instead of treating those score-side inputs as opaque
precomputed regions. -/
theorem triton_attention_bwd_score_python_test_shape_formula_summary
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    (∃ alg, (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
      MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹)
      128 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s QTile KTile MVec ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx)) := by
  constructor
  · exact triton_attention_bwd_score_formula_slice_toAlgorithm_supported
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile
      ((Real.sqrt (64 : ℝ))⁻¹) 128 64
  · exact triton_attention_bwd_score_python_test_shape_all_outputs_compute_correct
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile s hRegions



/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- Public Python forward summary for `triton_attention.py`.

The surface conjunct pins the faithful `_fwd_kernel` launch for the checked
shape `(B, H, T, D) = (2, 4, 128, 64)`, contiguous Q/K/V/O strides
`(32768, 8192, 64, 1)`, `BLOCK_M = BLOCK_N = 128`, and
`BLOCK_DMODEL = 64`. The output conjuncts read back the Python-observable
`Out`, `L`, and `M` stores from that full surface. -/
theorem triton_attention_forward_python_test_shape_output_summary
    (Q K V L M Out : RegionName) (s : BlockState)
    (hpid0 : s.pids 0 = 0) (hpid1 : s.pids 1 < 8)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (triton_attention_fwd_kernel Q K V L M Out
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 128 64 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s (s.pids 1 * 128) 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s (s.pids 1 * 128) 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16 (some (fwdOutSpec s Q K V idx))))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s (s.pids 1) 128 128 i))
      (expected := fun i : Fin 128 => fwdLSpec s Q K i)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s (s.pids 1) 128 128 i))
      (expected := fun i : Fin 128 => fwdMSpec s Q K i)) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact triton_attention_fwd_kernel_toAlgorithm_supported Q K V L M Out
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 128 64 128
  · exact triton_attention_forward_surface_out_python_test_shape_compute_correct
      Q K V L M Out s hpid0 hpid1 hLOut hMOut hLM hundef
  · exact triton_attention_forward_surface_l_python_test_shape_compute_correct
      Q K V L M Out s hpid0 hpid1 hLOut hMOut hLM hundef
  · exact triton_attention_forward_surface_m_python_test_shape_compute_correct
      Q K V L M Out s hpid0 hpid1 hLOut hMOut hLM hundef

/-- Public Python backward-preprocess summary for `triton_attention.py`.

This records the faithful `_bwd_preprocess` full surface at `BLOCK_M = 128`
and `D_HEAD = 64`, and connects both Python-observable `NewDO` and `Delta`
outputs directly to the produced full-surface values. -/
theorem triton_attention_bwd_preprocess_python_test_shape_output_summary
    (Out DO L NewDO Delta : RegionName) (s : BlockState) :
    (∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      128 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdPreprocessNewDOValue s Out DO L NewDO Delta 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 =>
        producedBwdPreprocessDeltaValue s Out DO L NewDO Delta 128 64 i)) := by
  constructor
  · exact triton_attention_bwd_preprocess_toAlgorithm_supported
      Out DO L NewDO Delta 128 64
  · exact triton_attention_bwd_preprocess_python_test_shape_all_outputs_compute_correct
      Out DO L NewDO Delta s

/-- Public Python backward-gradient summary for `triton_attention.py`.

The surface conjunct pins the checked Python launch of the main `_bwd_kernel`
for `(B, H, T, D) = (2, 4, 128, 64)` with `num_block = 1`; the output
conjuncts connect the Python-observable `DQ`, `DK`, and `DV` writes directly to
the produced full-kernel values. -/
theorem triton_attention_bwd_grads_python_test_shape_output_summary
    (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (s : BlockState) :
    (∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx)) := by
  constructor
  · exact triton_attention_bwd_kernel_toAlgorithm_supported Q K V Out DO DQ DK
      DV L M Delta ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128
  · exact triton_attention_bwd_grads_python_test_shape_all_outputs_compute_correct
      Q K V Out DO DQ DK DV L M Delta s

/-- Combined checked-shape backward summary for `triton_attention.py`.

This exposes the main `_bwd_kernel` surface, final `DQ`/`DK`/`DV` writebacks,
and the score-side `P`/`DS` arithmetic producer in one public target. -/
theorem triton_attention_bwd_python_test_shape_complete_summary
    (Q K V Out DO DQ DK DV L M Delta PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    ((∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx))) ∧
    ((∃ alg, (triton_attention_bwd_score_formula_slice Q K V DO M Delta
      PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice Q K V DO M Delta
        PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s Q K M ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice Q K V DO M Delta
        PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s Q K V DO M Delta ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx))) := by
  constructor
  · exact triton_attention_bwd_grads_python_test_shape_output_summary Q K V Out
      DO DQ DK DV L M Delta s
  · exact triton_attention_bwd_score_python_test_shape_formula_summary Q K V
      DO M Delta PTile DSTile s hRegions

end TestShape

end VeriTile.Bench.TritonBenchG.TritonAttention

