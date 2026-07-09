import VeriTile.Triton
import VeriTile.Examples.FlashAttention1

/-!
# `triton_attention` — strict per-kernel correctness

`triton_attention.py` is a full FlashAttention training pipeline of three
`@triton.jit` kernels: `_fwd_kernel` (online-softmax forward, stores the output
`Out` plus the running `L`/`M` log-sum-exp rows), `_bwd_preprocess` (computes
`NewDO = DO/L` and the per-row `Delta = sum(O·(DO/L))`), and `_bwd_kernel` (the main
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
triton_attention_bwd_grads_genuine_output_summary_general           ← ★ MAIN (bwd grads, multi-block dimension-general)
  └─ genuine input-memory closed forms: bwdKernelDQSpecG (priorDQ + Σ_J fp16(ds)·k);
       DV[J,e] = Σ_I fp16(p)·do, DK[J,e] = Σ_I fp16(ds)·q (genuine fp16 column sums)
  bwd score P/DS step: triton_attention_bwd_score_{p,ds}_formula_slice_compute_correct  ← closed-form P/DS

triton_attention_forward_output_summary_general                     ← ★ TOP (forward Out/L/M, dimension-general)
  └─ ta_execG → taPreLoop_evalG / ta_attn_stepG / ta_postLoopG (general streaming stack)
       └─ fwdOutSpecG / fwdLSpecG / fwdMSpecG (genuine input-memory causal specs)

triton_attention_bwd_preprocess_genuine_output_summary_general      ← ★ TOP (preprocess NewDO/Delta, dimension-general)
  └─ bwdPreprocessNewDOSpecG / bwdPreprocessDeltaSpecG (genuine input-memory specs)
```
(Offset injectivity taken as hypotheses of the main theorem.)

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
* **Backward grads** (`DQ`, `DK`, `DV`) are verified against explicit closed
  forms defined over the **input** memory (never re-reading `exec`): the ★ MAIN
  dimension-general `triton_attention_bwd_grads_genuine_output_summary_general`
  checks the stores against `DQ = priorDQ + Σ_J fp16(ds)·k` (`bwdKernelDQSpecG`),
  `DV[J,e] = Σ_I fp16(p)·do`, and `DK[J,e] = Σ_I fp16(ds)·q` (the genuine fp16
  column sums over all query rows).
* **Backward preprocess** and the **backward score `P`/`DS` step** are verified
  against explicit closed-form specs (`bwdScorePFormulaSpec`,
  `bwdScoreDSFormulaSpec`, and the `newdo`/`delta` formula slices), not opaque
  carriers — these inner arithmetic steps are checked, the surrounding loop
  composition is trusted.

Side conditions: the dimension-general summaries quantify over symbolic
`(B,H,T,D)`, block sizes, strides and `sm_scale`, taking tile-offset
injectivity as hypotheses of the main theorem; the
backward-grads summary additionally requires the score tiles distinct
(`PTile ≠ DSTile`).
-/

namespace VeriTile.Bench.TritonBenchG.TritonAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `triton_attention_forward_output_summary_general` (forward, dimension-general), `triton_attention_bwd_preprocess_genuine_output_summary_general`, `triton_attention_bwd_grads_genuine_output_summary_general` (backward gradients, multi-block general) -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => lStoreSpec s LPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => mStoreSpec s MPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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

The real Python `_bwd_kernel` rewinds the `q`/`do`/`dq` block pointers after the
inner loop with a *signed*, runtime-dependent `tl.advance(_, [lo + (1 - num_block)
* BLOCK_M, 0])` (it depends on the outer-loop counter `start_n`). That negative,
loop-counter-dependent rewind is not expressible as a static `tl.advance` delta.

Following the repo precedent (`attention_fwd_triton1` and the forward
`triton_attention` kernel), this surface re-models `q`/`do`/`dq` with a *dynamic
offset* referencing the loop counter: the three pointers are reconstructed via
`tl.make_block_ptr` at the start of each outer (`start_n`) iteration with
`offsets = (… + lo, 0)` where `lo = start_n * BLOCK_M`, then advanced
`[BLOCK_M, 0]` per inner step. The memory addresses accessed are byte-identical
to the Python advance+rewind flow, so this surface is *faithful at the observable
(memory) level for arbitrary `num_block`*. The `k`/`v`/`dk`/`dv` pointers advance
monotonically `[BLOCK_M, 0]` once per outer iteration and are unchanged. -/
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
  for start_n in range($(0), $(num_block), $(1)) {
    lo = start_n * $(BLOCK_M)
    q_tile_ptr = tl.make_block_ptr(base=Q,
      shape=($(D0), $(BLOCK_DMODEL)),
      strides=($(stride_qm), $(stride_qk)),
      offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d + lo, 0),
      block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
      order=(1, 0))
    do_tile_ptr = tl.make_block_ptr(base=DO,
      shape=($(D0), $(BLOCK_DMODEL)),
      strides=($(stride_qm), $(stride_qk)),
      offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d + lo, 0),
      block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
      order=(1, 0))
    dq_tile_ptr = tl.make_block_ptr(base=DQ,
      shape=($(D0), $(BLOCK_DMODEL)),
      strides=($(stride_qm), $(stride_qk)),
      offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d + lo, 0),
      block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
      order=(1, 0))
    DQ = DQ + off_z * $(stride_qz) + off_h * $(stride_qh)
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoFormulaSpec s DO L BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaFormulaSpec s Out DO L BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaStoreSpec s DeltaAcc BLOCK_M i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_delta_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_bwd_preprocess_delta_store_slice_correct
    DeltaAcc Delta BLOCK_M s i
  rw [hExec] at h
  exact Option.some.inj h

/-! ### Genuine closed-form preprocess specs (input-memory)

The two specs below are the GENUINE Python `_bwd_preprocess` closed forms,
written purely over the kernel's **input** regions `Out`, `DO`, `L` — they do
*not* reference the kernel's own `NewDO`/`Delta` output (no self-referential
executed-readback carrier).

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

/-- `Realizes_without_Rounding` form of the genuine `NewDO` correctness for the full surface. -/
theorem triton_attention_bwd_preprocess_newdo_genuine_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hND : NewDO ≠ Delta)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        bwdPreprocessNewDOSpecG s Out DO L BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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

/-- `Realizes_without_Rounding` form of the genuine `Delta` correctness for the full surface. -/
theorem triton_attention_bwd_preprocess_delta_genuine_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        bwdPreprocessDeltaSpecG s Out DO L BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
genuine input-memory closed forms, **not** a self-referential executed readback. -/
theorem triton_attention_bwd_preprocess_genuine_output_summary_general
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hND : NewDO ≠ Delta)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    (∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        bwdPreprocessNewDOSpecG s Out DO L BLOCK_M D_HEAD idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        some (PTile, bwdScoreOffset BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        bwdScorePFormulaSpec s QTile KTile MVec sm_scale BLOCK_M
          BLOCK_DMODEL idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        some (DSTile, bwdScoreOffset BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          sm_scale BLOCK_M BLOCK_DMODEL idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_dq_dot_step_slice DQPrev DS KTile DQ H
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh stride_qm
          stride_qk BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ H stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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

/-- Block-ptr base offset for the program: `off_z·32768 + off_h·8192`. -/
def bwdKBase (s : BlockState) : Nat :=
  s.pids 0 / 4 * 32768 + s.pids 0 % 4 * 8192

/-- fp16 round-trip on a real value, as performed by `tl.…to(tl.float16)`. -/
noncomputable def bwdFp16 (x : ℝ) : ℝ :=
  FloatDType.fp16.storeValue (FloatDType.real.cast FloatDType.fp16 (some x))

/-! ### ════════ Generalized backward-gradient spec layer (Phase 1) ════════

Dimension-general genuine backward-gradient specs over the **input** memory
`Q`/`K`/`V`/`DO`/`M`/`Delta`, parametric over `BM = BLOCK_M = BLOCK_N` (square
score blocks), `BD = BLOCK_DMODEL` (head dim), and `nb = num_block` (so the
sequence length is `N_CTX = nb * BM`).  Global query/key rows range over
`Fin (nb * BM)`; head channels over `Fin BD`.  Memory layout mirrors the pinned
specs: tile stride `BD`, base `bwdKBase`, `M`/`Delta` indexed at
`pids0 * (nb*BM) + I`.

These collapse to the pinned `Fin 128`/`Fin 64` specs at `BM = 128`, `BD = 64`,
`nb = 1`.  The honest closed forms hold for the genuine FlashAttention-1
backward math; the bridges below are the per-block partial-sum recurrences that
the eventual two-level loop invariant produces.

These defs and lemmas are **surface-independent** (they reference only the input
memory and the pinned spec layer, never the kernel AST), ported verbatim from
the parked `generalize/triton-bwd-grads` branch. -/

/-- Loaded `q[I,e] = Q[base + I·BD + e]` at global query row `I`. -/
noncomputable def bwdKernelQG (s : BlockState) (Q : RegionName) (BD : Nat)
    (I : Nat) (e : Nat) : ℝ :=
  s.readMem Q (bwdKBase s + I * BD + e)

/-- Loaded `k[J,e] = K[base + J·BD + e]` at global key row `J`. -/
noncomputable def bwdKernelKG (s : BlockState) (K : RegionName) (BD : Nat)
    (J : Nat) (e : Nat) : ℝ :=
  s.readMem K (bwdKBase s + J * BD + e)

/-- Loaded `v[J,e] = V[base + J·BD + e]`. -/
noncomputable def bwdKernelVG (s : BlockState) (V : RegionName) (BD : Nat)
    (J : Nat) (e : Nat) : ℝ :=
  s.readMem V (bwdKBase s + J * BD + e)

/-- Loaded `do[I,e] = DO[base + I·BD + e]`. -/
noncomputable def bwdKernelDOG (s : BlockState) (DO : RegionName) (BD : Nat)
    (I : Nat) (e : Nat) : ℝ :=
  s.readMem DO (bwdKBase s + I * BD + e)

/-- Loaded `m[I] = M[off_hz·N_CTX + I]` (`N_CTX = nb·BM`). -/
noncomputable def bwdKernelMG (s : BlockState) (M : RegionName) (NCTX : Nat)
    (I : Nat) : ℝ :=
  s.readMem M (s.pids 0 * NCTX + I)

/-- Loaded `Di[I] = Delta[off_hz·N_CTX + I]`. -/
noncomputable def bwdKernelDiG (s : BlockState) (Delta : RegionName) (NCTX : Nat)
    (I : Nat) : ℝ :=
  s.readMem Delta (s.pids 0 * NCTX + I)

/-- Prior (pre-accumulation) `dq[I,e] = DQ[base + I·BD + e]` — the `DQ` buffer
value the kernel's `+=` accumulates onto (same tile layout as `Q`/`DO`). -/
noncomputable def bwdKernelDQ0G (s : BlockState) (DQ : RegionName) (BD : Nat)
    (I : Nat) (e : Nat) : ℝ :=
  s.readMem DQ (bwdKBase s + I * BD + e)

/-- `qk[I,J] = Σ_e q[I,e]·k[J,e]` (the `tl.dot(q, trans(k))` score). -/
noncomputable def bwdKernelQKG (s : BlockState) (Q K : RegionName) (BD : Nat)
    (I J : Nat) : ℝ :=
  ∑ e : Fin BD, bwdKernelQG s Q BD I e.val * bwdKernelKG s K BD J e.val

/-- `p[I,J] = exp(qk·sm_scale − m[I])` with causal masking (`J ≤ I` keeps the
score, else `0`); mirrors the kernel `tl.where` / `tl.exp`. -/
noncomputable def bwdKernelPG (s : BlockState) (Q K M : RegionName) (BD NCTX : Nat)
    (sc : ℝ) (I J : Nat) : ℝ :=
  if J ≤ I then
    Real.exp (bwdKernelQKG s Q K BD I J * sc - bwdKernelMG s M NCTX I)
  else 0

/-- `dp[I,J] = (Σ_e do[I,e]·v[J,e]) − Di[I]`. -/
noncomputable def bwdKernelDPG (s : BlockState) (V DO Delta : RegionName) (BD NCTX : Nat)
    (I J : Nat) : ℝ :=
  (∑ e : Fin BD, bwdKernelDOG s DO BD I e.val * bwdKernelVG s V BD J e.val)
    - bwdKernelDiG s Delta NCTX I

/-- `ds[I,J] = p[I,J]·dp[I,J]·sm_scale`. -/
noncomputable def bwdKernelDSG (s : BlockState) (Q K V DO M Delta : RegionName)
    (BD NCTX : Nat) (sc : ℝ) (I J : Nat) : ℝ :=
  bwdKernelPG s Q K M BD NCTX sc I J * bwdKernelDPG s V DO Delta BD NCTX I J * sc

/-- **Genuine general `DQ` value.** `dq[I,e] = priorDQ[I,e] + Σ_J fp16(ds[I,J])·k[J,e]`,
summed over **all** global key rows `J ∈ [0, N_CTX)` (causal `p ⇒ ds` zeroes
`J>I`), stored real (no fp16 cast on the `DQ` store). -/
noncomputable def bwdKernelDQSpecG
    (s : BlockState) (Q K V DO M Delta DQ : RegionName) (BD NCTX : Nat) (sc : ℝ)
    (I e : Nat) : ℝ :=
  bwdKernelDQ0G s DQ BD I e +
    ∑ J : Fin NCTX,
      bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD NCTX sc I J.val) *
        bwdKernelKG s K BD J.val e

/-! #### Block-decomposition partial-sum bridges (two-level loop content)

The genuine `DV`/`DK` per-cell sums range over **all** global query rows
`I ∈ Fin (BM·nb)`.  The two-level loop produces these as an iterated
accumulation: for a fixed KV block, the inner Q-block loop accumulates the
per-block sub-sums.  These bridges rewrite the flat `Fin (BM·nb)` sum as a
`Fin nb × Fin BM` block-decomposed double sum (outer query block `m`, local row
`iL`, global row `= m·BM + iL`), exactly the order the loop visits.  They are the
backward analog of the forward `mPartialG_eq_blockSup` block-decomposition. -/

/-- A flat sum over global query rows `Fin (BM·nb)` decomposes blockwise into
`∑ (queryBlock m : Fin nb) ∑ (localRow iL : Fin BM) f (m·BM + iL)`.  The body is
indexed by the global row `(blockIndexEquiv BM nb).symm (m, iL)` whose value is
`m·BM + iL`. -/
theorem bwd_flatSum_eq_blockSum {BM nb : Nat} (f : Fin (BM * nb) → ℝ) :
    (∑ I : Fin (BM * nb), f I)
      = ∑ m : Fin nb, ∑ iL : Fin BM,
          f ((StreamingAccumulator.blockIndexEquiv BM nb).symm (m, iL)) := by
  rw [← Finset.sum_product', Finset.univ_product_univ]
  rw [← Equiv.sum_comp (StreamingAccumulator.blockIndexEquiv BM nb).symm f]

/-- The global row value of `(blockIndexEquiv BM nb).symm (m, iL)` is `m·BM + iL`. -/
theorem bwd_blockIndexEquiv_symm_val {BM nb : Nat} (m : Fin nb) (iL : Fin BM) :
    ((StreamingAccumulator.blockIndexEquiv BM nb).symm (m, iL)).val
      = m.val * BM + iL.val := by
  show ((Fin.castOrderIso (Nat.mul_comm BM nb)).toEquiv.symm
      (finProdFinEquiv (m, iL))).val = m.val * BM + iL.val
  show (finProdFinEquiv (m, iL)).val = m.val * BM + iL.val
  show iL.val + BM * m.val = m.val * BM + iL.val
  rw [Nat.mul_comm m.val BM]; omega

/-- **DQ block-decomposition bridge.** The genuine general `DQ` cell sum ranges
over all global **key** rows; for a fixed query block, the inner loop streams the
key blocks, so the natural decomposition is over key blocks (block `n`, local
col `jL`, global col `n·BM + jL`). -/
theorem bwdKernelDQSpecG_blockSum (s : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BD BM nb : Nat) (sc : ℝ) (I e : Nat) :
    bwdKernelDQSpecG s Q K V DO M Delta DQ BD (BM * nb) sc I e
      = s.readMem DQ (bwdKBase s + I * BD + e) +
        ∑ n : Fin nb, ∑ jL : Fin BM,
          bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I (n.val * BM + jL.val)) *
            bwdKernelKG s K BD (n.val * BM + jL.val) e := by
  simp only [bwdKernelDQSpecG, bwdKernelDQ0G]
  refine congrArg (s.readMem DQ (bwdKBase s + I * BD + e) + ·) ?_
  rw [bwd_flatSum_eq_blockSum
    (fun J : Fin (BM * nb) =>
      bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I J.val) *
        bwdKernelKG s K BD J.val e)]
  refine Finset.sum_congr rfl (fun n _ => Finset.sum_congr rfl (fun jL _ => ?_))
  rw [bwd_blockIndexEquiv_symm_val]

/-! #### Inner-loop accumulator infrastructure (surface-independent)

The inner `start_m` loop, for a fixed KV block `n` (key base `n·BM`), streams the
query blocks `m = n, n+1, …, num_block-1`, accumulating into the `dv`/`dk`
registers.  These defs name the per-query-block column sub-sums and their running
partial accumulation; the `_full` bridges show that after the inner loop runs
`nb - n` iterations the accumulator equals the full block-decomposed column sum
(the causally-zero lower blocks `m < n` are added for free), and the
`_fp16_eq_spec` bridges connect the fp16-cast accumulator to the genuine general
`DV`/`DK` specs.  Ported verbatim from the parked branch (spec-only). -/

/-- Per-query-block `dv` column sub-sum (KV block `n`, query block `m`,
cell `(j,e)`): `Σ_iL fp16(pG (m·BM+iL) (n·BM+j))·doG (m·BM+iL) e`. -/
noncomputable def bwdSubDv (s : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n m j e : Nat) : ℝ :=
  ∑ iL : Fin BM,
    bwdFp16 (bwdKernelPG s Q K M BD NCTX sc (m * BM + iL.val) (n * BM + j)) *
      bwdKernelDOG s DO BD (m * BM + iL.val) e

/-- Per-query-block `dk` column sub-sum (KV block `n`, query block `m`,
cell `(j,e)`): `Σ_iL fp16(dsG (m·BM+iL) (n·BM+j))·qG (m·BM+iL) e`. -/
noncomputable def bwdSubDk (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n m j e : Nat) : ℝ :=
  ∑ iL : Fin BM,
    bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD NCTX sc (m * BM + iL.val) (n * BM + j)) *
      bwdKernelQG s Q BD (m * BM + iL.val) e

/-- **Causal zero of `dv` sub-sum below the diagonal block.** For a query block
`m < n`, every query row `m·BM+iL` lies before every key row `n·BM+j`, so
`pG = 0` and the sub-sum vanishes. -/
theorem bwdSubDv_zero_of_lt (s : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n m j e : Nat) (hjm : j < BM) (hmn : m < n) :
    bwdSubDv s Q K M DO BM BD NCTX sc n m j e = 0 := by
  refine Finset.sum_eq_zero (fun iL _ => ?_)
  have hlt : m * BM + iL.val < n * BM + j := by
    have h1 : m * BM + iL.val < (m + 1) * BM := by
      have := iL.isLt; rw [Nat.add_mul, Nat.one_mul]; omega
    have h2 : (m + 1) * BM ≤ n * BM := Nat.mul_le_mul_right BM hmn
    omega
  have hp0 : bwdKernelPG s Q K M BD NCTX sc (m * BM + iL.val) (n * BM + j) = 0 := by
    simp only [bwdKernelPG, if_neg (by omega : ¬ n * BM + j ≤ m * BM + iL.val)]
  rw [hp0, show bwdFp16 0 = 0 from by simp only [bwdFp16, FloatDType.cast,
      FloatDType.ofWithBot, FloatDType.toWithBot, FloatDType.storeValue]; norm_num, zero_mul]

/-- **Causal zero of `dk` sub-sum below the diagonal block.** -/
theorem bwdSubDk_zero_of_lt (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n m j e : Nat) (hjm : j < BM) (hmn : m < n) :
    bwdSubDk s Q K V DO M Delta BM BD NCTX sc n m j e = 0 := by
  refine Finset.sum_eq_zero (fun iL _ => ?_)
  have hlt : m * BM + iL.val < n * BM + j := by
    have h1 : m * BM + iL.val < (m + 1) * BM := by
      have := iL.isLt; rw [Nat.add_mul, Nat.one_mul]; omega
    have h2 : (m + 1) * BM ≤ n * BM := Nat.mul_le_mul_right BM hmn
    omega
  have hds0 : bwdKernelDSG s Q K V DO M Delta BD NCTX sc (m * BM + iL.val) (n * BM + j) = 0 := by
    simp only [bwdKernelDSG, bwdKernelPG, if_neg (by omega : ¬ n * BM + j ≤ m * BM + iL.val),
      zero_mul]
  rw [hds0, show bwdFp16 0 = 0 from by simp only [bwdFp16, FloatDType.cast,
      FloatDType.ofWithBot, FloatDType.toWithBot, FloatDType.storeValue]; norm_num, zero_mul]

/-- Running `dv` accumulator cell after `t` inner iterations from KV block `n`:
the sum of the sub-sums of query blocks `n, n+1, …, n+t-1`. -/
noncomputable def bwdAccDvCell (s : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n t j e : Nat) : ℝ :=
  ∑ m ∈ Finset.range t, bwdSubDv s Q K M DO BM BD NCTX sc n (n + m) j e

/-- Running `dk` accumulator cell after `t` inner iterations from KV block `n`. -/
noncomputable def bwdAccDkCell (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n t j e : Nat) : ℝ :=
  ∑ m ∈ Finset.range t, bwdSubDk s Q K V DO M Delta BM BD NCTX sc n (n + m) j e

theorem bwdAccDvCell_succ (s : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n t j e : Nat) :
    bwdAccDvCell s Q K M DO BM BD NCTX sc n (t + 1) j e
      = bwdAccDvCell s Q K M DO BM BD NCTX sc n t j e
        + bwdSubDv s Q K M DO BM BD NCTX sc n (n + t) j e := by
  simp only [bwdAccDvCell, Finset.sum_range_succ]

theorem bwdAccDkCell_succ (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n t j e : Nat) :
    bwdAccDkCell s Q K V DO M Delta BM BD NCTX sc n (t + 1) j e
      = bwdAccDkCell s Q K V DO M Delta BM BD NCTX sc n t j e
        + bwdSubDk s Q K V DO M Delta BM BD NCTX sc n (n + t) j e := by
  simp only [bwdAccDkCell, Finset.sum_range_succ]

/-- At the end of the inner loop (`t = nb - n`, `n ≤ nb`), the running `dv`
accumulator equals the **full** block-decomposed column sum over all query
blocks `m ∈ Fin nb`: the lower blocks `m < n` are causally zero. -/
theorem bwdAccDvCell_full (s : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX nb : Nat) (sc : ℝ) (n j e : Nat) (hjm : j < BM) (hn : n ≤ nb) :
    bwdAccDvCell s Q K M DO BM BD NCTX sc n (nb - n) j e
      = ∑ m : Fin nb, bwdSubDv s Q K M DO BM BD NCTX sc n m.val j e := by
  rw [Finset.sum_fin_eq_sum_range]
  rw [← Finset.sum_range_add_sum_Ico _ hn]
  have hlow : (∑ m ∈ Finset.range n,
      (if h : m < nb then bwdSubDv s Q K M DO BM BD NCTX sc n m j e else 0)) = 0 := by
    refine Finset.sum_eq_zero (fun m hm => ?_)
    rw [Finset.mem_range] at hm
    by_cases h : m < nb
    · rw [dif_pos h, bwdSubDv_zero_of_lt s Q K M DO BM BD NCTX sc n m j e hjm hm]
    · rw [dif_neg h]
  rw [hlow, zero_add]
  rw [bwdAccDvCell]
  rw [Finset.sum_Ico_eq_sum_range]
  refine Finset.sum_congr rfl (fun t ht => ?_)
  rw [Finset.mem_range] at ht
  rw [dif_pos (by omega : n + t < nb)]

/-- At the end of the inner loop, the running `dk` accumulator equals the full
block-decomposed column sum over all query blocks. -/
theorem bwdAccDkCell_full (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX nb : Nat) (sc : ℝ) (n j e : Nat) (hjm : j < BM) (hn : n ≤ nb) :
    bwdAccDkCell s Q K V DO M Delta BM BD NCTX sc n (nb - n) j e
      = ∑ m : Fin nb, bwdSubDk s Q K V DO M Delta BM BD NCTX sc n m.val j e := by
  rw [Finset.sum_fin_eq_sum_range]
  rw [← Finset.sum_range_add_sum_Ico _ hn]
  have hlow : (∑ m ∈ Finset.range n,
      (if h : m < nb then bwdSubDk s Q K V DO M Delta BM BD NCTX sc n m j e else 0)) = 0 := by
    refine Finset.sum_eq_zero (fun m hm => ?_)
    rw [Finset.mem_range] at hm
    by_cases h : m < nb
    · rw [dif_pos h, bwdSubDk_zero_of_lt s Q K V DO M Delta BM BD NCTX sc n m j e hjm hm]
    · rw [dif_neg h]
  rw [hlow, zero_add]
  rw [bwdAccDkCell]
  rw [Finset.sum_Ico_eq_sum_range]
  refine Finset.sum_congr rfl (fun t ht => ?_)
  rw [Finset.mem_range] at ht
  rw [dif_pos (by omega : n + t < nb)]

/-- **`dv` accumulator → genuine raw column sum.** The end-of-loop `dv` cell (the
value stored, *before* the fp16 cast) equals the flat genuine column sum
`Σ_I fp16(pG I (n·BM+j))·doG I e` over all query rows `I ∈ Fin (BM·nb)`. -/
theorem bwdAccDvCell_full_eq_rawSum (s : BlockState) (Q K M DO : RegionName)
    (BM BD nb : Nat) (sc : ℝ) (n j e : Nat) (hjm : j < BM) (hn : n ≤ nb) :
    bwdAccDvCell s Q K M DO BM BD (BM * nb) sc n (nb - n) j e
      = ∑ I : Fin (BM * nb),
          bwdFp16 (bwdKernelPG s Q K M BD (BM * nb) sc I.val (n * BM + j)) *
            bwdKernelDOG s DO BD I.val e := by
  rw [bwdAccDvCell_full s Q K M DO BM BD (BM * nb) nb sc n j e hjm hn]
  rw [bwd_flatSum_eq_blockSum
    (fun I : Fin (BM * nb) =>
      bwdFp16 (bwdKernelPG s Q K M BD (BM * nb) sc I.val (n * BM + j)) *
        bwdKernelDOG s DO BD I.val e)]
  refine Finset.sum_congr rfl (fun m _ => ?_)
  rw [bwdSubDv]
  refine Finset.sum_congr rfl (fun iL _ => ?_)
  rw [bwd_blockIndexEquiv_symm_val]

/-- **`dk` accumulator → genuine raw column sum.** -/
theorem bwdAccDkCell_full_eq_rawSum (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD nb : Nat) (sc : ℝ) (n j e : Nat) (hjm : j < BM) (hn : n ≤ nb) :
    bwdAccDkCell s Q K V DO M Delta BM BD (BM * nb) sc n (nb - n) j e
      = ∑ I : Fin (BM * nb),
          bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I.val (n * BM + j)) *
            bwdKernelQG s Q BD I.val e := by
  rw [bwdAccDkCell_full s Q K V DO M Delta BM BD (BM * nb) nb sc n j e hjm hn]
  rw [bwd_flatSum_eq_blockSum
    (fun I : Fin (BM * nb) =>
      bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I.val (n * BM + j)) *
        bwdKernelQG s Q BD I.val e)]
  refine Finset.sum_congr rfl (fun m _ => ?_)
  rw [bwdSubDk]
  refine Finset.sum_congr rfl (fun iL _ => ?_)
  rw [bwd_blockIndexEquiv_symm_val]

/-- Single-key-block `DQ` contribution at global query row `I`, channel `e`
(KV block `n`): `Σ_jL fp16(dsG I (n·BM+jL))·kG (n·BM+jL) e`.  The inner-loop
read-modify-write of `DQ` adds exactly this for each query row it visits. -/
noncomputable def bwdDqKeyContrib (s0 : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n I e : Nat) : ℝ :=
  ∑ jL : Fin BM,
    bwdFp16 (bwdKernelDSG s0 Q K V DO M Delta BD NCTX sc I (n * BM + jL.val)) *
      bwdKernelKG s0 K BD (n * BM + jL.val) e

/-! ### ════════ Generalized backward-gradient AST layer (Phase 2) ════════

Dimension-general clones of the lowered `_bwd_kernel` loop bodies on the **new**
faithful surface (dynamic loop-counter `lo` offset for the `q`/`do`/`dq` block
pointers, built inside the outer body; no post-inner `[0,0]` rewinds), parametric
over `BM = BLOCK_M = BLOCK_N` (square score blocks), `BD = BLOCK_DMODEL` (head
dim) and `nb = num_block`.  Strides are pinned to the contiguous launch
(`stride_qm = BD`, `stride_qk = 1`) used by the top theorem, mirroring the
forward `taLoopBodyG`/`ta_body_splitG`.  At `BM = 128`, `BD = 64`, `nb = 1` these
`bwdInnerBodyG`/`bwdOuterBodyG`/`bwdPreLoopG` specialize to the Python test shape. -/

/-- General `make_block_ptr` op (k/v/dk/dv style, no `lo`): parent `[D0, BD]`,
block `[BM, BD]`, contiguous strides `[BD, 1]`, dynamic row offset
`off_z·stride_qz_2d + off_h·stride_qh_2d`, col `0`. -/
private def bwdMkPtrG (R : RegionName) (D0 BM BD : Nat) : Op .blockPtr [BM, BD] :=
  Op.makeBlockPtrDynOffsets R (Op.constNat 0) [D0, BD] [BM, BD] [BD, 1]
    [Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.ref .nat [] "stride_qz_2d"))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.ref .nat [] "stride_qh_2d")),
      Op.constNat 0]

/-- General in-loop `make_block_ptr` op (q/do/dq style): adds the dynamic loop
offset `lo = start_n · BM` to the base-row offset. -/
private def bwdMkPtrLoG (R : RegionName) (D0 BM BD : Nat) : Op .blockPtr [BM, BD] :=
  Op.makeBlockPtrDynOffsets R (Op.constNat 0) [D0, BD] [BM, BD] [BD, 1]
    [Op.add .nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.ref .nat [] "stride_qz_2d"))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.ref .nat [] "stride_qh_2d")))
        (Op.ref .nat [] "lo"),
      Op.constNat 0]

/-- General block-ptr tile value: every lane is the same `make_block_ptr`-style
block pointer into region `R` with parentShape `[D0, BD]`, blockShape `[BM, BD]`,
contiguous strides `[BD, 1]`, `baseOffset = 0`, and 2-D offsets
`[base/BD + rowOff, 0]`.  When `BD ∣ base` the address of lane `(i,e)` is
`(base/BD + rowOff + i)·BD + e = base + (rowOff + i)·BD + e`. -/
def bwdPtrTileG (R : RegionName) (base D0 BM BD rowOff : Nat) :
    Tile .blockPtr [BM, BD] :=
  ⟨fun _ : TileIndex [BM, BD] =>
    { region := R, baseOffset := 0, parentShape := [D0, BD],
      blockShape := [BM, BD], strides := [BD, 1],
      offsets := [base / BD + rowOff, 0] }⟩

/-- Advancing the general block-ptr tile by `[BM, 0]` shifts the row offset by
`BM`, yielding the same `bwdPtrTileG` at row `rowOff + BM`. -/
theorem bwdPtrTileG_advance (R : RegionName) (base D0 BM BD rowOff : Nat) :
    (⟨fun i => ((bwdPtrTileG R base D0 BM BD rowOff).data i).advance [BM, 0]⟩
        : Tile .blockPtr [BM, BD])
      = bwdPtrTileG R base D0 BM BD (rowOff + BM) := by
  refine Tile.ext (fun i => ?_)
  simp only [bwdPtrTileG, BlockPtr.advance_2d_offsets_int, BlockPtr.toNat_natCast_add_natCast,
    BlockPtr.toNat_natCast_add_zero, BlockPtr.toNat_zero_add_natCast, BlockPtr.mk.injEq,
    List.cons.injEq, and_true, true_and, Nat.add_assoc]

/-- General pre-loop statements of `_bwd_kernel` (new surface): `off_hz`/`off_z`/
`off_h`/`stride_qz_2d`/`stride_qh_2d` and the k/v/dk/dv block pointers (the
q/do/dq pointers are constructed in the outer body). -/
private def bwdPreLoopG (K V DK DV : RegionName)
    (stride_qz stride_qh D0 BM BD H : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "off_hz" (Op.programId 0),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "stride_qz_2d"
      (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat stride_qz) (Op.constNat BD)) (Op.constNat 1)),
    Stmt.assign .nat [] "stride_qh_2d"
      (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat stride_qh) (Op.constNat BD)) (Op.constNat 1)),
    Stmt.assign .blockPtr [BM, BD] "k_tile_ptr" (bwdMkPtrG K D0 BM BD),
    Stmt.assign .blockPtr [BM, BD] "v_tile_ptr" (bwdMkPtrG V D0 BM BD),
    Stmt.assign .blockPtr [BM, BD] "dk_tile_ptr" (bwdMkPtrG DK D0 BM BD),
    Stmt.assign .blockPtr [BM, BD] "dv_tile_ptr" (bwdMkPtrG DV D0 BM BD) ]

/-- General lowered inner (`start_m`) loop body of `_bwd_kernel`. -/
private noncomputable def bwdInnerBodyG (sc : ℝ) (BM BD : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "offs_m_curr"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m") (Op.ref .nat [BM] "offs_m")),
    Stmt.assign .real [BM, BD] "q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "q_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [BM, BM] "qk"
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.transpose (batch := []) (Op.ref .real [BM, BD] "k"))),
    Stmt.assign .real [BM, BM] "qk"
      (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m_curr"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BM] "offs_n")))
        (Op.ref .real [BM, BM] "qk") (Op.broadcast Op.negInf [BM, BM])),
    Stmt.assign .real [BM] "m"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "m_ptrs") (Op.ref .nat [BM] "offs_m_curr")))
        MaskOpt.none),
    Stmt.assign .real [BM, BM] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BM] "qk") (Op.const sc))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m")))),
    Stmt.assign .real [BM, BD] "do_val"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "do_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [BM, BD] "dv"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "dv")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real
            (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] "p"))))
          (Op.ref .real [BM, BD] "do_val"))),
    Stmt.assign .real [BM] "Di"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "D_ptrs") (Op.ref .nat [BM] "offs_m_curr")))
        MaskOpt.none),
    Stmt.assign .real [BM, BM] "dp"
      (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.full [BM, BM] (Op.const 0))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "Di"))),
    Stmt.assign .real [BM, BM] "dp"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BM] "dp")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "do_val")
          (Op.transpose (batch := []) (Op.ref .real [BM, BD] "v")))),
    Stmt.assign .real [BM, BM] "ds"
      (Op.mul .real Broadcast.scalarR
        (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BM, BM] "p") (Op.ref .real [BM, BM] "dp"))
        (Op.const sc)),
    Stmt.assign .real [BM, BD] "dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "dk")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real
            (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] "ds"))))
          (Op.ref .real [BM, BD] "q"))),
    Stmt.assign .real [BM, BD] "dq"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "dq_tile_ptr") []) MaskOpt.none),
    Stmt.assign .real [BM, BD] "dq"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "dq")
        (Op.dot (batch := [])
          (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] "ds")))
          (Op.ref .real [BM, BD] "k"))),
    Stmt.store .real [BM, BD] (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "dq_tile_ptr") [])
      (Op.ref .real [BM, BD] "dq") MaskOpt.none,
    Stmt.assign .ptr [BM, BD] "dq_ptrs"
      (Op.ptrAdd Broadcast.scalarR
        (Op.ref .ptr [BM, BD] "dq_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BM) (Op.constNat BD))),
    Stmt.assign .blockPtr [BM, BD] "q_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "q_tile_ptr") [BM, (0:Nat)]),
    Stmt.assign .blockPtr [BM, BD] "do_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "do_tile_ptr") [BM, (0:Nat)]),
    Stmt.assign .blockPtr [BM, BD] "dq_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "dq_tile_ptr") [BM, (0:Nat)]) ]

/-- General lowered outer (`start_n`) loop body of `_bwd_kernel` (new surface):
the q/do/dq block pointers are constructed here at dynamic row offset `lo`, the
inner loop streams `start_m` from `lo` to `nb·BM` step `BM`, and k/v/dk/dv
advance by `[BM,0]` per outer iter. -/
private noncomputable def bwdOuterBodyG (Q DO DQ Delta M : RegionName)
    (sc : ℝ) (stride_qz stride_qh D0 BM BD nb N_CTX : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "lo"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BM)),
    Stmt.assign .blockPtr [BM, BD] "q_tile_ptr" (bwdMkPtrLoG Q D0 BM BD),
    Stmt.assign .blockPtr [BM, BD] "do_tile_ptr" (bwdMkPtrLoG DO D0 BM BD),
    Stmt.assign .blockPtr [BM, BD] "dq_tile_ptr" (bwdMkPtrLoG DQ D0 BM BD),
    Stmt.assign .ptr [] "DQ"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase DQ)
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh)))),
    Stmt.assign .nat [BM] "offs_qm"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "lo") (Op.arange BM)),
    Stmt.assign .nat [BM] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BM)) (Op.arange BM)),
    Stmt.assign .nat [BM] "offs_m" (Op.arange BM),
    Stmt.assign .nat [BD] "offs_k" (Op.arange BD),
    Stmt.assign .ptr [BM, BD] "dq_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "DQ")
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_qm")) (Op.constNat BD))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "D_ptrs"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Delta)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))),
    Stmt.assign .ptr [] "m_ptrs"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase M)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))),
    Stmt.assign .real [BM, BD] "dv" (Op.full [BM, BD] (Op.const 0)),
    Stmt.assign .real [BM, BD] "dk" (Op.full [BM, BD] (Op.const 0)),
    Stmt.assign .real [BM, BD] "k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "k_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [BM, BD] "v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "v_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo")
      (Op.mul .nat Broadcast.nil (Op.constNat nb) (Op.constNat BM)) (Op.constNat BM)
      (bwdInnerBodyG sc BM BD),
    Stmt.assign .blockPtr [BM, BD] "k_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "k_tile_ptr") [BM, (0:Nat)]),
    Stmt.assign .blockPtr [BM, BD] "v_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "v_tile_ptr") [BM, (0:Nat)]),
    Stmt.store .fp16 [BM, BD] (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "dv_tile_ptr") [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] "dv")) MaskOpt.none,
    Stmt.store .fp16 [BM, BD] (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "dk_tile_ptr") [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] "dk")) MaskOpt.none,
    Stmt.assign .blockPtr [BM, BD] "dv_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "dv_tile_ptr") [BM, (0:Nat)]),
    Stmt.assign .blockPtr [BM, BD] "dk_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "dk_tile_ptr") [BM, (0:Nat)]) ]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General body split.** The general (contiguous-stride) backward kernel lowers
to `bwdPreLoopG ++ [forRange start_n 0 nb 1 bwdOuterBodyG]` on the new surface.
`stride_qm = stride_kn = stride_vk = BD`, `stride_qk = stride_kk = stride_vn = 1`;
`stride_qz`/`stride_qh` carry the per-batch/head pointer offset. The Python test
shape is `BM = 128`, `BD = 64`, `nb = 1`, `D0 = 1024`,
`N_CTX = 128`, `H = 4`, `stride_qz = 32768`, `stride_qh = 8192`. -/
theorem bwd_body_splitG (Q K V Out DO DQ DK DV L M Delta : RegionName) (sc : ℝ)
    (stride_qz stride_qh Z H N_CTX D0 nb BM BD : Nat) :
    (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
      stride_qz stride_qh BD 1 stride_qz stride_qh BD 1 stride_qz stride_qh BD 1
      Z H N_CTX D0 nb BM BD BM).toAlgKernel.body
      = bwdPreLoopG K V DK DV stride_qz stride_qh D0 BM BD H
        ++ [Stmt.forRange "start_n" 0 nb 1
              (bwdOuterBodyG Q DO DQ Delta M sc stride_qz stride_qh D0 BM BD nb N_CTX)] := by
  unfold triton_attention_bwd_kernel bwdPreLoopG bwdOuterBodyG bwdInnerBodyG
    bwdMkPtrG bwdMkPtrLoG
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

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General PreLoop execution** of `_bwd_kernel` (new surface). Steps the
deterministic preLoop, exposing `off_hz = pids 0`, `off_z`/`off_h`, the
`stride_qz_2d`/`stride_qh_2d` scalars, and the four k/v/dk/dv block pointers at
`bwdPtrTileG _ (bwdKBase s) D0 BM BD 0`. Specialized to `H = 4`,
`stride_qz = 32768`, `stride_qh = 8192` (so the index scalars match the pinned
`bwdKBase`); `hbase` carries the row-offset arithmetic. -/
theorem bwdPreLoopG_eval (s : BlockState) (K V DK DV : RegionName) (D0 BM BD : Nat)
    (hbase : (s.pids 0 / 4) * (32768 / BD) + (s.pids 0 % 4) * (8192 / BD) = bwdKBase s / BD)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (bwdPreLoopG K V DK DV 32768 8192 D0 BM BD 4) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4))
      ∧ s0.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4))
      ∧ s0.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD))
      ∧ s0.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD))
      ∧ s0.regs .blockPtr [BM, BD] "k_tile_ptr" = some (bwdPtrTileG K (bwdKBase s) D0 BM BD 0)
      ∧ s0.regs .blockPtr [BM, BD] "v_tile_ptr" = some (bwdPtrTileG V (bwdKBase s) D0 BM BD 0)
      ∧ s0.regs .blockPtr [BM, BD] "dk_tile_ptr" = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD 0)
      ∧ s0.regs .blockPtr [BM, BD] "dv_tile_ptr" = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD 0) := by
  unfold bwdPreLoopG bwdMkPtrG
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
  -- stride_qz_2d = 32768//BD//1 = 32768/BD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 32768) (Op.constNat BD)) (Op.constNat 1)) _
        = some (Tile.scalar (32768 / BD)) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      simp [Nat.div_one]))]
  -- stride_qh_2d = 8192//BD//1 = 8192/BD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat 8192) (Op.constNat BD)) (Op.constNat 1)) _
        = some (Tile.scalar (8192 / BD)) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      simp [Nat.div_one]))]
  -- 4 block pointers
  have hmk : ∀ (R : RegionName) (t : BlockState),
      t.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)) →
      t.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)) →
      t.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD)) →
      t.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD)) →
      evalOp (Op.makeBlockPtrDynOffsets R (Op.constNat 0) [D0, BD] [BM, BD] [BD, 1]
          [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.ref .nat [] "stride_qz_2d"))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.ref .nat [] "stride_qh_2d")),
            Op.constNat 0]) t
        = some (bwdPtrTileG R (bwdKBase s) D0 BM BD 0) := by
    intro R t hz hh hqz hqh
    rw [makeBlockPtr2_eval]
    simp only [evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, hz, hh, hqz, hqh,
      Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
      Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, NumericDType.mul]
    refine congrArg some ?_; ext idx
    simp only [bwdPtrTileG, BlockPtr.mk.injEq, List.cons.injEq, and_true, true_and, Nat.add_zero]
    rw [hbase]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk K _ (by simp [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_same]) (by simp [BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk V _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk DK _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (hmk DV _ (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]

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
needs the separately stored `M`: `log L + M = log(Σ exp(score))`.) The in-loop `l_rcp = 1/l_curr`
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
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [d, (0:Nat)]) s
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
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "k_tile_ptr") [(128:Nat), (0:Nat)]),
    Stmt.assign .blockPtr [128, 64] "v_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "v_tile_ptr") [(128:Nat), (0:Nat)]) ]

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

/-! ### ════════ General (symbolic-dimension) forward exec AST ════════

Dimension-general counterparts of `taPreLoop`/`taLoopBody`/`taPostLoop`/
`taKVPtrTile`, parameterized over `D0` (the 2D parent rows), `N_CTX`, `BLOCK_M`,
`BLOCK_N`, `BLOCK_DMODEL` (= `D_HEAD`) with contiguous strides
(`stride_qm = stride_kn = stride_vk = stride_om = BLOCK_DMODEL`,
`stride_qk = stride_kk = stride_vn = stride_on = 1`). They are the exact
parametric lowering of `triton_attention_fwd_kernel`, with the test-shape
numerals replaced by the corresponding dimension parameters. -/

/-- General K/V block-pointer tile at row offset `rowOff`. -/
def taKVPtrTileG (R : RegionName) (D0 BLOCK_N BLOCK_DMODEL rowOff : Nat) :
    Tile .blockPtr [BLOCK_N, BLOCK_DMODEL] :=
  ⟨fun _ : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
    { region := R, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL],
      blockShape := [BLOCK_N, BLOCK_DMODEL], strides := [BLOCK_DMODEL, 1],
      offsets := [rowOff, 0] }⟩

/-- General preLoop (13 deterministic prefix statements). -/
def taPreLoopG (Q K V Out : RegionName) (sc : ℝ)
    (stride_qh D0 BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .real [BLOCK_M] "m_prev"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_prev" (Op.full [BLOCK_M] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)),
    Stmt.assign .nat [] "stride_qh_2d"
      (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat stride_qh) (Op.constNat BLOCK_DMODEL)) (Op.constNat 1)),
    Stmt.assign .blockPtr [BLOCK_M, BLOCK_DMODEL] "q_tile_ptr"
      (Op.makeBlockPtrDynOffsets Q (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
        [Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)), Op.constNat 0]),
    Stmt.assign .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr"
      (Op.makeBlockPtrDynOffsets K (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
          Op.constNat 0]),
    Stmt.assign .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr"
      (Op.makeBlockPtrDynOffsets V (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
          Op.constNat 0]),
    Stmt.assign .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr"
      (Op.makeBlockPtrDynOffsets Out (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
        [Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)), Op.constNat 0]),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_DMODEL] "q_tile_ptr") []) MaskOpt.none) ]

/-- General loop body (19 statements). -/
def taLoopBodyG (sc : ℝ) (BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk" (Op.full [BLOCK_M, BLOCK_N] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "q")
          (Op.transpose (batch := []) (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "k")))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.const sc)),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.broadcast Op.negInf [BLOCK_M, BLOCK_N])),
    Stmt.assign .real [BLOCK_M] "m_curr"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))
          (Op.ref .real [BLOCK_M] "m_prev"))
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))
        (Op.ref .real [BLOCK_M] "m_prev")),
    Stmt.assign .real [BLOCK_M] "l_prev"
      (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BLOCK_M] "l_prev")
        (Op.exp (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "m_prev") (Op.ref .real [BLOCK_M] "m_curr")))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.exp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_curr")))),
    Stmt.assign .real [BLOCK_M] "l_curr"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "p"))
        (Op.ref .real [BLOCK_M] "l_prev")),
    Stmt.assign .real [BLOCK_M] "l_rcp"
      (Op.div .real Broadcast.scalarL (Op.const (1.0 : ℝ)) (Op.ref .real [BLOCK_M] "l_curr")),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_rcp"))),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.expandDim ⟨1, by simp⟩
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [BLOCK_M] "l_prev") (Op.ref .real [BLOCK_M] "l_rcp")))),
    Stmt.assign .fp16 [BLOCK_M, BLOCK_N] "p"
      (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr") [0, 1]) MaskOpt.none),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BLOCK_M, BLOCK_N] "p"))
          (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v"))),
    Stmt.assign .real [BLOCK_M] "l_prev" (Op.ref .real [BLOCK_M] "l_curr"),
    Stmt.assign .real [BLOCK_M] "m_prev" (Op.ref .real [BLOCK_M] "m_curr"),
    Stmt.assign .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr") [BLOCK_N, (0:Nat)]),
    Stmt.assign .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr") [BLOCK_N, (0:Nat)]) ]

/-- General postLoop (8 statements). -/
def taPostLoopG (L M Out : RegionName) (N_CTX BLOCK_M BLOCK_DMODEL : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
    Stmt.assign .ptr [BLOCK_M] "l_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase L)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))
          (Op.ref .nat [BLOCK_M] "offs_m"))),
    Stmt.assign .ptr [BLOCK_M] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))
          (Op.ref .nat [BLOCK_M] "offs_m"))),
    Stmt.store .real [BLOCK_M] (MemAccess.ptr (Op.ref .ptr [BLOCK_M] "l_ptrs"))
      (Op.ref .real [BLOCK_M] "l_prev") MaskOpt.none,
    Stmt.store .real [BLOCK_M] (MemAccess.ptr (Op.ref .ptr [BLOCK_M] "m_ptrs"))
      (Op.ref .real [BLOCK_M] "m_prev") MaskOpt.none,
    Stmt.assign .fp16 [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")),
    Stmt.store .fp16 [BLOCK_M, BLOCK_DMODEL] (MemAccess.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr") [0, 1])
      (Op.ref .fp16 [BLOCK_M, BLOCK_DMODEL] "acc") MaskOpt.none ]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General `q = tl.load(q_tile_ptr)`** → `fwdQTileG` cells (no boundary check). -/
theorem ta_load_q_evalG
    (Q : RegionName) (off_hz start_m stride_hz_2d D0 BLOCK_M BLOCK_DMODEL : Nat)
    (ptrOp : Op .blockPtr [BLOCK_M, BLOCK_DMODEL]) (s : BlockState)
    (hpids0 : s.pids 0 = start_m) (hpids1 : s.pids 1 = off_hz)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        { region := Q, baseOffset := 0,
          parentShape := [D0, BLOCK_DMODEL], blockShape := [BLOCK_M, BLOCK_DMODEL], strides := [BLOCK_DMODEL, 1],
          offsets := [off_hz * stride_hz_2d + start_m * BLOCK_M, 0] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          some (fwdQTileG s Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL idx)⟩ := by
  rw [load_blockPtr_Q_eval Q 0 D0 BLOCK_DMODEL BLOCK_M BLOCK_DMODEL
    BLOCK_DMODEL 1 (off_hz * stride_hz_2d + start_m * BLOCK_M) ptrOp s hp]
  refine congrArg some ?_
  ext idx
  simp only [fwdQTileG, hpids0, hpids1]
  congr 2
  ring

set_option maxRecDepth 8000 in
/-- **General body split.** The general (contiguous-stride) forward kernel lowers
to `taPreLoopG ++ forRangeDyn :: taPostLoopG`. The dynamic causal loop bound is
`(start_m + 1) · BLOCK_M`, step `BLOCK_N`. `Z`/`H` are arbitrary (unused in the
body); `stride_qz`/`stride_qh` carry the per-batch/head offset (unused by the
2D `make_block_ptr` row offset apart from `stride_qh_2d`). -/
theorem ta_body_splitG (Q K V L M Out : RegionName) (sc : ℝ)
    (stride_qz stride_qh Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    (triton_attention_fwd_kernel Q K V L M Out sc
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgKernel.body
      = taPreLoopG Q K V Out sc stride_qh D0 BLOCK_M BLOCK_N BLOCK_DMODEL
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BLOCK_M))
              (Op.constNat BLOCK_N) (taLoopBodyG sc BLOCK_M BLOCK_N BLOCK_DMODEL)
            :: taPostLoopG L M Out N_CTX BLOCK_M BLOCK_DMODEL) := by
  rfl

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

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General loop-body execution chain.** Symbolic-dimension mirror of
`taLoopBody_steps`, parametric in the block's `start_n = SN` and the loaded K/V
tile values (`kTfn`/`vTfn`, supplied via the `k`/`v` block-ptr load hypotheses
`hkload`/`hvload` so the caller can instantiate them at the global key offset). The
19 statements step `sin` to a final state exposing the `c+1` running registers. -/
theorem taLoopBody_stepsG (sc : ℝ) (BLOCK_M BLOCK_N BLOCK_DMODEL SN : Nat) (sin : BlockState)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N)
    (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (mtile ltile : Tile .real [BLOCK_M])
    (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (K V : RegionName) (D0 rowOff : Nat)
    (kTfn vTfn : TileIndex [BLOCK_N, BLOCK_DMODEL] → ℝ)
    (gm : Fin BLOCK_M → Nat)
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffm : sin.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec gm))
    (hoffn : sin.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hq : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile)
    (hmp : sin.regs .real [BLOCK_M] "m_prev" = some mtile)
    (hlp : sin.regs .real [BLOCK_M] "l_prev" = some ltile)
    (hacc : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (hkp : sin.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr" = some (taKVPtrTileG K D0 BLOCK_N BLOCK_DMODEL rowOff))
    (hvp : sin.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr" = some (taKVPtrTileG V D0 BLOCK_N BLOCK_DMODEL rowOff))
    (hrow : ∀ i : Fin BLOCK_N, rowOff + i.val < D0)
    (hkload : ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
        kTfn idx = sin.readMem K ((rowOff + idx.1.val) * BLOCK_DMODEL + idx.2.1.val))
    (hvload : ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
        vTfn idx = sin.readMem V ((rowOff + idx.1.val) * BLOCK_DMODEL + idx.2.1.val))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ (rmaxT : Tile .real [BLOCK_M]) (qk0 qk1 : Tile .real [BLOCK_M, BLOCK_N]),
      Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qk1 = some rmaxT ∧
      qk0 = Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
          (Tile.dot [] qtile (Tile.transpose [] (⟨fun idx => some (kTfn idx)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
        (Tile.scalar (some sc : WithBot ℝ)) ∧
      qk1 = (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        if SN + idx.2.1.val ≤ gm idx.1 then qk0.data idx else (⊥ : WithBot ℝ)⟩) ∧
      ∃ sF, stepStmts (taLoopBodyG sc BLOCK_M BLOCK_N BLOCK_DMODEL) sin = some sF
        ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
        ∧ ∃ (mcurrT lcurrT lrcpT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
              (acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL]),
            mcurrT = Tile.select
                (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) rmaxT mtile)
                rmaxT mtile
            ∧ pT = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                (Tile.uop WithBot.realExp
                  (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                    qk1 (Tile.expandDim ⟨1, by simp⟩ mcurrT)))
                (Tile.expandDim ⟨1, by simp⟩ lrcpT)
            ∧ lcurrT = Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
                (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                      qk1 (Tile.expandDim ⟨1, by simp⟩ mcurrT))))
                (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mcurrT)))
            ∧ lrcpT = Tile.bop NumericDType.real.div Broadcast.scalarL
                (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT
            ∧ acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                acctile
                (Tile.expandDim ⟨1, by simp⟩
                  (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
                    (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile
                      (Tile.uop WithBot.realExp
                        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mcurrT)))
                    lrcpT))
            ∧ sF.regs .real [BLOCK_M] "m_prev" = some mcurrT
            ∧ sF.regs .real [BLOCK_M] "l_prev" = some lcurrT
            ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some
                (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  acc1T (Tile.dot [] pT (⟨fun idx => some (vTfn idx)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL])))
            ∧ sF.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr" = some (taKVPtrTileG K D0 BLOCK_N BLOCK_DMODEL (rowOff + BLOCK_N))
            ∧ sF.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr" = some (taKVPtrTileG V D0 BLOCK_N BLOCK_DMODEL (rowOff + BLOCK_N))
            ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = sin.regs .real [BLOCK_M, BLOCK_DMODEL] "q"
            ∧ sF.regs .nat [BLOCK_M] "offs_m" = sin.regs .nat [BLOCK_M] "offs_m"
            ∧ sF.regs .nat [BLOCK_N] "offs_n" = sin.regs .nat [BLOCK_N] "offs_n"
            ∧ sF.regs .nat [] "off_hz" = sin.regs .nat [] "off_hz"
            ∧ sF.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr" = sin.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr" := by
  set kT : Tile .real [BLOCK_N, BLOCK_DMODEL] := ⟨fun idx => some (kTfn idx)⟩ with hkT
  set vT : Tile .real [BLOCK_N, BLOCK_DMODEL] := ⟨fun idx => some (vTfn idx)⟩ with hvT
  set qkdot : Tile .real [BLOCK_M, BLOCK_N] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
      (Tile.dot [] qtile (Tile.transpose [] kT)) with hqkdot
  set qk0 : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul Broadcast.scalarR
    qkdot (Tile.scalar (some sc : WithBot ℝ)) with hqk0
  set qk1 : Tile .real [BLOCK_M, BLOCK_N] := (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
    if SN + idx.2.1.val ≤ gm idx.1 then qk0.data idx else (⊥ : WithBot ℝ)⟩) with hqk1
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qk1 = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]⟩
  set mcurrT : Tile .real [BLOCK_M] := Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) rmaxT mtile) rmaxT mtile
    with hmc
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mcurrT) with hal
  set lprev1T : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
      ltile alphaT with hlp1
  set pexpT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        qk1 (Tile.expandDim ⟨1, by simp⟩ mcurrT)) with hpexp
  set lcurrT : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pexpT) lprev1T with hlc
  set lrcpT : Tile .real [BLOCK_M] := Tile.bop NumericDType.real.div Broadcast.scalarL
      (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT with hlr
  set pT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      pexpT (Tile.expandDim ⟨1, by simp⟩ lrcpT) with hpT
  set acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      acctile (Tile.expandDim ⟨1, by simp⟩
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) lprev1T lrcpT)) with hacc1
  refine ⟨rmaxT, qk0, qk1, hrm, rfl, rfl, ?_⟩
  -- execution chain
  unfold taLoopBodyG
  -- L1: k = load(k_tile_ptr [0,1]); discharge via the block-ptr-bc recipe against
  -- the entry register `hkp`, then per-cell `hkload` (entry-state readMem).
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr") [0, 1]) .none) sin
        = some kT from by
      rw [ta_load_blockPtr_bc_eval K 0 D0 BLOCK_DMODEL BLOCK_N BLOCK_DMODEL BLOCK_DMODEL 1 rowOff
        (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr") sin (by rw [evalOp_ref, hkp]; rfl)]
      refine congrArg some ?_; ext idx
      have hj : idx.2.1.val < BLOCK_DMODEL := idx.2.1.isLt
      have hi : rowOff + idx.1.val < D0 := hrow idx.1
      simp only [hi, hj, and_self, if_true, hkT, hkload idx, Nat.zero_add, Nat.mul_one]))]
  -- L2: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ta_qkzeros_eval _ BLOCK_M BLOCK_N))]
  -- L3: qk += dot(q, trans k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_qk_dot_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL (⟨fun _ => some (0:ℝ)⟩) qtile kT
      (by simp) (by simp [hq]) (by simp [hkT])))]
  -- L4: qk *= sc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_qk_scale_eval _ BLOCK_M BLOCK_N sc qkdot (by simp [hqkdot])))]
  -- L5: causal where
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_where_eval _ BLOCK_M BLOCK_N SN gm qk0 (by simp [hoffm]) (by simp [hoffn]) (by simp [hsn]) (by simp [hqk0])))]
  -- L6: m_curr = maximum
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_mij_eval _ BLOCK_M BLOCK_N mtile qk1 rmaxT (by simp [hmp]) (by simp [hqk1]) hrm))]
  -- L7: l_prev *= alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_alpha_eval _ BLOCK_M ltile mtile mcurrT (by simp [hlp]) (by simp [hmp]) (by simp [hmc])))]
  -- L8: p = exp(qk - m_curr)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_p_eval _ BLOCK_M BLOCK_N Nat.one_lt_two qk1 mcurrT (by simp [hqk1]) (by simp [hmc])))]
  -- L9: l_curr = sum p + l_prev
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_lij_eval _ BLOCK_M BLOCK_N pexpT lprev1T (by simp [hpexp]) (by simp [hlp1, hal])))]
  -- L10: l_rcp = 1/l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_lrcp_eval _ BLOCK_M lcurrT (by simp [hlc])))]
  -- L11: p *= l_rcp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_p_rcp_eval _ BLOCK_M BLOCK_N Nat.one_lt_two pexpT lrcpT (by simp [hpexp]) (by simp [hlr])))]
  -- L12: acc *= (l_prev * l_rcp)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_acc_rescale_eval _ BLOCK_M BLOCK_DMODEL Nat.one_lt_two acctile lprev1T lrcpT (by simp [hacc]) (by simp [hlp1, hal]) (by simp [hlr])))]
  -- L13: p -> fp16
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .fp16 [BLOCK_M, BLOCK_N] "p"
    (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")) _
    (⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ : Tile .fp16 [BLOCK_M, BLOCK_N])
    (ta_pfp16_eval _ BLOCK_M BLOCK_N pT (by simp [hpT])))]
  -- L14: v = load(v_tile_ptr [0,1]); the load reads `v_tile_ptr` (untouched by
  -- L1-L13) and the setReg tower preserves `sin`'s mem, so per-cell `hvload` applies.
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr") [0, 1]) MaskOpt.none) _
        = some vT from by
      rw [ta_load_blockPtr_bc_eval V 0 D0 BLOCK_DMODEL BLOCK_N BLOCK_DMODEL BLOCK_DMODEL 1 rowOff
        (Op.ref .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr") _
        (by rw [evalOp_ref]
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            rw [hvp]; rfl)]
      refine congrArg some ?_; ext idx
      have hj : idx.2.1.val < BLOCK_DMODEL := idx.2.1.isLt
      have hi : rowOff + idx.1.val < D0 := hrow idx.1
      simp only [hi, hj, and_self, if_true, BlockState.setReg_readMem, hvT, hvload idx,
        Nat.zero_add, Nat.mul_one]))]
  -- L15: acc += dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_acc_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL acc1T pT vT (by simp [hacc1]) (by simp [hpT]) (by simp [hvT])))]
  -- L16: l_prev = l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_reg_carry_eval _ BLOCK_M "l_curr" lcurrT (by simp [hlc])))]
  -- L17: m_prev = m_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_reg_carry_eval _ BLOCK_M "m_curr" mcurrT (by simp [hmc])))]
  -- L18: k_tile_ptr advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_advance_row_eval _ K 0 D0 BLOCK_DMODEL BLOCK_N BLOCK_DMODEL BLOCK_DMODEL 1 rowOff BLOCK_N "k_tile_ptr"
      (by simp [hkp, taKVPtrTileG])))]
  -- L19: v_tile_ptr advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_advance_row_eval _ V 0 D0 BLOCK_DMODEL BLOCK_N BLOCK_DMODEL BLOCK_DMODEL 1 rowOff BLOCK_N "v_tile_ptr"
      (by simp [hvp, taKVPtrTileG])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, mcurrT, lcurrT, lrcpT, pT, acc1T, rfl, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp [hmc]
  · simp [hlc]
  · simp [hacc1, hpT, hvT]
  · simp [taKVPtrTileG]
  · simp [taKVPtrTileG]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General PreLoop execution.** Symbolic-dimension mirror of `taPreLoop_eval`.
The 13 deterministic statements step a clean state to loop entry, exposing the
running-state inits, the loaded `q = fwdQTileG`, the K/V/Out block pointers at row
offset `off_hz·stride_hz_2d` (+`start_m·BLOCK_M` for Q/Out), and `off_hz`, where
`stride_hz_2d = stride_qh / BLOCK_DMODEL`. -/
theorem taPreLoop_evalG (s : BlockState) (Q K V Out : RegionName) (sc : ℝ)
    (stride_qh D0 BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (taPreLoopG Q K V Out sc stride_qh D0 BLOCK_M BLOCK_N BLOCK_DMODEL) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s0.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s0.regs .real [BLOCK_M] "m_prev" = some ⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M] "l_prev" = some ⟨fun _ : TileIndex [BLOCK_M] => some (0 : ℝ)⟩
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0 : ℝ)⟩
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          some (fwdQTileG s Q (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL BLOCK_M BLOCK_DMODEL idx)⟩
      ∧ s0.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr" = some
          (taKVPtrTileG K D0 BLOCK_N BLOCK_DMODEL (s.pids 1 * (stride_qh / BLOCK_DMODEL)))
      ∧ s0.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr" = some
          (taKVPtrTileG V D0 BLOCK_N BLOCK_DMODEL (s.pids 1 * (stride_qh / BLOCK_DMODEL)))
      ∧ s0.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            { region := Out, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL],
              blockShape := [BLOCK_M, BLOCK_DMODEL], strides := [BLOCK_DMODEL, 1],
              offsets := [s.pids 1 * (stride_qh / BLOCK_DMODEL) + s.pids 0 * BLOCK_M, 0] }⟩) := by
  unfold taPreLoopG
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: off_hz = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: offs_m = start_m*BLOCK_M + arange BLOCK_M
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)) _
        = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 3: offs_n = arange BLOCK_N
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BLOCK_N _))]
  -- stmt 4: m_prev = zeros + (-inf) = ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, WithBot.realAdd, Option.map₂_none_right]
      rfl))]
  -- stmt 5: l_prev = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  -- stmt 6: acc = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  -- stmt 7: stride_qh_2d = stride_qh // BLOCK_DMODEL // 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil
        (Op.floorDiv .nat Broadcast.nil (Op.constNat stride_qh) (Op.constNat BLOCK_DMODEL)) (Op.constNat 1)) _
        = some (Tile.scalar (stride_qh / BLOCK_DMODEL)) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex]
      simp [Nat.div_one]))]
  -- stmt 8: q_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets Q (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
          [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)), Op.constNat 0]) _
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            { region := Q, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL],
              blockShape := [BLOCK_M, BLOCK_DMODEL], strides := [BLOCK_DMODEL, 1],
              offsets := [s.pids 1 * (stride_qh / BLOCK_DMODEL) + s.pids 0 * BLOCK_M, 0] }⟩) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 9: k_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets K (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
            Op.constNat 0]) _
        = some (taKVPtrTileG K D0 BLOCK_N BLOCK_DMODEL (s.pids 1 * (stride_qh / BLOCK_DMODEL))) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 10: v_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets V (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"),
            Op.constNat 0]) _
        = some (taKVPtrTileG V D0 BLOCK_N BLOCK_DMODEL (s.pids 1 * (stride_qh / BLOCK_DMODEL))) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 11: out_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets Out (Op.constNat 0) [D0, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL] [BLOCK_DMODEL, 1]
          [Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.ref .nat [] "stride_qh_2d"))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)), Op.constNat 0]) _
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            { region := Out, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL],
              blockShape := [BLOCK_M, BLOCK_DMODEL], strides := [BLOCK_DMODEL, 1],
              offsets := [s.pids 1 * (stride_qh / BLOCK_DMODEL) + s.pids 0 * BLOCK_M, 0] }⟩) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
        Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 12: q = load(q_tile_ptr)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ta_load_q_evalG Q (s.pids 1) (s.pids 0) (stride_qh / BLOCK_DMODEL) D0 BLOCK_M BLOCK_DMODEL
      (Op.ref .blockPtr [BLOCK_M, BLOCK_DMODEL] "q_tile_ptr") _
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
  · rw [BlockState.setReg_same]
    refine congrArg some ?_; ext idx
    simp only [fwdQTileG, BlockState.setReg_readMem, BlockState.setReg_pids]
  · simp
  · simp
  · simp

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

/-! ## ════════ General forward invariant + step + postLoop + exec ════════

Dimension-general mirror of the pinned `taInvariant`/`ta_*`/`ta_exec` stack above.
The single test-shape block (`numKVBlocks = 1`) is replaced by an arbitrary number
of streaming KV blocks; the running registers after `c = i / BLOCK_N` iterations
bind to the `FA1MathCausal` causal accumulators
(`mPartial`/`lPartial`/`oPartial` at block count `c`, over the loaded span
`SEQ = BLOCK_N · numKVBlocks`). -/

open VeriTile.Examples.FA1MathCausal in
/-- General `q·kᵀ` dot cell (symbolic dims). -/
theorem ta_dot_score_cellG (BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT : TileIndex [BLOCK_N, BLOCK_DMODEL] → ℝ)
    (r : Fin BLOCK_M) (j : Fin BLOCK_N) :
    (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
        (Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))).data
        (r, j, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (r, e, PUnit.unit) * kT (j, e, PUnit.unit))) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BLOCK_DMODEL) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·)
          ((⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL]).data (r, e, PUnit.unit))
          ((Tile.transpose [] (⟨fun idx => some (kT idx)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL])).data (e, j, PUnit.unit))))
      = @Finset.sum (Fin BLOCK_DMODEL) (WithBot ℝ) _ Finset.univ
          (fun e => (some (qT (r, e, PUnit.unit) * kT (j, e, PUnit.unit)) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by
        rw [Tile.transpose_nil_data]; rfl)]
  rw [WithBot.sum_someTerm_eq_some]

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- General masked qk cell `qk1(r,j) = maskedScore qS qT kT sc r (blockIndex c jL)`.
The kernel's causal predicate `SN + j ≤ qS + r` (with `SN = c·BLOCK_N`) and the
in-range key index `blockIndex BLOCK_N numKVBlocks c _ j = c·BLOCK_N + j` match the
`FA1MathCausal.maskedScore` causal `≤` condition exactly. -/
theorem ta_qk1_cellG (BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ)
    (kT : TileIndex [BLOCK_N * numKVBlocks, BLOCK_DMODEL] → ℝ) (sc : ℝ) (qS : Nat)
    (c : Nat) (hc : c < numKVBlocks) (r : Fin BLOCK_M) (j : Fin BLOCK_N) :
    (if c * BLOCK_N + j.val ≤ qS + r.val then
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
            (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
              (Tile.transpose [] (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
                some (kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
          (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
      else (⊥ : WithBot ℝ))
      = maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) j) := by
  set jg := StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) j with hjg
  have hjgval : jg.val = c * BLOCK_N + j.val := rfl
  by_cases h : c * BLOCK_N + j.val ≤ qS + r.val
  · rw [if_pos h]
    rw [Tile.bop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarR, Tile.scalar_data,
      NumericDType.mul, NumericDType.add]
    rw [ta_dot_score_cellG BLOCK_M BLOCK_N BLOCK_DMODEL qT
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit)) r j]
    rw [maskedScore_of_le qS qT kT sc r jg (by rw [hjgval]; exact h)]
    simp only [StreamingAccumulator.scaledScore, WithBot.realAdd, WithBot.realMul,
      Option.map₂, Option.bind, Option.map]
    refine congrArg some ?_; ring
  · rw [if_neg h, maskedScore_of_not_le qS qT kT sc r jg (by rw [hjgval]; exact h)]

open VeriTile.Examples.FA1MathCausal in
/-- General `pexp` cell `exp(qk1(r,j) − Mr)` (with `Mr = mPartial (c+1)`). -/
theorem ta_pexp_cellG (BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ)
    (kT : TileIndex [BLOCK_N * numKVBlocks, BLOCK_DMODEL] → ℝ) (sc : ℝ) (qS : Nat)
    (c : Nat) (hc : c < numKVBlocks) (mNew : Fin BLOCK_M → WithBot ℝ) (r : Fin BLOCK_M) (j : Fin BLOCK_N) :
    WithBot.realExp (WithBot.realSub
        (if c * BLOCK_N + j.val ≤ qS + r.val then
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
                (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
                  (Tile.transpose [] (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
                    some (kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
              (Tile.scalar (some sc : WithBot ℝ))).data (r, j, PUnit.unit)
          else (⊥ : WithBot ℝ))
        (mNew r))
      = some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) j))
          (mNew r))).unbotD 0) := by
  rw [ta_qk1_cellG BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks qT kT sc qS c hc r j]
  exact realExp_eq_some_unbotD _

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- General masked exp block-sum cell `Σ_j pexp(r,j)`. -/
theorem ta_pexp_block_sumG (BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ)
    (kT : TileIndex [BLOCK_N * numKVBlocks, BLOCK_DMODEL] → ℝ) (sc : ℝ) (qS : Nat)
    (c : Nat) (hc : c < numKVBlocks) (mNew : Fin BLOCK_M → WithBot ℝ) (r : Fin BLOCK_M) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
        (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
              if c * BLOCK_N + idx.2.1.val ≤ qS + idx.1.val then
                (Tile.bop NumericDType.real.mul Broadcast.scalarR
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
                    (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
                      (Tile.transpose [] (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
                        some (kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
                  (Tile.scalar (some sc : WithBot ℝ))).data idx
              else (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
            (Tile.expandDim ⟨1, by simp⟩
              (⟨fun r : TileIndex [BLOCK_M] => mNew r.1⟩ : Tile .real [BLOCK_M]))))).data
        (r, PUnit.unit)
      = some ((Finset.univ : Finset (Fin BLOCK_N)).sum (fun jLocal =>
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal))
            (mNew r))).unbotD 0)) := by
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ k : Fin (TileShape.axisDim [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)),
      (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
              if c * BLOCK_N + idx.2.1.val ≤ qS + idx.1.val then
                (Tile.bop NumericDType.real.mul Broadcast.scalarR
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
                    (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
                      (Tile.transpose [] (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
                        some (kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
                  (Tile.scalar (some sc : WithBot ℝ))).data idx
              else (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
            (Tile.expandDim ⟨1, by simp⟩
              (⟨fun r : TileIndex [BLOCK_M] => mNew r.1⟩ : Tile .real [BLOCK_M])))).data
          (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) (r, PUnit.unit) k)
        = (some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) k))
            (mNew r))).unbotD 0) : WithBot ℝ) := by
    intro k
    rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
          (r, PUnit.unit) k) = (r, k, PUnit.unit) from rfl]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    exact ta_pexp_cellG BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks qT kT sc qS c hc mNew r k
  rw [Finset.sum_congr rfl (fun k _ => hcell k)]
  rw [WithBot.sum_someTerm_eq_some]
  rfl

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- General `p·v` dot cell. -/
theorem ta_pv_dot_blockG (BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ)
    (kT vT : TileIndex [BLOCK_N * numKVBlocks, BLOCK_DMODEL] → ℝ) (sc : ℝ) (qS : Nat)
    (c : Nat) (hc : c < numKVBlocks) (mNew : Fin BLOCK_M → WithBot ℝ)
    (r : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (lrcpT : Tile .real [BLOCK_M]) (lrcpVal : ℝ)
    (hlrcp : lrcpT.data (r, PUnit.unit) = some lrcpVal) :
    (Tile.dot []
        (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
                if c * BLOCK_N + idx.2.1.val ≤ qS + idx.1.val then
                  (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
                      (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
                        (Tile.transpose [] (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
                          some (kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
                    (Tile.scalar (some sc : WithBot ℝ))).data idx
                else (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
              (Tile.expandDim ⟨1, by simp⟩
                (⟨fun r : TileIndex [BLOCK_M] => mNew r.1⟩ : Tile .real [BLOCK_M]))))
          (Tile.expandDim ⟨1, by simp⟩ lrcpT))
        (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
          some (vT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL])).data (r, d, PUnit.unit)
      = some (lrcpVal * (Finset.univ : Finset (Fin BLOCK_N)).sum (fun jLocal =>
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal))
            (mNew r))).unbotD 0
            * vT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal, d, PUnit.unit))) := by
  rw [Tile.dot_nil_data]
  refine Eq.trans (b := @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ
      (fun j => (some (lrcpVal *
        ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) j))
          (mNew r))).unbotD 0
          * vT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hc) j, d, PUnit.unit))) : WithBot ℝ)))
    (Finset.sum_congr rfl (fun j (_ : j ∈ Finset.univ) => ?_)) ?_
  · rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.mul]
    rw [hlrcp]
    rw [Tile.uop_data, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub]
    rw [ta_pexp_cellG BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks qT kT sc qS c hc mNew r j]
    simp only [WithBot.realMul, Option.map₂_some_some]
    refine congrArg some ?_; ring
  · rw [WithBot.sum_someTerm_eq_some, ← Finset.mul_sum]

open VeriTile.Examples.FA1MathCausal in
/-- **General loop invariant** for the triton forward streaming loop. Counter
`i = c · BLOCK_N` (`c = i / BLOCK_N` blocks processed) over `numKVBlocks` blocks of
`BLOCK_N` keys (span `SEQ = BLOCK_N · numKVBlocks`). Running registers bind to
`FA1MathCausal` accumulators; K/V block pointers sit at row offset
`pids1·stride_hz_2d + i`. -/
noncomputable def taInvariantG
    (Q K V Out : RegionName) (s0 : BlockState) (sc : ℝ)
    (stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  let qS := s0.pids 0 * BLOCK_M
  let qT := fwdQTileG s0 Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL
  let kT := fwdKTileG s0 K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL
  let vT := fwdVTileG s0 V stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL
  s.pids = s0.pids ∧ i % BLOCK_N = 0 ∧ i ≤ BLOCK_N * numKVBlocks ∧
  (s.regs .real [BLOCK_M] "m_prev" = some ⟨fun r : TileIndex [BLOCK_M] =>
      mPartial BLOCK_N qS qT numKVBlocks kT sc (i / BLOCK_N) r.1⟩) ∧
  (s.regs .real [BLOCK_M] "l_prev" = some ⟨fun r : TileIndex [BLOCK_M] =>
      ((lPartial BLOCK_N qS qT numKVBlocks kT sc (i / BLOCK_N) r.1 : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      ((oPartial BLOCK_N qS qT numKVBlocks kT vT sc (i / BLOCK_N) idx
          / lPartial BLOCK_N qS qT numKVBlocks kT sc (i / BLOCK_N) idx.1 : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (qT idx)⟩) ∧
  (s.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => qS + r.val))) ∧
  (s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) ∧
  (s.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr" = some
    (taKVPtrTileG K D0 BLOCK_N BLOCK_DMODEL (s0.pids 1 * stride_hz_2d + i))) ∧
  (s.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr" = some
    (taKVPtrTileG V D0 BLOCK_N BLOCK_DMODEL (s0.pids 1 * stride_hz_2d + i))) ∧
  (s.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr" = some
    (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      { region := Out, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL],
        blockShape := [BLOCK_M, BLOCK_DMODEL], strides := [BLOCK_DMODEL, 1],
        offsets := [s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M, 0] }⟩)) ∧
  (s.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1))) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

open VeriTile.Examples.FA1MathCausal in
/-- **General invariant base case.** At `c = 0` the running registers are
`mPartial 0 = ⊥`, `lPartial 0 = 0`, `oPartial 0 / lPartial 0 = 0 / 0 = 0`. -/
theorem ta_invariant_zeroG
    (Q K V Out : RegionName) (sp : BlockState) (sc : ℝ)
    (stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (hm : sp.regs .real [BLOCK_M] "m_prev" = some ⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩)
    (hl : sp.regs .real [BLOCK_M] "l_prev" = some ⟨fun _ : TileIndex [BLOCK_M] => some (0 : ℝ)⟩)
    (hacc : sp.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0 : ℝ)⟩)
    (hq : sp.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (fwdQTileG sp Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL idx)⟩)
    (hoffm : sp.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => sp.pids 0 * BLOCK_M + r.val)))
    (hoffn : sp.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hkp : sp.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "k_tile_ptr" = some
        (taKVPtrTileG K D0 BLOCK_N BLOCK_DMODEL (sp.pids 1 * stride_hz_2d)))
    (hvp : sp.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "v_tile_ptr" = some
        (taKVPtrTileG V D0 BLOCK_N BLOCK_DMODEL (sp.pids 1 * stride_hz_2d)))
    (hop : sp.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr" = some
        (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          { region := Out, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL],
            blockShape := [BLOCK_M, BLOCK_DMODEL], strides := [BLOCK_DMODEL, 1],
            offsets := [sp.pids 1 * stride_hz_2d + sp.pids 0 * BLOCK_M, 0] }⟩))
    (hoh : sp.regs .nat [] "off_hz" = some (Tile.scalar (sp.pids 1)))
    (hundef : ∀ rg o, sp.undef rg o = 0) :
    taInvariantG Q K V Out sp sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks 0 sp := by
  unfold taInvariantG
  refine ⟨rfl, by simp, by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hm]; refine congrArg some ?_; ext r
    show (⊥ : WithBot ℝ) = mPartial BLOCK_N (sp.pids 0 * BLOCK_M) _ numKVBlocks _ sc (0 / BLOCK_N) r.1
    rw [Nat.zero_div]; rfl
  · rw [hl]; refine congrArg some ?_; ext r
    show some (0 : ℝ) = ((lPartial BLOCK_N (sp.pids 0 * BLOCK_M) _ numKVBlocks _ sc (0 / BLOCK_N) r.1 : ℝ) : WithBot ℝ)
    rw [Nat.zero_div]; rfl
  · rw [hacc]; refine congrArg some ?_; ext idx
    show some (0 : ℝ) = ((oPartial BLOCK_N (sp.pids 0 * BLOCK_M)
        (fwdQTileG sp Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL) numKVBlocks
        (fwdKTileG sp K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        (fwdVTileG sp V stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc (0 / BLOCK_N) idx
        / lPartial BLOCK_N (sp.pids 0 * BLOCK_M) (fwdQTileG sp Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL)
          numKVBlocks (fwdKTileG sp K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc (0 / BLOCK_N) idx.1 : ℝ) : WithBot ℝ)
    rw [Nat.zero_div]
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

open VeriTile.Examples.FA1MathCausal in
/-- If the causal m-free normalizer over the first `k` blocks is zero, then so is
the m-free output accumulator: a zero normalizer forces every causal indicator to
zero, killing each `oFree` term. -/
theorem ta_lFree_zero_imp_oFree_zero {M D Bk N : Nat}
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N) (idx : TileIndex [M, D])
    (h0 : lFree qStart Q K scale k hk idx.1 = 0) :
    oFree qStart Q K V scale k hk idx = 0 := by
  induction k with
  | zero => exact oFree_zero qStart Q K V scale idx
  | succ k ih =>
      have hk' : k ≤ N := Nat.le_of_succ_le hk
      rw [lFree_succ qStart Q K scale k hk idx.1] at h0
      -- both summands are ≥ 0; their sum is 0 ⇒ each is 0
      have hterm_nonneg : ∀ jL : Fin Bk,
          0 ≤ (if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
                Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
                  (StreamingAccumulator.blockIndex Bk N k hk jL))
              else 0) := by
        intro jL; by_cases hv : (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val
        · simp [hv, le_of_lt (Real.exp_pos _)]
        · simp [hv]
      have hblock_nonneg : 0 ≤ Finset.univ.sum (fun jL : Fin Bk =>
            if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
              Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
                (StreamingAccumulator.blockIndex Bk N k hk jL))
            else 0) :=
        Finset.sum_nonneg (fun jL _ => hterm_nonneg jL)
      have hlfree_nonneg : 0 ≤ lFree qStart Q K scale k hk' idx.1 := by
        unfold lFree; exact Finset.sum_nonneg (fun n _ => Finset.sum_nonneg (fun jL _ => by
          by_cases hv : (StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk') jL).val ≤ qStart + idx.1.val
          · simp [hv, le_of_lt (Real.exp_pos _)]
          · simp [hv]))
      have hlf0 : lFree qStart Q K scale k hk' idx.1 = 0 := by linarith
      have hblock0 : Finset.univ.sum (fun jL : Fin Bk =>
            if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
              Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
                (StreamingAccumulator.blockIndex Bk N k hk jL))
            else 0) = 0 := by linarith
      -- each indicator term is zero ⇒ the indicator was false ⇒ oFree block term zero
      have hterm0 : ∀ jL : Fin Bk,
          (if (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val then
            Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
              (StreamingAccumulator.blockIndex Bk N k hk jL))
          else 0) = 0 := fun jL =>
        (Finset.sum_eq_zero_iff_of_nonneg (fun jL' _ => hterm_nonneg jL')).mp hblock0 jL
          (Finset.mem_univ _)
      rw [oFree_succ qStart Q K V scale k hk idx, ih hk' hlf0]
      rw [zero_add]
      refine Finset.sum_eq_zero (fun jL _ => ?_)
      have := hterm0 jL
      by_cases hv : (StreamingAccumulator.blockIndex Bk N k hk jL).val ≤ qStart + idx.1.val
      · exfalso; rw [if_pos hv] at this; exact (Real.exp_ne_zero _) this
      · simp [hv]

open VeriTile.Examples.FA1MathCausal in
/-- **Acc-rescale cancel.** `(oPartial c / lPartial c) · lPartial c = oPartial c`,
valid even at `c = 0` (where `0/0·0 = 0 = oPartial 0`): if `lPartial c = 0` then
`oPartial c = 0` too (via the m-free factorization + the zero-normalizer lemma). -/
theorem ta_oPartial_div_lPartial_cancel {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (c : Nat) (hc : c ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    (oPartial Bk qStart Q numKVBlocks K V scale c idx
        / lPartial Bk qStart Q numKVBlocks K scale c idx.1)
      * lPartial Bk qStart Q numKVBlocks K scale c idx.1
      = oPartial Bk qStart Q numKVBlocks K V scale c idx := by
  by_cases hl : lPartial Bk qStart Q numKVBlocks K scale c idx.1 = 0
  · rw [hl, mul_zero]
    -- lPartial c = 0 ⇒ lFree c = 0 ⇒ oFree c = 0 ⇒ oPartial c = 0
    rw [lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale c hc idx.1] at hl
    have hlFree0 : lFree qStart Q K scale c hc idx.1 = 0 :=
      (mul_eq_zero.mp hl).resolve_left (Real.exp_ne_zero _)
    rw [oPartial_eq_mShifted hBk qStart Q numKVBlocks K V scale c hc idx,
        ta_lFree_zero_imp_oFree_zero qStart Q K V scale c hc idx hlFree0, mul_zero]
  · rw [div_mul_cancel₀ _ hl]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **General loop step.** Advancing the streaming counter from `i = c·BLOCK_N` to
`i + BLOCK_N` (block `c → c+1`), the loop body carries the `FA1MathCausal` causal
running registers from block count `c` to `c+1`. The running max becomes
`mPartial (c+1)`, the normalizer `lPartial (c+1)`, and `acc` the in-loop-`l_rcp`-
normalized ratio `oPartial (c+1) / lPartial (c+1)`. The honest boundary side
condition `pids1·stride_hz_2d + BLOCK_N·numKVBlocks ≤ D0` discharges the K/V load
bounds (faithful under the contiguous layout). -/
theorem ta_attn_stepG (Q K V Out : RegionName) (s0 : BlockState) (sc : ℝ)
    (stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBD : 0 < BLOCK_DMODEL)
    (hbound : s0.pids 1 * stride_hz_2d + BLOCK_N * numKVBlocks ≤ D0)
    (i : Nat) (s : BlockState) (hilt : i < BLOCK_N * numKVBlocks)
    (hinv : taInvariantG Q K V Out s0 sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks i s) :
    ∃ s', stepStmts (taLoopBodyG sc BLOCK_M BLOCK_N BLOCK_DMODEL)
        (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ taInvariantG Q K V Out s0 sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks (i + BLOCK_N) s' := by
  simp only [taInvariantG] at hinv
  obtain ⟨hpids, hmod, hile, hmp, hlp, hacc, hq, hoffm, hoffn, hkp, hvp, hop, hoh, hundef, hmem⟩ := hinv
  -- `i = c · BLOCK_N`
  set c := i / BLOCK_N with hcdef
  have hic : i = c * BLOCK_N := by
    rw [hcdef, Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]
  have hclt : c < numKVBlocks := by
    rw [hcdef]; exact (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm numKVBlocks BLOCK_N]; exact hilt)
  have hcle : c ≤ numKVBlocks := le_of_lt hclt
  have hc1 : (i + BLOCK_N) / BLOCK_N = c + 1 := by
    rw [hcdef, hic, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]
  set qS := s0.pids 0 * BLOCK_M with hqS
  set qT := fwdQTileG s0 Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL with hqT
  set kT := fwdKTileG s0 K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hkT
  set vT := fwdVTileG s0 V stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hvT
  set rowOff := s0.pids 1 * stride_hz_2d + i with hrowOff
  set se := s.setReg "start_n" .nat [] (Tile.scalar i) with hse
  -- per-block loaded tiles (kTfn/vTfn), expressed through `kT`/`vT` at `blockIndex c`
  set kTfn : TileIndex [BLOCK_N, BLOCK_DMODEL] → ℝ :=
    (fun idx => kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit)) with hkTfn
  set vTfn : TileIndex [BLOCK_N, BLOCK_DMODEL] → ℝ :=
    (fun idx => vT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit)) with hvTfn
  have hpids1 : se.pids 1 = s0.pids 1 := by rw [hse, BlockState.setReg_pids, hpids]
  have hpids0 : se.pids 0 = s0.pids 0 := by rw [hse, BlockState.setReg_pids, hpids]
  have hrow : ∀ j : Fin BLOCK_N, rowOff + j.val < D0 := by
    intro j; rw [hrowOff, hic]
    calc s0.pids 1 * stride_hz_2d + c * BLOCK_N + j.val
        < s0.pids 1 * stride_hz_2d + c * BLOCK_N + BLOCK_N := by omega
      _ = s0.pids 1 * stride_hz_2d + (c + 1) * BLOCK_N := by ring
      _ ≤ s0.pids 1 * stride_hz_2d + numKVBlocks * BLOCK_N := by
          have : (c + 1) * BLOCK_N ≤ numKVBlocks * BLOCK_N := Nat.mul_le_mul_right _ hclt
          omega
      _ = s0.pids 1 * stride_hz_2d + BLOCK_N * numKVBlocks := by rw [Nat.mul_comm numKVBlocks BLOCK_N]
      _ ≤ D0 := hbound
  -- readback equalities for the K/V per-cell loads (entry mem = s0 mem)
  have hmem_se : ∀ (R : RegionName) (o : Nat), se.readMem R o = s0.readMem R o := by
    intro R o; rw [hse]; simp only [BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem]
  have hkload : ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
      kTfn idx = se.readMem K ((rowOff + idx.1.val) * BLOCK_DMODEL + idx.2.1.val) := by
    intro idx; rw [hmem_se]
    simp only [hkTfn, hkT, fwdKTileG, hrowOff, hic, StreamingAccumulator.blockIndex]
    congr 1; ring
  have hvload : ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
      vTfn idx = se.readMem V ((rowOff + idx.1.val) * BLOCK_DMODEL + idx.2.1.val) := by
    intro idx; rw [hmem_se]
    simp only [hvTfn, hvT, fwdVTileG, hrowOff, hic, StreamingAccumulator.blockIndex]
    congr 1; ring
  -- entry running registers (after setReg start_n) feed taLoopBody_stepsG
  obtain ⟨rmaxT, qk0, qk1, hrm, hqk0eq, hqk1eq, sF, hchain, hpidsF, hmemF, hundefF,
      mcurrT, lcurrT, lrcpT, pT, acc1T, hmcd, hpTd, hlcd, hlrd, hacc1d,
      hmF, hlF, haccF, hkpF, hvpF, hqF, hoffmF, hoffnF, hohF, hopF⟩ :=
    taLoopBody_stepsG sc BLOCK_M BLOCK_N BLOCK_DMODEL i se hBM hBN
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (qT idx)⟩)
      (⟨fun r : TileIndex [BLOCK_M] => mPartial BLOCK_N qS qT numKVBlocks kT sc c r.1⟩)
      (⟨fun r : TileIndex [BLOCK_M] => ((lPartial BLOCK_N qS qT numKVBlocks kT sc c r.1 : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        ((oPartial BLOCK_N qS qT numKVBlocks kT vT sc c idx
            / lPartial BLOCK_N qS qT numKVBlocks kT sc c idx.1 : ℝ) : WithBot ℝ)⟩)
      K V D0 rowOff kTfn vTfn
      (fun r : Fin BLOCK_M => qS + r.val)
      (by rw [hse, BlockState.setReg_same])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hoffm])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hoffn])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hq])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hmp])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hlp])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hacc])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hkp, hrowOff])
      (by rw [hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hvp, hrowOff])
      hrow hkload hvload
      (by intro rg o; rw [hse, BlockState.setReg_undef]; exact hundef rg o)
  refine ⟨sF, hchain, ?_⟩
  -- qk1 cell = maskedScore at global key blockIndex c
  have hgkey : ∀ (j : Fin BLOCK_N), StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) j
      = (⟨c * BLOCK_N + j.val, by
          have : (c + 1) * BLOCK_N ≤ numKVBlocks * BLOCK_N := Nat.mul_le_mul_right _ hclt
          have hjv : j.val < BLOCK_N := j.isLt
          calc c * BLOCK_N + j.val < c * BLOCK_N + BLOCK_N := by omega
            _ = (c + 1) * BLOCK_N := by ring
            _ ≤ numKVBlocks * BLOCK_N := this
            _ = BLOCK_N * numKVBlocks := Nat.mul_comm _ _⟩ : Fin (BLOCK_N * numKVBlocks)) := by
    intro j; rfl
  have hqk1cell : ∀ (r : Fin BLOCK_M) (j : Fin BLOCK_N),
      qk1.data (r, j, PUnit.unit)
        = maskedScore qS qT kT sc r (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) j) := by
    intro r j
    rw [hqk1eq]
    simp only [hqk0eq, hic, hkTfn]
    have := ta_qk1_cellG BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks qT kT sc qS c hclt r j
    convert this using 3
  -- rmaxT cell = block sup = mPartial (c+1) right factor
  have hmne : ∀ r : Fin BLOCK_M, mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r ≠ ⊥ :=
    fun r => mPartial_succ_ne_bot (Bk := BLOCK_N) hBN qS qT numKVBlocks kT sc c (Nat.succ_le_iff.mpr hclt) r
  have hrmaxCell : ∀ r : Fin BLOCK_M,
      rmaxT.data (r, PUnit.unit)
        = (Finset.univ : Finset (Fin BLOCK_N)).sup
            (fun jLocal => maskedScore qS qT kT sc r
              (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) jLocal)) := by
    intro r
    unfold Tile.reduceMaxDrop at hrm
    rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)] at hrm
    rw [← Option.some.inj hrm]
    show (Finset.univ : Finset (Fin BLOCK_N)).sup' ⟨⟨0, hBN⟩, Finset.mem_univ _⟩
        (fun k => qk1.data ((r, k, PUnit.unit) : TileIndex [BLOCK_M, BLOCK_N])) = _
    rw [Finset.sup'_eq_sup]
    refine Finset.sup_congr rfl (fun j _ => ?_)
    exact hqk1cell r j
  -- mcurrT = mPartial (c+1) tile
  have hmcurrCell : ∀ r : Fin BLOCK_M,
      mcurrT.data (r, PUnit.unit) = mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r := by
    intro r
    rw [hmcd, Tile.select_data, Tile.cop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hrmaxCell r]
    rw [mPartial_succ_of_lt qS qT numKVBlocks kT sc c (Nat.succ_le_iff.mpr hclt) r]
    by_cases hgt : (Finset.univ : Finset (Fin BLOCK_N)).sup
        (fun jLocal => maskedScore qS qT kT sc r
          (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) jLocal))
        > mPartial BLOCK_N qS qT numKVBlocks kT sc c r
    · rw [decide_eq_true hgt]
      simp only [if_true]
      rw [max_eq_right (le_of_lt hgt)]
    · rw [decide_eq_false hgt]
      simp only [Bool.false_eq_true, if_false]
      rw [max_eq_left (not_lt.mp hgt)]
  have hmcurrTile : mcurrT = ⟨fun r : TileIndex [BLOCK_M] => mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r.1⟩ := by
    ext r; exact hmcurrCell r.1
  set mNew : Fin BLOCK_M → WithBot ℝ := fun r => mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r with hmNew
  -- rewrite qk1 to the canonical `c·BLOCK_N` masked if-form (for the block-sum bridges)
  have hqk1canon : qk1 = (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        if c * BLOCK_N + idx.2.1.val ≤ qS + idx.1.val then
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
              (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
                (Tile.transpose [] (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
                  some (kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
            (Tile.scalar (some sc : WithBot ℝ))).data idx
        else (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N]) := by
    rw [hqk1eq]; ext idx; simp only [hqk0eq, hic, hkTfn]
  -- lcurrT cell = some (lPartial (c+1))
  have hlcurrCell : ∀ r : Fin BLOCK_M,
      lcurrT.data (r, PUnit.unit) = some (lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r) := by
    intro r
    rw [hlcd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    rw [hqk1canon, hmcurrTile]
    erw [ta_pexp_block_sumG BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks qT kT sc qS c hclt mNew r]
    -- lprev1 = lPartial c * alpha, with alpha = exp(mPartial c - mPartial (c+1))
    rw [show (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
          (⟨fun r : TileIndex [BLOCK_M] => ((lPartial BLOCK_N qS qT numKVBlocks kT sc c r.1 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M])
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
              (⟨fun r : TileIndex [BLOCK_M] => mPartial BLOCK_N qS qT numKVBlocks kT sc c r.1⟩ : Tile .real [BLOCK_M])
              ⟨fun r : TileIndex [BLOCK_M] => mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r.1⟩))).data (r, PUnit.unit)
        = some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
              (mPartial BLOCK_N qS qT numKVBlocks kT sc c r) (mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r))).unbotD 0
            * lPartial BLOCK_N qS qT numKVBlocks kT sc c r) from by
      rw [Tile.bop_data, Tile.uop_data, Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub]
      rw [realExp_eq_some_unbotD (WithBot.realSub (mPartial BLOCK_N qS qT numKVBlocks kT sc c r)
            (mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r))]
      show WithBot.realMul (some (lPartial BLOCK_N qS qT numKVBlocks kT sc c r)) (some _) = some _
      rw [WithBot.realMul, Option.map₂_some_some]
      refine congrArg some ?_
      rw [mul_comm]
      rfl]
    simp only [NumericDType.add, WithBot.realAdd, Option.map₂_some_some]
    refine congrArg some ?_
    rw [lPartial_succ_of_lt qS qT numKVBlocks kT sc c (Nat.succ_le_iff.mpr hclt) r]
    simp only [hmNew]; ring
  -- lrcp cell = some (1 / lPartial (c+1))
  have hlrcpCell : ∀ r : Fin BLOCK_M,
      lrcpT.data (r, PUnit.unit) = some (1 / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r) := by
    intro r
    rw [hlrd, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarL, Tile.scalar_data,
      NumericDType.div]
    rw [hlcurrCell r]
    show WithBot.realDiv (some (1.0 : ℝ)) (some (lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r))
      = some (1 / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r)
    rw [WithBot.realDiv, Option.map₂_some_some]; norm_num
  -- final acc cell = some (oPartial (c+1) / lPartial (c+1))
  have haccCell : ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT (⟨fun idx => some (vTfn idx)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))).data idx
        = some (oPartial BLOCK_N qS qT numKVBlocks kT vT sc (c + 1) idx
            / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) idx.1) := by
    intro idx
    obtain ⟨r, d, u⟩ := idx; cases u
    rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    -- acc1T cell = some (alpha · oPartial c · (1/lPartial (c+1)))
    set alphaV : ℝ := (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
        (mPartial BLOCK_N qS qT numKVBlocks kT sc c r) (mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r))).unbotD 0 with halphaV
    -- mcurrT cell = mPartial (c+1)
    have hmcurrC : mcurrT.data (r, PUnit.unit) = mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r := hmcurrCell r
    have hacc1V : acc1T.data (r, d, PUnit.unit)
        = some (alphaV * oPartial BLOCK_N qS qT numKVBlocks kT vT sc c (r, d, PUnit.unit)
            * (1 / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r)) := by
      rw [hacc1d, Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub,
        Tile.expandDim_data, TileShape.dropInsertedIndex, Tile.bop_data, Tile.uop_data]
      rw [hlrcpCell r, hmcurrC]
      have hcancel := ta_oPartial_div_lPartial_cancel (Bk := BLOCK_N) hBN qS qT numKVBlocks kT vT sc c hcle (r, d, PUnit.unit)
      show WithBot.realMul (some (oPartial BLOCK_N qS qT numKVBlocks kT vT sc c (r, d, PUnit.unit)
              / lPartial BLOCK_N qS qT numKVBlocks kT sc c r))
            (WithBot.realMul (WithBot.realMul (some (lPartial BLOCK_N qS qT numKVBlocks kT sc c r))
              (WithBot.realExp (WithBot.realSub (mPartial BLOCK_N qS qT numKVBlocks kT sc c r)
                (mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r))))
              (some (1 / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r)))
          = _
      rw [realExp_eq_some_unbotD (WithBot.realSub _ _)]
      simp only [WithBot.realMul, Option.map₂_some_some, Option.map₂_some_coe]
      refine congrArg some ?_
      rw [show (WithBot.realExp (WithBot.realSub (mPartial BLOCK_N qS qT numKVBlocks kT sc c r)
              (mPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r))).unbotD 0 = alphaV from by
        rw [halphaV]; rfl]
      rw [show oPartial BLOCK_N qS qT numKVBlocks kT vT sc c (r, d, PUnit.unit)
              / lPartial BLOCK_N qS qT numKVBlocks kT sc c r
              * (lPartial BLOCK_N qS qT numKVBlocks kT sc c r * alphaV
                  * (1 / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r))
          = (oPartial BLOCK_N qS qT numKVBlocks kT vT sc c (r, d, PUnit.unit)
              / lPartial BLOCK_N qS qT numKVBlocks kT sc c r
              * lPartial BLOCK_N qS qT numKVBlocks kT sc c r)
            * (alphaV * (1 / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r)) from by ring]
      rw [hcancel]; ring
    rw [hacc1V]
    -- dot pT vT cell via ta_pv_dot_blockG
    rw [show pT = (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
                if c * BLOCK_N + idx.2.1.val ≤ qS + idx.1.val then
                  (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
                      (Tile.dot [] (⟨fun idx => some (qT idx)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL])
                        (Tile.transpose [] (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
                          some (kT (StreamingAccumulator.blockIndex BLOCK_N numKVBlocks c (Nat.succ_le_iff.mpr hclt) idx.1, idx.2.1, PUnit.unit))⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))))
                    (Tile.scalar (some sc : WithBot ℝ))).data idx
                else (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N])
              (Tile.expandDim ⟨1, by simp⟩
                (⟨fun r : TileIndex [BLOCK_M] => mNew r.1⟩ : Tile .real [BLOCK_M]))))
          (Tile.expandDim ⟨1, by simp⟩ lrcpT)) from by
      rw [hpTd, hmcurrTile, hqk1canon]]
    erw [ta_pv_dot_blockG BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks qT kT vT sc qS c hclt mNew r d lrcpT
      (1 / lPartial BLOCK_N qS qT numKVBlocks kT sc (c + 1) r) (hlrcpCell r)]
    rw [WithBot.realAdd, Option.map₂_some_some]
    refine congrArg some ?_
    -- oPartial (c+1) = alpha · oPartial c + Σ pexp·v
    rw [oPartial_succ_of_lt qS qT numKVBlocks kT vT sc c (Nat.succ_le_iff.mpr hclt) (r, d, PUnit.unit)]
    simp only [hmNew, halphaV]
    rw [add_div]
    congr 1
    · ring
    · ring
  -- assemble the invariant at i + BLOCK_N (c+1)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hse, BlockState.setReg_pids, hpids]
  · rw [hic, show c * BLOCK_N + BLOCK_N = (c + 1) * BLOCK_N from by ring, Nat.mul_mod_left]
  · rw [hic, show c * BLOCK_N + BLOCK_N = (c + 1) * BLOCK_N from by ring]
    have : (c + 1) * BLOCK_N ≤ numKVBlocks * BLOCK_N := Nat.mul_le_mul_right _ hclt
    rw [Nat.mul_comm numKVBlocks BLOCK_N] at this; exact this
  · rw [hmF, hmcurrTile, hc1]
  · rw [hlF, hc1]; refine congrArg some ?_; ext r; rw [hlcurrCell r.1]; rfl
  · rw [haccF, hc1]; refine congrArg some ?_; ext idx; rw [haccCell idx]; rfl
  · rw [hqF, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq
  · rw [hoffmF, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffm
  · rw [hoffnF, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn
  · rw [hkpF, hrowOff, hic]; congr 2; ring
  · rw [hvpF, hrowOff, hic]; congr 2; ring
  · rw [hopF, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hop
  · rw [hohF, hse, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoh
  · exact hundefF
  · rw [hmemF, hse]; funext region offset; rw [BlockState.setReg_mem]
    exact congrFun (congrFun hmem region) offset

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **General PostLoop execution + genuine readbacks.** Stepping the 8 post-loop
statements from a loop-end state satisfying `taInvariantG … (BLOCK_N·numKVBlocks)`
(full streaming scope, `c = numKVBlocks`), the kernel's `L`/`M`/`Out` stores hold the
genuine streaming values at block count `numKVBlocks`. The boundary side condition
`s0.pids 1·stride_hz_2d + BLOCK_M·(s0.pids 0 + 1) ≤ D0` is faithful (contiguous
layout). -/
theorem ta_postLoopG
    (Q K V L M Out : RegionName) (s0 : BlockState) (sc : ℝ)
    (stride_hz_2d D0 N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat) (s : BlockState)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBD : 0 < BLOCK_DMODEL)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val * 1))
    (hinv : taInvariantG Q K V Out s0 sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks
      (BLOCK_N * numKVBlocks) s) :
    ∃ sP, stepStmts (taPostLoopG L M Out N_CTX BLOCK_M BLOCK_DMODEL) s = some sP
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          active s0 (s0.pids 1 * stride_hz_2d) D0 BLOCK_M idx →
          sP.mem Out (outOffset s0 (s0.pids 1 * stride_hz_2d) BLOCK_DMODEL 1 BLOCK_M idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (oPartial BLOCK_N (s0.pids 0 * BLOCK_M)
                    (fwdQTileG s0 Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL) numKVBlocks
                    (fwdKTileG s0 K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
                    (fwdVTileG s0 V stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc numKVBlocks idx
                  / lPartial BLOCK_N (s0.pids 0 * BLOCK_M)
                    (fwdQTileG s0 Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL) numKVBlocks
                    (fwdKTileG s0 K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc numKVBlocks idx.1))))
      ∧ (∀ i : Fin BLOCK_M,
          sP.readMem L (lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i)
            = lPartial BLOCK_N (s0.pids 0 * BLOCK_M)
                (fwdQTileG s0 Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL) numKVBlocks
                (fwdKTileG s0 K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc numKVBlocks i)
      ∧ (∀ i : Fin BLOCK_M,
          sP.readMem M (lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i)
            = (mPartial BLOCK_N (s0.pids 0 * BLOCK_M)
                (fwdQTileG s0 Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL) numKVBlocks
                (fwdKTileG s0 K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL) sc numKVBlocks i).unbotD 0) := by
  simp only [taInvariantG] at hinv
  obtain ⟨hpids, _, _, hmp, hlp, hacc, hq, hoffm, hoffn, hkp, hvp, hop, hoh, hundef, hmem⟩ := hinv
  set qS := s0.pids 0 * BLOCK_M with hqS
  set qT := fwdQTileG s0 Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL with hqT
  set kT := fwdKTileG s0 K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hkT
  set vT := fwdVTileG s0 V stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hvT
  -- running registers at c = numKVBlocks
  have hcdiv : BLOCK_N * numKVBlocks / BLOCK_N = numKVBlocks := by
    rw [Nat.mul_comm]; exact Nat.mul_div_cancel numKVBlocks (by omega)
  rw [hcdiv] at hmp hlp hacc
  set mTile : Tile .real [BLOCK_M] := ⟨fun r : TileIndex [BLOCK_M] => mPartial BLOCK_N qS qT numKVBlocks kT sc numKVBlocks r.1⟩ with hmTile
  set lTile : Tile .real [BLOCK_M] := ⟨fun r : TileIndex [BLOCK_M] => ((lPartial BLOCK_N qS qT numKVBlocks kT sc numKVBlocks r.1 : ℝ) : WithBot ℝ)⟩ with hlTile
  set accTile : Tile .real [BLOCK_M, BLOCK_DMODEL] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      ((oPartial BLOCK_N qS qT numKVBlocks kT vT sc numKVBlocks idx / lPartial BLOCK_N qS qT numKVBlocks kT sc numKVBlocks idx.1 : ℝ) : WithBot ℝ)⟩ with haccTile
  unfold taPostLoopG
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: offs_m = start_m*BLOCK_M + arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)) _
        = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 2: l_ptrs = L + off_hz*N_CTX + offs_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase L)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))
          (Op.ref .nat [BLOCK_M] "offs_m"))) _
        = some (⟨fun r : TileIndex [BLOCK_M] => (Region.cast L, s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val))⟩
            : Tile .ptr [BLOCK_M]) from by
      simp only [evalOp, evalOp.eq_def, evalOp_ref, evalOp_constNat,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same, BlockState.setReg_pids, hoh, Option.bind_eq_bind, Option.bind_some,
        Option.map_some]
      refine congrArg some ?_; ext r
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.vec_data,
          Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, NumericDType.add, NumericDType.mul, Nat.zero_add]))]
  -- stmt 3: m_ptrs = M + off_hz*N_CTX + offs_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))
          (Op.ref .nat [BLOCK_M] "offs_m"))) _
        = some (⟨fun r : TileIndex [BLOCK_M] => (Region.cast M, s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val))⟩
            : Tile .ptr [BLOCK_M]) from by
      simp only [evalOp, evalOp.eq_def, evalOp_ref, evalOp_constNat,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same, BlockState.setReg_pids, hoh, Option.bind_eq_bind, Option.bind_some,
        Option.map_some]
      refine congrArg some ?_; ext r
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.vec_data,
          Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, NumericDType.add, NumericDType.mul, Nat.zero_add]))]
  set s3 := BlockState.setReg
      (BlockState.setReg
        (BlockState.setReg
          (BlockState.setReg s "start_m" .nat [] (Tile.scalar (s.pids 0)))
          "offs_m" .nat [BLOCK_M] (Tile.vec fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
        "l_ptrs" .ptr [BLOCK_M] ⟨fun r : TileIndex [BLOCK_M] => (Region.cast L, s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val))⟩)
      "m_ptrs" .ptr [BLOCK_M] ⟨fun r : TileIndex [BLOCK_M] => (Region.cast M, s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val))⟩
    with hs3
  have hlprev3 : s3.regs .real [BLOCK_M] "l_prev" = some lTile := by
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hlp]
  have hmprev3 : s3.regs .real [BLOCK_M] "m_prev" = some mTile := by
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hmp]
  have hlptr3 : s3.regs .ptr [BLOCK_M] "l_ptrs"
      = some (⟨fun r : TileIndex [BLOCK_M] => (Region.cast L, s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val))⟩
          : Tile .ptr [BLOCK_M]) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same]
  have hmptr3 : s3.regs .ptr [BLOCK_M] "m_ptrs"
      = some (⟨fun r : TileIndex [BLOCK_M] => (Region.cast M, s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val))⟩
          : Tile .ptr [BLOCK_M]) := by
    rw [hs3, BlockState.setReg_same]
  set lOffFn : TileIndex [BLOCK_M] → Nat := fun r => s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val) with hlOffFn
  -- stmt 4: store L via l_ptrs (l_prev)
  have hstore4 : stepStmt (Stmt.store .real [BLOCK_M] (MemAccess.ptr (Op.ref .ptr [BLOCK_M] "l_ptrs"))
      (Op.ref .real [BLOCK_M] "l_prev") MaskOpt.none) s3
      = some ((TileShape.allIndices [BLOCK_M]).foldl
          (fun acc r => acc.writeMemTyped .real (Region.cast L) (lOffFn r) (lTile.data r)) s3) := by
    unfold stepStmt
    simp only [evalOp_ref, hlprev3, hlptr3, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rfl
  rw [stepStmts.cons_some hstore4]
  set s4 := (TileShape.allIndices [BLOCK_M]).foldl
      (fun acc r => acc.writeMemTyped .real (Region.cast L) (lOffFn r) (lTile.data r)) s3 with hs4
  have hmprev4 : s4.regs .real [BLOCK_M] "m_prev" = some mTile := by
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]; exact hmprev3
  have hmptr4 : s4.regs .ptr [BLOCK_M] "m_ptrs"
      = some (⟨fun r : TileIndex [BLOCK_M] => (Region.cast M, s0.pids 1 * N_CTX + (s.pids 0 * BLOCK_M + r.1.val))⟩
          : Tile .ptr [BLOCK_M]) := by
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]; exact hmptr3
  -- stmt 5: store M via m_ptrs (m_prev)
  have hstore5 : stepStmt (Stmt.store .real [BLOCK_M] (MemAccess.ptr (Op.ref .ptr [BLOCK_M] "m_ptrs"))
      (Op.ref .real [BLOCK_M] "m_prev") MaskOpt.none) s4
      = some ((TileShape.allIndices [BLOCK_M]).foldl
          (fun acc r => acc.writeMemTyped .real (Region.cast M) (lOffFn r) (mTile.data r)) s4) := by
    unfold stepStmt
    simp only [evalOp_ref, hmprev4, hmptr4, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rfl
  rw [stepStmts.cons_some hstore5]
  set s5 := (TileShape.allIndices [BLOCK_M]).foldl
      (fun acc r => acc.writeMemTyped .real (Region.cast M) (lOffFn r) (mTile.data r)) s4 with hs5
  have hacc5 : s5.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some accTile := by
    rw [hs5]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hacc]
  have hout5 : s5.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr"
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          { region := Out, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL], blockShape := [BLOCK_M, BLOCK_DMODEL],
            strides := [BLOCK_DMODEL, 1], offsets := [s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M, 0] }⟩
          : Tile .blockPtr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [hs5]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs4]; simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs3]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hop]
  -- stmt 6: acc = castFloat real→fp16 acc
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")) s5
        = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx)⟩ : Tile .fp16 [BLOCK_M, BLOCK_DMODEL]) from by
      rw [evalOp_castFloat]; erw [evalOp_ref, hacc5]; rfl))]
  set s6 := s5.setReg "acc" .fp16 [BLOCK_M, BLOCK_DMODEL]
      ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx)⟩ with hs6
  have hout6 : s6.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr"
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          { region := Out, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL], blockShape := [BLOCK_M, BLOCK_DMODEL],
            strides := [BLOCK_DMODEL, 1], offsets := [s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M, 0] }⟩
          : Tile .blockPtr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [hs6, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hout5
  have hacc6 : s6.regs .fp16 [BLOCK_M, BLOCK_DMODEL] "acc"
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx)⟩ : Tile .fp16 [BLOCK_M, BLOCK_DMODEL]) := by
    rw [hs6, BlockState.setReg_same]
  set oOffFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx => (s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val * 1 with hoOffFn
  set oValFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → TileCarrier TileDType.fp16 :=
    fun idx => FloatDType.real.cast FloatDType.fp16 (accTile.data idx) with hoValFn
  set oMask : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M + idx.1.val < D0 ∧ 0 + idx.2.1.val < BLOCK_DMODEL with hoMask
  -- stmt 7: store Out via out_tile_ptr [0,1] (boundary-checked, masked)
  have hstore7 : stepStmt (Stmt.store .fp16 [BLOCK_M, BLOCK_DMODEL]
      (MemAccess.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr") [0, 1])
      (Op.ref .fp16 [BLOCK_M, BLOCK_DMODEL] "acc") MaskOpt.none) s6
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
          (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 Out (oOffFn idx) (oValFn idx) else acc) s6) := by
    have hval : evalOp (Op.ref .fp16 [BLOCK_M, BLOCK_DMODEL] "acc") s6 = some (⟨oValFn⟩ : Tile .fp16 [BLOCK_M, BLOCK_DMODEL]) := by
      rw [evalOp_ref]; exact hacc6
    have hopval : evalOp (Op.ref .blockPtr [BLOCK_M, BLOCK_DMODEL] "out_tile_ptr") s6
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            { region := Out, baseOffset := 0, parentShape := [D0, BLOCK_DMODEL], blockShape := [BLOCK_M, BLOCK_DMODEL],
              strides := [BLOCK_DMODEL, 1], offsets := [s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M, 0] }⟩
            : Tile .blockPtr [BLOCK_M, BLOCK_DMODEL]) := by rw [evalOp_ref]; exact hout6
    unfold stepStmt
    simp only [hval, hopval, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s6 ?_
    intro acc idx _
    simp only [Bool.true_and, TileShape.blockPtr_inBounds_2d_offsets_index,
      TileShape.blockPtr_address_2d_row_offset_index]
    have hoff : (0 + (s0.pids 1 * stride_hz_2d + s0.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val * 1)
        = oOffFn idx := by simp only [hoOffFn]; ring
    by_cases hmk : oMask idx
    · rw [decide_eq_true (by simpa only [hoMask] using hmk), hoff, if_pos hmk]; rfl
    · rw [decide_eq_false (by simpa only [hoMask] using hmk), if_neg hmk]; rfl
  erw [stepStmts.cons_some hstore7, stepStmts.nil]
  set s7 := (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
      (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 Out (oOffFn idx) (oValFn idx) else acc) s6
    with hs7
  refine ⟨s7, rfl, ?_, ?_, ?_⟩
  · -- Out readback
    intro idx hActive
    have hinjO : Function.Injective oOffFn := by rw [hoOffFn]; exact houtinj
    have hmaskIdx : oMask idx := by
      rw [hoMask]; refine ⟨?_, by have := idx.2.1.isLt; omega⟩
      simp only [active, rowIndex] at hActive
      omega
    rw [hs7]
    rw [show outOffset s0 (s0.pids 1 * stride_hz_2d) BLOCK_DMODEL 1 BLOCK_M idx = oOffFn idx from by
      simp only [outOffset, rowIndex, dIndex, hoOffFn]; ring]
    rw [scatter_memcell_fp16_prop_masked_nd s6 oOffFn oValFn oMask hinjO idx]
    rw [if_pos hmaskIdx]
    simp only [hoValFn, haccTile, FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
      FloatDType.ofWithBot, FloatDType.toWithBot]
    rfl
  · -- L readback
    intro i
    have hrawInj : Function.Injective (fun idx : TileIndex [BLOCK_M] => lOffFn idx) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hh
      simp only [hlOffFn] at hh
      obtain rfl : a = b := Fin.ext (by omega)
      rfl
    rw [show s7.readMem L (lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i)
          = s4.readMem L (lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i) from by
      unfold BlockState.readMem
      rw [hs7, foldl_writeMemTyped_fp16_mask_other_region oMask oOffFn oValFn _ hLOut]
      rw [hs6, BlockState.setReg_mem, hs5,
        foldl_writeMemTyped_real_other_region (W := Region.cast M) (R := L) lOffFn
          (fun r => mTile.data r) _ (by intro hc; exact hLM (by simpa using hc.symm))]]
    rw [hs4]
    rw [show lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i = lOffFn (i, PUnit.unit) from by
      simp only [lRowOffset, hlOffFn, hpids]]
    simp only [BlockState.writeMemTyped_real]
    erw [BlockState.scatter_readback_nd (region := Region.cast L) s3 lOffFn
      (fun idx : TileIndex [BLOCK_M] => FloatDType.real.storeValue (lTile.data idx)) hrawInj (i, PUnit.unit)]
    simp only [hlTile, FloatDType.real_storeValue, WithBot.unbotD_coe]
  · -- M readback
    intro i
    have hrawInj : Function.Injective (fun idx : TileIndex [BLOCK_M] => lOffFn idx) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hh
      simp only [hlOffFn] at hh
      obtain rfl : a = b := Fin.ext (by omega)
      rfl
    rw [show s7.readMem M (lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i)
          = s5.readMem M (lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i) from by
      unfold BlockState.readMem
      rw [hs7, foldl_writeMemTyped_fp16_mask_other_region oMask oOffFn oValFn _ hMOut]
      rw [hs6, BlockState.setReg_mem]]
    rw [hs5]
    rw [show lRowOffset s0 (s0.pids 1) N_CTX BLOCK_M i = lOffFn (i, PUnit.unit) from by
      simp only [lRowOffset, hlOffFn, hpids]]
    simp only [BlockState.writeMemTyped_real]
    erw [BlockState.scatter_readback_nd (region := Region.cast M) s4 lOffFn
      (fun idx : TileIndex [BLOCK_M] => FloatDType.real.storeValue (mTile.data idx)) hrawInj (i, PUnit.unit)]
    simp only [hmTile, FloatDType.real_storeValue]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
open VeriTile.Examples.FA1MathCausal in
/-- **General full-kernel forward execution chain.** Running the lowered
`_fwd_kernel` body (`taPreLoopG ++ forRangeDyn(0, (pids0+1)·BLOCK_M, BLOCK_N) ::
taPostLoopG`) from a clean state `s`, with `numKVBlocks = (pids0+1)·BLOCK_M /
BLOCK_N` causal KV blocks (`SEQ = BLOCK_N·numKVBlocks = (pids0+1)·BLOCK_M`), reaches
a final state `sF` whose `Out`/`L`/`M` stores hold the genuine general closed-form
values `fwdOutSpecG`/`fwdLSpecG`/`fwdMSpecG`. Honest side conditions: positive
block dims, `BLOCK_N ∣ (pids0+1)·BLOCK_M`, contiguous layout, the output-offset
injectivity, and the boundary `pids1·stride_hz_2d + BLOCK_M·(pids0+1) ≤ D0`. -/
theorem ta_execG (Q K V L M Out : RegionName) (s : BlockState) (sc : ℝ)
    (stride_qz stride_qh Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBD : 0 < BLOCK_DMODEL)
    (hdvd : BLOCK_N ∣ (s.pids 0 + 1) * BLOCK_M)
    (hbound : s.pids 1 * (stride_qh / BLOCK_DMODEL) + (s.pids 0 + 1) * BLOCK_M ≤ D0)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s.pids 1 * (stride_qh / BLOCK_DMODEL) + s.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val * 1))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N) s = some sF
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          active s (s.pids 1 * (stride_qh / BLOCK_DMODEL)) D0 BLOCK_M idx →
          sF.mem Out (outOffset s (s.pids 1 * (stride_qh / BLOCK_DMODEL)) BLOCK_DMODEL 1 BLOCK_M idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (fwdOutSpecG s Q K V (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL
                    ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc idx))))
      ∧ (∀ i : Fin BLOCK_M, sF.readMem L (lRowOffset s (s.pids 1) N_CTX BLOCK_M i)
          = fwdLSpecG s Q K (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc
              (Nat.mul_pos (Nat.succ_pos _) hBM) i)
      ∧ (∀ i : Fin BLOCK_M, sF.readMem M (lRowOffset s (s.pids 1) N_CTX BLOCK_M i)
          = fwdMSpecG s Q K (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc
              (Nat.mul_pos (Nat.succ_pos _) hBM) i) := by
  set stride_hz_2d := stride_qh / BLOCK_DMODEL with hstr
  set numKVBlocks := (s.pids 0 + 1) * BLOCK_M / BLOCK_N with hnum
  have hSEQ : BLOCK_N * numKVBlocks = (s.pids 0 + 1) * BLOCK_M := by
    rw [hnum, Nat.mul_div_cancel' hdvd]
  have hnumpos : 0 < numKVBlocks := by
    rw [hnum]; exact Nat.div_pos (Nat.le_of_dvd (by positivity) hdvd) hBN
  -- decompose the body
  have hbody : exec (triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N) s
        = stepStmts (taPreLoopG Q K V Out sc stride_qh D0 BLOCK_M BLOCK_N BLOCK_DMODEL
            ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
                  (Op.mul .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BLOCK_M))
                  (Op.constNat BLOCK_N) (taLoopBodyG sc BLOCK_M BLOCK_N BLOCK_DMODEL)
                :: taPostLoopG L M Out N_CTX BLOCK_M BLOCK_DMODEL)) s := by
    show stepStmts (triton_attention_fwd_kernel Q K V L M Out sc
          stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
          stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
          Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgKernel.body s = _
    rw [ta_body_splitG]
  rw [hbody]
  -- preLoop
  obtain ⟨sp, hpre, hsppids, hspmem, hspundef, hsmStart, hohp, homp, honp, hmpp, hlpp, haccp,
      hqp, hkpp, hvpp, hopp⟩ := taPreLoop_evalG s Q K V Out sc stride_qh D0 BLOCK_M BLOCK_N BLOCK_DMODEL hundef
  rw [stepStmts.append_some hpre]
  -- invariant base case at counter 0
  have hinv0 : taInvariantG Q K V Out sp sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks 0 sp := by
    refine ta_invariant_zeroG Q K V Out sp sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks
      ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ hspundef
    · exact hmpp
    · exact hlpp
    · exact haccp
    · rw [hqp]; refine congrArg some ?_; ext idx
      simp only [fwdQTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    · rw [homp, hsppids]
    · exact honp
    · rw [hkpp, hsppids, hstr]
    · rw [hvpp, hsppids, hstr]
    · rw [hopp, hsppids, hstr]
    · rw [hohp, hsppids]
  -- run the loop via forRangeDyn_inv
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BLOCK_M))
      (stepOp := Op.constNat BLOCK_N)
      (P := fun i st => taInvariantG Q K V Out sp sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks i st)
      (s_init := sp)
      (by rw [evalOp_constNat])
      (by rw [evalOp_mul, evalOp_add, evalOp_ref, hsmStart]
          simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
          refine congrArg some ?_
          ext u
          simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
            NumericDType.add, NumericDType.mul]
          show (s.pids 0 + 1) * BLOCK_M = BLOCK_N * numKVBlocks
          rw [hSEQ])
      (by rw [evalOp_constNat])
      (by omega)
      hinv0
      (fun i st hi hP => ta_attn_stepG Q K V Out sp sc stride_hz_2d D0 BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks
        hBM hBN hBD (by rw [hsppids, hstr, hSEQ]; exact hbound) i st hi hP)
  rw [stepStmts.cons_some hloop]
  -- final counter = SEQ
  have hfinal : final = BLOCK_N * numKVBlocks := by
    simp only [taInvariantG] at hinvL
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    rw [hSEQ] at hfin hle ⊢
    omega
  subst hfinal
  -- postLoop
  obtain ⟨sF, hpostStep, hO, hLrb, hMrb⟩ :=
    ta_postLoopG Q K V L M Out sp sc stride_hz_2d D0 N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks sL
      hBM hBN hBD hLOut hMOut hLM
      (by simp only [hsppids]; exact houtinj) hinvL
  refine ⟨sF, hpostStep, ?_, ?_, ?_⟩
  · -- Out readback
    intro idx hActive
    have htileQ : fwdQTileG sp Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL
        = fwdQTileG s Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL := by
      funext i; simp only [fwdQTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileK : fwdKTileG sp K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL
        = fwdKTileG s K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL := by
      funext i; simp only [fwdKTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileV : fwdVTileG sp V stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL
        = fwdVTileG s V stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL := by
      funext i; simp only [fwdVTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have hooff : outOffset sp (sp.pids 1 * stride_hz_2d) BLOCK_DMODEL 1 BLOCK_M idx
        = outOffset s (s.pids 1 * stride_hz_2d) BLOCK_DMODEL 1 BLOCK_M idx := by
      simp only [outOffset, rowIndex, hsppids]
    have hOidx := hO idx (by simp only [active, rowIndex, hsppids]; simpa only [active, rowIndex] using hActive)
    rw [hooff] at hOidx
    rw [hOidx]
    refine congrArg (MemCell.of .fp16) (congrArg (FloatDType.real.cast FloatDType.fp16) (congrArg some ?_))
    rw [show (s.pids 0 + 1) * BLOCK_M = BLOCK_N * numKVBlocks from hSEQ.symm]
    rw [fwdOutSpecG_eq_streaming s Q K V stride_hz_2d BLOCK_DMODEL BLOCK_N numKVBlocks BLOCK_M BLOCK_DMODEL hBN hnumpos sc idx]
    rw [htileQ, htileK, htileV, hsppids]
  · -- L readback
    intro i
    have htileQ : fwdQTileG sp Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL
        = fwdQTileG s Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL := by
      funext i; simp only [fwdQTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileK : fwdKTileG sp K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL
        = fwdKTileG s K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL := by
      funext i; simp only [fwdKTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have hloff : lRowOffset sp (sp.pids 1) N_CTX BLOCK_M i = lRowOffset s (s.pids 1) N_CTX BLOCK_M i := by
      simp only [lRowOffset, rowIndex, hsppids]
    have hLi := hLrb i
    rw [hloff] at hLi
    rw [hLi, lPartial_eq_fwdLSpecG sp Q K stride_hz_2d BLOCK_DMODEL BLOCK_N numKVBlocks BLOCK_M BLOCK_DMODEL hBN hnumpos sc i]
    -- align the expected SEQ `(pids0+1)*BLOCK_M` with the bridge's `BLOCK_N*numKVBlocks`
    have hcast : ∀ (SEQ : Nat) (heq : BLOCK_N * numKVBlocks = SEQ) (hpos : 0 < SEQ),
        fwdLSpecG sp Q K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL sc
            (heq ▸ hpos) i
          = fwdLSpecG s Q K stride_hz_2d BLOCK_DMODEL SEQ BLOCK_M BLOCK_DMODEL sc hpos i := by
      intro SEQ heq hpos; subst heq
      simp only [fwdLSpecG, fwdMSpecG, fwdCausalSetG, htileQ, htileK, hsppids]
    exact hcast ((s.pids 0 + 1) * BLOCK_M) hSEQ (Nat.mul_pos (Nat.succ_pos _) hBM)
  · -- M readback
    intro i
    have htileQ : fwdQTileG sp Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL
        = fwdQTileG s Q stride_hz_2d BLOCK_DMODEL BLOCK_M BLOCK_DMODEL := by
      funext i; simp only [fwdQTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have htileK : fwdKTileG sp K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL
        = fwdKTileG s K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_DMODEL := by
      funext i; simp only [fwdKTileG]; unfold BlockState.readMem; rw [hspmem, hsppids]
    have hloff : lRowOffset sp (sp.pids 1) N_CTX BLOCK_M i = lRowOffset s (s.pids 1) N_CTX BLOCK_M i := by
      simp only [lRowOffset, rowIndex, hsppids]
    have hMi := hMrb i
    rw [hloff] at hMi
    rw [hMi, ← fwdMSpecG_eq_mPartial sp Q K stride_hz_2d BLOCK_DMODEL BLOCK_N numKVBlocks BLOCK_M BLOCK_DMODEL hBN hnumpos sc i]
    have hcast : ∀ (SEQ : Nat) (heq : BLOCK_N * numKVBlocks = SEQ) (hpos : 0 < SEQ),
        fwdMSpecG sp Q K stride_hz_2d BLOCK_DMODEL (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL sc
            (heq ▸ hpos) i
          = fwdMSpecG s Q K stride_hz_2d BLOCK_DMODEL SEQ BLOCK_M BLOCK_DMODEL sc hpos i := by
      intro SEQ heq hpos; subst heq
      simp only [fwdMSpecG, fwdCausalSetG, htileQ, htileK, hsppids]
    exact hcast ((s.pids 0 + 1) * BLOCK_M) hSEQ (Nat.mul_pos (Nat.succ_pos _) hBM)

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General `tl.where(offs_m_curr ≥ offs_n, qk, -inf)` eval** with arbitrary
`offs_n[j] = gn j` (the general outer body sets `offs_n[j] = kRow + j`).  Kept
lanes satisfy `gn j ≤ gm i`. -/
theorem bwd_where_evalG (s : BlockState) (BM BN : Nat)
    (gm : Fin BM → Nat) (gn : Fin BN → Nat) (qktile : Tile .real [BM, BN])
    (hom : s.regs .nat [BM] "offs_m_curr" = some (Tile.vec gm))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec gn))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m_curr"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) s
      = some ⟨fun idx : TileIndex [BM, BN] =>
          if gn idx.2.1 ≤ gm idx.1 then qktile.data idx else (⊥ : WithBot ℝ)⟩ := by
  have hexpM : @evalOp TileDType.nat [BM, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m_curr")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec gm)) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hom
  have hexpN : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec gn)) :=
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
  by_cases h : gn idx.2.1 ≤ gm idx.1
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

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General** `qk = tl.dot(a, tl.trans(b))` raw-dot eval (no scale/zero-seed),
parametric over `[BM, BD]` operand shape (result `[BM, BM]`). -/
theorem bwd_dot_trans_evalG (s : BlockState) (BM BD : Nat) (na nb : RegName)
    (atile btile : Tile .real [BM, BD])
    (ha : s.regs .real [BM, BD] na = some atile)
    (hb : s.regs .real [BM, BD] nb = some btile) :
    @evalOp TileDType.real [BM, BM]
        (Op.dot (batch := []) (Op.ref .real [BM, BD] na)
          (Op.transpose (batch := []) (Op.ref .real [BM, BD] nb))) s
      = some (Tile.dot [] atile (Tile.transpose [] btile)) := by
  have hbr : evalOp (Op.ref .real [BM, BD] nb) s = some btile := by rw [evalOp_ref, hb]
  have har : evalOp (Op.ref .real [BM, BD] na) s = some atile := by rw [evalOp_ref, ha]
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [BM, BD] nb)) s
      = some (Tile.transpose [] btile) := by
    erw [evalOp_transpose [] (Op.ref .real [BM, BD] nb), hbr]; rfl
  erw [evalOp_dot [] (Op.ref .real [BM, BD] na)
    (Op.transpose (batch := []) (Op.ref .real [BM, BD] nb)), har, htr]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General** `acc += tl.dot((tl.trans(x.to(fp16))).to(real), y)` eval, parametric
over `[BM, BM]` score-tile `x` and `[BM, BD]` `acc`/`y` (matches `dv`/`dk`). -/
theorem bwd_trans_fp16_dot_evalG (s : BlockState) (BM BD : Nat) (nacc nx ny : RegName)
    (acctile : Tile .real [BM, BD]) (xtile : Tile .real [BM, BM]) (ytile : Tile .real [BM, BD])
    (hacc : s.regs .real [BM, BD] nacc = some acctile)
    (hx : s.regs .real [BM, BM] nx = some xtile)
    (hy : s.regs .real [BM, BD] ny = some ytile) :
    @evalOp TileDType.real [BM, BD]
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BM, BD] nacc)
          (Op.dot (batch := [])
            (Op.castFloat .fp16 .real
              (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx))))
            (Op.ref .real [BM, BD] ny))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot []
            (⟨fun idx : TileIndex [BM, BM] =>
                FloatDType.fp16.cast FloatDType.real
                  ((Tile.transpose []
                    (⟨fun jdx : TileIndex [BM, BM] =>
                        FloatDType.real.cast FloatDType.fp16 (xtile.data jdx)⟩ : Tile .fp16 [BM, BM])).data idx)⟩
              : Tile .real [BM, BM])
            ytile)) := by
  have haccr : evalOp (Op.ref .real [BM, BD] nacc) s = some acctile := by rw [evalOp_ref, hacc]
  have hyr : evalOp (Op.ref .real [BM, BD] ny) s = some ytile := by rw [evalOp_ref, hy]
  have hxr : evalOp (Op.ref .real [BM, BM] nx) s = some xtile := by rw [evalOp_ref, hx]
  have hcast : @evalOp TileDType.fp16 [BM, BM]
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx)) s
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (xtile.data idx)⟩ : Tile .fp16 [BM, BM]) := by
    erw [evalOp_castFloat .real .fp16 (Op.ref .real [BM, BM] nx), hxr]; rfl
  have htr : @evalOp TileDType.fp16 [BM, BM]
      (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx))) s
      = some (Tile.transpose [] (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (xtile.data idx)⟩ : Tile .fp16 [BM, BM])) := by
    erw [evalOp_transpose [] (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx)), hcast]; rfl
  have hcastback : @evalOp TileDType.real [BM, BM]
      (Op.castFloat .fp16 .real
        (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx)))) s
      = some (⟨fun idx : TileIndex [BM, BM] =>
          FloatDType.fp16.cast FloatDType.real
            ((Tile.transpose []
              (⟨fun jdx : TileIndex [BM, BM] =>
                  FloatDType.real.cast FloatDType.fp16 (xtile.data jdx)⟩ : Tile .fp16 [BM, BM])).data idx)⟩
        : Tile .real [BM, BM]) := by
    erw [evalOp_castFloat .fp16 .real
      (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx))), htr]; rfl
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := [])
        (Op.castFloat .fp16 .real
          (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx))))
        (Op.ref .real [BM, BD] ny)) s
      = some (Tile.dot []
          (⟨fun idx : TileIndex [BM, BM] =>
              FloatDType.fp16.cast FloatDType.real
                ((Tile.transpose []
                  (⟨fun jdx : TileIndex [BM, BM] =>
                      FloatDType.real.cast FloatDType.fp16 (xtile.data jdx)⟩ : Tile .fp16 [BM, BM])).data idx)⟩
            : Tile .real [BM, BM]) ytile) := by
    erw [evalOp_dot []
      (Op.castFloat .fp16 .real
        (Op.transpose (batch := []) (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nx))))
      (Op.ref .real [BM, BD] ny), hcastback, hyr]; rfl
  rw [evalOp_add]
  simp only [haccr, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General** `dq += tl.dot((ds.to(fp16)).to(real), k)` eval (no transpose),
parametric over `[BM, BM]` `ds` and `[BM, BD]` `dq`/`k`. -/
theorem bwd_fp16_dot_evalG (s : BlockState) (BM BD : Nat) (ndq nds nk : RegName)
    (dqtile : Tile .real [BM, BD]) (dstile : Tile .real [BM, BM]) (ktile : Tile .real [BM, BD])
    (hdq : s.regs .real [BM, BD] ndq = some dqtile)
    (hds : s.regs .real [BM, BM] nds = some dstile)
    (hk : s.regs .real [BM, BD] nk = some ktile) :
    @evalOp TileDType.real [BM, BD]
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BM, BD] ndq)
          (Op.dot (batch := [])
            (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nds)))
            (Op.ref .real [BM, BD] nk))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) dqtile
          (Tile.dot []
            (⟨fun idx : TileIndex [BM, BM] =>
                FloatDType.fp16.cast FloatDType.real
                  (FloatDType.real.cast FloatDType.fp16 (dstile.data idx))⟩ : Tile .real [BM, BM])
            ktile)) := by
  have hdqr : evalOp (Op.ref .real [BM, BD] ndq) s = some dqtile := by rw [evalOp_ref, hdq]
  have hkr : evalOp (Op.ref .real [BM, BD] nk) s = some ktile := by rw [evalOp_ref, hk]
  have hdsr : evalOp (Op.ref .real [BM, BM] nds) s = some dstile := by rw [evalOp_ref, hds]
  have hcast : @evalOp TileDType.fp16 [BM, BM]
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nds)) s
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (dstile.data idx)⟩ : Tile .fp16 [BM, BM]) := by
    erw [evalOp_castFloat .real .fp16 (Op.ref .real [BM, BM] nds), hdsr]; rfl
  have hcastback : @evalOp TileDType.real [BM, BM]
      (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nds))) s
      = some (⟨fun idx : TileIndex [BM, BM] =>
          FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 (dstile.data idx))⟩ : Tile .real [BM, BM]) := by
    erw [evalOp_castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nds)), hcast]
    refine congrArg some ?_; ext idx; rfl
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := [])
        (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nds)))
        (Op.ref .real [BM, BD] nk)) s
      = some (Tile.dot []
          (⟨fun idx : TileIndex [BM, BM] =>
              FloatDType.fp16.cast FloatDType.real
                (FloatDType.real.cast FloatDType.fp16 (dstile.data idx))⟩ : Tile .real [BM, BM]) ktile) := by
    erw [evalOp_dot []
      (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BM] nds)))
      (Op.ref .real [BM, BD] nk), hcastback, hkr]; rfl
  rw [evalOp_add]
  simp only [hdqr, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

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
/-- **General `dp = dp + dot(do, trans(v))` eval**, parametric over `[BM, BD]`. -/
theorem bwd_dpdot_evalG (s : BlockState) (BM BD : Nat) (dptile : Tile .real [BM, BM])
    (dotile vtile : Tile .real [BM, BD])
    (hdp : s.regs .real [BM, BM] "dp" = some dptile)
    (hdo : s.regs .real [BM, BD] "do_val" = some dotile)
    (hv : s.regs .real [BM, BD] "v" = some vtile) :
    @evalOp TileDType.real [BM, BM]
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BM, BM] "dp")
          (Op.dot (batch := []) (Op.ref .real [BM, BD] "do_val")
            (Op.transpose (batch := []) (Op.ref .real [BM, BD] "v")))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) dptile
          (Tile.dot [] dotile (Tile.transpose [] vtile))) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hdp, Option.bind_eq_bind, Option.bind_some]
  rw [bwd_dot_trans_evalG s BM BD "do_val" "v" dotile vtile hdo hv]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General `ds = (p · dp) · sc` eval**, parametric over `[BM, BM]`. -/
theorem bwd_ds_evalG (s : BlockState) (BM : Nat) (sc : ℝ)
    (ptile dptile : Tile .real [BM, BM])
    (hp : s.regs .real [BM, BM] "p" = some ptile)
    (hdp : s.regs .real [BM, BM] "dp" = some dptile) :
    @evalOp TileDType.real [BM, BM]
        (Op.mul .real Broadcast.scalarR
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BM] "p") (Op.ref .real [BM, BM] "dp"))
          (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            ptile dptile)
          (Tile.scalar (some sc : WithBot ℝ))) := by
  rw [evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, evalOp_const, hp, hdp, Option.bind_eq_bind, Option.bind_some]


set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General boundary-checked block-ptr `.real` load** of `bwdPtrTileG R D0 BM BD
rowOff`.  Each lane `(i,e)` is in bounds (`rowOff + i < D0`, `e < BD`) and reads
`R[(rowOff + i)·BD + e]`. -/
theorem bwd_load_bc_evalG
    (R : RegionName) (base D0 BM BD rowOff : Nat) (ptrOp : Op .blockPtr [BM, BD]) (s t : BlockState)
    (hbdvd : BD ∣ base)
    (hmem : t.mem = s.mem)
    (hrow : ∀ i : Fin BM, base / BD + rowOff + i.val < D0)
    (hp : evalOp ptrOp t = some (bwdPtrTileG R base D0 BM BD rowOff)) :
    evalOp (.load .real (.blockPtr ptrOp [0, 1]) .none) t
      = some ⟨fun idx : TileIndex [BM, BD] =>
          some (s.readMem R (base + (rowOff + idx.1.val) * BD + idx.2.1.val))⟩ := by
  have hp' : evalOp ptrOp t = some
      ⟨fun _ : TileIndex [BM, BD] =>
        { region := R, baseOffset := 0, parentShape := [D0, BD],
          blockShape := [BM, BD], strides := [BD, 1],
          offsets := [base / BD + rowOff, 0] }⟩ := by
    rw [hp]; simp only [bwdPtrTileG]
  rw [ta_load_blockPtr_bc_eval R 0 D0 BD BM BD BD 1 (base / BD + rowOff) ptrOp t hp']
  refine congrArg some ?_
  ext idx
  have he : idx.2.1.val < BD := idx.2.1.isLt
  have hi : base / BD + rowOff + idx.1.val < D0 := hrow idx.1
  simp only [hi, he, and_self, if_true]
  simp only [BlockState.readMem, hmem]
  congr 1
  rw [show base / BD + rowOff + idx.1.val = base / BD + (rowOff + idx.1.val) from by omega,
    Nat.add_mul, Nat.div_mul_cancel hbdvd]
  ring

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General no-boundary-check block-ptr `.real` load** of `bwdPtrTileG R D0 BM BD
rowOff` (the `dq = tl.load(dq_tile_ptr)` load).  Reads each lane `(i,e)` at
`R[(rowOff + i)·BD + e]`. -/
theorem bwd_load_nobc_evalG
    (R : RegionName) (base D0 BM BD rowOff : Nat) (ptrOp : Op .blockPtr [BM, BD]) (s t : BlockState)
    (hbdvd : BD ∣ base)
    (hmem : t.mem = s.mem)
    (hp : evalOp ptrOp t = some (bwdPtrTileG R base D0 BM BD rowOff)) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) t
      = some ⟨fun idx : TileIndex [BM, BD] =>
          some (s.readMem R (base + (rowOff + idx.1.val) * BD + idx.2.1.val))⟩ := by
  have hp' : evalOp ptrOp t = some
      ⟨fun _ : TileIndex [BM, BD] =>
        { region := R, baseOffset := 0, parentShape := [D0, BD],
          blockShape := [BM, BD], strides := [BD, 1],
          offsets := [base / BD + rowOff, 0] }⟩ := by
    rw [hp]; simp only [bwdPtrTileG]
  simp only [evalOp, hp', Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [BlockPtr.inBounds_nil_checkedAxes, Bool.and_true, Bool.true_and, if_true,
    TileShape.blockPtr_address_2d_row_offset_index, BlockState.readMemValue_real]
  simp only [BlockState.readMem, hmem]
  congr 1
  rw [show base / BD + rowOff + idx.1.val = base / BD + (rowOff + idx.1.val) from by omega,
    Nat.add_mul, Nat.div_mul_cancel hbdvd]
  ring

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General no-boundary-check `.real` block-ptr store** of `bwdPtrTileG R D0 BM BD
rowOff`: writes each lane `(i,e)` to `R[(rowOff + i)·BD + e]` with value
`(vt.data idx).unbotD 0`. -/
theorem bwd_store_dq_evalG (s sPre : BlockState) (R : RegionName) (base D0 BM BD rowOff : Nat)
    (hbdvd : BD ∣ base)
    (vt : Tile .real [BM, BD])
    (hptr : sPre.regs .blockPtr [BM, BD] "dq_tile_ptr" = some (bwdPtrTileG R base D0 BM BD rowOff))
    (hval : sPre.regs .real [BM, BD] "dq" = some vt) :
    stepStmt (Stmt.store .real [BM, BD]
        (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] "dq_tile_ptr") [])
        (Op.ref .real [BM, BD] "dq") MaskOpt.none) sPre
      = some ((TileShape.allIndices [BM, BD]).foldl
          (fun acc idx => acc.writeMem R (base + (rowOff + idx.1.val) * BD + idx.2.1.val)
            ((vt.data idx).unbotD 0)) sPre) := by
  unfold stepStmt
  simp only [evalOp_ref, hptr, hval, Option.bind_eq_bind, Option.bind_some, Option.map]
  refine congrArg some ?_
  refine List.foldl_ext _ _ _ (fun acc idx _ => ?_)
  rw [bwdPtrTileG]
  simp only [BlockPtr.inBounds_nil_checkedAxes, Bool.and_true, Bool.true_and, if_true,
    TileShape.blockPtr_address_2d_row_offset_index, BlockState.writeMemTyped_real,
    FloatDType.real_storeValue]
  congr 1
  rw [show base / BD + rowOff + idx.1.val = base / BD + (rowOff + idx.1.val) from by omega,
    Nat.add_mul, Nat.div_mul_cancel hbdvd]
  ring

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General fp16 boundary-checked block-ptr store** (dimension-general,
`bwd_store_fp16_bc_evalG`).  Storing `castFloat .real .fp16 (ref valNm)` through a
`bwdPtrTileG R base D0 BM BD rowOff` block pointer (boundary check `[0,1]`) writes,
for every in-bounds lane `(i,e)`, an fp16 `MemCell` at `R[base + (rowOff+i)·BD + e]`.
Under `rowOff + i < D0` (the `hrow` side-condition) every lane is in bounds, so the
readback at any lane is `MemCell.of .fp16 (real.cast fp16 (vt.data idx))`. -/
private theorem bwd_store_fp16_bc_evalG (R : RegionName) (base D0 BM BD rowOff : Nat)
    (sPre : BlockState) (ptrNm valNm : RegName) (vt : Tile .real [BM, BD])
    (hbdvd : BD ∣ base)
    (hrow : ∀ i : Fin BM, base / BD + rowOff + i.val < D0)
    (hptr : sPre.regs .blockPtr [BM, BD] ptrNm = some (bwdPtrTileG R base D0 BM BD rowOff))
    (hval : sPre.regs .real [BM, BD] valNm = some vt)
    (hsome : ∀ idx : TileIndex [BM, BD], ∃ r : ℝ, vt.data idx = some r) :
    ∃ sF, stepStmt (Stmt.store .fp16 [BM, BD]
        (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] ptrNm) [0, 1])
        (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) MaskOpt.none) sPre
      = some sF
      ∧ sF.pids = sPre.pids
      ∧ (∀ idx : TileIndex [BM, BD],
          sF.mem R (base + (rowOff + idx.1.val) * BD + idx.2.1.val)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16 (vt.data idx)))
      ∧ (∀ (R' : RegionName) (o : Nat), R' ≠ R → sF.mem R' o = sPre.mem R' o)
      ∧ (∀ o : Nat, (∀ idx : TileIndex [BM, BD],
            o ≠ base + (rowOff + idx.1.val) * BD + idx.2.1.val) →
          sF.mem R o = sPre.mem R o) := by
  set oOffFn : TileIndex [BM, BD] → Nat :=
    fun idx => base + (rowOff + idx.1.val) * BD + idx.2.1.val with hoOffFn
  set oValFn : TileIndex [BM, BD] → TileCarrier TileDType.fp16 :=
    fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx) with hoValFn
  set oMask : TileIndex [BM, BD] → Prop :=
    fun idx => base / BD + rowOff + idx.1.val < D0 ∧ 0 + idx.2.1.val < BD with hoMask
  have hstore : stepStmt (Stmt.store .fp16 [BM, BD]
      (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] ptrNm) [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) MaskOpt.none) sPre
      = some ((TileShape.allIndices [BM, BD]).foldl
          (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 R (oOffFn idx) (oValFn idx) else acc) sPre) := by
    have hvalop : evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) sPre
        = some (⟨oValFn⟩ : Tile .fp16 [BM, BD]) := by
      rw [evalOp_castFloat]; erw [evalOp_ref, hval]; rfl
    have hopval : evalOp (Op.ref .blockPtr [BM, BD] ptrNm) sPre
        = some (bwdPtrTileG R base D0 BM BD rowOff) := by rw [evalOp_ref]; exact hptr
    unfold stepStmt
    simp only [hopval, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    erw [hvalop]
    rw [Option.bind_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ sPre ?_
    intro acc idx _
    simp only [bwdPtrTileG, Bool.true_and,
      TileShape.blockPtr_inBounds_2d_offsets_index,
      TileShape.blockPtr_address_2d_row_offset_index]
    have hoff : 0 + (base / BD + rowOff + idx.1.val) * BD + idx.2.1.val * 1 = oOffFn idx := by
      simp only [hoOffFn]
      rw [show base / BD + rowOff + idx.1.val = base / BD + (rowOff + idx.1.val) from by omega,
        Nat.add_mul, Nat.div_mul_cancel hbdvd]
      ring
    by_cases hmk : oMask idx
    · rw [decide_eq_true (by simpa only [hoMask] using hmk), hoff]
      rw [if_pos rfl, if_pos hmk]
    · rw [decide_eq_false (by simpa only [hoMask] using hmk)]
      rw [if_neg (by decide), if_neg hmk]
  rw [hstore]
  set sF := (TileShape.allIndices [BM, BD]).foldl
      (fun acc idx => if oMask idx then acc.writeMemTyped .fp16 R (oOffFn idx) (oValFn idx) else acc) sPre
    with hsF
  have hinj : Function.Injective oOffFn := by
    rw [hoOffFn]
    rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
    simp only at h
    have hkey' : (rowOff + ma) * BD + da = (rowOff + mb) * BD + db := by omega
    have hmod1 : ((rowOff + ma) * BD + da) % BD = da := by
      rw [Nat.mul_add_mod']; exact Nat.mod_eq_of_lt hda
    have hmod2 : ((rowOff + mb) * BD + db) % BD = db := by
      rw [Nat.mul_add_mod']; exact Nat.mod_eq_of_lt hdb
    have hd : da = db := by rw [← hmod1, ← hmod2, hkey']
    subst db
    have hmeq : (rowOff + ma) * BD = (rowOff + mb) * BD := by omega
    have hm : ma = mb := by
      have := Nat.eq_of_mul_eq_mul_right (by omega : 0 < BD) hmeq
      omega
    subst mb; rfl
  have hPids : sF.pids = sPre.pids := by
    rw [hsF]
    suffices h : ∀ (l : List (TileIndex [BM, BD])) (t : BlockState),
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
  have hMem : ∀ idx : TileIndex [BM, BD],
      sF.mem R (base + (rowOff + idx.1.val) * BD + idx.2.1.val)
        = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16 (vt.data idx)) := by
    intro idx
    have hmaskIdx : oMask idx := by
      rw [hoMask]; exact ⟨hrow idx.1, by have := idx.2.1.isLt; omega⟩
    rw [hsF]
    rw [show base + (rowOff + idx.1.val) * BD + idx.2.1.val = oOffFn idx from rfl]
    rw [scatter_memcell_fp16_prop_masked_nd sPre oOffFn oValFn oMask hinj idx]
    rw [if_pos hmaskIdx]
    obtain ⟨r, hr⟩ := hsome idx
    simp only [hoValFn, hr, FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
      FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.unbotD_some]
  have hOther : ∀ (R' : RegionName) (o : Nat), R' ≠ R → sF.mem R' o = sPre.mem R' o := by
    intro R' o hne
    rw [hsF, foldl_writeMemTyped_fp16_mask_other_region oMask oOffFn oValFn _ hne]
  have hOffImg : ∀ o : Nat, (∀ idx : TileIndex [BM, BD],
        o ≠ base + (rowOff + idx.1.val) * BD + idx.2.1.val) →
      sF.mem R o = sPre.mem R o := by
    intro o ho
    rw [hsF]
    have hstep' :
        (fun (acc : BlockState) (k : TileIndex [BM, BD]) =>
          if oMask k then acc.writeMemTyped .fp16 R (oOffFn k) (oValFn k) else acc)
          =
        (fun (acc : BlockState) (k : TileIndex [BM, BD]) =>
          if decide (oMask k) then acc.writeMemTyped .fp16 R (oOffFn k) (oValFn k) else acc) := by
      funext acc k; by_cases hk : oMask k <;> simp [hk]
    rw [hstep']
    refine foldl_writeMemTyped_fp16_preserves oOffFn oValFn
      (fun k => decide (oMask k)) o _ sPre ?_
    intro k _ _
    simp only [hoOffFn]
    exact fun h => ho k h.symm
  exact ⟨sF, rfl, hPids, hMem, hOther, hOffImg⟩

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General fp16-store register preservation** (dimension-general,
`bwd_store_fp16_regs_eqG`).  An fp16 block-ptr store leaves every register untouched. -/
private theorem bwd_store_fp16_regs_eqG (sPre : BlockState) (BM BD : Nat)
    (ptrTile : Tile .blockPtr [BM, BD]) (ptrNm valNm : RegName) (vt : Tile .real [BM, BD])
    (hptr : sPre.regs .blockPtr [BM, BD] ptrNm = some ptrTile)
    (hval : sPre.regs .real [BM, BD] valNm = some vt)
    (sF : BlockState)
    (hstep : stepStmt (Stmt.store .fp16 [BM, BD]
        (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] ptrNm) [0, 1])
        (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) MaskOpt.none) sPre = some sF)
    {dt : TileDType} {sh : TileShape} {nm' : RegName} :
    sF.regs dt sh nm' = sPre.regs dt sh nm' := by
  have hvalop : evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) sPre
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [BM, BD]) := by
    rw [evalOp_castFloat]; erw [evalOp_ref, hval]; rfl
  have hopval : evalOp (Op.ref .blockPtr [BM, BD] ptrNm) sPre
      = some ptrTile := by rw [evalOp_ref]; exact hptr
  rw [show stepStmt (Stmt.store .fp16 [BM, BD]
      (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] ptrNm) [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) MaskOpt.none) sPre
      = some ((TileShape.allIndices [BM, BD]).foldl
          (fun acc i =>
            let bp := ptrTile.data i
            let idx := TileShape.indexToList [BM, BD] i
            if («true» && bp.inBounds idx [0, 1]) = «true» then
              acc.writeMemTyped .fp16 bp.region (bp.address idx)
                ((⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [BM, BD]).data i)
            else acc) sPre) from by
        unfold stepStmt
        simp only [hopval, Option.bind_eq_bind, Option.bind_some, Option.map_some]
        erw [hvalop]
        rw [Option.bind_some]] at hstep
  obtain rfl : sF = _ := Option.some.inj hstep.symm
  suffices h : ∀ (l : List (TileIndex [BM, BD])) (t : BlockState),
      (l.foldl (fun acc i =>
        let bp := ptrTile.data i
        let idx := TileShape.indexToList [BM, BD] i
        if («true» && bp.inBounds idx [0, 1]) = «true» then
          acc.writeMemTyped .fp16 bp.region (bp.address idx)
            ((⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [BM, BD]).data i)
        else acc) t).regs dt sh nm' = t.regs dt sh nm' by exact h _ sPre
  intro l
  induction l with
  | nil => intro t; rfl
  | cons hd tl ih =>
      intro t
      rw [List.foldl_cons, ih]
      by_cases hb : («true» && (ptrTile.data hd).inBounds (TileShape.indexToList [BM, BD] hd) [0, 1]) = «true»
      · rw [if_pos hb, BlockState.writeMemTyped_regs]
      · rw [if_neg hb]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General fp16-store `undef` preservation.** An fp16 block-ptr store leaves the
`undef` map untouched. -/
private theorem bwd_store_fp16_undef_eqG (sPre : BlockState) (BM BD : Nat)
    (ptrTile : Tile .blockPtr [BM, BD]) (ptrNm valNm : RegName) (vt : Tile .real [BM, BD])
    (hptr : sPre.regs .blockPtr [BM, BD] ptrNm = some ptrTile)
    (hval : sPre.regs .real [BM, BD] valNm = some vt)
    (sF : BlockState)
    (hstep : stepStmt (Stmt.store .fp16 [BM, BD]
        (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] ptrNm) [0, 1])
        (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) MaskOpt.none) sPre = some sF)
    (rg : RegionName) (o : Nat) :
    sF.undef rg o = sPre.undef rg o := by
  have hvalop : evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) sPre
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [BM, BD]) := by
    rw [evalOp_castFloat]; erw [evalOp_ref, hval]; rfl
  have hopval : evalOp (Op.ref .blockPtr [BM, BD] ptrNm) sPre
      = some ptrTile := by rw [evalOp_ref]; exact hptr
  rw [show stepStmt (Stmt.store .fp16 [BM, BD]
      (MemAccess.blockPtr (Op.ref .blockPtr [BM, BD] ptrNm) [0, 1])
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BD] valNm)) MaskOpt.none) sPre
      = some ((TileShape.allIndices [BM, BD]).foldl
          (fun acc i =>
            let bp := ptrTile.data i
            let idx := TileShape.indexToList [BM, BD] i
            if («true» && bp.inBounds idx [0, 1]) = «true» then
              acc.writeMemTyped .fp16 bp.region (bp.address idx)
                ((⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [BM, BD]).data i)
            else acc) sPre) from by
        unfold stepStmt
        simp only [hopval, Option.bind_eq_bind, Option.bind_some, Option.map_some]
        erw [hvalop]
        rw [Option.bind_some]] at hstep
  obtain rfl : sF = _ := Option.some.inj hstep.symm
  suffices h : ∀ (l : List (TileIndex [BM, BD])) (t : BlockState),
      (l.foldl (fun acc i =>
        let bp := ptrTile.data i
        let idx := TileShape.indexToList [BM, BD] i
        if («true» && bp.inBounds idx [0, 1]) = «true» then
          acc.writeMemTyped .fp16 bp.region (bp.address idx)
            ((⟨fun idx => FloatDType.real.cast FloatDType.fp16 (vt.data idx)⟩ : Tile .fp16 [BM, BD]).data i)
        else acc) t).undef rg o = t.undef rg o by exact h _ sPre
  intro l
  induction l with
  | nil => intro t; rfl
  | cons hd tl ih =>
      intro t
      rw [List.foldl_cons, ih]
      by_cases hb : («true» && (ptrTile.data hd).inBounds (TileShape.indexToList [BM, BD] hd) [0, 1]) = «true»
      · rw [if_pos hb, BlockState.writeMemTyped_undef]
      · rw [if_neg hb]

/-! ### ════════ General backward inner-body register tiles ════════

Dimension-general (`BM`/`BD`) clones of the pinned `bwdInner{Qk,Qk1,P,Dp,Ds,Dv,Dk,Dq}`
register tiles, parametric over the global query-block base row `qRow` (= `start_m`)
and key-block base row `kRow` (= `lo = start_n·BM`).  Each cell lemma collapses to the
Phase-1 `G` spec helpers (`bwdKernelQKG`/`PG`/`DPG`/`DSG`) at global rows
`I = qRow + iL`, `J = kRow + jL`.  Score blocks are square `[BM, BM]`; input/grad
tiles `[BM, BD]`.  The fp16 round-trips use `bwd_fp16_roundtrip`. -/

/-- The raw `qk` dot tile (`Σ_e q·k`), square block `[BM, BM]`. -/
noncomputable def bwdInnerQkG (s : BlockState) (Q K : RegionName) (BM BD qRow kRow : Nat) :
    Tile .real [BM, BM] :=
  Tile.dot []
    (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelQG s Q BD (qRow + idx.1.val) idx.2.1.val)⟩
      : Tile .real [BM, BD])
    (Tile.transpose []
      (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + idx.1.val) idx.2.1.val)⟩
        : Tile .real [BM, BD]))

/-- The causal-masked `qk` tile (`-inf` ⇒ `⊥` where `kRow+jL > qRow+iL`). -/
noncomputable def bwdInnerQk1G (s : BlockState) (Q K : RegionName) (BM BD qRow kRow : Nat) :
    Tile .real [BM, BM] :=
  ⟨fun idx : TileIndex [BM, BM] =>
    if kRow + idx.2.1.val ≤ qRow + idx.1.val then (bwdInnerQkG s Q K BM BD qRow kRow).data idx
    else (⊥ : WithBot ℝ)⟩

/-- The `p` tile `exp(qk1·sc − m[:,None])`. -/
noncomputable def bwdInnerPG (s : BlockState) (Q K M : RegionName) (BM BD NCTX qRow kRow : Nat) (sc : ℝ) :
    Tile .real [BM, BM] :=
  Tile.uop WithBot.realExp
    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Tile.bop NumericDType.real.mul Broadcast.scalarR (bwdInnerQk1G s Q K BM BD qRow kRow)
        (Tile.scalar (some sc : WithBot ℝ)))
      (Tile.expandDim ⟨1, by simp⟩
        (⟨fun i : TileIndex [BM] => some (bwdKernelMG s M NCTX (qRow + i.1.val))⟩ : Tile .real [BM])))

/-- The `dp` tile `(0 − Di[:,None]) + dot(do, trans(v))`. -/
noncomputable def bwdInnerDpG (s : BlockState) (V DO Delta : RegionName) (BM BD NCTX qRow kRow : Nat) :
    Tile .real [BM, BM] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (⟨fun _ : TileIndex [BM, BM] => some (0:ℝ)⟩ : Tile .real [BM, BM])
      (Tile.expandDim ⟨1, by simp⟩
        (⟨fun i : TileIndex [BM] => some (bwdKernelDiG s Delta NCTX (qRow + i.1.val))⟩ : Tile .real [BM])))
    (Tile.dot []
      (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelDOG s DO BD (qRow + idx.1.val) idx.2.1.val)⟩
        : Tile .real [BM, BD])
      (Tile.transpose []
        (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelVG s V BD (kRow + idx.1.val) idx.2.1.val)⟩
          : Tile .real [BM, BD])))

/-- The `ds` tile `p · dp · sc`. -/
noncomputable def bwdInnerDsG (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) : Tile .real [BM, BM] :=
  Tile.bop NumericDType.real.mul Broadcast.scalarR
    (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc) (bwdInnerDpG s V DO Delta BM BD NCTX qRow kRow))
    (Tile.scalar (some sc : WithBot ℝ))

/-- Cell value of the raw `qk` dot. -/
theorem bwdInnerQkG_cell (s : BlockState) (Q K : RegionName) (BM BD qRow kRow : Nat)
    (i j : Fin BM) :
    (bwdInnerQkG s Q K BM BD qRow kRow).data (i, j, PUnit.unit)
      = some (bwdKernelQKG s Q K BD (qRow + i.val) (kRow + j.val)) := by
  simp only [bwdInnerQkG, Tile.dot, Tile.transpose, bwdKernelQKG, bwdKernelQG, bwdKernelKG,
    Option.map₂_some_some, WithBot.sum_someTerm_eq_some]

/-- Cell value of the masked `p` tile (matches `bwdKernelPG`). -/
theorem bwdInnerPG_cell (s : BlockState) (Q K M : RegionName) (BM BD NCTX qRow kRow : Nat) (sc : ℝ)
    (i j : Fin BM) :
    (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc).data (i, j, PUnit.unit)
      = some (bwdKernelPG s Q K M BD NCTX sc (qRow + i.val) (kRow + j.val)) := by
  have hqk1 : (bwdInnerQk1G s Q K BM BD qRow kRow).data (i, j, PUnit.unit)
      = if kRow + j.val ≤ qRow + i.val then some (bwdKernelQKG s Q K BD (qRow + i.val) (kRow + j.val))
        else (⊥ : WithBot ℝ) := by
    simp only [bwdInnerQk1G]
    by_cases h : kRow + j.val ≤ qRow + i.val
    · rw [if_pos h, if_pos h, bwdInnerQkG_cell]
    · rw [if_neg h, if_neg h]
  simp only [bwdInnerPG, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar, bwdKernelPG, hqk1, Tile.expandDim_data, TileShape.dropInsertedIndex_succ,
    TileShape.dropInsertedIndex_nil, WithBot.realMul, WithBot.realSub]
  by_cases h : kRow + j.val ≤ qRow + i.val
  · simp only [if_pos h, NumericDType.mul, NumericDType.sub, WithBot.realMul,
      WithBot.realSub, Option.map₂_some_some, WithBot.realExp_some]
  · simp only [if_neg h, NumericDType.mul, NumericDType.sub, WithBot.realMul,
      WithBot.realSub, Option.map₂_none_left, Option.map₂_none_right,
      WithBot.realExp_bot]
    rfl

/-- Cell value of the `dp` tile (matches `bwdKernelDPG`). -/
theorem bwdInnerDpG_cell (s : BlockState) (V DO Delta : RegionName) (BM BD NCTX qRow kRow : Nat)
    (i j : Fin BM) :
    (bwdInnerDpG s V DO Delta BM BD NCTX qRow kRow).data (i, j, PUnit.unit)
      = some (bwdKernelDPG s V DO Delta BD NCTX (qRow + i.val) (kRow + j.val)) := by
  simp only [bwdInnerDpG, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.expandDim_data, TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    Tile.dot, Tile.transpose, bwdKernelDPG, bwdKernelDOG, bwdKernelVG, bwdKernelDiG,
    NumericDType.add, NumericDType.sub,
    WithBot.realAdd, WithBot.realSub, Option.map₂_some_some, WithBot.sum_someTerm_eq_some,
    Option.map₂_some_coe, Option.map₂_coe_some]
  congr 1
  ring

/-- Cell value of the `ds` tile (matches `bwdKernelDSG`). -/
theorem bwdInnerDsG_cell (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (i j : Fin BM) :
    (bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc).data (i, j, PUnit.unit)
      = some (bwdKernelDSG s Q K V DO M Delta BD NCTX sc (qRow + i.val) (kRow + j.val)) := by
  simp only [bwdInnerDsG, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar,
    NumericDType.mul, WithBot.realMul]
  rw [bwdInnerPG_cell, bwdInnerDpG_cell]
  simp only [bwdKernelDSG, Option.map₂_some_some, Option.map₂_some_coe]

/-- General `dv` accumulator delta after the inner body: `acc + dot((trans(p.to
fp16)).to real, do)`.  Cell `(jL,e)` adds `Σ_iL fp16(pG (qRow+iL) (kRow+jL))·
doG (qRow+iL) e` to the prior accumulator cell `acc[jL,e]`. -/
noncomputable def bwdInnerDvG (s : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (acc : Tile .real [BM, BD]) :
    Tile .real [BM, BD] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    acc
    (Tile.dot []
      (⟨fun idx : TileIndex [BM, BM] =>
          FloatDType.fp16.cast FloatDType.real
            ((Tile.transpose []
              (⟨fun jdx : TileIndex [BM, BM] =>
                  FloatDType.real.cast FloatDType.fp16 ((bwdInnerPG s Q K M BM BD NCTX qRow kRow sc).data jdx)⟩
                : Tile .fp16 [BM, BM])).data idx)⟩ : Tile .real [BM, BM])
      (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelDOG s DO BD (qRow + idx.1.val) idx.2.1.val)⟩
        : Tile .real [BM, BD]))

/-- General `dk` accumulator delta: `acc + dot((trans(ds.to fp16)).to real, q)`. -/
noncomputable def bwdInnerDkG (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (acc : Tile .real [BM, BD]) :
    Tile .real [BM, BD] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    acc
    (Tile.dot []
      (⟨fun idx : TileIndex [BM, BM] =>
          FloatDType.fp16.cast FloatDType.real
            ((Tile.transpose []
              (⟨fun jdx : TileIndex [BM, BM] =>
                  FloatDType.real.cast FloatDType.fp16 ((bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc).data jdx)⟩
                : Tile .fp16 [BM, BM])).data idx)⟩ : Tile .real [BM, BM])
      (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelQG s Q BD (qRow + idx.1.val) idx.2.1.val)⟩
        : Tile .real [BM, BD]))

/-- General `dq` register tile: `dq_loaded + dot((ds.to fp16).to real, k)`, where
`dq_loaded[iL,e] = DQ[bwdKBase s + (qRow+iL)·BD + e]`. -/
noncomputable def bwdInnerDqG (s : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) : Tile .real [BM, BD] :=
  Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (⟨fun idx : TileIndex [BM, BD] =>
        some (s.readMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩ : Tile .real [BM, BD])
    (Tile.dot []
      (⟨fun idx : TileIndex [BM, BM] =>
          FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 ((bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc).data idx))⟩
        : Tile .real [BM, BM])
      (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + idx.1.val) idx.2.1.val)⟩
        : Tile .real [BM, BD]))

/-- Cell value of the `dv` accumulator delta:
`dv[jL,e] = acc[jL,e] + Σ_iL fp16(pG (qRow+iL) (kRow+jL))·doG (qRow+iL) e`. -/
theorem bwdInnerDvG_cell (s : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (acc : Tile .real [BM, BD]) (j : Fin BM) (e : Fin BD) :
    (bwdInnerDvG s Q K M DO BM BD NCTX qRow kRow sc acc).data (j, e, PUnit.unit)
      = Option.map₂ (· + ·) (acc.data (j, e, PUnit.unit))
          (some (∑ iL : Fin BM,
            bwdFp16 (bwdKernelPG s Q K M BD NCTX sc (qRow + iL.val) (kRow + j.val)) *
              bwdKernelDOG s DO BD (qRow + iL.val) e.val)) := by
  simp only [bwdInnerDvG, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  have hsum : (∑ iL : Fin BM, Option.map₂ (· * ·)
        ((⟨fun idx : TileIndex [BM, BM] =>
            FloatDType.fp16.cast FloatDType.real
              ((Tile.transpose []
                (⟨fun jdx : TileIndex [BM, BM] =>
                    FloatDType.real.cast FloatDType.fp16 ((bwdInnerPG s Q K M BM BD NCTX qRow kRow sc).data jdx)⟩
                  : Tile .fp16 [BM, BM])).data idx)⟩ : Tile .real [BM, BM]).data (j, iL, PUnit.unit))
        ((⟨fun idx : TileIndex [BM, BD] => some (bwdKernelDOG s DO BD (qRow + idx.1.val) idx.2.1.val)⟩
            : Tile .real [BM, BD]).data (iL, e, PUnit.unit)) : WithBot ℝ)
      = some (∑ iL : Fin BM,
          bwdFp16 (bwdKernelPG s Q K M BD NCTX sc (qRow + iL.val) (kRow + j.val)) *
            bwdKernelDOG s DO BD (qRow + iL.val) e.val) := by
    rw [← WithBot.sum_someTerm_eq_some]
    refine Finset.sum_congr rfl (fun iL _ => ?_)
    simp only [Tile.transpose]
    rw [bwdInnerPG_cell, bwd_fp16_roundtrip]
    simp only [Option.map₂_some_some]
  rw [hsum]

/-- Cell value of the `dk` accumulator delta. -/
theorem bwdInnerDkG_cell (s : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (acc : Tile .real [BM, BD]) (j : Fin BM) (e : Fin BD) :
    (bwdInnerDkG s Q K V DO M Delta BM BD NCTX qRow kRow sc acc).data (j, e, PUnit.unit)
      = Option.map₂ (· + ·) (acc.data (j, e, PUnit.unit))
          (some (∑ iL : Fin BM,
            bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD NCTX sc (qRow + iL.val) (kRow + j.val)) *
              bwdKernelQG s Q BD (qRow + iL.val) e.val)) := by
  simp only [bwdInnerDkG, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  have hsum : (∑ iL : Fin BM, Option.map₂ (· * ·)
        ((⟨fun idx : TileIndex [BM, BM] =>
            FloatDType.fp16.cast FloatDType.real
              ((Tile.transpose []
                (⟨fun jdx : TileIndex [BM, BM] =>
                    FloatDType.real.cast FloatDType.fp16 ((bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc).data jdx)⟩
                  : Tile .fp16 [BM, BM])).data idx)⟩ : Tile .real [BM, BM]).data (j, iL, PUnit.unit))
        ((⟨fun idx : TileIndex [BM, BD] => some (bwdKernelQG s Q BD (qRow + idx.1.val) idx.2.1.val)⟩
            : Tile .real [BM, BD]).data (iL, e, PUnit.unit)) : WithBot ℝ)
      = some (∑ iL : Fin BM,
          bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD NCTX sc (qRow + iL.val) (kRow + j.val)) *
            bwdKernelQG s Q BD (qRow + iL.val) e.val) := by
    rw [← WithBot.sum_someTerm_eq_some]
    refine Finset.sum_congr rfl (fun iL _ => ?_)
    simp only [Tile.transpose]
    rw [bwdInnerDsG_cell, bwd_fp16_roundtrip]
    simp only [Option.map₂_some_some]
  rw [hsum]

/-- Cell value of the `dq` tile:
`dq[iL,e] = DQ[bwdKBase + (qRow+iL)·BD + e] + Σ_jL fp16(dsG (qRow+iL) (kRow+jL))·kG (kRow+jL) e`. -/
theorem bwdInnerDqG_cell (s : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (i : Fin BM) (e : Fin BD) :
    (bwdInnerDqG s Q K V DO M Delta DQ BM BD NCTX qRow kRow sc).data (i, e, PUnit.unit)
      = some (s.readMem DQ (bwdKBase s + (qRow + i.val) * BD + e.val) +
          ∑ jL : Fin BM,
            bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD NCTX sc (qRow + i.val) (kRow + jL.val)) *
              bwdKernelKG s K BD (kRow + jL.val) e.val) := by
  simp only [bwdInnerDqG, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  have hsum : (∑ jL : Fin BM, Option.map₂ (· * ·)
        ((⟨fun idx : TileIndex [BM, BM] =>
            FloatDType.fp16.cast FloatDType.real
              (FloatDType.real.cast FloatDType.fp16 ((bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc).data idx))⟩
          : Tile .real [BM, BM]).data (i, jL, PUnit.unit))
        ((⟨fun idx : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + idx.1.val) idx.2.1.val)⟩
            : Tile .real [BM, BD]).data (jL, e, PUnit.unit)) : WithBot ℝ)
      = some (∑ jL : Fin BM,
          bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD NCTX sc (qRow + i.val) (kRow + jL.val)) *
            bwdKernelKG s K BD (kRow + jL.val) e.val) := by
    rw [← WithBot.sum_someTerm_eq_some]
    refine Finset.sum_congr rfl (fun jL _ => ?_)
    show Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 ((bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc).data (i, jL, PUnit.unit))))
        (some (bwdKernelKG s K BD (kRow + jL.val) e.val)) = _
    rw [bwdInnerDsG_cell, bwd_fp16_roundtrip]
    simp only [Option.map₂_some_some]

  rw [hsum]
  simp only [Option.map₂_some_some]




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

/-- A `writeMem` scatter over the same region preserves any offset disjoint from
every written offset (no-mask form matching the `bwd` `DQ` store). -/
private theorem foldl_writeMem_disjoint_offset {α : Type} (W : RegionName)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (l : List α) :
    ∀ (s : BlockState) (o : Nat), (∀ k, k ∈ l → o ≠ offsetFn k) →
      ((l.foldl (fun acc k => acc.writeMem W (offsetFn k) (valueFn k)) s).mem W o)
        = s.mem W o := by
  induction l with
  | nil => intro s o _; rfl
  | cons hd tl ih =>
      intro s o hoff
      rw [List.foldl_cons,
        ih _ _ (fun k hk => hoff k (List.mem_cons_of_mem hd hk)), BlockState.writeMem]
      simp only [true_and, if_neg (hoff hd List.mem_cons_self)]



/-! ### ════════ General inner Q-block step (Phase 2, item 3) ════════

Dimension-general (`BM`/`BD`) inner Q-block step `bwd_inner_stepG`, parametric over the
counter `start_m = qRow` (global query-block base row), the key-block base row
`kRow = lo = start_n·BM`, and arbitrary prior `dv`/`dk` accumulator tiles.  The
body is structurally `bwdInnerBodyG`; the proof reuses the general statement-evals
and register tiles built above.  We split `bwdInnerBodyG` into the same four
chunks for heartbeat budget. -/

/-- General chunk slices of `bwdInnerBodyG`. -/
private noncomputable def bwdInnerChunk1G (sc : ℝ) (BM BD : Nat) : List Stmt :=
  (bwdInnerBodyG sc BM BD).take 5
private noncomputable def bwdInnerChunk2G (sc : ℝ) (BM BD : Nat) : List Stmt :=
  ((bwdInnerBodyG sc BM BD).drop 5).take 5
private noncomputable def bwdInnerChunk3G (sc : ℝ) (BM BD : Nat) : List Stmt :=
  ((bwdInnerBodyG sc BM BD).drop 10).take 5
private noncomputable def bwdInnerChunk4G (sc : ℝ) (BM BD : Nat) : List Stmt :=
  (bwdInnerBodyG sc BM BD).drop 15

private theorem bwdInnerBody_chunksG (sc : ℝ) (BM BD : Nat) :
    bwdInnerBodyG sc BM BD
      = bwdInnerChunk1G sc BM BD ++ bwdInnerChunk2G sc BM BD
        ++ bwdInnerChunk3G sc BM BD ++ bwdInnerChunk4G sc BM BD := by
  unfold bwdInnerChunk1G bwdInnerChunk2G bwdInnerChunk3G bwdInnerChunk4G
  rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General chunk 1 step.** -/
private theorem bwdInnerChunk1_stepG (s sin : BlockState) (Q K M : RegionName)
    (BM BD D0 NCTX qRow kRow : Nat) (sc : ℝ)
    (hbdvd : BD ∣ bwdKBase s)
    (hrow : ∀ i : Fin BM, bwdKBase s / BD + qRow + i.val < D0)
    (hmem : sin.mem = s.mem)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar qRow))
    (hom : sin.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => i.val)))
    (hon : sin.regs .nat [BM] "offs_n" = some (Tile.vec (fun j : Fin BM => kRow + j.val)))
    (hqptr : sin.regs .blockPtr [BM, BD] "q_tile_ptr"
      = some (bwdPtrTileG Q (bwdKBase s) D0 BM BD qRow))
    (hk : sin.regs .real [BM, BD] "k"
      = some (⟨fun ij : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + ij.1.val) ij.2.1.val)⟩))
    (hmptr : sin.regs .ptr [] "m_ptrs" = some (Tile.scalar (M, s.pids 0 * NCTX))) :
    ∃ s1, stepStmts (bwdInnerChunk1G sc BM BD) sin = some s1
      ∧ s1.pids = sin.pids ∧ s1.mem = sin.mem ∧ s1.undef = sin.undef
      ∧ s1.regs .nat [BM] "offs_m_curr" = some (Tile.vec (fun i : Fin BM => qRow + i.val))
      ∧ s1.regs .real [BM, BD] "q"
          = some (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelQG s Q BD (qRow + idx.1.val) idx.2.1.val)⟩)
      ∧ s1.regs .real [BM, BM] "qk" = some (bwdInnerQk1G s Q K BM BD qRow kRow)
      ∧ s1.regs .real [BM] "m"
          = some (⟨fun i : TileIndex [BM] => some (bwdKernelMG s M NCTX (qRow + i.1.val))⟩)
      ∧ (∀ {dt sh nm}, nm ≠ "offs_m_curr" → nm ≠ "q" → nm ≠ "qk" → nm ≠ "m" →
          s1.regs dt sh nm = sin.regs dt sh nm) := by
  unfold bwdInnerChunk1G bwdInnerBodyG
  simp only [List.take]
  have hvecmc : evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m")
      (Op.ref .nat [BM] "offs_m")) sin = some (Tile.vec (fun i : Fin BM => qRow + i.val)) := by
    rw [bwd_offsmcurr_eval sin BM qRow (fun i : Fin BM => i.val) hsm hom]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hvecmc)]
  set s0 := sin.setReg "offs_m_curr" .nat [BM] (Tile.vec (fun i : Fin BM => qRow + i.val)) with hs0
  have homc0 : s0.regs .nat [BM] "offs_m_curr" = some (Tile.vec (fun i : Fin BM => qRow + i.val)) := by
    rw [hs0]; simp only [BlockState.setReg_same]
  -- q load
  rw [stepStmts.cons_some (show stepStmt _ s0
      = some (s0.setReg "q" .real [BM, BD]
          (⟨fun idx : TileIndex [BM, BD] =>
            some (s.readMem Q (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩)) from
    stepStmt_assign_eq_some
      (bwd_load_bc_evalG Q (bwdKBase s) D0 BM BD qRow (Op.ref .blockPtr [BM, BD] "q_tile_ptr") s s0
        hbdvd (by rw [hs0]; exact hmem) hrow
        (by rw [evalOp_ref, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hqptr)))]
  set s1 := s0.setReg "q" .real [BM, BD]
    (⟨fun idx : TileIndex [BM, BD] =>
      some (s.readMem Q (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩) with hs1
  -- qk dot
  rw [stepStmts.cons_some (show stepStmt _ s1
      = some (s1.setReg "qk" .real [BM, BM] (bwdInnerQkG s Q K BM BD qRow kRow)) from
    stepStmt_assign_eq_some (by
      rw [bwd_dot_trans_evalG s1 BM BD "q" "k"
        (⟨fun idx : TileIndex [BM, BD] => some (s.readMem Q (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩)
        (⟨fun ij : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + ij.1.val) ij.2.1.val)⟩)
        (by rw [hs1]; simp only [BlockState.setReg_same])
        (by rw [hs1, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hk)]
      rfl))]
  set s2 := s1.setReg "qk" .real [BM, BM] (bwdInnerQkG s Q K BM BD qRow kRow) with hs2
  -- qk where-mask
  rw [stepStmts.cons_some (show stepStmt _ s2
      = some (s2.setReg "qk" .real [BM, BM] (bwdInnerQk1G s Q K BM BD qRow kRow)) from
    stepStmt_assign_eq_some (by
      have := bwd_where_evalG s2 BM BM (fun i : Fin BM => qRow + i.val) (fun j : Fin BM => kRow + j.val)
        (bwdInnerQkG s Q K BM BD qRow kRow)
        (by rw [hs2, hs1];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact homc0)
        (by rw [hs2, hs1, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hon)
        (by rw [hs2]; simp only [BlockState.setReg_same])
      rw [this]; refine congrArg some ?_; ext idx; simp only [bwdInnerQk1G]))]
  set s3 := s2.setReg "qk" .real [BM, BM] (bwdInnerQk1G s Q K BM BD qRow kRow) with hs3
  have hs3mem : ∀ R o, s3.readMem R o = s.readMem R o := by
    intro R o; rw [hs3, hs2, hs1, hs0]
    simp only [BlockState.readMem, BlockState.setReg_mem, hmem]
  -- m load
  rw [stepStmts.cons_some (show stepStmt _ s3
      = some (s3.setReg "m" .real [BM]
          (⟨fun i : TileIndex [BM] => some (bwdKernelMG s M NCTX (qRow + i.1.val))⟩)) from
    stepStmt_assign_eq_some (by
      have hml := bwd_ptr_load_eval s3 BM "m_ptrs" M (s.pids 0 * NCTX)
        (fun i : Fin BM => qRow + i.val)
        (by rw [hs3, hs2, hs1, hs0];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hmptr)
        (by rw [hs3, hs2, hs1];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact homc0)
      rw [hml]; refine congrArg some ?_; ext i
      simp only [bwdKernelMG, hs3mem M (s.pids 0 * NCTX + (qRow + i.1.val))])),
    stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact homc0
  · rw [hs3, hs2]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [hs1]; simp only [BlockState.setReg_same]
    refine congrArg some ?_; ext idx; simp only [bwdKernelQG]
  · rw [hs3]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  · simp only [BlockState.setReg_same]
  · intro dt sh nm h1 h2 h3 h4
    rw [hs3, hs2, hs1, hs0]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General chunk 2 step.** Steps `p`, `do_val`, `dv`-accumulate (onto prior
`accDv`), `Di`, `dp`-init. -/
private theorem bwdInnerChunk2_stepG (s sin : BlockState) (Q K M DO Delta : RegionName)
    (BM BD D0 NCTX qRow kRow : Nat) (sc : ℝ) (accDv : Tile .real [BM, BD])
    (hbdvd : BD ∣ bwdKBase s)
    (hrow : ∀ i : Fin BM, bwdKBase s / BD + qRow + i.val < D0)
    (hmem : sin.mem = s.mem)
    (homc : sin.regs .nat [BM] "offs_m_curr" = some (Tile.vec (fun i : Fin BM => qRow + i.val)))
    (hqk : sin.regs .real [BM, BM] "qk" = some (bwdInnerQk1G s Q K BM BD qRow kRow))
    (hm : sin.regs .real [BM] "m"
      = some (⟨fun i : TileIndex [BM] => some (bwdKernelMG s M NCTX (qRow + i.1.val))⟩))
    (hdoptr : sin.regs .blockPtr [BM, BD] "do_tile_ptr"
      = some (bwdPtrTileG DO (bwdKBase s) D0 BM BD qRow))
    (hdv : sin.regs .real [BM, BD] "dv" = some accDv)
    (hDptr : sin.regs .ptr [] "D_ptrs" = some (Tile.scalar (Delta, s.pids 0 * NCTX))) :
    ∃ s2, stepStmts (bwdInnerChunk2G sc BM BD) sin = some s2
      ∧ s2.pids = sin.pids ∧ s2.mem = sin.mem ∧ s2.undef = sin.undef
      ∧ s2.regs .real [BM, BM] "p" = some (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc)
      ∧ s2.regs .real [BM, BD] "do_val"
          = some (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelDOG s DO BD (qRow + idx.1.val) idx.2.1.val)⟩)
      ∧ s2.regs .real [BM, BD] "dv" = some (bwdInnerDvG s Q K M DO BM BD NCTX qRow kRow sc accDv)
      ∧ s2.regs .real [BM, BM] "dp"
          = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (⟨fun _ : TileIndex [BM, BM] => some (0:ℝ)⟩ : Tile .real [BM, BM])
              (Tile.expandDim ⟨1, by simp⟩
                (⟨fun i : TileIndex [BM] => some (bwdKernelDiG s Delta NCTX (qRow + i.1.val))⟩ : Tile .real [BM])))
      ∧ (∀ {dt sh nm}, nm ≠ "p" → nm ≠ "do_val" → nm ≠ "dv" → nm ≠ "Di" → nm ≠ "dp" →
          s2.regs dt sh nm = sin.regs dt sh nm) := by
  unfold bwdInnerChunk2G bwdInnerBodyG
  simp only [List.drop, List.take]
  -- p = exp(qk·sc − m[:,None])
  rw [stepStmts.cons_some (show stepStmt _ sin
      = some (sin.setReg "p" .real [BM, BM] (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc)) from
    stepStmt_assign_eq_some (by
      rw [bwd_p_eval sin BM BM (by simp) sc (bwdInnerQk1G s Q K BM BD qRow kRow)
        (⟨fun i : TileIndex [BM] => some (bwdKernelMG s M NCTX (qRow + i.1.val))⟩) hqk hm]; rfl))]
  set s_p := sin.setReg "p" .real [BM, BM] (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc) with hs_p
  -- do_val load
  rw [stepStmts.cons_some (show stepStmt _ s_p
      = some (s_p.setReg "do_val" .real [BM, BD]
          (⟨fun idx : TileIndex [BM, BD] =>
            some (s.readMem DO (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩)) from
    stepStmt_assign_eq_some
      (bwd_load_bc_evalG DO (bwdKBase s) D0 BM BD qRow (Op.ref .blockPtr [BM, BD] "do_tile_ptr") s s_p
        hbdvd (by rw [hs_p]; exact hmem) hrow
        (by rw [evalOp_ref, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hdoptr)))]
  set s_do := s_p.setReg "do_val" .real [BM, BD]
    (⟨fun idx : TileIndex [BM, BD] =>
      some (s.readMem DO (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩) with hs_do
  -- dv accumulate
  rw [stepStmts.cons_some (show stepStmt _ s_do
      = some (s_do.setReg "dv" .real [BM, BD] (bwdInnerDvG s Q K M DO BM BD NCTX qRow kRow sc accDv)) from
    stepStmt_assign_eq_some
      (bwd_trans_fp16_dot_evalG s_do BM BD "dv" "p" "do_val"
        accDv
        (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc)
        (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelDOG s DO BD (qRow + idx.1.val) idx.2.1.val)⟩)
        (by rw [hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hdv)
        (by rw [hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
              not_false_eq_true])
        (by rw [hs_do]; simp only [BlockState.setReg_same]
            refine congrArg some ?_; ext idx; simp only [bwdKernelDOG])))]
  set s_dv := s_do.setReg "dv" .real [BM, BD] (bwdInnerDvG s Q K M DO BM BD NCTX qRow kRow sc accDv) with hs_dv
  have hs_dvmem : ∀ R o, s_dv.readMem R o = s.readMem R o := by
    intro R o; rw [hs_dv, hs_do, hs_p]
    simp only [BlockState.readMem, BlockState.setReg_mem, hmem]
  -- Di load
  rw [stepStmts.cons_some (show stepStmt _ s_dv
      = some (s_dv.setReg "Di" .real [BM]
          (⟨fun i : TileIndex [BM] => some (s.readMem Delta (s.pids 0 * NCTX + (qRow + i.1.val)))⟩)) from
    stepStmt_assign_eq_some (by
      have hdil := bwd_ptr_load_eval s_dv BM "D_ptrs" Delta (s.pids 0 * NCTX)
        (fun i : Fin BM => qRow + i.val)
        (by rw [hs_dv, hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact hDptr)
        (by rw [hs_dv, hs_do, hs_p];
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            exact homc)
      rw [hdil]; refine congrArg some ?_; ext i
      simp only [hs_dvmem Delta (s.pids 0 * NCTX + (qRow + i.1.val))]))]
  set s_Di := s_dv.setReg "Di" .real [BM]
    (⟨fun i : TileIndex [BM] => some (s.readMem Delta (s.pids 0 * NCTX + (qRow + i.1.val)))⟩) with hs_Di
  -- dp init
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bwd_dpinit_eval s_Di BM BM (by simp)
      (⟨fun i : TileIndex [BM] => some (s.readMem Delta (s.pids 0 * NCTX + (qRow + i.1.val)))⟩)
      (by rw [hs_Di]; simp only [BlockState.setReg_same]))), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · rw [hs_Di, hs_dv, hs_do]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hs_p]; simp only [BlockState.setReg_same]
  · rw [hs_Di, hs_dv]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hs_do]; simp only [BlockState.setReg_same]
    refine congrArg some ?_; ext idx; simp only [bwdKernelDOG]
  · rw [hs_Di]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hs_dv]; simp only [BlockState.setReg_same]
  · simp only [BlockState.setReg_same]
    refine congrArg some ?_
    refine congrArg _ ?_
    refine congrArg _ ?_
    ext i; simp only [bwdKernelDiG]
  · intro dt sh nm h1 h2 h3 h4 h5
    rw [hs_Di, hs_dv, hs_do, hs_p]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h5,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General chunk 3 step.** Steps `dp`-dot, `ds`, `dk`-accumulate (onto prior
`accDk`), `dq` load, `dq`-accumulate. -/
private theorem bwdInnerChunk3_stepG (s sin : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD D0 NCTX qRow kRow : Nat) (sc : ℝ) (accDk : Tile .real [BM, BD])
    (hbdvd : BD ∣ bwdKBase s)
    (hmem : sin.mem = s.mem)
    (hp : sin.regs .real [BM, BM] "p" = some (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc))
    (hdoval : sin.regs .real [BM, BD] "do_val"
      = some (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelDOG s DO BD (qRow + idx.1.val) idx.2.1.val)⟩))
    (hdp : sin.regs .real [BM, BM] "dp"
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (⟨fun _ : TileIndex [BM, BM] => some (0:ℝ)⟩ : Tile .real [BM, BM])
          (Tile.expandDim ⟨1, by simp⟩
            (⟨fun i : TileIndex [BM] => some (bwdKernelDiG s Delta NCTX (qRow + i.1.val))⟩ : Tile .real [BM]))))
    (hdk : sin.regs .real [BM, BD] "dk" = some accDk)
    (hq : sin.regs .real [BM, BD] "q"
      = some (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelQG s Q BD (qRow + idx.1.val) idx.2.1.val)⟩))
    (hv : sin.regs .real [BM, BD] "v"
      = some (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelVG s V BD (kRow + idx.1.val) idx.2.1.val)⟩))
    (hk : sin.regs .real [BM, BD] "k"
      = some (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + idx.1.val) idx.2.1.val)⟩))
    (hdqptr : sin.regs .blockPtr [BM, BD] "dq_tile_ptr"
      = some (bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow)) :
    ∃ s3, stepStmts (bwdInnerChunk3G sc BM BD) sin = some s3
      ∧ s3.pids = sin.pids ∧ s3.mem = sin.mem ∧ s3.undef = sin.undef
      ∧ s3.regs .real [BM, BD] "dk" = some (bwdInnerDkG s Q K V DO M Delta BM BD NCTX qRow kRow sc accDk)
      ∧ s3.regs .real [BM, BD] "dq" = some (bwdInnerDqG s Q K V DO M Delta DQ BM BD NCTX qRow kRow sc)
      ∧ s3.regs .blockPtr [BM, BD] "dq_tile_ptr" = some (bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow)
      ∧ (∀ {dt sh nm}, nm ≠ "dp" → nm ≠ "ds" → nm ≠ "dk" → nm ≠ "dq" →
          s3.regs dt sh nm = sin.regs dt sh nm) := by
  unfold bwdInnerChunk3G bwdInnerBodyG
  simp only [List.drop, List.take]
  -- dp = dp + dot(do, trans(v))  (defeq bwdInnerDpG)
  rw [stepStmts.cons_some (show stepStmt _ sin
      = some (sin.setReg "dp" .real [BM, BM] (bwdInnerDpG s V DO Delta BM BD NCTX qRow kRow)) from
    stepStmt_assign_eq_some
      (bwd_dpdot_evalG sin BM BD
        (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (⟨fun _ : TileIndex [BM, BM] => some (0:ℝ)⟩ : Tile .real [BM, BM])
          (Tile.expandDim ⟨1, by simp⟩
            (⟨fun i : TileIndex [BM] => some (bwdKernelDiG s Delta NCTX (qRow + i.1.val))⟩ : Tile .real [BM])))
        (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelDOG s DO BD (qRow + idx.1.val) idx.2.1.val)⟩)
        (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelVG s V BD (kRow + idx.1.val) idx.2.1.val)⟩)
        hdp hdoval hv))]
  set s1 := sin.setReg "dp" .real [BM, BM] (bwdInnerDpG s V DO Delta BM BD NCTX qRow kRow) with hs1
  -- ds = (p · dp) · sc  (defeq bwdInnerDsG)
  rw [stepStmts.cons_some (show stepStmt _ s1
      = some (s1.setReg "ds" .real [BM, BM] (bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc)) from
    stepStmt_assign_eq_some
      (bwd_ds_evalG s1 BM sc (bwdInnerPG s Q K M BM BD NCTX qRow kRow sc) (bwdInnerDpG s V DO Delta BM BD NCTX qRow kRow)
        (by rw [hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hp)
        (by rw [hs1]; simp only [BlockState.setReg_same])))]
  set s2 := s1.setReg "ds" .real [BM, BM] (bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc) with hs2
  -- dk accumulate
  rw [stepStmts.cons_some (show stepStmt _ s2
      = some (s2.setReg "dk" .real [BM, BD] (bwdInnerDkG s Q K V DO M Delta BM BD NCTX qRow kRow sc accDk)) from
    stepStmt_assign_eq_some
      (bwd_trans_fp16_dot_evalG s2 BM BD "dk" "ds" "q"
        accDk
        (bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc)
        (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelQG s Q BD (qRow + idx.1.val) idx.2.1.val)⟩)
        (by rw [hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hdk)
        (by rw [hs2]; simp only [BlockState.setReg_same])
        (by rw [hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hq)))]
  set s3 := s2.setReg "dk" .real [BM, BD] (bwdInnerDkG s Q K V DO M Delta BM BD NCTX qRow kRow sc accDk) with hs3
  -- dq load
  rw [stepStmts.cons_some (show stepStmt _ s3
      = some (s3.setReg "dq" .real [BM, BD]
          (⟨fun idx : TileIndex [BM, BD] =>
            some (s.readMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩)) from
    stepStmt_assign_eq_some
      (bwd_load_nobc_evalG DQ (bwdKBase s) D0 BM BD qRow (Op.ref .blockPtr [BM, BD] "dq_tile_ptr") s s3
        hbdvd (by rw [hs3, hs2, hs1]; exact hmem)
        (by rw [evalOp_ref, hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hdqptr)))]
  set s4 := s3.setReg "dq" .real [BM, BD]
    (⟨fun idx : TileIndex [BM, BD] =>
      some (s.readMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩) with hs4
  -- dq accumulate
  rw [stepStmts.cons_some (show stepStmt _ s4
      = some (s4.setReg "dq" .real [BM, BD] (bwdInnerDqG s Q K V DO M Delta DQ BM BD NCTX qRow kRow sc)) from
    stepStmt_assign_eq_some
      (bwd_fp16_dot_evalG s4 BM BD "dq" "ds" "k"
        (⟨fun idx : TileIndex [BM, BD] =>
          some (s.readMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val))⟩)
        (bwdInnerDsG s Q K V DO M Delta BM BD NCTX qRow kRow sc)
        (⟨fun idx : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + idx.1.val) idx.2.1.val)⟩)
        (by rw [hs4]; simp only [BlockState.setReg_same])
        (by rw [hs4, hs3]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
            rw [hs2]; simp only [BlockState.setReg_same])
        (by rw [hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hk))),
    stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · rw [hs4, hs3]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true]
  · rw [hs4]; simp only [BlockState.setReg_same]
  · rw [hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hdqptr
  · intro dt sh nm h1 h2 h3 h4
    rw [hs4, hs3, hs2, hs1]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General chunk 4 step.** Steps the `dq` store and the four pointer advances.
The final `DQ` memory holds `dqSpec` at each lane; memory in any other region is
unchanged; `q`/`do`/`dq` block pointers continue to exist; all other registers are
preserved. -/
private theorem bwdInnerChunk4_stepG (s sin : BlockState) (Q DO DQ : RegionName)
    (BM BD D0 qRow : Nat) (sc : ℝ)
    (hbdvd : BD ∣ bwdKBase s)
    (dqt : Tile .real [BM, BD]) (dqSpec : TileIndex [BM, BD] → ℝ)
    (hmem : sin.mem = s.mem)
    (hdq : sin.regs .real [BM, BD] "dq" = some dqt)
    (hdqcell : ∀ idx : TileIndex [BM, BD], (dqt.data idx).unbotD 0 = dqSpec idx)
    (hdqptr : sin.regs .blockPtr [BM, BD] "dq_tile_ptr" = some (bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow))
    (hdqptrs : ∃ t, sin.regs .ptr [BM, BD] "dq_ptrs" = some t)
    (hqptr : sin.regs .blockPtr [BM, BD] "q_tile_ptr" = some (bwdPtrTileG Q (bwdKBase s) D0 BM BD qRow))
    (hdoptr : sin.regs .blockPtr [BM, BD] "do_tile_ptr" = some (bwdPtrTileG DO (bwdKBase s) D0 BM BD qRow)) :
    ∃ s4, stepStmts (bwdInnerChunk4G sc BM BD) sin = some s4
      ∧ s4.pids = sin.pids ∧ s4.undef = sin.undef
      ∧ (∀ idx : TileIndex [BM, BD],
          s4.readMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val) = dqSpec idx)
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → s4.mem R o = sin.mem R o)
      ∧ (∀ o : Nat, (∀ idx : TileIndex [BM, BD],
            o ≠ bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val) →
          s4.readMem DQ o = sin.readMem DQ o)
      ∧ s4.regs .blockPtr [BM, BD] "q_tile_ptr" = some (bwdPtrTileG Q (bwdKBase s) D0 BM BD (qRow + BM))
      ∧ s4.regs .blockPtr [BM, BD] "do_tile_ptr" = some (bwdPtrTileG DO (bwdKBase s) D0 BM BD (qRow + BM))
      ∧ s4.regs .blockPtr [BM, BD] "dq_tile_ptr" = some (bwdPtrTileG DQ (bwdKBase s) D0 BM BD (qRow + BM))
      ∧ (∃ tp, s4.regs .ptr [BM, BD] "dq_ptrs" = some tp)
      ∧ (∀ {dt sh nm}, nm ≠ "dq" → nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
          nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" → s4.regs dt sh nm = sin.regs dt sh nm) := by
  unfold bwdInnerChunk4G bwdInnerBodyG
  simp only [List.drop]
  obtain ⟨tdqptrs, htdqptrs⟩ := hdqptrs
  set tqp := bwdPtrTileG Q (bwdKBase s) D0 BM BD qRow with htqpdef
  set tdop := bwdPtrTileG DO (bwdKBase s) D0 BM BD qRow with htdopdef
  have htqp : sin.regs .blockPtr [BM, BD] "q_tile_ptr" = some tqp := hqptr
  have htdop : sin.regs .blockPtr [BM, BD] "do_tile_ptr" = some tdop := hdoptr
  -- store
  rw [stepStmts.cons_some (bwd_store_dq_evalG s sin DQ (bwdKBase s) D0 BM BD qRow hbdvd dqt hdqptr hdq)]
  set sStore := (TileShape.allIndices [BM, BD]).foldl
    (fun acc idx => acc.writeMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val)
      ((dqt.data idx).unbotD 0)) sin with hsStore
  have hStpids : sStore.pids = sin.pids := by
    rw [hsStore]; exact foldl_writeMem_pids_eq _ _ _ _ sin
  have hStundef : sStore.undef = sin.undef := by
    rw [hsStore]; funext R o; exact foldl_writeMem_undef DQ _ _ _ sin R o
  have hStregs : ∀ {dt sh nm}, sStore.regs dt sh nm = sin.regs dt sh nm := by
    intro dt sh nm; rw [hsStore]; exact BlockState.foldl_writeMem_regs _ _ _ _ sin dt sh nm
  have hStDQ : ∀ idx : TileIndex [BM, BD],
      sStore.readMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val) = dqSpec idx := by
    intro idx
    rw [hsStore, BlockState.readMem]
    have hinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
        bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val) := by
      rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
      simp only at h
      have hkey : (qRow + ma) * BD + da = (qRow + mb) * BD + db := by omega
      have hda' : da < BD := hda
      have hdb' : db < BD := hdb
      have hmod1 : ((qRow + ma) * BD + da) % BD = da := by
        rw [Nat.mul_add_mod']; exact Nat.mod_eq_of_lt hda'
      have hmod2 : ((qRow + mb) * BD + db) % BD = db := by
        rw [Nat.mul_add_mod']; exact Nat.mod_eq_of_lt hdb'
      have hdaeq : da = db := by rw [← hmod1, ← hmod2, hkey]
      obtain rfl : da = db := hdaeq
      have hmeq : (qRow + ma) * BD = (qRow + mb) * BD := by omega
      obtain rfl : ma = mb := by
        have := Nat.eq_of_mul_eq_mul_right (by omega : 0 < BD) hmeq
        omega
      rfl
    have hsc := BlockState.scatter_readback_nd (region := DQ) sin
      (fun idx : TileIndex [BM, BD] => bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val)
      (fun idx : TileIndex [BM, BD] => (dqt.data idx).unbotD 0) hinj idx
    simp only [BlockState.readMem] at hsc
    rw [hsc]; exact hdqcell idx
  have hStother : ∀ (R : RegionName) (o : Nat), R ≠ DQ → sStore.mem R o = sin.mem R o := by
    intro R o hRne; rw [hsStore]; exact foldl_writeMem_other_region DQ _ _ _ sin R o hRne
  have hStdisjoint : ∀ o : Nat, (∀ idx : TileIndex [BM, BD],
        o ≠ bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val) →
      sStore.readMem DQ o = sin.readMem DQ o := by
    intro o hoff
    unfold BlockState.readMem
    rw [hsStore]
    rw [foldl_writeMem_disjoint_offset DQ
      (fun idx : TileIndex [BM, BD] => bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val)
      (fun idx : TileIndex [BM, BD] => (dqt.data idx).unbotD 0)
      (TileShape.allIndices [BM, BD]) sin o (fun k _ => hoff k)]
  -- advances: dq_ptrs, q, do, dq_tile_ptr
  have htdqptrsSt : sStore.regs .ptr [BM, BD] "dq_ptrs" = some tdqptrs := hStregs.trans htdqptrs
  have htqpSt : sStore.regs .blockPtr [BM, BD] "q_tile_ptr" = some tqp := hStregs.trans htqp
  have htdopSt : sStore.regs .blockPtr [BM, BD] "do_tile_ptr" = some tdop := hStregs.trans htdop
  have htdqtpSt : sStore.regs .blockPtr [BM, BD] "dq_tile_ptr"
      = some (bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow) := hStregs.trans hdqptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BD] "dq_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BM) (Op.constNat BD))) sStore
      = some (Tile.ptrAdd Broadcast.scalarR tdqptrs (Tile.scalar (BM * BD))) from by
      rw [evalOp_ptrAdd]; simp only [evalOp_ref, htdqptrsSt, evalOp_mul, evalOp_constNat,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  set sA := sStore.setReg "dq_ptrs" .ptr [BM, BD]
    (Tile.ptrAdd Broadcast.scalarR tdqptrs (Tile.scalar (BM * BD))) with hsA
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "q_tile_ptr") [BM, (0:Nat)]) sA
      = some ⟨fun i => (tqp.data i).advance [BM, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htqpSt, Option.bind_eq_bind, Option.bind_some, Nat.cast_zero]))]
  set sB := sA.setReg "q_tile_ptr" .blockPtr [BM, BD] ⟨fun i => (tqp.data i).advance [BM, 0]⟩ with hsB
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "do_tile_ptr") [BM, (0:Nat)]) sB
      = some ⟨fun i => (tdop.data i).advance [BM, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsB, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdopSt, Option.bind_eq_bind, Option.bind_some, Nat.cast_zero]))]
  set sC := sB.setReg "do_tile_ptr" .blockPtr [BM, BD] ⟨fun i => (tdop.data i).advance [BM, 0]⟩ with hsC
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "dq_tile_ptr") [BM, (0:Nat)]) sC
      = some ⟨fun i => ((bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow).data i).advance [BM, 0]⟩ from by
      rw [advanceBlockPtr_eval, hsC, hsB, hsA]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        htdqtpSt, Option.bind_eq_bind, Option.bind_some, Nat.cast_zero])), stepStmts.nil]
  set sD := sC.setReg "dq_tile_ptr" .blockPtr [BM, BD]
    ⟨fun i => ((bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow).data i).advance [BM, 0]⟩ with hsD
  have hmemD : ∀ R o, sD.mem R o = sStore.mem R o := by
    intro R o; rw [hsD, hsC, hsB, hsA]; simp only [BlockState.setReg_mem]
  refine ⟨sD, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [show sD.pids = sStore.pids from by rw [hsD, hsC, hsB, hsA]; rfl]; exact hStpids
  · rw [show sD.undef = sStore.undef from by rw [hsD, hsC, hsB, hsA]; rfl]; exact hStundef
  · intro idx
    rw [BlockState.readMem, hmemD, ← BlockState.readMem, hStDQ idx]
  · intro R o hRne
    rw [hmemD, hStother R o hRne]
  · intro o hoff
    rw [show sD.readMem DQ o = sStore.readMem DQ o from by
      unfold BlockState.readMem; rw [hmemD], hStdisjoint o hoff]
  · rw [show sD.regs .blockPtr [BM, BD] "q_tile_ptr"
          = some (⟨fun i => (tqp.data i).advance [BM, 0]⟩ : Tile .blockPtr [BM, BD]) from by
        simp only [hsD, hsC, hsB, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, BlockState.setReg_same],
      htqpdef, bwdPtrTileG_advance]
  · rw [show sD.regs .blockPtr [BM, BD] "do_tile_ptr"
          = some (⟨fun i => (tdop.data i).advance [BM, 0]⟩ : Tile .blockPtr [BM, BD]) from by
        simp only [hsD, hsC, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, BlockState.setReg_same],
      htdopdef, bwdPtrTileG_advance]
  · rw [show sD.regs .blockPtr [BM, BD] "dq_tile_ptr"
          = some (⟨fun i => ((bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow).data i).advance [BM, 0]⟩
              : Tile .blockPtr [BM, BD]) from by
        rw [hsD, BlockState.setReg_same],
      bwdPtrTileG_advance]
  · refine ⟨Tile.ptrAdd Broadcast.scalarR tdqptrs (Tile.scalar (BM * BD)), ?_⟩
    rw [show sD.regs .ptr [BM, BD] "dq_ptrs" = sA.regs .ptr [BM, BD] "dq_ptrs" from by
        rw [hsD, hsC, hsB]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true],
      hsA, BlockState.setReg_same]
  · intro dt sh nm h1 h2 h3 h4 h5
    rw [hsD, hsC, hsB, hsA]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h5,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2]
    exact hStregs

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General inner-body step.** From a state at counter `start_m = qRow` (the
current query block) and fixed key block base `kRow`, exposing the loaded `k`/`v`
tiles, the `q`/`do`/`dq` block pointers at row `qRow`, the `m`/`D` pointers, and
arbitrary prior `dv`/`dk` accumulators `accDv`/`accDk`, stepping `bwdInnerBodyG`
reaches a state whose `dv`/`dk` registers hold the block-`qRow` accumulation
(`bwdInnerDvG`/`bwdInnerDkG` onto the priors), whose `DQ` memory holds the
genuine `bwdInnerDqG` cell value, and which preserves `pids`/`undef`, `mem` off
`DQ`, and every register not assigned by the body (in particular `k`/`v` and the
KV/grad block pointers).  Dimension-general inner Q-block step. -/
private theorem bwd_inner_stepG (s sin : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD D0 NCTX qRow kRow : Nat) (sc : ℝ)
    (accDv accDk : Tile .real [BM, BD])
    (hbdvd : BD ∣ bwdKBase s)
    (hrow : ∀ i : Fin BM, bwdKBase s / BD + qRow + i.val < D0)
    (hmem : sin.mem = s.mem)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar qRow))
    (hom : sin.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => i.val)))
    (hon : sin.regs .nat [BM] "offs_n" = some (Tile.vec (fun j : Fin BM => kRow + j.val)))
    (hqptr : sin.regs .blockPtr [BM, BD] "q_tile_ptr" = some (bwdPtrTileG Q (bwdKBase s) D0 BM BD qRow))
    (hdoptr : sin.regs .blockPtr [BM, BD] "do_tile_ptr" = some (bwdPtrTileG DO (bwdKBase s) D0 BM BD qRow))
    (hdqptr : sin.regs .blockPtr [BM, BD] "dq_tile_ptr" = some (bwdPtrTileG DQ (bwdKBase s) D0 BM BD qRow))
    (hk : sin.regs .real [BM, BD] "k"
      = some (⟨fun ij : TileIndex [BM, BD] => some (bwdKernelKG s K BD (kRow + ij.1.val) ij.2.1.val)⟩))
    (hv : sin.regs .real [BM, BD] "v"
      = some (⟨fun ij : TileIndex [BM, BD] => some (bwdKernelVG s V BD (kRow + ij.1.val) ij.2.1.val)⟩))
    (hdv : sin.regs .real [BM, BD] "dv" = some accDv)
    (hdk : sin.regs .real [BM, BD] "dk" = some accDk)
    (hmptr : sin.regs .ptr [] "m_ptrs" = some (Tile.scalar (M, s.pids 0 * NCTX)))
    (hDptr : sin.regs .ptr [] "D_ptrs" = some (Tile.scalar (Delta, s.pids 0 * NCTX)))
    (hdqptrs : ∃ t, sin.regs .ptr [BM, BD] "dq_ptrs" = some t) :
    ∃ sF, stepStmts (bwdInnerBodyG sc BM BD) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.undef = sin.undef
      ∧ sF.regs .real [BM, BD] "dv" = some (bwdInnerDvG s Q K M DO BM BD NCTX qRow kRow sc accDv)
      ∧ sF.regs .real [BM, BD] "dk" = some (bwdInnerDkG s Q K V DO M Delta BM BD NCTX qRow kRow sc accDk)
      ∧ (∀ idx : TileIndex [BM, BD],
          sF.readMem DQ (bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val)
            = WithBot.unbotD 0 ((bwdInnerDqG s Q K V DO M Delta DQ BM BD NCTX qRow kRow sc).data idx))
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → sF.mem R o = sin.mem R o)
      ∧ (∀ o : Nat, (∀ idx : TileIndex [BM, BD],
            o ≠ bwdKBase s + (qRow + idx.1.val) * BD + idx.2.1.val) →
          sF.readMem DQ o = sin.readMem DQ o)
      ∧ sF.regs .blockPtr [BM, BD] "q_tile_ptr" = some (bwdPtrTileG Q (bwdKBase s) D0 BM BD (qRow + BM))
      ∧ sF.regs .blockPtr [BM, BD] "do_tile_ptr" = some (bwdPtrTileG DO (bwdKBase s) D0 BM BD (qRow + BM))
      ∧ sF.regs .blockPtr [BM, BD] "dq_tile_ptr" = some (bwdPtrTileG DQ (bwdKBase s) D0 BM BD (qRow + BM))
      ∧ (∃ tp, sF.regs .ptr [BM, BD] "dq_ptrs" = some tp)
      ∧ (∀ {dt sh nm}, nm ≠ "offs_m_curr" → nm ≠ "q" → nm ≠ "qk" → nm ≠ "m" →
          nm ≠ "p" → nm ≠ "do_val" → nm ≠ "Di" → nm ≠ "dp" → nm ≠ "ds" →
          nm ≠ "dk" → nm ≠ "dv" → nm ≠ "dq" → nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
          nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" →
          sF.regs dt sh nm = sin.regs dt sh nm) := by
  rw [bwdInnerBody_chunksG, List.append_assoc, List.append_assoc]
  obtain ⟨tdqptrs, htdqptrs⟩ := hdqptrs
  -- chunk 1
  obtain ⟨s1, hc1, hp1pids, hp1mem, hp1undef, h1omc, h1q, h1qk, h1m, h1rest⟩ :=
    bwdInnerChunk1_stepG s sin Q K M BM BD D0 NCTX qRow kRow sc hbdvd hrow hmem hsm hom hon hqptr hk hmptr
  rw [stepStmts.append_some hc1]
  have h1mem : s1.mem = s.mem := by rw [hp1mem, hmem]
  -- chunk 2
  obtain ⟨s2, hc2, hp2pids, hp2mem, hp2undef, h2p, h2do, h2dv, h2dp, h2rest⟩ :=
    bwdInnerChunk2_stepG s s1 Q K M DO Delta BM BD D0 NCTX qRow kRow sc accDv hbdvd hrow h1mem h1omc h1qk h1m
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdoptr)
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdv)
      (by rw [h1rest (by decide) (by decide) (by decide) (by decide)]; exact hDptr)
  rw [stepStmts.append_some hc2]
  have h2mem : s2.mem = s.mem := by rw [hp2mem, h1mem]
  -- chunk 3
  obtain ⟨s3, hc3, hp3pids, hp3mem, hp3undef, h3dk, h3dq, h3dqptr, h3rest⟩ :=
    bwdInnerChunk3_stepG s s2 Q K V DO M Delta DQ BM BD D0 NCTX qRow kRow sc accDk hbdvd h2mem h2p h2do h2dp
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
  obtain ⟨s4, hc4, hp4pids, hp4undef, h4dq, h4mem, h4disj, h4q, h4do, h4dqp, h4dqptrs, h4rest⟩ :=
    bwdInnerChunk4_stepG s s3 Q DO DQ BM BD D0 qRow sc hbdvd (bwdInnerDqG s Q K V DO M Delta DQ BM BD NCTX qRow kRow sc)
      (fun idx => WithBot.unbotD 0 ((bwdInnerDqG s Q K V DO M Delta DQ BM BD NCTX qRow kRow sc).data idx))
      h3mem h3dq (fun idx => rfl) h3dqptr
      ⟨tdqptrs, by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact htdqptrs⟩
      (by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hqptr)
      (by rw [h3rest (by decide) (by decide) (by decide) (by decide),
            h2rest (by decide) (by decide) (by decide) (by decide) (by decide),
            h1rest (by decide) (by decide) (by decide) (by decide)]; exact hdoptr)
  refine ⟨s4, hc4, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
  · intro o hoff
    rw [h4disj o hoff,
      show s3.readMem DQ o = sin.readMem DQ o from by
        unfold BlockState.readMem; rw [h3mem, ← hmem]]
  · exact h4q
  · exact h4do
  · exact h4dqp
  · exact h4dqptrs
  · intro dt sh nm h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16
    rw [h4rest h12 h13 h14 h15 h16,
      h3rest h8 h9 h10 h12,
      h2rest h5 h6 h11 h7 h8,
      h1rest h1 h2 h3 h4]

/-- **General inner-loop invariant** at counter `i` (key block `n` fixed,
`kRow = n·BM`).  Carries: the dv/dk accumulators at the running partial column
sums (`t = i/BM - n` query blocks processed), the q/do/dq block pointers at row
`i`, the fixed k/v tiles, the loop-invariant offs/ptr registers, the preserved
`pids`/`undef`, the input memory unchanged off `DQ`, and the running `DQ` memory:
visited query rows (blocks `[n, i/BM)`, i.e. global rows `n·BM ≤ I < i`) have
received this KV block's `bwdDqKeyContrib`, unvisited rows still hold the entry
value `s0`'s `DQ`. -/
noncomputable def bwdInnerInvariantG
    (s0 : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD D0 NCTX n num_block : Nat) (sc : ℝ)
    (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ i % BM = 0 ∧ n * BM ≤ i ∧ i ≤ num_block * BM ∧
  (∀ R o, R ≠ DQ → s.mem R o = s0.mem R o) ∧
  (∀ I e : Nat, I < num_block * BM → e < BD →
    s.readMem DQ (bwdKBase s0 + I * BD + e)
      = s0.readMem DQ (bwdKBase s0 + I * BD + e)
        + (if n * BM ≤ I ∧ I < i then bwdDqKeyContrib s0 Q K V DO M Delta BM BD NCTX sc n I e else 0)) ∧
  (∀ rg o, s.undef rg o = 0) ∧
  (s.regs .nat [BM] "offs_m" = some (Tile.vec (fun iL : Fin BM => iL.val))) ∧
  (s.regs .nat [BM] "offs_n" = some (Tile.vec (fun j : Fin BM => n * BM + j.val))) ∧
  (s.regs .blockPtr [BM, BD] "q_tile_ptr" = some (bwdPtrTileG Q (bwdKBase s0) D0 BM BD i)) ∧
  (s.regs .blockPtr [BM, BD] "do_tile_ptr" = some (bwdPtrTileG DO (bwdKBase s0) D0 BM BD i)) ∧
  (s.regs .blockPtr [BM, BD] "dq_tile_ptr" = some (bwdPtrTileG DQ (bwdKBase s0) D0 BM BD i)) ∧
  (s.regs .real [BM, BD] "k"
    = some (⟨fun ij : TileIndex [BM, BD] => some (bwdKernelKG s0 K BD (n * BM + ij.1.val) ij.2.1.val)⟩)) ∧
  (s.regs .real [BM, BD] "v"
    = some (⟨fun ij : TileIndex [BM, BD] => some (bwdKernelVG s0 V BD (n * BM + ij.1.val) ij.2.1.val)⟩)) ∧
  (s.regs .real [BM, BD] "dv"
    = some (⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDvCell s0 Q K M DO BM BD NCTX sc n (i / BM - n) idx.1.val idx.2.1.val)⟩)) ∧
  (s.regs .real [BM, BD] "dk"
    = some (⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDkCell s0 Q K V DO M Delta BM BD NCTX sc n (i / BM - n) idx.1.val idx.2.1.val)⟩)) ∧
  (s.regs .ptr [] "m_ptrs" = some (Tile.scalar (M, s0.pids 0 * NCTX))) ∧
  (s.regs .ptr [] "D_ptrs" = some (Tile.scalar (Delta, s0.pids 0 * NCTX))) ∧
  (∃ t, s.regs .ptr [BM, BD] "dq_ptrs" = some t)

/-- **`dv` accumulator recurrence.** At query block `n+t` (counter `(n+t)·BM`) and
KV block `n`, `bwdInnerDvG` folds the block-`(n+t)` sub-sum onto the running
accumulator tile, advancing `bwdAccDvCell` from `t` to `t+1`. -/
theorem bwdInnerDvG_accStep (s0 : BlockState) (Q K M DO : RegionName)
    (BM BD NCTX n t : Nat) (sc : ℝ) :
    bwdInnerDvG s0 Q K M DO BM BD NCTX ((n + t) * BM) (n * BM) sc
        (⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDvCell s0 Q K M DO BM BD NCTX sc n t idx.1.val idx.2.1.val)⟩)
      = (⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDvCell s0 Q K M DO BM BD NCTX sc n (t + 1) idx.1.val idx.2.1.val)⟩) := by
  refine Tile.ext (fun idx => ?_)
  obtain ⟨j, e, u⟩ := idx
  rw [bwdInnerDvG_cell]
  simp only [Option.map₂_some_some]
  rw [bwdAccDvCell_succ]
  refine congrArg some ?_
  congr 1

/-- **`dk` accumulator recurrence.** -/
theorem bwdInnerDkG_accStep (s0 : BlockState) (Q K V DO M Delta : RegionName)
    (BM BD NCTX n t : Nat) (sc : ℝ) :
    bwdInnerDkG s0 Q K V DO M Delta BM BD NCTX ((n + t) * BM) (n * BM) sc
        (⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDkCell s0 Q K V DO M Delta BM BD NCTX sc n t idx.1.val idx.2.1.val)⟩)
      = (⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDkCell s0 Q K V DO M Delta BM BD NCTX sc n (t + 1) idx.1.val idx.2.1.val)⟩) := by
  refine Tile.ext (fun idx => ?_)
  obtain ⟨j, e, u⟩ := idx
  rw [bwdInnerDkG_cell]
  simp only [Option.map₂_some_some]
  rw [bwdAccDkCell_succ]
  refine congrArg some ?_
  congr 1

/-! #### State-congruence of kernel/accumulator tiles off the `DQ` region.

When the inner loop steps the `DQ` store between iterations, the running state
`sCur` differs from the inner-loop entry state only in the `DQ` region (and in
loop registers).  All the `bwdKernel*G` loads (and hence the `dv`/`dk`/`dq`
tiles) read only `Q`/`K`/`V`/`DO`/`M`/`Delta` plus `pids` — never `DQ` — so they
are invariant under such state changes.  These congruences let the per-iteration
`bwd_inner_stepG` (driven with reference state `sCur`, so `hmem` is `rfl`) report
its results in the fixed entry-state vocabulary `s0`. -/

/-- `bwdKBase` depends only on `pids`. -/
theorem bwdKBase_congr {s s0 : BlockState} (hpids : s.pids = s0.pids) :
    bwdKBase s = bwdKBase s0 := by simp only [bwdKBase, hpids]

theorem bwdKernelKG_congr {s s0 : BlockState} (K : RegionName) (BD J e : Nat)
    (hpids : s.pids = s0.pids) (hK : s.mem K = s0.mem K) :
    bwdKernelKG s K BD J e = bwdKernelKG s0 K BD J e := by
  simp only [bwdKernelKG, BlockState.readMem, hK, bwdKBase_congr hpids]

theorem bwdKernelQG_congr {s s0 : BlockState} (Q : RegionName) (BD I e : Nat)
    (hpids : s.pids = s0.pids) (hQ : s.mem Q = s0.mem Q) :
    bwdKernelQG s Q BD I e = bwdKernelQG s0 Q BD I e := by
  simp only [bwdKernelQG, BlockState.readMem, hQ, bwdKBase_congr hpids]

theorem bwdKernelVG_congr {s s0 : BlockState} (V : RegionName) (BD J e : Nat)
    (hpids : s.pids = s0.pids) (hV : s.mem V = s0.mem V) :
    bwdKernelVG s V BD J e = bwdKernelVG s0 V BD J e := by
  simp only [bwdKernelVG, BlockState.readMem, hV, bwdKBase_congr hpids]

theorem bwdKernelDOG_congr {s s0 : BlockState} (DO : RegionName) (BD I e : Nat)
    (hpids : s.pids = s0.pids) (hDO : s.mem DO = s0.mem DO) :
    bwdKernelDOG s DO BD I e = bwdKernelDOG s0 DO BD I e := by
  simp only [bwdKernelDOG, BlockState.readMem, hDO, bwdKBase_congr hpids]

theorem bwdKernelMG_congr {s s0 : BlockState} (M : RegionName) (NCTX I : Nat)
    (hpids : s.pids = s0.pids) (hM : s.mem M = s0.mem M) :
    bwdKernelMG s M NCTX I = bwdKernelMG s0 M NCTX I := by
  simp only [bwdKernelMG, BlockState.readMem, hM, hpids]

theorem bwdKernelDiG_congr {s s0 : BlockState} (Delta : RegionName) (NCTX I : Nat)
    (hpids : s.pids = s0.pids) (hD : s.mem Delta = s0.mem Delta) :
    bwdKernelDiG s Delta NCTX I = bwdKernelDiG s0 Delta NCTX I := by
  simp only [bwdKernelDiG, BlockState.readMem, hD, hpids]

theorem bwdKernelPG_congr {s s0 : BlockState} (Q K M : RegionName) (BD NCTX : Nat) (sc : ℝ)
    (I J : Nat) (hpids : s.pids = s0.pids)
    (hQ : s.mem Q = s0.mem Q) (hK : s.mem K = s0.mem K) (hM : s.mem M = s0.mem M) :
    bwdKernelPG s Q K M BD NCTX sc I J = bwdKernelPG s0 Q K M BD NCTX sc I J := by
  simp only [bwdKernelPG, bwdKernelQKG]
  by_cases h : J ≤ I
  · rw [if_pos h, if_pos h, bwdKernelMG_congr M NCTX I hpids hM]
    refine congrArg (fun x => Real.exp (x * sc - _)) ?_
    refine Finset.sum_congr rfl (fun e _ => ?_)
    rw [bwdKernelQG_congr Q BD I e.val hpids hQ, bwdKernelKG_congr K BD J e.val hpids hK]
  · rw [if_neg h, if_neg h]

theorem bwdKernelDSG_congr {s s0 : BlockState} (Q K V DO M Delta : RegionName) (BD NCTX : Nat)
    (sc : ℝ) (I J : Nat) (hpids : s.pids = s0.pids)
    (hQ : s.mem Q = s0.mem Q) (hK : s.mem K = s0.mem K) (hV : s.mem V = s0.mem V)
    (hDO : s.mem DO = s0.mem DO) (hM : s.mem M = s0.mem M) (hDe : s.mem Delta = s0.mem Delta) :
    bwdKernelDSG s Q K V DO M Delta BD NCTX sc I J
      = bwdKernelDSG s0 Q K V DO M Delta BD NCTX sc I J := by
  simp only [bwdKernelDSG, bwdKernelDPG]
  rw [bwdKernelPG_congr Q K M BD NCTX sc I J hpids hQ hK hM,
    bwdKernelDiG_congr Delta NCTX I hpids hDe]
  have hsum : (∑ e : Fin BD, bwdKernelDOG s DO BD I e.val * bwdKernelVG s V BD J e.val)
      = ∑ e : Fin BD, bwdKernelDOG s0 DO BD I e.val * bwdKernelVG s0 V BD J e.val := by
    refine Finset.sum_congr rfl (fun e _ => ?_)
    rw [bwdKernelDOG_congr DO BD I e.val hpids hDO, bwdKernelVG_congr V BD J e.val hpids hV]
  rw [hsum]

/-- `bwdInnerDvG` is invariant under state changes off `Q`/`K`/`M`/`DO`. -/
theorem bwdInnerDvG_congr {s s0 : BlockState} (Q K M DO : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (acc : Tile .real [BM, BD])
    (hpids : s.pids = s0.pids)
    (hQ : s.mem Q = s0.mem Q) (hK : s.mem K = s0.mem K) (hM : s.mem M = s0.mem M)
    (hDO : s.mem DO = s0.mem DO) :
    bwdInnerDvG s Q K M DO BM BD NCTX qRow kRow sc acc
      = bwdInnerDvG s0 Q K M DO BM BD NCTX qRow kRow sc acc := by
  refine Tile.ext (fun idx => ?_)
  obtain ⟨j, e, u⟩ := idx
  rw [bwdInnerDvG_cell, bwdInnerDvG_cell]
  refine congrArg (Option.map₂ (· + ·) (acc.data (j, e, PUnit.unit))) (congrArg some ?_)
  refine Finset.sum_congr rfl (fun iL _ => ?_)
  rw [bwdKernelPG_congr Q K M BD NCTX sc _ _ hpids hQ hK hM,
    bwdKernelDOG_congr DO BD _ e.val hpids hDO]

/-- `bwdInnerDkG` is invariant under state changes off `Q`/`K`/`V`/`DO`/`M`/`Delta`. -/
theorem bwdInnerDkG_congr {s s0 : BlockState} (Q K V DO M Delta : RegionName)
    (BM BD NCTX qRow kRow : Nat) (sc : ℝ) (acc : Tile .real [BM, BD])
    (hpids : s.pids = s0.pids)
    (hQ : s.mem Q = s0.mem Q) (hK : s.mem K = s0.mem K) (hV : s.mem V = s0.mem V)
    (hDO : s.mem DO = s0.mem DO) (hM : s.mem M = s0.mem M) (hDe : s.mem Delta = s0.mem Delta) :
    bwdInnerDkG s Q K V DO M Delta BM BD NCTX qRow kRow sc acc
      = bwdInnerDkG s0 Q K V DO M Delta BM BD NCTX qRow kRow sc acc := by
  refine Tile.ext (fun idx => ?_)
  obtain ⟨j, e, u⟩ := idx
  rw [bwdInnerDkG_cell, bwdInnerDkG_cell]
  refine congrArg (Option.map₂ (· + ·) (acc.data (j, e, PUnit.unit))) (congrArg some ?_)
  refine Finset.sum_congr rfl (fun iL _ => ?_)
  rw [bwdKernelDSG_congr Q K V DO M Delta BD NCTX sc _ _ hpids hQ hK hV hDO hM hDe,
    bwdKernelQG_congr Q BD _ e.val hpids hQ]

/-- `bwdDqKeyContrib` is invariant under state changes off `Q`/`K`/`V`/`DO`/`M`/`Delta`. -/
theorem bwdDqKeyContrib_congr {s s0 : BlockState} (Q K V DO M Delta : RegionName)
    (BM BD NCTX : Nat) (sc : ℝ) (n I e : Nat) (hpids : s.pids = s0.pids)
    (hQ : s.mem Q = s0.mem Q) (hK : s.mem K = s0.mem K) (hV : s.mem V = s0.mem V)
    (hDO : s.mem DO = s0.mem DO) (hM : s.mem M = s0.mem M) (hDe : s.mem Delta = s0.mem Delta) :
    bwdDqKeyContrib s Q K V DO M Delta BM BD NCTX sc n I e
      = bwdDqKeyContrib s0 Q K V DO M Delta BM BD NCTX sc n I e := by
  simp only [bwdDqKeyContrib]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  rw [bwdKernelDSG_congr Q K V DO M Delta BD NCTX sc I (n * BM + jL.val) hpids hQ hK hV hDO hM hDe,
    bwdKernelKG_congr K BD (n * BM + jL.val) e hpids hK]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General inner-loop step** (the `forRangeDyn_inv` `h_step`).  Advancing the
counter from `i` to `i + BM` (query block `i/BM → i/BM + 1`) preserves the inner
invariant: the body runs `bwd_inner_stepG` with reference state `sin` itself (so
its `hmem` is `rfl`), then the off-`DQ` congruences re-express the results in the
fixed entry vocabulary `s0`, the accumulator recurrences advance `dv`/`dk`, and
the `DQ` read-modify-write extends the running per-key-block contribution to the
newly-visited block. -/
private theorem bwdInnerInvariant_stepG (s0 : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD D0 NCTX n num_block : Nat) (sc : ℝ)
    (hBM : 0 < BM) (hbdvd : BD ∣ bwdKBase s0)
    (hbound : bwdKBase s0 / BD + num_block * BM ≤ D0)
    (hQDQ : Q ≠ DQ) (hKDQ : K ≠ DQ) (hVDQ : V ≠ DQ) (hDODQ : DO ≠ DQ)
    (hMDQ : M ≠ DQ) (hDeDQ : Delta ≠ DQ)
    (i : Nat) (s : BlockState) (hilt : i < num_block * BM)
    (hinv : bwdInnerInvariantG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc i s) :
    ∃ s', stepStmts (bwdInnerBodyG sc BM BD) (s.setReg "start_m" .nat [] (Tile.scalar i)) = some s'
      ∧ bwdInnerInvariantG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc (i + BM) s' := by
  obtain ⟨hpids, hmod, hni, hile, hmemoff, hdq, hundef, homm, honn, hqp, hdop, hdqp,
    hk, hv, hdv, hdk, hmptr, hDptr, hdqptrs⟩ := hinv
  -- counter algebra: i = (n + t) * BM, with t = i/BM - n
  set t := i / BM - n with htdef
  have hdvd : BM ∣ i := Nat.dvd_of_mod_eq_zero hmod
  have hnle : n ≤ i / BM := (Nat.le_div_iff_mul_le hBM).mpr hni
  have hidiv : i / BM = n + t := by omega
  have hieq : i = (n + t) * BM := by
    conv_lhs => rw [← Nat.div_mul_cancel hdvd]
    rw [hidiv]
  have hidivBM : (i + BM) / BM = n + t + 1 := by
    rw [Nat.add_div_right _ hBM, hidiv]
  -- reference state for bwd_inner_stepG is `sin` itself (hmem becomes rfl)
  set sin := s.setReg "start_m" .nat [] (Tile.scalar i) with hsin
  have hsinpids : sin.pids = s0.pids := by rw [hsin]; simpa using hpids
  have hsinmemoff : ∀ R o, R ≠ DQ → sin.mem R o = s0.mem R o := by
    intro R o hR; rw [hsin]; simp only [BlockState.setReg_mem]; exact hmemoff R o hR
  have hsinmemfun : ∀ R : RegionName, R ≠ DQ → sin.mem R = s0.mem R := by
    intro R hR; funext o; exact hsinmemoff R o hR
  -- register readbacks in `sin` (setReg start_m preserves all but start_m)
  have hsin_reg : ∀ {dt sh nm}, nm ≠ "start_m" → sin.regs dt sh nm = s.regs dt sh nm := by
    intro dt sh nm h; rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h]
  -- i + BM ≤ num_block * BM  (block n+t is strictly below num_block)
  have hntlt : n + t < num_block := by
    have : (n + t) * BM < num_block * BM := by rw [← hieq]; exact hilt
    exact Nat.lt_of_mul_lt_mul_right this
  have hiBMle : i + BM ≤ num_block * BM := by
    calc i + BM = (n + t + 1) * BM := by rw [hieq]; ring
      _ ≤ num_block * BM := Nat.mul_le_mul_right BM hntlt
  -- bwdKBase sin = bwdKBase s0
  have hkbeq : bwdKBase sin = bwdKBase s0 := bwdKBase_congr hsinpids
  have hbdvd_sin : BD ∣ bwdKBase sin := by rw [hkbeq]; exact hbdvd
  -- prove hrow bound (global row form)
  have hrow : ∀ iL : Fin BM, bwdKBase sin / BD + i + iL.val < D0 := by
    intro iL
    have hiL := iL.isLt
    rw [hkbeq]
    omega
  -- the prior accumulators (as tiles), in `s0` vocabulary
  set accDv : Tile .real [BM, BD] := ⟨fun idx : TileIndex [BM, BD] =>
    some (bwdAccDvCell s0 Q K M DO BM BD NCTX sc n t idx.1.val idx.2.1.val)⟩ with haccDv
  set accDk : Tile .real [BM, BD] := ⟨fun idx : TileIndex [BM, BD] =>
    some (bwdAccDkCell s0 Q K V DO M Delta BM BD NCTX sc n t idx.1.val idx.2.1.val)⟩ with haccDk
  -- apply the inner-body step with reference = sin
  obtain ⟨sF, hstep, hFpids, hFundef, hFdv, hFdk, hFdq, hFmem, hFdisj,
      hFqp, hFdop, hFdqp, hFdqptrs, hFrest⟩ :=
    bwd_inner_stepG sin sin Q K V DO M Delta DQ BM BD D0 NCTX i (n * BM) sc accDv accDk
      hbdvd_sin hrow rfl
      (by rw [hsin, BlockState.setReg_same])
      (by rw [hsin_reg (by decide)]; exact homm)
      (by rw [hsin_reg (by decide)]; exact honn)
      (by rw [hsin_reg (by decide), hqp, bwdKBase_congr hsinpids])
      (by rw [hsin_reg (by decide), hdop, bwdKBase_congr hsinpids])
      (by rw [hsin_reg (by decide), hdqp, bwdKBase_congr hsinpids])
      (by rw [hsin_reg (by decide), hk]; refine congrArg some (Tile.ext (fun ij => ?_))
          simp only [bwdKernelKG_congr K BD _ ij.2.1.val hsinpids (hsinmemfun K hKDQ)])
      (by rw [hsin_reg (by decide), hv]; refine congrArg some (Tile.ext (fun ij => ?_))
          simp only [bwdKernelVG_congr V BD _ ij.2.1.val hsinpids (hsinmemfun V hVDQ)])
      (by rw [hsin_reg (by decide), hdv])
      (by rw [hsin_reg (by decide), hdk])
      (by rw [hsin_reg (by decide), hmptr]; rw [hsinpids])
      (by rw [hsin_reg (by decide), hDptr]; rw [hsinpids])
      (by obtain ⟨tt, htt⟩ := hdqptrs; exact ⟨tt, by rw [hsin_reg (by decide)]; exact htt⟩)
  refine ⟨sF, hstep, ?_⟩
  -- assemble the invariant at i + BM
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hFpids]; exact hsinpids
  · rw [Nat.add_mod_right]; exact hmod
  · omega
  · exact hiBMle
  · -- mem off DQ preserved
    intro R o hR
    rw [hFmem R o hR, hsinmemoff R o hR]
  · -- DQ running formula at i + BM
    intro I e hI he
    have hBD : 0 < BD := lt_of_le_of_lt (Nat.zero_le e) he
    by_cases hvis : i ≤ I ∧ I < i + BM
    · -- newly visited block: I = i + iL for some iL < BM
      obtain ⟨hIge, hIlt⟩ := hvis
      set iL : Fin BM := ⟨I - i, by omega⟩ with hiL
      have hIeq : I = i + iL.val := by simp only [hiL]; omega
      -- LHS: rewrite I → i + iL throughout, express via bwdInnerDqG readback
      rw [hIeq]
      rw [show bwdKBase s0 = bwdKBase sin from (bwdKBase_congr hsinpids).symm]
      rw [hFdq ⟨iL, ⟨e, he⟩, PUnit.unit⟩]
      rw [bwdInnerDqG_cell]
      simp only [WithBot.unbotD_some]
      rw [show bwdKBase sin = bwdKBase s0 from bwdKBase_congr hsinpids]
      -- sin.readMem DQ at row i+iL (≥ i) equals s0 value (unvisited at counter i)
      have hsinDQ : sin.readMem DQ (bwdKBase s0 + (i + iL.val) * BD + e)
          = s0.readMem DQ (bwdKBase s0 + (i + iL.val) * BD + e) := by
        rw [show sin.readMem DQ (bwdKBase s0 + (i + iL.val) * BD + e)
              = s.readMem DQ (bwdKBase s0 + (i + iL.val) * BD + e) from by
            unfold BlockState.readMem; rw [hsin]; simp only [BlockState.setReg_mem]]
        rw [hdq (i + iL.val) e (by omega) he, if_neg (by omega)]
        ring
      rw [hsinDQ, if_pos (show n * BM ≤ i + iL.val ∧ i + iL.val < i + BM from ⟨by omega, by omega⟩)]
      refine congrArg (s0.readMem DQ (bwdKBase s0 + (i + iL.val) * BD + e) + ·) ?_
      simp only [bwdDqKeyContrib]
      refine Finset.sum_congr rfl (fun jL _ => ?_)
      rw [bwdKernelDSG_congr Q K V DO M Delta BD NCTX sc _ _ hsinpids
          (hsinmemfun Q hQDQ) (hsinmemfun K hKDQ) (hsinmemfun V hVDQ)
          (hsinmemfun DO hDODQ) (hsinmemfun M hMDQ) (hsinmemfun Delta hDeDQ),
        bwdKernelKG_congr K BD _ e hsinpids (hsinmemfun K hKDQ)]
    · -- unvisited block: address disjoint from the store image; preserved
      have hdisj : ∀ idx : TileIndex [BM, BD],
          bwdKBase s0 + I * BD + e ≠ bwdKBase sin + (i + idx.1.val) * BD + idx.2.1.val := by
        intro idx hcontra
        rw [show bwdKBase sin = bwdKBase s0 from bwdKBase_congr hsinpids] at hcontra
        have he2 : idx.2.1.val < BD := idx.2.1.isLt
        have hkey : I * BD + e = (i + idx.1.val) * BD + idx.2.1.val := by omega
        have hmod1 : (I * BD + e) % BD = e := by
          rw [Nat.mul_add_mod']; exact Nat.mod_eq_of_lt he
        have hmod2 : ((i + idx.1.val) * BD + idx.2.1.val) % BD = idx.2.1.val := by
          rw [Nat.mul_add_mod']; exact Nat.mod_eq_of_lt he2
        have heq : e = idx.2.1.val := by rw [← hmod1, ← hmod2, hkey]
        have hIeq2 : I = i + idx.1.val := by
          have : I * BD = (i + idx.1.val) * BD := by omega
          exact Nat.eq_of_mul_eq_mul_right hBD this
        have := idx.1.isLt; omega
      rw [hFdisj (bwdKBase s0 + I * BD + e) hdisj]
      rw [show sin.readMem DQ (bwdKBase s0 + I * BD + e)
            = s.readMem DQ (bwdKBase s0 + I * BD + e) from by
          unfold BlockState.readMem; rw [hsin]; simp only [BlockState.setReg_mem]]
      rw [hdq I e hI he]
      by_cases hlo : n * BM ≤ I ∧ I < i
      · rw [if_pos hlo, if_pos ⟨hlo.1, by omega⟩]
      · rw [if_neg hlo, if_neg (by
          rintro ⟨hge, hlt2⟩
          exact hlo ⟨hge, by omega⟩)]
  · rw [hFundef]; exact hundef
  · rw [hFrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hsin_reg (by decide)]; exact homm
  · rw [hFrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hsin_reg (by decide)]; exact honn
  · rw [hFqp, bwdKBase_congr hsinpids]
  · rw [hFdop, bwdKBase_congr hsinpids]
  · rw [hFdqp, bwdKBase_congr hsinpids]
  · rw [hFrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hsin_reg (by decide), hk]
  · rw [hFrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hsin_reg (by decide), hv]
  · -- dv accumulator advances t → t+1
    rw [hFdv]
    rw [bwdInnerDvG_congr Q K M DO BM BD NCTX i (n * BM) sc accDv hsinpids
        (hsinmemfun Q hQDQ) (hsinmemfun K hKDQ) (hsinmemfun M hMDQ) (hsinmemfun DO hDODQ)]
    rw [show (i + BM) / BM - n = t + 1 from by rw [hidivBM]; omega]
    rw [haccDv, hieq, bwdInnerDvG_accStep]
  · -- dk accumulator advances t → t+1
    rw [hFdk]
    rw [bwdInnerDkG_congr Q K V DO M Delta BM BD NCTX i (n * BM) sc accDk hsinpids
        (hsinmemfun Q hQDQ) (hsinmemfun K hKDQ) (hsinmemfun V hVDQ) (hsinmemfun DO hDODQ)
        (hsinmemfun M hMDQ) (hsinmemfun Delta hDeDQ)]
    rw [show (i + BM) / BM - n = t + 1 from by rw [hidivBM]; omega]
    rw [haccDk, hieq, bwdInnerDkG_accStep]
  · rw [hFrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hsin_reg (by decide)]; exact hmptr
  · rw [hFrest (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hsin_reg (by decide)]; exact hDptr
  · -- dq_ptrs exists (modified by the body; supplied by the inner step)
    exact hFdqptrs

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General inner-loop driver.** Runs the *whole* inner `forRangeDyn` over
`start_m ∈ [n·BM, num_block·BM)` (step `BM`) for KV block `n`, from an entry
state `s` satisfying `bwdInnerInvariantG … (n·BM)` and exposing `lo = n·BM`.
The result is a final state satisfying `bwdInnerInvariantG … (num_block·BM)` —
i.e. the `dv`/`dk` registers hold the running accumulation over *all* query
blocks `[n, num_block)` (= the full column sums, lower blocks being causally
zero), and `DQ` holds the entry value plus this KV block's contribution for
every query row `I ≥ n·BM`. -/
private theorem bwdInnerLoop_driveG (s0 : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD D0 NCTX n num_block : Nat) (sc : ℝ)
    (hBM : 0 < BM) (hbdvd : BD ∣ bwdKBase s0)
    (hbound : bwdKBase s0 / BD + num_block * BM ≤ D0) (hn : n ≤ num_block)
    (hQDQ : Q ≠ DQ) (hKDQ : K ≠ DQ) (hVDQ : V ≠ DQ) (hDODQ : DO ≠ DQ)
    (hMDQ : M ≠ DQ) (hDeDQ : Delta ≠ DQ)
    (s : BlockState)
    (hlo : s.regs .nat [] "lo" = some (Tile.scalar (n * BM)))
    (hinv : bwdInnerInvariantG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc (n * BM) s) :
    ∃ sF, stepStmt (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo")
        (Op.mul .nat Broadcast.nil (Op.constNat num_block) (Op.constNat BM)) (Op.constNat BM)
        (bwdInnerBodyG sc BM BD)) s = some sF
      ∧ bwdInnerInvariantG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc (num_block * BM) sF := by
  obtain ⟨final, sFinal, hstep, hfinalge, hPfinal⟩ :=
    forRangeDyn_inv (idx := "start_m") (startOp := Op.ref .nat [] "lo")
      (stopOp := Op.mul .nat Broadcast.nil (Op.constNat num_block) (Op.constNat BM))
      (stepOp := Op.constNat BM) (start := n * BM) (stop := num_block * BM) (step := BM)
      (body := bwdInnerBodyG sc BM BD)
      (P := fun i st => bwdInnerInvariantG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc i st)
      (s_init := s)
      (by rw [evalOp_ref, hlo])
      (by simp only [evalOp_mul, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
          refine congrArg some (Tile.ext (fun u => ?_))
          rw [Tile.scalar_data]
          simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
            NumericDType.mul])
      (by rw [evalOp_constNat])
      (by omega)
      hinv
      (fun i st hi hP => bwdInnerInvariant_stepG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc
        hBM hbdvd hbound hQDQ hKDQ hVDQ hDODQ hMDQ hDeDQ i st hi hP)
  -- final counter equals num_block * BM (invariant pins i ≤ num_block * BM)
  have hfinalle : final ≤ num_block * BM := hPfinal.2.2.2.1
  have hfeq : final = num_block * BM := le_antisymm hfinalle hfinalge
  refine ⟨sFinal, hstep, ?_⟩
  rw [← hfeq]; exact hPfinal

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Inner-loop frame.** Re-running the inner `forRangeDyn` from an entry state
satisfying the invariant, the loop never touches the KV block pointers
`k_tile_ptr`/`v_tile_ptr`/`dk_tile_ptr`/`dv_tile_ptr` nor the `DV`/`DK` memory
regions (the inner body only writes `DQ` and advances q/do/dq), so the unique
final state `sF` (= the one produced by `bwdInnerLoop_driveG`) carries those
registers/regions unchanged from the entry state. -/
private theorem bwdInnerLoop_frameG (s0 : BlockState) (Q K V DO M Delta DQ : RegionName)
    (BM BD D0 NCTX n num_block : Nat) (sc : ℝ)
    (hBM : 0 < BM) (hbdvd : BD ∣ bwdKBase s0)
    (hbound : bwdKBase s0 / BD + num_block * BM ≤ D0) (hn : n ≤ num_block)
    (hQDQ : Q ≠ DQ) (hKDQ : K ≠ DQ) (hVDQ : V ≠ DQ) (hDODQ : DO ≠ DQ)
    (hMDQ : M ≠ DQ) (hDeDQ : Delta ≠ DQ)
    (s : BlockState)
    (hlo : s.regs .nat [] "lo" = some (Tile.scalar (n * BM)))
    (hinv : bwdInnerInvariantG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc (n * BM) s) :
    ∃ sF, stepStmt (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo")
        (Op.mul .nat Broadcast.nil (Op.constNat num_block) (Op.constNat BM)) (Op.constNat BM)
        (bwdInnerBodyG sc BM BD)) s = some sF
      ∧ (∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
          nm ≠ "start_m" → nm ≠ "offs_m_curr" → nm ≠ "q" → nm ≠ "qk" → nm ≠ "m" →
          nm ≠ "p" → nm ≠ "do_val" → nm ≠ "Di" → nm ≠ "dp" → nm ≠ "ds" →
          nm ≠ "dk" → nm ≠ "dv" → nm ≠ "dq" → nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
          nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" →
          sF.regs dt sh nm = s.regs dt sh nm)
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → sF.mem R o = s.mem R o) := by
  -- frame predicate: facts preserved across iterations (relative to entry `s`)
  set Frame : Nat → BlockState → Prop := fun _ st =>
    (∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
      nm ≠ "start_m" → nm ≠ "offs_m_curr" → nm ≠ "q" → nm ≠ "qk" → nm ≠ "m" →
      nm ≠ "p" → nm ≠ "do_val" → nm ≠ "Di" → nm ≠ "dp" → nm ≠ "ds" →
      nm ≠ "dk" → nm ≠ "dv" → nm ≠ "dq" → nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
      nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" →
      st.regs dt sh nm = s.regs dt sh nm)
    ∧ (∀ (R : RegionName) (o : Nat), R ≠ DQ → st.mem R o = s.mem R o) with hFrame
  obtain ⟨final, sFinal, hstep, _, hPfinal⟩ :=
    forRangeDyn_inv (idx := "start_m") (startOp := Op.ref .nat [] "lo")
      (stopOp := Op.mul .nat Broadcast.nil (Op.constNat num_block) (Op.constNat BM))
      (stepOp := Op.constNat BM) (start := n * BM) (stop := num_block * BM) (step := BM)
      (body := bwdInnerBodyG sc BM BD)
      (P := fun i st => bwdInnerInvariantG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc i st
        ∧ Frame i st)
      (s_init := s)
      (by rw [evalOp_ref, hlo])
      (by simp only [evalOp_mul, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
          refine congrArg some (Tile.ext (fun u => ?_))
          rw [Tile.scalar_data]
          simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
            NumericDType.mul])
      (by rw [evalOp_constNat])
      (by omega)
      ⟨hinv, ⟨fun _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => rfl, fun _ _ _ => rfl⟩⟩
      (fun i st hi hP => by
        obtain ⟨hinvi, hframei⟩ := hP
        -- invariant step gives the unique next state s'
        obtain ⟨s', hstep', hinv'⟩ :=
          bwdInnerInvariant_stepG s0 Q K V DO M Delta DQ BM BD D0 NCTX n num_block sc
            hBM hbdvd hbound hQDQ hKDQ hVDQ hDODQ hMDQ hDeDQ i st hi hinvi
        refine ⟨s', hstep', hinv', ?_, ?_⟩
        · -- frame: registers outside the inner body's modified set are untouched
          intro dt sh nm h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17
          -- re-run the body via bwd_inner_stepG to expose the register frame
          obtain ⟨hpids, hmod, hni, hile, hmemoff, hdq, hundef, homm, honn, hqp, hdop, hdqp,
            hk, hv, hdv, hdk, hmptr, hDptr, hdqptrs⟩ := hinvi
          set sin := st.setReg "start_m" .nat [] (Tile.scalar i) with hsin
          have hsinpids : sin.pids = s0.pids := by rw [hsin]; simpa using hpids
          have hsinmemoff : ∀ R o, R ≠ DQ → sin.mem R o = s0.mem R o := by
            intro R o hR; rw [hsin]; simp only [BlockState.setReg_mem]; exact hmemoff R o hR
          have hsinmemfun : ∀ R : RegionName, R ≠ DQ → sin.mem R = s0.mem R := by
            intro R hR; funext o; exact hsinmemoff R o hR
          have hsin_reg : ∀ {dt sh nm}, nm ≠ "start_m" → sin.regs dt sh nm = st.regs dt sh nm := by
            intro dt sh nm h; rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h]
          have hkbeq : bwdKBase sin = bwdKBase s0 := bwdKBase_congr hsinpids
          have hbdvd_sin : BD ∣ bwdKBase sin := by rw [hkbeq]; exact hbdvd
          have hiBMle : i + BM ≤ num_block * BM := by
            have hdvd : BM ∣ i := Nat.dvd_of_mod_eq_zero hmod
            obtain ⟨q, hq⟩ := hdvd
            have hqlt : q < num_block := by
              have hi' : q * BM < num_block * BM := by rw [hq, Nat.mul_comm BM q] at hi; exact hi
              exact Nat.lt_of_mul_lt_mul_right hi'
            calc i + BM = (q + 1) * BM := by rw [hq, Nat.mul_comm BM q]; ring
              _ ≤ num_block * BM := Nat.mul_le_mul_right BM hqlt
          have hrow : ∀ iL : Fin BM, bwdKBase sin / BD + i + iL.val < D0 := by
            intro iL; have hiL := iL.isLt; rw [hkbeq]; omega
          set accDv : Tile .real [BM, BD] := ⟨fun idx : TileIndex [BM, BD] =>
            some (bwdAccDvCell s0 Q K M DO BM BD NCTX sc n (i / BM - n) idx.1.val idx.2.1.val)⟩ with haccDv
          set accDk : Tile .real [BM, BD] := ⟨fun idx : TileIndex [BM, BD] =>
            some (bwdAccDkCell s0 Q K V DO M Delta BM BD NCTX sc n (i / BM - n) idx.1.val idx.2.1.val)⟩ with haccDk
          obtain ⟨sF2, hstep2, _, _, _, _, _, _, _, _, _, _, _, hFrest⟩ :=
            bwd_inner_stepG sin sin Q K V DO M Delta DQ BM BD D0 NCTX i (n * BM) sc accDv accDk
              hbdvd_sin hrow rfl
              (by rw [hsin, BlockState.setReg_same])
              (by rw [hsin_reg (by decide)]; exact homm)
              (by rw [hsin_reg (by decide)]; exact honn)
              (by rw [hsin_reg (by decide), hqp, bwdKBase_congr hsinpids])
              (by rw [hsin_reg (by decide), hdop, bwdKBase_congr hsinpids])
              (by rw [hsin_reg (by decide), hdqp, bwdKBase_congr hsinpids])
              (by rw [hsin_reg (by decide), hk]; refine congrArg some (Tile.ext (fun ij => ?_))
                  simp only [bwdKernelKG_congr K BD _ ij.2.1.val hsinpids (hsinmemfun K hKDQ)])
              (by rw [hsin_reg (by decide), hv]; refine congrArg some (Tile.ext (fun ij => ?_))
                  simp only [bwdKernelVG_congr V BD _ ij.2.1.val hsinpids (hsinmemfun V hVDQ)])
              (by rw [hsin_reg (by decide), hdv])
              (by rw [hsin_reg (by decide), hdk])
              (by rw [hsin_reg (by decide), hmptr]; rw [hsinpids])
              (by rw [hsin_reg (by decide), hDptr]; rw [hsinpids])
              (by obtain ⟨tt, htt⟩ := hdqptrs; exact ⟨tt, by rw [hsin_reg (by decide)]; exact htt⟩)
          -- the two body executions agree (stepStmts is a function)
          have hbodyeq : stepStmts (bwdInnerBodyG sc BM BD) sin = some s' := by
            rw [hsin]; exact hstep'
          have hs2eq : sF2 = s' := by
            have := hstep2.symm.trans hbodyeq; exact Option.some.inj this
          subst hs2eq
          -- nm is outside the body's modified set: preserved through the body and to entry
          rw [hFrest h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17,
            hsin_reg h1]
          exact hframei.1 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17
        · -- frame: DV/DK (and all non-DQ) memory preserved
          intro R o hR
          -- the invariant itself records mem-off-DQ = s0 at every state; chain through entry
          obtain ⟨_, _, _, _, hmemoff', _⟩ := hinv'
          obtain ⟨_, _, _, _, hmemoffi, _⟩ := hinvi
          rw [hmemoff' R o hR, ← hmemoffi R o hR]
          exact (hframei.2 R o hR))
  -- the unique final state coincides with the driver's
  exact ⟨sFinal, hstep, hPfinal.2.1, hPfinal.2.2⟩

/-! ### ════════ General multi-block backward grads: outer body / exec / top ════════ -/

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 8000 in
/-- **General outer-body step** for KV block `n` (`start_n = n`). At symbolic
`BM`/`BD`/`nb` with a dynamic row offset `lo = n·BM` (the Python test shape is the
`nb = 1`, `n = 0` case): the q/do/dq block pointers
are (re-)built at `bwdPtrTileG _ (bwdKBase s) D0 BM BD (n·BM)`, the whole inner
`forRangeDyn` is driven by `bwdInnerLoop_driveG`, and the fp16 `DV`/`DK` stores
land the genuine general column sums `bwdKernelD{V,K}SpecG` at every key row
`n·BM + jL`. -/
private theorem bwdOuterBodyG_step (s sOuter : BlockState) (Q K V DO DQ DK DV M Delta : RegionName)
    (BM BD D0 nb N_CTX n : Nat) (sc : ℝ)
    (hBM : 0 < BM) (hbdvd : BD ∣ bwdKBase s)
    (hbound : bwdKBase s / BD + nb * BM ≤ D0) (hn : n < nb)
    (hNCTX : N_CTX = BM * nb)
    (hbase : (s.pids 0 / 4) * (32768 / BD) + (s.pids 0 % 4) * (8192 / BD)
        = bwdKBase s / BD)
    (hQDQ : Q ≠ DQ) (hKDQ : K ≠ DQ) (hVDQ : V ≠ DQ) (hDODQ : DO ≠ DQ)
    (hMDQ : M ≠ DQ) (hDeDQ : Delta ≠ DQ)
    (hDVDQ : DV ≠ DQ) (hDKDQ : DK ≠ DQ) (hDVDK : DV ≠ DK) (hDKDV : DK ≠ DV)
    (hpidsEq : sOuter.pids = s.pids)
    (hmem : ∀ R o, R ≠ DV → R ≠ DK → R ≠ DQ → sOuter.mem R o = s.mem R o)
    -- input-region memory agrees with the clean reference (input regions disjoint from DV/DK/DQ)
    (hinmem : ∀ R o, R = Q ∨ R = K ∨ R = V ∨ R = DO ∨ R = M ∨ R = Delta →
        sOuter.mem R o = s.mem R o)
    (hundef : ∀ rg o, sOuter.undef rg o = 0)
    -- prior DQ contributions from key blocks < n already accumulated in memory
    (hDQprior : ∀ I e : Nat, I < nb * BM → e < BD →
        sOuter.readMem DQ (bwdKBase s + I * BD + e)
          = s.readMem DQ (bwdKBase s + I * BD + e)
            + (∑ n' ∈ Finset.range n, ∑ jL : Fin BM,
                bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD N_CTX sc I (n' * BM + jL.val)) *
                  bwdKernelKG s K BD (n' * BM + jL.val) e))
    (hsn : sOuter.regs .nat [] "start_n" = some (Tile.scalar n))
    (hohz : sOuter.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)))
    (hoffz : sOuter.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)))
    (hoffh : sOuter.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)))
    (hsqz : sOuter.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD)))
    (hsqh : sOuter.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD)))
    (hkptr : sOuter.regs .blockPtr [BM, BD] "k_tile_ptr"
        = some (bwdPtrTileG K (bwdKBase s) D0 BM BD (n * BM)))
    (hvptr : sOuter.regs .blockPtr [BM, BD] "v_tile_ptr"
        = some (bwdPtrTileG V (bwdKBase s) D0 BM BD (n * BM)))
    (hdkptr : sOuter.regs .blockPtr [BM, BD] "dk_tile_ptr"
        = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)))
    (hdvptr : sOuter.regs .blockPtr [BM, BD] "dv_tile_ptr"
        = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM))) :
    ∃ sF, stepStmts (bwdOuterBodyG Q DO DQ Delta M sc 32768 8192 D0 BM BD nb N_CTX) sOuter = some sF
      ∧ sF.pids = s.pids
      ∧ (∀ rg o, sF.undef rg o = 0)
      -- DV fp16 store at key rows n·BM + jL (raw genuine column sum, cast to fp16)
      ∧ (∀ (jL : Fin BM) (e : Nat), e < BD →
          sF.mem DV (bwdKBase s + (n * BM + jL.val) * BD + e)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ I : Fin N_CTX,
                  bwdFp16 (bwdKernelPG s Q K M BD N_CTX sc I.val (n * BM + jL.val)) *
                    bwdKernelDOG s DO BD I.val e))))
      -- DK fp16 store at key rows n·BM + jL
      ∧ (∀ (jL : Fin BM) (e : Nat), e < BD →
          sF.mem DK (bwdKBase s + (n * BM + jL.val) * BD + e)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ I : Fin N_CTX,
                  bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD N_CTX sc I.val (n * BM + jL.val)) *
                    bwdKernelQG s Q BD I.val e))))
      -- regions other than DV/DK/DQ fully preserved
      ∧ (∀ (R' : RegionName) (o : Nat), R' ≠ DV → R' ≠ DK → R' ≠ DQ → sF.mem R' o = sOuter.mem R' o)
      ∧ (∀ o : Nat, (∀ (jL : Fin BM) (e : Fin BD),
            o ≠ bwdKBase s + (n * BM + jL.val) * BD + e.val) →
          sF.mem DV o = sOuter.mem DV o)
      ∧ (∀ o : Nat, (∀ (jL : Fin BM) (e : Fin BD),
            o ≠ bwdKBase s + (n * BM + jL.val) * BD + e.val) →
          sF.mem DK o = sOuter.mem DK o)
      -- DQ readback: prior contributions extended by key block n
      ∧ (∀ I e : Nat, I < nb * BM → e < BD →
          sF.readMem DQ (bwdKBase s + I * BD + e)
            = s.readMem DQ (bwdKBase s + I * BD + e)
              + (∑ n' ∈ Finset.range (n + 1), ∑ jL : Fin BM,
                  bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD N_CTX sc I (n' * BM + jL.val)) *
                    bwdKernelKG s K BD (n' * BM + jL.val) e))
      -- advanced k/v/dk/dv ptrs at row (n+1)·BM
      ∧ sF.regs .blockPtr [BM, BD] "k_tile_ptr"
          = some (bwdPtrTileG K (bwdKBase s) D0 BM BD ((n + 1) * BM))
      ∧ sF.regs .blockPtr [BM, BD] "v_tile_ptr"
          = some (bwdPtrTileG V (bwdKBase s) D0 BM BD ((n + 1) * BM))
      ∧ sF.regs .blockPtr [BM, BD] "dk_tile_ptr"
          = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD ((n + 1) * BM))
      ∧ sF.regs .blockPtr [BM, BD] "dv_tile_ptr"
          = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD ((n + 1) * BM))
      ∧ sF.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0))
      ∧ sF.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4))
      ∧ sF.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4))
      ∧ sF.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD))
      ∧ sF.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD)) := by
  have hpids0 : sOuter.pids 0 = s.pids 0 := congrFun hpidsEq 0
  have hnle : n ≤ nb := Nat.le_of_lt hn
  have hnBMlt : n * BM < nb * BM := Nat.mul_lt_mul_of_lt_of_le hn (le_refl BM) hBM
  -- bwdKBase sOuter = bwdKBase s
  have hkbeq : bwdKBase sOuter = bwdKBase s := bwdKBase_congr hpidsEq
  unfold bwdOuterBodyG
  -- stmt 1: lo = start_n * BM = n * BM
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BM)) sOuter
        = some (Tile.scalar (n * BM)) from by
      simp only [evalOp_mul, evalOp_ref, hsn, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.mul]))]
  set s1 := sOuter.setReg "lo" .nat [] (Tile.scalar (n * BM)) with hs1
  -- helper: bwdMkPtrLoG R evaluates to bwdPtrTileG R (bwdKBase s) D0 BM BD (n·BM)
  have hmkLo : ∀ (R : RegionName) (t : BlockState),
      t.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)) →
      t.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)) →
      t.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD)) →
      t.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD)) →
      t.regs .nat [] "lo" = some (Tile.scalar (n * BM)) →
      evalOp (bwdMkPtrLoG R D0 BM BD) t = some (bwdPtrTileG R (bwdKBase s) D0 BM BD (n * BM)) := by
    intro R t hz hh hqz hqh hlo
    unfold bwdMkPtrLoG
    rw [makeBlockPtr2_eval]
    simp only [evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, hz, hh, hqz, hqh, hlo,
      Option.bind_eq_bind, Option.bind_some, List.mapM_cons, List.mapM_nil,
      Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, NumericDType.mul]
    refine congrArg some ?_; ext idx
    simp only [bwdPtrTileG, BlockPtr.mk.injEq, List.cons.injEq, and_true, true_and]
    rw [hbase]
  have hoffz1 : s1.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffz
  have hoffh1 : s1.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffh
  have hsqz1 : s1.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsqz
  have hsqh1 : s1.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsqh
  have hlo1 : s1.regs .nat [] "lo" = some (Tile.scalar (n * BM)) := by
    rw [hs1, BlockState.setReg_same]
  -- stmt 2: q_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (hmkLo Q s1 hoffz1 hoffh1 hsqz1 hsqh1 hlo1))]
  set s1q := s1.setReg "q_tile_ptr" .blockPtr [BM, BD] (bwdPtrTileG Q (bwdKBase s) D0 BM BD (n * BM)) with hs1q
  -- stmt 3: do_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (hmkLo DO s1q
    (by rw [hs1q, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffz1)
    (by rw [hs1q, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffh1)
    (by rw [hs1q, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsqz1)
    (by rw [hs1q, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsqh1)
    (by rw [hs1q, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hlo1)))]
  set s1do := s1q.setReg "do_tile_ptr" .blockPtr [BM, BD] (bwdPtrTileG DO (bwdKBase s) D0 BM BD (n * BM)) with hs1do
  -- stmt 4: dq_tile_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (hmkLo DQ s1do
    (by rw [hs1do, hs1q]; iterate 2 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hoffz1)
    (by rw [hs1do, hs1q]; iterate 2 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hoffh1)
    (by rw [hs1do, hs1q]; iterate 2 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hsqz1)
    (by rw [hs1do, hs1q]; iterate 2 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hsqh1)
    (by rw [hs1do, hs1q]; iterate 2 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hlo1)))]
  set s1dq := s1do.setReg "dq_tile_ptr" .blockPtr [BM, BD] (bwdPtrTileG DQ (bwdKBase s) D0 BM BD (n * BM)) with hs1dq
  have hlodq : s1dq.regs .nat [] "lo" = some (Tile.scalar (n * BM)) := by
    rw [hs1dq, hs1do, hs1q]; iterate 3 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hlo1
  have hsndq : s1dq.regs .nat [] "start_n" = some (Tile.scalar n) := by
    rw [hs1dq, hs1do, hs1q, hs1]; iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hsn
  have hoffzdq : s1dq.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)) := by
    rw [hs1dq, hs1do, hs1q]; iterate 3 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hoffz1
  have hoffhdq : s1dq.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)) := by
    rw [hs1dq, hs1do, hs1q]; iterate 3 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hoffh1
  -- stmt 5: DQ reassign (existence only)
  obtain ⟨tDQ, htDQ⟩ : ∃ v, evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase DQ)
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 32768))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 8192)))) s1dq
      = some v := by
    simp only [evalOp, evalOp_constNat, evalOp_add, evalOp_mul, evalOp_ref, hoffzdq, hoffhdq,
      Option.bind_eq_bind, Option.bind_some]
    exact ⟨_, rfl⟩
  rw [stepStmts.cons_some (stepStmt_assign_eq_some htDQ)]
  set sDQp := s1dq.setReg "DQ" .ptr [] tDQ with hsDQp
  have hlodqp : sDQp.regs .nat [] "lo" = some (Tile.scalar (n * BM)) := by
    rw [hsDQp, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hlodq
  have hsndqp : sDQp.regs .nat [] "start_n" = some (Tile.scalar n) := by
    rw [hsDQp, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsndq
  -- stmt 6: offs_qm = lo + arange = n·BM + arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "lo") (Op.arange BM)) sDQp
        = some (Tile.vec (fun iL : Fin BM => n * BM + iL.val)) from by
      simp only [evalOp_add, evalOp_ref, hlodqp, evalOp_arange,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext iL
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add]))]
  set s2 := sDQp.setReg "offs_qm" .nat [BM] (Tile.vec (fun iL : Fin BM => n * BM + iL.val)) with hs2
  -- stmt 7: offs_n = start_n*BM + arange = n·BM + arange
  have hsn2 : s2.regs .nat [] "start_n" = some (Tile.scalar n) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsndqp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BM)) (Op.arange BM)) s2
        = some (Tile.vec (fun j : Fin BM => n * BM + j.val)) from by
      rw [evalOp_add, evalOp_mul, evalOp_ref, hsn2]
      simp only [evalOp_constNat, evalOp_arange, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))]
  set s3 := s2.setReg "offs_n" .nat [BM] (Tile.vec (fun j : Fin BM => n * BM + j.val)) with hs3
  -- stmt 8: offs_m = arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BM) s3 = some (Tile.vec (fun i : Fin BM => i.val)) from
      evalOp_arange BM s3))]
  set s4 := s3.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => i.val)) with hs4
  -- stmt 9: offs_k = arange BD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BD) s4 = some (Tile.vec (fun e : Fin BD => e.val)) from
      evalOp_arange BD s4))]
  set s5 := s4.setReg "offs_k" .nat [BD] (Tile.vec (fun e : Fin BD => e.val)) with hs5
  -- stmt 10: dq_ptrs (existence only)
  obtain ⟨tdqp2, htdqp2⟩ : ∃ v, evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "DQ")
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_qm")) (Op.constNat BD))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_k")) (Op.constNat 1)))) s5
      = some v := by
    have hDQ5 : s5.regs .ptr [] "DQ" = some tDQ := by
      rw [hs5, hs4, hs3, hs2]
      iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hsDQp, BlockState.setReg_same]
    have hqm5 : s5.regs .nat [BM] "offs_qm" = some (Tile.vec (fun iL : Fin BM => n * BM + iL.val)) := by
      rw [hs5, hs4, hs3]
      iterate 3 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs2, BlockState.setReg_same]
    have hk5 : s5.regs .nat [BD] "offs_k" = some (Tile.vec (fun e : Fin BD => e.val)) := by
      rw [hs5, BlockState.setReg_same]
    obtain ⟨offv, hoffv⟩ : ∃ offv, evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_qm")) (Op.constNat BD))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_k")) (Op.constNat 1))) s5
        = some offv := by
      have hm1 : evalOp (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_qm")) (Op.constNat BD)) s5
          = some (Tile.bop NumericDType.nat.mul Broadcast.scalarR
              (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun iL : Fin BM => n * BM + iL.val))) (Tile.scalar BD)) := by
        rw [evalOp_mul]
        rw [Option.bind_eq_bind, Option.bind_eq_some_iff]
        exact ⟨_, evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hqm5, by
          rw [evalOp_constNat]; simp only [Option.bind_eq_bind, Option.bind_some]⟩
      have hm0 : evalOp (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_k")) (Op.constNat 1)) s5
          = some (Tile.bop NumericDType.nat.mul Broadcast.scalarR
              (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun e : Fin BD => e.val))) (Tile.scalar 1)) := by
        rw [evalOp_mul]
        rw [Option.bind_eq_bind, Option.bind_eq_some_iff]
        exact ⟨_, evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hk5, by
          rw [evalOp_constNat]; simp only [Option.bind_eq_bind, Option.bind_some]⟩
      refine ⟨Tile.bop NumericDType.nat.add (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Tile.bop NumericDType.nat.mul Broadcast.scalarR
          (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun iL : Fin BM => n * BM + iL.val))) (Tile.scalar BD))
        (Tile.bop NumericDType.nat.mul Broadcast.scalarR
          (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun e : Fin BD => e.val))) (Tile.scalar 1)), ?_⟩
      rw [evalOp_add, Option.bind_eq_bind]
      refine Option.bind_eq_some_iff.mpr ⟨_, hm1, ?_⟩
      rw [Option.bind_eq_bind]
      exact Option.bind_eq_some_iff.mpr ⟨_, hm0, rfl⟩
    rw [evalOp]
    rw [evalOp_ref, hDQ5, hoffv]
    simp only [Option.bind_eq_bind, Option.bind_some]
    exact ⟨_, rfl⟩
  rw [stepStmts.cons_some (stepStmt_assign_eq_some htdqp2)]
  set s6 := s5.setReg "dq_ptrs" .ptr [BM, BD] tdqp2 with hs6
  -- stmt 11: D_ptrs = (Delta, off_hz*N_CTX)
  have hohz6 : s6.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)) := by
    rw [hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    iterate 10 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hohz
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Delta)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))) s6
        = some (Tile.scalar (Delta, s.pids 0 * N_CTX)) from by
      simp only [evalOp, evalOp_ref, hohz6, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, Region.cast_id,
        Nat.zero_add]))]
  set s7 := s6.setReg "D_ptrs" .ptr [] (Tile.scalar (Delta, s.pids 0 * N_CTX)) with hs7
  -- stmt 12: m_ptrs = (M, off_hz*N_CTX)
  have hohz7 : s7.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)) := by
    rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hohz6
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase M)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat N_CTX))) s7
        = some (Tile.scalar (M, s.pids 0 * N_CTX)) from by
      simp only [evalOp, evalOp_ref, hohz7, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, Region.cast_id,
        Nat.zero_add]))]
  set s8 := s7.setReg "m_ptrs" .ptr [] (Tile.scalar (M, s.pids 0 * N_CTX)) with hs8
  -- stmt 13: dv = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BD] (Op.const 0)) s8
        = some (⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ : Tile .real [BM, BD]) from by
      rw [evalOp_full, evalOp_const]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]))]
  set s9 := s8.setReg "dv" .real [BM, BD] ⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ with hs9
  -- stmt 14: dk = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BD] (Op.const 0)) s9
        = some (⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ : Tile .real [BM, BD]) from by
      rw [evalOp_full, evalOp_const]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun u => ?_))
      rw [Tile.scalar_data]))]
  set s10 := s9.setReg "dk" .real [BM, BD] ⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ with hs10
  have hmem10fromOuter : s10.mem = sOuter.mem := by
    funext R o
    rw [hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    simp only [BlockState.setReg_mem]
  -- input-region memory (≠ DV/DK/DQ) agrees with s
  have hmem10in : ∀ R o, R ≠ DV → R ≠ DK → R ≠ DQ → s10.mem R o = s.mem R o := by
    intro R o h1 h2 h3
    rw [show s10.mem R o = sOuter.mem R o from by rw [hmem10fromOuter]]; exact hmem R o h1 h2 h3
  have hkbeq' : bwdKBase sOuter = bwdKBase s := hkbeq
  -- stmt 15: k load (reference sOuter; loaded tile = bwdKernelKG sOuter)
  have hkptr10 : s10.regs .blockPtr [BM, BD] "k_tile_ptr"
      = some (bwdPtrTileG K (bwdKBase sOuter) D0 BM BD (n * BM)) := by
    rw [hkbeq', hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    iterate 14 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hkptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bwd_load_bc_evalG K (bwdKBase sOuter) D0 BM BD (n * BM) (Op.ref .blockPtr [BM, BD] "k_tile_ptr")
      sOuter s10 (by rw [hkbeq']; exact hbdvd) hmem10fromOuter
      (by intro i; have := i.isLt; rw [hkbeq']
          calc bwdKBase s / BD + n * BM + i.val < bwdKBase s / BD + n * BM + BM := by omega
            _ = bwdKBase s / BD + (n + 1) * BM := by ring
            _ ≤ bwdKBase s / BD + nb * BM := Nat.add_le_add_left (Nat.mul_le_mul_right BM hn) _
            _ ≤ D0 := hbound)
      (by rw [evalOp_ref]; exact hkptr10)))]
  set s11 := s10.setReg "k" .real [BM, BD]
    ⟨fun idx : TileIndex [BM, BD] => some (sOuter.readMem K (bwdKBase sOuter + (n * BM + idx.1.val) * BD + idx.2.1.val))⟩
    with hs11
  -- stmt 16: v load
  have hmem11fromOuter : s11.mem = sOuter.mem := by rw [hs11]; exact hmem10fromOuter
  have hvptr11 : s11.regs .blockPtr [BM, BD] "v_tile_ptr"
      = some (bwdPtrTileG V (bwdKBase sOuter) D0 BM BD (n * BM)) := by
    rw [hkbeq', hs11, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    iterate 14 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hvptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bwd_load_bc_evalG V (bwdKBase sOuter) D0 BM BD (n * BM) (Op.ref .blockPtr [BM, BD] "v_tile_ptr")
      sOuter s11 (by rw [hkbeq']; exact hbdvd) hmem11fromOuter
      (by intro i; have := i.isLt; rw [hkbeq']
          calc bwdKBase s / BD + n * BM + i.val < bwdKBase s / BD + n * BM + BM := by omega
            _ = bwdKBase s / BD + (n + 1) * BM := by ring
            _ ≤ bwdKBase s / BD + nb * BM := Nat.add_le_add_left (Nat.mul_le_mul_right BM hn) _
            _ ≤ D0 := hbound)
      (by rw [evalOp_ref]; exact hvptr11)))]
  set s12 := s11.setReg "v" .real [BM, BD]
    ⟨fun idx : TileIndex [BM, BD] => some (sOuter.readMem V (bwdKBase sOuter + (n * BM + idx.1.val) * BD + idx.2.1.val))⟩
    with hs12
  -- memory / pids facts for s12 (= sOuter off registers)
  have hmem12fromOuter : s12.mem = sOuter.mem := by rw [hs12]; exact hmem11fromOuter
  have hmem12s : ∀ R o, R ≠ DV → R ≠ DK → R ≠ DQ → s12.mem R o = s.mem R o := by
    intro R o h1 h2 h3
    rw [show s12.mem R o = sOuter.mem R o from by rw [hmem12fromOuter]]; exact hmem R o h1 h2 h3
  have hpids12 : s12.pids = s.pids := by
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    simp only [BlockState.setReg_pids]; exact hpidsEq
  have hpids12Outer : s12.pids = sOuter.pids := by rw [hpids12]; exact hpidsEq.symm
  have hundef12 : ∀ rg o, s12.undef rg o = 0 := by
    intro rg o
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    simp only [BlockState.setReg_undef]; exact hundef rg o
  have hkbeq12 : bwdKBase s12 = bwdKBase s := bwdKBase_congr hpids12
  have hbdvd12 : BD ∣ bwdKBase s12 := by rw [hkbeq12]; exact hbdvd
  -- register readback helper for s12 (all set-regs since sDQp; off-(set names))
  -- lo readback in s12
  have hlo12 : s12.regs .nat [] "lo" = some (Tile.scalar (n * BM)) := by
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp]
    iterate 12 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hlodq
  -- assemble entry invariant for the inner driver (reference state s0 := sOuter)
  have hentry : bwdInnerInvariantG sOuter Q K V DO M Delta DQ BM BD D0 N_CTX n nb sc (n * BM) s12 := by
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact hpids12Outer
    · exact Nat.mul_mod_left n BM
    · exact le_refl _
    · exact Nat.le_of_lt hnBMlt
    · intro R o hR; rw [hmem12fromOuter]
    · -- DQ readback at entry: contribution if-then-else is 0 (no I with n·BM ≤ I < n·BM)
      intro I e hI he
      rw [BlockState.readMem, hmem12fromOuter, ← BlockState.readMem,
        if_neg (by omega : ¬(n * BM ≤ I ∧ I < n * BM)), add_zero]
    · exact hundef12
    · -- offs_m
      rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5]
      iterate 8 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs4, BlockState.setReg_same]
    · -- offs_n = n·BM + j
      rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4]
      iterate 9 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs3, BlockState.setReg_same]
    · -- q_tile_ptr at row n·BM
      rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do]
      iterate 14 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs1q, BlockState.setReg_same, hkbeq]
    · -- do_tile_ptr at row n·BM
      rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq]
      iterate 13 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs1do, BlockState.setReg_same, hkbeq]
    · -- dq_tile_ptr at row n·BM
      rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp]
      iterate 12 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs1dq, BlockState.setReg_same, hkbeq]
    · -- k tile = bwdKernelKG sOuter (n·BM + i) e
      rw [hs12, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs11, BlockState.setReg_same]
      rfl
    · -- v tile
      rw [hs12, BlockState.setReg_same]
      rfl
    · -- dv reg = accDvCell at t=0 = 0 (full-0 tile)
      rw [hs12, hs11, hs10]
      iterate 3 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs9, BlockState.setReg_same]
      refine congrArg some (Tile.ext (fun idx => ?_))
      refine congrArg some ?_
      simp only [bwdAccDvCell]
      rw [show n * BM / BM - n = 0 from by rw [Nat.mul_div_cancel _ hBM]; omega]
      simp only [Finset.range_zero, Finset.sum_empty]
    · -- dk reg = accDkCell at t=0 = 0
      rw [hs12, hs11]
      iterate 2 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs10, BlockState.setReg_same]
      refine congrArg some (Tile.ext (fun idx => ?_))
      refine congrArg some ?_
      simp only [bwdAccDkCell]
      rw [show n * BM / BM - n = 0 from by rw [Nat.mul_div_cancel _ hBM]; omega]
      simp only [Finset.range_zero, Finset.sum_empty]
    · -- m_ptrs = (M, pids0 * NCTX)
      rw [hs12, hs11, hs10, hs9]
      iterate 4 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs8, BlockState.setReg_same, hpids0]
    · -- D_ptrs = (Delta, pids0 * NCTX)
      rw [hs12, hs11, hs10, hs9, hs8]
      iterate 5 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs7, BlockState.setReg_same, hpids0]
    · -- dq_ptrs exists
      refine ⟨tdqp2, ?_⟩
      rw [hs12, hs11, hs10, hs9, hs8, hs7]
      iterate 6 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs6, BlockState.setReg_same]
  -- drive the inner loop
  obtain ⟨sInner, hInnerStep, hInnerInv⟩ :=
    bwdInnerLoop_driveG sOuter Q K V DO M Delta DQ BM BD D0 N_CTX n nb sc hBM
      (by rw [hkbeq]; exact hbdvd) (by rw [hkbeq]; exact hbound) hnle
      hQDQ hKDQ hVDQ hDODQ hMDQ hDeDQ s12 hlo12 hentry
  rw [stepStmts.cons_some hInnerStep]
  -- unpack final inner invariant
  obtain ⟨hInpids, _, _, _, hInmemoff, hInDQ, hInundef, _, _, _, _, _,
    hInk, hInv2, hIndv, hIndk, _, _, _⟩ := hInnerInv
  -- frame: KV pointers + DV/DK memory preserved across the inner loop
  obtain ⟨sInner', hInnerStep', hInframeReg, hInframeMem⟩ :=
    bwdInnerLoop_frameG sOuter Q K V DO M Delta DQ BM BD D0 N_CTX n nb sc hBM
      (by rw [hkbeq]; exact hbdvd) (by rw [hkbeq]; exact hbound) hnle
      hQDQ hKDQ hVDQ hDODQ hMDQ hDeDQ s12 hlo12 hentry
  obtain rfl : sInner = sInner' := Option.some.inj (hInnerStep.symm.trans hInnerStep')
  -- s12 KV pointer readbacks
  have hs12_kp : s12.regs .blockPtr [BM, BD] "k_tile_ptr"
      = some (bwdPtrTileG K (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    iterate 16 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hkptr
  have hs12_vp : s12.regs .blockPtr [BM, BD] "v_tile_ptr"
      = some (bwdPtrTileG V (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    iterate 16 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hvptr
  have hs12_dkp : s12.regs .blockPtr [BM, BD] "dk_tile_ptr"
      = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    iterate 16 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hdkptr
  have hs12_dvp : s12.regs .blockPtr [BM, BD] "dv_tile_ptr"
      = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    iterate 16 rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hdvptr
  -- KV pointer readbacks in sInner (via frame)
  have hInkp : sInner.regs .blockPtr [BM, BD] "k_tile_ptr"
      = some (bwdPtrTileG K (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hInframeReg (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]; exact hs12_kp
  have hInvp : sInner.regs .blockPtr [BM, BD] "v_tile_ptr"
      = some (bwdPtrTileG V (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hInframeReg (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]; exact hs12_vp
  have hIndkp : sInner.regs .blockPtr [BM, BD] "dk_tile_ptr"
      = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hInframeReg (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]; exact hs12_dkp
  have hIndvp : sInner.regs .blockPtr [BM, BD] "dv_tile_ptr"
      = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hInframeReg (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]; exact hs12_dvp
  -- off_hz/off_z/off_h/stride_qz_2d/stride_qh_2d recoveries (via frame, then s12 chain)
  have hframeOut : ∀ (nm : RegName), nm ≠ "start_m" → nm ≠ "offs_m_curr" → nm ≠ "q" → nm ≠ "qk" →
      nm ≠ "m" → nm ≠ "p" → nm ≠ "do_val" → nm ≠ "Di" → nm ≠ "dp" → nm ≠ "ds" →
      nm ≠ "dk" → nm ≠ "dv" → nm ≠ "dq" → nm ≠ "dq_ptrs" → nm ≠ "q_tile_ptr" →
      nm ≠ "do_tile_ptr" → nm ≠ "dq_tile_ptr" →
      sInner.regs .nat [] nm = s12.regs .nat [] nm := by
    intro nm a b c d e f g h i j k l m o p q r
    exact hInframeReg a b c d e f g h i j k l m o p q r
  have hs12_nat : ∀ (nm : RegName), nm ≠ "lo" → nm ≠ "q_tile_ptr" → nm ≠ "do_tile_ptr" →
      nm ≠ "dq_tile_ptr" → nm ≠ "DQ" → nm ≠ "offs_qm" → nm ≠ "offs_n" → nm ≠ "offs_m" →
      nm ≠ "offs_k" → nm ≠ "dq_ptrs" → nm ≠ "D_ptrs" → nm ≠ "m_ptrs" → nm ≠ "dv" →
      nm ≠ "dk" → nm ≠ "k" → nm ≠ "v" →
      s12.regs .nat [] nm = sOuter.regs .nat [] nm := by
    intro nm a b c d e f g h i j k l m o p q
    rw [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hsDQp, hs1dq, hs1do, hs1q, hs1]
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ q, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ p,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ o, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ m,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ l, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ k,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ j, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ i,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ g,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ f, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ e,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ c,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ b, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ a]
  have hInohz : sInner.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)) := by
    rw [hframeOut "off_hz" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    rw [hs12_nat "off_hz" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hohz
  have hInoffz : sInner.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)) := by
    rw [hframeOut "off_z" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    rw [hs12_nat "off_z" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hoffz
  have hInoffh : sInner.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)) := by
    rw [hframeOut "off_h" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    rw [hs12_nat "off_h" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hoffh
  have hInsqz : sInner.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD)) := by
    rw [hframeOut "stride_qz_2d" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    rw [hs12_nat "stride_qz_2d" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hsqz
  have hInsqh : sInner.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD)) := by
    rw [hframeOut "stride_qh_2d" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    rw [hs12_nat "stride_qh_2d" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hsqh
  -- dv/dk register tiles at exit (i/BM - n = nb - n)
  have hnbn : nb * BM / BM - n = nb - n := by rw [Nat.mul_div_cancel _ hBM]
  -- stmt: k advance [BM, 0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "k_tile_ptr") [(BM:Nat), (0:Nat)]) sInner
      = some ⟨fun i => ((bwdPtrTileG K (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ from by
      rw [advanceBlockPtr_eval]
      simp only [evalOp_ref, hInkp, Option.bind_eq_bind, Option.bind_some]))]
  set sa1 := sInner.setReg "k_tile_ptr" .blockPtr [BM, BD]
    ⟨fun i => ((bwdPtrTileG K (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ with hsa1
  -- stmt: v advance [BM, 0]
  have hsa1_vp : sa1.regs .blockPtr [BM, BD] "v_tile_ptr"
      = some (bwdPtrTileG V (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hsa1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hInvp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "v_tile_ptr") [(BM:Nat), (0:Nat)]) sa1
      = some ⟨fun i => ((bwdPtrTileG V (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ from by
      rw [advanceBlockPtr_eval]
      simp only [evalOp_ref, hsa1_vp, Option.bind_eq_bind, Option.bind_some]))]
  set sa2 := sa1.setReg "v_tile_ptr" .blockPtr [BM, BD]
    ⟨fun i => ((bwdPtrTileG V (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ with hsa2
  -- readbacks in sa2 (after k/v advances)
  have hsa2_regs : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
      nm ≠ "k_tile_ptr" → nm ≠ "v_tile_ptr" → sa2.regs dt sh nm = sInner.regs dt sh nm := by
    intro dt sh nm h4 h5
    rw [hsa2, hsa1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h5,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4]
  have hsa2_mem : ∀ R o, sa2.mem R o = sInner.mem R o := by
    intro R o; rw [hsa2, hsa1]; simp only [BlockState.setReg_mem]
  -- dv register tile = accDvCell sOuter n (nb-n)
  have hdv2 : sa2.regs .real [BM, BD] "dv"
      = some (⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDvCell sOuter Q K M DO BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩) := by
    rw [hsa2_regs (by decide) (by decide)]
    rw [hIndv, hnbn]
  have hdk2 : sa2.regs .real [BM, BD] "dk"
      = some (⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDkCell sOuter Q K V DO M Delta BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩) := by
    rw [hsa2_regs (by decide) (by decide)]
    rw [hIndk, hnbn]
  have hdvtp2 : sa2.regs .blockPtr [BM, BD] "dv_tile_ptr"
      = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hsa2_regs (by decide) (by decide)]; exact hIndvp
  have hdktp2 : sa2.regs .blockPtr [BM, BD] "dk_tile_ptr"
      = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hsa2_regs (by decide) (by decide)]; exact hIndkp
  -- hrow bound for the fp16 stores
  have hrowDV : ∀ i : Fin BM, bwdKBase s / BD + n * BM + i.val < D0 := by
    intro i; have := i.isLt
    calc bwdKBase s / BD + n * BM + i.val < bwdKBase s / BD + n * BM + BM := by omega
      _ = bwdKBase s / BD + (n + 1) * BM := by ring
      _ ≤ bwdKBase s / BD + nb * BM := Nat.add_le_add_left (Nat.mul_le_mul_right BM hn) _
      _ ≤ D0 := hbound
  -- DV fp16 store
  obtain ⟨sDV, hDVstep, hDVpids, hDVmem, hDVother, hDVoffimg⟩ :=
    bwd_store_fp16_bc_evalG DV (bwdKBase s) D0 BM BD (n * BM) sa2 "dv_tile_ptr" "dv"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDvCell sOuter Q K M DO BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hbdvd hrowDV hdvtp2 hdv2
      (fun idx => ⟨_, rfl⟩)
  rw [stepStmts.cons_some hDVstep]
  -- DK fp16 store
  have hdktpDV : sDV.regs .blockPtr [BM, BD] "dk_tile_ptr"
      = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) := by
    have := bwd_store_fp16_regs_eqG sa2 BM BD (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) "dv_tile_ptr" "dv"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDvCell sOuter Q K M DO BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hdvtp2 hdv2 sDV hDVstep (dt := .blockPtr) (sh := [BM, BD]) (nm' := "dk_tile_ptr")
    rw [this]; exact hdktp2
  have hdk2DV : sDV.regs .real [BM, BD] "dk"
      = some (⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDkCell sOuter Q K V DO M Delta BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩) := by
    have := bwd_store_fp16_regs_eqG sa2 BM BD (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) "dv_tile_ptr" "dv"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDvCell sOuter Q K M DO BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hdvtp2 hdv2 sDV hDVstep (dt := .real) (sh := [BM, BD]) (nm' := "dk")
    rw [this]; exact hdk2
  obtain ⟨sDK, hDKstep, hDKpids, hDKmem, hDKother, hDKoffimg⟩ :=
    bwd_store_fp16_bc_evalG DK (bwdKBase s) D0 BM BD (n * BM) sDV "dk_tile_ptr" "dk"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDkCell sOuter Q K V DO M Delta BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hbdvd hrowDV hdktpDV hdk2DV
      (fun idx => ⟨_, rfl⟩)
  rw [stepStmts.cons_some hDKstep]
  -- final two advances (dv_tile_ptr, dk_tile_ptr)
  have hdvtpDK : sDK.regs .blockPtr [BM, BD] "dv_tile_ptr"
      = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [bwd_store_fp16_regs_eqG sDV BM BD (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) "dk_tile_ptr" "dk"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDkCell sOuter Q K V DO M Delta BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hdktpDV hdk2DV sDK hDKstep (dt := .blockPtr) (sh := [BM, BD]) (nm' := "dv_tile_ptr")]
    rw [bwd_store_fp16_regs_eqG sa2 BM BD (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) "dv_tile_ptr" "dv"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDvCell sOuter Q K M DO BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hdvtp2 hdv2 sDV hDVstep (dt := .blockPtr) (sh := [BM, BD]) (nm' := "dv_tile_ptr")]
    exact hdvtp2
  have hdktpDK : sDK.regs .blockPtr [BM, BD] "dk_tile_ptr"
      = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [bwd_store_fp16_regs_eqG sDV BM BD (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) "dk_tile_ptr" "dk"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDkCell sOuter Q K V DO M Delta BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hdktpDV hdk2DV sDK hDKstep (dt := .blockPtr) (sh := [BM, BD]) (nm' := "dk_tile_ptr")]
    exact hdktpDV
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "dv_tile_ptr") [(BM:Nat), (0:Nat)]) sDK
      = some ⟨fun i => ((bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ from by
      rw [advanceBlockPtr_eval]; simp only [evalOp_ref, hdvtpDK, Option.bind_eq_bind, Option.bind_some]))]
  set sb1 := sDK.setReg "dv_tile_ptr" .blockPtr [BM, BD]
    ⟨fun i => ((bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ with hsb1
  have hdktpb1 : sb1.regs .blockPtr [BM, BD] "dk_tile_ptr"
      = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) := by
    rw [hsb1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hdktpDK
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BD] "dk_tile_ptr") [(BM:Nat), (0:Nat)]) sb1
      = some ⟨fun i => ((bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ from by
      rw [advanceBlockPtr_eval]
      simp only [evalOp_ref, hdktpb1, Option.bind_eq_bind, Option.bind_some])), stepStmts.nil]
  set sb2 := sb1.setReg "dk_tile_ptr" .blockPtr [BM, BD]
    ⟨fun i => ((bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)).data i).advance [(BM:Nat), (0:Nat)]⟩ with hsb2
  -- final state assembly
  have hb2mem : ∀ R o, sb2.mem R o = sDK.mem R o := by
    intro R o; rw [hsb2, hsb1]; simp only [BlockState.setReg_mem]
  -- register readback helper for sb2: nat-[] regs survive advances + fp16 stores
  have hsb2_reg : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
      nm ≠ "dv_tile_ptr" → nm ≠ "dk_tile_ptr" → sb2.regs dt sh nm = sDK.regs dt sh nm := by
    intro dt sh nm hdv hdk
    rw [hsb2, hsb1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hdk,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hdv]
  have hsDK_reg : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
      sDK.regs dt sh nm = sDV.regs dt sh nm :=
    fun {dt sh nm} => bwd_store_fp16_regs_eqG sDV BM BD (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM))
      "dk_tile_ptr" "dk"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDkCell sOuter Q K V DO M Delta BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hdktpDV hdk2DV sDK hDKstep
  have hsDV_reg : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
      sDV.regs dt sh nm = sa2.regs dt sh nm :=
    fun {dt sh nm} => bwd_store_fp16_regs_eqG sa2 BM BD (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM))
      "dv_tile_ptr" "dv"
      ⟨fun idx : TileIndex [BM, BD] =>
        some (bwdAccDvCell sOuter Q K M DO BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
      hdvtp2 hdv2 sDV hDVstep
  -- general sb2 → sInner reg readback for names outside the touched set
  have hsb2_to_sInner : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName},
      nm ≠ "dv_tile_ptr" → nm ≠ "dk_tile_ptr" → nm ≠ "k_tile_ptr" → nm ≠ "v_tile_ptr" →
      sb2.regs dt sh nm = sInner.regs dt sh nm := by
    intro dt sh nm h1 h2 h3 h4
    rw [hsb2_reg h1 h2, hsDK_reg, hsDV_reg, hsa2_regs h3 h4]
  -- mem readback chain for sb2: only DV/DK regions written this iter
  have hsb2memDV : ∀ o, sb2.mem DV o = sDK.mem DV o := fun o => hb2mem DV o
  have hsb2memDK : ∀ o, sb2.mem DK o = sDK.mem DK o := fun o => hb2mem DK o
  -- s.mem R = sOuter.mem R (region-level, for congr)
  have hmemfun' : ∀ R : RegionName, R = Q ∨ R = K ∨ R = V ∨ R = DO ∨ R = M ∨ R = Delta →
      s.mem R = sOuter.mem R := by
    intro R hR; funext o; exact (hinmem R o hR).symm
  -- the inner-loop final DQ readback (in sOuter / s vocabulary)
  have hsInnerDQ : ∀ I e : Nat, I < nb * BM → e < BD →
      sInner.readMem DQ (bwdKBase s + I * BD + e)
        = sOuter.readMem DQ (bwdKBase s + I * BD + e)
          + (if n * BM ≤ I ∧ I < nb * BM
              then bwdDqKeyContrib sOuter Q K V DO M Delta BM BD N_CTX sc n I e else 0) := by
    intro I e hI he
    have hd := hInDQ I e hI he
    rw [← hkbeq]
    exact hd
  refine ⟨sb2, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    rw [hsb2, hsb1]; simp only [BlockState.setReg_pids]
    rw [hDKpids, hDVpids]
    rw [hsa2, hsa1]; simp only [BlockState.setReg_pids]
    exact hInpids.trans hpidsEq
  · -- undef
    intro rg o
    rw [hsb2, hsb1]; simp only [BlockState.setReg_undef]
    -- fp16 stores preserve undef; advances preserve undef; inner-loop frame: undef from invariant
    have hDKundef : sDK.undef rg o = 0 := by
      rw [bwd_store_fp16_undef_eqG sDV BM BD (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) "dk_tile_ptr" "dk"
        ⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDkCell sOuter Q K V DO M Delta BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
        hdktpDV hdk2DV sDK hDKstep rg o]
      rw [bwd_store_fp16_undef_eqG sa2 BM BD (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) "dv_tile_ptr" "dv"
        ⟨fun idx : TileIndex [BM, BD] =>
          some (bwdAccDvCell sOuter Q K M DO BM BD N_CTX sc n (nb - n) idx.1.val idx.2.1.val)⟩
        hdvtp2 hdv2 sDV hDVstep rg o]
      rw [hsa2, hsa1]; simp only [BlockState.setReg_undef]
      exact hInundef rg o
    exact hDKundef
  · -- DV fp16 store at key rows n·BM + jL (raw genuine column sum)
    intro jL e he
    rw [hsb2memDV, hDKother DV (bwdKBase s + (n * BM + jL.val) * BD + e) hDVDK]
    have := hDVmem ⟨jL, ⟨e, he⟩, PUnit.unit⟩
    simp only at this
    rw [this]
    refine congrArg (MemCell.of .fp16) (congrArg (FloatDType.real.cast FloatDType.fp16) ?_)
    refine congrArg some ?_
    rw [hNCTX, bwdAccDvCell_full_eq_rawSum sOuter Q K M DO BM BD nb sc n jL.val e jL.isLt hnle]
    refine Finset.sum_congr rfl (fun I _ => ?_)
    rw [bwdKernelPG_congr Q K M BD (BM * nb) sc I.val (n * BM + jL.val) hpidsEq.symm
        (hmemfun' Q (Or.inl rfl)) (hmemfun' K (Or.inr (Or.inl rfl)))
        (hmemfun' M (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))),
      bwdKernelDOG_congr DO BD I.val e hpidsEq.symm
        (hmemfun' DO (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))]
  · -- DK fp16 store
    intro jL e he
    rw [hsb2memDK, hDKmem ⟨jL, ⟨e, he⟩, PUnit.unit⟩]
    refine congrArg (MemCell.of .fp16) (congrArg (FloatDType.real.cast FloatDType.fp16) ?_)
    refine congrArg some ?_
    rw [hNCTX, bwdAccDkCell_full_eq_rawSum sOuter Q K V DO M Delta BM BD nb sc n jL.val e jL.isLt hnle]
    refine Finset.sum_congr rfl (fun I _ => ?_)
    rw [bwdKernelDSG_congr Q K V DO M Delta BD (BM * nb) sc I.val (n * BM + jL.val) hpidsEq.symm
        (hmemfun' Q (Or.inl rfl)) (hmemfun' K (Or.inr (Or.inl rfl)))
        (hmemfun' V (Or.inr (Or.inr (Or.inl rfl)))) (hmemfun' DO (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
        (hmemfun' M (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))
        (hmemfun' Delta (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))),
      bwdKernelQG_congr Q BD I.val e hpidsEq.symm (hmemfun' Q (Or.inl rfl))]
  · -- regions other than DV/DK/DQ fully preserved
    intro R' o hRDV hRDK hRDQ
    rw [hb2mem, hDKother R' o hRDK, hDVother R' o hRDV, hsa2_mem,
      hInframeMem R' o hRDQ, show s12.mem R' o = sOuter.mem R' o from by rw [hmem12fromOuter]]
  · -- DV preserved off the freshly-written key block
    intro o ho
    rw [hsb2memDV, hDKother DV o hDVDK, hDVoffimg o (fun idx => ho idx.1 idx.2.1), hsa2_mem,
      hInframeMem DV o hDVDQ, show s12.mem DV o = sOuter.mem DV o from by rw [hmem12fromOuter]]
  · -- DK preserved off the freshly-written key block
    intro o ho
    rw [hsb2memDK, hDKoffimg o (fun idx => ho idx.1 idx.2.1)]
    -- sDV.mem DK = sa2.mem DK (DV store off region DK), then inner-frame
    rw [hDVother DK o hDKDV, hsa2_mem, hInframeMem DK o hDKDQ,
      show s12.mem DK o = sOuter.mem DK o from by rw [hmem12fromOuter]]
  · -- DQ readback: prior contributions extended by key block n
    intro I e hI he
    -- sb2.readMem DQ = sInner.readMem DQ (DQ untouched by advances/stores)
    have hDQmemEq : sb2.mem DQ (bwdKBase s + I * BD + e) = sInner.mem DQ (bwdKBase s + I * BD + e) := by
      rw [hb2mem, hDKother DQ _ (Ne.symm hDKDQ), hDVother DQ _ (Ne.symm hDVDQ), hsa2_mem]
    rw [show sb2.readMem DQ (bwdKBase s + I * BD + e)
          = sInner.readMem DQ (bwdKBase s + I * BD + e) from by
        unfold BlockState.readMem; rw [hDQmemEq]]
    rw [hsInnerDQ I e hI he]
    -- sOuter.readMem DQ = s.readMem DQ + prior (blocks < n)
    rw [hDQprior I e hI he]
    by_cases hvis : n * BM ≤ I ∧ I < nb * BM
    · rw [if_pos hvis]
      -- sum over range (n+1) = sum over range n + block-n term
      rw [Finset.sum_range_succ]
      rw [add_assoc]
      refine congrArg (s.readMem DQ (bwdKBase s + I * BD + e) + ·) ?_
      refine congrArg (_ + ·) ?_
      -- block-n term = bwdDqKeyContrib (convert sOuter → s, then unfold)
      rw [bwdDqKeyContrib_congr Q K V DO M Delta BM BD N_CTX sc n I e hpidsEq
        (hmemfun' Q (Or.inl rfl)).symm (hmemfun' K (Or.inr (Or.inl rfl))).symm
        (hmemfun' V (Or.inr (Or.inr (Or.inl rfl)))).symm
        (hmemfun' DO (Or.inr (Or.inr (Or.inr (Or.inl rfl))))).symm
        (hmemfun' M (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))).symm
        (hmemfun' Delta (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))).symm]
      simp only [bwdDqKeyContrib]
    · rw [if_neg hvis]
      -- I < n·BM (since I < nb·BM): the block-n term is 0, range(n+1) = range n + 0
      have hIlt : I < n * BM := by
        rcases Nat.lt_or_ge I (n * BM) with h | h
        · exact h
        · exact absurd ⟨h, hI⟩ hvis
      rw [Finset.sum_range_succ, add_zero]
      refine congrArg (s.readMem DQ (bwdKBase s + I * BD + e) + ·) ?_
      -- block-n contribution (j ≥ n·BM > I) is causally zero
      refine (add_zero _).symm.trans ?_
      refine congrArg (_ + ·) ?_
      symm
      refine Finset.sum_eq_zero (fun jL _ => ?_)
      have hds0 : bwdKernelDSG s Q K V DO M Delta BD N_CTX sc I (n * BM + jL.val) = 0 := by
        simp only [bwdKernelDSG, bwdKernelPG, if_neg (by omega : ¬ n * BM + jL.val ≤ I), zero_mul]
      rw [hds0, show bwdFp16 0 = 0 from by simp only [bwdFp16, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot, FloatDType.storeValue]; norm_num, zero_mul]
  · -- k_tile_ptr advanced to (n+1)·BM
    -- chain sb2 → sa1 (k untouched after sa1)
    have hk_sb2 : sb2.regs .blockPtr [BM, BD] "k_tile_ptr" = sa1.regs .blockPtr [BM, BD] "k_tile_ptr" := by
      rw [hsb2_reg (by decide) (by decide), hsDK_reg, hsDV_reg,
        hsa2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    rw [hk_sb2, hsa1, BlockState.setReg_same]
    rw [show ((n : Nat) + 1) * BM = n * BM + BM from by ring]
    exact congrArg some (bwdPtrTileG_advance K (bwdKBase s) D0 BM BD (n * BM))
  · -- v_tile_ptr advanced to (n+1)·BM
    have hv_sb2 : sb2.regs .blockPtr [BM, BD] "v_tile_ptr" = sa2.regs .blockPtr [BM, BD] "v_tile_ptr" := by
      rw [hsb2_reg (by decide) (by decide), hsDK_reg, hsDV_reg]
    rw [hv_sb2, hsa2, BlockState.setReg_same]
    rw [show ((n : Nat) + 1) * BM = n * BM + BM from by ring]
    exact congrArg some (bwdPtrTileG_advance V (bwdKBase s) D0 BM BD (n * BM))
  · -- dk_tile_ptr advanced to (n+1)·BM
    rw [hsb2, BlockState.setReg_same]
    rw [show ((n : Nat) + 1) * BM = n * BM + BM from by ring]
    exact congrArg some (bwdPtrTileG_advance DK (bwdKBase s) D0 BM BD (n * BM))
  · -- dv_tile_ptr advanced to (n+1)·BM
    rw [hsb2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hsb1, BlockState.setReg_same]
    rw [show ((n : Nat) + 1) * BM = n * BM + BM from by ring]
    exact congrArg some (bwdPtrTileG_advance DV (bwdKBase s) D0 BM BD (n * BM))
  · -- off_hz preserved
    rw [hsb2_to_sInner (by decide) (by decide) (by decide) (by decide)]; exact hInohz
  · -- off_z
    rw [hsb2_to_sInner (by decide) (by decide) (by decide) (by decide)]; exact hInoffz
  · -- off_h
    rw [hsb2_to_sInner (by decide) (by decide) (by decide) (by decide)]; exact hInoffh
  · -- stride_qz_2d
    rw [hsb2_to_sInner (by decide) (by decide) (by decide) (by decide)]; exact hInsqz
  · -- stride_qh_2d
    rw [hsb2_to_sInner (by decide) (by decide) (by decide) (by decide)]; exact hInsqh

/-- **Outer-loop invariant** for the KV-block `start_n` loop. After `n` blocks:
`DV`/`DK` memory at every already-visited key row `J < n·BM` holds the genuine
fp16 column sum (raw real sum cast to fp16); `DQ` rows accumulate the
contributions of key blocks `< n`; the k/v/dk/dv block pointers sit at row
`n·BM`; the index/stride scalars persist; all other memory is unchanged. -/
private def bwdOuterInvariantG
    (s : BlockState) (Q K V DO DQ DK DV M Delta : RegionName)
    (BM BD D0 nb : Nat) (sc : ℝ) (n : Nat) (st : BlockState) : Prop :=
  st.pids = s.pids ∧ (∀ rg o, st.undef rg o = 0) ∧
  (∀ J e : Nat, J < n * BM → e < BD →
    st.mem DV (bwdKBase s + J * BD + e)
      = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ I : Fin (BM * nb),
            bwdFp16 (bwdKernelPG s Q K M BD (BM * nb) sc I.val J) * bwdKernelDOG s DO BD I.val e)))) ∧
  (∀ J e : Nat, J < n * BM → e < BD →
    st.mem DK (bwdKBase s + J * BD + e)
      = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ I : Fin (BM * nb),
            bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I.val J) * bwdKernelQG s Q BD I.val e)))) ∧
  (∀ J e : Nat, J ≥ n * BM → J < nb * BM → e < BD →
    st.mem DV (bwdKBase s + J * BD + e) = s.mem DV (bwdKBase s + J * BD + e)) ∧
  (∀ J e : Nat, J ≥ n * BM → J < nb * BM → e < BD →
    st.mem DK (bwdKBase s + J * BD + e) = s.mem DK (bwdKBase s + J * BD + e)) ∧
  (∀ I e : Nat, I < nb * BM → e < BD →
    st.readMem DQ (bwdKBase s + I * BD + e)
      = s.readMem DQ (bwdKBase s + I * BD + e)
        + (∑ n' ∈ Finset.range n, ∑ jL : Fin BM,
            bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I (n' * BM + jL.val)) *
              bwdKernelKG s K BD (n' * BM + jL.val) e)) ∧
  (∀ (R' : RegionName) (o : Nat), R' ≠ DV → R' ≠ DK → R' ≠ DQ → st.mem R' o = s.mem R' o) ∧
  st.regs .blockPtr [BM, BD] "k_tile_ptr" = some (bwdPtrTileG K (bwdKBase s) D0 BM BD (n * BM)) ∧
  st.regs .blockPtr [BM, BD] "v_tile_ptr" = some (bwdPtrTileG V (bwdKBase s) D0 BM BD (n * BM)) ∧
  st.regs .blockPtr [BM, BD] "dk_tile_ptr" = some (bwdPtrTileG DK (bwdKBase s) D0 BM BD (n * BM)) ∧
  st.regs .blockPtr [BM, BD] "dv_tile_ptr" = some (bwdPtrTileG DV (bwdKBase s) D0 BM BD (n * BM)) ∧
  st.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 0)) ∧
  st.regs .nat [] "off_z" = some (Tile.scalar (s.pids 0 / 4)) ∧
  st.regs .nat [] "off_h" = some (Tile.scalar (s.pids 0 % 4)) ∧
  st.regs .nat [] "stride_qz_2d" = some (Tile.scalar (32768 / BD)) ∧
  st.regs .nat [] "stride_qh_2d" = some (Tile.scalar (8192 / BD))

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 8000 in
/-- **General multi-block backward exec.** Running the full general `_bwd_kernel`
(contiguous strides `[BD,1]`, `stride_qz = 32768`, `stride_qh = 8192`, `H = 4`,
`N_CTX = BM·nb`) from a clean honest state reaches a final state whose `DV`/`DK`
memory (fp16 `MemCell` level) holds the genuine general column sums at every key
row, and whose `DQ` memory (real) holds `bwdKernelDQSpecG`. Driven by the outer
`forRange_inv` over KV blocks `start_n ∈ [0, nb)` with `bwdOuterInvariantG`. -/
theorem bwd_grads_execG (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (BM BD D0 nb : Nat) (sc : ℝ)
    (hBM : 0 < BM) (hBD : 0 < BD) (hnb : 0 < nb) (hbdvd : BD ∣ bwdKBase s)
    (hbound : bwdKBase s / BD + nb * BM ≤ D0)
    (hbase : (s.pids 0 / 4) * (32768 / BD) + (s.pids 0 % 4) * (8192 / BD) = bwdKBase s / BD)
    (hQDQ : Q ≠ DQ) (hKDQ : K ≠ DQ) (hVDQ : V ≠ DQ) (hDODQ : DO ≠ DQ)
    (hMDQ : M ≠ DQ) (hDeDQ : Delta ≠ DQ)
    (hDVDQ : DV ≠ DQ) (hDKDQ : DK ≠ DQ) (hDVDK : DV ≠ DK) (hDKDV : DK ≠ DV)
    -- input regions disjoint from the gradient outputs DV/DK
    (hin : ∀ R : RegionName, R = Q ∨ R = K ∨ R = V ∨ R = DO ∨ R = M ∨ R = Delta →
        R ≠ DV ∧ R ≠ DK ∧ R ≠ DQ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM) s = some sF
      ∧ (∀ I e : Nat, I < nb * BM → e < BD →
          sF.readMem DQ (bwdKBase s + I * BD + e)
            = bwdKernelDQSpecG s Q K V DO M Delta DQ BD (BM * nb) sc I e)
      ∧ (∀ J e : Nat, J < nb * BM → e < BD →
          sF.mem DV (bwdKBase s + J * BD + e)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ I : Fin (BM * nb),
                  bwdFp16 (bwdKernelPG s Q K M BD (BM * nb) sc I.val J) * bwdKernelDOG s DO BD I.val e))))
      ∧ (∀ J e : Nat, J < nb * BM → e < BD →
          sF.mem DK (bwdKBase s + J * BD + e)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (∑ I : Fin (BM * nb),
                  bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I.val J) * bwdKernelQG s Q BD I.val e)))) := by
  -- body split → preLoop ++ outer forRange
  have hbody : exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM) s
      = stepStmts (bwdPreLoopG K V DK DV 32768 8192 D0 BM BD 4
          ++ [Stmt.forRange "start_n" 0 nb 1 (bwdOuterBodyG Q DO DQ Delta M sc 32768 8192 D0 BM BD nb (BM * nb))]) s := by
    show stepStmts (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM).toAlgKernel.body s = _
    rw [bwd_body_splitG]
  rw [hbody]
  obtain ⟨s0, hpre, hs0pids, hs0mem, hs0undef, hohz, hoffz, hoffh, hsqz, hsqh,
    hkp, hvp, hdkp, hdvp⟩ := bwdPreLoopG_eval s K V DK DV D0 BM BD hbase hundef
  rw [stepStmts.append_some hpre]
  -- outer forRange over start_n ∈ [0, nb)
  obtain ⟨final, sFinal, hforstep, hfinalge, hPfinal⟩ :=
    forRange_inv (idx := "start_n") (start := 0) (stop := nb) (step := 1)
      (body := bwdOuterBodyG Q DO DQ Delta M sc 32768 8192 D0 BM BD nb (BM * nb))
      (P := fun n st => bwdOuterInvariantG s Q K V DO DQ DK DV M Delta BM BD D0 nb sc n st)
      (s_init := s0) (by decide)
      -- h_init: invariant at n = 0
      (by
        refine ⟨hs0pids, hs0undef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hohz, hoffz, hoffh, hsqz, hsqh⟩
        · intro J e hJ he; simp only [Nat.zero_mul, Nat.not_lt_zero] at hJ
        · intro J e hJ he; simp only [Nat.zero_mul, Nat.not_lt_zero] at hJ
        · intro J e _ _ _; rw [hs0mem]
        · intro J e _ _ _; rw [hs0mem]
        · intro I e _ _
          simp only [Finset.range_zero, Finset.sum_empty, add_zero]
          unfold BlockState.readMem; rw [hs0mem]
        · intro R' o _ _ _; rw [hs0mem]
        · rw [Nat.zero_mul]; exact hkp
        · rw [Nat.zero_mul]; exact hvp
        · rw [Nat.zero_mul]; exact hdkp
        · rw [Nat.zero_mul]; exact hdvp)
      -- h_step
      (fun n st hn hP => by
        obtain ⟨hpids, hsundef, hDVlo, hDKlo, hDVhi, hDKhi, hDQacc, hother,
          hkptr, hvptr, hdkptr, hdvptr, hohz', hoffz', hoffh', hsqz', hsqh'⟩ := hP
        set stn := st.setReg "start_n" .nat [] (Tile.scalar n) with hstn
        -- readbacks in stn (setReg start_n preserves all but start_n)
        have hstn_reg : ∀ {dt sh nm}, nm ≠ "start_n" → stn.regs dt sh nm = st.regs dt sh nm := by
          intro dt sh nm h; rw [hstn, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h]
        have hstnmem : ∀ R o, stn.mem R o = st.mem R o := by
          intro R o; rw [hstn]; simp only [BlockState.setReg_mem]
        have hstnpids : stn.pids = s.pids := by rw [hstn, BlockState.setReg_pids]; exact hpids
        have hstnundef : ∀ rg o, stn.undef rg o = 0 := by
          intro rg o; rw [hstn]; simp only [BlockState.setReg_undef]; exact hsundef rg o
        -- apply the outer body step at block n
        obtain ⟨sF, hstep, hFpids, hFundef, hFDV, hFDK, hFother, hFDVoff, hFDKoff, hFDQ,
          hFkp, hFvp, hFdkp, hFdvp, hFohz, hFoffz, hFoffh, hFsqz, hFsqh⟩ :=
          bwdOuterBodyG_step s stn Q K V DO DQ DK DV M Delta BM BD D0 nb (BM * nb) n sc
            hBM hbdvd hbound hn rfl hbase
            hQDQ hKDQ hVDQ hDODQ hMDQ hDeDQ hDVDQ hDKDQ hDVDK hDKDV
            hstnpids
            (fun R o h1 h2 h3 => (hstnmem R o).trans (hother R o h1 h2 h3))
            (fun R o hRin => (hstnmem R o).trans (hother R o (hin R hRin).1 (hin R hRin).2.1 (hin R hRin).2.2))
            hstnundef
            -- hDQprior from invariant
            (by intro I e hI he
                rw [show stn.readMem DQ (bwdKBase s + I * BD + e) = st.readMem DQ (bwdKBase s + I * BD + e) from by
                    unfold BlockState.readMem; rw [hstnmem]]
                exact hDQacc I e hI he)
            (by rw [hstn, BlockState.setReg_same])
            (by rw [hstn_reg (by decide)]; exact hohz')
            (by rw [hstn_reg (by decide)]; exact hoffz')
            (by rw [hstn_reg (by decide)]; exact hoffh')
            (by rw [hstn_reg (by decide)]; exact hsqz')
            (by rw [hstn_reg (by decide)]; exact hsqh')
            (by rw [hstn_reg (by decide)]; exact hkptr)
            (by rw [hstn_reg (by decide)]; exact hvptr)
            (by rw [hstn_reg (by decide)]; exact hdkptr)
            (by rw [hstn_reg (by decide)]; exact hdvptr)
        refine ⟨sF, hstep, hFpids, hFundef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hFohz, hFoffz, hFoffh, hFsqz, hFsqh⟩
        · -- DV at blocks < (n+1): split into < n (old) and = n (new)
          intro J e hJ he
          by_cases hlt : J < n * BM
          · -- old block: preserved off the freshly-written key block n
            rw [hFDVoff (bwdKBase s + J * BD + e) (by
              intro jL eF
              have : (n * BM + jL.val) * BD + eF.val ≠ J * BD + e := by
                have hjL := jL.isLt; have heF := eF.isLt
                have hge : n * BM ≤ n * BM + jL.val := by omega
                intro hc
                have : J * BD + e < (n * BM) * BD := by
                  have : J + 1 ≤ n * BM := hlt
                  calc J * BD + e < J * BD + BD := by omega
                    _ = (J + 1) * BD := by ring
                    _ ≤ (n * BM) * BD := Nat.mul_le_mul_right BD this
                have : (n * BM + jL.val) * BD ≥ (n * BM) * BD := Nat.mul_le_mul_right BD hge
                omega
              omega)]
            rw [hstnmem]; exact hDVlo J e hlt he
          · -- new block n: J = n*BM + jL
            have hJge : n * BM ≤ J := by omega
            have hjval : J - n * BM < BM := by
              have : J < (n + 1) * BM := hJ
              have : J < n * BM + BM := by rw [Nat.add_mul, Nat.one_mul] at this; exact this
              omega
            have := hFDV ⟨J - n * BM, hjval⟩ e he
            simp only at this
            rw [show n * BM + (J - n * BM) = J from by omega] at this
            exact this
        · -- DK at blocks < (n+1)
          intro J e hJ he
          by_cases hlt : J < n * BM
          · rw [hFDKoff (bwdKBase s + J * BD + e) (by
              intro jL eF
              have hjL := jL.isLt; have heF := eF.isLt
              have hge : n * BM ≤ n * BM + jL.val := by omega
              intro hc
              have hlb : J * BD + e < (n * BM) * BD := by
                have hJ1 : J + 1 ≤ n * BM := hlt
                calc J * BD + e < J * BD + BD := by omega
                  _ = (J + 1) * BD := by ring
                  _ ≤ (n * BM) * BD := Nat.mul_le_mul_right BD hJ1
              have : (n * BM + jL.val) * BD ≥ (n * BM) * BD := Nat.mul_le_mul_right BD hge
              omega)]
            rw [hstnmem]; exact hDKlo J e hlt he
          · have hJge : n * BM ≤ J := by omega
            have hjval : J - n * BM < BM := by
              have : J < n * BM + BM := by rw [Nat.add_mul, Nat.one_mul] at hJ; exact hJ
              omega
            have := hFDK ⟨J - n * BM, hjval⟩ e he
            simp only at this
            rw [show n * BM + (J - n * BM) = J from by omega] at this
            exact this
        · -- DV unchanged at blocks ≥ (n+1)
          intro J e hJ hJlt he
          rw [Nat.add_mul, Nat.one_mul] at hJ
          have hJgen : ¬ (∃ (jL : Fin BM) (eF : Fin BD), bwdKBase s + J * BD + e
              = bwdKBase s + (n * BM + jL.val) * BD + eF.val) := by
            rintro ⟨jL, eF, hc⟩
            have hjL := jL.isLt; have heF := eF.isLt
            have hlt : n * BM + jL.val + 1 ≤ J := by omega
            have hmul : (n * BM + jL.val + 1) * BD ≤ J * BD := Nat.mul_le_mul_right BD hlt
            rw [Nat.add_mul, Nat.one_mul] at hmul
            omega
          rw [hFDVoff (bwdKBase s + J * BD + e) (by intro jL eF hc; exact hJgen ⟨jL, eF, hc⟩)]
          rw [hstnmem]
          exact hDVhi J e (by omega) hJlt he
        · -- DK unchanged at blocks ≥ (n+1)
          intro J e hJ hJlt he
          rw [Nat.add_mul, Nat.one_mul] at hJ
          have hJgen : ¬ (∃ (jL : Fin BM) (eF : Fin BD), bwdKBase s + J * BD + e
              = bwdKBase s + (n * BM + jL.val) * BD + eF.val) := by
            rintro ⟨jL, eF, hc⟩
            have hjL := jL.isLt; have heF := eF.isLt
            have hlt : n * BM + jL.val + 1 ≤ J := by omega
            have hmul : (n * BM + jL.val + 1) * BD ≤ J * BD := Nat.mul_le_mul_right BD hlt
            rw [Nat.add_mul, Nat.one_mul] at hmul
            omega
          rw [hFDKoff (bwdKBase s + J * BD + e) (by intro jL eF hc; exact hJgen ⟨jL, eF, hc⟩)]
          rw [hstnmem]
          exact hDKhi J e (by omega) hJlt he
        · -- DQ accumulates through block n
          intro I e hI he
          rw [hFDQ I e hI he]
        · -- regions other than DV/DK/DQ preserved
          intro R' o hRDV hRDK hRDQ
          rw [hFother R' o hRDV hRDK hRDQ, hstnmem]; exact hother R' o hRDV hRDK hRDQ
        · rw [hFkp]
        · rw [hFvp]
        · rw [hFdkp]
        · rw [hFdvp])
  -- final invariant at `final ≥ nb`; the loop ran exactly `nb` blocks
  rw [stepStmts.cons_some hforstep, stepStmts.nil]
  -- final = nb (invariant ptrs pin n, but we just need the n = nb facts via final ≥ nb and the
  -- invariant's "blocks < n" coverage). Use final ≥ nb so all key rows < nb*BM are covered.
  obtain ⟨_, _, hDVfin, hDKfin, _, _, hDQfin, _, _, _, _, _, _, _, _, _, _⟩ := hPfinal
  refine ⟨sFinal, rfl, ?_, ?_, ?_⟩
  · -- DQ readback = bwdKernelDQSpecG
    intro I e hI he
    rw [hDQfin I e hI he, bwdKernelDQSpecG_blockSum]
    refine congrArg (s.readMem DQ (bwdKBase s + I * BD + e) + ·) ?_
    -- ∑ n' ∈ range final = ∑ n' : Fin nb  (the tail n' ≥ nb is causally zero: I < nb·BM ≤ n'·BM)
    rw [Finset.sum_fin_eq_sum_range]
    rw [← Finset.sum_range_add_sum_Ico _ hfinalge]
    have htail : (∑ n' ∈ Finset.Ico nb final, ∑ jL : Fin BM,
        bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I (n' * BM + jL.val)) *
          bwdKernelKG s K BD (n' * BM + jL.val) e) = 0 := by
      refine Finset.sum_eq_zero (fun n' hn' => ?_)
      rw [Finset.mem_Ico] at hn'
      refine Finset.sum_eq_zero (fun jL _ => ?_)
      have hIlt : I < n' * BM + jL.val := by
        have : nb * BM ≤ n' * BM := Nat.mul_le_mul_right BM hn'.1
        omega
      have hds0 : bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I (n' * BM + jL.val) = 0 := by
        simp only [bwdKernelDSG, bwdKernelPG, if_neg (by omega : ¬ n' * BM + jL.val ≤ I), zero_mul]
      rw [hds0, show bwdFp16 0 = 0 from by simp only [bwdFp16, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot, FloatDType.storeValue]; norm_num, zero_mul]
    rw [htail, add_zero]
    refine Finset.sum_congr rfl (fun n' hn' => ?_)
    rw [Finset.mem_range] at hn'
    rw [dif_pos hn']
  · intro J e hJ he
    exact hDVfin J e (lt_of_lt_of_le hJ (Nat.mul_le_mul_right BM hfinalge)) he
  · intro J e hJ he
    exact hDKfin J e (lt_of_lt_of_le hJ (Nat.mul_le_mul_right BM hfinalge)) he

set_option maxHeartbeats 1600000 in
open VeriTile.Examples.FA1MathCausal in
/-- **★ MAIN (forward, dimension-general).** Public symbolic-dimension forward
output summary for `triton_attention.py`'s `_fwd_kernel`. For arbitrary symbolic
`N_CTX`/`D_HEAD = BLOCK_DMODEL`/`BLOCK_M`/`BLOCK_N` and a contiguous layout, the
lowered kernel's `Out`/`L`/`M` writes Realize the genuine general closed-form
specs `fwdOutSpecG`/`fwdLSpecG`/`fwdMSpecG` — defined purely over the **input**
`Q`/`K`/`V` memory (causal natural-exp attention over the streaming KV span
`SEQ = (pids0+1)·BLOCK_M`), never over the kernel's own readback. Honest side
conditions: positive block dims, `BLOCK_N ∣ (pids0+1)·BLOCK_M`, contiguous
strides, output-offset injectivity, and the boundary
`pids1·(stride_qh/BLOCK_DMODEL) + (pids0+1)·BLOCK_M ≤ D0`. -/
theorem triton_attention_forward_output_summary_general
    (Q K V L M Out : RegionName) (s : BlockState) (sc : ℝ)
    (stride_qz stride_qh Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBD : 0 < BLOCK_DMODEL)
    (hdvd : BLOCK_N ∣ (s.pids 0 + 1) * BLOCK_M)
    (hbound : s.pids 1 * (stride_qh / BLOCK_DMODEL) + (s.pids 0 + 1) * BLOCK_M ≤ D0)
    (hLOut : L ≠ Out) (hMOut : M ≠ Out) (hLM : M ≠ L)
    (houtinj : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        (s.pids 1 * (stride_qh / BLOCK_DMODEL) + s.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val * 1))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (triton_attention_fwd_kernel Q K V L M Out sc
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s (s.pids 1 * (stride_qh / BLOCK_DMODEL)) D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s (s.pids 1 * (stride_qh / BLOCK_DMODEL)) BLOCK_DMODEL 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (fwdOutSpecG s Q K V (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL
              ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc idx))))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s (s.pids 1) N_CTX BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        fwdLSpecG s Q K (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc
          (Nat.mul_pos (Nat.succ_pos _) hBM) i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_fwd_kernel Q K V L M Out sc
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
        Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s (s.pids 1) N_CTX BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        fwdMSpecG s Q K (stride_qh / BLOCK_DMODEL) BLOCK_DMODEL ((s.pids 0 + 1) * BLOCK_M) BLOCK_M BLOCK_DMODEL sc
          (Nat.mul_pos (Nat.succ_pos _) hBM) i)) := by
  obtain ⟨sF, hstep, hO, hLrb, hMrb⟩ :=
    ta_execG Q K V L M Out s sc stride_qz stride_qh Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N
      hBM hBN hBD hdvd hbound hLOut hMOut hLM houtinj hundef
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact triton_attention_fwd_kernel_toAlgorithm_supported Q K V L M Out sc
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      stride_qz stride_qh BLOCK_DMODEL 1 stride_qz stride_qh BLOCK_DMODEL 1
      Z H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N
  · -- Out
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    rw [hstep] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_memcell]
    exact hO idx hActive
  · -- L
    unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro i
    rw [hstep] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    exact hLrb i
  · -- M
    unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro i
    rw [hstep] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    exact hMrb i

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
set_option maxHeartbeats 1600000 in
/-- **★ MAIN (backward gradients, multi-block general).** Public symbolic-dimension
backward-gradient summary for `triton_attention.py`'s `_bwd_kernel` over a full
`N_CTX = BLOCK_M · num_block` sequence (the multi-block KV/Q streaming loop).
For symbolic `num_block`/`N_CTX`/`BLOCK_M`/`BLOCK_DMODEL` with contiguous strides:

* `DQ` (stored `.real`) reads back as a **real** equal to the genuine general
  `bwdKernelDQSpecG` (`priorDQ + Σ_J fp16(ds)·k`, summed over **all** key rows);
* `DV`/`DK` are stored `tl.float16`, so read back at the **fp16 `MemCell`** level as
  `MemCell.of fp16 (real.cast fp16 (some <genuine real column sum>))` — the raw
  column sums `DV[J,e] = Σ_I fp16(p[I,J])·do[I,e]`,
  `DK[J,e] = Σ_I fp16(ds[I,J])·q[I,e]` over all query rows `I ∈ Fin (BLOCK_M·num_block)`.

Honest side conditions: positive block dims and `num_block`, `BD ∣ bwdKBase`, the
streaming boundary `bwdKBase/BD + num_block·BLOCK_M ≤ D0`, the index/stride
arithmetic `hbase`, input/output region disjointness, and the honest pids grid. All
specs are defined purely over the **input** `Q`/`K`/`V`/`DO`/`M`/`Delta`/`DQ`
memory — never over the kernel's own `exec` readback. -/
theorem triton_attention_bwd_grads_genuine_output_summary_general
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) (sc : ℝ)
    (BM BD D0 nb : Nat)
    (hBM : 0 < BM) (hBD : 0 < BD) (hnb : 0 < nb) (hbdvd : BD ∣ bwdKBase s)
    (hbound : bwdKBase s / BD + nb * BM ≤ D0)
    (hbase : (s.pids 0 / 4) * (32768 / BD) + (s.pids 0 % 4) * (8192 / BD) = bwdKBase s / BD)
    (hQDQ : Q ≠ DQ) (hKDQ : K ≠ DQ) (hVDQ : V ≠ DQ) (hDODQ : DO ≠ DQ)
    (hMDQ : M ≠ DQ) (hDeDQ : Delta ≠ DQ)
    (hDVDQ : DV ≠ DQ) (hDKDQ : DK ≠ DQ) (hDVDK : DV ≠ DK) (hDKDV : DK ≠ DV)
    (hin : ∀ R : RegionName, R = Q ∨ R = K ∨ R = V ∨ R = DO ∨ R = M ∨ R = Delta →
        R ≠ DV ∧ R ≠ DK ∧ R ≠ DQ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM)
        (fun idx : TileIndex [BM * nb, BD] => (DQ, bwdKBase s + idx.1.val * BD + idx.2.1.val)))
      (expected := fun idx : TileIndex [BM * nb, BD] =>
        bwdKernelDQSpecG s Q K V DO M Delta DQ BD (BM * nb) sc idx.1.val idx.2.1.val)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM)
        (fun idx : TileIndex [BM * nb, BD] => (DV, bwdKBase s + idx.1.val * BD + idx.2.1.val)))
      (expected := fun idx : TileIndex [BM * nb, BD] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ I : Fin (BM * nb),
            bwdFp16 (bwdKernelPG s Q K M BD (BM * nb) sc I.val idx.1.val) *
              bwdKernelDOG s DO BD I.val idx.2.1.val))))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta sc
        32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
        2 4 (BM * nb) D0 nb BM BD BM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BM * nb, BD] => idx.1.val < nb * BM)
        (fun idx : TileIndex [BM * nb, BD] => (DK, bwdKBase s + idx.1.val * BD + idx.2.1.val)))
      (expected := fun idx : TileIndex [BM * nb, BD] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (∑ I : Fin (BM * nb),
            bwdFp16 (bwdKernelDSG s Q K V DO M Delta BD (BM * nb) sc I.val idx.1.val) *
              bwdKernelQG s Q BD I.val idx.2.1.val))))) := by
  obtain ⟨sF, hexec, hDQ, hDV, hDK⟩ :=
    bwd_grads_execG s Q K V Out DO DQ DK DV L M Delta BM BD D0 nb sc
      hBM hBD hnb hbdvd hbound hbase hQDQ hKDQ hVDQ hDODQ hMDQ hDeDQ
      hDVDQ hDKDQ hDVDK hDKDV hin hundef
  -- coordinate bounds for a [BM*nb, BD] index
  have hidx1 : ∀ idx : TileIndex [BM * nb, BD], idx.1.val < BM * nb := fun idx => by
    have := idx.1.isLt; simpa [Nat.mul_comm] using this
  have hidx2 : ∀ idx : TileIndex [BM * nb, BD], idx.2.1.val < BD := fun idx => idx.2.1.isLt
  have hnbBM : nb * BM = BM * nb := Nat.mul_comm nb BM
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- toAlgorithm conjunct
    exact triton_attention_bwd_kernel_toAlgorithm_supported Q K V Out DO DQ DK DV L M Delta sc
      32768 8192 BD 1 32768 8192 BD 1 32768 8192 BD 1
      2 4 (BM * nb) D0 nb BM BD BM
  · -- DQ: real readback = bwdKernelDQSpecG
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx _hActive
    rw [hexec] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    exact hDQ idx.1.val idx.2.1.val (by rw [hnbBM]; exact hidx1 idx) (hidx2 idx)
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
    exact hDV idx.1.val idx.2.1.val (by rw [hnbBM]; exact hidx1 idx) (hidx2 idx)
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
    exact hDK idx.1.val idx.2.1.val (by rw [hnbBM]; exact hidx1 idx) (hidx2 idx)

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.TritonAttention
