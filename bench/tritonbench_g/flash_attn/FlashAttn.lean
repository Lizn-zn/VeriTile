import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention

/-!
# `flash_attn` — strict per-kernel correctness

`flash_attn.py`'s `_fwd_kernel` is a FlashAttention forward over
`tl.make_block_ptr` tiles: program `(start_m, off_bs_head)` loads its query
block, streams `K`/`V` blocks (`hi = (start_m+1)·BLOCK_M` when `IS_CAUSAL`,
else `SEQLEN`) running the online-softmax recurrence (running `max`, `denom`,
`out_buffer`) with `qk_scale = sm_scale · log2(e)` and per-block causal
masking, then stores the normalized `out_buffer / denom` to `O` and the
log-sum-exp `max + log2(denom)` to `L`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`grid = (cdiv(SEQLEN, BLOCK_M), BS·HEAD, 1)`, the
`num_warps`/`num_stages` choice, and how the runtime composes per-program
writes into the `O`/`L` buffers) is the *trusted boundary*, not a proof
obligation here. Because the program ids `start_m`/`off_bs_head` are
universally quantified (via `s`), the per-program statements cover every
program of the grid.

## Genuine closed-form spec (NOT self-referential)

The online-softmax recurrence this kernel runs (`qk = where(off_m ≥ start_n+off_n,
q·kᵀ·qk_scale, -inf)`, running `max`/`denom`/`out_buffer`, final `out_buffer /
denom`) computes a genuine base-2 attention closed form, with `qk_scale =
sm_scale · log2(e)` folded into `q`:

* **case 2 (`IS_CAUSAL = false`)** — every key contributes:
  `flashAttnOValueSpec` ≡ `attentionRealBase2PerKeyScale (qTile) (kTile) (vTile)
  (fun _ => qk_scale)` of the loaded Q/K/V tiles (constant per-key scale).
* **case 1 (`IS_CAUSAL = true`)** — key `j` contributes only when
  `j ≤ qStart + i` (the `-inf` mask zeroes future keys):
  `flashAttnOValueSpecCausal` ≡ `attentionRealBase2PerKeyScaleCausal …`.

These are the genuine `expected` values, defined over the loaded tiles, **not**
the kernel's own executed output. The mathematical heart (online-softmax fold ==
batch base-2 softmax, causal and non-causal) is proved sorry-free in
`VeriTile/Triton/Math/Attention.lean`
(`attentionRealBase2PerKeyScale_eq_streaming`,
`attentionRealBase2PerKeyScaleCausal_eq_streaming`, `osBlockStep_foldl_eq_batch`).

## Proof architecture

```
flash_attn_output_store_slice_compute_correct      ← O store (out_buffer / denom), genuine readback
  └─ flash_attn_output_store_slice_correct           algorithm-layer readback per lane
flash_attn_l_store_slice_compute_correct           ← L store (max + log2 denom), genuine readback
  └─ flash_attn_l_store_slice_correct
flash_attn_python_case{i}_surface_toAlgorithm_supported   surface lowers to the algorithm layer
flash_attn_python_case{i}_store_summary                   package the genuine store-only facts
```
(Offset injectivity discharged by `flash_attn_python_{output,l}_offset_injective`.)

## Scope status

This file currently delivers, **sorry-free**: (1) the faithful DSL surface and its
algorithm lowering; (2) the genuine closed-form spec definitions above plus their
streaming-softmax bridge in `Math/Attention.lean`; (3) the genuine final-store
readback `Realizes` theorems for both the `O` (`out_buffer / denom`) and `L`
(`max + log2 denom`) stores at the correct, injective offsets; (4) the
`exec`-side **block-pointer evaluation foundation** for the loop-invariant proof:
the general block-ptr construction/advance/load `evalOp` reductions now live
sorry-free in `VeriTile/Triton/Semantics/BlockPtrEval.lean` +
`VeriTile/Triton/Core/{Types,Shape}.lean` (`address_2d_offsets`,
`advance_2d_offsets`, the `blockPtr_*_index` forms, `inBounds_nil_*`), the
dynamic-loop driver `forRangeDyn_inv` lives in
`VeriTile/Triton/LoopInvariant.lean`, and the flash-attn-specific recipes
`flash_makeBlockPtrDyn_eval` / `flash_makeBlockPtr_rowcol_eval` /
`flash_advance_{col,row}_eval` / `flash_load_{K,Q}_eval` (below) specialize them
to this kernel's exact `make_block_ptr` AST.

Additionally banked sorry-free toward the invariant skeleton (this run): the
per-row online-softmax **running-state recurrence** `flashKV`/`flashKeysUpto`/
`flashState` (the `osStep` fold over the causally-filtered streamed key prefix),
its one-block advance `flashState_succ` (= the kernel's per-block update
`osBlockStep`, via the new `Math/Attention.lean` lemma `osBlockStep_eq_foldl_osStep`
+ the sorted-window split `flashKeysUpto_succ`), the full-window bridges
`flashState_full_eq_spec{,_causal}` (final state reads off `flashAttnOValueSpec{,
Causal}`), the `max`/`denom` channel-independence lemmas
`flashState_{fst,snd_fst}_eq`, and the `attnInvariant` definition binding the
kernel's 14 live registers to `flashState` after `c` blocks.

The **remaining stage** (not yet closed) is the exec-side `preLoop`/`attn_step`/
`attn_postLoop` proofs over this `attnInvariant` (threading the loop-body op-eval
recipes — including the causal `tl.where(off_m ≥ start_n+off_n, qk, -inf)` mask —
and the WithBot `reduceMaxDrop`/`realExp2 ⊥ = 0`/masked-dot bridges into one
`flashState_succ` step), plus the dual `O`/`L` stores composed via
`forRangeDyn_inv` and read off through `flashState_full_eq_spec{,_causal}`. That
mirrors `VeriTile/Examples/AttentionForwardClosedForm.lean`'s preLoop/step/postLoop
skeleton, now retargeted onto the block-pointer foundation and the `flashState`
recurrence above. No self-referential / tautological summary is asserted in its
place.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `exp2`, `log2`, `tl.dot`,
and the `sm_scale · log2(e)` scaling are not modeled at the bit level);
`@triton.autotune`/`num_warps`/`num_stages` are not modeled. Side conditions:
layout `(BS,HEAD,SEQLEN,DIM) = (2,2,128,64)`, `BLOCK_M = 128`, `BLOCK_N = 64`,
strides `(16384, 8192, 64, 1)`; case 1 is `IS_CAUSAL = true`, case 2 is `false`.
-/

namespace VeriTile.Bench.TritonBenchG.FlashAttn

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful DSL port of `flash_attn.py`'s `_fwd_kernel`. -/
def flash_attn_fwd_kernel_surface
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      _BS _HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  qkv_base_offset = off_bs_head * $(stride_q_head)
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, (out_buffer).to(tl.float16))
}

/-- The full flash-attention forward surface lowers to the algorithm layer. -/
theorem flash_attn_fwd_kernel_surface_toAlgorithm_supported
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ∃ alg, (flash_attn_fwd_kernel_surface Q K V L O sm_scale stride_q_bs
      stride_q_head stride_q_seqlen stride_q_dim stride_k_bs stride_k_head
      stride_k_seqlen stride_k_dim stride_v_bs stride_v_head stride_v_seqlen
      stride_v_dim stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N IS_CAUSAL).toAlgorithm? =
        Except.ok alg := by
  simp [flash_attn_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `flash_attn.py`'s
`_fwd_kernel`.

The full kernel streams over K/V blocks, computes a numerically stable attention
accumulator, and also writes the log-sum-exp vector `L`. This slice starts after
`out_buffer = out_buffer / denom[:, None]` with a precomputed `OutBuffer` tile
and proves the final unmasked `O_block_ptr` writeback. It preserves the source
base offset, which is derived from `stride_q_head`. The inner `tl.float32`
online-softmax state and K/V dot loop are outside this slice. -/
def flash_attn_output_store_slice
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(DIM))
  out_buffer = tl.load(OutBuffer + off_bs_head * $(stride_buf_h) +
      offs_m[:, None] * $(stride_buf_m) + offs_d[None, :] * $(stride_buf_d))
  tl.store(O + off_bs_head * $(stride_q_head) +
      offs_m[:, None] * $(stride_o_seqlen) + offs_d[None, :] * $(stride_o_dim),
      (out_buffer).to(tl.float16))
}

/-- Surface transcription of `flash_attn.py`'s final `L` vector store.

The full kernel computes the streaming row max and denominator, then stores
`max + tl.math.log2(denom)` into `L + off_bs_head * SEQLEN + off_m`. This
surface starts from precomputed `Max` and `Denom` row tiles and preserves that
addressing. -/
def flash_attn_l_store_slice
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_row = tl.load(Max + off_bs_head * $(stride_max_h) + off_m * $(stride_max_m))
  denom = tl.load(Denom + off_bs_head * $(stride_den_h) + off_m * $(stride_den_m))
  tl.store(L + off_bs_head * $(SEQLEN) + off_m, max_row + tl.log2(denom))
}

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  idx.2.1.val

def bufferOffset
    (s : BlockState)
    (stride_buf_h stride_buf_m stride_buf_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_buf_h +
    mIndex s BLOCK_M idx.1 * stride_buf_m + dIndex idx * stride_buf_d

def outOffset
    (s : BlockState)
    (stride_q_head stride_o_seqlen stride_o_dim BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_q_head +
    mIndex s BLOCK_M idx.1 * stride_o_seqlen + dIndex idx * stride_o_dim

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, offsetFn k ≠ o :=
        fun k hk => h k (List.mem_cons_of_mem hd hk)
      have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
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

private theorem scatter_memcell_fp16_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i)
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i))) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i))
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := by
    intro k hk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (offsetFn i) l₂ _ h_l2_not_in]
  unfold BlockState.writeMemTyped BlockState.writeMemAs
  change
    (if region = region ∧ offsetFn i = offsetFn i then
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
    else
      (List.foldl
        (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
        s l₁).mem region (offsetFn i))
      =
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [if_pos ⟨rfl, rfl⟩]

/-- Algorithm-layer correctness for the final FlashAttention output store. -/
theorem flash_attn_output_store_slice_correct
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, DIM],
      let outAddr := outOffset s stride_q_head stride_o_seqlen stride_o_dim
        BLOCK_M idx
      (exec (flash_attn_output_store_slice OutBuffer O stride_buf_h
            stride_buf_m stride_buf_d stride_q_head stride_o_seqlen
            stride_o_dim BLOCK_M DIM) s).map (·.mem O outAddr)
        = some (MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem OutBuffer
                (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))) := by
  intro idx
  simp [exec, flash_attn_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, mIndex, dIndex,
        bufferOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, DIM] → Nat :=
    fun idx =>
      s.pids 1 * stride_q_head +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_o_seqlen +
        idx.2.1.val * stride_o_dim
  let valueFn : TileIndex [BLOCK_M, DIM] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (s.readMem OutBuffer
          (s.pids 1 * stride_buf_h +
            (s.pids 0 * BLOCK_M + idx.1.val) * stride_buf_m +
            idx.2.1.val * stride_buf_d)))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMemTyped .fp16 O (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, DIM])).mem O (offsetFn idx) =
    MemCell.of .fp16
      (FloatDType.real.cast FloatDType.fp16
        (some (s.readMem OutBuffer
          (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))
  rw [scatter_memcell_fp16_nd _ _ _ hOffsetInj idx]
  simp [valueFn, bufferOffset, mIndex, dIndex, FloatDType.cast,
    FloatDType.ofReal, FloatDType.storeValue, FloatDType.ofWithBot, FloatDType.toWithBot]

/-- Compute-facing correctness for the final FlashAttention output store. -/
theorem flash_attn_output_store_slice_compute_correct
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O stride_buf_h
        stride_buf_m stride_buf_d stride_q_head stride_o_seqlen stride_o_dim
        BLOCK_M DIM)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] =>
        some (O, outOffset s stride_q_head stride_o_seqlen stride_o_dim
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_attn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := flash_attn_output_store_slice_correct OutBuffer O stride_buf_h
    stride_buf_m stride_buf_d stride_q_head stride_o_seqlen stride_o_dim
    BLOCK_M DIM s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Output offset for the FlashAttention `L` row store. -/
def lOffset (s : BlockState) (SEQLEN BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * SEQLEN + mIndex s BLOCK_M i

/-- Source offset for the precomputed `Max` row read. -/
def maxOffset
    (s : BlockState) (stride_max_h stride_max_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_max_h + mIndex s BLOCK_M i * stride_max_m

/-- Source offset for the precomputed `Denom` row read. -/
def denomOffset
    (s : BlockState) (stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_den_h + mIndex s BLOCK_M i * stride_den_m

/-- Spec for the `L` row store value: `max + log(denom) / log(2)`, mirroring
the kernel's `tl.log2` semantics (`Real.log x / Real.log 2`). -/
noncomputable def lStoreSpec
    (s : BlockState) (Max Denom : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  s.readMem Max (maxOffset s stride_max_h stride_max_m BLOCK_M i) +
    Real.log (s.readMem Denom
      (denomOffset s stride_den_h stride_den_m BLOCK_M i)) / Real.log 2

/-- Algorithm-layer correctness for the `L` row store slice. -/
theorem flash_attn_l_store_slice_correct
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lOffset s SEQLEN BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lOffset s SEQLEN BLOCK_M i
      (exec (flash_attn_l_store_slice Max Denom L stride_max_h stride_max_m
            stride_den_h stride_den_m SEQLEN BLOCK_M) s).map (·.readMem L outAddr)
        = some
            (lStoreSpec s Max Denom stride_max_h stride_max_m stride_den_h
              stride_den_m BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        s.pids 1 * SEQLEN + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 1 * SEQLEN + s.pids 0 * BLOCK_M + a.val =
        s.pids 1 * SEQLEN + s.pids 0 * BLOCK_M + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [exec, flash_attn_l_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul]
  simp only [lOffset, mIndex, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [lStoreSpec, maxOffset, denomOffset, mIndex, Nat.add_assoc]

/-- Compute-facing correctness for the `L` row store slice. -/
theorem flash_attn_l_store_slice_compute_correct
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lOffset s SEQLEN BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L stride_max_h
        stride_max_m stride_den_h stride_den_m SEQLEN BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i =>
        lStoreSpec s Max Denom stride_max_h stride_max_m stride_den_h
          stride_den_m BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_attn_l_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := flash_attn_l_store_slice_correct Max Denom L stride_max_h
    stride_max_m stride_den_h stride_den_m SEQLEN BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

theorem flash_attn_python_output_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] => outOffset s 8192 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem flash_attn_python_l_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 128 => lOffset s 128 128 i) := by
  intro a b h
  simp [lOffset, mIndex] at h
  exact Fin.ext h

theorem flash_attn_python_output_store_compute_correct
    (OutBuffer O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O
        8192 64 1 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s 8192 64 1 128 idx))))) := by
  exact flash_attn_output_store_slice_compute_correct OutBuffer O
    8192 64 1 8192 64 1 128 64 s
    (flash_attn_python_output_offset_injective s)

theorem flash_attn_python_l_store_compute_correct
    (Max Denom L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L 128 1 128 1 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i =>
        lStoreSpec s Max Denom 128 1 128 1 128 i) := by
  exact flash_attn_l_store_slice_compute_correct Max Denom L
    128 1 128 1 128 128 s
    (flash_attn_python_l_offset_injective s)

/-- Python case 1 full surface lowering: causal forward attention for
`B=2,H=2,SEQLEN=128,DIM=64`, `BLOCK_M=128`, `BLOCK_N=64`. -/
theorem flash_attn_python_case1_surface_toAlgorithm_supported
    (Q K V L O : RegionName) :
    ∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.true).toAlgorithm? =
        Except.ok alg := by
  exact flash_attn_fwd_kernel_surface_toAlgorithm_supported Q K V L O (1.0 : ℝ)
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    2 2 128 128 64 64 Bool.true

/-- Python case 2 full surface lowering: non-causal forward attention for the
same checked layout. -/
theorem flash_attn_python_case2_surface_toAlgorithm_supported
    (Q K V L O : RegionName) :
    ∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.false).toAlgorithm? =
        Except.ok alg := by
  exact flash_attn_fwd_kernel_surface_toAlgorithm_supported Q K V L O (1.0 : ℝ)
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    2 2 128 128 64 64 Bool.false

/-- Python case 1 store-slice coverage retained for the final-store proof. -/
theorem flash_attn_python_case1_store_summary
    (Q K V L O OutBuffer Max Denom : RegionName) (s : BlockState) :
    (∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.true).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O
        8192 64 1 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s 8192 64 1 128 idx)))))) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L 128 1 128 1 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i =>
        lStoreSpec s Max Denom 128 1 128 1 128 i)) := by
  constructor
  · exact flash_attn_python_case1_surface_toAlgorithm_supported Q K V L O
  constructor
  · exact flash_attn_python_output_store_compute_correct OutBuffer O s
  · exact flash_attn_python_l_store_compute_correct Max Denom L s

/-- Python case 2 store-slice coverage retained for the final-store proof. -/
theorem flash_attn_python_case2_store_summary
    (Q K V L O OutBuffer Max Denom : RegionName) (s : BlockState) :
    (∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O
        8192 64 1 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s 8192 64 1 128 idx)))))) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L 128 1 128 1 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i =>
        lStoreSpec s Max Denom 128 1 128 1 128 i)) := by
  constructor
  · exact flash_attn_python_case2_surface_toAlgorithm_supported Q K V L O
  constructor
  · exact flash_attn_python_output_store_compute_correct OutBuffer O s
  · exact flash_attn_python_l_store_compute_correct Max Denom L s

/-! ## Genuine closed-form output spec (replaces the self-referential summary)

The kernel's online-softmax recurrence computes a base-2 attention closed form
over the loaded Q/K/V tiles, with the scale `qk_scale = sm_scale · log2(e)`
folded into `q`. These definitions give the genuine `expected` `O`-store value —
defined over the loaded tiles, **not** the kernel's own executed output — for the
two Python cases. The streaming-softmax math heart that justifies them
(online-softmax fold == batch base-2 softmax, causal and non-causal) is proved
sorry-free in `VeriTile/Triton/Math/Attention.lean`. -/

open VeriTile.Triton (attentionRealBase2PerKeyScale attentionRealBase2PerKeyScaleCausal)

/-- The base-2 log-of-`e` constant the kernel folds into `qk_scale`
(`q = (q · sm_scale · 1.44269504).to(fp16)`). This is the *exact decimal literal*
`1.44269504` the Triton source uses (a truncation of the true `log2(e) = 1/log 2 ≈
1.4426950408889634`); the spec's per-key scale `sm_scale · log2e` is therefore the
genuine scale the kernel actually computes, folded into `q`. -/
def log2e : ℝ := 1.44269504

/-- Per-(batch,head) base offset `off_bs_head · stride_q_head = pid₁ · 8192` for
the Python layout. -/
def flashBaseOffset (s : BlockState) (stride_q_head : Nat) : Nat :=
  s.pids 1 * stride_q_head

/-- Loaded `Q` tile as a function of memory. Under the Python layout
(`stride_q_seqlen = DIM`, `stride_q_dim = 1`) row `i`, head lane `e` of the block
sits at `base + (pid₀·BLOCK_M + i)·DIM + e`. -/
noncomputable def qTile (s : BlockState) (Q : RegionName)
    (stride_q_head DIM BLOCK_M : Nat) : TileIndex [BLOCK_M, DIM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (flashBaseOffset s stride_q_head + mIndex s BLOCK_M i * DIM + e.val)

/-- Loaded `K` tile (key `j`, head lane `e`) at `base + j·DIM + e`. -/
noncomputable def kTile (s : BlockState) (K : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (flashBaseOffset s stride_q_head + j.val * DIM + e.val)

/-- Loaded `V` tile (key `j`, channel `d`) at `base + j·DIM + d`. -/
noncomputable def vTile (s : BlockState) (V : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (flashBaseOffset s stride_q_head + j.val * DIM + d.val)

/-- Genuine non-causal (`IS_CAUSAL = false`, Python case 2) closed-form `O`-store
value: the base-2 attention of the loaded Q/K/V tiles with the constant per-key
scale `qk_scale = sm_scale · log2(e)`. Every key contributes. -/
noncomputable def flashAttnOValueSpec
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : ℝ :=
  attentionRealBase2PerKeyScale
    (qTile s Q stride_q_head DIM BLOCK_M)
    (kTile s K stride_q_head DIM SEQLEN)
    (vTile s V stride_q_head DIM SEQLEN)
    (fun _ : Fin SEQLEN => sm_scale * log2e)
    idx

/-- Genuine causal (`IS_CAUSAL = true`, Python case 1) closed-form `O`-store value:
the base-2 attention restricted to keys `j ≤ pid₀·BLOCK_M + i` — the per-element
`tl.where(off_m ≥ start_n + off_n, qk, -inf)` mask zeroes future keys. -/
noncomputable def flashAttnOValueSpecCausal
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : ℝ :=
  attentionRealBase2PerKeyScaleCausal
    (qTile s Q stride_q_head DIM BLOCK_M)
    (kTile s K stride_q_head DIM SEQLEN)
    (vTile s V stride_q_head DIM SEQLEN)
    (fun _ : Fin SEQLEN => sm_scale * log2e)
    (s.pids 0 * BLOCK_M)
    idx

/-- The genuine non-causal `O`-value spec unfolds to the streaming online-softmax
fold over every key (the form the `exec`-side loop produces). Sorry-free bridge to
`Math/Attention.lean`; the remaining `exec` proof has to identify the kernel's
`out_buffer/denom` with this fold. -/
theorem flashAttnOValueSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (i : Fin BLOCK_M) (d : Fin DIM) :
    flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M
        (i, d, PUnit.unit)
      = (let st := (VeriTile.Triton.attnKeyList
            (qTile s Q stride_q_head DIM BLOCK_M)
            (kTile s K stride_q_head DIM SEQLEN)
            (vTile s V stride_q_head DIM SEQLEN)
            (fun _ : Fin SEQLEN => sm_scale * log2e) i d).foldl
          VeriTile.Triton.osStep (0, 0, 0)
         st.2.2 / st.2.1) :=
  VeriTile.Triton.attentionRealBase2PerKeyScale_eq_streaming _ _ _ _ i d

/-- The genuine causal `O`-value spec unfolds to the streaming fold over the
causally filtered key list. Sorry-free bridge to `Math/Attention.lean`. -/
theorem flashAttnOValueSpecCausal_eq_streaming
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (i : Fin BLOCK_M) (d : Fin DIM) :
    flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M
        (i, d, PUnit.unit)
      = (let st := (VeriTile.Triton.attnKeyListCausal
            (qTile s Q stride_q_head DIM BLOCK_M)
            (kTile s K stride_q_head DIM SEQLEN)
            (vTile s V stride_q_head DIM SEQLEN)
            (fun _ : Fin SEQLEN => sm_scale * log2e)
            (s.pids 0 * BLOCK_M) i d).foldl
          VeriTile.Triton.osStep (0, 0, 0)
         st.2.2 / st.2.1) :=
  VeriTile.Triton.attentionRealBase2PerKeyScaleCausal_eq_streaming _ _ _ _ _ i d

/-! ## exec-side block-pointer eval recipes (toward the loop-invariant proof)

The remaining gap is the `exec`-side proof that the `make_block_ptr` streaming
loop realizes `flashAttnOValueSpec{,Causal}`. The reusable foundation for that
proof — the block-pointer construction/advance/load `evalOp` reductions — now
lives sorry-free in `VeriTile/Triton/Semantics/BlockPtrEval.lean` and
`VeriTile/Triton/Core/{Types,Shape}.lean`. These local wrappers specialize them
to `flash_attn`'s exact AST: `Q`/`O` use `makeBlockPtrDynOffsets`
(`offsets=(start_m·BLOCK_M, 0)`), `K`/`V` use `makeBlockPtrDyn`
(`offsets=(0,0)`), and the per-block step is `tl.advance`. They are the
flash-attn analogues of the `attention_kernel` branch's
`makeBlockPtrDyn_eval`/`makeBlockPtr_rowcol_eval`/`advance_col_eval`/
`advance_row_eval`, retargeted here. -/

set_option maxHeartbeats 1000000 in
/-- **`K`/`V` block-ptr construction** (`offsets=(0,0)`, dynamic base):
`tl.make_block_ptr(base=K+qkv_base_offset, …)` evaluates to the constant
`BlockPtr` tile with `baseOffset = base`. -/
theorem flash_makeBlockPtrDyn_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape)
    (strides offsets : List Nat) (s : BlockState) (base : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base)) :
    evalOp (.makeBlockPtrDyn region baseOffset parentShape blockShape strides offsets) s
      = some (⟨fun _ : TileIndex blockShape =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := offsets }⟩) := by
  simp only [evalOp, hb, Option.bind]
  rfl

set_option maxHeartbeats 1000000 in
/-- **`Q`/`O` block-ptr construction** (`offsets=(start_m·BLOCK_M, 0)`, dynamic
row offset, literal `0` column): packages the constant `BlockPtr` tile with the
resolved row offset. -/
theorem flash_makeBlockPtr_rowcol_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape) (strides : List Nat)
    (rowOp : Op .nat []) (s : BlockState) (base rowOff : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base))
    (hr : evalOp rowOp s = some (Tile.scalar rowOff)) :
    evalOp (.makeBlockPtrDynOffsets region baseOffset parentShape blockShape strides
        [rowOp, Op.constNat 0]) s
      = some (⟨fun _ : TileIndex blockShape =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := [rowOff, 0] }⟩) := by
  rw [makeBlockPtr2_eval]
  simp only [hb, hr, evalOp, Option.bind, List.mapM, List.mapM.loop, Tile.scalar]
  rfl

set_option maxHeartbeats 1000000 in
/-- **`tl.advance(K_block_ptr, [0, BLOCK_N])`**: advances the column offset of a
`[rowOff=0, colOff]` block pointer by `BLOCK_N`. -/
theorem flash_advance_col_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS colOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [0, colOff] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [0, d]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [0, colOff + d] }⟩) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockPtr.advance_2d_offsets]

set_option maxHeartbeats 1000000 in
/-- **`tl.advance(V_block_ptr, [BLOCK_N, 0])`**: advances the row offset of a
`[rowOff, colOff=0]` block pointer by `BLOCK_N`. -/
theorem flash_advance_row_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS rowOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [rowOff, 0] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [d, 0]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [rowOff + d, 0] }⟩) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockPtr.advance_2d_offsets]

set_option maxHeartbeats 1000000 in
/-- **`k = tl.load(K_block_ptr)`** (no boundary check): a `[0, colOff]`-offset
block-ptr load reads `readMem` at the contiguous per-lane address
`base + i·strideT + (colOff + j)·strideS`. With `K`'s `strides=(1, DIM)` this is
key `colOff + j`, head-lane `i` — exactly a `kTile` read after `colOff` advances.
Directly specializes the library lemma `load_blockPtr_K_eval`. -/
theorem flash_load_K_eval
    (region : RegionName) (base rows cols BT BS strideT strideS colOff : Nat)
    (ptrOp : Op .blockPtr [BT, BS]) (s : BlockState)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [0, colOff] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [BT, BS] =>
          some (s.readMem region
            (base + idx.1.val * strideT + (colOff + idx.2.1.val) * strideS))⟩ :=
  load_blockPtr_K_eval region base rows cols BT BS strideT strideS colOff ptrOp s hp

set_option maxHeartbeats 1000000 in
/-- **`q = tl.load(Q_block_ptr)`** / `v = tl.load(V_block_ptr)` (no boundary
check): a `[rowOff, 0]`-offset block-ptr load reads `readMem` at
`base + (rowOff + i)·strideT + j·strideS`. With `Q`/`V`'s `strides=(DIM, 1)` this
is row `rowOff + i`, channel `j`. Directly specializes `load_blockPtr_Q_eval`. -/
theorem flash_load_Q_eval
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff : Nat)
    (ptrOp : Op .blockPtr [BT, BS]) (s : BlockState)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, 0] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [BT, BS] =>
          some (s.readMem region
            (base + (rowOff + idx.1.val) * strideT + idx.2.1.val * strideS))⟩ :=
  load_blockPtr_Q_eval region base rows cols BT BS strideT strideS rowOff ptrOp s hp

/-! ## Loop-body per-statement op-eval recipes

The 15-statement `forRangeDyn` body of `flash_attn`'s `_fwd_kernel` evaluates,
statement by statement, via these recipes (the flash-attn analogues of the
`AttentionForwardClosedForm` template's `*_op_eval` family, retargeted onto the
block-pointer foundation and the causal `ifThen`/`where` `-inf` mask). Each is a
standalone `evalOp` reduction with abstract register-readback hypotheses, so the
step lemma threads them through `stepStmts.cons_some` without ever reducing a
nested `setReg` literal state. -/

/-- `evalOp` helper for `tl.math.exp2` (`Op.exp2`): `Tile.uop realExp2`. -/
theorem evalOp_exp2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- `evalOp` helper for the `>=` causal predicate (`Op.ge`), which has no
dedicated `@[simp]` form (like `floorDiv`/`mod`). -/
theorem evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` helper for `tl.math.log2` (`Op.log2`): `Tile.uop realLog2`. -/
theorem evalOp_log2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.log2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realLog2 va)) := by
  simp [evalOp]

/-- **`qk = tl.zeros([BLOCK_M, BLOCK_N])` statement eval** (loop body L2): the
all-`0` tile, matching the neutral pre-mask scores. -/
theorem flash_qkzeros_op_eval (s : BlockState) (BM BN : Nat) :
    evalOp (Op.full [BM, BN] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- **Causal `where` statement eval** (loop body L3, inside `ifThen IS_CAUSAL`):
`qk = tl.where(off_m[:,None] >= start_n + off_n[None,:], qk, -inf)`. Given the
`off_m`/`off_n` index vectors, `start_n = SN`, and the running `qk` tile, the
masked tile selects `qk` where `off_m_i ≥ SN + j` and `⊥` (`-inf`) otherwise. -/
theorem flash_where_op_eval (s : BlockState) (BM BN SN : Nat)
    (gm : Fin BM → Nat) (qktile : Tile .real [BM, BN])
    (hom : s.regs .nat [BM] "off_m" = some (Tile.vec gm))
    (hon : s.regs .nat [BN] "off_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "off_m"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "off_n"))))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) s
      = some ⟨fun idx : TileIndex [BM, BN] =>
          if SN + idx.2.1.val ≤ gm idx.1 then qktile.data idx else (⊥ : WithBot ℝ)⟩ := by
  have hexpM : @evalOp TileDType.nat [BM, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "off_m")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec gm)) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hom
  have hexpN : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "off_n")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hon
  have haddN : @evalOp TileDType.nat [1, BN]
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "off_n"))) s
        = some (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar SN)
            (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val)))) := by
    rw [evalOp_add]
    rw [show evalOp (Op.ref .nat [] "start_n") s = some (Tile.scalar SN) from by rw [evalOp_ref, hsn]]
    rw [hexpN]
    rfl
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where, evalOp_ge]
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

/-- **`qk += tl.dot(q, k)` statement eval** (loop body L4): adds the `q·k` dot to
the (possibly causally-masked) `qk` tile. `q` is fp16 (the scaled/cast query). -/
theorem flash_qkdot_op_eval (s : BlockState) (BM BN DIM : Nat)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .fp16 [BM, DIM]) (ktile : Tile .real [DIM, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .fp16 [BM, DIM] "q" = some qtile)
    (hk : s.regs .real [DIM, BN] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, DIM] "q"))
          (Op.ref .real [DIM, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qktile (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, DIM]) ktile)) := by
  have hcb : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, DIM] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, DIM]) := by
    rw [evalOp_castFloat]; simp [hq]
  have hcb2 : @evalOp TileDType.real [BM, DIM] (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, DIM] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, DIM]) := hcb
  have hdotN : evalOp (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, DIM] "q")) (Op.ref .real [DIM, BN] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, DIM]) ktile) := by
    rw [evalOp_dot]; simp [hcb2, hk]
  have hdotN2 : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, DIM] "q")) (Op.ref .real [DIM, BN] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, DIM]) ktile) := hdotN
  rw [evalOp_add]; simp only [evalOp_ref, hqk, hdotN2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`max_new = tl.maximum(max, tl.max(qk, 1))` statement eval** (loop body L5).
Same `where`/`gt`/`reduceMax` shape as the template's `mij_op_eval`. -/
theorem flash_maxnew_op_eval (s : BlockState) (BM BN : Nat)
    (mtile : Tile .real [BM]) (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hmax : s.regs .real [BM] "max" = some mtile)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "max")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")))
        (Op.ref .real [BM] "max")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) := by
  have hrmaxN : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [BM]
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmax, hrmax, Option.bind_eq_bind, Option.bind_some]

/-- **`alpha = tl.math.exp2(max - max_new)` statement eval** (loop body L6). -/
theorem flash_alpha_op_eval (s : BlockState) (BM : Nat) (mtile mnewtile : Tile .real [BM])
    (hmax : s.regs .real [BM] "max" = some mtile) (hmnew : s.regs .real [BM] "max_new" = some mnewtile) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "max") (Op.ref .real [BM] "max_new"))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewtile)) := by
  rw [evalOp_exp2]; simp [evalOp_sub, hmax, hmnew]

/-- **`nume = tl.math.exp2(qk - max_new[:, None])` statement eval** (loop body L7).
Composes the `qk - max_new[:,None]` shift (template `qk2_op_eval` shape) with `exp2`. -/
theorem flash_nume_op_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM, BN]) (mnewtile : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) (hmnew : s.regs .real [BM] "max_new" = some mnewtile) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "max_new")))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qktile
          (Tile.expandDim ⟨1, hax⟩ mnewtile))) := by
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "max_new")) s
      = some (Tile.expandDim ⟨1, hax⟩ mnewtile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmnew
  rw [evalOp_exp2, evalOp_sub]
  simp only [evalOp_ref, hqk, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`out_scale = denom * 0 + alpha` statement eval** (loop body L8): the `denom*0`
zeroes out, leaving `alpha` (a `0·denom + alpha` form the kernel uses to keep
`out_scale` the right shape). -/
theorem flash_outscale_op_eval (s : BlockState) (BM : Nat) (dtile atile : Tile .real [BM])
    (hd : s.regs .real [BM] "denom" = some dtile) (ha : s.regs .real [BM] "alpha" = some atile) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM] "denom") (Op.const 0))
        (Op.ref .real [BM] "alpha")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul Broadcast.scalarR dtile (Tile.scalar (some 0))) atile) := by
  rw [evalOp_add]; simp [evalOp_mul, evalOp_ref, evalOp_const, hd, ha]

/-- **`out_buffer *= out_scale[:, None]` statement eval** (loop body L9). Mirrors
the template's `acc1_op_eval` (rescale by the row factor). -/
theorem flash_outbuf_rescale_op_eval (s : BlockState) (BM DIM : Nat) (hax : 1 < [BM].length.succ)
    (obtile : Tile .real [BM, DIM]) (ostile : Tile .real [BM])
    (hob : s.regs .real [BM, DIM] "out_buffer" = some obtile) (hos : s.regs .real [BM] "out_scale" = some ostile) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM, DIM] "out_buffer")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "out_scale"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) obtile
          (Tile.expandDim ⟨1, hax⟩ ostile)) := by
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "out_scale")) s
      = some (Tile.expandDim ⟨1, hax⟩ ostile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hos
  rw [evalOp_mul]; simp only [evalOp_ref, hob, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`nume = (nume).to(tl.float16)` statement eval** (loop body, `nume` fp16
round-trip before the value dot). Template `pfp16_op_eval` analogue. -/
theorem flash_numefp16_op_eval (s : BlockState) (BM BN : Nat) (ntile : Tile .real [BM, BN])
    (hn : s.regs .real [BM, BN] "nume" = some ntile) :
    evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "nume")) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ntile.data i)⟩ : Tile .fp16 [BM, BN]) := by
  rw [evalOp_castFloat]; simp [hn]

/-- **`out_buffer += tl.dot((nume).to(fp16), v)` statement eval** (loop body L10).
Template `acc2_op_eval` analogue: the `nume` fp16 round-trip is identity in the
model, the dot accumulates the value contribution. -/
theorem flash_outbuf_acc_op_eval (s : BlockState) (BM BN DIM : Nat)
    (ob1tile : Tile .real [BM, DIM]) (ntile : Tile .real [BM, BN]) (vtile : Tile .real [BN, DIM])
    (hob : s.regs .real [BM, DIM] "out_buffer" = some ob1tile)
    (hnf16 : s.regs .fp16 [BM, BN] "nume" = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ntile.data i)⟩ : Tile .fp16 [BM, BN]))
    (hv : s.regs .real [BN, DIM] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) (Op.ref .real [BM, DIM] "out_buffer")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "nume")) (Op.ref .real [BN, DIM] "v"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) ob1tile
          (Tile.dot [] ntile vtile)) := by
  have hcb : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "nume")) s = some ntile := by
    rw [evalOp_castFloat]; simp [hnf16]; ext i; simp [FloatDType.cast]
  have hcb2 : @evalOp TileDType.real [BM, BN] (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "nume")) s = some ntile := hcb
  have hdotN : evalOp (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "nume")) (Op.ref .real [BN, DIM] "v")) s
      = some (Tile.dot [] ntile vtile) := by rw [evalOp_dot]; simp [hcb2, hv]
  have hdotN2 : @evalOp TileDType.real [BM, DIM]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "nume")) (Op.ref .real [BN, DIM] "v")) s
      = some (Tile.dot [] ntile vtile) := hdotN
  rw [evalOp_add]; simp only [evalOp_ref, hob, hdotN2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`denom = denom * alpha + tl.sum(nume, 1)` statement eval** (loop body L11).
Template `li_op_eval`/`lij_op_eval` shapes composed: the running denominator
rescales by `alpha` and adds the new block's row-sum of `nume`. -/
theorem flash_denom_op_eval (s : BlockState) (BM BN : Nat)
    (dtile atile : Tile .real [BM]) (ntile : Tile .real [BM, BN])
    (hd : s.regs .real [BM] "denom" = some dtile) (ha : s.regs .real [BM] "alpha" = some atile)
    (hn : s.regs .real [BM, BN] "nume" = some ntile) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "denom") (Op.ref .real [BM] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "nume"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) dtile atile)
          (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ntile)) := by
  have hsumN : @evalOp TileDType.real [BM]
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "nume")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ntile) := by
    have : @evalOp TileDType.real [BM]
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "nume")) s
        = (do let vx ← evalOp (Op.ref .real [BM, BN] "nume") s;
              some (Tile.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false vx)) :=
      evalOp_reduceSum _ _ _ _
    rw [this]; simp only [evalOp_ref, hn, Option.bind_some, Option.bind_eq_bind]; rfl
  rw [evalOp_add]
  simp only [evalOp_mul, evalOp_ref, hd, ha, hsumN, Option.bind_eq_bind, Option.bind_some]; rfl

/-! ## Per-row online-softmax running state (the loop invariant's register math)

The loop carries, per output row `i`, the running `osStep` state over all keys
streamed so far (`global index < c · BLOCK_N`) that pass the causal filter
(`j ≤ qStart + i`). The invariant binds the kernel's `max`/`denom`/`out_buffer`
registers to the three components of this fold. `osStep` is the banked per-key
recurrence from `Math/Attention.lean`; here we specialize its `(score, value)`
inputs to the flash kernel's per-key score `(sm_scale·log2e)·(q row i · k row j)`
and value `V[j, d]`, with future keys (`> qStart + i`) or out-of-window keys
(`≥ hi`) simply not emitted. -/

open VeriTile.Triton (osStep pow2 attnKeyListCausal attnKeyList)

/-- The `(score, value)` pair the kernel streams for output `(i, d)` at *global*
key `j`: score `scale · Σ_e q[i,e]·k[j,e]`, value `V[j, d]`. (`scale` already
folds `qk_scale = sm_scale · log2e`.) -/
noncomputable def flashKV
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (i : Fin BLOCK_M) (d : Fin DIM) (j : Fin SEQLEN) : ℝ × ℝ :=
  (scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))

/-- Causal per-row key list over the *window* `[0, hi)`: keys `j < hi` with
`j ≤ qStart + i`, in index order. After `c` blocks `hi = c · BLOCK_N`, this is
the prefix the kernel has streamed. -/
noncomputable def flashKeysUpto
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    List (ℝ × ℝ) :=
  (List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
    if j.val < hi ∧ (causal → j.val ≤ qStart + i.val) then
      some (flashKV qT kT vT scale i d j)
    else none)

/-- Running per-row online-softmax state after streaming the window `[0, hi)`. -/
noncomputable def flashState
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    ℝ × ℝ × ℝ :=
  (flashKeysUpto qT kT vT scale causal qStart hi i d).foldl osStep (0, 0, 0)

/-- **⊥-seeded running max** of the streamed key prefix `[0, hi)`, exactly the
value the kernel carries in its `max` register: `max = tl.zeros([BM]) − inf`
seeds at `⊥` (NOT real `0` like `flashState.1`, which is the `blockMax 0` fold).
The `WithBot` `⊔`-fold (`foldr ⊔ ⊥`) of the coerced per-key scores; `⊥` on the
empty / `hi = 0` window (matching the kernel's preLoop init). -/
noncomputable def flashRunningMax
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    WithBot ℝ :=
  ((flashKeysUpto qT kT vT scale causal qStart hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- The `⊥`-seeded running max at the empty / `hi = 0` window is `⊥` — the kernel's
preLoop `max` init (`tl.zeros − inf`). The base case making `attnInvariant … 0`
satisfiable (`some 0 ≠ ⊥` was the prior bug). -/
theorem flashRunningMax_zero
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    flashRunningMax qT kT vT scale causal qStart 0 i d = ⊥ := by
  unfold flashRunningMax flashKeysUpto
  rw [show (List.finRange SEQLEN).filterMap
        (fun j : Fin SEQLEN => if j.val < 0 ∧ (causal → j.val ≤ qStart + i.val)
          then some (flashKV qT kT vT scale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- At the full window `hi = SEQLEN`, the non-causal key list is the full
`attnKeyList`. -/
theorem flashKeysUpto_full
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (qS : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    flashKeysUpto qT kT vT scale Bool.false qS SEQLEN i d
      = attnKeyList qT kT vT (fun _ : Fin SEQLEN => scale) i d := by
  unfold flashKeysUpto attnKeyList flashKV
  rw [List.ofFn_eq_map]
  exact List.filterMap_eq_map_iff_forall_eq_some.mpr (fun j _ => by simp [j.isLt])

/-- At the full window `hi = SEQLEN`, the causal key list is `attnKeyListCausal`. -/
theorem flashKeysUpto_full_causal
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (qS : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    flashKeysUpto qT kT vT scale Bool.true qS SEQLEN i d
      = attnKeyListCausal qT kT vT (fun _ : Fin SEQLEN => scale) qS i d := by
  unfold flashKeysUpto attnKeyListCausal flashKV
  apply List.filterMap_congr
  intro j _
  simp [j.isLt]

/-- The full-window non-causal final state reads off `flashAttnOValueSpec`. -/
theorem flashState_full_eq_spec
    (s : BlockState) (Q K V : RegionName) (sm_scale : ℝ)
    (stride_q_head : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    (let st := flashState (qTile s Q stride_q_head DIM BLOCK_M)
        (kTile s K stride_q_head DIM SEQLEN) (vTile s V stride_q_head DIM SEQLEN)
        (sm_scale * log2e) Bool.false 0 SEQLEN i d
     st.2.2 / st.2.1)
      = flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M
          (i, d, PUnit.unit) := by
  rw [flashAttnOValueSpec_eq_streaming]
  simp only [flashState]
  rw [flashKeysUpto_full]

/-- The full-window causal final state reads off `flashAttnOValueSpecCausal`. -/
theorem flashState_full_eq_spec_causal
    (s : BlockState) (Q K V : RegionName) (sm_scale : ℝ)
    (stride_q_head : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    (let st := flashState (qTile s Q stride_q_head DIM BLOCK_M)
        (kTile s K stride_q_head DIM SEQLEN) (vTile s V stride_q_head DIM SEQLEN)
        (sm_scale * log2e) Bool.true (s.pids 0 * BLOCK_M) SEQLEN i d
     st.2.2 / st.2.1)
      = flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M
          (i, d, PUnit.unit) := by
  rw [flashAttnOValueSpecCausal_eq_streaming]
  simp only [flashState]
  rw [flashKeysUpto_full_causal]

/-- Block-`c` per-row key list: keys with `c·BLOCK_N ≤ j < (c+1)·BLOCK_N` passing
the causal filter, the keys the loop's `c`-th iteration streams. -/
noncomputable def flashBlock
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    List (ℝ × ℝ) :=
  (List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
    if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val) then
      some (flashKV qT kT vT scale i d j)
    else none)

/-- Generic threshold-split for a `.val`-ascending list: the `j.val < hi₂` window
filterMap splits into the `j.val < t` prefix and the `t ≤ j.val < hi₂` block,
provided `t ≤ hi₂`. By induction using the ascending (`Pairwise <`) order — once an
element clears `t`, the whole tail does too. -/
private theorem filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (t hi₂ : Nat) (Q : Fin n → Prop) [DecidablePred Q]
    (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if Q j ∧ j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if Q j ∧ j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if t ≤ j.val ∧ j.val < hi₂ ∧ Q j then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hlt : a.val < t
    · -- head goes to the prefix
      rw [ih htl]
      have hnb : ¬ (t ≤ a.val ∧ a.val < hi₂ ∧ Q a) := fun h => (Nat.not_le.mpr hlt) h.1
      rw [if_neg hnb]
      by_cases hQ : Q a
      · have h2 : a.val < hi₂ := lt_of_lt_of_le hlt hle
        rw [if_pos (And.intro hQ h2 : Q a ∧ a.val < hi₂),
          if_pos (And.intro hQ hlt : Q a ∧ a.val < t)]
        rfl
      · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => hQ h.1),
          if_neg (fun h : Q a ∧ a.val < t => hQ h.1)]
    · -- head clears `t`; so does the whole tail, making the prefix empty
      have hge : t ≤ a.val := Nat.not_lt.mp hlt
      have htail_prefix : tl.filterMap (fun j => if Q j ∧ j.val < t then some (g j) else none) = [] := by
        apply List.filterMap_eq_nil_iff.mpr
        intro b hb
        have hab : a.val < b.val := hahead b hb
        have hbt : ¬ (b.val < t) := by omega
        simp [hbt]
      rw [ih htl, htail_prefix]
      have hnp : ¬ (Q a ∧ a.val < t) := fun h => hlt h.2
      rw [if_neg hnp]
      by_cases hQ : Q a
      · by_cases h2 : a.val < hi₂
        · rw [if_pos (And.intro hQ h2 : Q a ∧ a.val < hi₂),
            if_pos (And.intro hge (And.intro h2 hQ) : t ≤ a.val ∧ a.val < hi₂ ∧ Q a)]
          rfl
        · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => h2 h.2),
            if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ ∧ Q a => h2 h.2.1)]
      · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => hQ h.1),
          if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ ∧ Q a => hQ h.2.2)]

/-- **Window split** (`hi = c·BLOCK_N`): the keys streamed through `c+1` blocks are
those through `c` blocks followed by block `c`. -/
theorem flashKeysUpto_succ
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    flashKeysUpto qT kT vT scale causal qStart ((c + 1) * BLOCK_N) i d
      = flashKeysUpto qT kT vT scale causal qStart (c * BLOCK_N) i d
        ++ flashBlock qT kT vT scale causal qStart BLOCK_N c i d := by
  unfold flashKeysUpto flashBlock
  rw [show (List.finRange SEQLEN).filterMap
        (fun j : Fin SEQLEN => if j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
          then some (flashKV qT kT vT scale i d j) else none)
      = (List.finRange SEQLEN).filterMap
        (fun j : Fin SEQLEN => if (causal → j.val ≤ qStart + i.val) ∧ j.val < (c + 1) * BLOCK_N
          then some (flashKV qT kT vT scale i d j) else none)
      from List.filterMap_congr (fun j _ => by simp only [and_comm])]
  rw [filterMap_window_split (List.finRange SEQLEN) (List.pairwise_lt_finRange SEQLEN)
    (c * BLOCK_N) ((c + 1) * BLOCK_N) (fun j => causal → j.val ≤ qStart + i.val)
    (fun j => flashKV qT kT vT scale i d j) (by nlinarith [Nat.zero_le BLOCK_N])]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.filterMap_congr; intro j _; simp only [and_comm]
  · apply List.filterMap_congr; intro j _
    by_cases h1 : c * BLOCK_N ≤ j.val <;> by_cases h2 : j.val < (c + 1) * BLOCK_N <;>
      by_cases h3 : (causal → j.val ≤ qStart + i.val) <;> simp [h1, h2, h3, and_assoc]

/-- **One-block invariant advance** (pure math): `flashState` after `c+1` blocks
is `osBlockStep` (= the kernel's per-block update) applied to `flashState` after
`c` blocks with block `c`'s keys. -/
theorem flashState_succ
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    flashState qT kT vT scale causal qStart ((c + 1) * BLOCK_N) i d
      = VeriTile.Triton.osBlockStep
          (flashState qT kT vT scale causal qStart (c * BLOCK_N) i d)
          (flashBlock qT kT vT scale causal qStart BLOCK_N c i d) := by
  unfold flashState
  rw [flashKeysUpto_succ, List.foldl_append,
    VeriTile.Triton.osBlockStep_eq_foldl_osStep]

/-- The score-projection (`.1`) of the per-row key list is channel-independent. -/
theorem flashKeysUpto_map_fst_eq
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin DIM) :
    (flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => p.1)
      = (flashKeysUpto qT kT vT scale causal qStart hi i d').map (fun p => p.1) := by
  unfold flashKeysUpto flashKV
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases h : j.val < hi ∧ (causal → j.val ≤ qStart + i.val) <;> simp [h]

/-- `blockMax` depends only on the score projections. -/
private theorem blockMax_eq_foldl_map (m₀ : ℝ) (block : List (ℝ × ℝ)) :
    VeriTile.Triton.blockMax m₀ block = (block.map (fun p => p.1)).foldl max m₀ := by
  unfold VeriTile.Triton.blockMax
  induction block generalizing m₀ with
  | nil => rfl
  | cons a t ih => simp only [List.foldl_cons, List.map_cons]; exact ih (max m₀ a.1)

/-- The running `max` component of `flashState` is channel-independent. -/
theorem flashState_fst_eq
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin DIM) :
    (flashState qT kT vT scale causal qStart hi i d).1
      = (flashState qT kT vT scale causal qStart hi i d').1 := by
  unfold flashState
  rw [VeriTile.Triton.osStep_foldl_fst, VeriTile.Triton.osStep_foldl_fst,
    blockMax_eq_foldl_map, blockMax_eq_foldl_map,
    flashKeysUpto_map_fst_eq qT kT vT scale causal qStart hi i d d']

/-- The running denominator component (`.2.1`) of `flashState` is
channel-independent. -/
theorem flashState_snd_fst_eq
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin DIM) :
    (flashState qT kT vT scale causal qStart hi i d).2.1
      = (flashState qT kT vT scale causal qStart hi i d').2.1 := by
  have hcons : ∀ dd : Fin DIM,
      (flashState qT kT vT scale causal qStart hi i dd).2.1
        = pow2 (-(flashState qT kT vT scale causal qStart hi i dd).1)
          * (0 + ((flashKeysUpto qT kT vT scale causal qStart hi i dd).map
              (fun p => pow2 p.1)).sum) := by
    intro dd
    exact (VeriTile.Triton.osStep_foldl_consistent
      (flashKeysUpto qT kT vT scale causal qStart hi i dd) 0 0 0 0 0 (by simp) (by simp)).1
  rw [hcons d, hcons d', flashState_fst_eq qT kT vT scale causal qStart hi i d d']
  congr 2
  rw [show (fun p : ℝ × ℝ => pow2 p.1) = pow2 ∘ (fun p => p.1) from rfl,
    ← List.map_map, ← List.map_map,
    flashKeysUpto_map_fst_eq qT kT vT scale causal qStart hi i d d']

/-- One ⊥-seeded online-softmax step: like `osStep`, but the running max lives in
`WithBot ℝ` (seeded `⊥`), so `α = realExp2(m ⊖ m')` is `0` on the first block —
faithful to the kernel's `max` register (`tl.zeros − inf`) and `denom`/`acc`
(seeded `0`). The real-0-seed `osStep` fold (`flashState`) carries the **wrong**
max shift (`blockMax 0`, not `flashRunningMax`); `flashStateBot` is the faithful
recurrence the loop invariant must bind to. -/
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

/-- `flashStateBot` — the ⊥-seeded running `(max, denom, acc)` after streaming the
window `[0, hi)`. Faithful to the kernel's register recurrence (`max` seeded `⊥`,
`denom`/`acc` seeded `0`). -/
noncomputable def flashStateBot
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    WithBot ℝ × ℝ × ℝ :=
  (flashKeysUpto qT kT vT scale causal qStart hi i d).foldl osStepBot (⊥, 0, 0)

/-- **Loop invariant** for the flash-attn streaming loop (counter
`i = block c · BLOCK_N`, window `hi_c = i`). Binds the kernel's 14 live registers
after `c` blocks: program ids, the loaded+scaled `q`, the index vectors
`off_m`/`off_n`, `qk_scale`, `lo`/`hi`, the three block pointers (`K`/`V` advanced
by `c·BLOCK_N`, `Q` fixed), and — the heart — `max`/`denom`/`out_buffer` equal the
three components of the ⊥-seeded `flashStateBot` over the first `i` keys (`max` =
`flashRunningMax`, the `WithBot ⊔`-fold; `denom`/`out_buffer` = the ⊥-seeded
`denom`/`acc`, per row / per `(row, channel)`). Memory/undef preserved. Strides specialized
to the Python layout (`stride_q_seqlen = DIM`, `stride_q_dim = 1`); the per-key
score scale is `qk_scale = sm_scale · log2e`. -/
noncomputable def attnInvariant
    (Q K V : RegionName) (s0 : BlockState) (sm_scale : ℝ)
    (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N hiTotal : Nat) (causal : Bool)
    (hDIM : 0 < DIM)
    (i : Nat) (s : BlockState) : Prop :=
  let base := flashBaseOffset s0 stride_q_head
  let qStart := s0.pids 0 * BLOCK_M
  let qT := qTile s0 Q stride_q_head DIM BLOCK_M
  let kT := kTile s0 K stride_q_head DIM SEQLEN
  let vT := vTile s0 V stride_q_head DIM SEQLEN
  let scale := sm_scale * log2e
  s.pids = s0.pids ∧ i % BLOCK_N = 0 ∧ i ≤ hiTotal ∧
  (s.regs .real [BLOCK_M] "max" = some ⟨fun r : TileIndex [BLOCK_M] =>
      flashRunningMax qT kT vT scale causal qStart i r.1 ⟨0, hDIM⟩⟩) ∧
  (s.regs .real [BLOCK_M] "denom" = some ⟨fun r : TileIndex [BLOCK_M] =>
      ((flashStateBot qT kT vT scale causal qStart i r.1 ⟨0, hDIM⟩).2.1 : ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, DIM] "out_buffer" = some ⟨fun idx : TileIndex [BLOCK_M, DIM] =>
      ((flashStateBot qT kT vT scale causal qStart i idx.1 idx.2.1).2.2 : ℝ)⟩) ∧
  (s.regs .fp16 [BLOCK_M, DIM] "q" = some ⟨fun idx : TileIndex [BLOCK_M, DIM] =>
      FloatDType.real.cast FloatDType.fp16
        (some (scale * qT idx))⟩) ∧
  (s.regs .nat [BLOCK_M] "off_m" = some (Tile.vec (fun r : Fin BLOCK_M => qStart + r.val))) ∧
  (s.regs .nat [BLOCK_N] "off_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) ∧
  (s.regs .blockPtr [DIM, BLOCK_N] "K_block_ptr" = some
    (⟨fun _ : TileIndex [DIM, BLOCK_N] =>
      { region := K, baseOffset := base, parentShape := [DIM, SEQLEN],
        blockShape := [DIM, BLOCK_N], strides := [1, DIM], offsets := [0, i] }⟩)) ∧
  (s.regs .blockPtr [BLOCK_N, DIM] "V_block_ptr" = some
    (⟨fun _ : TileIndex [BLOCK_N, DIM] =>
      { region := V, baseOffset := base, parentShape := [SEQLEN, DIM],
        blockShape := [BLOCK_N, DIM], strides := [DIM, 1], offsets := [i, 0] }⟩)) ∧
  (s.regs .blockPtr [BLOCK_M, DIM] "Q_block_ptr" = some
    (⟨fun _ : TileIndex [BLOCK_M, DIM] =>
      { region := Q, baseOffset := base, parentShape := [SEQLEN, DIM],
        blockShape := [BLOCK_M, DIM], strides := [DIM, 1], offsets := [qStart, 0] }⟩)) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-! ## exec-side loop-invariant skeleton (in progress)

The compiled body of `flash_attn_fwd_kernel_surface` (verified by direct
inspection of `(surface …).toAlgKernel.body`) is a 22-statement list:

```
preLoop (16 stmts, 0–15):
  0  start_m         = program_id 0
  1  off_bs_head     = program_id 1
  2  qkv_base_offset = off_bs_head * stride_q_head
  3  Q_block_ptr     = make_block_ptr(Q + base, offsets=(start_m*BLOCK_M, 0))
  4  K_block_ptr     = make_block_ptr(K + base, offsets=(0, 0))
  5  V_block_ptr     = make_block_ptr(V + base, offsets=(0, 0))
  6  off_m           = start_m*BLOCK_M + arange BLOCK_M
  7  off_n           = arange BLOCK_N
  8  max             = full 0 + (-inf)
  9  denom           = full 0
  10 out_buffer      = full 0
  11 qk_scale        = sm_scale * 1.44269504
  12 q               = load Q_block_ptr            (no mask)
  13 q               = (q * qk_scale).to fp16
  14 lo              = 0
  15 hi              = (start_m+1)*BLOCK_M  [causal]  /  SEQLEN  [non-causal]
loop (16):
  forRangeDyn start_n lo hi BLOCK_N loopBody   (loopBody = 15 stmts)
postLoop (5 stmts, 17–21):
  17 out_buffer      = out_buffer / denom[:, None]
  18 l_ptr           = L + off_bs_head*SEQLEN + off_m
  19 store L         l_ptr (max + log2 denom)
  20 O_block_ptr     = make_block_ptr(O + base, offsets=(start_m*BLOCK_M, 0))
  21 store O         O_block_ptr (out_buffer.to fp16)
loopBody (15):
  L0 k = load K_block_ptr ; L1 v = load V_block_ptr ; L2 qk = full 0 ;
  L3 ifThen IS_CAUSAL { qk = where(off_m[:,None] >= start_n+off_n[None,:], qk, -inf) } ;
  L4 qk = qk + dot q k ; L5 max_new = maximum(max, reduceMax qk 1) ;
  L6 alpha = exp2(max - max_new) ; L7 nume = exp2(qk - max_new[:,None]) ;
  L8 out_scale = denom*0 + alpha ; L9 out_buffer = out_buffer * out_scale[:,None] ;
  L10 out_buffer = out_buffer + dot (nume.to fp16) v ; L11 denom = denom*alpha + sum nume 1 ;
  L12 max = max_new ; L13 K_block_ptr = advance(K_block_ptr, [0, BLOCK_N]) ;
  L14 V_block_ptr = advance(V_block_ptr, [BLOCK_N, 0]).
```

The body decomposes by `rfl` into `take 16 ++ (forRangeDyn … :: postLoop)`, so the
loop driver `forRangeDyn_inv` applies with the abstract `loopBody`/postLoop supplied
at the call site. The genuine remaining work — the per-statement op-eval recipes
for the 15-stmt loop body (threading the causal `ifThen`/`where` `-inf` mask into
the `osBlockStep`/`attnKeyListCausal` fold) and the `attnInvariant`/`preLoop`/
`attn_step`/`attn_postLoop` skeleton composed via `forRangeDyn_inv` and bridged
through `flashAttnOValueSpec{,Causal}_eq_streaming` — is tracked here. -/

/-- The compiled body splits as `take 16 ++ drop 16`, with `drop 16` a `forRangeDyn`
followed by the 5 post-loop statements. Pure `List` identity (`take_append_drop`),
independent of any transcription. -/
theorem flash_body_split
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (IS_CAUSAL : Bool) :
    (flash_attn_fwd_kernel_surface Q K V L O sm_scale s0 s1 s2 s3 s4 s5 s6 s7
        s8 s9 s10 s11 s12 s13 s14 s15 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N
        IS_CAUSAL).toAlgKernel.body
      = (flash_attn_fwd_kernel_surface Q K V L O sm_scale s0 s1 s2 s3 s4 s5 s6 s7
          s8 s9 s10 s11 s12 s13 s14 s15 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N
          IS_CAUSAL).toAlgKernel.body.take 16
        ++ (flash_attn_fwd_kernel_surface Q K V L O sm_scale s0 s1 s2 s3 s4 s5 s6 s7
              s8 s9 s10 s11 s12 s13 s14 s15 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N
              IS_CAUSAL).toAlgKernel.body.drop 16 :=
  (List.take_append_drop 16 _).symm

/-- The deterministic scalar prefix (statements 0–2: `start_m`, `off_bs_head`,
`qkv_base_offset`) transcribes exactly to the lowered ops, by `rfl`. The base case
of the preLoop transcription; later prefix statements (the block-pointer
constructions 3–5, the index vectors 6–7, the `max`/`denom`/`out_buffer` inits
8–10, `qk_scale` 11, the `q` load+scale 12–13, and `lo`/`hi` 14–15) extend this
list and step via the recipes above (`flash_makeBlockPtr*_eval`, `flash_load_Q_eval`,
…). -/
theorem flash_prefix_scalars_eq (Q K V L O : RegionName) (sm_scale : ℝ)
    (sqbs sqh sqsl sqd skbs skh sksl skd svbs svh svsl svd sobs soh sosl sod
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (IS_CAUSAL : Bool) :
    (flash_attn_fwd_kernel_surface Q K V L O sm_scale sqbs sqh sqsl sqd skbs skh sksl skd
        svbs svh svsl svd sobs soh sosl sod BS HEAD SEQLEN BLOCK_M DIM BLOCK_N IS_CAUSAL).toAlgKernel.body.take 3
      = [ Stmt.assign .nat [] "start_m" (Op.programId 0),
          Stmt.assign .nat [] "off_bs_head" (Op.programId 1),
          Stmt.assign .nat [] "qkv_base_offset"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_bs_head") (Op.constNat sqh)) ] :=
  rfl

/-! ## Block-reduction bridges (flashBlock ↔ Fin BLOCK_N masked reductions)

The kernel's loop body reduces a `BLOCK_N`-lane masked score row over `Fin BLOCK_N`
(lane `jL` ↦ global key `c·BLOCK_N + jL`); the `osBlockStep` math uses the
`flashBlock` list (a `Fin SEQLEN` causal filterMap of the window
`[c·BLOCK_N, (c+1)·BLOCK_N)`). These bridges equate the two by reindexing the
window onto `Fin BLOCK_N`. -/

open VeriTile.Triton (osBlockStep blockMax pow2 osStep)

/-- Sum over `Fin BLOCK_N` reindexes the window `[c·BLOCK_N, (c+1)·BLOCK_N) ⊆
Fin SEQLEN`: lane `jL` ↦ global key `c·BLOCK_N + jL`. -/
theorem flash_window_sum_reindex (BLOCK_N c SEQLEN : Nat)
    (hwin : (c + 1) * BLOCK_N ≤ SEQLEN) (g : Nat → ℝ) :
    (∑ jL : Fin BLOCK_N, g (c * BLOCK_N + jL.val))
      = ∑ j : Fin SEQLEN, (if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N then g j.val else 0) := by
  have hmul : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
  rw [← Finset.sum_filter]
  refine Finset.sum_bij
    (i := fun jL _ => (⟨c * BLOCK_N + jL.val, by
      have h1 := jL.isLt
      have h2 : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
      omega⟩ : Fin SEQLEN)) ?_ ?_ ?_ ?_
  · intro jL _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have := jL.isLt; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BLOCK_N + a.val = c * BLOCK_N + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N := (Finset.mem_filter.mp hj).2
    exact ⟨⟨j.val - c * BLOCK_N, by omega⟩, Finset.mem_univ _, by apply Fin.ext; simp only; omega⟩
  · intro jL _; rfl

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem flash_filterMap_finRange_sum {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 := by
  rw [List.map_filterMap]
  rw [show (fun j : Fin n => Option.map h (if p j then some (g j) else none))
        = (fun j : Fin n => if p j then some (h (g j)) else none) from by
    funext j; by_cases hj : p j <;> simp [hj]]
  rw [show (((List.finRange n).filterMap (fun j => if p j then some (h (g j)) else none))).sum
        = ((List.finRange n).map (fun j => if p j then h (g j) else 0)).sum from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : p a <;> simp [ha, ih]]
  rw [← List.sum_ofFn]; congr 1; rw [List.ofFn_eq_map]

/-- The map-and-sum of `flashBlock` equals a `Fin BLOCK_N`-masked `Finset.sum`,
reindexing the window onto lanes `jL : Fin BLOCK_N` (key `c·BLOCK_N + jL`). -/
theorem flashBlock_map_sum
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (hwin : (c + 1) * BLOCK_N ≤ SEQLEN) (h : ℝ × ℝ → ℝ) :
    ((flashBlock qT kT vT scale causal qStart BLOCK_N c i d).map h).sum
      = ∑ jL : Fin BLOCK_N,
          (if (causal → c * BLOCK_N + jL.val ≤ qStart + i.val) then
            h ((scale * Finset.univ.sum (fun e : Fin DIM =>
                  qT (i, e, PUnit.unit) * kT (⟨c * BLOCK_N + jL.val, by
                    have := jL.isLt; have : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
                    omega⟩, e, PUnit.unit)),
                vT (⟨c * BLOCK_N + jL.val, by
                  have := jL.isLt; have : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
                  omega⟩, d, PUnit.unit)))
           else 0) := by
  rw [flashBlock, flash_filterMap_finRange_sum SEQLEN
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val))
    (fun j => flashKV qT kT vT scale i d j) h]
  -- pull the window guard out of the conjunction into a filter on Fin SEQLEN
  have hmul : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
  rw [show (∑ j : Fin SEQLEN, if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
            then h (flashKV qT kT vT scale i d j) else 0)
        = ∑ j ∈ Finset.univ.filter (fun j : Fin SEQLEN => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N),
            (if (causal → j.val ≤ qStart + i.val) then h (flashKV qT kT vT scale i d j) else 0) from by
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hwj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N
    · by_cases hcj : (causal → j.val ≤ qStart + i.val)
      · rw [if_pos ⟨hwj.1, hwj.2, hcj⟩, if_pos hwj, if_pos hcj]
      · rw [if_neg (fun hh => hcj hh.2.2), if_pos hwj, if_neg hcj]
    · rw [if_neg (fun hh => hwj ⟨hh.1, hh.2.1⟩), if_neg hwj]]
  -- bijection: jL : Fin BLOCK_N ↦ c·BLOCK_N + jL into the filtered window
  symm
  refine Finset.sum_bij
    (i := fun jL _ => (⟨c * BLOCK_N + jL.val, by
      have h1 := jL.isLt
      have h2 : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
      omega⟩ : Fin SEQLEN)) ?_ ?_ ?_ ?_
  · intro jL _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have := jL.isLt; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BLOCK_N + a.val = c * BLOCK_N + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N := (Finset.mem_filter.mp hj).2
    exact ⟨⟨j.val - c * BLOCK_N, by omega⟩, Finset.mem_univ _, by apply Fin.ext; simp only; omega⟩
  · intro jL _
    simp only [flashKV]

/-- A real `foldl max` over a list, coerced to `WithBot`, is `max` of the seed
with the `WithBot` `foldr` of the coerced list (associativity/commutativity of
`max`). The bridge between `blockMax` (a real foldl) and the kernel's `WithBot`
running max. -/
theorem flash_foldl_max_coe (m0 : ℝ) (L : List ℝ) :
    ((L.foldl (fun a x => max a x) m0 : ℝ) : WithBot ℝ)
      = max ((m0 : ℝ) : WithBot ℝ) ((L.map (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥) := by
  induction L generalizing m0 with
  | nil => simp
  | cons a t ih =>
    simp only [List.foldl_cons, List.map_cons, List.foldr_cons]
    rw [ih (max m0 a), WithBot.coe_max]
    rw [show (((a : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun x => ((x : ℝ) : WithBot ℝ)) t)))
          = max ((a : ℝ) : WithBot ℝ) ((List.map (fun x => ((x : ℝ) : WithBot ℝ)) t).foldr (· ⊔ ·) ⊥) from rfl]
    rw [← max_assoc]

/-- The `WithBot` `foldr` of a causally-filtered score list (coerced) equals the
`Finset.sup` over `Fin n` of the lane terms (`⊥` on filtered-out lanes). -/
theorem flash_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
    (sc : Fin n → ℝ) :
    (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) := by
  rw [show (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = (List.finRange n).foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : P a <;> simp [ha, ih]]
  -- finRange foldr = Finset.sup
  apply le_antisymm
  · induction (List.finRange n) with
    | nil => simp
    | cons a t ih =>
      simp only [List.foldr_cons]
      exact sup_le (Finset.le_sup (f := fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
        (Finset.mem_univ a)) ih
  · apply Finset.sup_le
    intro j _
    have key : ∀ (l : List (Fin n)), j ∈ l →
        (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
          ≤ l.foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ := by
      intro l hl
      induction l with
      | nil => simp at hl
      | cons a t ih =>
        simp only [List.foldr_cons]
        rcases List.mem_cons.mp hl with h | h
        · subst h; exact le_sup_left
        · exact le_trans (ih h) le_sup_right
    exact key _ (List.mem_finRange j)

/-- Reindex a windowed `Finset.sup` over `Fin SEQLEN` (lanes `c·BLOCK_N ≤ j <
(c+1)·BLOCK_N` with a value-level guard `Qc`) onto `Fin BLOCK_N` (lane `jL` ↦
key `c·BLOCK_N + jL`); out-of-window lanes contribute `⊥`. -/
theorem flash_window_sup_reindex (BLOCK_N c SEQLEN : Nat) (hwin : (c + 1) * BLOCK_N ≤ SEQLEN)
    (F : Nat → WithBot ℝ) (Qc : Nat → Prop) [DecidablePred Qc] :
    Finset.univ.sup (fun j : Fin SEQLEN =>
        if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ Qc j.val then F j.val else ⊥)
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if Qc (c * BLOCK_N + jL.val) then F (c * BLOCK_N + jL.val) else ⊥) := by
  have hmul : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ Qc j.val
    · rw [if_pos hj]
      have hjL : j.val - c * BLOCK_N < BLOCK_N := by omega
      refine le_trans ?_ (Finset.le_sup
        (f := fun jL : Fin BLOCK_N => if Qc (c * BLOCK_N + jL.val) then F (c * BLOCK_N + jL.val) else ⊥)
        (Finset.mem_univ (⟨j.val - c * BLOCK_N, hjL⟩ : Fin BLOCK_N)))
      simp only
      rw [show c * BLOCK_N + (j.val - c * BLOCK_N) = j.val from by omega, if_pos hj.2.2]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le
    intro jL _
    by_cases hq : Qc (c * BLOCK_N + jL.val)
    · rw [if_pos hq]
      have hb : c * BLOCK_N + jL.val < SEQLEN := by have := jL.isLt; omega
      refine le_trans ?_ (Finset.le_sup
        (f := fun j : Fin SEQLEN =>
          if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ Qc j.val then F j.val else ⊥)
        (Finset.mem_univ (⟨c * BLOCK_N + jL.val, hb⟩ : Fin SEQLEN)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega, hq⟩)]
    · rw [if_neg hq]; exact bot_le

/-- **Block running-max bridge.** `↑(blockMax m₀ block)` for `block` the
causally-filtered `flashBlock` window equals `max (some m₀)` of the kernel's
`WithBot` `Finset.sup` over `Fin BLOCK_N` of the masked score row. Combines the
foldl→coe-foldr bridge, the filterMap-foldr→sup bridge, and the window reindex —
the math bridge for the kernel's `max_new = tl.maximum(max, tl.max(qk,1))`. -/
theorem flashBlock_blockMax
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (m₀ : ℝ) (hwin : (c + 1) * BLOCK_N ≤ SEQLEN) :
    ((blockMax m₀ (flashBlock qT kT vT scale causal qStart BLOCK_N c i d) : ℝ) : WithBot ℝ)
      = max ((m₀ : ℝ) : WithBot ℝ)
          (Finset.univ.sup (fun jL : Fin BLOCK_N =>
            if (causal → c * BLOCK_N + jL.val ≤ qStart + i.val) then
              ((scale * Finset.univ.sum (fun e : Fin DIM =>
                  qT (i, e, PUnit.unit) * kT (⟨c * BLOCK_N + jL.val, by
                    have := jL.isLt; have : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
                    omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
            else ⊥)) := by
  classical
  set F : Nat → WithBot ℝ := fun jg =>
    if h : jg < SEQLEN then
      ((scale * Finset.univ.sum (fun e : Fin DIM =>
          qT (i, e, PUnit.unit) * kT (⟨jg, h⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
    else ⊥ with hF
  rw [show blockMax m₀ (flashBlock qT kT vT scale causal qStart BLOCK_N c i d)
        = ((flashBlock qT kT vT scale causal qStart BLOCK_N c i d).map (fun p => p.1)).foldl
            (fun a x => max a x) m₀ from by unfold blockMax; rw [List.foldl_map]]
  rw [flash_foldl_max_coe]
  congr 1
  rw [show (flashBlock qT kT vT scale causal qStart BLOCK_N c i d).map (fun p => p.1)
        = (List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
            if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
            then some (scale * Finset.univ.sum (fun e : Fin DIM =>
                  qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit))) else none) from by
    unfold flashBlock flashKV
    rw [List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val) <;>
      simp [hj]]
  rw [flash_filterMap_foldr_sup SEQLEN
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val))
    (fun j => scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)))]
  -- rewrite the Fin SEQLEN windowed sup body as `if window∧Qc then F j.val else ⊥`
  rw [show (Finset.univ.sup (fun j : Fin SEQLEN =>
        if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
        then ((scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else ⊥))
      = Finset.univ.sup (fun j : Fin SEQLEN =>
          if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
          then F j.val else ⊥) from by
    apply Finset.sup_congr rfl
    intro j _
    by_cases hw : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
    · rw [if_pos hw, if_pos hw, hF]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [flash_window_sup_reindex BLOCK_N c SEQLEN hwin F
    (fun jg => causal → jg ≤ qStart + i.val)]
  -- the BLOCK_N sup bodies agree (F at c·BN+jL = the explicit score)
  apply Finset.sup_congr rfl
  intro jL _
  have hb : c * BLOCK_N + jL.val < SEQLEN := by
    have h1 := jL.isLt
    have h2 : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
    omega
  by_cases hq : (causal → c * BLOCK_N + jL.val ≤ qStart + i.val)
  · rw [if_pos hq, if_pos hq, hF]; simp only [dif_pos hb]
  · rw [if_neg hq, if_neg hq]

/-! ## ⊥-seeded running-max advance (binds the kernel's `max` register)

The kernel's `max` register seeds at `⊥` (`tl.zeros − inf`), not real `0`, so the
loop invariant binds it to `flashRunningMax` (the `WithBot ⊔`-fold), not
`flashState.1` (the `blockMax 0` real fold). These reuse the banked block-reduction
bridges (`flash_filterMap_foldr_sup` / `flash_window_sup_reindex`) in their `⊥`-seed
form. -/

/-- `flashRunningMax` is channel-independent (depends only on the score
projections, like `flashState_fst_eq`). -/
theorem flashRunningMax_eq
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin DIM) :
    flashRunningMax qT kT vT scale causal qStart hi i d
      = flashRunningMax qT kT vT scale causal qStart hi i d' := by
  unfold flashRunningMax
  rw [show (fun p : ℝ × ℝ => ((p.1 : ℝ) : WithBot ℝ))
        = (fun x : ℝ => ((x : ℝ) : WithBot ℝ)) ∘ (fun p => p.1) from rfl,
    ← List.map_map, ← List.map_map,
    flashKeysUpto_map_fst_eq qT kT vT scale causal qStart hi i d d']

/-- **One-block advance** of the `⊥`-seeded running max: streaming block `c`
(`[c·BLOCK_N, (c+1)·BLOCK_N)`) advances `flashRunningMax` by `⊔` with that block's
masked score `Finset.sup` over the `BLOCK_N` lanes — exactly the kernel's
`max_new = tl.maximum(max, tl.max(qk, 1))`. Reuses the banked `flash_filterMap_foldr_sup`
+ `flash_window_sup_reindex` bridges, `⊥`-seed variant. -/
theorem flashRunningMax_succ
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (hwin : (c + 1) * BLOCK_N ≤ SEQLEN) :
    flashRunningMax qT kT vT scale causal qStart ((c + 1) * BLOCK_N) i d
      = flashRunningMax qT kT vT scale causal qStart (c * BLOCK_N) i d
        ⊔ Finset.univ.sup (fun jL : Fin BLOCK_N =>
            if (causal → c * BLOCK_N + jL.val ≤ qStart + i.val) then
              ((scale * Finset.univ.sum (fun e : Fin DIM =>
                  qT (i, e, PUnit.unit) * kT (⟨c * BLOCK_N + jL.val, by
                    have := jL.isLt; have : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
                    omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
            else ⊥) := by
  unfold flashRunningMax
  rw [flashKeysUpto_succ, List.map_append, List.foldr_append]
  have hblock : ((flashBlock qT kT vT scale causal qStart BLOCK_N c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if (causal → c * BLOCK_N + jL.val ≤ qStart + i.val) then
            ((scale * Finset.univ.sum (fun e : Fin DIM =>
                qT (i, e, PUnit.unit) * kT (⟨c * BLOCK_N + jL.val, by
                  have := jL.isLt; have : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
                  omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
          else ⊥) := by
    rw [show (flashBlock qT kT vT scale causal qStart BLOCK_N c i d).map
          (fun p => ((p.1 : ℝ) : WithBot ℝ))
        = ((List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
            if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
            then some ((scale * Finset.univ.sum (fun e : Fin DIM =>
                  qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)) : ℝ)) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
      unfold flashBlock flashKV
      rw [List.map_filterMap, List.map_filterMap]
      apply List.filterMap_congr
      intro j _
      by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val) <;>
        simp [hj]]
    rw [flash_filterMap_foldr_sup SEQLEN
      (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val))
      (fun j => scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)))]
    classical
    set F : Nat → WithBot ℝ := fun jg =>
      if h : jg < SEQLEN then
        ((scale * Finset.univ.sum (fun e : Fin DIM =>
            qT (i, e, PUnit.unit) * kT (⟨jg, h⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
      else ⊥ with hF
    rw [show (Finset.univ.sup (fun j : Fin SEQLEN =>
          if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
          then ((scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)) : ℝ) : WithBot ℝ)
          else ⊥))
        = Finset.univ.sup (fun j : Fin SEQLEN =>
            if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
            then F j.val else ⊥) from by
      apply Finset.sup_congr rfl
      intro j _
      by_cases hw : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ (causal → j.val ≤ qStart + i.val)
      · rw [if_pos hw, if_pos hw, hF]; simp only [dif_pos j.isLt]
      · rw [if_neg hw, if_neg hw]]
    rw [flash_window_sup_reindex BLOCK_N c SEQLEN hwin F
      (fun jg => causal → jg ≤ qStart + i.val)]
    apply Finset.sup_congr rfl
    intro jL _
    have hb : c * BLOCK_N + jL.val < SEQLEN := by
      have h1 := jL.isLt
      have h2 : (c + 1) * BLOCK_N = c * BLOCK_N + BLOCK_N := by ring
      omega
    by_cases hq : (causal → c * BLOCK_N + jL.val ≤ qStart + i.val)
    · rw [if_pos hq, if_pos hq, hF]; simp only [dif_pos hb]
    · rw [if_neg hq, if_neg hq]
  rw [hblock]
  generalize hpre : (flashKeysUpto qT kT vT scale causal qStart (c * BLOCK_N) i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ)) = preL
  clear hblock hpre
  induction preL with
  | nil => simp
  | cons a t ih => simp only [List.foldr_cons, ih]; rw [sup_assoc]

/-! ## ⊥-seeded online-softmax state (binds the kernel's `denom`/`out_buffer`)

The kernel seeds its running max at `⊥` (`tl.zeros − inf`), and its `denom`/`acc`
registers at real `0`. Each block it rescales by `α = exp2(max ⊖ max_new)` —
with `max = ⊥` on block 0 the rescale is `realExp2 ⊥ = 0`, killing the seed. The
`flashState` recurrence (the *real-0-seeded* `osStep` fold) carries the **wrong**
max shift (`blockMax 0`, not `flashRunningMax`), so its `denom`/`acc` do not equal
the kernel's. `flashStateBot` is the faithful ⊥-seeded recurrence; its final
`acc/denom` ratio and `max + log2 denom` log-sum-exp agree with `flashState`'s
(the `pow2(−M)` common factor cancels in the ratio / telescopes in the L value),
which is what reconnects it to the closed-form spec. The `osStepBot`/`flashStateBot`
definitions live above (before `attnInvariant`, which binds to them). -/

/-- The running `max` component of `flashStateBot` is exactly `flashRunningMax`
(the `⊥`-seeded `⊔`-fold). -/
theorem flashStateBot_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded consistency.** Folding `osStepBot` from a start `(m, l, acc)` whose
`l`/`acc` are anchored to the true (max-free) batch denominator `L` / accumulator
`T` via the `⊥`-aware factor keeps that invariant: `l = κ(m)·L`, `acc = κ(m)·T`
with `κ ⊥ = 0`, `κ (some r) = pow2(−r)`. (`κ` is `(realExp2 (realSub ⊥ m ... ))`,
spelled directly.) -/
theorem osStepBot_foldl_consistent (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let st := xs.foldl osStepBot (m, l, acc)
    st.2.1 = (st.1.elim 0 (fun r => pow2 (-r))) * (L + (xs.map (fun p => pow2 p.1)).sum) ∧
    st.2.2 = (st.1.elim 0 (fun r => pow2 (-r))) * (T + (xs.map (fun p => pow2 p.1 * p.2)).sum) := by
  induction xs generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons x xs ih =>
    obtain ⟨s, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((s : ℝ) : WithBot ℝ) with hm'
    -- m' is finite (it is at least `some s`)
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨s, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a s, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => pow2 (-r)) = pow2 (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    have hp : pow2 (s - m'.unbotD 0) = pow2 (-mr) * pow2 s := by
      rw [hunbot, ← pow2_add]; ring_nf
    -- the two updated values land on `pow2(-mr)·(L+pow2 s)` and `pow2(-mr)·(T+pow2 s·v)`
    have hl' : l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (s - m'.unbotD 0) = pow2 (-mr) * (L + pow2 s) := by
      cases m with
      | bot =>
        rw [hmL rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a s) * Real.log 2) = pow2 (a - max a s) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hl, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have hacc' : acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (s - m'.unbotD 0) * v = pow2 (-mr) * (T + pow2 s * v) := by
      cases m with
      | bot =>
        rw [hmT rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a s) * Real.log 2) = pow2 (a - max a s) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hacc, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have step := ih m'
      (l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (s - m'.unbotD 0))
      (acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (s - m'.unbotD 0) * v)
      (T + pow2 s * v) (L + pow2 s) (by rw [hl', hκm']) (by rw [hacc', hκm'])
      (by rw [hmr]; simp) (by rw [hmr]; simp)
    simpa [List.foldl_cons, osStepBot, hm', List.map_cons, add_assoc] using step

/-- The `WithBot ⊔`-fold is seed/direction-agnostic: `foldl (⊔)` from `⊥` equals
`foldr (⊔)` from `⊥`. -/
theorem foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih =>
      intro m
      simp only [List.foldl_cons, List.foldr_cons, ih]
      rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- The ⊥-seeded running `max` of `flashStateBot` is exactly `flashRunningMax`. -/
theorem flashStateBot_fst_eq_runningMax
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    (flashStateBot qT kT vT scale causal qStart hi i d).1
      = flashRunningMax qT kT vT scale causal qStart hi i d := by
  rw [flashStateBot, flashStateBot_fst, flashRunningMax, foldl_sup_bot_eq_foldr]

/-- The ⊥-seeded denominator equals `κ(flashRunningMax)·Σpow2 score`. -/
theorem flashStateBot_snd_fst
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    (flashStateBot qT kT vT scale causal qStart hi i d).2.1
      = ((flashRunningMax qT kT vT scale causal qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * ((flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1)).sum := by
  have h := (osStepBot_foldl_consistent (flashKeysUpto qT kT vT scale causal qStart hi i d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).1
  rw [flashStateBot]
  rw [show (List.foldl osStepBot (⊥, 0, 0) (flashKeysUpto qT kT vT scale causal qStart hi i d)).2.1
        = _ from h]
  rw [show (List.foldl osStepBot (⊥, 0, 0) (flashKeysUpto qT kT vT scale causal qStart hi i d)).1
        = flashRunningMax qT kT vT scale causal qStart hi i d from by
    rw [flashStateBot_fst, flashRunningMax, foldl_sup_bot_eq_foldr]]
  rw [zero_add]

/-- The ⊥-seeded accumulator equals `κ(flashRunningMax)·Σpow2 score·v`. -/
theorem flashStateBot_snd_snd
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    (flashStateBot qT kT vT scale causal qStart hi i d).2.2
      = ((flashRunningMax qT kT vT scale causal qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * ((flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum := by
  have h := (osStepBot_foldl_consistent (flashKeysUpto qT kT vT scale causal qStart hi i d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).2
  rw [flashStateBot]
  rw [show (List.foldl osStepBot (⊥, 0, 0) (flashKeysUpto qT kT vT scale causal qStart hi i d)).2.2
        = _ from h]
  rw [show (List.foldl osStepBot (⊥, 0, 0) (flashKeysUpto qT kT vT scale causal qStart hi i d)).1
        = flashRunningMax qT kT vT scale causal qStart hi i d from by
    rw [flashStateBot_fst, flashRunningMax, foldl_sup_bot_eq_foldr]]
  rw [zero_add]

/-- The ratio `acc/denom` of the ⊥-seeded state equals that of the real-0-seeded
`flashState` — the seed cancels (`pow2` never zero). Valid whenever the window is
nonempty (`flashRunningMax ≠ ⊥`). -/
theorem flashStateBot_ratio_eq
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (hne : flashRunningMax qT kT vT scale causal qStart hi i d ≠ ⊥) :
    (flashStateBot qT kT vT scale causal qStart hi i d).2.2
        / (flashStateBot qT kT vT scale causal qStart hi i d).2.1
      = (flashState qT kT vT scale causal qStart hi i d).2.2
        / (flashState qT kT vT scale causal qStart hi i d).2.1 := by
  rw [flashStateBot_snd_fst, flashStateBot_snd_snd]
  have hcL := (VeriTile.Triton.osStep_foldl_consistent
    (flashKeysUpto qT kT vT scale causal qStart hi i d) 0 0 0 0 0 (by simp) (by simp)).1
  have hcT := (VeriTile.Triton.osStep_foldl_consistent
    (flashKeysUpto qT kT vT scale causal qStart hi i d) 0 0 0 0 0 (by simp) (by simp)).2
  rw [flashState]
  rw [show (List.foldl osStep (0, 0, 0) (flashKeysUpto qT kT vT scale causal qStart hi i d)).2.1
        = _ from hcL,
      show (List.foldl osStep (0, 0, 0) (flashKeysUpto qT kT vT scale causal qStart hi i d)).2.2
        = _ from hcT]
  cases hM : flashRunningMax qT kT vT scale causal qStart hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    rw [show ((↑r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) from rfl]
    simp only [zero_add]
    rw [mul_div_mul_left _ _ (ne_of_gt (pow2_pos _)),
        mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]

/-- **The full-window non-causal ⊥-seeded final state reads off the closed-form
spec.** `flashStateBot.acc / flashStateBot.denom = flashAttnOValueSpec`. -/
theorem flashStateBot_full_eq_spec
    (s : BlockState) (Q K V : RegionName) (sm_scale : ℝ)
    (stride_q_head : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (hne : flashRunningMax (qTile s Q stride_q_head DIM BLOCK_M)
        (kTile s K stride_q_head DIM SEQLEN) (vTile s V stride_q_head DIM SEQLEN)
        (sm_scale * log2e) Bool.false 0 SEQLEN i d ≠ ⊥) :
    (let st := flashStateBot (qTile s Q stride_q_head DIM BLOCK_M)
        (kTile s K stride_q_head DIM SEQLEN) (vTile s V stride_q_head DIM SEQLEN)
        (sm_scale * log2e) Bool.false 0 SEQLEN i d
     st.2.2 / st.2.1)
      = flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M
          (i, d, PUnit.unit) := by
  simp only
  rw [flashStateBot_ratio_eq _ _ _ _ _ _ _ _ _ hne]
  exact flashState_full_eq_spec s Q K V sm_scale stride_q_head i d

/-- **The full-window causal ⊥-seeded final state reads off the causal spec.** -/
theorem flashStateBot_full_eq_spec_causal
    (s : BlockState) (Q K V : RegionName) (sm_scale : ℝ)
    (stride_q_head : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (hne : flashRunningMax (qTile s Q stride_q_head DIM BLOCK_M)
        (kTile s K stride_q_head DIM SEQLEN) (vTile s V stride_q_head DIM SEQLEN)
        (sm_scale * log2e) Bool.true (s.pids 0 * BLOCK_M) SEQLEN i d ≠ ⊥) :
    (let st := flashStateBot (qTile s Q stride_q_head DIM BLOCK_M)
        (kTile s K stride_q_head DIM SEQLEN) (vTile s V stride_q_head DIM SEQLEN)
        (sm_scale * log2e) Bool.true (s.pids 0 * BLOCK_M) SEQLEN i d
     st.2.2 / st.2.1)
      = flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M
          (i, d, PUnit.unit) := by
  simp only
  rw [flashStateBot_ratio_eq _ _ _ _ _ _ _ _ _ hne]
  exact flashState_full_eq_spec_causal s Q K V sm_scale stride_q_head i d

/-- **L-store value (log-sum-exp) is seed-independent.** The kernel stores
`max + log2 denom`; for the ⊥-seeded state this is `M_bot + log2(κ(M_bot)·Σ) =
log2 Σpow2 score` (the `−M_bot` shift cancels), the genuine log-sum-exp, whenever
the window is nonempty and `Σpow2 score > 0`. -/
theorem flashStateBot_logsumexp
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (mr : ℝ) (hM : flashRunningMax qT kT vT scale causal qStart hi i d = (mr : WithBot ℝ)) :
    mr + Real.log ((flashStateBot qT kT vT scale causal qStart hi i d).2.1) / Real.log 2
      = Real.log (((flashKeysUpto qT kT vT scale causal qStart hi i d).map
          (fun p => pow2 p.1)).sum) / Real.log 2 := by
  rw [flashStateBot_snd_fst, hM]
  rw [show ((↑mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-mr) from rfl]
  -- the sum is > 0 only when the list is nonempty; factor
  -- log(pow2(-mr)*S) = log(pow2(-mr)) + log S when both positive.
  by_cases hS : ((flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1)).sum = 0
  · -- sum of strictly-positive terms is 0 ⟹ the key list is empty ⟹ running max ⊥,
    -- contradicting `hM : … = ↑mr`.
    exfalso
    have hempty : flashKeysUpto qT kT vT scale causal qStart hi i d = [] := by
      by_contra hne
      obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
      have hpos : 0 < pow2 p.1 := pow2_pos _
      have hmem : pow2 p.1 ∈ (flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1) :=
        List.mem_map_of_mem hp
      have hnn : ∀ x ∈ (flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1), (0:ℝ) ≤ x := by
        intro x hx; simp only [List.mem_map] at hx; obtain ⟨q, _, rfl⟩ := hx; exact le_of_lt (pow2_pos _)
      have := List.single_le_sum hnn _ hmem
      rw [hS] at this
      exact absurd (le_antisymm this (le_of_lt hpos)) (ne_of_gt hpos)
    rw [flashRunningMax, hempty] at hM
    simp only [List.map_nil, List.foldr_nil] at hM
    exact absurd hM (WithBot.bot_ne_coe)
  · have hSpos : 0 < ((flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1)).sum := by
      have hnn : 0 ≤ ((flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1)).sum :=
        List.sum_nonneg (by intro x hx; simp only [List.mem_map] at hx; obtain ⟨p, _, rfl⟩ := hx; exact le_of_lt (pow2_pos _))
      exact lt_of_le_of_ne hnn (Ne.symm hS)
    rw [Real.log_mul (ne_of_gt (pow2_pos _)) (ne_of_gt hSpos)]
    rw [show pow2 (-mr) = Real.exp (Real.log 2 * (-mr)) from rfl, Real.log_exp]
    field_simp
    ring

/-- **One-block advance** of the ⊥-seeded state: streaming block `c` folds that
block's keys (via `osStepBot`) onto the state after `c` blocks. (`foldl_append` over
the `flashKeysUpto_succ` window split — the ⊥-seed analogue of `flashState_succ`.) -/
theorem flashStateBot_succ
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    flashStateBot qT kT vT scale causal qStart ((c + 1) * BLOCK_N) i d
      = (flashBlock qT kT vT scale causal qStart BLOCK_N c i d).foldl osStepBot
          (flashStateBot qT kT vT scale causal qStart (c * BLOCK_N) i d) := by
  unfold flashStateBot
  rw [flashKeysUpto_succ, List.foldl_append]

/-- The ⊥-seeded state at the empty / `hi = 0` window is `(⊥, 0, 0)` — the kernel's
preLoop init (`max = tl.zeros − inf`, `denom = 0`, `out_buffer = 0`). The base case
making `attnInvariant … 0` satisfiable under the ⊥-seed rebind. -/
theorem flashStateBot_zero
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    flashStateBot qT kT vT scale causal qStart 0 i d = (⊥, 0, 0) := by
  unfold flashStateBot flashKeysUpto
  rw [show (List.finRange SEQLEN).filterMap
        (fun j : Fin SEQLEN => if j.val < 0 ∧ (causal → j.val ≤ qStart + i.val)
          then some (flashKV qT kT vT scale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-! ## STEP C — exec-side stepping (preLoop → attn_step → postLoop → top theorem)

Mirrors the worked `attention_score` exec-assembly (`score_preLoop_eval`,
`score_loop_eval`, `score_post_eval`, `attention_score_case1_exec_eq_closedForm`,
`attention_score_case1_genuine_compute_correct`), retargeted onto the
block-pointer foundation + `flashStateBot`/`flashRunningMax` recurrence and the
`forRangeDyn_inv` dynamic-loop driver. Specialized to the Python layout
`(BS,HEAD,SEQLEN,DIM)=(2,2,128,64)`, `BLOCK_M=128`, `BLOCK_N=64`,
`stride_q_head=8192`; `sm_scale`/`IS_CAUSAL` kept as parameters. -/

open VeriTile.Triton (osStep pow2 attnKeyList attnKeyListCausal osBlockStep)

/-- `evalOp` of a `constBool` literal. -/
theorem flash_evalOp_constBool (b : Bool) (s : BlockState) :
    evalOp (Op.constBool b) s = some (Tile.scalar b) := by
  unfold evalOp; rfl

/-- `ifThen true body` steps the body. -/
theorem flash_stepStmt_ifThen_true (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (Op.constBool Bool.true) body) s = stepStmts body s := by
  unfold stepStmt; rw [flash_evalOp_constBool]; rfl

/-- `ifThen false body` is a no-op. -/
theorem flash_stepStmt_ifThen_false (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (Op.constBool Bool.false) body) s = some s := by
  unfold stepStmt; rw [flash_evalOp_constBool]; rfl

/-- `scalarBop` helper (nil-broadcast binary op on scalars). -/
theorem flash_scalarBop {dtype : TileDType}
    (f : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (a b : TileCarrier dtype) :
    Tile.bop f Broadcast.nil (Tile.scalar a) (Tile.scalar b) = Tile.scalar (f a b) := rfl

/-- **`q = (q * qk_scale).to fp16` statement eval** (preLoop stmt 13): the loaded
`q` tile is scaled by the scalar `qk_scale` then round-tripped to fp16. -/
theorem flash_qscale_op_eval (s : BlockState) (qtile : Tile .real [128, 64]) (sc : ℝ)
    (hq : s.regs .real [128, 64] "q" = some qtile)
    (hqs : s.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) :
    evalOp (Op.castFloat .real .fp16
        (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 64] "q") (Op.ref .real [] "qk_scale"))) s
      = some (⟨fun idx : TileIndex [128, 64] =>
          FloatDType.real.cast FloatDType.fp16
            ((qtile.data idx).bind (fun x => some (x * sc)))⟩ : Tile .fp16 [128, 64]) := by
  have hmul : evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 64] "q")
        (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := by
    rw [evalOp_mul]; simp only [evalOp_ref, hq, hqs, Option.bind_eq_bind, Option.bind_some]
  have hmul2 : @evalOp FloatDType.real.toTileDType [128, 64]
        (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 64] "q") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := hmul
  rw [evalOp_castFloat, hmul2]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

/-- The 16 lowered preLoop statements of the Python-shape flash-attn body
(`IS_CAUSAL` left as a parameter; only `hi` (stmt 15) depends on it). -/
def flashPreLoop (Q K V : RegionName) (sm_scale : ℝ) (IS_CAUSAL : Bool) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_bs_head" (Op.programId 1),
    Stmt.assign .nat [] "qkv_base_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_bs_head") (Op.constNat 8192)),
    Stmt.assign .blockPtr [128, 64] "Q_block_ptr"
      (Op.makeBlockPtrDynOffsets Q (Op.ref .nat [] "qkv_base_offset") [128, 64] [128, 64] [64, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128), Op.constNat 0]),
    Stmt.assign .blockPtr [64, 64] "K_block_ptr"
      (Op.makeBlockPtrDyn K (Op.ref .nat [] "qkv_base_offset") [64, 128] [64, 64] [1, 64] [0, 0]),
    Stmt.assign .blockPtr [64, 64] "V_block_ptr"
      (Op.makeBlockPtrDyn V (Op.ref .nat [] "qkv_base_offset") [128, 64] [64, 64] [64, 1] [0, 0]),
    Stmt.assign .nat [128] "off_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)),
    Stmt.assign .nat [64] "off_n" (Op.arange 64),
    Stmt.assign .real [128] "max"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf),
    Stmt.assign .real [128] "denom" (Op.full [128] (Op.const 0)),
    Stmt.assign .real [128, 64] "out_buffer" (Op.full [128, 64] (Op.const 0)),
    Stmt.assign .real [] "qk_scale"
      (Op.mul .real Broadcast.nil (Op.const sm_scale) (Op.const 1.44269504)),
    Stmt.assign .real [128, 64] "q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "Q_block_ptr") []) MaskOpt.none),
    Stmt.assign .fp16 [128, 64] "q"
      (Op.castFloat .real .fp16
        (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 64] "q") (Op.ref .real [] "qk_scale"))),
    Stmt.assign .nat [] "lo" (Op.constNat 0),
    Stmt.assign .nat [] "hi"
      ((Op.constBool IS_CAUSAL).ite
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
        (Op.constNat 128)) ]

/-- The lowered Python-shape flash-attn body `take 16` is exactly `flashPreLoop`. -/
theorem flashPreLoop_check (Q K V L O : RegionName) (sm_scale : ℝ) (IS_CAUSAL : Bool) :
    (flash_attn_fwd_kernel_surface Q K V L O sm_scale
        16384 8192 64 1 16384 8192 64 1 16384 8192 64 1 16384 8192 64 1
        2 2 128 128 64 64 IS_CAUSAL).toAlgKernel.body.take 16
      = flashPreLoop Q K V sm_scale IS_CAUSAL := by
  cases IS_CAUSAL <;> rfl

/-- The resolved value of the `hi` register after preLoop. -/
def flashHi (s : BlockState) (IS_CAUSAL : Bool) : Nat :=
  if IS_CAUSAL then (s.pids 0 + 1) * 128 else 128

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution.** The 16 deterministic preLoop statements step a clean
state to the loop-entry state, exposing every register readback `attnInvariant …
0` / the loop body needs: the running `max`/`denom`/`out_buffer` registers carry
the ⊥-seed init (`full ⊥`, `full 0`, `full 0`), `q` is the scaled fp16 cast, the
index vectors / three block pointers are set, and `lo`/`hi` resolve. -/
theorem flash_preLoop_eval
    (s : BlockState) (Q K V : RegionName) (sm_scale : ℝ) (IS_CAUSAL : Bool)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (flashPreLoop Q K V sm_scale IS_CAUSAL) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "off_bs_head" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [] "qkv_base_offset" = some (Tile.scalar (s.pids 1 * 8192))
      ∧ s0.regs .real [128] "max" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [128] "denom" = some ⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩
      ∧ s0.regs .real [128, 64] "out_buffer" = some ⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩
      ∧ s0.regs .fp16 [128, 64] "q" = some ⟨fun idx : TileIndex [128, 64] =>
          FloatDType.real.cast FloatDType.fp16
            (some (sm_scale * log2e * qTile s Q 8192 64 128 idx))⟩
      ∧ s0.regs .nat [128] "off_m" = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val))
      ∧ s0.regs .nat [64] "off_n" = some (Tile.vec (fun j : Fin 64 => j.val))
      ∧ s0.regs .blockPtr [64, 64] "K_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := K, baseOffset := s.pids 1 * 8192, parentShape := [64, 128],
              blockShape := [64, 64], strides := [1, 64], offsets := [0, 0] }⟩)
      ∧ s0.regs .blockPtr [64, 64] "V_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := V, baseOffset := s.pids 1 * 8192, parentShape := [128, 64],
              blockShape := [64, 64], strides := [64, 1], offsets := [0, 0] }⟩)
      ∧ s0.regs .blockPtr [128, 64] "Q_block_ptr" = some
          (⟨fun _ : TileIndex [128, 64] =>
            { region := Q, baseOffset := s.pids 1 * 8192, parentShape := [128, 64],
              blockShape := [128, 64], strides := [64, 1], offsets := [s.pids 0 * 128, 0] }⟩)
      ∧ s0.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.44269504)))
      ∧ s0.regs .nat [] "lo" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "hi" = some (Tile.scalar (flashHi s IS_CAUSAL)) := by
  unfold flashPreLoop
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: off_bs_head = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: qkv_base_offset = off_bs_head * 8192
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_bs_head") (Op.constNat 8192)) _
        = some (Tile.scalar (s.pids 1 * 8192)) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 3: Q_block_ptr = makeBlockPtrDynOffsets Q qkv [start_m*128, 0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets Q (Op.ref .nat [] "qkv_base_offset") [128, 64] [128, 64]
        [64, 1] [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128), Op.constNat 0]) _
        = some (⟨fun _ : TileIndex [128, 64] =>
            { region := Q, baseOffset := s.pids 1 * 8192, parentShape := [128, 64],
              blockShape := [128, 64], strides := [64, 1], offsets := [s.pids 0 * 128, 0] }⟩
            : Tile .blockPtr [128, 64]) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_ref, evalOp_constNat, evalOp_mul, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind,
        Option.bind_some, List.mapM_cons, List.mapM_nil, BlockState.setReg_pids, flash_scalarBop]
      refine congrArg some ?_; ext idx; rfl))]
  -- stmt 4: K_block_ptr = makeBlockPtrDyn K qkv [0,0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_makeBlockPtrDyn_eval K (Op.ref .nat [] "qkv_base_offset") [64, 128] [64, 64] [1, 64] [0, 0] _
      (s.pids 1 * 8192) (by rw [evalOp_ref]; simp)))]
  -- stmt 5: V_block_ptr = makeBlockPtrDyn V qkv [0,0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_makeBlockPtrDyn_eval V (Op.ref .nat [] "qkv_base_offset") [128, 64] [64, 64] [64, 1] [0, 0] _
      (s.pids 1 * 8192) (by rw [evalOp_ref]; simp)))]
  -- stmt 6: off_m = start_m*128 + arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)) _
        = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val)) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, evalOp_arange, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 7: off_n = arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 64) _ = some (Tile.vec (fun j : Fin 64 => j.val)) from evalOp_arange 64 _))]
  -- stmt 8: max = full 0 + (-inf) = full ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩ : Tile .real [128]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rfl))]
  -- stmt 9: denom = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩ : Tile .real [128]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 10: out_buffer = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 64] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ : Tile .real [128, 64]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 11: qk_scale = sm_scale * 1.44269504
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.nil (Op.const sm_scale) (Op.const 1.44269504)) _
        = some (Tile.scalar (some (sm_scale * 1.44269504))) from by
      rw [evalOp_mul]
      simp only [evalOp_const, Option.bind_eq_bind, Option.bind_some, flash_scalarBop]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- stmt 12: q = load Q_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "Q_block_ptr") [])
        MaskOpt.none) _
        = some (⟨fun idx : TileIndex [128, 64] =>
            some (s.readMem Q (s.pids 1 * 8192 + (s.pids 0 * 128 + idx.1.val) * 64 + idx.2.1.val * 1))⟩
            : Tile .real [128, 64]) from by
      rw [flash_load_Q_eval Q (s.pids 1 * 8192) 128 64 128 64 64 1 (s.pids 0 * 128)
        (Op.ref .blockPtr [128, 64] "Q_block_ptr") _ (by rw [evalOp_ref]; simp)]
      refine congrArg some ?_; ext idx
      simp [BlockState.readMem, BlockState.setReg_mem]))]
  -- stmt 13: q = (q * qk_scale).to fp16
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.castFloat .real .fp16
        (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 64] "q") (Op.ref .real [] "qk_scale"))) _
        = some (⟨fun idx : TileIndex [128, 64] =>
            FloatDType.real.cast FloatDType.fp16
              (some (sm_scale * log2e * qTile s Q 8192 64 128 idx))⟩ : Tile .fp16 [128, 64]) from by
      rw [flash_qscale_op_eval _
        (⟨fun idx : TileIndex [128, 64] =>
          some (s.readMem Q (s.pids 1 * 8192 + (s.pids 0 * 128 + idx.1.val) * 64 + idx.2.1.val * 1))⟩
          : Tile .real [128, 64])
        (sm_scale * 1.44269504)
        (by rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same])]
      refine congrArg some ?_; ext idx
      simp only [Option.bind, Option.map, qTile, flashBaseOffset, mIndex, log2e]
      ring_nf))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.constNat 0) _ = _ from evalOp_constNat 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.constBool IS_CAUSAL).ite
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
        (Op.constNat 128)) _
        = some (Tile.scalar (flashHi s IS_CAUSAL)) from by
      conv_lhs => unfold evalOp
      rw [flash_evalOp_constBool]
      simp only [Tile.scalar_data, Option.bind_eq_bind, Option.bind_some]
      cases IS_CAUSAL
      · simp only [Bool.false_eq_true, if_false, flashHi]
        exact evalOp_constNat 128 _
      · simp only [if_true, flashHi]
        rw [evalOp_mul, evalOp_add]
        simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
          ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
          Option.bind_eq_bind, Option.bind_some, flash_scalarBop]
        rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  all_goals
    simp only [FloatDType.toTileDType, BlockState.setReg_ne_name, BlockState.setReg_same,
      BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
      and_self, and_true, true_and]

/-- The 15 lowered loop-body statements of the Python-shape flash-attn body
(`IS_CAUSAL` parameter only affects the `ifThen` mask, stmt L3). -/
def flashLoopBody (IS_CAUSAL : Bool) : List Stmt :=
  [ Stmt.assign .real [64, 64] "k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") []) MaskOpt.none),
    Stmt.assign .real [64, 64] "v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") []) MaskOpt.none),
    Stmt.assign .real [128, 64] "qk" (Op.full [128, 64] (Op.const 0)),
    Stmt.ifThen (Op.constBool IS_CAUSAL)
      [ Stmt.assign .real [128, 64] "qk"
          (Op.where
            (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "off_m"))
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "off_n"))))
            (Op.ref .real [128, 64] "qk") (Op.broadcast Op.negInf [128, 64])) ],
    Stmt.assign .real [128, 64] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "qk")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [128, 64] "q"))
          (Op.ref .real [64, 64] "k"))),
    Stmt.assign .real [128] "max_new"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "max")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "qk")))
        (Op.ref .real [128] "max")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "qk"))),
    Stmt.assign .real [128] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "max")
        (Op.ref .real [128] "max_new"))),
    Stmt.assign .real [128, 64] "nume"
      (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "max_new")))),
    Stmt.assign .real [128] "out_scale"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real Broadcast.scalarR (Op.ref .real [128] "denom") (Op.const 0))
        (Op.ref .real [128] "alpha")),
    Stmt.assign .real [128, 64] "out_buffer"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "out_buffer")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "out_scale"))),
    Stmt.assign .real [128, 64] "out_buffer"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "out_buffer")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume"))) (Op.ref .real [64, 64] "v"))),
    Stmt.assign .real [128] "denom"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "denom")
          (Op.ref .real [128] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "nume"))),
    Stmt.assign .real [128] "max" (Op.ref .real [128] "max_new"),
    Stmt.assign .blockPtr [64, 64] "K_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") [0, 64]),
    Stmt.assign .blockPtr [64, 64] "V_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") [64, 0]) ]

/-- The lowered body `drop 16` is `forRangeDyn … flashLoopBody :: postLoop`. -/
theorem flashLoopBody_check (Q K V L O : RegionName) (sm_scale : ℝ) (IS_CAUSAL : Bool) :
    (flash_attn_fwd_kernel_surface Q K V L O sm_scale
        16384 8192 64 1 16384 8192 64 1 16384 8192 64 1 16384 8192 64 1
        2 2 128 128 64 64 IS_CAUSAL).toAlgKernel.body.drop 16
      = Stmt.forRangeDyn "start_n" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
          (Op.constNat 64) (flashLoopBody IS_CAUSAL)
        :: (flash_attn_fwd_kernel_surface Q K V L O sm_scale
            16384 8192 64 1 16384 8192 64 1 16384 8192 64 1 16384 8192 64 1
            2 2 128 128 64 64 IS_CAUSAL).toAlgKernel.body.drop 17 := by
  cases IS_CAUSAL <;> rfl

/-! ### attn_step — one loop iteration advances the invariant

The loop body, entered with `start_n = i` (= block `c·BLOCK_N`, `i = c·64`),
streams block `c`'s keys and advances the ⊥-seeded `flashStateBot`/`flashRunningMax`
running state by one block. The kernel's explicit per-block tile arithmetic
(`max_new = max(max, blockMax)`, `denom' = denom·α + Σnume`,
`out_buffer' = out_buffer·α + dot(nume, v)`, with `α = exp2(max − max_new)`,
`nume = exp2(qk − max_new)`) realizes exactly the ⊥-aware block update — proved by
expanding both sides onto the banked closed forms
`flashStateBot_{snd_fst,snd_snd}` (`= κ(M)·Σ pow2 score [· v]`) and the
window-split / block bridges (`flashKeysUpto_succ`, `flashRunningMax_succ`,
`flashBlock_blockMax`, `flashBlock_map_sum`). -/

/-- Overwriting a register slot shadows the inner write (BlockState ext). -/
theorem flash_setReg_shadow (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (a b : Tile dtype shape) :
    (s.setReg name dtype shape a).setReg name dtype shape b = s.setReg name dtype shape b := by
  have hregs : ∀ (dt : TileDType) (sh : TileShape) (nm : RegName),
      ((s.setReg name dtype shape a).setReg name dtype shape b).regs dt sh nm
        = (s.setReg name dtype shape b).regs dt sh nm := by
    intro dt sh nm
    by_cases hnm : nm = name
    · subst hnm
      by_cases hdt : dt = dtype
      · subst hdt
        by_cases hsh : sh = shape
        · subst hsh; simp only [BlockState.setReg_same]
        · simp only [BlockState.setReg_ne_shape _ _ _ _ _ _ _ _ rfl rfl hsh]
      · simp only [BlockState.setReg_ne_dtype _ _ _ _ _ _ _ _ rfl hdt]
    · simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hnm]
  exact BlockState.ext (fun _ _ => by simp only [BlockState.setReg_mem]) hregs
    (fun _ => by simp only [BlockState.setReg_pids])
    (fun _ _ => by simp only [BlockState.setReg_undef])

set_option maxHeartbeats 1000000 in
/-- **L10 (out_buffer accumulate, inline double cast)**: `out_buffer += dot(
(nume.to fp16).to real, v)`. The real→fp16→real round-trip of `nume` is identity
in the model, so the dot accumulates `dot(nume, v)` onto the running buffer. -/
theorem flash_outbuf_acc2_op_eval (s : BlockState)
    (ob1tile : Tile .real [128, 64]) (ntile : Tile .real [128, 64]) (vtile : Tile .real [64, 64])
    (hob : s.regs .real [128, 64] "out_buffer" = some ob1tile)
    (hn : s.regs .real [128, 64] "nume" = some ntile)
    (hv : s.regs .real [64, 64] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) (Op.ref .real [128, 64] "out_buffer")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume"))) (Op.ref .real [64, 64] "v"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) ob1tile
          (Tile.dot [] ntile vtile)) := by
  have hcastInner : evalOp (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume")) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ntile.data i)⟩ : Tile .fp16 [128, 64]) := by
    rw [evalOp_castFloat]; simp [evalOp_ref, hn]
  have hcast2 : evalOp (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume"))) s
      = some ntile := by
    rw [evalOp_castFloat, hcastInner]
    refine congrArg some ?_; ext i; simp [FloatDType.cast]
  have hcast2ann : @evalOp TileDType.real [128, 64]
      (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume"))) s
      = some ntile := hcast2
  have hdotN : evalOp (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume"))) (Op.ref .real [64, 64] "v")) s
      = some (Tile.dot [] ntile vtile) := by
    rw [evalOp_dot]; simp [hcast2ann, hv]
  have hdotN2 : @evalOp TileDType.real [128, 64]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume"))) (Op.ref .real [64, 64] "v")) s
      = some (Tile.dot [] ntile vtile) := hdotN
  rw [evalOp_add]; simp only [evalOp_ref, hob, hdotN2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- The L2–L3 qk-seed + causal-mask prefix of `flashLoopBody`. -/
def flashQkSeedStmts (IS_CAUSAL : Bool) : List Stmt :=
  [ Stmt.assign .real [128, 64] "qk" (Op.full [128, 64] (Op.const 0)),
    Stmt.ifThen (Op.constBool IS_CAUSAL)
      [ Stmt.assign .real [128, 64] "qk"
          (Op.where
            (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "off_m"))
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "off_n"))))
            (Op.ref .real [128, 64] "qk") (Op.broadcast Op.negInf [128, 64])) ] ]

/-- The L4–L14 tail of `flashLoopBody` (after the L0–L3 loads + qk seed/mask):
`qk += dot q k`, the running-state update (`max_new`/`alpha`/`nume`/`out_scale`/
`out_buffer`/`denom`/`max`), and the two block-pointer advances. -/
def flashLoopBodyTail (IS_CAUSAL : Bool) : List Stmt :=
  [ Stmt.assign .real [128, 64] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "qk")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [128, 64] "q"))
          (Op.ref .real [64, 64] "k"))),
    Stmt.assign .real [128] "max_new"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "max")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "qk")))
        (Op.ref .real [128] "max")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "qk"))),
    Stmt.assign .real [128] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "max")
        (Op.ref .real [128] "max_new"))),
    Stmt.assign .real [128, 64] "nume"
      (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "max_new")))),
    Stmt.assign .real [128] "out_scale"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real Broadcast.scalarR (Op.ref .real [128] "denom") (Op.const 0))
        (Op.ref .real [128] "alpha")),
    Stmt.assign .real [128, 64] "out_buffer"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "out_buffer")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "out_scale"))),
    Stmt.assign .real [128, 64] "out_buffer"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 64] "out_buffer")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "nume"))) (Op.ref .real [64, 64] "v"))),
    Stmt.assign .real [128] "denom"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "denom")
          (Op.ref .real [128] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "nume"))),
    Stmt.assign .real [128] "max" (Op.ref .real [128] "max_new"),
    Stmt.assign .blockPtr [64, 64] "K_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") [0, 64]),
    Stmt.assign .blockPtr [64, 64] "V_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") [64, 0]) ]

/-- `flashLoopBody` is the 2 loads, then the qk seed/mask prefix, then the tail. -/
theorem flashLoopBody_eq_tail (IS_CAUSAL : Bool) :
    flashLoopBody IS_CAUSAL
      = (Stmt.assign .real [64, 64] "k"
          (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") []) MaskOpt.none))
        :: (Stmt.assign .real [64, 64] "v"
          (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") []) MaskOpt.none))
        :: (flashQkSeedStmts IS_CAUSAL ++ flashLoopBodyTail IS_CAUSAL) := by
  cases IS_CAUSAL <;> rfl

/-- The masked seed tile after L2 (`qk = full 0`) and L3 (causal `where`):
cell `(i,j)` is `some 0` when `(IS_CAUSAL → SN+j ≤ gm i)`, else `⊥` (`-inf`). -/
noncomputable def flashMaskedSeed (IS_CAUSAL : Bool) (SN : Nat) (gm : Fin 128 → Nat) :
    Tile .real [128, 64] :=
  ⟨fun idx : TileIndex [128, 64] =>
    if (IS_CAUSAL → SN + idx.2.1.val ≤ gm idx.1) then some (0 : ℝ) else (⊥ : WithBot ℝ)⟩

set_option maxHeartbeats 1000000 in
/-- **L2–L3 (qk seed + causal mask).** Stepping `[qk = full 0, ifThen IS_CAUSAL {…}]`
from a state with `off_m`/`off_n`/`start_n` set yields `s.setReg "qk" flashMaskedSeed`.
The all-`0` seed of L2 makes the `qk` slots clean, so the no-op (non-causal) and
the `where`-rewrite (causal) both land on the single `setReg "qk" flashMaskedSeed`. -/
theorem flash_qk_seed_steps (IS_CAUSAL : Bool) (s : BlockState) (SN : Nat)
    (gm : Fin 128 → Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hom : s.regs .nat [128] "off_m" = some (Tile.vec gm))
    (hon : s.regs .nat [64] "off_n" = some (Tile.vec (fun j : Fin 64 => j.val))) :
    stepStmts (flashQkSeedStmts IS_CAUSAL) s
      = some (s.setReg "qk" .real [128, 64] (flashMaskedSeed IS_CAUSAL SN gm)) := by
  unfold flashQkSeedStmts
  set zeroQk : Tile .real [128, 64] := ⟨fun _ : TileIndex [128, 64] => some (0 : ℝ)⟩ with hzeroQk
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 64] (Op.const 0)) s = some zeroQk from by
      simp [evalOp_full, evalOp_const, hzeroQk]))]
  match IS_CAUSAL with
  | Bool.false =>
    rw [stepStmts.cons_some (show stepStmt (Stmt.ifThen (Op.constBool Bool.false) _) _ = some _
        from flash_stepStmt_ifThen_false _ _), stepStmts.nil]
    refine congrArg some ?_
    refine congrArg (s.setReg "qk" .real [128, 64]) ?_
    ext idx; simp only [hzeroQk, flashMaskedSeed, Bool.false_eq_true, false_implies, if_true]
  | Bool.true =>
    rw [stepStmts.cons_some (show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) _ = some _
        from by rw [flash_stepStmt_ifThen_true]
                rw [stepStmts.cons_some (stepStmt_assign_eq_some
                  (flash_where_op_eval _ 128 64 SN gm zeroQk
                    (by simp [BlockState.setReg_ne_name, hom])
                    (by simp [BlockState.setReg_ne_name, hon])
                    (by simp [BlockState.setReg_ne_name, hsn])
                    (by rw [BlockState.setReg_same])))]
                rw [stepStmts.nil]), stepStmts.nil]
    refine congrArg some ?_
    rw [flash_setReg_shadow]
    refine congrArg (s.setReg "qk" .real [128, 64]) ?_
    ext idx
    simp only [hzeroQk, flashMaskedSeed, true_implies, Tile.mk.injEq]

/-- The per-cell `qk` value the kernel's L2–L4 produce for global key `SN + j`,
output row `i`: `if (causal → SN+j ≤ gm i) then some(dot) else ⊥`. The masked
score: `⊥` (`-inf`) on future causal keys, else the real `q·k` dot. -/
noncomputable def flashQkCell (IS_CAUSAL : Bool) (SN : Nat) (gm : Fin 128 → Nat)
    (qtile : Tile .fp16 [128, 64]) (ktile : Tile .real [64, 64])
    (i : Fin 128) (j : Fin 64) : WithBot ℝ :=
  if (IS_CAUSAL → SN + j.val ≤ gm i) then
    (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [128, 64]) ktile).data (i, j, PUnit.unit)
  else (⊥ : WithBot ℝ)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Loop-body execution chain.** The 15 lowered `flashLoopBody` statements step
the iteration-entry state `sin` (with `start_n = SN`, and the invariant's register
readbacks `max`/`denom`/`out_buffer`/`q`/`off_m`/`off_n`/`K_block_ptr`/
`V_block_ptr`) to a final state `sF`, exposing the symbolic `max`/`denom`/
`out_buffer` register values (the kernel's per-block tile arithmetic over the
masked `qk` cell `flashQkCell`) plus the advanced block pointers. Threaded through
`stepStmts.cons_some` via the banked op-eval recipes; the causal `ifThen` mask
folds into `flashQkCell`. -/
theorem flashLoopBody_steps (IS_CAUSAL : Bool) (sin : BlockState) (SN : Nat)
    (gm : Fin 128 → Nat)
    (Kreg Vreg : RegionName) (kbase vbase kcol vrow : Nat)
    (qtile : Tile .fp16 [128, 64]) (mtile dtile : Tile .real [128])
    (obtile : Tile .real [128, 64]) (ktile vtile : Tile .real [64, 64])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hom : sin.regs .nat [128] "off_m" = some (Tile.vec gm))
    (hon : sin.regs .nat [64] "off_n" = some (Tile.vec (fun j : Fin 64 => j.val)))
    (hmax : sin.regs .real [128] "max" = some mtile)
    (hden : sin.regs .real [128] "denom" = some dtile)
    (hob : sin.regs .real [128, 64] "out_buffer" = some obtile)
    (hq : sin.regs .fp16 [128, 64] "q" = some qtile)
    (hKp : sin.regs .blockPtr [64, 64] "K_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Kreg, baseOffset := kbase, parentShape := [64, 128],
          blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol] }⟩))
    (hVp : sin.regs .blockPtr [64, 64] "V_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Vreg, baseOffset := vbase, parentShape := [128, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [vrow, 0] }⟩))
    (hkload : ∀ idx : TileIndex [64, 64],
      ktile.data idx = some (sin.readMem Kreg (kbase + idx.1.val * 1 + (kcol + idx.2.1.val) * 64)))
    (hvload : ∀ idx : TileIndex [64, 64],
      vtile.data idx = some (sin.readMem Vreg (vbase + (vrow + idx.1.val) * 64 + idx.2.1.val * 1)))
    (Qptr : Tile .blockPtr [128, 64]) (hQp : sin.regs .blockPtr [128, 64] "Q_block_ptr" = some Qptr)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (flashLoopBody IS_CAUSAL) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .fp16 [128, 64] "q" = some qtile
      ∧ sF.regs .blockPtr [128, 64] "Q_block_ptr" = some Qptr
      ∧ sF.regs .nat [128] "off_m" = some (Tile.vec gm)
      ∧ sF.regs .nat [64] "off_n" = some (Tile.vec (fun j : Fin 64 => j.val))
      ∧ sF.regs .blockPtr [64, 64] "K_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Kreg, baseOffset := kbase, parentShape := [64, 128],
              blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol + 64] }⟩)
      ∧ sF.regs .blockPtr [64, 64] "V_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Vreg, baseOffset := vbase, parentShape := [128, 64],
              blockShape := [64, 64], strides := [64, 1], offsets := [vrow + 64, 0] }⟩)
      ∧ ∃ (qkT : Tile .real [128, 64]) (rmaxT mnewT alphaT : Tile .real [128])
            (numeT : Tile .real [128, 64]) (ostileT : Tile .real [128]),
          (∀ i : Fin 128, ∀ j : Fin 64,
            qkT.data (i, j, PUnit.unit) = flashQkCell IS_CAUSAL SN gm qtile ktile i j)
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qkT = some rmaxT
          ∧ mnewT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT)
          ∧ numeT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT))
          ∧ ostileT = Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul Broadcast.scalarR dtile (Tile.scalar (some 0))) alphaT
          ∧ sF.regs .real [128] "max" = some mnewT
          ∧ sF.regs .real [128] "denom" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) dtile alphaT)
              (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) numeT))
          ∧ sF.regs .real [128, 64] "out_buffer" = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) obtile (Tile.expandDim ⟨1, by simp⟩ ostileT))
              (Tile.dot [] numeT vtile)) := by
  -- the q·k dot tile (cast of fp16 q against the loaded k)
  set qrealT : Tile .real [128, 64] :=
    ⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ with hqreal
  set dotT : Tile .real [128, 64] := Tile.dot [] qrealT ktile with hdotT
  -- the qk tile after L2–L4: full-0 (+ causal mask) + dot
  set qkT : Tile .real [128, 64] :=
    ⟨fun idx => flashQkCell IS_CAUSAL SN gm qtile ktile idx.1 idx.2.1⟩ with hqkT
  have hqkData : ∀ i : Fin 128, ∀ j : Fin 64,
      qkT.data (i, j, PUnit.unit) = flashQkCell IS_CAUSAL SN gm qtile ktile i j := fun _ _ => rfl
  -- reduceMax exists
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [128, 64] (⟨1, by simp⟩ : Fin [128, 64].length) from by decide)]⟩
  set mnewT : Tile .real [128] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmnew
  set alphaT : Tile .real [128] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT) with halpha
  set numeT : Tile .real [128, 64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT)) with hnume
  set ostileT : Tile .real [128] := Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul Broadcast.scalarR dtile (Tile.scalar (some 0))) alphaT with hostile
  rw [flashLoopBody_eq_tail]
  -- L0: k = load K_block_ptr  (rewrite directly to ktile)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") []) MaskOpt.none) sin
        = some ktile from by
      rw [flash_load_K_eval Kreg kbase 64 128 64 64 1 64 kcol
        (Op.ref .blockPtr [64, 64] "K_block_ptr") sin (by rw [evalOp_ref]; simp [hKp])]
      refine congrArg some ?_; ext idx; rw [hkload idx]))]
  -- L1: v = load V_block_ptr  (rewrite directly to vtile)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") []) MaskOpt.none) _
        = some vtile from by
      rw [flash_load_Q_eval Vreg vbase 128 64 64 64 64 1 vrow
        (Op.ref .blockPtr [64, 64] "V_block_ptr") _ (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hVp])]
      refine congrArg some ?_; ext idx
      simp only [BlockState.setReg_readMem]
      rw [hvload idx]))]
  -- L2–L3: qk seed + causal mask  →  flashMaskedSeed (via the 2-statement prefix)
  rw [stepStmts.append_some (l2 := flashLoopBodyTail IS_CAUSAL)
    (flash_qk_seed_steps IS_CAUSAL _ SN gm
      (by simp [BlockState.setReg_ne_name, hsn])
      (by simp [BlockState.setReg_ne_name, hom])
      (by simp [BlockState.setReg_ne_name, hon]))]
  unfold flashLoopBodyTail
  -- L4: qk += dot q k   →  qkT (= flashQkCell tile)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [128, 64] "qk")
          (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [128, 64] "q"))
            (Op.ref .real [64, 64] "k"))) _ = some qkT from by
      rw [flash_qkdot_op_eval _ 128 64 64 (flashMaskedSeed IS_CAUSAL SN gm) qtile ktile
        (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hq])
        (by simp [BlockState.setReg_ne_name])]
      refine congrArg some ?_; ext idx
      simp only [hqkT, flashQkCell, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        flashMaskedSeed, NumericDType.add]
      by_cases h : (IS_CAUSAL → SN + idx.2.1.val ≤ gm idx.1)
      · rw [if_pos h, if_pos h]
        cases hd : (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [128, 64]) ktile).data idx with
        | none => simp [WithBot.realAdd, Option.map₂, Option.bind]
        | some r => simp [WithBot.realAdd, Option.map₂, Option.bind]
      · rw [if_neg h, if_neg h]
        simp only [WithBot.realAdd, Option.map₂, Option.bind]
        rfl))]
  -- L5: max_new = maximum(max, reduceMax qk 1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_maxnew_op_eval _ 128 64 mtile qkT rmaxT
      (by simp [BlockState.setReg_ne_name, hmax]) (by rw [BlockState.setReg_same]) hrm))]
  -- L6: alpha = exp2(max - max_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_alpha_op_eval _ 128 mtile mnewT
      (by simp [BlockState.setReg_ne_name, hmax]) (by simp [BlockState.setReg_same, hmnew])))]
  -- L7: nume = exp2(qk - max_new[:,None])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_nume_op_eval _ 128 64 (by simp) qkT mnewT
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name, hmnew])))]
  -- L8: out_scale = denom * 0 + alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_outscale_op_eval _ 128 dtile alphaT
      (by simp [BlockState.setReg_ne_name, hden]) (by simp [BlockState.setReg_ne_name, halpha])))]
  -- L9: out_buffer *= out_scale[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_outbuf_rescale_op_eval _ 128 64 (by simp) obtile ostileT
      (by simp [BlockState.setReg_ne_name, hob]) (by simp [BlockState.setReg_same, hostile])))]
  -- L10: out_buffer += dot((nume.to fp16).to real, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_outbuf_acc2_op_eval _
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) obtile (Tile.expandDim ⟨1, by simp⟩ ostileT))
      numeT vtile
      (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hnume])
      (by simp [BlockState.setReg_ne_name])))]
  -- L11: denom = denom * alpha + sum nume 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_denom_op_eval _ 128 64 dtile alphaT numeT
      (by simp [BlockState.setReg_ne_name, hden]) (by simp [BlockState.setReg_ne_name, halpha])
      (by simp [BlockState.setReg_ne_name, hnume])))]
  -- L12: max = max_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "max_new") _ = some mnewT from by
      rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hmnew]))]
  -- L13: K_block_ptr = advance(K_block_ptr, [0, 64])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_advance_col_eval _ Kreg kbase 64 128 64 64 1 64 kcol 64 "K_block_ptr"
      (by simp [BlockState.setReg_ne_name, hKp])))]
  -- L14: V_block_ptr = advance(V_block_ptr, [64, 0])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (flash_advance_row_eval _ Vreg vbase 128 64 64 64 64 1 vrow 64 "V_block_ptr"
      (by simp [BlockState.setReg_ne_name, hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, qkT, rmaxT, mnewT, alphaT, numeT, ostileT,
    hqkData, hrm, rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef, hundef]
  · simp [BlockState.setReg_ne_name, hq]
  · simp [BlockState.setReg_ne_name, hQp]
  · simp [BlockState.setReg_ne_name, hom]
  · simp [BlockState.setReg_ne_name, hon]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]

/-- The `WithBot ⊔`-fold of a block's coerced scores equals `↑(block max)` over the
running max `m`, OR `m` if the block is empty — phrased as: the kernel's
`mnewT = m ⊔ blockSup` equals the running max after folding `osStepBot` over the
block (the `.1` channel of `flashStateBot_succ`). -/
theorem osStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl osStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [flashStateBot_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

/-- **The block-at-once update equals the key-by-key `osStepBot` fold** (the C.2
heart). For a block with max `M' = m ⊔ blockSup` and a state `(m, l, acc)` anchored
to the true denominator/accumulator via `l = κ(m)·L`, `acc = κ(m)·T`, the kernel's
one-shot rescale-and-add (`l·α + Σ exp2(s−M')`, `acc·α + Σ exp2(s−M')·v`, with
`α = realExp2(m ⊖ M')`) lands on `block.foldl osStepBot (m, l, acc)`. Both sides are
`(M', κ(M')·(L+Σpow2 s), κ(M')·(T+Σpow2 s·v))` by `osStepBot_foldl_consistent`. -/
theorem osStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let M' := m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (M',
     l * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0))).sum,
     acc * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum)
      = block.foldl osStepBot (m, l, acc) := by
  intro M'
  -- the fold's three channels via consistency
  have hfst : (block.foldl osStepBot (m, l, acc)).1 = M' := by
    rw [osStepBot_block_fst]
  obtain ⟨hfold_l, hfold_acc⟩ := osStepBot_foldl_consistent block m l acc T L hl hacc hmL hmT
  rw [hfst] at hfold_l hfold_acc
  -- the sums over the (coerced) block vanish when M' = ⊥ (block is empty)
  have hM'eq : M' = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := rfl
  cases hM' : M' with
  | bot =>
    -- M' = ⊥ forces block = [] (any block element would push the sup above ⊥)
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have : ((a.1 : ℝ) : WithBot ℝ) ≤ M' := by
          rw [hM'eq]
          exact le_sup_of_le_right (by simp only [List.map_cons, List.foldr_cons]; exact le_sup_left)
        rw [hM'] at this
        exact absurd (le_bot_iff.mp this) (WithBot.coe_ne_bot)
    have hm0 : m = ⊥ := by
      rw [hM'eq, hempty] at hM'
      simpa only [List.map_nil, List.foldr_nil, sup_bot_eq] using hM'
    have hl0 : l = 0 := by rw [hl, hm0]; simp [hmL hm0]
    have hacc0 : acc = 0 := by rw [hacc, hm0]; simp [hmT hm0]
    subst hempty
    rw [hl0, hacc0]
    simp only [List.foldl_nil, List.map_nil, List.sum_nil, add_zero, mul_zero, zero_mul]
    rw [hm0]
  | coe Mr =>
    rw [hM'] at hfst hfold_l hfold_acc
    -- M' finite: l·α = pow2(-Mr)·L, acc·α = pow2(-Mr)·T (covering m = ⊥ via L = T = 0)
    have hlα : l * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * L := by
      cases hm : m with
      | bot =>
        rw [hl, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmL hm]; ring
      | coe a =>
        rw [hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have haccα : acc * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * T := by
      cases hm : m with
      | bot =>
        rw [hacc, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmT hm]; ring
      | coe a =>
        rw [hacc, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have hsumL : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0))).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1)).sum := by
      have := sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun _ => 1)
      simp only [mul_one] at this
      rw [this, WithBot.unbotD_coe]
    have hsumT : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0) * p.2)).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1 * p.2)).sum := by
      rw [sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun p => p.2), WithBot.unbotD_coe]
    refine Prod.ext hfst.symm (Prod.ext ?_ ?_)
    · rw [hfold_l, hlα, hsumL, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring
    · rw [hfold_acc, haccα, hsumT, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring

/-- **Reduce-max row bridge.** `tl.max(qk, 1)` at row `r`: the `reduceMaxDrop`
of a tile whose row-`r` cells equal `g jL` reads off `Finset.univ.sup g`. -/
theorem flash_reduceMaxDrop_row (qk : Tile .real [128, 64])
    (rmaxT : Tile .real [128])
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qk = some rmaxT)
    (r : Fin 128) (g : Fin 64 → WithBot ℝ)
    (hqk : ∀ jL : Fin 64, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [128, 64] (⟨1, by simp⟩ : Fin [128, 64].length) from by decide)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- **The `q·k` dot cell is the `flashKV` score.** With `q = cast_fp16(scale·qT)`
loaded and the K block reading `kT` at global key `c·64 + jL`, the dot of the
cast-back-to-real `q` against the loaded `k` at cell `(r, jL)` is
`some (scale · Σ_e qT(r,e)·kT(c·64+jL, e))` — the `flashKV` score `.1`. -/
theorem flash_dot_score_cell (s0 : BlockState) (Q K : RegionName) (scale : ℝ) (c : Nat)
    (r : Fin 128) (jL : Fin 64) (hjL : c * 64 + jL.val < 128)
    (qtile : Tile .fp16 [128, 64]) (ktile : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [128, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some (scale * qTile s0 Q 8192 64 128 idx))⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64))) :
    (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [128, 64]) ktile).data (r, jL, PUnit.unit)
      = some (scale * Finset.univ.sum (fun e : Fin 64 =>
          qTile s0 Q 8192 64 128 (r, e, PUnit.unit) * kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·)
          ((⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [128, 64]).data (r, e, PUnit.unit))
          (ktile.data (e, jL, PUnit.unit))))
      = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun e => (some (scale * qTile s0 Q 8192 64 128 (r, e, PUnit.unit)
            * kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by
        simp only [hq, hk (e, jL, PUnit.unit), FloatDType.cast, FloatDType.real_ofWithBot,
          FloatDType.toWithBot, FloatDType.real_toWithBot, Option.map₂]
        rw [show kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)
              = s0.readMem K (s0.pids 1 * 8192 + (c * 64 + jL.val) * 64 + e.val) from by
          simp only [kTile, flashBaseOffset]]
        simp only [Option.bind, Option.map]
        refine congrArg some ?_
        rw [show e.val * 1 = e.val from by ring]
        ring_nf)]
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun e => (some (scale * qTile s0 Q 8192 64 128 (r, e, PUnit.unit)
            * kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)) : WithBot ℝ)))
      = ((Finset.univ.sum (fun e : Fin 64 => scale * qTile s0 Q 8192 64 128 (r, e, PUnit.unit)
          * kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
      from (WithBot.coe_sum Finset.univ _).symm]
  rw [Finset.mul_sum]
  refine congrArg some ?_
  exact Finset.sum_congr rfl (fun e _ => by ring)

/-- Canonical axis-1 index of `[128, 64]` (the reduce/sum axis). Naming it makes the
proof term stable, so `Tile.reduceSumDrop flashAx1 …` matches across lemmas (unlike
`⟨1, by simp⟩`, whose `by simp` proof varies per elaboration). -/
def flashAx1 : Fin [128, 64].length := ⟨1, by decide⟩

/-! ### attn_step channel bridges (denom / out_buffer)

The loop body's symbolic `denom`/`out_buffer` registers (after `flashLoopBody_steps`)
carry the kernel's per-block rescale-and-add tile arithmetic over the masked
`flashQkCell`. These bridges read them off `flashStateBot((c+1)*64)` via
`osStepBot_block_eq` + `flashStateBot_succ`, reducing the kernel's `Fin 64` masked
sum to the `flashBlock` list sum through `flashBlock_map_sum`. -/

/-- Any member of a `WithBot ℝ` list is `≤` its `foldr (⊔) ⊥`. -/
theorem flash_mem_le_foldr_sup (a : WithBot ℝ) :
    ∀ (L : List (WithBot ℝ)), a ∈ L → a ≤ L.foldr (· ⊔ ·) ⊥ := by
  intro L
  induction L with
  | nil => intro h; simp at h
  | cons x t ih =>
    intro h
    simp only [List.foldr_cons]
    rcases List.mem_cons.mp h with h | h
    · rw [h]; exact le_sup_left
    · exact le_trans (ih h) le_sup_right

/-- The ⊥-seeded running max over a nonempty window `[0, hi)` with `hi > 0` is `≠ ⊥`:
key `0` is always present (`0 < hi`) and passes the causal filter (`0 ≤ qStart + i`),
so the score list is nonempty. -/
theorem flashRunningMax_ne_bot
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (hhi : 0 < hi) (hSEQ : 0 < SEQLEN) :
    flashRunningMax qT kT vT scale causal qStart hi i d ≠ ⊥ := by
  unfold flashRunningMax flashKeysUpto
  have hmem : (⟨0, hSEQ⟩ : Fin SEQLEN) ∈ List.finRange SEQLEN := List.mem_finRange _
  -- the score of key 0 is a member of the coerced list, and it's ≠ ⊥
  set L := ((List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
      if j.val < hi ∧ (causal → j.val ≤ qStart + i.val)
      then some (flashKV qT kT vT scale i d j) else none)).map
      (fun p => ((p.1 : ℝ) : WithBot ℝ)) with hL
  have hmemL : ((flashKV qT kT vT scale i d ⟨0, hSEQ⟩).1 : WithBot ℝ) ∈ L := by
    rw [hL, List.mem_map]
    refine ⟨flashKV qT kT vT scale i d ⟨0, hSEQ⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    exact ⟨⟨0, hSEQ⟩, hmem, by
      rw [if_pos (show ((⟨0, hSEQ⟩ : Fin SEQLEN).val < hi ∧
        (causal → (⟨0, hSEQ⟩ : Fin SEQLEN).val ≤ qStart + i.val)) from ⟨hhi, fun _ => Nat.zero_le _⟩)]⟩
  -- foldr (⊔) ⊥ ≥ any member
  have hle : ((flashKV qT kT vT scale i d ⟨0, hSEQ⟩).1 : WithBot ℝ) ≤ L.foldr (· ⊔ ·) ⊥ :=
    flash_mem_le_foldr_sup _ L hmemL
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) (WithBot.coe_ne_bot)

/-- **`mnewT = flashRunningMax((c+1)·64)`.** The kernel `max_new = maximum(max, max(qk,1))`
register (given `max = flashRunningMax(c·64)` and `reduceMax qk = block sup` of the
masked `flashQkCell` scores) equals the ⊥-seeded running max after `c+1` blocks. -/
theorem flash_mnewT_eq (s0 : BlockState) (Q K : RegionName) (scale : ℝ) (causal : Bool)
    (vT : TileIndex [128, 64] → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (r : Fin 128)
    (qtile : Tile .fp16 [128, 64]) (ktile : Tile .real [64, 64])
    (mtile rmaxT : Tile .real [128])
    (hq : qtile = ⟨fun idx : TileIndex [128, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some (scale * qTile s0 Q 8192 64 128 idx))⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hmtile : mtile.data (r, PUnit.unit)
        = flashRunningMax (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) vT
            scale causal (s0.pids 0 * 128) (c * 64) r ⟨0, by norm_num⟩)
    (hrmax : rmaxT.data (r, PUnit.unit)
        = Finset.univ.sup (fun jL : Fin 64 =>
            flashQkCell causal (c * 64) (fun rr : Fin 128 => s0.pids 0 * 128 + rr.val) qtile ktile r jL)) :
    (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT).data (r, PUnit.unit)
      = flashRunningMax (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) vT
          scale causal (s0.pids 0 * 128) ((c + 1) * 64) r ⟨0, by norm_num⟩ := by
  -- block sup of qkCell scores = the flashRunningMax_succ block term
  have hblock : Finset.univ.sup (fun jL : Fin 64 =>
        flashQkCell causal (c * 64) (fun rr : Fin 128 => s0.pids 0 * 128 + rr.val) qtile ktile r jL)
      = Finset.univ.sup (fun jL : Fin 64 =>
          if (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val) then
            ((scale * Finset.univ.sum (fun e : Fin 64 =>
                qTile s0 Q 8192 64 128 (r, e, PUnit.unit) *
                  kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, by
                    have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
          else ⊥) := by
    refine Finset.sup_congr rfl (fun jL _ => ?_)
    unfold flashQkCell
    by_cases hcond : (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val)
    · rw [if_pos hcond, if_pos hcond,
        flash_dot_score_cell s0 Q K scale c r jL (by
            have := jL.isLt; omega) qtile ktile hq hk]
      rfl
    · rw [if_neg hcond, if_neg hcond]
  rw [flashRunningMax_succ (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) vT
      scale causal (s0.pids 0 * 128) 64 c r ⟨0, by norm_num⟩ hc1]
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmtile, hrmax, hblock]
  set M := flashRunningMax (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) vT
      scale causal (s0.pids 0 * 128) (c * 64) r ⟨0, by norm_num⟩ with hM
  set S := Finset.univ.sup (fun jL : Fin 64 =>
      if (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val) then
        ((scale * Finset.univ.sum (fun e : Fin 64 =>
            qTile s0 Q 8192 64 128 (r, e, PUnit.unit) *
              kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, by
                have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
      else ⊥) with hS
  by_cases h : M ≤ S
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- The kernel's `Σ_jL nume[r,jL]` over the `Fin 64` masked block equals the
`flashBlock` list sum of `pow2(score − M')`. The masked (`⊥`) cells contribute
`realExp2(⊥ − M') = some 0`. -/
theorem flash_nume_row_sum (s0 : BlockState) (Q K V : RegionName) (scale : ℝ) (causal : Bool)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (r : Fin 128) (d : Fin 64)
    (qtile : Tile .fp16 [128, 64]) (ktile : Tile .real [64, 64])
    (mnewT : Tile .real [128]) (numeT : Tile .real [128, 64])
    (hq : qtile = ⟨fun idx : TileIndex [128, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some (scale * qTile s0 Q 8192 64 128 idx))⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (Mr : ℝ)
    (hmnew : mnewT.data (r, PUnit.unit) = (Mr : WithBot ℝ))
    (hnume : ∀ jL : Fin 64, numeT.data (r, jL, PUnit.unit)
        = WithBot.realExp2 (WithBot.realSub
            (flashQkCell causal (c * 64) (fun rr : Fin 128 => s0.pids 0 * 128 + rr.val) qtile ktile r jL)
            (mnewT.data (r, PUnit.unit)))) :
    (Tile.reduceSumDrop flashAx1 numeT).data (r, PUnit.unit)
      = some ((flashBlock (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
          scale causal (s0.pids 0 * 128) 64 c r d).map (fun p => pow2 (p.1 - Mr))).sum := by
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin 64,
      numeT.data (TileShape.insertAxisIndex [128, 64] flashAx1 (r, PUnit.unit) jL)
        = some (if (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val)
            then pow2 ((scale * Finset.univ.sum (fun e : Fin 64 =>
                  qTile s0 Q 8192 64 128 (r, e, PUnit.unit) *
                    kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, by
                      have := jL.isLt; omega⟩, e, PUnit.unit))) - Mr)
            else 0) := by
    intro jL
    rw [show (TileShape.insertAxisIndex [128, 64] flashAx1 (r, PUnit.unit) jL) = (r, jL, PUnit.unit) from rfl]
    rw [hnume jL, hmnew]
    unfold flashQkCell
    by_cases hcond : (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val)
    · rw [if_pos hcond, if_pos hcond,
        flash_dot_score_cell s0 Q K scale c r jL (by
          have := jL.isLt; omega) qtile ktile hq hk]
      simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
      refine congrArg some ?_; simp only [pow2]; ring_nf
    · rw [if_neg hcond, if_neg hcond]
      rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [flashBlock_map_sum (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
      scale causal (s0.pids 0 * 128) 64 c r d hc1 (fun p => pow2 (p.1 - Mr))]
  rfl

/-- The kernel's `Σ_jL nume[r,jL]·v[jL,d]` over the `Fin 64` masked block equals the
`flashBlock` list sum of `pow2(score − M')·v`. The `nume.to fp16 → real` round-trip
is identity in the model, and the masked (`⊥`) cells contribute `0`. -/
theorem flash_acc_dot_block (s0 : BlockState) (Q K V : RegionName) (scale : ℝ) (causal : Bool)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (r : Fin 128) (d : Fin 64)
    (qtile : Tile .fp16 [128, 64]) (ktile vtile : Tile .real [64, 64])
    (mnewT : Tile .real [128]) (numeT : Tile .real [128, 64])
    (hq : qtile = ⟨fun idx : TileIndex [128, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some (scale * qTile s0 Q 8192 64 128 idx))⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hv : ∀ idx : TileIndex [64, 64],
        vtile.data idx = some (s0.readMem V (s0.pids 1 * 8192 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)))
    (Mr : ℝ)
    (hmnew : mnewT.data (r, PUnit.unit) = (Mr : WithBot ℝ))
    (hnume : ∀ jL : Fin 64, numeT.data (r, jL, PUnit.unit)
        = WithBot.realExp2 (WithBot.realSub
            (flashQkCell causal (c * 64) (fun rr : Fin 128 => s0.pids 0 * 128 + rr.val) qtile ktile r jL)
            (mnewT.data (r, PUnit.unit)))) :
    (Tile.dot [] numeT vtile).data (r, d, PUnit.unit)
      = some ((flashBlock (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
          scale causal (s0.pids 0 * 128) 64 c r d).map (fun p => pow2 (p.1 - Mr) * p.2)).sum := by
  rw [Tile.dot_nil_data]
  -- per-lane: nume[r,jL]·v[jL,d] = some(masked pow2(score-Mr)·v)
  have hcell : ∀ jL : Fin 64,
      Option.map₂ (· * ·) (numeT.data (r, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (if (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val)
            then pow2 ((scale * Finset.univ.sum (fun e : Fin 64 =>
                  qTile s0 Q 8192 64 128 (r, e, PUnit.unit) *
                    kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, by
                      have := jL.isLt; omega⟩, e, PUnit.unit))) - Mr)
                * vTile s0 V 8192 64 128 (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)
            else 0) := by
    intro jL
    rw [hnume jL, hmnew, hv (jL, d, PUnit.unit)]
    unfold flashQkCell
    have hvval : s0.readMem V (s0.pids 1 * 8192 + (c * 64 + jL.val) * 64 + d.val * 1)
        = vTile s0 V 8192 64 128 (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit) := by
      simp only [vTile, flashBaseOffset]; ring_nf
    by_cases hcond : (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val)
    · rw [if_pos hcond, if_pos hcond,
        flash_dot_score_cell s0 Q K scale c r jL (by have := jL.isLt; omega) qtile ktile hq hk]
      simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
      rw [hvval]; refine congrArg some ?_; simp only [pow2]; ring_nf
    · rw [if_neg hcond, if_neg hcond]
      simp only [WithBot.realSub_bot_left, WithBot.realExp2_bot, Option.map₂, Option.bind, Option.map,
        zero_mul]
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (numeT.data (r, k, PUnit.unit)) (vtile.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ (fun jL =>
          (some (if (causal → c * 64 + jL.val ≤ s0.pids 0 * 128 + r.val)
            then pow2 ((scale * Finset.univ.sum (fun e : Fin 64 =>
                  qTile s0 Q 8192 64 128 (r, e, PUnit.unit) *
                    kTile s0 K 8192 64 128 (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mr)
                * vTile s0 V 8192 64 128 (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)
            else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hcell jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [flashBlock_map_sum (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
      scale causal (s0.pids 0 * 128) 64 c r d hc1 (fun p => pow2 (p.1 - Mr) * p.2)]

/-- The ⊥-seeded denominator after `c` blocks is anchored to the max-free batch
denominator: `l = κ(M_c)·L_c`. (Reads off `flashStateBot_snd_fst`.) -/
theorem flash_denom_anchor
    (qT : TileIndex [128, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin 128) (d : Fin 64) :
    (flashStateBot qT kT vT scale causal qStart hi i d).2.1
      = ((flashStateBot qT kT vT scale causal qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [flashStateBot_snd_fst, flashStateBot_fst_eq_runningMax, zero_add]

/-- The ⊥-seeded accumulator after `c` blocks: `acc = κ(M_c)·T_c`. -/
theorem flash_acc_anchor
    (qT : TileIndex [128, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin 128) (d : Fin 64) :
    (flashStateBot qT kT vT scale causal qStart hi i d).2.2
      = ((flashStateBot qT kT vT scale causal qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [flashStateBot_snd_snd, flashStateBot_fst_eq_runningMax, zero_add]

/-- If the ⊥-seeded running max over `[0, hi)` is `⊥`, the key list is empty, hence
its `pow2`-score sum (resp. `·v` sum) is `0`. -/
theorem flashKeysUpto_sum_zero_of_bot
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM)
    (hbot : flashRunningMax qT kT vT scale causal qStart hi i d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((flashKeysUpto qT kT vT scale causal qStart hi i d).map h).sum = 0 := by
  rw [show flashKeysUpto qT kT vT scale causal qStart hi i d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (flashKeysUpto qT kT vT scale causal qStart hi i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := flash_mem_le_foldr_sup _ _ hmem
  rw [← flashRunningMax, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

set_option maxHeartbeats 1000000 in
/-- **`denom = flashStateBot((c+1)·64).2.1`.** The kernel's `denom' = denom·α + Σnume`
register, with `denom = flashStateBot(c·64).2.1`, `α = realExp2(M_c − M_{c+1})`, and
`Σnume` the masked block sum, equals the ⊥-seeded denominator after `c+1` blocks
(`osStepBot_block_eq` over `flashBlock`, anchored via `flash_denom_anchor`). -/
theorem flash_denom_reg_eq (s0 : BlockState) (Q K V : RegionName) (scale : ℝ) (causal : Bool)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (r : Fin 128)
    (qtile : Tile .fp16 [128, 64]) (ktile : Tile .real [64, 64])
    (dtile mtile mnewT alphaT : Tile .real [128]) (numeT : Tile .real [128, 64])
    (hq : qtile = ⟨fun idx : TileIndex [128, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some (scale * qTile s0 Q 8192 64 128 idx))⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hdtile : dtile.data (r, PUnit.unit) = some
        ((flashStateBot (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
            scale causal (s0.pids 0 * 128) (c * 64) r ⟨0, by norm_num⟩).2.1))
    (hmtile : mtile.data (r, PUnit.unit)
        = flashRunningMax (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
            scale causal (s0.pids 0 * 128) (c * 64) r ⟨0, by norm_num⟩)
    (hmnew : mnewT.data (r, PUnit.unit)
        = flashRunningMax (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
            scale causal (s0.pids 0 * 128) ((c + 1) * 64) r ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT))
    (hnume : ∀ jL : Fin 64, numeT.data (r, jL, PUnit.unit)
        = WithBot.realExp2 (WithBot.realSub
            (flashQkCell causal (c * 64) (fun rr : Fin 128 => s0.pids 0 * 128 + rr.val) qtile ktile r jL)
            (mnewT.data (r, PUnit.unit)))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) dtile alphaT)
        (Tile.reduceSumDrop flashAx1 numeT)).data (r, PUnit.unit)
      = some ((flashStateBot (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
          scale causal (s0.pids 0 * 128) ((c + 1) * 64) r ⟨0, by norm_num⟩).2.1) := by
  set qT := qTile s0 Q 8192 64 128
  set kT := kTile s0 K 8192 64 128
  set vT := vTile s0 V 8192 64 128
  set qS := s0.pids 0 * 128
  -- abbreviations
  set m := flashStateBot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩ |>.1 with hm_def
  set Mc := flashRunningMax qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩ with hMc
  set Mc1 := flashRunningMax qT kT vT scale causal qS ((c + 1) * 64) r ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, flashStateBot_fst_eq_runningMax]
  -- Mc1 ≠ ⊥  (window through (c+1)·64 is nonempty)
  have hne : Mc1 ≠ ⊥ := flashRunningMax_ne_bot qT kT vT scale causal qS ((c + 1) * 64) r ⟨0, by norm_num⟩
    (by positivity) (by norm_num)
  obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
    cases hh : Mc1 with
    | bot => exact absurd hh hne
    | coe x => exact ⟨x, rfl⟩
  -- the running-max one-block advance: Mc1 = M' of osStepBot_block_eq.
  -- Derived from `flashStateBot_succ` + `osStepBot_block_fst` (no explicit window math).
  have hMsucc : Mc1 = m ⊔ ((flashBlock qT kT vT scale causal qS 64 c r ⟨0, by norm_num⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (flashStateBot qT kT vT scale causal qS ((c + 1) * 64) r ⟨0, by norm_num⟩).1 := by
      rw [hMc1, flashStateBot_fst_eq_runningMax]
    rw [h1, flashStateBot_succ,
      osStepBot_block_fst m
        ((flashStateBot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩).2.1)
        ((flashStateBot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩).2.2)]
  -- alpha value
  have halphaVal : alphaT.data (r, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]
    show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmnew,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  -- the reduceSum = block sum
  have hsum := flash_nume_row_sum s0 Q K V scale causal c hc1 r ⟨0, by norm_num⟩ qtile ktile mnewT numeT
    hq hk Mr (by rw [hmnew, hMr]) (by intro jL; rw [hnume jL])
  -- osStepBot_block_eq for the .2.1 channel
  have hblockEq := osStepBot_block_eq m
    ((flashStateBot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩).2.1)
    ((flashStateBot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩).2.2)
    ((flashKeysUpto qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((flashKeysUpto qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩).map (fun p => pow2 p.1)).sum
    (flashBlock qT kT vT scale causal qS 64 c r ⟨0, by norm_num⟩)
    (by rw [flash_denom_anchor, zero_add, hm_def])
    (by rw [flash_acc_anchor, zero_add, hm_def])
    (fun hbot => flashKeysUpto_sum_zero_of_bot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩
      (by rw [← flashStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => flashKeysUpto_sum_zero_of_bot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩
      (by rw [← flashStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  -- M' = Mc1
  rw [← hMsucc] at hblockEq
  -- final: target .2.1 via flashStateBot_succ
  rw [show (flashStateBot qT kT vT scale causal qS ((c + 1) * 64) r ⟨0, by norm_num⟩).2.1
        = (Mc1, (flashStateBot qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((flashBlock qT kT vT scale causal qS 64 c r ⟨0, by norm_num⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [flashStateBot_succ]; rw [← hblockEq]]
  -- now the kernel side
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hdtile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [hMr, WithBot.unbotD_coe]

set_option maxHeartbeats 1000000 in
/-- **`out_buffer = flashStateBot((c+1)·64).2.2`** (per channel `d`). The kernel's
`out_buffer' = out_buffer·out_scale + dot(nume, v)` register, with `out_buffer =
flashStateBot(c·64).2.2`, `out_scale = denom·0 + α = α`, and `dot(nume, v)` the
masked block PV sum, equals the ⊥-seeded accumulator after `c+1` blocks. -/
theorem flash_acc_reg_eq (s0 : BlockState) (Q K V : RegionName) (scale : ℝ) (causal : Bool)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (r : Fin 128) (d : Fin 64)
    (qtile : Tile .fp16 [128, 64]) (ktile vtile : Tile .real [64, 64])
    (obtile : Tile .real [128, 64]) (dtile mtile mnewT alphaT ostileT : Tile .real [128])
    (numeT : Tile .real [128, 64])
    (hq : qtile = ⟨fun idx : TileIndex [128, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some (scale * qTile s0 Q 8192 64 128 idx))⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hv : ∀ idx : TileIndex [64, 64],
        vtile.data idx = some (s0.readMem V (s0.pids 1 * 8192 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)))
    (hobtile : obtile.data (r, d, PUnit.unit) = some
        ((flashStateBot (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
            scale causal (s0.pids 0 * 128) (c * 64) r d).2.2))
    (rdenom : ℝ) (hdtile : dtile.data (r, PUnit.unit) = some rdenom)
    (hmtile : mtile.data (r, PUnit.unit)
        = flashRunningMax (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
            scale causal (s0.pids 0 * 128) (c * 64) r ⟨0, by norm_num⟩)
    (hmnew : mnewT.data (r, PUnit.unit)
        = flashRunningMax (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
            scale causal (s0.pids 0 * 128) ((c + 1) * 64) r ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT))
    (hostile : ostileT = Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul Broadcast.scalarR dtile (Tile.scalar (some 0))) alphaT)
    (hnume : ∀ jL : Fin 64, numeT.data (r, jL, PUnit.unit)
        = WithBot.realExp2 (WithBot.realSub
            (flashQkCell causal (c * 64) (fun rr : Fin 128 => s0.pids 0 * 128 + rr.val) qtile ktile r jL)
            (mnewT.data (r, PUnit.unit)))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) obtile
          (Tile.expandDim ⟨1, by simp⟩ ostileT))
        (Tile.dot [] numeT vtile)).data (r, d, PUnit.unit)
      = some ((flashStateBot (qTile s0 Q 8192 64 128) (kTile s0 K 8192 64 128) (vTile s0 V 8192 64 128)
          scale causal (s0.pids 0 * 128) ((c + 1) * 64) r d).2.2) := by
  set qT := qTile s0 Q 8192 64 128
  set kT := kTile s0 K 8192 64 128
  set vT := vTile s0 V 8192 64 128
  set qS := s0.pids 0 * 128
  set m := flashStateBot qT kT vT scale causal qS (c * 64) r d |>.1 with hm_def
  set Mc := flashRunningMax qT kT vT scale causal qS (c * 64) r ⟨0, by norm_num⟩ with hMc
  set Mc1 := flashRunningMax qT kT vT scale causal qS ((c + 1) * 64) r ⟨0, by norm_num⟩ with hMc1
  -- m (at channel d) = Mc (channel-independent running max)
  have hmMc : m = Mc := by
    rw [hm_def, hMc, flashStateBot_fst_eq_runningMax, flashRunningMax_eq qT kT vT scale causal qS (c * 64) r d ⟨0, by norm_num⟩]
  have hne : Mc1 ≠ ⊥ := flashRunningMax_ne_bot qT kT vT scale causal qS ((c + 1) * 64) r ⟨0, by norm_num⟩
    (by positivity) (by norm_num)
  obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
    cases hh : Mc1 with
    | bot => exact absurd hh hne
    | coe x => exact ⟨x, rfl⟩
  -- M' = Mc1 (one-block running-max advance, via osStepBot_block_fst + flashStateBot_succ)
  have hMsucc : Mc1 = m ⊔ ((flashBlock qT kT vT scale causal qS 64 c r d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (flashStateBot qT kT vT scale causal qS ((c + 1) * 64) r d).1 := by
      rw [hMc1, flashStateBot_fst_eq_runningMax, flashRunningMax_eq qT kT vT scale causal qS ((c+1)*64) r ⟨0, by norm_num⟩ d]
    rw [h1, flashStateBot_succ,
      osStepBot_block_fst m
        ((flashStateBot qT kT vT scale causal qS (c * 64) r d).2.1)
        ((flashStateBot qT kT vT scale causal qS (c * 64) r d).2.2)]
  -- ostile (out_scale) value = α = realExp2(realSub m Mc1)
  have hαsome' : WithBot.realExp2 (WithBot.realSub m Mc1)
      = some ((WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0) := by
    cases WithBot.realSub m Mc1 <;> rfl
  have halphaVal : alphaT.data (r, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]
    show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmnew,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hostileVal : ostileT.data (r, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [hostile, Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
    rw [halphaVal, hαsome', Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarR, Tile.scalar_data,
      NumericDType.mul, hdtile]
    simp only [WithBot.realMul, WithBot.realAdd, Option.map₂, Option.bind, Option.map, mul_zero,
      zero_add]
  -- the dot = block PV sum
  have hdot := flash_acc_dot_block s0 Q K V scale causal c hc1 r d qtile ktile vtile mnewT numeT
    hq hk hv Mr (by rw [hmnew, hMr]) (by intro jL; rw [hnume jL])
  -- osStepBot_block_eq for the .2.2 channel
  have hblockEq := osStepBot_block_eq m
    ((flashStateBot qT kT vT scale causal qS (c * 64) r d).2.1)
    ((flashStateBot qT kT vT scale causal qS (c * 64) r d).2.2)
    ((flashKeysUpto qT kT vT scale causal qS (c * 64) r d).map (fun p => pow2 p.1 * p.2)).sum
    ((flashKeysUpto qT kT vT scale causal qS (c * 64) r d).map (fun p => pow2 p.1)).sum
    (flashBlock qT kT vT scale causal qS 64 c r d)
    (by rw [flash_denom_anchor, zero_add, hm_def])
    (by rw [flash_acc_anchor, zero_add, hm_def])
    (fun hbot => flashKeysUpto_sum_zero_of_bot qT kT vT scale causal qS (c * 64) r d
      (by rw [← flashStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => flashKeysUpto_sum_zero_of_bot qT kT vT scale causal qS (c * 64) r d
      (by rw [← flashStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  -- target .2.2 via flashStateBot_succ
  rw [show (flashStateBot qT kT vT scale causal qS ((c + 1) * 64) r d).2.2
        = (Mc1, _,
            (flashStateBot qT kT vT scale causal qS (c * 64) r d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((flashBlock qT kT vT scale causal qS 64 c r d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [flashStateBot_succ]; rw [← hblockEq]]
  -- kernel side
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hobtile, hostileVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [hMr, WithBot.unbotD_coe]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Step lemma**: the loop body advances `attnInvariant` by one key block.
Combines the validated execution chain `flashLoopBody_steps` (exposing the symbolic
`max`/`denom`/`out_buffer` registers) with the channel bridges `flash_mnewT_eq`
(running max), `flash_denom_reg_eq` (denom), `flash_acc_reg_eq` (out_buffer), all
reading off `flashStateBot((c+1)·64)`/`flashRunningMax((c+1)·64)`. Block pointers
advance `i → i + 64`; the off_m/off_n/q registers are preserved. -/
theorem flash_attn_step (Q K V : RegionName) (s0 : BlockState) (sm_scale : ℝ) (IS_CAUSAL : Bool)
    (hDIM : 0 < 64) (i : Nat) (s : BlockState) (hiTotal : Nat)
    (hilt : i < hiTotal) (hhi : hiTotal ≤ 128) (hhimod : hiTotal % 64 = 0)
    (hinv : attnInvariant Q K V s0 sm_scale 8192 128 128 64 64 hiTotal IS_CAUSAL hDIM i s) :
    ∃ s', stepStmts (flashLoopBody IS_CAUSAL) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ attnInvariant Q K V s0 sm_scale 8192 128 128 64 64 hiTotal IS_CAUSAL hDIM (i + 64) s' := by
  simp only [attnInvariant, log2e] at hinv
  obtain ⟨hpids, hmod, hile, hmax, hden, hob, hq, hom, hon, hKp, hVp, hQp, hundef, hmem⟩ := hinv
  set c := i / 64 with hc_def
  have hi : i = c * 64 := by omega
  have hc1 : (c + 1) * 64 ≤ 128 := by omega
  set base := flashBaseOffset s0 8192 with hbase
  set qS := s0.pids 0 * 128 with hqS
  set scale := sm_scale * log2e with hscale
  set qT := qTile s0 Q 8192 64 128 with hqT
  set kT := kTile s0 K 8192 64 128 with hkT
  set vT := vTile s0 V 8192 64 128 with hvT
  -- base offset is `pids 1 * 8192`
  have hbaseEq : base = s0.pids 1 * 8192 := by simp [hbase, flashBaseOffset]
  -- run the execution chain
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hqF, hQpF, homF, honF, hKpF, hVpF,
      qkT, rmaxT, mnewT, alphaT, numeT, ostileT,
      hqkData, hrm, hmnewd, halphad, hnumed, hostiled, hmaxF, hdenF, hobF⟩ :=
    flashLoopBody_steps IS_CAUSAL (s.setReg "start_n" .nat [] (Tile.scalar i)) i
      (fun r : Fin 128 => qS + r.val) K V base base i i
      (⟨fun idx : TileIndex [128, 64] =>
          FloatDType.real.cast FloatDType.fp16 (some (scale * qT idx))⟩)
      (⟨fun r : TileIndex [128] => flashRunningMax qT kT vT (sm_scale * log2e) IS_CAUSAL qS i r.1 ⟨0, hDIM⟩⟩)
      (⟨fun r : TileIndex [128] => ((flashStateBot qT kT vT (sm_scale * log2e) IS_CAUSAL qS i r.1 ⟨0, hDIM⟩).2.1 : ℝ)⟩)
      (⟨fun idx : TileIndex [128, 64] => ((flashStateBot qT kT vT (sm_scale * log2e) IS_CAUSAL qS i idx.1 idx.2.1).2.2 : ℝ)⟩)
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩)
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩)
      (by simp [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hon)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmax)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hden)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hob)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; rw [hKp])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; rw [hVp])
      (fun idx => rfl)
      (fun idx => rfl)
      _
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hQp)
      (by intro rg o; rw [BlockState.setReg_undef]; exact hundef rg o)
  refine ⟨sF, hchain, ?_⟩
  -- the chain's k/v load tiles read s0's memory (sin.mem = s.mem = s0.mem)
  have hrmemK : ∀ idx : TileIndex [64, 64],
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩ : Tile .real [64, 64]).data idx
        = some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)) := by
    intro idx
    simp only [BlockState.setReg_readMem]
    rw [show (s.readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))
          = s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64) from by
      unfold BlockState.readMem; rw [hmem, hbaseEq, hi]]
  have hrmemV : ∀ idx : TileIndex [64, 64],
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩ : Tile .real [64, 64]).data idx
        = some (s0.readMem V (s0.pids 1 * 8192 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)) := by
    intro idx
    simp only [BlockState.setReg_readMem]
    rw [show (s.readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))
          = s0.readMem V (s0.pids 1 * 8192 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1) from by
      unfold BlockState.readMem; rw [hmem, hbaseEq, hi]]
  -- q tile readback fact (cast fp16 of scale·qT)
  set qtileF : Tile .fp16 [128, 64] :=
    ⟨fun idx : TileIndex [128, 64] => FloatDType.real.cast FloatDType.fp16 (some (scale * qT idx))⟩ with hqtileF
  set ktileF : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩ with hktileF
  set vtileF : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩ with hvtileF
  -- mnewT cell value = flashRunningMax((c+1)·64)
  have hmnewcell : ∀ r : Fin 128, mnewT.data (r, PUnit.unit)
      = flashRunningMax qT kT vT scale IS_CAUSAL qS ((c + 1) * 64) r ⟨0, hDIM⟩ := by
    intro r
    rw [hmnewd]
    refine flash_mnewT_eq s0 Q K scale IS_CAUSAL vT c hc1 r qtileF ktileF _ rmaxT hqtileF hrmemK ?_ ?_
    · rw [hi]
    · refine flash_reduceMaxDrop_row qkT rmaxT hrm r
        (fun jL => flashQkCell IS_CAUSAL (c * 64) (fun rr : Fin 128 => qS + rr.val) qtileF ktileF r jL) ?_
      intro jL
      rw [hqkData r jL, hi]
  -- nume cell readback
  have hnumecell : ∀ r : Fin 128, ∀ jL : Fin 64, numeT.data (r, jL, PUnit.unit)
      = WithBot.realExp2 (WithBot.realSub
          (flashQkCell IS_CAUSAL (c * 64) (fun rr : Fin 128 => qS + rr.val) qtileF ktileF r jL)
          (mnewT.data (r, PUnit.unit))) := by
    intro r jL
    rw [hnumed]
    show WithBot.realExp2 _ = _
    simp only [Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, NumericDType.sub, hqkData r jL, hi]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, BlockState.setReg_pids, hpids]
  · omega
  · omega
  · -- max = flashRunningMax((i+64))
    rw [hmaxF]; refine congrArg some ?_; ext r
    rw [hmnewcell r.1]; congr 1; omega
  · -- denom = flashStateBot((i+64)).2.1
    rw [hdenF]; refine congrArg some ?_; ext r
    rw [show ((i + 64) : Nat) = (c + 1) * 64 from by omega]
    refine flash_denom_reg_eq s0 Q K V scale IS_CAUSAL c hc1 r.1 qtileF ktileF
      (⟨fun rr : TileIndex [128] => ((flashStateBot qT kT vT scale IS_CAUSAL qS i rr.1 ⟨0, hDIM⟩).2.1 : ℝ)⟩)
      (⟨fun rr : TileIndex [128] => flashRunningMax qT kT vT scale IS_CAUSAL qS i rr.1 ⟨0, hDIM⟩⟩)
      mnewT alphaT numeT hqtileF hrmemK ?_ ?_ ?_ halphad ?_
    · show some _ = some _; rw [hi]
    · show flashRunningMax _ _ _ _ _ _ _ _ _ = _; rw [hi]
    · rw [hmnewcell r.1]
    · intro jL; rw [hnumecell r.1 jL]
  · -- out_buffer = flashStateBot((i+64)).2.2
    rw [hobF]; refine congrArg some ?_; ext idx
    rw [show ((i + 64) : Nat) = (c + 1) * 64 from by omega]
    refine flash_acc_reg_eq s0 Q K V scale IS_CAUSAL c hc1 idx.1 idx.2.1 qtileF ktileF vtileF
      (⟨fun ii : TileIndex [128, 64] => ((flashStateBot qT kT vT scale IS_CAUSAL qS i ii.1 ii.2.1).2.2 : ℝ)⟩)
      (⟨fun rr : TileIndex [128] => ((flashStateBot qT kT vT scale IS_CAUSAL qS i rr.1 ⟨0, hDIM⟩).2.1 : ℝ)⟩)
      (⟨fun rr : TileIndex [128] => flashRunningMax qT kT vT scale IS_CAUSAL qS i rr.1 ⟨0, hDIM⟩⟩)
      mnewT alphaT ostileT numeT hqtileF hrmemK hrmemV ?_
      ((flashStateBot qT kT vT scale IS_CAUSAL qS i idx.1 ⟨0, hDIM⟩).2.1) ?_ ?_ ?_ halphad hostiled ?_
    · show some _ = some _; rw [hi]
    · show some _ = some _; rw [hi]
    · show flashRunningMax _ _ _ _ _ _ _ _ _ = _; rw [hi]
    · rw [hmnewcell idx.1]
    · intro jL; rw [hnumecell idx.1 jL]
  · rw [hqF]
  · rw [homF]
  · rw [honF]
  · -- K_block_ptr advanced (kcol = i, so kcol + 64 = i + 64)
    rw [hKpF]
  · -- V_block_ptr advanced
    rw [hVpF]
  · -- Q_block_ptr preserved (not touched by the loop body)
    rw [hQpF]
  · exact hundefF
  · rw [hmemF]; funext region offset
    rw [BlockState.setReg_mem]
    have : s.mem = s0.mem := hmem
    rw [this]

/-- The 5 lowered post-loop statements (17–21): `out_buffer /= denom[:, None]`,
the `l_ptr`/`L`-store of `max + log2 denom`, and the `O_block_ptr`/`O`-store of
`out_buffer.to fp16`. -/
def flashPostLoop (L O : RegionName) : List Stmt :=
  [ Stmt.assign .real [128, 64] "out_buffer"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "out_buffer")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "denom"))),
    Stmt.assign .ptr [128] "l_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase L)
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_bs_head") (Op.constNat 128))
          (Op.ref .nat [128] "off_m"))),
    Stmt.store .real [128] (MemAccess.ptr (Op.ref .ptr [128] "l_ptr"))
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [128] "max")
        (Op.log2 (Op.ref .real [128] "denom"))) MaskOpt.none,
    Stmt.assign .blockPtr [128, 64] "O_block_ptr"
      (Op.makeBlockPtrDynOffsets O (Op.ref .nat [] "qkv_base_offset") [128, 64] [128, 64] [64, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128), Op.constNat 0]),
    Stmt.store .fp16 [128, 64] (MemAccess.blockPtr (Op.ref .blockPtr [128, 64] "O_block_ptr") [])
      (Op.castFloat .real .fp16 (Op.ref .real [128, 64] "out_buffer")) MaskOpt.none ]

/-- The lowered body `drop 17` is exactly `flashPostLoop`. -/
theorem flashPostLoop_check (Q K V L O : RegionName) (sm_scale : ℝ) (IS_CAUSAL : Bool) :
    (flash_attn_fwd_kernel_surface Q K V L O sm_scale
        16384 8192 64 1 16384 8192 64 1 16384 8192 64 1 16384 8192 64 1
        2 2 128 128 64 64 IS_CAUSAL).toAlgKernel.body.drop 17
      = flashPostLoop L O := by
  cases IS_CAUSAL <;> rfl

end VeriTile.Bench.TritonBenchG.FlashAttn
