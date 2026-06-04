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

The **remaining stage** (not yet closed) is the per-statement loop-body op-eval
recipes (including threading the per-element causal `tl.where(off_m ≥ start_n+off_n,
qk, -inf)` mask into the `osBlockStep`/`attnKeyListCausal` fold) and the
`attnInvariant`/`preLoop`/`attn_step`/`attn_postLoop` invariant skeleton over the
14-register `out_buffer` kernel with BOTH the `O` and `L` stores, composed via
`forRangeDyn_inv` — i.e. proving `out_buffer / denom` at the final store equals
`flashAttnOValueSpec{,Causal}`. That mirrors
`VeriTile/Examples/AttentionForwardClosedForm.lean`'s preLoop/step/postLoop
skeleton, now retargeted onto the block-pointer foundation above. No
self-referential / tautological summary is asserted in its place.

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

/-- Base-2 logarithm of `e` (`log2(e) = 1 / log 2`), the constant `qk_scale` folds
in (`q = (q · sm_scale · 1.44269504).to(fp16)`). -/
noncomputable def log2e : ℝ := 1 / Real.log 2

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

end VeriTile.Bench.TritonBenchG.FlashAttn
