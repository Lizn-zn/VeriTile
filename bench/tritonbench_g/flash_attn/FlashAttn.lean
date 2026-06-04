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

end VeriTile.Bench.TritonBenchG.FlashAttn
