import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant
import VeriTile.Triton.Semantics.BlockPtrEval

/-!
# `attention_kernel` — strict per-kernel correctness

`_fwd_kernel_aligned` is a flash-attention forward kernel with a fused
relative-position bias `B0` (`rel_h + rel_w`). Program `(start_m, off_hz)` loads
a `BLOCK_M`-row `Q` tile for one (batch, head), scales it by
`qk_scale = sm_scale · log2(e)`, then over the key/value context (stepping by
`BLOCK_N`) runs the online-softmax recurrence — block scores
`qk = dot(q, k) + (b0 + b1)·log2(e)`, running max `m_i`, denominator `l_i`,
accumulator `acc` with `exp2(qk - m_i_new)` weights — and finally stores
`acc / l_i` to `Out`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_aligned[grid](...)`, the grid over
`(cdiv(N_CTX, BLOCK_M), Z·H)`, block scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `start_m`/`off_hz` are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
attention_kernel_python_test_shape_output_summary            ← TOP THEOREM
  ├─ attention_kernel_fwd_kernel_aligned_surface_toAlgorithm_supported   surface lowers to algorithm layer
  └─ attention_kernel_fwd_kernel_aligned_python_test_shape_compute_correct
       └─ attention_kernel_fwd_kernel_aligned_surface_compute_correct
            └─ (full surface produces producedOutputValue at the Out store)

attention_kernel_final_store_python_test_shape_compute_correct
  └─ attention_kernel_final_store_slice_compute_correct       ← ComputeCorrect over the final Out store
       └─ attention_kernel_final_store_slice_correct          ← algorithm-layer readback (normalizedAccValue)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `OUT_DTYPE`
(`fp16`/`bf16`) casts collapse to the identity post-erasure; `@triton.autotune`
/ `num_warps`/`num_stages` are not modeled. The output summary is stated at the
Python test shape (`B=2, H=4, N_CTX=128, D_MODEL=128, BLOCK_M=BLOCK_N=64`,
`sm_scale=0.1`, `P_SEQ=0`, `fp16`, contiguous per-head strides `(16384,128,1)`).
The surface theorem captures the full single-program bias-augmented
online-softmax body via `producedOutputValue`; the `final_store` lemmas isolate
the final `acc / l_i` store (`normalizedAccValue`). This is a single-program
scope (the store is unmasked at this shape since `N_CTX` is a multiple of
`BLOCK_M`); cross-program composition into the full output is the trusted host
boundary.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful DSL port of `attention_kernel.py`'s `_fwd_kernel_aligned`. -/
def attention_kernel_fwd_kernel_aligned_surface
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(OUT_DTYPE)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=OUT_DTYPE)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    qk += ((b0 + b1) * 1.44269504)

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(OUT_DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(OUT_DTYPE))
}

/-- The full aligned attention-kernel surface lowers to the algorithm layer. -/
theorem attention_kernel_fwd_kernel_aligned_surface_toAlgorithm_supported
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) :
    ∃ alg, (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
      stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
      stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
      BLOCK_M BLOCK_N out_dtype).toAlgorithm? = Except.ok alg := by
  simp [attention_kernel_fwd_kernel_aligned_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Final output-store slice of `attention_kernel.py`'s `_fwd_kernel_aligned`.

This slice includes the Python `acc = acc / l_i[:, None]` statement immediately
before the unmasked block writeback into `Out`. Producing the unnormalized
streaming-softmax `Acc` and `L` inputs remains the narrower recurrence
obligation. -/
def attention_kernel_final_store_slice
    (Acc L Out : RegionName)
    (stride_acc_h stride_acc_m stride_acc_k
      stride_l_h stride_l_m
      stride_oh stride_om stride_on
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(Acc + off_hz * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k))
  l_i = tl.load(L + off_hz * $(stride_l_h) + offs_m * $(stride_l_m))
  acc = acc / l_i[:, None]
  tl.store(Out + off_hz * $(stride_oh) +
      offs_m[:, None] * $(stride_om) + offs_k[None, :] * $(stride_on),
      (acc).to(Out.dtype.element_ty))
}

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def accOffset
    (s : BlockState)
    (stride_acc_h stride_acc_m stride_acc_k BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 1 * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + kIndex idx * stride_acc_k

def lOffset
    (s : BlockState)
    (stride_l_h stride_l_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_l_h + mIndex s BLOCK_M i * stride_l_m

noncomputable def normalizedAccValue
    (s : BlockState) (Acc L : RegionName)
    (stride_acc_h stride_acc_m stride_acc_k stride_l_h stride_l_m
      BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  s.readMem Acc
      (accOffset s stride_acc_h stride_acc_m stride_acc_k BLOCK_M idx) /
    s.readMem L (lOffset s stride_l_h stride_l_m BLOCK_M idx.1)

def outOffset
    (s : BlockState)
    (stride_oh stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 1 * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + kIndex idx * stride_on

def surfaceOutOffset
    (s : BlockState)
    (stride_qh stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 1 * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + kIndex idx * stride_on

noncomputable def producedOutputValue
    (s : BlockState) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  match exec (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out
      sm_scale stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
      stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
      BLOCK_M BLOCK_N out_dtype) s with
  | some s' => s'.readMem Out (surfaceOutOffset s stride_qh stride_om stride_on BLOCK_M idx)
  | none => 0.0

/-- Algorithm-layer correctness for the final output store. -/
theorem attention_kernel_final_store_slice_correct
    (Acc L Out : RegionName)
    (stride_acc_h stride_acc_m stride_acc_k
      stride_l_h stride_l_m
      stride_oh stride_om stride_on
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s stride_oh stride_om stride_on BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s stride_oh stride_om stride_on BLOCK_M idx
      (exec (attention_kernel_final_store_slice Acc L Out stride_acc_h
            stride_acc_m stride_acc_k stride_l_h stride_l_m stride_oh
            stride_om stride_on BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (normalizedAccValue s Acc L stride_acc_h stride_acc_m
            stride_acc_k stride_l_h stride_l_m BLOCK_M idx) := by
  intro idx
  simp [exec, attention_kernel_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.div,
        mIndex, kIndex, accOffset, lOffset, normalizedAccValue, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 1 * stride_oh + (s.pids 0 * BLOCK_M + idx.1.val) * stride_om +
        idx.2.1.val * stride_on
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      s.readMem Acc
        (s.pids 1 * stride_acc_h +
          (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
          idx.2.1.val * stride_acc_k) /
        s.readMem L
          (s.pids 1 * stride_l_h +
            (s.pids 0 * BLOCK_M + idx.1.val) * stride_l_m)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, mIndex, kIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem Out (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    normalizedAccValue s Acc L stride_acc_h stride_acc_m stride_acc_k
      stride_l_h stride_l_m BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, accOffset, lOffset, normalizedAccValue, mIndex, kIndex]

/-- Compute-facing correctness for the final output store. -/
theorem attention_kernel_final_store_slice_compute_correct
    (Acc L Out : RegionName)
    (stride_acc_h stride_acc_m stride_acc_k
      stride_l_h stride_l_m
      stride_oh stride_om stride_on
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s stride_oh stride_om stride_on BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_final_store_slice Acc L Out stride_acc_h
        stride_acc_m stride_acc_k stride_l_h stride_l_m stride_oh stride_om
        stride_on BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (Out, outOffset s stride_oh stride_om stride_on BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        normalizedAccValue s Acc L stride_acc_h stride_acc_m stride_acc_k
          stride_l_h stride_l_m BLOCK_M idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_kernel_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_kernel_final_store_slice_correct Acc L Out stride_acc_h
    stride_acc_m stride_acc_k stride_l_h stride_l_m stride_oh stride_om stride_on BLOCK_M
    BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem attention_kernel_fwd_kernel_aligned_surface_compute_correct
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out
        sm_scale stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
        stride_vh stride_vk stride_vn stride_oh stride_om stride_on
        stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
        BLOCK_DMODEL BLOCK_M BLOCK_N out_dtype)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (Out, surfaceOutOffset s stride_qh stride_om stride_on BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        producedOutputValue s Q K V B0 Out sm_scale stride_qh stride_qm
          stride_qk stride_kh stride_kn stride_kk stride_vh stride_vk
          stride_vn stride_oh stride_om stride_on stride_b0h stride_b0m
          Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M
          BLOCK_N out_dtype idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_kernel_fwd_kernel_aligned_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedOutputValue, hExec]

/-! ## Python test-shape wrapper

`attention_kernel.py`'s checked test uses `B = 2`, `H = 4`, `N_CTX = 128`,
`D_MODEL = 128`, `BLOCK_M = 64`, and `BLOCK_N = 64`. Contiguous
`[B, H, N_CTX, D_MODEL]` tensors are passed to the kernel with per-head strides
`(16384, 128, 1)`. -/

theorem attention_kernel_final_store_python_test_shape_compute_correct
    (Acc L Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_final_store_slice Acc L Out
        16384 128 1 128 1 16384 128 1 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [64, 128] =>
        some (Out, outOffset s 16384 128 1 64 idx))
      (expected := fun idx : TileIndex [64, 128] =>
        normalizedAccValue s Acc L 16384 128 1 128 1 64 idx) := by
  apply attention_kernel_final_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attention_kernel_fwd_kernel_aligned_python_test_shape_compute_correct
    (Q K V B0 Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out
        0.1 16384 128 1 16384 128 1 16384 128 1 16384 128 1
        16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 128] =>
        some (Out, surfaceOutOffset s 16384 128 1 64 idx))
      (expected := fun idx : TileIndex [64, 128] =>
        producedOutputValue s Q K V B0 Out 0.1 16384 128 1 16384 128 1
          16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64
          64 FloatDType.fp16 idx) := by
  exact attention_kernel_fwd_kernel_aligned_surface_compute_correct
    Q K V B0 Out 0.1 16384 128 1 16384 128 1 16384 128 1
    16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
    FloatDType.fp16 s

/-- Public Python test-shape summary for `attention_kernel.py`.

This end-to-end summary records the faithful aligned attention surface for the
checked relative-position-bias launch and ties the Q/K/V streaming-softmax
producer path directly to the observable final `Out` writeback. -/
theorem attention_kernel_python_test_shape_output_summary
    (Q K V B0 Out : RegionName) (s : BlockState) :
    (∃ alg, (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out
      0.1 16384 128 1 16384 128 1 16384 128 1 16384 128 1
      16384 128 2 4 128 0 64 128 128 64 64
      FloatDType.fp16).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out
        0.1 16384 128 1 16384 128 1 16384 128 1 16384 128 1
        16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 128] =>
        some (Out, surfaceOutOffset s 16384 128 1 64 idx))
      (expected := fun idx : TileIndex [64, 128] =>
        producedOutputValue s Q K V B0 Out 0.1 16384 128 1 16384 128 1
          16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64
          64 FloatDType.fp16 idx) := by
  constructor
  · exact attention_kernel_fwd_kernel_aligned_surface_toAlgorithm_supported
      Q K V B0 Out 0.1 16384 128 1 16384 128 1 16384 128 1
      16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
      FloatDType.fp16
  · exact attention_kernel_fwd_kernel_aligned_python_test_shape_compute_correct
      Q K V B0 Out s

/-! ## Genuine closed-form correctness — reusable foundation

The theorems above state correctness against `producedOutputValue`, which is the
kernel's *own executed* `Out` readback (a self-referential summary). The section
below builds the foundation for replacing that with a **genuine** closed-form
claim: the kernel computes the base-2 streaming softmax `attnGenScore fscore V`
of the loaded Q/K/V tiles, with the kernel's actual per-key score `fscore`
(scaled dot plus the additive relative-position bias `(b0 + b1)·log2 e`).

The pure-math heart (`VeriTile.Triton.attnGenScore`, `closed_form_g`,
`attnGenScore_eq_streaming` in `Math/Attention.lean`) and the block-pointer
`evalOp` reduction lemmas (`Semantics/BlockPtrEval.lean`) are complete and
sorry-free; this section supplies the kernel-specific `evalOp` reductions
(`makeBlockPtrDyn`/`makeBlockPtr`-with-dynamic-row-offset and the K/V/Q
block-pointer loads at their resolved contiguous addresses) plus the genuine
score/tile specification `fscore`/`qRaw`/`kFlat`/`vFlat`/`b0Val`/`b1Val`. The
remaining `exec`-side assembly (preLoop base case, the 15-statement loop-body
invariant step over `forRangeDyn`, the `acc /= l_i` + block-pointer-store
epilogue, and the `ComputeCorrect.Realizes` bench bridge) mirrors
`VeriTile.Examples.AttentionForwardClosedForm`, swapping in the generalized
`mPg`/`lPg`/`oPg` invariant, the `forRangeDyn` loop driver, and these block-ptr
lemmas. -/

namespace ClosedForm

open VeriTile.Triton

/-- **`makeBlockPtrDyn` eval** (the `K`/`V` block pointers — static offsets,
dynamic base): evaluates the base-offset op and packages the constant
`BlockPtr` tile. -/
theorem makeBlockPtrDyn_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape)
    (strides offsets : List Nat) (s : BlockState) (base : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base)) :
    evalOp (.makeBlockPtrDyn region baseOffset parentShape blockShape strides offsets) s
      = some (⟨fun _ : TileIndex blockShape =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := offsets }⟩) := by
  simp only [evalOp, hb, Option.bind]
  rfl

/-- **`makeBlockPtrDynOffsets` (dynamic row offset, literal `0` column) eval**
(the `Q`/`O` block pointers): packages the constant `BlockPtr` tile with the
resolved row offset. -/
theorem makeBlockPtr_rowcol_eval (region : RegionName) (baseOffset : Op .nat [])
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

/-- **`tl.advance [0, d]` of a `[rowOff=0, colOff]` block pointer** (the `K`
pointer step): advances the column offset by `d`. -/
theorem advance_col_eval (s : BlockState) (region : RegionName)
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

/-- **`tl.advance [d, 0]` of a `[rowOff, colOff=0]` block pointer** (the `V`
pointer step): advances the row offset by `d`. -/
theorem advance_row_eval (s : BlockState) (region : RegionName)
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

/-! ### Genuine kernel specification

The loaded Q/K/V/B0 tiles read from the *input* state `s0`, and the kernel's
genuine per-key score `fscore` (used as the `score` argument of the generalized
streaming-softmax spec `VeriTile.Triton.attnGenScore`). -/

/-- Loaded (pre-scale) Q tile: row `r`, head lane `e`. -/
noncomputable def qRaw (s0 : BlockState) (Q : RegionName)
    (q_offset BLOCK_M HEAD_DIM : Nat) (start_m : Nat) :
    TileIndex [BLOCK_M, HEAD_DIM] → ℝ :=
  fun (r, e, _) => s0.readMem Q (q_offset + (start_m * BLOCK_M + r.val) * HEAD_DIM + e.val)

/-- Loaded K tile as a flat per-key function over `[HEAD_DIM, N_CTX]`. -/
noncomputable def kFlat (s0 : BlockState) (K : RegionName)
    (kv_offset HEAD_DIM N_CTX : Nat) :
    Fin HEAD_DIM → Fin N_CTX → ℝ :=
  fun e j => s0.readMem K (kv_offset + e.val + j.val * HEAD_DIM)

/-- Loaded V tile, flat per-key over `[N_CTX, HEAD_DIM]`. -/
noncomputable def vFlat (s0 : BlockState) (V : RegionName)
    (kv_offset HEAD_DIM N_CTX : Nat) :
    TileIndex [N_CTX, HEAD_DIM] → ℝ :=
  fun (j, d, _) => s0.readMem V (kv_offset + j.val * HEAD_DIM + d.val)

/-- Per-row, per-block-column bias `b0` read (`c = j / BLOCK_N`). -/
noncomputable def b0Val (s0 : BlockState) (B0 : RegionName)
    (b_offset BLOCK_M stride_b0m : Nat) (start_m : Nat)
    (r : Fin BLOCK_M) (c : Nat) : ℝ :=
  s0.readMem B0 (b_offset + (start_m * BLOCK_M + r.val) * stride_b0m + c)

/-- Per-row, per-lane bias `b1` read at lane `jL` (`jL = j % BLOCK_N`). -/
noncomputable def b1Val (s0 : BlockState) (B0 : RegionName)
    (b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE : Nat) (start_m : Nat)
    (r : Fin BLOCK_M) (jL : Nat) : ℝ :=
  s0.readMem B0
    (b_offset + (start_m * BLOCK_M + r.val) * stride_b0m + (jL % BIAS_LAST_SIZE + BIAS_LAST_SIZE))

/-- **Genuine per-key score** `fscore r j` of `_fwd_kernel_aligned`:
`qk_scale·(Σ_e Q[r,e]·K[e,j]) + (b0[r, j/BN] + b1[r, j%BN])·log2 e`,
with `qk_scale = sm_scale · log2 e` already folded into the pre-scaled `q`. This
is the `score` argument of `VeriTile.Triton.attnGenScore`, whose batch base-2
softmax `(Σ 2^fscore · V) / (Σ 2^fscore)` is the kernel's closed form (see
`closed_form_g`). -/
noncomputable def fscore (s0 : BlockState) (Q K B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m : Nat)
    (start_m : Nat)
    (r : Fin BLOCK_M) (j : Fin N_CTX) : ℝ :=
  (sm_scale * 1.44269504) *
      Finset.univ.sum (fun e : Fin HEAD_DIM =>
        qRaw s0 Q q_offset BLOCK_M HEAD_DIM start_m (r, e, PUnit.unit)
          * kFlat s0 K kv_offset HEAD_DIM N_CTX e j)
    + (b0Val s0 B0 b_offset BLOCK_M stride_b0m start_m r (j.val / BLOCK_N)
        + b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE start_m r (j.val % BLOCK_N))
      * 1.44269504

/-- **Closed-form target.** The kernel's genuine specification is
`attnGenScore (fscore …) (vFlat …)`: the base-2 streaming softmax over the
genuine per-key score `fscore` (scaled dot + additive bias). The generalized
math lemma `closed_form_g` establishes that the running `oPg / lPg` recurrence
the kernel's loop maintains converges to exactly this value after all
`numKVBlocks` key blocks — which is what the remaining `exec`-side assembly
discharges. -/
noncomputable def attentionKernelSpec (s0 : BlockState) (Q K V B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m : Nat)
    (start_m : Nat) : TileIndex [BLOCK_M, HEAD_DIM] → ℝ :=
  attnGenScore
    (fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m start_m)
    (vFlat s0 V kv_offset HEAD_DIM N_CTX)

/-- **Closed-form readout.** After all `numKVBlocks` key blocks the kernel's
running `oPg / lPg` ratio (which the loop invariant maintains) is exactly the
genuine spec `attentionKernelSpec` (= `attnGenScore fscore vFlat`). Direct
corollary of the generalized math lemma `closed_form_g`. -/
theorem attentionKernelSpec_eq_ratio (s0 : BlockState) (Q K V B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM BIAS_LAST_SIZE stride_b0m nB : Nat)
    (start_m : Nat) (hBN : 0 < BLOCK_N) (hnB : 1 ≤ nB)
    (i : Fin BLOCK_M) (d : Fin HEAD_DIM) :
    let score := fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m start_m
    let V' := vFlat s0 V kv_offset HEAD_DIM (BLOCK_N * nB)
    oPg score V' i d nB / lPg score i nB
      = attentionKernelSpec s0 Q K V B0 sm_scale q_offset kv_offset b_offset
          BLOCK_M BLOCK_N HEAD_DIM (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m start_m (i, d, PUnit.unit) :=
  closed_form_g _ _ hBN hnB i d

/-- **Kernel-faithful running denominator.** Same recurrence as the math `lPg`
but seeded `0` (the kernel inits `l_i = 0`, whereas the generalized `lPg` seeds
`1`). This is the value the loop invariant must carry in the `l_i` register. -/
noncomputable def lPgK {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        alphaPg score i c * lPgK score i c +
          Finset.univ.sum (fun a : Fin BN =>
            pow2 (score i (gkey BN nB c (by omega) a) - mRg score i (c + 1)))
      else lPgK score i c

/-- The seed difference is killed by `alphaPg 0 = 0`: for `1 ≤ c ≤ nB` the
kernel's seed-`0` denominator agrees with the math `lPg` (seed `1`). So the
final `c = nB ≥ 1` readout is unaffected by the kernel's `l_i = 0` initialization,
and `closed_form_g` / `attentionKernelSpec_eq_ratio` apply unchanged. -/
theorem lPgK_eq_lPg {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) :
    ∀ c, 1 ≤ c → c ≤ nB → lPgK score i c = lPg score i c := by
  intro c hc1 hc
  induction c, hc1 using Nat.le_induction with
  | base => simp only [lPgK, lPg, dif_pos hc, alphaPg_zero score i, zero_mul]
  | succ c hc1 ih =>
    have ihc := ih (by omega)
    simp only [lPgK, lPg, dif_pos hc, ihc]

/-! ### Loop invariant skeleton

The `forRangeDyn` loop-invariant `Prop` carried across key blocks. The counter `i`
ranges over `0, BLOCK_N, 2·BLOCK_N, …` (`c = i / BLOCK_N` blocks consumed); the
streaming registers `m_i`/`l_i`/`acc` equal the generalized partials
`mPg`/`lPgK`/`oPg` of the genuine score `fscore` and value tile `vFlat` over the
first `c` blocks (kernel seed `l_i = 0`, hence `lPgK` not `lPg`), and the K/V
block pointers have advanced `c` column/row steps of `BLOCK_N`. The loop-invariant
registers (`q` pre-scaled+loaded, `b1`, `b_offset`, `start_m`/`off_hz`) are fixed.
`preLoop` establishes `attnKernelInvariant … 0`, `attn_step` advances it by one
block, and `attn_postLoop` reads the closed form off `… numKVBlocks` via
`attentionKernelSpec_eq_ratio`. -/
noncomputable def attnKernelInvariant
    (s0 : BlockState) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM P_SEQ BIAS_LAST_SIZE stride_b0m stride_kk stride_kn
      stride_vk stride_vn numKVBlocks : Nat)
    (start_m : Nat) (i : Nat) (s : BlockState) : Prop :=
  let nB := numKVBlocks; let c := i / BLOCK_N; let N_CTX := BLOCK_N * numKVBlocks
  let score := fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
    BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m start_m
  let V' := vFlat s0 V kv_offset HEAD_DIM N_CTX
  s.pids = s0.pids ∧ i = c * BLOCK_N ∧ c ≤ nB ∧
  (s.regs .real [BLOCK_M] "m_i" = some ⟨fun r : TileIndex [BLOCK_M] => mPg BLOCK_N nB score r.1 c⟩) ∧
  (s.regs .real [BLOCK_M] "l_i" = some ⟨fun r : TileIndex [BLOCK_M] => ((lPgK score r.1 c : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, HEAD_DIM] "acc" = some ⟨fun idx : TileIndex [BLOCK_M, HEAD_DIM] =>
        ((oPg score V' idx.1 idx.2.1 c : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .blockPtr [HEAD_DIM, BLOCK_N] "K_block_ptr" = some
      (⟨fun _ : TileIndex [HEAD_DIM, BLOCK_N] =>
        { region := K, baseOffset := kv_offset,
          parentShape := [HEAD_DIM, N_CTX + P_SEQ], blockShape := [HEAD_DIM, BLOCK_N],
          strides := [stride_kk, stride_kn], offsets := [0, c * BLOCK_N] }⟩)) ∧
  (s.regs .blockPtr [BLOCK_N, HEAD_DIM] "V_block_ptr" = some
      (⟨fun _ : TileIndex [BLOCK_N, HEAD_DIM] =>
        { region := V, baseOffset := kv_offset,
          parentShape := [N_CTX + P_SEQ, HEAD_DIM], blockShape := [BLOCK_N, HEAD_DIM],
          strides := [stride_vk, stride_vn], offsets := [c * BLOCK_N, 0] }⟩)) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-! ### preLoop prefix (deterministic prologue → invariant base seeds)

The first ten prologue statements are deterministic (no loads): the four scalar
offsets, the three block pointers, and the `m_i`/`l_i`/`acc` initializers. They
step with the validated recipe (`stepStmts.cons_some (stepStmt_assign_eq_some
…)` per statement, threading register reads through the accumulated `setReg`
chain) using the block-pointer eval lemmas above. The result is the base-case
register seeding for the loop invariant: `m_i = ⊥` (= `mPg … 0`),
`l_i = 0` (the kernel's seed; `lPgK … 0`), `acc = 0` (= `oPg … 0`), and the
`K`/`V` block pointers at their initial `[0,0]` offsets. -/

/-- Scalar `ref · constNat` eval (the `q_offset` / `kv_offset` / `start_m·BLOCK_M`
offsets). -/
theorem evalOp_mul_ref_const (s : BlockState) (name : RegName) (a c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar a)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (a * c)) := by
  rw [evalOp_mul, evalOp_ref, hr, evalOp_constNat]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop_data, NumericDType.mul]
  rfl

set_option maxHeartbeats 1000000 in
/-- **preLoop prefix** (prologue statements 0–9): steps the deterministic
offset/pointer/init prologue to a state carrying the invariant base seeds. -/
theorem preLoop_prefix (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (out_dtype : FloatDType) (s : BlockState) :
    ∃ s10, stepStmts ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h stride_b0m
        Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N
        out_dtype).toAlgKernel.body.take 10) s = some s10
      ∧ s10.pids = s.pids
      ∧ s10.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s10.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s10.regs .nat [] "q_offset" = some (Tile.scalar (s.pids 1 * stride_qh))
      ∧ s10.regs .nat [] "kv_offset" = some (Tile.scalar (s.pids 1 * stride_kh))
      ∧ s10.regs .blockPtr [BLOCK_DMODEL, BLOCK_N] "K_block_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            { region := K, baseOffset := s.pids 1 * stride_kh,
              parentShape := [BLOCK_DMODEL, N_CTX + P_SEQ], blockShape := [BLOCK_DMODEL, BLOCK_N],
              strides := [stride_kk, stride_kn], offsets := [0, 0] }⟩)
      ∧ s10.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "V_block_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            { region := V, baseOffset := s.pids 1 * stride_kh,
              parentShape := [N_CTX + P_SEQ, BLOCK_DMODEL], blockShape := [BLOCK_N, BLOCK_DMODEL],
              strides := [stride_vk, stride_vn], offsets := [0, 0] }⟩)
      ∧ s10.regs .real [BLOCK_M] "m_i" = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩)
      ∧ s10.regs .real [BLOCK_M] "l_i" = some (⟨fun _ : TileIndex [BLOCK_M] => some (0:ℝ)⟩)
      ∧ s10.regs .real [BLOCK_M, BLOCK_DMODEL] "acc"
          = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0:ℝ)⟩)
      ∧ s10.undef = s.undef
      ∧ s10.mem = s.mem := by
  rw [show ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h stride_b0m
        Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N
        out_dtype).toAlgKernel.body.take 10)
      = [ Stmt.assign .nat [] "start_m" (Op.programId 0),
          Stmt.assign .nat [] "off_hz" (Op.programId 1),
          Stmt.assign .nat [] "q_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat stride_qh)),
          Stmt.assign .nat [] "kv_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat stride_kh)),
          Stmt.assign .blockPtr [BLOCK_M, BLOCK_DMODEL] "Q_block_ptr"
            (Op.makeBlockPtrDynOffsets Q (Op.ref .nat [] "q_offset") [N_CTX, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL]
              [stride_qm, stride_qk] [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M), Op.constNat 0]),
          Stmt.assign .blockPtr [BLOCK_DMODEL, BLOCK_N] "K_block_ptr"
            (Op.makeBlockPtrDyn K (Op.ref .nat [] "kv_offset") [BLOCK_DMODEL, N_CTX + P_SEQ] [BLOCK_DMODEL, BLOCK_N]
              [stride_kk, stride_kn] [0, 0]),
          Stmt.assign .blockPtr [BLOCK_N, BLOCK_DMODEL] "V_block_ptr"
            (Op.makeBlockPtrDyn V (Op.ref .nat [] "kv_offset") [N_CTX + P_SEQ, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL]
              [stride_vk, stride_vn] [0, 0]),
          Stmt.assign .real [BLOCK_M] "m_i" (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
          Stmt.assign .real [BLOCK_M] "l_i" (Op.full [BLOCK_M] (Op.const 0)),
          Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) ] from rfl]
  have hmi : ∀ s' : BlockState, evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) s'
      = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) := by
    intro s'
    simp only [evalOp_add, evalOp_full, evalOp_negInf, evalOp_const, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext r; simp only [Tile.bop_data, NumericDType.add]; rfl
  have hli : ∀ s' : BlockState, evalOp (Op.full [BLOCK_M] (Op.const 0)) s'
      = some (⟨fun _ : TileIndex [BLOCK_M] => some (0:ℝ)⟩ : Tile .real [BLOCK_M]) := by
    intro s'; simp [evalOp_full, evalOp_const, Option.bind]
  have hacc : ∀ s' : BlockState, evalOp (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) s'
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0:ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL]) := by
    intro s'; simp [evalOp_full, evalOp_const, Option.bind]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (s := s) (by rw [evalOp_programId])),
    stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (evalOp_mul_ref_const _ "off_hz" (s.pids 1) stride_qh (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (evalOp_mul_ref_const _ "off_hz" (s.pids 1) stride_kh (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtr_rowcol_eval Q _ [N_CTX, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL] [stride_qm, stride_qk] _ _
        (s.pids 1 * stride_qh) (s.pids 0 * BLOCK_M)
        (by simp [BlockState.setReg]) (evalOp_mul_ref_const _ "start_m" (s.pids 0) BLOCK_M (by simp [BlockState.setReg])))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtrDyn_eval K _ [BLOCK_DMODEL, N_CTX + P_SEQ] [BLOCK_DMODEL, BLOCK_N] [stride_kk, stride_kn] [0,0] _
        (s.pids 1 * stride_kh) (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtrDyn_eval V _ [N_CTX + P_SEQ, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL] [stride_vk, stride_vn] [0,0] _
        (s.pids 1 * stride_kh) (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (hmi _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (hli _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (hacc _)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [BlockState.setReg, BlockState.setReg_same, BlockState.setReg_ne_name]

/-! ### preLoop tail loads (statements 10–18 eval helpers)

The prologue's final block (statements 10–18: `qk_scale`, the Q block-pointer
load + scale + `OUT_DTYPE` cast, the `lo`/`hi` constants, the bias arange/offset
prep, and the `b1` region load) needs three eval reductions beyond the
deterministic-prologue recipe used by `preLoop_prefix`:

* the two generalized canonical-axis `expandDim` reductions over an arbitrary
  `nat` operand (`evalOp_expandDim_one_nat'`/`evalOp_expandDim_zero_nat'`) — the
  generic `evalOp_expandDim` cannot `rw`/`simp`-match the `triton{}`-elaborated
  surface because its result shape `TileShape.insertAxis [M] 1 1` is only
  defeq-not-syntactic to `[M, 1]`; these state the concrete-shape form (the
  `b1`/`b0` offset uses `expandDim` of a *computed* `mul`/`add`, not a bare
  register, so the existing register-only specializations don't apply);
* the `b1` (and identically-shaped `b0`) relative-position-bias region load
  (`load_b1_eval`): reads `B0` at the resolved contiguous per-lane address
  `b_offset + (start_m·BLOCK_M + r)·stride_b0m + (jL % BIAS_LAST_SIZE +
  BIAS_LAST_SIZE)` — i.e. exactly `b1Val`'s address (the `b0` block-column read
  `… + start_n // BLOCK_N` is the `floorDiv` variant). -/

/-- Generalized canonical axis-1 `expandDim` eval over an arbitrary `nat`
operand (explicit `[M, 1]` result shape). -/
@[simp] theorem evalOp_expandDim_one_nat' {M : Nat} (e : Op .nat [M]) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ e) s =
      (evalOp e s).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  conv_lhs => unfold evalOp
  cases evalOp e s with
  | none => rfl
  | some v =>
      simp only [Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_; ext i; simp [Tile.expandDim]

/-- Generalized canonical axis-0 `expandDim` eval over an arbitrary `nat`
operand (explicit `[1, D]` result shape). -/
@[simp] theorem evalOp_expandDim_zero_nat' {D : Nat} (e : Op .nat [D]) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ e) s =
      (evalOp e s).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  conv_lhs => unfold evalOp
  cases evalOp e s with
  | none => rfl
  | some v =>
      simp only [Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_; ext i; simp [Tile.expandDim]

/-- **`b1` relative-position-bias region load** (statement 18; the `b0` per-block
read is the `floorDiv` analogue). Resolves the `B0 + b_offset + (start_m·BLOCK_M
+ b_ptr_offsets_m)·stride_b0m[:,None] + b_ptr_offsets_n_1[None,:]` pointer
arithmetic to the per-lane `readMem` at `b_offset + (start_m·BLOCK_M + r)·
stride_b0m + (jL % BIAS_LAST_SIZE + BIAS_LAST_SIZE)` (= `b1Val`'s address). -/
theorem load_b1_eval (s : BlockState) (B0 : RegionName)
    (BLOCK_M BLOCK_N stride_b0m BIAS_LAST_SIZE smbm boff : Nat)
    (hbo : s.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hm : s.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val)))
    (hn1 : s.regs .nat [BLOCK_N] "b_ptr_offsets_n_1"
      = some (Tile.vec (fun jL : Fin BLOCK_N => jL.val % BIAS_LAST_SIZE + BIAS_LAST_SIZE))) :
    evalOp (Op.load .real (.region B0
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
                (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "b_ptr_offsets_n_1"))))
        .none) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some (s.readMem B0
            (boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m
              + (idx.2.1.val % BIAS_LAST_SIZE + BIAS_LAST_SIZE)))⟩) := by
  rw [evalOp_load_region_none]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat',
    evalOp_expandDim_zero_nat', hbo, hsm, hm, hn1, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, Tile.vec,
    NumericDType.add, NumericDType.mul, BlockState.readMemValue_real, Region.cast_id]

end ClosedForm

end VeriTile.Bench.TritonBenchG.AttentionKernel
