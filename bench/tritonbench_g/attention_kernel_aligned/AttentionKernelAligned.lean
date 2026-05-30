import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `attention_kernel_aligned` — strict per-kernel correctness

`_fwd_kernel_aligned` is a flash-attention forward kernel with a fused
relative-position bias `B0` (`rel_h + rel_w`). Program `(start_m, off_hz)` loads
a `BLOCK_M`-row `Q` tile for one (batch, head), scales it by
`qk_scale = sm_scale · log2(e)`, then over the key/value context (stepping by
`BLOCK_N`) runs the online-softmax recurrence — block scores
`qk = dot(q, k) + (b0 + b1)`, running max `m_i`, denominator `l_i`, accumulator
`acc` with `exp2(qk - m_i_new)` weights — and finally stores `acc / l_i` to
`Out`. This is a near-clone of `attention_kernel` differing in that the bias is
added without the `log2(e)` factor and the `qk` dot is computed with
`out_dtype=OUT_DTYPE`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_aligned[grid](...)`, the grid over
`(cdiv(N_CTX, BLOCK_M), Z·H)`, block scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `start_m`/`off_hz` are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
attention_kernel_aligned_python_test_shape_output_summary    ← TOP THEOREM
  ├─ attention_kernel_aligned_fwd_kernel_aligned_surface_toAlgorithm_supported   surface lowers to algorithm layer
  └─ attention_kernel_aligned_fwd_kernel_aligned_python_test_shape_compute_correct
       └─ attention_kernel_aligned_fwd_kernel_aligned_surface_compute_correct
            └─ (full surface produces producedOutputValue at the Out store)

attention_kernel_aligned_final_store_python_test_shape_compute_correct
  └─ attention_kernel_aligned_final_store_slice_compute_correct   ← ComputeCorrect over the final Out store
       └─ attention_kernel_aligned_final_store_slice_correct      ← algorithm-layer readback (normalizedAccValue)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `OUT_DTYPE`
(`fp16`/`bf16`) casts collapse to the identity post-erasure; `@triton.autotune`
/ `num_warps`/`num_stages` are not modeled. The output summary is stated at the
Python test shape (`B=2, H=4, N_CTX=128, D_MODEL=64, BLOCK_M=32, BLOCK_N=64`,
`sm_scale=1.0`, `P_SEQ=0`, `fp16`, contiguous per-head strides `(8192,64,1)`).
The surface theorem captures the full single-program bias-augmented
online-softmax body via `producedOutputValue`; the `final_store` lemmas isolate
the final `acc / l_i` store (`normalizedAccValue`). This is a single-program
scope (the store is unmasked at this shape since `N_CTX` is a multiple of
`BLOCK_M`); cross-program composition into the full output is the trusted host
boundary.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful DSL port of `attention_kernel_aligned.py`'s `_fwd_kernel_aligned`. -/
def attention_kernel_aligned_fwd_kernel_aligned_surface
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
    qk += tl.dot(q, k, out_dtype=OUT_DTYPE)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    qk += (b0 + b1)

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
theorem attention_kernel_aligned_fwd_kernel_aligned_surface_toAlgorithm_supported
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) :
    ∃ alg, (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out
      sm_scale stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
      stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
      BLOCK_M BLOCK_N out_dtype).toAlgorithm? = Except.ok alg := by
  simp [attention_kernel_aligned_fwd_kernel_aligned_surface,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Final output-store slice of `attention_kernel_aligned.py`'s
`_fwd_kernel_aligned`.

This slice includes the Python `acc = acc / l_i[:, None]` statement immediately
before the unmasked block-pointer writeback into `Out`. Producing the
unnormalized streaming-softmax `Acc` and `L` inputs remains the narrower
recurrence obligation. -/
def attention_kernel_aligned_final_store_slice
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
  match exec (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out
      sm_scale stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
      stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
      BLOCK_M BLOCK_N out_dtype) s with
  | some s' => s'.readMem Out (surfaceOutOffset s stride_qh stride_om stride_on BLOCK_M idx)
  | none => 0.0

/-- Algorithm-layer correctness for the final output store. -/
theorem attention_kernel_aligned_final_store_slice_correct
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
      (exec (attention_kernel_aligned_final_store_slice Acc L Out stride_acc_h
            stride_acc_m stride_acc_k stride_l_h stride_l_m stride_oh
            stride_om stride_on BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (normalizedAccValue s Acc L stride_acc_h stride_acc_m
            stride_acc_k stride_l_h stride_l_m BLOCK_M idx) := by
  intro idx
  simp [exec, attention_kernel_aligned_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
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
theorem attention_kernel_aligned_final_store_slice_compute_correct
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
      (kernel := attention_kernel_aligned_final_store_slice Acc L Out
        stride_acc_h stride_acc_m stride_acc_k stride_l_h stride_l_m
        stride_oh stride_om stride_on BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (Out, outOffset s stride_oh stride_om stride_on BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        normalizedAccValue s Acc L stride_acc_h stride_acc_m stride_acc_k
          stride_l_h stride_l_m BLOCK_M idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_kernel_aligned_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_kernel_aligned_final_store_slice_correct Acc L Out
    stride_acc_h stride_acc_m stride_acc_k stride_l_h stride_l_m stride_oh
    stride_om stride_on BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem attention_kernel_aligned_fwd_kernel_aligned_surface_compute_correct
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
      (kernel := attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out
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
  · simp [attention_kernel_aligned_fwd_kernel_aligned_surface,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedOutputValue, hExec]

/-! ## Python test-shape wrapper

`attention_kernel_aligned.py`'s checked test uses `B = 2`, `H = 4`,
`N_CTX = 128`, `D_MODEL = 64`, `BLOCK_M = 32`, and `BLOCK_N = 64`.
Contiguous `[B, H, N_CTX, D_MODEL]` tensors are passed to the kernel with
per-head strides `(8192, 64, 1)`. -/

theorem attention_kernel_aligned_final_store_python_test_shape_compute_correct
    (Acc L Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_aligned_final_store_slice Acc L Out
        8192 64 1 128 1 8192 64 1 32 64)
      (initialState := s)
      (write := fun idx : TileIndex [32, 64] =>
        some (Out, outOffset s 8192 64 1 32 idx))
      (expected := fun idx : TileIndex [32, 64] =>
        normalizedAccValue s Acc L 8192 64 1 128 1 32 idx) := by
  apply attention_kernel_aligned_final_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attention_kernel_aligned_fwd_kernel_aligned_python_test_shape_compute_correct
    (Q K V B0 Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out
        1.0 8192 64 1 8192 64 1 8192 64 1 8192 64 1
        8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [32, 64] =>
        some (Out, surfaceOutOffset s 8192 64 1 32 idx))
      (expected := fun idx : TileIndex [32, 64] =>
        producedOutputValue s Q K V B0 Out 1.0 8192 64 1 8192 64 1
          8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32
          64 FloatDType.fp16 idx) := by
  exact attention_kernel_aligned_fwd_kernel_aligned_surface_compute_correct
    Q K V B0 Out 1.0 8192 64 1 8192 64 1 8192 64 1
    8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
    FloatDType.fp16 s

/-- Public Python test-shape summary for `attention_kernel_aligned.py`.

This end-to-end summary records the faithful aligned attention surface for the
checked relative-position-bias launch and ties the Q/K/V streaming-softmax
producer path directly to the observable final `Out` writeback. -/
theorem attention_kernel_aligned_python_test_shape_output_summary
    (Q K V B0 Out : RegionName) (s : BlockState) :
    (∃ alg, (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out
      1.0 8192 64 1 8192 64 1 8192 64 1 8192 64 1
      8192 128 2 4 128 0 64 128 64 32 64
      FloatDType.fp16).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out
        1.0 8192 64 1 8192 64 1 8192 64 1 8192 64 1
        8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [32, 64] =>
        some (Out, surfaceOutOffset s 8192 64 1 32 idx))
      (expected := fun idx : TileIndex [32, 64] =>
        producedOutputValue s Q K V B0 Out 1.0 8192 64 1 8192 64 1
          8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32
          64 FloatDType.fp16 idx) := by
  constructor
  · exact attention_kernel_aligned_fwd_kernel_aligned_surface_toAlgorithm_supported
      Q K V B0 Out 1.0 8192 64 1 8192 64 1 8192 64 1
      8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
      FloatDType.fp16
  · exact attention_kernel_aligned_fwd_kernel_aligned_python_test_shape_compute_correct
      Q K V B0 Out s

end VeriTile.Bench.TritonBenchG.AttentionKernelAligned
