import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant
import VeriTile.Triton.Semantics.BlockPtrEval

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
attention_kernel_aligned_python_test_shape_output_summary    ← TOP THEOREM (genuine, NON-self-referential)
  ├─ attention_kernel_aligned_fwd_kernel_aligned_surface_toAlgorithm_supported   surface lowers to algorithm layer
  └─ ClosedForm.aligned_genuine_output_compute_correct
       └─ ClosedForm.aligned_exec    ← whole-kernel exec assembly (preLoop + forRangeDyn + postLoop)
            └─ (every Out lane = genuine closed form `alignedClosedForm`)

genuine producer closed form (sorry-free; exec assembly now connected):
  alignedClosedForm  := attentionRealBase2ScalarScaleBias (loaded Q/K/V) (sm_scale·log2e) (rel_h+rel_w bias)
  alignedClosedForm_eq_streaming  → Math/Attention.lean (osStep fold == batch base-2 softmax)
  aligned_exec : preLoop (→ invariant 0) + forRangeDyn_inv over aligned_step + aligned_postLoop;
    attnGenScore_eq_alignedClosedForm bridges the genuine `fscore` softmax to `alignedClosedForm`
    (with `log2e = 1.44269504`, the kernel's literal `qk_scale` constant).
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `OUT_DTYPE`
(`fp16`/`bf16`) casts collapse to the identity post-erasure; `@triton.autotune`
/ `num_warps`/`num_stages` are not modeled. The output summary is stated at the
Python test shape (`B=2, H=4, N_CTX=128, D_MODEL=64, BLOCK_M=32, BLOCK_N=64`,
`sm_scale=1.0`, `P_SEQ=0`, `fp16`, contiguous per-head strides `(8192,64,1)`).
The public summary asserts the **genuine** closed form: every observable `Out`
lane equals the base-2 streaming-softmax `alignedClosedForm` of the loaded Q/K/V
tiles under the scalar score scale `sm_scale·log2(e)` and the fused `rel_h+rel_w`
bias — discharged whole-kernel by `ClosedForm.aligned_exec`, NOT a self-referential
readback. The `producedOutputValue` definition and `*_surface_compute_correct`
lemmas below are kept only as internal *execution observations*. The genuine
producer closed form is `alignedClosedForm`, with the streaming bridge
`alignedClosedForm_eq_streaming` and the exec-side assembly
`ClosedForm.aligned_exec` both proved sorry-free. This is a
single-program scope (the store is unmasked at this shape since `N_CTX` is a
multiple of `BLOCK_M`); cross-program composition into the full output is the
trusted host boundary.
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

/-- **Execution observation (NOT a specification).** The kernel's own executed
`Out` value at lane `idx` — i.e. `exec(surface) |> readMem Out`. This is
self-referential (`Out = whatever the kernel writes`) and is therefore used only
as an internal observation, never as the public summary's `expected`. The genuine
closed-form specification is `alignedClosedForm`. -/
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

/-! ## Genuine closed-form output spec (NOT self-referential)

`producedOutputValue` above is the kernel's own executed `Out` value; it is the
*observation*, not a specification. The definitions below give the **genuine
closed form** the aligned online-softmax recurrence computes — the base-2
softmax over the loaded `Q`/`K`/`V` tiles with the scalar score scale
`qk_scale = sm_scale · log2(e)` folded into `q` and the fused `rel_h + rel_w`
position bias `b0 + b1` added to the score — expressed over the loaded memory
tiles, independent of the kernel's execution. The streaming-softmax math heart
that justifies it (`osStep` fold == batch base-2 softmax) is proved sorry-free in
`VeriTile/Triton/Math/Attention.lean`
(`attentionRealBase2ScalarScaleBias_eq_streaming`).

Under the Python launch layout (contiguous per-head `Q`/`K`/`V` with
`stride_qm = stride_kn = stride_vk = BLOCK_DMODEL`, head stride `1`, and
`P_SEQ = 0` so `S = N_CTX`), key `j`, head lane `e` of the per-`(batch,head)`
tile sits at `pid₁ · stride_qh + j · BLOCK_DMODEL + e`. -/

open VeriTile.Triton (attentionRealBase2ScalarScaleBias)

/-- The kernel's literal base-2-`e` constant `1.44269504` (the truncated
`log2(e) = 1 / log 2` the `@triton.jit` source folds into `q` via
`qk_scale = sm_scale · 1.44269504`, `q = (q · qk_scale).to(...)`). The genuine
closed form `alignedClosedForm` below is stated with exactly this constant so
that the executed `qk_scale` register and the spec's score scale coincide
literally (no `log2(e)`-rounding gap), matching the kernel faithfully. -/
noncomputable def log2e : ℝ := 1.44269504

/-- Loaded `Q` tile: block row `i`, head lane `e` at
`pid₁ · stride_qh + (pid₀ · BLOCK_M + i) · BLOCK_DMODEL + e`. -/
noncomputable def alignedQTile (s : BlockState) (Q : RegionName)
    (stride_qh BLOCK_DMODEL BLOCK_M : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (s.pids 1 * stride_qh + mIndex s BLOCK_M i * BLOCK_DMODEL + e.val)

/-- Loaded `K` tile: key `j`, head lane `e` at
`pid₁ · stride_qh + j · BLOCK_DMODEL + e`. -/
noncomputable def alignedKTile (s : BlockState) (K : RegionName)
    (stride_qh BLOCK_DMODEL S : Nat) :
    TileIndex [S, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (s.pids 1 * stride_qh + j.val * BLOCK_DMODEL + e.val)

/-- Loaded `V` tile: key `j`, channel `d` at
`pid₁ · stride_qh + j · BLOCK_DMODEL + d`. -/
noncomputable def alignedVTile (s : BlockState) (V : RegionName)
    (stride_qh BLOCK_DMODEL S : Nat) :
    TileIndex [S, BLOCK_DMODEL] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (s.pids 1 * stride_qh + j.val * BLOCK_DMODEL + d.val)

/-- Fused relative-position bias `bias i j = b0 + b1` added to the `(i, j)`
score. Mirrors the kernel: `b0` indexes the block column `j / BLOCK_N`, `b1`
indexes the per-lane column `(j % BLOCK_N) % BIAS_LAST_SIZE + BIAS_LAST_SIZE`,
both at row `pid₀ · BLOCK_M + i` of the `B0` table
(`b_offset = pid₁ · stride_b0h`, row stride `stride_b0m`). -/
noncomputable def alignedBias (s : BlockState) (B0 : RegionName)
    (stride_b0h stride_b0m BIAS_LAST_SIZE BLOCK_M BLOCK_N S : Nat) :
    Fin BLOCK_M → Fin S → ℝ :=
  fun i j =>
    let row := s.pids 1 * stride_b0h + mIndex s BLOCK_M i * stride_b0m
    s.readMem B0 (row + j.val / BLOCK_N) +
      s.readMem B0 (row + (j.val % BLOCK_N) % BIAS_LAST_SIZE + BIAS_LAST_SIZE)

/-- **Genuine closed-form `Out`-store value** for `attention_kernel_aligned`: the
base-2 attention of the loaded `Q`/`K`/`V` tiles, with the constant scalar score
scale `sm_scale · log2(e)` and the fused `rel_h + rel_w` bias `b0 + b1`. This is
the value the streaming softmax `acc / l_i` computes — defined over the loaded
tiles, NOT the kernel's own executed output (`producedOutputValue`). -/
noncomputable def alignedClosedForm
    (s : BlockState) (Q K V B0 : RegionName) (sm_scale : ℝ)
    (stride_qh stride_b0h stride_b0m
      N_CTX BIAS_LAST_SIZE BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealBase2ScalarScaleBias
    (alignedQTile s Q stride_qh BLOCK_DMODEL BLOCK_M)
    (alignedKTile s K stride_qh BLOCK_DMODEL N_CTX)
    (alignedVTile s V stride_qh BLOCK_DMODEL N_CTX)
    (sm_scale * log2e)
    (alignedBias s B0 stride_b0h stride_b0m BIAS_LAST_SIZE BLOCK_M BLOCK_N N_CTX)
    idx

/-- The genuine `Out`-value closed form unfolds to the streaming online-softmax
`acc / l` fold over every key (the form the `exec`-side loop produces). Sorry-free
bridge to `Math/Attention.lean`; the remaining `exec` obligation has to identify
the kernel's running `acc / l_i` with this fold. -/
theorem alignedClosedForm_eq_streaming
    (s : BlockState) (Q K V B0 : RegionName) (sm_scale : ℝ)
    (stride_qh stride_b0h stride_b0m
      N_CTX BIAS_LAST_SIZE BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    alignedClosedForm s Q K V B0 sm_scale stride_qh stride_b0h stride_b0m
        N_CTX BIAS_LAST_SIZE BLOCK_DMODEL BLOCK_M BLOCK_N (i, d, PUnit.unit)
      = (let st := (VeriTile.Triton.attnKeyListBias
            (alignedQTile s Q stride_qh BLOCK_DMODEL BLOCK_M)
            (alignedKTile s K stride_qh BLOCK_DMODEL N_CTX)
            (alignedVTile s V stride_qh BLOCK_DMODEL N_CTX)
            (sm_scale * log2e)
            (alignedBias s B0 stride_b0h stride_b0m BIAS_LAST_SIZE BLOCK_M BLOCK_N N_CTX)
            i d).foldl VeriTile.Triton.osStep (0, 0, 0)
         st.2.2 / st.2.1) :=
  VeriTile.Triton.attentionRealBase2ScalarScaleBias_eq_streaming _ _ _ _ _ i d

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


/-! ## Genuine closed-form exec assembly (mirrors `attention_kernel` / PR #304)

The section below builds the **genuine** end-to-end exec correctness: every
observable `Out` lane of `_fwd_kernel_aligned` equals the banked closed form
`alignedClosedForm` (the base-2 streaming softmax of the loaded `Q`/`K`/`V` tiles
under the scalar score scale `sm_scale · log2(e)` plus the fused `rel_h + rel_w`
bias `b0 + b1`). It mirrors PR #304 (`attention_kernel`, the same exp2-noncausal-
bias sub-family) statement-for-statement, adapting (a) the Python test shape
(`BLOCK_M = 32`, `HEAD_DIM = BLOCK_DMODEL = 64`, `BLOCK_N = 64`, per-head strides
`(8192, 64, 1)`, `sm_scale = 1.0`); and (b) the bias add — aligned does
`qk += (b0 + b1)` (additive, **no** `· log2(e)` factor) where `attention_kernel`
does `qk += (b0 + b1) · 1.44269504`.

Per the modeling boundary, `log2e` is the kernel's literal `1.44269504`, so the
score scale `sm_scale · log2e` the spec carries is exactly the `qk_scale`
register the kernel computes. -/

namespace ClosedForm

open VeriTile.Triton

/-! ### Block-pointer `evalOp` reductions (K/V/Q/O pointers) -/

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

/-! ### Genuine kernel specification (loaded tiles + the aligned per-key score) -/

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
`qk_scale·(Σ_e Q[r,e]·K[e,j]) + (b0[r, j/BN] + b1[r, j%BN])`, with
`qk_scale = sm_scale · 1.44269504` already folded into the pre-scaled `q` and the
fused `rel_h + rel_w` bias added **without** the `log2(e)` factor (the one
difference from `attention_kernel`). The base-2 batch softmax of this score is
the kernel's closed form. -/
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

/-- **Closed-form readout.** After all `nB` key blocks the running `oPg / lPg`
ratio equals `attnGenScore (fscore …) (vFlat …)` (the base-2 softmax of the
genuine aligned per-key score). Direct corollary of `closed_form_g`. -/
theorem fscore_ratio_eq_attnGenScore (s0 : BlockState) (Q K V B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM BIAS_LAST_SIZE stride_b0m nB : Nat)
    (start_m : Nat) (hBN : 0 < BLOCK_N) (hnB : 1 ≤ nB)
    (i : Fin BLOCK_M) (d : Fin HEAD_DIM) :
    let score := fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m start_m
    let V' := vFlat s0 V kv_offset HEAD_DIM (BLOCK_N * nB)
    oPg score V' i d nB / lPg score i nB
      = attnGenScore score V' (i, d, PUnit.unit) :=
  closed_form_g _ _ hBN hnB i d

/-- **Kernel-faithful running denominator** (kernel seed `l_i = 0`). -/
noncomputable def lPgK {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        alphaPg score i c * lPgK score i c +
          Finset.univ.sum (fun a : Fin BN =>
            pow2 (score i (gkey BN nB c (by omega) a) - mRg score i (c + 1)))
      else lPgK score i c

theorem lPgK_eq_lPg {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) :
    ∀ c, 1 ≤ c → c ≤ nB → lPgK score i c = lPg score i c := by
  intro c hc1 hc
  induction c, hc1 using Nat.le_induction with
  | base => simp only [lPgK, lPg, dif_pos hc, alphaPg_zero score i, zero_mul]
  | succ c hc1 ih =>
    have ihc := ih (by omega)
    simp only [lPgK, lPg, dif_pos hc, ihc]

/-! ### Generic streaming-softmax register bridges (mirror #304) -/

theorem realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

theorem realExp2_unbotD_coe (r : ℝ) :
    (WithBot.realExp2 ((r : ℝ) : WithBot ℝ)).unbotD 0 = pow2 r := by
  simp [pow2, mul_comm]

theorem withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

theorem dot_pv (BM BN BD : Nat) (p : Tile .real [BM,BN]) (v : Tile .real [BN,BD])
    (r : Fin BM) (d : Fin BD) (fp fv : Fin BN → ℝ)
    (hp : ∀ jL : Fin BN, p.data (r, jL, PUnit.unit) = some (fp jL))
    (hv : ∀ jL : Fin BN, v.data (jL, d, PUnit.unit) = some (fv jL)) :
    (Tile.dot [] p v).data (r, d, PUnit.unit) = some (Finset.univ.sum fun jL : Fin BN => fp jL * fv jL) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (p.data (r, k, PUnit.unit)) (v.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ (fun k => (some (fp k * fv k) : WithBot ℝ))
      from Finset.sum_congr rfl (fun k _ => by rw [hp k, hv k]; rfl)]
  exact withBot_sum_some _

theorem exp2_some {M N : Nat} (h : Fin M → Fin N → ℝ) (x : Tile .real [M, N])
    (i : Fin M) (j : Fin N) (hx : x.data (i, j, PUnit.unit) = some (h i j)) :
    (Tile.uop WithBot.realExp2 x).data (i, j, PUnit.unit) = some (pow2 (h i j)) := by
  show WithBot.realExp2 _ = _
  rw [hx, WithBot.realExp2_some]
  refine congrArg some ?_
  rw [show pow2 (h i j) = Real.exp (Real.log 2 * (h i j)) from rfl, mul_comm]

theorem reduceMaxDrop_data_row (BM BN : Nat) (hBN : 0 < BN) (qk : Tile .real [BM, BN])
    (rmaxT : Tile .real [BM]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qk = some rmaxT)
    (r : Fin BM) (g : Fin BN → WithBot ℝ) (hqk : ∀ jL : Fin BN, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [BM,BN] (⟨1, by simp⟩ : Fin [BM,BN].length) from hBN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

theorem mijg_eq {Mq : Nat} (BN nB c : Nat) (hc : c < nB)
    (score : Fin Mq → Fin (BN*nB) → ℝ)
    (m_i rmaxT : Tile .real [Mq]) (r : Fin Mq)
    (hmi : m_i.data (r, PUnit.unit) = mPg BN nB score r c)
    (hrmax : rmaxT.data (r, PUnit.unit)
        = Finset.univ.sup (fun a : Fin BN => ((score r (gkey BN nB c hc a) : ℝ) : WithBot ℝ))) :
    (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) m_i rmaxT) m_i rmaxT).data (r, PUnit.unit)
      = mPg BN nB score r (c+1) := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrmax]
  rw [mPg, dif_pos (by omega : c + 1 ≤ nB)]
  by_cases h : mPg BN nB score r c ≤ Finset.univ.sup (fun a : Fin BN => ((score r (gkey BN nB c hc a) : ℝ) : WithBot ℝ))
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

theorem mPg_succ_ne_bot {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN*nB) → ℝ) (r : Fin Mq) (c : Nat)
    (hc : mPg BN nB score r c ≠ ⊥) : mPg BN nB score r (c+1) ≠ ⊥ := by
  rw [mPg]
  by_cases h : c + 1 ≤ nB
  · rw [dif_pos h]
    intro hbot; rw [max_eq_bot] at hbot; exact hc hbot.1
  · rw [dif_neg h]; exact hc

theorem alphag_eq {Mq : Nat} (BN nB c : Nat)
    (score : Fin Mq → Fin (BN*nB) → ℝ)
    (m_i m_ij : Tile .real [Mq]) (r : Fin Mq)
    (hmi : m_i.data (r, PUnit.unit) = mPg BN nB score r c)
    (hmij : m_ij.data (r, PUnit.unit) = mPg BN nB score r (c+1)) :
    (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) m_i m_ij)).data (r, PUnit.unit)
      = some (alphaPg score r c) := by
  show WithBot.realExp2 _ = _
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmi, hmij, NumericDType.sub]
  unfold alphaPg
  cases hm : mPg BN nB score r c with
  | bot => rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
  | coe a =>
    show WithBot.realExp2 (WithBot.realSub (↑a) (mPg BN nB score r (c+1))) = some (pow2 (a - mRg score r (c+1)))
    rw [show mRg score r (c+1) = (mPg BN nB score r (c+1)).unbotD 0 from rfl]
    have hne : mPg BN nB score r (c+1) ≠ ⊥ := mPg_succ_ne_bot score r c (by rw [hm]; exact WithBot.coe_ne_bot)
    cases hm1 : mPg BN nB score r (c+1) with
    | bot => exact absurd hm1 hne
    | coe b =>
      rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
      refine congrArg some ?_
      rw [show pow2 (a - b) = Real.exp (Real.log 2 * (a - b)) from rfl, mul_comm]

set_option maxHeartbeats 1600000 in
theorem lig_eq {Mq : Nat} (BN nB c : Nat) (hBN : 0 < BN) (hc : c < nB)
    (score : Fin Mq → Fin (BN*nB) → ℝ)
    (m_i m_ij l_i : Tile .real [Mq]) (qk : Tile .real [Mq, BN]) (r : Fin Mq)
    (hmi : m_i.data (r, PUnit.unit) = mPg BN nB score r c)
    (hmij : m_ij.data (r, PUnit.unit) = mPg BN nB score r (c+1))
    (hli : l_i.data (r, PUnit.unit) = some (lPgK score r c))
    (hqk : ∀ a : Fin BN, qk.data (r, a, PUnit.unit) = some (score r (gkey BN nB c hc a))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
       (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) l_i
         (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) m_i m_ij)))
       (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [Mq,BN].length)
         (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qk
           (Tile.expandDim (⟨1, by simp⟩ : Fin [Mq].length.succ) m_ij))))).data (r, PUnit.unit)
      = some (lPgK score r (c+1)) := by
  have hmijReal : m_ij.data (r, PUnit.unit) = some (mRg score r (c+1)) := by
    rw [hmij]
    rw [mPg_eq_coe score hBN r (c+1) (by omega) (by omega)]; rfl
  have halpha := alphag_eq BN nB c score m_i m_ij r hmi hmij
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [halpha, hli]
  simp only [Tile.reduceSumDrop_data, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.expandDim_data, TileShape.dropInsertedIndex,
    TileShape.insertAxisIndex, hqk, hmijReal, NumericDType.sub, WithBot.realSub,
    Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
  simp only [withBot_sum_some]
  rw [lPgK, dif_pos (by omega : c + 1 ≤ nB)]
  simp only [NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [mul_comm (lPgK score r c)]
  congr 1
  exact Finset.sum_congr rfl (fun k _ => by rw [pow2, mul_comm])

set_option maxHeartbeats 1600000 in
theorem accg_eq {Mq Dh : Nat} (BN nB c : Nat) (hc : c < nB)
    (score : Fin Mq → Fin (BN*nB) → ℝ) (V : TileIndex [BN*nB, Dh] → ℝ)
    (acc : Tile .real [Mq,Dh]) (alpha : Tile .real [Mq]) (p : Tile .real [Mq,BN]) (v : Tile .real [BN,Dh])
    (r : Fin Mq) (d : Fin Dh)
    (hacc : acc.data (r, d, PUnit.unit) = some (oPg score V r d c))
    (halpha : alpha.data (r, PUnit.unit) = some (alphaPg score r c))
    (hp : ∀ a : Fin BN, p.data (r, a, PUnit.unit) = some (pow2 (score r (gkey BN nB c hc a) - mRg score r (c+1))))
    (hv : ∀ a : Fin BN, v.data (a, d, PUnit.unit) = some (V (gkey BN nB c hc a, d, PUnit.unit))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
       (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acc
         (Tile.expandDim (⟨1, by simp⟩ : Fin [Mq].length.succ) alpha))
       (Tile.dot [] p v)).data (r, d, PUnit.unit)
      = some (oPg score V r d (c+1)) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, halpha]
  rw [dot_pv Mq BN Dh p v r d (fun a => pow2 (score r (gkey BN nB c hc a) - mRg score r (c+1)))
      (fun a => V (gkey BN nB c hc a, d, PUnit.unit)) hp hv]
  rw [hacc]
  rw [oPg, dif_pos (by omega : c + 1 ≤ nB)]
  simp only [NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [mul_comm (oPg score V r d c)]


/-! ### Loop invariant skeleton -/

noncomputable def alignedInvariant
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
  (s.regs .fp16 [BLOCK_M, HEAD_DIM] "q" = some ⟨fun idx : TileIndex [BLOCK_M, HEAD_DIM] =>
      FloatDType.real.cast FloatDType.fp16
        (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset BLOCK_M HEAD_DIM start_m idx))⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_N] "b1" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE start_m idx.1 idx.2.1.val)⟩) ∧
  (s.regs .nat [] "b_offset" = some (Tile.scalar b_offset)) ∧
  (s.regs .nat [] "start_m" = some (Tile.scalar start_m)) ∧
  (s.regs .nat [] "q_offset" = some (Tile.scalar q_offset)) ∧
  (s.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val))) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-! ### preLoop prefix (deterministic prologue 0-9 → invariant base seeds) -/

theorem evalOp_mul_ref_const (s : BlockState) (name : RegName) (a c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar a)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (a * c)) := by
  rw [evalOp_mul, evalOp_ref, hr, evalOp_constNat]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop_data, NumericDType.mul]
  rfl

set_option maxHeartbeats 1600000 in
theorem preLoop_prefix (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (out_dtype : FloatDType) (s : BlockState) :
    ∃ s10, stepStmts ((attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h stride_b0m
        Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N
        out_dtype).toAlgKernel.body.take 10) s = some s10
      ∧ s10.pids = s.pids
      ∧ s10.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s10.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s10.regs .nat [] "q_offset" = some (Tile.scalar (s.pids 1 * stride_qh))
      ∧ s10.regs .nat [] "kv_offset" = some (Tile.scalar (s.pids 1 * stride_kh))
      ∧ s10.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "Q_block_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            { region := Q, baseOffset := s.pids 1 * stride_qh,
              parentShape := [N_CTX, BLOCK_DMODEL], blockShape := [BLOCK_M, BLOCK_DMODEL],
              strides := [stride_qm, stride_qk], offsets := [s.pids 0 * BLOCK_M, 0] }⟩)
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
  rw [show ((attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
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
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [BlockState.setReg, BlockState.setReg_same, BlockState.setReg_ne_name]


/-! ### preLoop tail loads (statements 10-18 eval helpers) -/

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

theorem evalOp_floorDiv' {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

theorem load_b0_eval (s : BlockState) (B0 : RegionName)
    (BLOCK_M stride_b0m smbm boff snv : Nat) (hax : 1 < [BLOCK_M].length.succ)
    (hbo : s.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar snv))
    (hm : s.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val))) :
    evalOp ((Op.load .real (.region B0
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, hax⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
                (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64))))
        .none : Op .real [BLOCK_M, 1])) s
      = some (⟨fun idx : TileIndex [BLOCK_M, 1] =>
          some (s.readMem B0
            (boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + snv / 64))⟩ : Tile .real [BLOCK_M, 1]) := by
  rw [evalOp_load_region_none]
  have hoff : evalOp ((Op.add .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
          (Op.expandDim ⟨1, hax⟩ (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
              (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
        (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64))) : Op .nat [BLOCK_M, 1]) s
      = some (⟨fun idx : TileIndex [BLOCK_M, 1] =>
          boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + snv / 64⟩ : Tile .nat [BLOCK_M, 1]) := by
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    rw [evalOp_ref, hsm]
    simp only [evalOp_ref, evalOp_constNat, hbo, hsn, hm, Option.bind, Option.map]
    refine congrArg some ?_
    ext idx
    simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
      Broadcast.rightIndex, Tile.scalar, Tile.vec, NumericDType.add, NumericDType.mul,
      IntegralDType.floorDiv]
  rw [hoff]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [BlockState.readMemValue_real, Region.cast_id]

theorem evalOp_exp2' {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- The lowered 15-statement `forRangeDyn` body of the aligned Python-shape
kernel. Differs from `attention_kernel` only in the bias add (L5: `qk += (b0 +
b1)`, **no** `· 1.44269504`) and the tile shapes (`BLOCK_M=32`, `HEAD_DIM=64`). -/
def attnLoopBody (B0 : RegionName) : List Stmt :=
  [ Stmt.assign .real [64, 64] "k"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") []) .none),
    Stmt.assign .real [64, 64] "v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") []) .none),
    Stmt.assign .fp16 [32, 64] "qk" (Op.full [32, 64] (Op.castFloat .real .fp16 (Op.const 0))),
    Stmt.assign .real [32, 64] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "qk"))
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "q"))
          (Op.ref .real [64, 64] "k"))),
    Stmt.assign .real [32, 1] "b0"
      (Op.load .real (.region B0
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 32))
                (Op.ref .nat [32] "b_ptr_offsets_m")) (Op.constNat 128))))
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64))))
        .none),
    Stmt.assign .real [32, 64] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [32, 64] "qk")
        (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.ref .real [32, 1] "b0") (Op.ref .real [32, 64] "b1"))),
    Stmt.assign .real [32] "m_i_new"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [32] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [32, 64].length) Bool.false (Op.ref .real [32, 64] "qk")))
        (Op.ref .real [32] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [32, 64].length) Bool.false (Op.ref .real [32, 64] "qk"))),
    Stmt.assign .real [32] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [32] "m_i")
        (Op.ref .real [32] "m_i_new"))),
    Stmt.assign .real [32, 64] "p"
      (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [32, 64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [32] "m_i_new")))),
    Stmt.assign .real [32, 64] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [32, 64] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [32] "alpha"))),
    Stmt.assign .real [32, 64] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [32, 64] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [32, 64] "p"))) (Op.ref .real [64, 64] "v"))),
    Stmt.assign .real [32] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [32] "l_i")
          (Op.ref .real [32] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [32, 64].length) Bool.false (Op.ref .real [32, 64] "p"))),
    Stmt.assign .real [32] "m_i" (Op.ref .real [32] "m_i_new"),
    Stmt.assign .blockPtr [64, 64] "K_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") [0, 64]),
    Stmt.assign .blockPtr [64, 64] "V_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") [64, 0]) ]

theorem attnLoopBody_check (Q K V B0 Out : RegionName) :
    (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body.drop 19
      = Stmt.forRangeDyn "start_n" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
          (Op.constNat 64) (attnLoopBody B0)
        :: (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
            8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
            FloatDType.fp16).toAlgKernel.body.drop 20 := by
  rfl

def attnPostLoop (Out : RegionName) : List Stmt :=
  [ Stmt.assign .real [32, 64] "acc"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [32, 64] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [32] "l_i"))),
    Stmt.assign .blockPtr [32, 64] "O_block_ptr"
      (Op.makeBlockPtrDynOffsets Out (Op.ref .nat [] "q_offset") [128, 64] [32, 64]
        [64, 1] [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 32), Op.constNat 0]),
    Stmt.store .fp16 [32, 64] (.blockPtr (Op.ref .blockPtr [32, 64] "O_block_ptr") [])
      (Op.castFloat .real .fp16 (Op.ref .real [32, 64] "acc")) .none ]

theorem attnPostLoop_check (Q K V B0 Out : RegionName) :
    (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body.drop 20 = attnPostLoop Out := by
  rfl


/-! ### Loop-body op-eval recipes (aligned Python shape `BLOCK_M=32`, `HEAD_DIM=BLOCK_N=64`) -/

theorem qk_dot_op_eval (s : BlockState) (qtile : Tile .fp16 [32, 64]) (ktile : Tile .real [64, 64])
    (hqk : s.regs .fp16 [32, 64] "qk" = some ⟨fun _ : TileIndex [32, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some 0)⟩)
    (hq : s.regs .fp16 [32, 64] "q" = some qtile)
    (hk : s.regs .real [64, 64] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "qk"))
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "q"))
          (Op.ref .real [64, 64] "k"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [32, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
          (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [32, 64]) ktile)) := by
  have hczA : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "qk")) s
      = some (⟨fun _ : TileIndex [32, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩ : Tile .real [32, 64]) := by
    rw [evalOp_castFloat]; simp [hqk]
  have hcz : @evalOp TileDType.real [32, 64] (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "qk")) s
      = some (⟨fun _ : TileIndex [32, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩ : Tile .real [32, 64]) := hczA
  have hcbA : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [32, 64]) := by
    rw [evalOp_castFloat]; simp [hq]
  have hcb2 : @evalOp TileDType.real [32, 64] (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [32, 64]) := hcbA
  have hdotN : evalOp (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "q")) (Op.ref .real [64, 64] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [32, 64]) ktile) := by
    rw [evalOp_dot]; simp [hcb2, hk]
  have hdotN2 : @evalOp TileDType.real [32, 64]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "q")) (Op.ref .real [64, 64] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [32, 64]) ktile) := hdotN
  rw [evalOp_add, hcz, hdotN2]; rfl

/-- **`qk += (b0 + b1)`** (aligned L5): additive relative-position bias, **no**
`· 1.44269504` factor (the lone difference from `attention_kernel`). -/
theorem qk_bias_op_eval (s : BlockState) (qktile : Tile .real [32, 64])
    (b0tile : Tile .real [32, 1]) (b1tile : Tile .real [32, 64])
    (hqk : s.regs .real [32, 64] "qk" = some qktile)
    (hb0 : s.regs .real [32, 1] "b0" = some b0tile)
    (hb1 : s.regs .real [32, 64] "b1" = some b1tile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [32, 64] "qk")
        (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.ref .real [32, 1] "b0") (Op.ref .real [32, 64] "b1"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qktile
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0tile b1tile)) := by
  rw [evalOp_add]
  simp only [evalOp_add, evalOp_ref, hqk, hb0, hb1, Option.bind_eq_bind, Option.bind_some]

theorem ak_ref_op_eval (s : BlockState) {dtype : TileDType} {shape : TileShape}
    (name : RegName) (v : Tile dtype shape) (h : s.regs dtype shape name = some v) :
    evalOp (Op.ref dtype shape name) s = some v := by rw [evalOp_ref, h]

theorem ak_mnew_op_eval (s : BlockState) (BM BN : Nat)
    (mtile : Tile .real [BM]) (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hmax : s.regs .real [BM] "m_i" = some mtile)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")))
        (Op.ref .real [BM] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) := by
  have hrmaxN : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [BM]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmax, hrmax, Option.bind_eq_bind, Option.bind_some]

theorem ak_alpha_op_eval (s : BlockState) (BM : Nat) (mtile mnewtile : Tile .real [BM])
    (hmax : s.regs .real [BM] "m_i" = some mtile) (hmnew : s.regs .real [BM] "m_i_new" = some mnewtile) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_i_new"))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewtile)) := by
  rw [evalOp_exp2', evalOp_sub]; simp [evalOp_ref, hmax, hmnew]

theorem ak_p_op_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM, BN]) (mnewtile : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) (hmnew : s.regs .real [BM] "m_i_new" = some mnewtile) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_i_new")))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qktile
          (Tile.expandDim ⟨1, hax⟩ mnewtile))) := by
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_i_new")) s
      = some (Tile.expandDim ⟨1, hax⟩ mnewtile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmnew
  rw [evalOp_exp2', evalOp_sub]
  simp only [evalOp_ref, hqk, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl

theorem ak_accmul_op_eval (s : BlockState) (BM DIM : Nat) (hax : 1 < [BM].length.succ)
    (acctile : Tile .real [BM, DIM]) (atile : Tile .real [BM])
    (hacc : s.regs .real [BM, DIM] "acc" = some acctile) (ha : s.regs .real [BM] "alpha" = some atile) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM, DIM] "acc")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile
          (Tile.expandDim ⟨1, hax⟩ atile)) := by
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "alpha")) s
      = some (Tile.expandDim ⟨1, hax⟩ atile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ ha
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl

theorem ak_accadd_op_eval (s : BlockState) (BM BN DIM : Nat)
    (acc1tile : Tile .real [BM, DIM]) (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, DIM])
    (hacc : s.regs .real [BM, DIM] "acc" = some acc1tile)
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hv : s.regs .real [BN, DIM] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) (Op.ref .real [BM, DIM] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) (Op.ref .real [BN, DIM] "v"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc1tile
          (Tile.dot [] ptile vtile)) := by
  have hcastInner : evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p")) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ : Tile .fp16 [BM, BN]) := by
    rw [evalOp_castFloat]; simp [evalOp_ref, hp]
  have hpf16 : evalOp (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) s = some ptile := by
    rw [evalOp_castFloat, hcastInner]
    refine congrArg some ?_; ext i; simp [FloatDType.cast]
  have hpf16' : @evalOp TileDType.real [BM, BN]
      (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) s = some ptile := hpf16
  have hdotN : evalOp
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) (Op.ref .real [BN, DIM] "v")) s
      = some (Tile.dot [] ptile vtile) := by
    rw [evalOp_dot]; simp [hpf16', evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, DIM]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) (Op.ref .real [BN, DIM] "v")) s
      = some (Tile.dot [] ptile vtile) := hdotN
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

theorem ak_li_op_eval (s : BlockState) (BM BN : Nat)
    (ltile atile : Tile .real [BM]) (ptile : Tile .real [BM, BN])
    (hl : s.regs .real [BM] "l_i" = some ltile) (ha : s.regs .real [BM] "alpha" = some atile)
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile atile)
          (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile)) := by
  have hsumN0 : evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile) := by
    rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_eq_bind, Option.bind_some]; rfl
  have hsumN : @evalOp TileDType.real [BM]
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile) := hsumN0
  rw [evalOp_add]
  simp only [evalOp_mul, evalOp_ref, hl, ha, hsumN, Option.bind_eq_bind, Option.bind_some]; rfl


set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
theorem attnLoopBody_steps (B0 : RegionName) (sin : BlockState) (SN : Nat)
    (Kreg Vreg : RegionName) (kbase vbase kcol vrow kcols vrows : Nat)
    (boff smbm : Nat)
    (qtile : Tile .fp16 [32, 64]) (mtile ltile : Tile .real [32])
    (acctile : Tile .real [32, 64]) (ktile : Tile .real [64, 64]) (vtile : Tile .real [64, 64])
    (b1tile : Tile .real [32, 64])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hbo : sin.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hm : sin.regs .nat [32] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin 32 => r.val)))
    (hmax : sin.regs .real [32] "m_i" = some mtile)
    (hl : sin.regs .real [32] "l_i" = some ltile)
    (hacc : sin.regs .real [32, 64] "acc" = some acctile)
    (hq : sin.regs .fp16 [32, 64] "q" = some qtile)
    (hb1 : sin.regs .real [32, 64] "b1" = some b1tile)
    (hKp : sin.regs .blockPtr [64, 64] "K_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Kreg, baseOffset := kbase, parentShape := [64, kcols],
          blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol] }⟩))
    (hVp : sin.regs .blockPtr [64, 64] "V_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Vreg, baseOffset := vbase, parentShape := [vrows, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [vrow, 0] }⟩))
    (hkload : ∀ idx : TileIndex [64, 64],
      ktile.data idx = some (sin.readMem Kreg (kbase + idx.1.val * 1 + (kcol + idx.2.1.val) * 64)))
    (hvload : ∀ idx : TileIndex [64, 64],
      vtile.data idx = some (sin.readMem Vreg (vbase + (vrow + idx.1.val) * 64 + idx.2.1.val * 1)))
    (qoffV : Nat) (hqo : sin.regs .nat [] "q_offset" = some (Tile.scalar qoffV))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (attnLoopBody B0) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .fp16 [32, 64] "q" = some qtile
      ∧ sF.regs .real [32, 64] "b1" = some b1tile
      ∧ sF.regs .nat [] "b_offset" = some (Tile.scalar boff)
      ∧ sF.regs .nat [] "start_m" = some (Tile.scalar smbm)
      ∧ sF.regs .nat [] "q_offset" = some (Tile.scalar qoffV)
      ∧ sF.regs .nat [32] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin 32 => r.val))
      ∧ sF.regs .blockPtr [64, 64] "K_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Kreg, baseOffset := kbase, parentShape := [64, kcols],
              blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol + 64] }⟩)
      ∧ sF.regs .blockPtr [64, 64] "V_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Vreg, baseOffset := vbase, parentShape := [vrows, 64],
              blockShape := [64, 64], strides := [64, 1], offsets := [vrow + 64, 0] }⟩)
      ∧ ∃ (qkT : Tile .real [32, 64]) (rmaxT mnewT alphaT : Tile .real [32])
            (pT : Tile .real [32, 64]),
          (∀ i : Fin 32, ∀ j : Fin 64,
            qkT.data (i, j, PUnit.unit)
              = (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (⟨fun _ : TileIndex [32, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
                    (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [32, 64]) ktile))
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil))
                    (⟨fun idx : TileIndex [32, 1] =>
                      some (sin.readMem B0 (boff + (smbm * 32 + idx.1.val) * 128 + SN / 64))⟩)
                    b1tile)).data (i, j, PUnit.unit))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [32, 64].length) qkT = some rmaxT
          ∧ mnewT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT))
          ∧ sF.regs .real [32] "m_i" = some mnewT
          ∧ sF.regs .real [32, 64] "acc" = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
              (Tile.dot [] pT vtile))
          ∧ sF.regs .real [32] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT)
              (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [32, 64].length) pT)) := by
  set b0T : Tile .real [32, 1] :=
    ⟨fun idx : TileIndex [32, 1] => some (sin.readMem B0 (boff + (smbm * 32 + idx.1.val) * 128 + SN / 64))⟩ with hb0T
  set qkdotT : Tile .real [32, 64] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (⟨fun _ : TileIndex [32, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
      (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [32, 64]) ktile) with hqkdotT
  set qkT : Tile .real [32, 64] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkdotT
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0T b1tile) with hqkT
  have hqkData : ∀ i : Fin 32, ∀ j : Fin 64, qkT.data (i, j, PUnit.unit)
      = (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkdotT
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0T b1tile)).data (i, j, PUnit.unit) := fun _ _ => rfl
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [32, 64].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [32, 64] (⟨1, by simp⟩ : Fin [32, 64].length) from by decide)]⟩
  set mnewT : Tile .real [32] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmnew
  set alphaT : Tile .real [32] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT) with halpha
  set pT : Tile .real [32, 64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT)) with hpT
  unfold attnLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 64] "K_block_ptr") []) .none) sin
        = some ktile from by
      rw [load_blockPtr_K_eval Kreg kbase 64 (kcols) 64 64 1 64 kcol
        (Op.ref .blockPtr [64, 64] "K_block_ptr") sin (by rw [evalOp_ref]; simp [hKp])]
      refine congrArg some ?_; ext idx; rw [hkload idx]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 64] "V_block_ptr") []) .none) _
        = some vtile from by
      rw [load_blockPtr_Q_eval Vreg vbase (vrows) 64 64 64 64 1 vrow
        (Op.ref .blockPtr [64, 64] "V_block_ptr") _ (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hVp])]
      refine congrArg some ?_; ext idx
      simp only [BlockState.setReg_readMem]; rw [hvload idx]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show @evalOp TileDType.fp16 [32, 64] (Op.full [32, 64] (Op.castFloat .real .fp16 (Op.const 0)))
          ((sin.setReg "k" .real [64, 64] ktile).setReg "v" .real [64, 64] vtile)
        = some (⟨fun _ : TileIndex [32, 64] => FloatDType.real.cast FloatDType.fp16 (some 0)⟩ : Tile .fp16 [32, 64]) from by
      conv_lhs => unfold evalOp
      conv_lhs => unfold evalOp
      conv_lhs => unfold evalOp
      simp only [Option.bind_eq_bind]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "qk"))
          (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [32, 64] "q"))
            (Op.ref .real [64, 64] "k"))) _ = some qkdotT from by
      rw [qk_dot_op_eval _ qtile ktile (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, hq]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])]))]
  rw [stepStmts.cons_some
    (stepStmt_assign_eq_some (v := b0T)
      (by
        rw [load_b0_eval
          ((((sin.setReg "k" .real [64, 64] ktile).setReg "v" .real [64, 64] vtile).setReg "qk"
              .fp16 [32, 64] (⟨fun _ : TileIndex [32, 64] => FloatDType.real.cast FloatDType.fp16 (some 0)⟩)).setReg
            "qk" .real [32, 64] qkdotT)
          B0 32 128 smbm boff SN attnLoopBody._proof_1
          (by simp [BlockState.setReg_ne_name, hbo]) (by simp [BlockState.setReg_ne_name, hsm])
          (by simp [BlockState.setReg_ne_name, hsn]) (by simp [BlockState.setReg_ne_name, hm])]
        refine congrArg some ?_; ext idx
        simp only [hb0T, BlockState.setReg_readMem] : _ = some b0T))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [32, 64] "qk")
          (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.ref .real [32, 1] "b0") (Op.ref .real [32, 64] "b1"))) _
        = some qkT from by
      rw [qk_bias_op_eval _ qkdotT b0T b1tile (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hb1])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_mnew_op_eval _ 32 64 mtile qkT rmaxT
      (by simp [BlockState.setReg_ne_name, hmax]) (by rw [BlockState.setReg_same]) hrm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_alpha_op_eval _ 32 mtile mnewT
      (by simp [BlockState.setReg_ne_name, hmax]) (by simp [BlockState.setReg_same, hmnew])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_p_op_eval _ 32 64 (by simp) qkT mnewT
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name, hmnew])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_accmul_op_eval _ 32 64 (by simp) acctile alphaT
      (by simp [BlockState.setReg_ne_name, hacc]) (by simp [BlockState.setReg_same, halpha])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_accadd_op_eval _ 32 64 64
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
      pT vtile
      (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hpT])
      (by simp [BlockState.setReg_ne_name])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_li_op_eval _ 32 64 ltile alphaT pT
      (by simp [BlockState.setReg_ne_name, hl]) (by simp [BlockState.setReg_ne_name, halpha])
      (by simp [BlockState.setReg_ne_name, hpT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_ref_op_eval _ "m_i_new" mnewT (by rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (advance_col_eval _ Kreg kbase 64 (kcols) 64 64 1 64 kcol 64 "K_block_ptr"
      (by simp [BlockState.setReg_ne_name, hKp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (advance_row_eval _ Vreg vbase (vrows) 64 64 64 64 1 vrow 64 "V_block_ptr"
      (by simp [BlockState.setReg_ne_name, hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, qkT, rmaxT, mnewT, alphaT, pT,
    hqkData, hrm, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef, hundef]
  · simp [BlockState.setReg_ne_name, hq]
  · simp [BlockState.setReg_ne_name, hb1]
  · simp [BlockState.setReg_ne_name, hbo]
  · simp [BlockState.setReg_ne_name, hsm]
  · simp [BlockState.setReg_ne_name, hqo]
  · simp [BlockState.setReg_ne_name, hm]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]


/-! ### preLoop — full 19-statement prologue → `alignedInvariant … 0` -/

def attnPreLoopTail (B0 : RegionName) : List Stmt :=
  [ Stmt.assign .real [] "qk_scale" (Op.mul .real Broadcast.nil (Op.const 1.0) (Op.const 1.44269504)),
    Stmt.assign .real [32, 64] "q"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [32, 64] "Q_block_ptr") []) .none),
    Stmt.assign .fp16 [32, 64] "q"
      (Op.castFloat .real .fp16 (Op.mul .real Broadcast.scalarR (Op.ref .real [32, 64] "q") (Op.ref .real [] "qk_scale"))),
    Stmt.assign .nat [] "lo" (Op.constNat 0),
    Stmt.assign .nat [] "hi" (Op.constNat 128),
    Stmt.assign .nat [32] "b_ptr_offsets_m" (Op.arange 32),
    Stmt.assign .nat [] "b_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 8192)),
    Stmt.assign .nat [64] "b_ptr_offsets_n_1"
      (Op.add .nat Broadcast.scalarR
        (Op.mod .nat Broadcast.scalarR (Op.arange 64) (Op.constNat 64)) (Op.constNat 64)),
    Stmt.assign .real [32, 64] "b1"
      (Op.load .real (.region B0
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 32))
                (Op.ref .nat [32] "b_ptr_offsets_m")) (Op.constNat 128))))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "b_ptr_offsets_n_1"))))
        .none) ]

theorem attnPreLoopTail_check (Q K V B0 Out : RegionName) :
    ((attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body.take 19).drop 10 = attnPreLoopTail B0 := by
  rfl

theorem ak_qscale_op_eval (s : BlockState) (qtile : Tile .real [32, 64]) (sc : ℝ)
    (hq : s.regs .real [32, 64] "q" = some qtile)
    (hqs : s.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) :
    evalOp (Op.castFloat .real .fp16
        (Op.mul .real Broadcast.scalarR (Op.ref .real [32, 64] "q") (Op.ref .real [] "qk_scale"))) s
      = some (⟨fun idx : TileIndex [32, 64] =>
          FloatDType.real.cast FloatDType.fp16 ((qtile.data idx).bind (fun x => some (x * sc)))⟩ : Tile .fp16 [32, 64]) := by
  have hmul0 : evalOp
        (Op.mul .real Broadcast.scalarR (Op.ref .real [32, 64] "q") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := by
    rw [evalOp_mul]; simp only [evalOp_ref, hq, hqs, Option.bind_eq_bind, Option.bind_some]
  have hmul : @evalOp FloatDType.real.toTileDType [32, 64]
        (Op.mul .real Broadcast.scalarR (Op.ref .real [32, 64] "q") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := hmul0
  rw [evalOp_castFloat, hmul]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

theorem evalOp_mod' {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

theorem ak_bn1_op_eval (s : BlockState) :
    evalOp (Op.add .nat Broadcast.scalarR
        (Op.mod .nat Broadcast.scalarR (Op.arange 64) (Op.constNat 64)) (Op.constNat 64)) s
      = some (Tile.vec (fun jL : Fin 64 => jL.val % 64 + 64)) := by
  rw [evalOp_add, evalOp_mod']
  simp only [evalOp_arange, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, IntegralDType.nat_mod]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
theorem preLoop (Q K V B0 Out : RegionName) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts ((attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body.take 19) s = some s0
      ∧ s0.regs .nat [] "lo" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "hi" = some (Tile.scalar 128)
      ∧ alignedInvariant s Q K V B0 Out 1.0
          (s.pids 1 * 8192) (s.pids 1 * 8192) (s.pids 1 * 8192)
          32 64 64 0 64 128 1 64 64 1 2 (s.pids 0) 0 s0 := by
  rw [show (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body.take 19
      = (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
          8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
          FloatDType.fp16).toAlgKernel.body.take 10 ++ attnPreLoopTail B0 from by
    conv_lhs => rw [← List.take_append_drop 10
      ((attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
          8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
          FloatDType.fp16).toAlgKernel.body.take 19)]
    rw [List.take_take, attnPreLoopTail_check Q K V B0 Out]
    norm_num]
  obtain ⟨s10, hpre, hpids, hstartm, hoffhz, hqoff, hkvoff, hQp, hKp, hVp, hmi, hli, hacc, hundef10, hmem10⟩ :=
    preLoop_prefix Q K V B0 Out 1.0 8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64 FloatDType.fp16 s
  rw [stepStmts.append_some hpre]
  unfold attnPreLoopTail
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.nil (Op.const 1.0) (Op.const 1.44269504)) s10
        = some (Tile.scalar (some (1.0 * 1.44269504))) from by
      rw [evalOp_mul]; simp only [evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [32, 64] "Q_block_ptr") []) .none)
          (s10.setReg "qk_scale" .real [] (Tile.scalar (some (1.0 * 1.44269504))))
        = some (⟨fun idx : TileIndex [32, 64] =>
            some ((s10.setReg "qk_scale" .real [] (Tile.scalar (some (1.0 * 1.44269504)))).readMem Q
              (s.pids 1 * 8192 + (s.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1))⟩
            : Tile .real [32, 64]) from
      load_blockPtr_Q_eval Q (s.pids 1 * 8192) 128 64 32 64 64 1 (s.pids 0 * 32)
        (Op.ref .blockPtr [32, 64] "Q_block_ptr") _
        (by rw [evalOp_ref]; rw [show (s10.setReg "qk_scale" .real [] (Tile.scalar (some (1.0 * 1.44269504)))).regs .blockPtr [32, 64] "Q_block_ptr" = s10.regs .blockPtr [32, 64] "Q_block_ptr" from by simp [BlockState.setReg_ne_name]]; exact hQp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.castFloat .real .fp16 (Op.mul .real Broadcast.scalarR (Op.ref .real [32, 64] "q") (Op.ref .real [] "qk_scale"))) _
        = some (⟨fun idx : TileIndex [32, 64] =>
            FloatDType.real.cast FloatDType.fp16
              (some (1.0 * 1.44269504 * qRaw s Q (s.pids 1 * 8192) 32 64 (s.pids 0) idx))⟩ : Tile .fp16 [32, 64]) from by
      rw [ak_qscale_op_eval _
        (⟨fun idx : TileIndex [32, 64] =>
          some ((s10.setReg "qk_scale" .real [] (Tile.scalar (some (1.0 * 1.44269504)))).readMem Q
            (s.pids 1 * 8192 + (s.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1))⟩ : Tile .real [32, 64])
        (1.0 * 1.44269504)
        (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
      refine congrArg some ?_; ext idx
      simp only [Option.bind, Option.map, qRaw]
      have hrm : (s10.setReg "qk_scale" .real [] (Tile.scalar (some (1.0 * 1.44269504)))).readMem Q
            (s.pids 1 * 8192 + (s.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1)
          = s.readMem Q (s.pids 1 * 8192 + (s.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val) := by
        rw [BlockState.setReg_readMem]
        unfold BlockState.readMem; rw [hmem10]; congr 1; ring
      rw [hrm]; exact congrArg _ (congrArg _ (mul_comm _ _))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 128 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange 32 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (evalOp_mul_ref_const _ "off_hz" (s.pids 1) 8192
      (by simp [BlockState.setReg_ne_name, hoffhz])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ak_bn1_op_eval _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_b1_eval _ B0 32 64 128 64 (s.pids 0) (s.pids 1 * 8192)
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by simp [BlockState.setReg_ne_name, hstartm])
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]; exact hpids
  · rfl
  · norm_num
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
    rw [hmi]; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
    rw [hli]; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
    rw [hacc]; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hKp]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hVp]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, FloatDType.toTileDType_fp16]
  · rw [BlockState.setReg_same]
    refine congrArg some ?_; ext idx
    simp only [b1Val, BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem10]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]; exact hstartm
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]; exact hqoff
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
  · intro rg o
    show s10.undef rg o = 0
    rw [hundef10]; exact hundef rg o
  · show s10.mem = s.mem
    exact hmem10


/-! ### `attn_qk_cell`: the body's `qk` cell = `fscore` at the global key -/

set_option maxHeartbeats 1600000 in
theorem attn_qk_cell (s0 : BlockState) (Q K B0 : RegionName) (sm_scale : ℝ)
    (q_offset kv_offset b_offset : Nat) (nB c bcol : Nat) (hc : c < nB)
    (qtile : Tile .fp16 [32, 64]) (ktile : Tile .real [64, 64])
    (b1tile : Tile .real [32, 64])
    (hbcol : bcol = c)
    (hq : qtile = ⟨fun idx : TileIndex [32, 64] =>
        FloatDType.real.cast FloatDType.fp16
          (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset 32 64 (s0.pids 0) idx))⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (kv_offset + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hb1 : b1tile = ⟨fun idx : TileIndex [32, 64] =>
        some (b1Val s0 B0 b_offset 32 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩)
    (i : Fin 32) (jL : Fin 64) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [32, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
          (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [32, 64]) ktile))
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (⟨fun idx : TileIndex [32, 1] =>
            some (s0.readMem B0 (b_offset + (s0.pids 0 * 32 + idx.1.val) * 128 + bcol))⟩)
          b1tile)).data (i, jL, PUnit.unit)
      = some (fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
          32 64 64 (64 * nB) 64 128 (s0.pids 0) i (gkey 64 nB c hc jL)) := by
  have hdot : (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [32, 64]) ktile).data (i, jL, PUnit.unit)
      = some (sm_scale * 1.44269504 * Finset.univ.sum (fun e : Fin 64 =>
          qRaw s0 Q q_offset 32 64 (s0.pids 0) (i, e, PUnit.unit)
            * kFlat s0 K kv_offset 64 (64 * nB) e (gkey 64 nB c hc jL))) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·)
            ((⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [32, 64]).data (i, e, PUnit.unit))
            (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
          (fun e => (some (sm_scale * 1.44269504 * (qRaw s0 Q q_offset 32 64 (s0.pids 0) (i, e, PUnit.unit)
              * kFlat s0 K kv_offset 64 (64 * nB) e (gkey 64 nB c hc jL))) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          simp only [hq, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
            FloatDType.real_toWithBot, Option.map₂]
          rw [hk (e, jL, PUnit.unit)]
          simp only [Option.bind, Option.map]
          refine congrArg some ?_
          simp only [qRaw, kFlat, gkey]
          rw [show kv_offset + e.val * 1 + (c * 64 + jL.val) * 64
                = kv_offset + e.val + (c * 64 + jL.val) * 64 from by ring]
          ring)]
    rw [withBot_sum_some]
    refine congrArg some ?_
    rw [Finset.mul_sum]
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar,
    NumericDType.add, NumericDType.mul, hb1, hdot, b1Val, b0Val]
  simp only [FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
    WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  simp only [fscore, qRaw, kFlat, b0Val, b1Val, gkey, hbcol]
  have hjLlt := jL.isLt
  rw [show (c * 64 + jL.val) / 64 = c from by
    rw [Nat.mul_comm c 64, Nat.mul_add_div (by norm_num : (0:Nat) < 64), Nat.div_eq_of_lt jL.isLt]
    omega]
  rw [show (c * 64 + jL.val) % 64 = jL.val from by
    rw [Nat.mul_comm c 64, Nat.mul_add_mod, Nat.mod_eq_of_lt jL.isLt]]
  ring


/-! ### `attn_step`: the loop body advances the invariant by one block -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
theorem attn_step (Q K V B0 Out : RegionName) (s0 : BlockState)
    (nB : Nat) (i : Nat) (s : BlockState) (hilt : i < 64 * nB)
    (hinv : alignedInvariant s0 Q K V B0 Out 1.0
        (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
        32 64 64 0 64 128 1 64 64 1 nB (s0.pids 0) i s) :
    ∃ s', stepStmts (attnLoopBody B0) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ alignedInvariant s0 Q K V B0 Out 1.0
          (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
          32 64 64 0 64 128 1 64 64 1 nB (s0.pids 0) (i + 64) s' := by
  have hBN : (0:Nat) < 64 := by norm_num
  have hc : i / 64 < nB := (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + 64) / 64 = i / 64 + 1 := Nat.add_div_right i hBN
  simp only [alignedInvariant] at hinv
  obtain ⟨hpids, hieq, hcle, hmi, hli, hacc, hKp, hVp, hq, hb1, hbo, hsm, hqo, hm, hundef, hmem⟩ := hinv
  set sc := fscore s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
    32 64 64 (64 * nB) 64 128 (s0.pids 0) with hscdef
  set V' := vFlat s0 V (s0.pids 1 * 8192) 64 (64 * nB) with hVpdef
  have hrmem : ∀ (R : RegionName) (o : Nat),
      (s.setReg "start_n" .nat [] (Tile.scalar i)).readMem R o = s0.readMem R o := by
    intro R o; simp only [BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hmi' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [32] "m_i"
      = some (⟨fun r : TileIndex [32] => mPg 64 nB sc r.1 (i / 64)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi
  have hli' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [32] "l_i"
      = some (⟨fun r : TileIndex [32] => ((lPgK sc r.1 (i / 64) : ℝ) : WithBot ℝ)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli
  have hacc' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [32, 64] "acc"
      = some (⟨fun idx : TileIndex [32, 64] => ((oPg sc V' idx.1 idx.2.1 (i / 64) : ℝ) : WithBot ℝ)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc
  have hKp' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .blockPtr [64, 64] "K_block_ptr"
      = some (⟨fun _ : TileIndex [64, 64] =>
        { region := K, baseOffset := s0.pids 1 * 8192, parentShape := [64, 64 * nB + 0],
          blockShape := [64, 64], strides := [1, 64], offsets := [0, i / 64 * 64] }⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hKp]
  have hVp' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .blockPtr [64, 64] "V_block_ptr"
      = some (⟨fun _ : TileIndex [64, 64] =>
        { region := V, baseOffset := s0.pids 1 * 8192, parentShape := [64 * nB + 0, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [i / 64 * 64, 0] }⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hVp]
  have hq' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .fp16 [32, 64] "q"
      = some (⟨fun idx : TileIndex [32, 64] =>
        FloatDType.real.cast FloatDType.fp16
          (some (1.0 * 1.44269504 * qRaw s0 Q (s0.pids 1 * 8192) 32 64 (s0.pids 0) idx))⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq
  have hb1' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [32, 64] "b1"
      = some (⟨fun idx : TileIndex [32, 64] =>
        some (b1Val s0 B0 (s0.pids 1 * 8192) 32 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hb1
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hqF, hb1F, hboF, hsmF, hqoF, hmF, hKpF, hVpF,
      qkT, rmaxT, mnewT, alphaT, pT, hqkData, hrm, hmnewd, halphad, hpTd, hm_iF, haccF, hl_iF⟩ :=
    attnLoopBody_steps B0 (s.setReg "start_n" .nat [] (Tile.scalar i)) i
      K V (s0.pids 1 * 8192) (s0.pids 1 * 8192) (i / 64 * 64) (i / 64 * 64) (64 * nB + 0) (64 * nB + 0)
      (s0.pids 1 * 8192) (s0.pids 0)
      (⟨fun idx : TileIndex [32, 64] =>
        FloatDType.real.cast FloatDType.fp16
          (some (1.0 * 1.44269504 * qRaw s0 Q (s0.pids 1 * 8192) 32 64 (s0.pids 0) idx))⟩)
      (⟨fun r : TileIndex [32] => mPg 64 nB sc r.1 (i / 64)⟩)
      (⟨fun r : TileIndex [32] => ((lPgK sc r.1 (i / 64) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [32, 64] => ((oPg sc V' idx.1 idx.2.1 (i / 64) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [64, 64] =>
        some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (i / 64 * 64 + idx.2.1.val) * 64))⟩)
      (⟨fun idx : TileIndex [64, 64] =>
        some (s0.readMem V (s0.pids 1 * 8192 + (i / 64 * 64 + idx.1.val) * 64 + idx.2.1.val * 1))⟩)
      (⟨fun idx : TileIndex [32, 64] =>
        some (b1Val s0 B0 (s0.pids 1 * 8192) 32 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩)
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hbo)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsm)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hm)
      hmi' hli' hacc' hq' hb1' hKp' hVp'
      (fun idx => congrArg some (hrmem K _).symm) (fun idx => congrArg some (hrmem V _).symm)
      (s0.pids 1 * 8192) (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqo)
      (by intro rg o; simp [BlockState.setReg_undef, hundef])
  refine ⟨sF, hchain, ?_⟩
  have hqk : ∀ (r : Fin 32) (jL : Fin 64),
      qkT.data (r, jL, PUnit.unit) = some (sc r (gkey 64 nB (i / 64) hc jL)) := by
    intro r jL
    rw [hqkData r jL]
    have := attn_qk_cell s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
      nB (i / 64) (i / 64) hc
      (⟨fun idx : TileIndex [32, 64] =>
        FloatDType.real.cast FloatDType.fp16
          (some (1.0 * 1.44269504 * qRaw s0 Q (s0.pids 1 * 8192) 32 64 (s0.pids 0) idx))⟩)
      (⟨fun idx : TileIndex [64, 64] =>
        some (s0.readMem K (s0.pids 1 * 8192 + idx.1.val * 1 + (i / 64 * 64 + idx.2.1.val) * 64))⟩)
      (⟨fun idx : TileIndex [32, 64] =>
        some (b1Val s0 B0 (s0.pids 1 * 8192) 32 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩)
      rfl rfl (fun idx => rfl) rfl r jL
    simp only [hscdef]
    simp only [hrmem] at this ⊢
    exact this
  have hmij_eq : ∀ idx : TileIndex [32], mnewT.data idx
      = mPg 64 nB sc idx.1 (i / 64 + 1) := by
    intro idx
    rw [hmnewd]
    refine mijg_eq 64 nB (i / 64) hc sc _ rmaxT idx.1 ?_ ?_
    · rfl
    · exact reduceMaxDrop_data_row 32 64 hBN _ rmaxT hrm idx.1 _ (fun jL => hqk idx.1 jL)
  simp only [alignedInvariant]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF]; exact hpids
  · rw [hc1, Nat.add_one_mul, ← hieq]
  · rw [hc1]; omega
  · rw [hm_iF, hc1]; refine congrArg some ?_; ext idx; exact hmij_eq idx
  · rw [hl_iF, hc1]; refine congrArg some ?_; ext idx
    rw [halphad, hpTd]
    exact lig_eq 64 nB (i / 64) hBN hc sc _ mnewT _ qkT idx.1 rfl (hmij_eq idx) rfl (fun jL => hqk idx.1 jL)
  · rw [haccF, hc1]; refine congrArg some ?_
    ext idx
    have hv : ∀ jL : Fin 64,
        (⟨fun idx : TileIndex [64, 64] =>
          some (s0.readMem V (s0.pids 1 * 8192 + (i / 64 * 64 + idx.1.val) * 64 + idx.2.1.val * 1))⟩
          : Tile .real [64, 64]).data (jL, idx.2.1, PUnit.unit)
          = some (V' (gkey 64 nB (i / 64) hc jL, idx.2.1, PUnit.unit)) := by
      intro jL
      refine congrArg some ?_
      simp only [hVpdef, vFlat, gkey]
      rw [show s0.pids 1 * 8192 + (i / 64 * 64 + jL.val) * 64 + idx.2.1.val * 1
            = s0.pids 1 * 8192 + (i / 64 * 64 + jL.val) * 64 + idx.2.1.val from by ring]
    refine accg_eq 64 nB (i / 64) hc sc V' _ alphaT pT _ idx.1 idx.2.1 rfl ?_ ?_ hv
    · rw [halphad]
      exact alphag_eq 64 nB (i / 64) sc _ mnewT idx.1 rfl (hmij_eq (idx.1, PUnit.unit))
    · intro jL
      rw [hpTd]
      refine exp2_some (fun a b => sc a (gkey 64 nB (i / 64) hc b) - mRg sc a (i / 64 + 1)) _ idx.1 jL ?_
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk idx.1 jL,
        hmij_eq (idx.1, PUnit.unit),
        mPg_eq_coe sc hBN idx.1 (i / 64 + 1) (Nat.le_add_left 1 (i / 64)) hc,
        NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]
  · rw [hKpF, hc1, show i / 64 * 64 + 64 = (i / 64 + 1) * 64 from by ring]
  · rw [hVpF, hc1, show i / 64 * 64 + 64 = (i / 64 + 1) * 64 from by ring]
  · rw [hqF]
  · rw [hb1F]
  · rw [hboF]
  · rw [hsmF]
  · rw [hqoF]
  · rw [hmF]
  · exact hundefF
  · rw [hmemF]; exact hmem


/-! ### Bridge: `attnGenScore fscore vFlat` = banked `alignedClosedForm` -/

set_option maxRecDepth 8000 in
/-- The genuine `attnGenScore` of the aligned per-key score equals the banked
closed form `alignedClosedForm` (with `log2e = 1.44269504`), at the Python test
shape. Both unfold to the same `exp(log 2 · (sm_scale·1.44269504·raw + bias))`
batch base-2 softmax over the loaded tiles. -/
theorem attnGenScore_eq_alignedClosedForm (s0 : BlockState) (Q K V B0 : RegionName)
    (i : Fin 32) (d : Fin 64) :
    attnGenScore
        (fscore s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
          32 64 64 (64 * 2) 64 128 (s0.pids 0))
        (vFlat s0 V (s0.pids 1 * 8192) 64 (64 * 2)) (i, d, PUnit.unit)
      = alignedClosedForm s0 Q K V B0 1.0 8192 8192 128 128 64 64 32 64 (i, d, PUnit.unit) := by
  have hw : ∀ j : Fin (64 * 2),
      pow2 (fscore s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
              32 64 64 (64 * 2) 64 128 (s0.pids 0) i j)
        = Real.exp (Real.log 2 * (1.0 * log2e *
            Finset.univ.sum (fun e : Fin 64 =>
              alignedQTile s0 Q 8192 64 32 (i, e, PUnit.unit)
                * alignedKTile s0 K 8192 64 128 (j, e, PUnit.unit))
            + alignedBias s0 B0 8192 128 64 32 64 128 i j)) := by
    intro j
    rw [pow2]
    refine congrArg _ (congrArg _ ?_)
    rw [fscore, log2e, alignedBias, mIndex]
    refine congrArg₂ _ ?_ ?_
    · refine congrArg₂ _ rfl (Finset.sum_congr rfl (fun e _ => ?_))
      simp only [qRaw, kFlat, alignedQTile, alignedKTile, mIndex]
      ring_nf
    · simp only [b0Val, b1Val, mIndex]
      rw [Nat.add_assoc (s0.pids 1 * 8192 + (s0.pids 0 * 32 + i.val) * 128) (j.val % 64 % 64) 64]
  simp only [attnGenScore, alignedClosedForm, attentionRealBase2ScalarScaleBias]
  rw [show (Finset.univ.sum (fun j : Fin (64 * 2) => pow2 (fscore s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192) 32 64 64 (64 * 2) 64 128 (s0.pids 0) i j)))
        = Finset.univ.sum (fun j : Fin (64 * 2) => Real.exp (Real.log 2 * (1.0 * log2e * Finset.univ.sum (fun e : Fin 64 => alignedQTile s0 Q 8192 64 32 (i, e, PUnit.unit) * alignedKTile s0 K 8192 64 128 (j, e, PUnit.unit)) + alignedBias s0 B0 8192 128 64 32 64 128 i j)))
      from Finset.sum_congr rfl (fun j _ => hw j)]
  rw [show (Finset.univ.sum (fun j : Fin (64 * 2) => pow2 (fscore s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192) 32 64 64 (64 * 2) 64 128 (s0.pids 0) i j) * vFlat s0 V (s0.pids 1 * 8192) 64 (64 * 2) (j, d, PUnit.unit)))
        = Finset.univ.sum (fun j : Fin (64 * 2) => Real.exp (Real.log 2 * (1.0 * log2e * Finset.univ.sum (fun e : Fin 64 => alignedQTile s0 Q 8192 64 32 (i, e, PUnit.unit) * alignedKTile s0 K 8192 64 128 (j, e, PUnit.unit)) + alignedBias s0 B0 8192 128 64 32 64 128 i j)) * alignedVTile s0 V 8192 64 128 (j, d, PUnit.unit))
      from Finset.sum_congr rfl (fun j _ => by rw [hw j]; refine congrArg₂ _ rfl ?_; simp only [vFlat, alignedVTile, mIndex])]

/-! ### fp16 block-store readback lemmas -/

theorem foldl_writeMemTyped_fp16_preserves' {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, offsetFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k)) s).mem region o)
      = s.mem region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
    have htl : ∀ k ∈ tl, offsetFn k ≠ o := fun k hk => h k (List.mem_cons_of_mem hd hk)
    rw [List.foldl_cons, ih _ htl]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    change (if region = region ∧ o = offsetFn hd then _ else s.mem region o) = s.mem region o
    rw [if_neg]; rintro ⟨_, h_eq⟩; exact hhd h_eq.symm

theorem scatter_memcell_fp16_nd' {region : RegionName} {shape : TileShape}
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
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := by
    intro k hk heq; have hki : k = i := h_inj heq; subst hki; exact hi_notin_l2 hk
  rw [foldl_writeMemTyped_fp16_preserves' offsetFn valueFn (offsetFn i) l₂ _ h_l2_not_in]
  unfold BlockState.writeMemTyped BlockState.writeMemAs
  change
    (if region = region ∧ offsetFn i = offsetFn i then
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
    else _) = _
  rw [if_pos ⟨rfl, rfl⟩]

/-! ### `attn_postLoop`: `acc /= l_i` + block store = `alignedClosedForm` -/

set_option maxHeartbeats 1600000 in
theorem attn_postLoop (Q K V B0 Out : RegionName) (s0 : BlockState)
    (nB : Nat) (hnB : 1 ≤ nB) (st : BlockState)
    (hinv : alignedInvariant s0 Q K V B0 Out 1.0
        (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
        32 64 64 0 64 128 1 64 64 1 nB (s0.pids 0) (64 * nB) st) :
    ∃ sfin, stepStmts (attnPostLoop Out) st = some sfin
      ∧ ∀ idx : TileIndex [32, 64],
          sfin.mem Out (s0.pids 1 * 8192 + (s0.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (attnGenScore
                  (fscore s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
                    32 64 64 (64 * nB) 64 128 (s0.pids 0))
                  (vFlat s0 V (s0.pids 1 * 8192) 64 (64 * nB)) idx))) := by
  have hBN : (0:Nat) < 64 := by norm_num
  have hcnB : (64 * nB) / 64 = nB := by rw [Nat.mul_comm, Nat.mul_div_cancel _ hBN]
  simp only [alignedInvariant, hcnB] at hinv
  obtain ⟨hpids, hieq, hcle, hmi, hli, hacc, hKp, hVp, hq, hb1, hbo, hsm, hqo, hm, hundef, hmem⟩ := hinv
  set sc := fscore s0 Q K B0 1.0 (s0.pids 1 * 8192) (s0.pids 1 * 8192) (s0.pids 1 * 8192)
    32 64 64 (64 * nB) 64 128 (s0.pids 0) with hscdef
  set V' := vFlat s0 V (s0.pids 1 * 8192) 64 (64 * nB) with hVpdef
  set acc' : Tile .real [32, 64] :=
    ⟨fun idx : TileIndex [32, 64] => ((oPg sc V' idx.1 idx.2.1 nB / lPgK sc idx.1 nB : ℝ) : WithBot ℝ)⟩
    with hacc'def
  unfold attnPostLoop
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [32, 64] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [32] "l_i"))) st = some acc' from by
      have hexp : @evalOp TileDType.real [32, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [32] "l_i")) st
          = some (Tile.expandDim ⟨1, by simp⟩
              (⟨fun r : TileIndex [32] => ((lPgK sc r.1 nB : ℝ) : WithBot ℝ)⟩ : Tile .real [32])) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      rw [evalOp_div]
      simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
        Option.map₂, Option.bind, Option.map, hacc'def]
      rfl))]
  set st1 := st.setReg "acc" .real [32, 64] acc' with hst1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_rowcol_eval Out (Op.ref .nat [] "q_offset") [128, 64] [32, 64] [64, 1]
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 32)) st1
      (s0.pids 1 * 8192) (s0.pids 0 * 32)
      (by rw [evalOp_ref]; simp only [hst1, BlockState.setReg_ne_name]; exact hqo)
      (evalOp_mul_ref_const st1 "start_m" (s0.pids 0) 32
        (by simp only [hst1, BlockState.setReg_ne_name]; exact hsm))))]
  set st2 := st1.setReg "O_block_ptr" .blockPtr [32, 64]
      ⟨fun _ : TileIndex [32, 64] =>
        { region := Out, baseOffset := s0.pids 1 * 8192, parentShape := [128, 64],
          blockShape := [32, 64], strides := [64, 1], offsets := [s0.pids 0 * 32, 0] }⟩
    with hst2
  set oValFn : TileIndex [32, 64] → TileCarrier TileDType.fp16 :=
    fun idx => FloatDType.real.cast FloatDType.fp16
      (some (oPg sc V' idx.1 idx.2.1 nB / lPgK sc idx.1 nB)) with hoValFn
  set oOffFn : TileIndex [32, 64] → Nat :=
    fun idx => s0.pids 1 * 8192 + (s0.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1 with hoOffFn
  have hOpref : @evalOp TileDType.blockPtr [32, 64] (Op.ref .blockPtr [32, 64] "O_block_ptr") st2
      = some (⟨fun _ : TileIndex [32, 64] =>
          { region := Out, baseOffset := s0.pids 1 * 8192, parentShape := [128, 64],
            blockShape := [32, 64], strides := [64, 1], offsets := [s0.pids 0 * 32, 0] }⟩
          : Tile .blockPtr [32, 64]) := by rw [evalOp_ref, hst2, BlockState.setReg_same]
  have haccref : @evalOp TileDType.real [32, 64] (Op.ref .real [32, 64] "acc") st2 = some acc' := by
    rw [evalOp_ref, hst2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hst1,
      BlockState.setReg_same]
  have hval : evalOp (Op.castFloat .real .fp16 (Op.ref .real [32, 64] "acc")) st2
        = some (⟨oValFn⟩ : Tile .fp16 [32, 64]) := by
    rw [evalOp_castFloat]; erw [haccref]; rfl
  have hstore : stepStmt (Stmt.store .fp16 [32, 64]
      (MemAccess.blockPtr (Op.ref .blockPtr [32, 64] "O_block_ptr") [])
      (Op.castFloat .real .fp16 (Op.ref .real [32, 64] "acc")) MaskOpt.none) st2
      = some ((TileShape.allIndices [32, 64]).foldl
          (fun acc idx => acc.writeMemTyped .fp16 Out (oOffFn idx) (oValFn idx)) st2) := by
    unfold stepStmt
    erw [hval]
    simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
    erw [hOpref]
    simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ st2 ?_
    intro acc idx _
    simp only [TileShape.blockPtr_inBounds_nil_index, Bool.and_true, Bool.true_and,
      TileShape.blockPtr_address_2d_row_offset_index, hoOffFn, if_true]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinjO : Function.Injective oOffFn := by
    rintro ⟨⟨a, ha⟩, ⟨b, hb⟩, _⟩ ⟨⟨c, hc⟩, ⟨d, hd⟩, _⟩ heq
    simp only [hoOffFn] at heq
    have hac : a = c ∧ b = d := by
      set P := s0.pids 0 * 32 with hP
      omega
    obtain ⟨rfl, rfl⟩ := hac
    rfl
  rw [show s0.pids 1 * 8192 + (s0.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1 = oOffFn idx from rfl]
  rw [scatter_memcell_fp16_nd' (region := Out) st2 oOffFn oValFn hinjO idx]
  refine congrArg (MemCell.of .fp16) ?_
  rw [hoValFn]
  simp only [FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue, FloatDType.ofWithBot,
    FloatDType.toWithBot, WithBot.unbotD_coe]
  refine congrArg (FloatDType.real.cast FloatDType.fp16) ?_
  refine congrArg some ?_
  rw [lPgK_eq_lPg sc idx.1 nB hnB le_rfl]
  exact closed_form_g sc V' hBN hnB idx.1 idx.2.1


/-! ### Whole-kernel exec assembly + genuine closed-form correctness -/

set_option maxHeartbeats 1600000 in
theorem aligned_exec (Q K V B0 Out : RegionName) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body s = some sF
      ∧ ∀ idx : TileIndex [32, 64],
          sF.mem Out (s.pids 1 * 8192 + (s.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (alignedClosedForm s Q K V B0 1.0 8192 8192 128 128 64 64 32 64 idx))) := by
  rw [← List.take_append_drop 19 (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body]
  obtain ⟨s0, hpre, hlo, hhi, hinv0⟩ := preLoop Q K V B0 Out s hundef
  rw [stepStmts.append_some hpre]
  rw [attnLoopBody_check Q K V B0 Out, show (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out 1.0
        8192 64 1 8192 64 1 8192 64 1 8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16).toAlgKernel.body.drop 20 = attnPostLoop Out from attnPostLoop_check Q K V B0 Out]
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n")
      (startOp := Op.ref .nat [] "lo") (stopOp := Op.ref .nat [] "hi")
      (stepOp := Op.constNat 64)
      (P := fun i st => alignedInvariant s Q K V B0 Out 1.0
        (s.pids 1 * 8192) (s.pids 1 * 8192) (s.pids 1 * 8192)
        32 64 64 0 64 128 1 64 64 1 2 (s.pids 0) i st)
      (s_init := s0)
      (by rw [evalOp_ref, hlo])
      (by rw [evalOp_ref, hhi])
      (by rw [evalOp_constNat])
      (by norm_num)
      hinv0
      (fun i st hi hP => attn_step Q K V B0 Out s 2 i st (by omega) hP)
  rw [stepStmts.cons_some hloop]
  have hfinal : final = 64 * 2 := by
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    omega
  subst hfinal
  obtain ⟨sF, hpostStep, hO⟩ := attn_postLoop Q K V B0 Out s 2 (by norm_num) sL hinvL
  refine ⟨sF, hpostStep, ?_⟩
  rintro ⟨a, b, u⟩
  cases u
  rw [hO (a, b, PUnit.unit), attnGenScore_eq_alignedClosedForm s Q K V B0 a b]

set_option maxHeartbeats 1600000 in
/-- **Genuine closed-form `Out`-store correctness for `attention_kernel_aligned`.**
Every output lane realizes the banked `alignedClosedForm` (base-2 streaming
softmax of the loaded Q/K/V tiles under the scalar score scale `sm_scale·log2(e)`
and the fused `rel_h + rel_w` bias). NOT a self-referential readback. -/
theorem aligned_genuine_output_compute_correct
    (Q K V B0 Out : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out
        1.0 8192 64 1 8192 64 1 8192 64 1 8192 64 1
        8192 128 2 4 128 0 64 128 64 32 64
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [32, 64] =>
        some (Out, surfaceOutOffset s 8192 64 1 32 idx))
      (expected := fun idx : TileIndex [32, 64] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (alignedClosedForm s Q K V B0 1.0 8192 8192 128 128 64 64 32 64 idx)))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_kernel_aligned_fwd_kernel_aligned_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  obtain ⟨sF, hstep, hO⟩ := aligned_exec Q K V B0 Out s hundef
  rw [show exec _ s = stepStmts _ s from rfl, hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_memcell]
  rw [show surfaceOutOffset s 8192 64 1 32 idx
        = s.pids 1 * 8192 + (s.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val * 1 from by
    simp [surfaceOutOffset, mIndex, kIndex]]
  exact hO idx

end ClosedForm

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

/-- **Public Python test-shape summary for `attention_kernel_aligned.py`
(NON-self-referential).**

This end-to-end summary records the faithful aligned attention surface for the
checked relative-position-bias launch and asserts that every observable `Out`
lane holds the **genuine** closed-form base-2 streaming-softmax attention
`alignedClosedForm` (= `attentionRealBase2ScalarScaleBias` of the loaded
`Q`/`K`/`V` tiles under the scalar score scale `sm_scale · log2(e)` and the fused
`rel_h + rel_w` bias `b0 + b1`) — NOT the kernel's own executed readback:

1. the faithful aligned attention surface lowers to the algorithm layer
   (`toAlgorithm? = Except.ok alg`); and
2. the whole-kernel `Out` writeback realizes `alignedClosedForm`, discharged by
   the sorry-free exec assembly `ClosedForm.aligned_genuine_output_compute_correct`
   (preLoop → `forRangeDyn_inv` over `aligned_step` → `aligned_postLoop`), with
   the streaming heart `alignedClosedForm_eq_streaming` (→ `Math/Attention.lean`).

`log2e` is the kernel's literal `1.44269504`, so the score scale the spec carries
is exactly the executed `qk_scale` register. -/
theorem attention_kernel_aligned_python_test_shape_output_summary
    (Q K V B0 Out : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
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
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (alignedClosedForm s Q K V B0 1.0 8192 8192 128 128 64 64 32 64 idx)))) := by
  refine ⟨?_, ?_⟩
  · exact attention_kernel_aligned_fwd_kernel_aligned_surface_toAlgorithm_supported
      Q K V B0 Out 1.0 8192 64 1 8192 64 1 8192 64 1
      8192 64 1 8192 128 2 4 128 0 64 128 64 32 64
      FloatDType.fp16
  · exact ClosedForm.aligned_genuine_output_compute_correct Q K V B0 Out s hundef

end VeriTile.Bench.TritonBenchG.AttentionKernelAligned
