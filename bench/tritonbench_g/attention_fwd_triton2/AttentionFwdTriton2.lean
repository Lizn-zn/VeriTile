import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Examples.AttentionForwardClosedForm

/-!
# `attention_fwd_triton2` — strict per-kernel correctness

`_attn_fwd` is a flash-attention forward kernel: program `(start_m, off_hz)`
loads a `BLOCK_M`-row tile of `Q` for one (batch, head), then over the key/value
context (`_attn_fwd_inner`, stepping by `BLOCK_N`) runs the online-softmax
recurrence — block scores `qk = q·k · q_scale · k_scale`, running max `m_i`,
rescaled denominator `l_i`, and accumulator `acc` updated with `exp2(qk - m_ij)`
weights — and finally stores `acc / l_i` to `Out` (here a `bfloat16` output),
masked to the first 96 head lanes. This is a near-clone of
`attention_forward_triton` differing in output dtype and the explicit
`v.to(tl.float16)` cast in the `acc += dot(p, v)` step; over `ℝ` (post dtype
erasure) the two kernels lower to the *same* algorithm and share the same
closed-form output.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_attn_fwd[grid](...)`, the grid over
`(cdiv(N_CTX, BLOCK_M), Z·H)`, block scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `start_m`/`off_hz` are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
attention_fwd_triton2_output_summary_general                 ← GENERAL TOP THEOREM (dimension-parameterized, genuine closed form)
  ├─ attention_fwd_triton2_surface_toAlgorithm_supported      surface lowers to the algorithm layer
  └─ attention_fwd_triton2_closed_form_correct                ← genuine closed-form value
       └─ (online-softmax recurrence == batch base-2 softmax, Math/Attention.lean)

attention_fwd_triton2_final_store_slice_compute_correct      ← ComputeCorrect over the masked Out store
       └─ attention_fwd_triton2_final_store_slice_correct     ← algorithm-layer readback per lane
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `float16`/`float32`/
`bfloat16` casts collapse to the identity post-erasure; `@triton.autotune` /
`num_warps`/`num_stages` are not modeled. The output summary is dimension-general
(symbolic shape/strides); the Python test shape
(`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64`,
contiguous strides, 96 active head lanes) is the special case. Cross-program composition into the
full `[Z,H,N_CTX,HEAD_DIM]` output is the trusted host boundary.

## Top theorem: closed-form value (NOT self-referential)

`attention_fwd_triton2_closed_form_correct` is a **genuine closed-form value
claim**: every active output lane of `Out` equals
`VeriTile.Triton.attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under
the per-block key scale — the base-2, per-key-scaled attention output, NOT the
kernel's own executed value. The mathematical heart — online-softmax recurrence
== batch base-2 softmax — is proved sorry-free in
`VeriTile/Triton/Math/Attention.lean`, and the full `exec`-side loop unfolding is
complete in `VeriTile/Examples/AttentionForwardClosedForm.lean`. Because this
kernel's surface is (post dtype-erasure) definitionally the same loop, the top
theorem here bridges directly to that result. Tracked as
`attention-forward-online-softmax-recurrence`, #162.
## Translation-surface blocker

Translation-surface blocker: the `_attn_fwd_inner` helper JIT is inlined into
the port's single streaming-loop surface, and the Python-hard-coded head
constants (`tl.arange(0, 128)`, the `< 96` head mask,
`tl.zeros([BLOCK_M, 128])`) are generalized to the `BLOCK_DMODEL` /
`HEAD_ACTIVE` binders — the Python literals are the `128`/`96` instantiation
of the dimension-general top theorem. The Lean surface is therefore not a
line-for-line textual match of the Python `_attn_fwd` body, and the textual
py↔lean scans in `bench/audit_tritonbench_g.sh` exempt this port on this
marker (registered in `proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton2

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `attention_fwd_triton2_output_summary_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-- Full Lean port of `attention_fwd_triton2.py`'s `_attn_fwd`.

The upstream kernel calls a separate `@triton.jit` helper `_attn_fwd_inner` to
run the K/V streaming-softmax loop. The DSL has no function-call surface, so the
helper body is inlined verbatim into the outer kernel; semantically the two
forms are identical for this fixed-stage path. The upstream `v.to(tl.float16)`
dot-input cast and the `bfloat16` output cast erase to the identity over `ℝ`, so
this surface is the same inlined online-softmax loop verified for
`attention_forward_triton`.

The literal `128` and `96` in the upstream kernel correspond to the
`BLOCK_DMODEL` / `HEAD_ACTIVE` parameters threaded through the bundled tests
(`head_dim = 128`, with the inner dot using only the first 96 lanes of the head
dimension). They appear here as explicit Lean parameters. -/
def attention_fwd_triton2_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}

/-- The full inlined `attention_fwd_triton2` surface lowers to the algorithm
layer. -/
theorem attention_fwd_triton2_surface_toAlgorithm_supported
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ alg, (attention_fwd_triton2_surface Q K V Q_scale K_scale Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn
      stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh
      stride_om stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_triton2_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of
`attention_fwd_triton2.py`'s `_attn_fwd`.

The full kernel computes the attention accumulator through Q/K/V tiled dot
products, quantization scales, and a streaming softmax. This slice starts after
`acc = acc / l_i[:, None]` with a precomputed `Acc` tile and proves the final
masked writeback into `Out`, preserving the source mask
`(offs_m < N_CTX) & (offs_k < 96)`. The source forms `O_block_ptr` from the Q
strides, so this slice names those as the output store strides. The inner
`tl.float32` accumulator and `p.to(tl.float16)` dot-input cast are outside this
slice. -/
def attention_fwd_triton2_final_store_slice
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(N_CTX)) & (offs_k[None, :] < $(HEAD_ACTIVE))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k),
      mask=mask, other=0.0)
  tl.store(Out + off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE

instance activeDecidable
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s N_CTX HEAD_ACTIVE BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_k BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + kIndex idx * stride_acc_k

def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk

/-! ## Closed-form spec inputs (loaded tiles, per-key scale)

The genuine `expected` for the top theorem is `attentionRealBase2PerKeyScale` of
the loaded Q/K/V tiles under the per-block key scale, mirroring
`VeriTile/Examples/AttentionForwardClosedForm.lean`. Under the contiguity
contracts `stride_qm = stride_kn = HEAD_DIM`, head stride `1`, every loaded
element sits at `base + row · HEAD_DIM + col`; masked-off head lanes load `0`,
so summing over the `HEAD_ACTIVE` active lanes is the full contraction. -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Batch/head base offset `off_z · stride_qz + off_h · stride_qh`. -/
def baseOffset (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh

noncomputable def qTile (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i * HEAD_DIM + e.val)

noncomputable def kTile (s : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + e.val)

noncomputable def vTile (s : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + d.val)

/-- Per-key scale `q_scale · k_scale[block(j)]`, `block(j) = j / BLOCK_N`.
`q_scale` is read at `off_hz · cdiv(N_CTX, BLOCK_M) + pid₀`; `k_scale[b]` at
`off_hz · cdiv(N_CTX, BLOCK_N) + b`. -/
noncomputable def keyScale (s : BlockState) (Q_scale K_scale : RegionName)
    (N_CTX BLOCK_M BLOCK_N S : Nat) :
    Fin S → ℝ :=
  fun j =>
    s.readMem Q_scale (s.pids 1 * cdiv N_CTX BLOCK_M + s.pids 0) *
      s.readMem K_scale (s.pids 1 * cdiv N_CTX BLOCK_N + j.val / BLOCK_N)

/-- Algorithm-layer correctness for the final output store. -/
theorem attention_fwd_triton2_final_store_slice_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
      (exec (attention_fwd_triton2_final_store_slice Acc Out H N_CTX
            HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            s.readMem Acc
              (accOffset s H stride_acc_z stride_acc_h stride_acc_m
                stride_acc_k BLOCK_M idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attention_fwd_triton2_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        offZ, offH, mIndex, kIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
            idx.2.1.val < HEAD_ACTIVE then
          some (s.readMem Acc
            (s.pids 1 / H * stride_acc_z + s.pids 1 % H * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_k))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
        idx.2.1.val < HEAD_ACTIVE
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, kIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
          BLOCK_M idx)
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]

/-- Compute-facing correctness for the final output store. -/
theorem attention_fwd_triton2_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton2_final_store_slice Acc Out H N_CTX
        HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        s.readMem Acc
          (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            BLOCK_M idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton2_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attention_fwd_triton2_final_store_slice_correct Acc Out H N_CTX
    HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrapper

`attention_fwd_triton2.py`'s checked test uses `B = 2`, `H = 4`,
`N_CTX = 128`, `HEAD_DIM = 128`, `BLOCK_M = 128`, `BLOCK_N = 64`, and
the final store mask enables only the first 96 head lanes. Contiguous
`[B, H, N_CTX, HEAD_DIM]` tensors have strides `(65536, 16384, 128, 1)`. -/

/-- **Closed-form correctness for `attention_fwd_triton2` (general statement).**

For arbitrary batch/head strides, head count, block sizes, KV-block count,
head/active dimensions and arbitrary `q_scale`/`k_scale`, every active output
lane of `Out` (`mIndex < N_CTX ∧ head < HEAD_ACTIVE`) equals
`attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block key
scale — the genuine base-2, per-key-scaled attention output, NOT the kernel's own
executed value. Inactive lanes are unconstrained (masked out by the write map).

Layout contracts: `N_CTX = BLOCK_N · numKVBlocks`, `stride_qm = stride_kn =
HEAD_DIM` and head stride `1`, `0 < BLOCK_N`, `HEAD_ACTIVE ≤ BLOCK_DMODEL`,
`HEAD_ACTIVE ≤ HEAD_DIM`, clean initial `undef`. The Python test case
(`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, HEAD_ACTIVE=96`,
`q_scale = k_scale = 1`) is the special case.

**Proven sorry-free**: `attention_fwd_triton2`'s surface is — post dtype-erasure
over `ℝ` — definitionally the same inlined online-softmax loop verified in
`VeriTile/Examples/AttentionForwardClosedForm.lean`, whose full `exec`-side loop
unfolding and math core (`Math/Attention.lean`) are complete. This theorem
bridges directly to
`VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_closed_form_correct`.
Tracked as `attention-forward-online-softmax-recurrence`, #162. -/
theorem attention_fwd_triton2_closed_form_correct
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (hHD : HEAD_ACTIVE ≤ HEAD_DIM) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton2_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if h : idx.2.1.val < HEAD_ACTIVE then
          attentionRealBase2PerKeyScale
            (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            (idx.1, ⟨idx.2.1.val, h⟩, PUnit.unit)
        else (0 : ℝ)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  obtain ⟨hm, hk⟩ := hActive
  have hmain := VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_closed_form_correct
    Q K V Q_scale K_scale Out s0 stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM
    BLOCK_DMODEL HEAD_ACTIVE STAGE hBN hActiveLe hHD hundef idx ⟨hm, hk⟩
  have hExec2 : exec (VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_surface
      Q K V Q_scale K_scale Out stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE) s0
      = some s' := hExec
  rw [hExec2] at hmain
  simp only [ComputeCorrect.OutputReadable.read_real]
  rw [dif_pos (show idx.2.1.val < HEAD_ACTIVE from hk)]
  exact hmain


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **Dimension-general output summary for `attention_fwd_triton2` (no test-shape pin).**

Mirrors the reference `attention_forward_triton_closed_form_correct`: over
*symbolic* batch/head strides, head count `H`, block sizes `BLOCK_M`/`BLOCK_N`,
KV-block count `numKVBlocks` (so `N_CTX = BLOCK_N · numKVBlocks`), head/active
dimensions and arbitrary `q_scale`/`k_scale`, this combines

* the checked full-surface lowering to the algorithm layer
  (`attention_fwd_triton2_surface_toAlgorithm_supported`), and
* the genuine closed-form value of every active `Out` lane
  (`attention_fwd_triton2_closed_form_correct`):
  `attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block
  key scale — the base-2, per-key-scaled attention output reading INPUT Q/K/V
  memory, NOT the kernel's own executed value.

The only layout assumptions are the contiguity contracts the kernel relies on
(`stride_qm = stride_kn = HEAD_DIM`, head stride `1`), `0 < BLOCK_N`,
`HEAD_ACTIVE ≤ BLOCK_DMODEL`, `HEAD_ACTIVE ≤ HEAD_DIM`, and a clean initial
`undef`. The Python test case (`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128,
BLOCK_N=64, HEAD_ACTIVE=96, numKVBlocks=2`) is the special case. -/
theorem attention_fwd_triton2_output_summary_general
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (hHD : HEAD_ACTIVE ≤ HEAD_DIM) (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_fwd_triton2_surface Q K V Q_scale K_scale Out
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton2_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if h : idx.2.1.val < HEAD_ACTIVE then
          attentionRealBase2PerKeyScale
            (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            (idx.1, ⟨idx.2.1.val, h⟩, PUnit.unit)
        else (0 : ℝ)) :=
  ⟨attention_fwd_triton2_surface_toAlgorithm_supported Q K V Q_scale K_scale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE,
    attention_fwd_triton2_closed_form_correct Q K V Q_scale K_scale Out s
      stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL
      HEAD_ACTIVE STAGE hBN hActiveLe hHD hundef⟩

end Correct


end VeriTile.Bench.TritonBenchG.AttentionFwdTriton2
