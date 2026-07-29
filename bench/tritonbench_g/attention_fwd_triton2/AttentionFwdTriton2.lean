import VeriTile.Triton
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

section Correct_without_Rounding

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
specification attention_fwd_triton2_output_summary_general
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
    ComputeCorrect.Realizes_without_Rounding
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

end Correct_without_Rounding

section IOFace

open scoped VeriTile.Triton.StreamMasked3DKernelIO₅

open VeriTile.Examples.AttentionForwardClosedForm
  (attnLoopBody attnAccAssign attnStoreStmt attn_body_split preLoop_scalars
   qptrs_eval qscaleptr_eval kptrs_eval kscaleptr_eval vptrs_eval
   mi_init_eval li_init_eval acc_init_eval qmask_eval kmask_eval vmask_eval
   load_ptr_mask_real load_ptr_none_real
   qk_op_eval mij_op_eval qk2_op_eval p_op_eval lij_op_eval alpha_op_eval
   li_op_eval acc1_op_eval pfp16_op_eval acc2_op_eval kptr_adv_eval ksptr_adv_eval
   attention_forward_triton_surface attention_forward_triton_closed_form_correct)

/-! ## Cast-free collapse under `R.round .fp16 = id`

The exec-side foundation (`VeriTile/Examples/AttentionForwardClosedForm.lean`)
is stated at the exact `exec`/`stepStmts` level. Under the file's declared fp16
modeling boundary `R.round .fp16 = id` every rounding site of the surface
collapses to the exact cast, so `execR R` reduces to `exec` and the exact closed
form transports verbatim. -/

/-- The `R`-cast on the `.real` channel never rounds. -/
private theorem aft2IO_Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .real = id from by
    funext y; cases y <;> simp [RoundingModel.roundW]]
  rfl

/-- With `R.round .fp16 = id`, the `R`-cast into the fp16 grid is the exact
cast (the in-loop rounding-event site collapses). -/
private theorem aft2IO_Rcast_real_fp16 (R : RoundingModel) (hfp16 : R.round .fp16 = id) :
    R.cast .real .fp16 = FloatDType.cast .real .fp16 := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .fp16 = id from by
    funext y; cases y <;> simp [RoundingModel.roundW, hfp16]]
  rfl

/-- The `R`-cast out of the fp16 grid lands on the `.real` channel, which
never rounds. -/
private theorem aft2IO_Rcast_fp16_real (R : RoundingModel) :
    R.cast .fp16 .real = FloatDType.cast .fp16 .real := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .real = id from by
    funext y; cases y <;> simp [RoundingModel.roundW]]
  rfl

/-- `.real` stores never round: `writeMemTypedR` delegates to the exact write. -/
private theorem aft2IO_wmtR_real (R : RoundingModel) (s : BlockState)
    (region : RegionName) (o : Nat) (v : TileCarrier .real) :
    s.writeMemTypedR R .real region o v = s.writeMemTyped .real region o v := rfl

/-! ## The surface's statement decomposition

The prologue is transcribed literally (statements 0–10 = the scalar offsets and
index vectors, statements 11–21 = the six pointer seeds, the three accumulator
seeds and the two `Q`/`Q_scale` loads); the streaming body and the two postLoop
statements are the foundation's `attnLoopBody` / `attnAccAssign` /
`attnStoreStmt`. -/

/-- Prologue statements 0–10: the eight scalar offsets and the three index
vectors. -/
private def aft2PreHead
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "qvk_offset" (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))),
    Stmt.assign .nat [] "vk_offset"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat HEAD_DIM)),
    Stmt.assign .nat [] "q_scale_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat (BLOCK_N * numKVBlocks)) (Op.constNat BLOCK_M))
          (Op.constNat 1)) (Op.constNat BLOCK_M))),
    Stmt.assign .nat [] "k_scale_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat (BLOCK_N * numKVBlocks)) (Op.constNat BLOCK_N))
          (Op.constNat 1)) (Op.constNat BLOCK_N))),
    Stmt.assign .nat [BLOCK_M] "offs_m" (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
        (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_k" (Op.arange BLOCK_DMODEL) ]

/-- Prologue statements 11–19: the six pointer seeds and the three accumulator
seeds (all load-free). -/
private def aft2PtrSeeds (Q K V Q_scale K_scale Out : RegionName)
    (HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
              (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k"))
            (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Q_scale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.constNat HEAD_DIM)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase K_scale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k"))
            (Op.constNat 1)))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
              (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k"))
            (Op.constNat 1)))),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) ]

/-- Prologue statements 20–21: the `q` and `q_scale` loads. -/
private def aft2Loads (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (.ptr (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"))
        (.mask (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
            (Op.constNat (BLOCK_N * numKVBlocks)))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
              (Op.constNat HEAD_ACTIVE)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (Op.ref .ptr [] "Q_scale_ptr")) .none) ]

/-- The load-free prologue prefix (statements 0–19). -/
private def aft2Pre20 (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat) :
    List Stmt :=
  aft2PreHead stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks
    ++ aft2PtrSeeds Q K V Q_scale K_scale Out HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL

/-- The whole prologue (statements 0–21). -/
private def aft2PreLoop (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) :
    List Stmt :=
  aft2Pre20 Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks
    ++ aft2Loads BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks

/-- The two postLoop statements (`acc /= l_i` and the masked terminal store). -/
private def aft2PostLoop (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) :
    List Stmt :=
  [ attnAccAssign BLOCK_M BLOCK_DMODEL,
    attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE ]

/-- Body decomposition of the whole surface: prologue ++ (KV loop :: postLoop).
By `rfl` (the surface's `toAlgKernel.body` reduces to the literal list). -/
private theorem aft2_body_split (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE
      STAGE : Nat) :
    (attention_fwd_triton2_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body
      = aft2PreLoop Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
        ++ (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
              (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
            :: aft2PostLoop BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) := rfl

set_option maxHeartbeats 4000000 in
/-- Every prologue statement is cast-free (`.nat`/`.real`/`.ptr` register
arithmetic and the two `.real` loads — no rounding site in sight). -/
private theorem aft2IO_preLoop_stmt_castFree (R : RoundingModel)
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) :
    ∀ st ∈ aft2PreLoop Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE numKVBlocks,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [aft2PreLoop, aft2Pre20, aft2PreHead, aft2PtrSeeds, aft2Loads, List.mem_append,
    List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with (h | h) | h
  · rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · rcases h with rfl | rfl <;>
      simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 4000000 in
/-- Every loop-body statement is cast-free **given `R.round .fp16 = id`**: the
three `castFloat` sites (the fp32 wrapper on the `qk` dot, statement 13's
`p = (p).to(tl.float16)`, and the `fp16 → real` re-widening inside statement
14's dot) collapse via the three `aft2IO_Rcast_*` lemmas. -/
private theorem aft2IO_loopBody_stmt_castFree (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks : Nat) :
    ∀ st ∈ attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [attnLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      aft2IO_Rcast_real_real R, aft2IO_Rcast_real_fp16 R hfp16, aft2IO_Rcast_fp16_real R]

set_option maxHeartbeats 4000000 in
/-- Both postLoop statements are cast-free: the `acc /= l_i` divide is register
arithmetic and the masked terminal store is `.real`-typed
(`aft2IO_wmtR_real`). -/
private theorem aft2IO_postLoop_stmt_castFree (R : RoundingModel)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) :
    ∀ st ∈ aft2PostLoop BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [aft2PostLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl <;>
    simp only [attnAccAssign, attnStoreStmt, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      aft2IO_wmtR_real R]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). -/
private theorem aft2IO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact aft2IO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

/-- The prologue collapses onto the exact stepper. -/
private theorem aft2IO_preLoop_castFree (R : RoundingModel)
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (t : BlockState) :
    stepStmtsR R (aft2PreLoop Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M
        BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) t
      = stepStmts (aft2PreLoop Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M
        BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) t :=
  aft2IO_stepStmtsR_castFree_of_stmts R _
    (aft2IO_preLoop_stmt_castFree R Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) t

/-- The loop body collapses onto the exact stepper (under `hfp16`). -/
private theorem aft2IO_loopBody_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks : Nat) (t : BlockState) :
    stepStmtsR R (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks) t
      = stepStmts (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks) t :=
  aft2IO_stepStmtsR_castFree_of_stmts R _
    (aft2IO_loopBody_stmt_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
      numKVBlocks) t

/-- The static streaming `forRange` statement is cast-free given
`R.round .fp16 = id` (static bounds, cast-free body through
`stepForRangeAuxR_castFree`). -/
private theorem aft2IO_loopStmt_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks : Nat) :
    ∀ u, stepStmtR R
        (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
          (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) u
      = stepStmt
        (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
          (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) u := by
  intro u
  rw [stepStmtR_forRange,
    stepForRangeAuxR_castFree R _
      (aft2IO_loopBody_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
        numKVBlocks) "start_n",
    ← stepForRangeAux.forRange_unfold]

/-- The postLoop collapses onto the exact stepper. -/
private theorem aft2IO_postLoop_castFree (R : RoundingModel)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) (t : BlockState) :
    stepStmtsR R (aft2PostLoop BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) t
      = stepStmts (aft2PostLoop BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) t :=
  aft2IO_stepStmtsR_castFree_of_stmts R _
    (aft2IO_postLoop_stmt_castFree R BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) t

set_option maxHeartbeats 4000000 in
/-- The `attention_fwd_triton2` surface sits inside the flat-memory bridge's
covered fragment (plain `ptrAdd` walks only — no `ptrSub`, no atomics, no block
pointers). -/
private theorem aft2IO_flattenOk (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE
      STAGE : Nat) :
    ((attention_fwd_triton2_surface Q K V Q_scale K_scale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [aft2_body_split]
  simp [aft2PreLoop, aft2Pre20, aft2PreHead, aft2PtrSeeds, aft2Loads, aft2PostLoop, attnLoopBody,
    attnAccAssign, attnStoreStmt, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-! ## Memory helpers: the seeded pointer tiles

The safety walk needs the *addresses* the kernel touches, so the pointer
registers are pinned exactly (their values are the window-bound obligations)
while the value registers only need bare existence. -/

/-- Lift a register readback through a `setReg` to a different name. -/
private theorem aft2_regs_setReg_chain {d d' : TileDType} {sh sh' : TileShape}
    {n n' : RegName} {s : BlockState} {v : Tile d sh} {w : Tile d' sh'}
    (hne : n ≠ n') (h : s.regs d sh n = some v) :
    (s.setReg n' d' sh' w).regs d sh n = some v := by
  simp only [BlockState.setReg_ne_name, ne_eq, hne, not_false_eq_true, h]

/-- The shared batch/head plane base `qvk_offset`. -/
private def aft2Base (s0 : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  s0.pids 1 / H * stride_qz + s0.pids 1 % H * stride_qh

/-- The `Q_ptrs` / `O_block_ptr` row-major query window: cell `(i, e)` →
`base + (pid₀·BLOCK_M + i)·HEAD_DIM + e`. -/
private def aft2RowPtr (s0 : BlockState) (Rg : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    Tile .ptr [BLOCK_M, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
    (Rg.cast, aft2Base s0 H stride_qz stride_qh
      + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val)⟩

/-- The transposed `K_ptrs` window after `c` block advances: cell `(e, jL)` →
`base + e + (c·BLOCK_N + jL)·HEAD_DIM`. -/
private def aft2KPtrs (s0 : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile .ptr [BLOCK_DMODEL, BLOCK_N] :=
  ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
    (K.cast, aft2Base s0 H stride_qz stride_qh + idx.1.val
      + (c * BLOCK_N + idx.2.1.val) * HEAD_DIM)⟩

/-- The `V_ptrs` window after `c` block advances: cell `(jL, d)` →
`base + (c·BLOCK_N + jL)·HEAD_DIM + d`. -/
private def aft2VPtrs (s0 : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile .ptr [BLOCK_N, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
    (V.cast, aft2Base s0 H stride_qz stride_qh + (c * BLOCK_N + idx.1.val) * HEAD_DIM
      + idx.2.1.val)⟩

/-- A scalar-width scale pointer at slot `off + c`. -/
private def aft2SPtr (Rg : RegionName) (off c : Nat) : Tile .ptr [] :=
  ⟨fun _ : TileIndex [] => (Rg.cast, off + c)⟩

/-- `K_ptrs` advances one block per step. -/
private theorem aft2KPtrs_succ (s0 : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (aft2KPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N
        BLOCK_DMODEL c) (Tile.scalar (BLOCK_N * HEAD_DIM))
      = aft2KPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [aft2KPtrs, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
    ring

/-- `V_ptrs` advances one block per step. -/
private theorem aft2VPtrs_succ (s0 : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (aft2VPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N
        BLOCK_DMODEL c) (Tile.scalar (BLOCK_N * HEAD_DIM))
      = aft2VPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [aft2VPtrs, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
    ring

/-- `K_scale_ptr` advances one slot per step. -/
private theorem aft2SPtr_succ (Rg : RegionName) (off c : Nat) :
    Tile.ptrAdd Broadcast.nil (aft2SPtr Rg off c) (Tile.scalar 1) = aft2SPtr Rg off (c + 1) := by
  ext idx
  · rfl
  · simp only [aft2SPtr, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    omega

/-- The prologue head is `body.take 11`. -/
private theorem aft2_preHead_eq (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE
      STAGE : Nat) :
    (attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 11
      = aft2PreHead stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks := rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Weak load-free prologue walk**: from an *arbitrary* launch state (no
clean-`undef` pin — the skin's safety obligation quantifies over every state)
the 20 load-free prologue statements step successfully, pinning the six pointer
registers exactly and the three accumulator seeds up to existence. -/
private theorem aft2IO_pre20_evalW (s : BlockState) (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat) :
    ∃ s20, stepStmts (aft2Pre20 Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks) s = some s20
      ∧ s20.pids = s.pids ∧ s20.mem = s.mem
      ∧ s20.regs .nat [BLOCK_M] "offs_m"
          = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s20.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ (∃ mT : Tile .real [BLOCK_M], s20.regs .real [BLOCK_M] "m_i" = some mT)
      ∧ (∃ lT : Tile .real [BLOCK_M], s20.regs .real [BLOCK_M] "l_i" = some lT)
      ∧ (∃ aT : Tile .real [BLOCK_M, BLOCK_DMODEL],
          s20.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT)
      ∧ s20.regs .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
          = some (aft2RowPtr s Q H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL)
      ∧ s20.regs .ptr [] "Q_scale_ptr"
          = some (aft2SPtr Q_scale (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M) (s.pids 0))
      ∧ s20.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
          = some (aft2KPtrs s K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s20.regs .ptr [] "K_scale_ptr"
          = some (aft2SPtr K_scale (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N) 0)
      ∧ s20.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
          = some (aft2VPtrs s V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s20.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
          = some (aft2RowPtr s Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL) := by
  obtain ⟨s11, h11, hpids, hstart, hqvk, hqso, hkso, hm, hn, hk, -, hmem⟩ :=
    preLoop_scalars Q K V Q_scale K_scale Out s stride_qz stride_qh 0 H BLOCK_M BLOCK_N
      numKVBlocks HEAD_DIM BLOCK_DMODEL 0 0
  rw [aft2_preHead_eq Q K V Q_scale K_scale Out stride_qz stride_qh 0 H BLOCK_M BLOCK_N
    numKVBlocks HEAD_DIM BLOCK_DMODEL 0 0] at h11
  have hqvk' : s11.regs .nat [] "qvk_offset"
      = some (Tile.scalar (aft2Base s H stride_qz stride_qh)) := hqvk
  have hqso' : s11.regs .nat [] "q_scale_offset"
      = some (Tile.scalar (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M)) := hqso
  have hkso' : s11.regs .nat [] "k_scale_offset"
      = some (Tile.scalar (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N)) := hkso
  unfold aft2Pre20 aft2PtrSeeds
  rw [stepStmts.append_some h11,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qptrs_eval s11 Q BLOCK_M BLOCK_DMODEL HEAD_DIM (aft2Base s H stride_qz stride_qh)
        (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val) hqvk' hm hk)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qscaleptr_eval _ Q_scale (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M) (s.pids 0)
        (by simp [hqso']) (by simp [hstart]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (kptrs_eval _ K BLOCK_DMODEL BLOCK_N HEAD_DIM (aft2Base s H stride_qz stride_qh)
        (by simp [hqvk']) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (kscaleptr_eval _ K_scale (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N)
        (by simp [hkso']))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (vptrs_eval _ V BLOCK_N BLOCK_DMODEL HEAD_DIM (aft2Base s H stride_qz stride_qh)
        (by simp [hqvk']) (by simp [hn]) (by simp [hk]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qptrs_eval _ Out BLOCK_M BLOCK_DMODEL HEAD_DIM (aft2Base s H stride_qz stride_qh)
        (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val)
        (by simp [hqvk']) (by simp [hm]) (by simp [hk]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (mi_init_eval _ BLOCK_M)),
    stepStmts.cons_some (stepStmt_assign_eq_some (li_init_eval _ BLOCK_M)),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BLOCK_M BLOCK_DMODEL)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact hpids
  · exact hmem
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hn]
  · exact ⟨(⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M]),
      by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, BlockState.setReg_same]⟩
  · exact ⟨(⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M]),
      by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, BlockState.setReg_same]⟩
  · exact ⟨(⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩ :
        Tile .real [BLOCK_M, BLOCK_DMODEL]),
      by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, BlockState.setReg_same]⟩
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    · rfl
    · simp only [aft2RowPtr, Nat.mul_one]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    · rfl
    · simp only [aft2SPtr, Tile.scalar]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    · rfl
    · simp only [aft2KPtrs, Nat.zero_mul, Nat.zero_add]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    · rfl
    · simp only [aft2SPtr, Tile.scalar, Nat.add_zero]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    · rfl
    · simp only [aft2VPtrs, Nat.mul_one, Nat.zero_mul, Nat.zero_add]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    · rfl
    · simp only [aft2RowPtr, Nat.mul_one]

/-! ## The safety walk (weak invariant)

The skin's `hts` obligation quantifies over **arbitrary** launch states (no
clean-`undef` pin), so the exec stack's `attnInvariant` is unavailable there.
The safety walk instead runs on the *shape* half: exact pins for the
loop-carried pointer/index registers (whose addresses are the bound
obligations) plus bare existence for the value registers. -/

/-- The value a masked `.real` pointer load produces on an **arbitrary** state:
active lanes read memory, inactive lanes fall through to `undef` (the exec
stack's `load_ptr_mask_real` is the `undef = 0` specialization). Only `mem` and
`undef` are consulted, so the value is stated against a `base` state that the
walk's `setReg` chains agree with. -/
private noncomputable def aft2LoadVal {shape : TileShape} (base : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape) : Tile .real shape :=
  ⟨fun i => if masks.data i then base.readMemValue .real (ptrs.data i).1 (ptrs.data i).2
      else some (base.undef (ptrs.data i).1 (ptrs.data i).2)⟩

/-- **Rewrite-based masked-load recipe** (no clean-`undef` pin): a masked `.real`
pointer load always evaluates, to `aft2LoadVal`. -/
private theorem aft2IO_load_mask_val {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s base : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks)
    (hrm : s.mem = base.mem) (hu : s.undef = base.undef) :
    evalOp (Op.load .real (.ptr ptrOp) (.mask maskOp)) s
      = some (aft2LoadVal base ptrs masks) := by
  have hras : ∀ (d : FloatDType) (r : RegionName) (o : Nat),
      s.readMemAs d r o = base.readMemAs d r o := by
    intro d r o
    simp only [BlockState.readMemAs, hrm]
  simp only [evalOp, hp, hm, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [aft2LoadVal, BlockState.readMemValue, hras, hu, if_true]

/-- Safety-walk loop invariant: counter alignment/bound, exact pins for the
index vectors, the three streamed pointers and the output pointer (whose
addresses are the bound obligations), and bare existence for the value
registers. -/
private def aft2SafeInv (K K_scale V Out : RegionName) (s0 : BlockState)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  c % BLOCK_N = 0 ∧ c ≤ BLOCK_N * numKVBlocks ∧
  s.regs .nat [BLOCK_M] "offs_m"
    = some (Tile.vec (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)) ∧
  s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) ∧
  (∃ mT : Tile .real [BLOCK_M], s.regs .real [BLOCK_M] "m_i" = some mT) ∧
  (∃ lT : Tile .real [BLOCK_M], s.regs .real [BLOCK_M] "l_i" = some lT) ∧
  (∃ aT : Tile .real [BLOCK_M, BLOCK_DMODEL],
    s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT) ∧
  (∃ qT : Tile .real [BLOCK_M, BLOCK_DMODEL],
    s.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qT) ∧
  (∃ qsT : ℝ, s.regs .real [] "q_scale" = some (Tile.scalar (some qsT))) ∧
  s.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
    = some (aft2KPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)) ∧
  s.regs .ptr [] "K_scale_ptr"
    = some (aft2SPtr K_scale (s0.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N) (c / BLOCK_N)) ∧
  s.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
    = some (aft2VPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)) ∧
  s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
    = some (aft2RowPtr s0 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL) ∧
  s.mem = s0.mem

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Weak prologue**: from an arbitrary launch state the full prologue steps to
a state satisfying `aft2SafeInv … 0`. -/
private theorem aft2IO_preLoop_evalW (s : BlockState) (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) :
    ∃ sp, stepStmts (aft2PreLoop Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) s = some sp
      ∧ aft2SafeInv K K_scale V Out s H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks 0 sp := by
  obtain ⟨s20, h20, hpids20, hmem20, hoffsm, hoffsn, hmi, hli, hacc, hqp, hqsp, hkp, hksp,
    hvp, hop⟩ :=
    aft2IO_pre20_evalW s Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M
      BLOCK_N BLOCK_DMODEL numKVBlocks
  unfold aft2PreLoop aft2Loads
  rw [stepStmts.append_some h20]
  have hqload := aft2IO_load_mask_val (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs") _ s20 s20 _ _
    (by rw [evalOp_ref]; exact hqp)
    (qmask_eval s20 BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
      (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val) hoffsm) rfl rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hqload),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (load_ptr_none_real (Op.ref .ptr [] "Q_scale_ptr") _ _
        (by rw [evalOp_ref]; exact aft2_regs_setReg_chain (by decide) hqsp))),
    stepStmts.nil]
  obtain ⟨mT, hmi⟩ := hmi
  obtain ⟨lT, hli⟩ := hli
  obtain ⟨aT, hacc⟩ := hacc
  refine ⟨_, rfl, Nat.zero_mod _, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_⟩
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hoffsm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hoffsn]
  · exact ⟨mT, aft2_regs_setReg_chain (by decide) (aft2_regs_setReg_chain (by decide) hmi)⟩
  · exact ⟨lT, aft2_regs_setReg_chain (by decide) (aft2_regs_setReg_chain (by decide) hli)⟩
  · exact ⟨aT, aft2_regs_setReg_chain (by decide) (aft2_regs_setReg_chain (by decide) hacc)⟩
  · exact ⟨_, aft2_regs_setReg_chain (by decide) (BlockState.setReg_same _ _ _ _ _)⟩
  · refine ⟨s20.readMem Q_scale
      (s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M + s.pids 0), ?_⟩
    rw [BlockState.setReg_same]
    refine congrArg some ?_
    ext i
    simp only [aft2SPtr, Tile.scalar, Region.cast_id, BlockState.setReg_mem, BlockState.readMem]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      Nat.zero_div, hkp]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      Nat.zero_div, hksp]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      Nat.zero_div, hvp]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hop]
  · exact hmem20

set_option maxHeartbeats 4000000 in
/-- The 20 load-free prologue statements are safe at every state. -/
private theorem aft2IO_pre20_stmt_safe (R : RoundingModel) (bounds : RegionBounds)
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat) :
    ∀ st ∈ aft2Pre20 Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
        BLOCK_DMODEL numKVBlocks,
      ∀ u : BlockState, Stmt.TraceSafeR R bounds st u := by
  intro st hst u
  simp only [aft2Pre20, aft2PreHead, aft2PtrSeeds, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false] at hst
  rcases hst with h | h
  · rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

/-- `evalOpR` value of the shared `q`-load / terminal-store mask under an
`offs_m` pin (the op is cast-free, so `R` is inert). -/
private theorem aft2IO_qmask_evalR (R : RoundingModel) (u : BlockState)
    (p₀ N_CTX HEAD_ACTIVE BLOCK_M BLOCK_DMODEL : Nat)
    (hoffsm : u.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => p₀ * BLOCK_M + r.val))) :
    evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (decide (p₀ * BLOCK_M + idx.1.val < N_CTX) && decide (idx.2.1.val < HEAD_ACTIVE))⟩ := by
  rw [show evalOpR R (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      from by simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact qmask_eval u BLOCK_M BLOCK_DMODEL N_CTX HEAD_ACTIVE
    (fun r : Fin BLOCK_M => p₀ * BLOCK_M + r.val) hoffsm

/-- `evalOpR` value of the in-loop `k_mask` op under `offs_n`/`start_n` pins. -/
private theorem aft2IO_kmask_evalR (R : RoundingModel) (u : BlockState)
    (SN N_CTX HEAD_ACTIVE BLOCK_N BLOCK_DMODEL : Nat)
    (hoffsn : u.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOpR R (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = some ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
          (decide (idx.2.1.val < N_CTX - SN) && decide (idx.1.val < HEAD_ACTIVE))⟩ := by
  rw [show evalOpR R (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = evalOp (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      from by simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact kmask_eval u BLOCK_DMODEL BLOCK_N N_CTX SN HEAD_ACTIVE hoffsn hsn

/-- `evalOpR` value of the in-loop `v`-load mask under `offs_n`/`start_n` pins. -/
private theorem aft2IO_vmask_evalR (R : RoundingModel) (u : BlockState)
    (SN N_CTX HEAD_ACTIVE BLOCK_N BLOCK_DMODEL : Nat)
    (hoffsn : u.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
          (decide (idx.1.val < N_CTX - SN) && decide (idx.2.1.val < HEAD_ACTIVE))⟩ := by
  rw [show evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      from by simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact vmask_eval u BLOCK_N BLOCK_DMODEL N_CTX SN HEAD_ACTIVE hoffsn hsn

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body execution chain**: from register pins alone (no
clean-`undef` pin) the 19-statement body steps to some state, advancing the
three streamed pointers by one block and preserving the loop-invariant
registers. This is the exec stack's `attn_loopBody_steps` with the two masked
loads weakened to `aft2LoadVal` (their *values* need `undef`; the walk does
not). -/
private theorem aft2IO_loopBody_stepsW (BM BN BD nB HA HD : Nat) (hBN : 0 < BN)
    (hax : 1 < [BM].length.succ) (sin : BlockState)
    (SN : TileCarrier .nat) (qsv : ℝ)
    (Kptr : Tile .ptr [BD, BN]) (Ksptr : Tile .ptr []) (Vptr : Tile .ptr [BN, BD])
    (qtile : Tile .real [BM, BD]) (mtile ltile : Tile .real [BM]) (acctile : Tile .real [BM, BD])
    (hoffs : sin.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hKp : sin.regs .ptr [BD, BN] "K_ptrs" = some Kptr)
    (hKs : sin.regs .ptr [] "K_scale_ptr" = some Ksptr)
    (hVp : sin.regs .ptr [BN, BD] "V_ptrs" = some Vptr)
    (hq : sin.regs .real [BM, BD] "q" = some qtile)
    (hqs : sin.regs .real [] "q_scale" = some (Tile.scalar (some qsv)))
    (hmi : sin.regs .real [BM] "m_i" = some mtile)
    (hli : sin.regs .real [BM] "l_i" = some ltile)
    (hacc : sin.regs .real [BM, BD] "acc" = some acctile) :
    ∃ sF, stepStmts (attnLoopBody BM BN BD HA HD nB) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem
      ∧ sF.regs .ptr [BD, BN] "K_ptrs"
          = some (Tile.ptrAdd Broadcast.scalarR Kptr (Tile.scalar (BN * HD)))
      ∧ sF.regs .ptr [] "K_scale_ptr" = some (Tile.ptrAdd Broadcast.nil Ksptr (Tile.scalar 1))
      ∧ sF.regs .ptr [BN, BD] "V_ptrs"
          = some (Tile.ptrAdd Broadcast.scalarR Vptr (Tile.scalar (BN * HD)))
      ∧ sF.regs .real [BM, BD] "q" = some qtile
      ∧ sF.regs .real [] "q_scale" = some (Tile.scalar (some qsv))
      ∧ sF.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
      ∧ sF.regs .nat [BM] "offs_m" = sin.regs .nat [BM] "offs_m"
      ∧ sF.regs .ptr [BM, BD] "O_block_ptr" = sin.regs .ptr [BM, BD] "O_block_ptr"
      ∧ (∃ t : Tile .real [BM], sF.regs .real [BM] "m_i" = some t)
      ∧ (∃ t : Tile .real [BM], sF.regs .real [BM] "l_i" = some t)
      ∧ (∃ t : Tile .real [BM, BD], sF.regs .real [BM, BD] "acc" = some t) := by
  set kmaskT : Tile .bool [BD, BN] :=
    ⟨fun idx => (decide (idx.2.1.val < BN * nB - SN) && decide (idx.1.val < HA))⟩ with hkm
  set kloadT : Tile .real [BD, BN] := aft2LoadVal sin Kptr kmaskT with hkl
  set ksv : ℝ := sin.readMem (Ksptr.data PUnit.unit).1 (Ksptr.data PUnit.unit).2 with hksv
  set qkT : Tile .real [BM, BN] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile kloadT).data i)⟩ :
          Tile .real [BM, BN])
        (Tile.scalar (some qsv : WithBot ℝ))) (Tile.scalar (some ksv : WithBot ℝ)) with hqk
  obtain ⟨rmaxT, hrm⟩ :
      ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkT = some t :=
    ⟨_, by
      unfold Tile.reduceMaxDrop
      rw [dif_pos (show 0 < TileShape.axisDim [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length)
        from hBN)]⟩
  set mijT : Tile .real [BM] :=
    Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
      mtile rmaxT with hmij
  set qk2T : Tile .real [BM, BN] :=
    Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT
      (Tile.expandDim ⟨1, hax⟩ mijT) with hqk2
  set pT : Tile .real [BM, BN] := Tile.uop WithBot.realExp2 qk2T with hpT
  set lijT : Tile .real [BM] :=
    Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pT with hlij
  set alphaT : Tile .real [BM] := Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with hal
  set acc1T : Tile .real [BM, BD] :=
    Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile
      (Tile.expandDim ⟨1, hax⟩ alphaT) with hacc1
  set vmaskT : Tile .bool [BN, BD] :=
    ⟨fun idx => (decide (idx.1.val < BN * nB - SN) && decide (idx.2.1.val < HA))⟩ with hvm
  set vloadT : Tile .real [BN, BD] := aft2LoadVal sin Vptr vmaskT with hvl
  unfold attnLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by
        rw [evalOp_ref]; exact hsn)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (kmask_eval _ BD BN (BN * nB) SN HA (by simp [hoffs]) (by simp))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aft2IO_load_mask_val (Op.ref .ptr [BD, BN] "K_ptrs") _ _ sin Kptr kmaskT
        (by rw [evalOp_ref]; simp [hKp]) (by rw [evalOp_ref]; simp [hkm])
        (by rfl) (by rfl))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (load_ptr_none_real (Op.ref .ptr [] "K_scale_ptr") _ Ksptr (by rw [evalOp_ref]; simp [hKs]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qk_op_eval _ BM BN BD qtile kloadT qsv ksv (by simp [hq]) (by simp [hkl]) (by simp [hqs])
        (by simp [hksv]; ext z; rfl))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (mij_op_eval _ BM BN mtile qkT rmaxT (by simp [hmi]) (by simp [hqk]) hrm)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (qk2_op_eval _ BM BN hax qkT mijT (by simp [hqk]) (by simp [hmij]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (p_op_eval _ BM BN qk2T (by simp [hqk2]))),
    stepStmts.cons_some (@stepStmt_assign_eq_some .real [BM] "l_ij"
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p")) _
      lijT (lij_op_eval _ BM BN pT (by simp [hpT]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (alpha_op_eval _ BM mtile mijT (by simp [hmi]) (by simp [hmij]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (li_op_eval _ BM ltile alphaT lijT (by simp [hli]) (by simp [hal]) (by simp [hlij]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (acc1_op_eval _ BM BD hax acctile alphaT (by simp [hacc]) (by simp [hal]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aft2IO_load_mask_val (Op.ref .ptr [BN, BD] "V_ptrs") _ _ sin Vptr vmaskT
        (by rw [evalOp_ref]; simp [hVp])
        (by exact vmask_eval _ BN BD (BN * nB) SN HA (by simp [hoffs]) (by simp))
        (by rfl) (by rfl))),
    stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BM, BN] "p"
      (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p")) _
      (⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ : Tile .fp16 [BM, BN])
      (pfp16_op_eval _ BM BN pT (by simp [hpT]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (acc2_op_eval _ BM BN BD acc1T pT vloadT (by simp [hacc1]) (by simp [hpT]) (by simp [hvl]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .real [BM] "m_ij") _ = some mijT from by
        rw [evalOp_ref]; simp [hmij])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (kptr_adv_eval _ BD BN BN HD Kptr "K_ptrs" (by simp [hKp]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (ksptr_adv_eval _ Ksptr "K_scale_ptr" (by simp [hKs]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (kptr_adv_eval _ BN BD BN HD Vptr "V_ptrs" (by simp [hVp]))),
    stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · simp
  · simp
  · simp
  · simp [hq]
  · simp [hqs]
  · simp [hoffs]
  · simp
  · simp
  · exact ⟨mijT, by simp⟩
  · exact ⟨Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT) lijT,
      by simp⟩
  · exact ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      acc1T (Tile.dot [] pT vloadT), by simp⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body step**: the body advances `aft2SafeInv` by one key block. -/
private theorem aft2IO_stepW (K K_scale V Out : RegionName) (s0 : BlockState)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (c : Nat) (s : BlockState) (hc : c < BLOCK_N * numKVBlocks)
    (hP : aft2SafeInv K K_scale V Out s0 H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s) :
    ∃ s', stepStmts (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (s.setReg "start_n" .nat [] (Tile.scalar c)) = some s'
      ∧ aft2SafeInv K K_scale V Out s0 H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks (c + BLOCK_N) s' := by
  obtain ⟨hmod, hle, hoffsm, hoffsn, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩, ⟨qT, hq⟩, ⟨qsv, hqs⟩,
    hKp, hKsp, hVp, hOp, hmem⟩ := hP
  have hcBN : c / BLOCK_N * BLOCK_N = c := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
  have hcdivlt : c / BLOCK_N < numKVBlocks := by
    have h1 : c / BLOCK_N * BLOCK_N < numKVBlocks * BLOCK_N := by
      rw [hcBN, Nat.mul_comm numKVBlocks BLOCK_N]
      exact hc
    exact lt_of_mul_lt_mul_right h1 (Nat.zero_le BLOCK_N)
  have hdiv : (c + BLOCK_N) / BLOCK_N = c / BLOCK_N + 1 := by
    conv_lhs => rw [show c = c / BLOCK_N * BLOCK_N from hcBN.symm]
    rw [Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]
  have hle' : c + BLOCK_N ≤ BLOCK_N * numKVBlocks := by
    have h1 : (c / BLOCK_N + 1) * BLOCK_N ≤ numKVBlocks * BLOCK_N :=
      Nat.mul_le_mul_right BLOCK_N hcdivlt
    calc c + BLOCK_N = c / BLOCK_N * BLOCK_N + BLOCK_N := by rw [hcBN]
      _ = (c / BLOCK_N + 1) * BLOCK_N := by ring
      _ ≤ numKVBlocks * BLOCK_N := h1
      _ = BLOCK_N * numKVBlocks := Nat.mul_comm _ _
  obtain ⟨sF, hchain, hpidsF, hmemF, hFKp, hFKsp, hFVp, hFq, hFqs, hFoffsn, hFoffsm, hFOp,
    hFmi, hFli, hFacc⟩ :=
    aft2IO_loopBody_stepsW BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks HEAD_ACTIVE HEAD_DIM hBN
      (by simp) (s.setReg "start_n" .nat [] (Tile.scalar c)) c qsv
      (aft2KPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N))
      (aft2SPtr K_scale (s0.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N) (c / BLOCK_N))
      (aft2VPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N))
      qT mT lT aT
      (aft2_regs_setReg_chain (by decide) hoffsn)
      (BlockState.setReg_same _ _ _ _ _)
      (aft2_regs_setReg_chain (by decide) hKp)
      (aft2_regs_setReg_chain (by decide) hKsp)
      (aft2_regs_setReg_chain (by decide) hVp)
      (aft2_regs_setReg_chain (by decide) hq)
      (aft2_regs_setReg_chain (by decide) hqs)
      (aft2_regs_setReg_chain (by decide) hmi)
      (aft2_regs_setReg_chain (by decide) hli)
      (aft2_regs_setReg_chain (by decide) hacc)
  refine ⟨sF, hchain, ?_, hle', ?_, hFoffsn, hFmi, hFli, hFacc, ⟨qT, hFq⟩, ⟨qsv, hFqs⟩, ?_, ?_,
    ?_, ?_, ?_⟩
  · rw [Nat.add_mod_right]; exact hmod
  · rw [hFoffsm, aft2_regs_setReg_chain (by decide) hoffsm]
  · rw [hFKp, aft2KPtrs_succ, hdiv]
  · rw [hFKsp, aft2SPtr_succ, hdiv]
  · rw [hFVp, aft2VPtrs_succ, hdiv]
  · rw [hFOp, aft2_regs_setReg_chain (by decide) hOp]
  · rw [hmemF]; exact hmem

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body safety**: from an `aft2SafeInv` state every statement of the
streamed body is trace-safe — the `k`/`k_scale`/`v` loads' active lanes are
bounded by the skin's `read2`/`read5`/`read3` window bounds. -/
private theorem aft2IO_bodySafeW (R : RoundingModel) (bounds : RegionBounds)
    (K K_scale V Out : RegionName) (s0 : BlockState)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (c : Nat) (s : BlockState) (hc : c < BLOCK_N * numKVBlocks)
    (hP : aft2SafeInv K K_scale V Out s0 H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s)
    (hbK : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      aft2Base s0 H stride_qz stride_qh
        + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM < bounds K)
    (hbKS : ∀ (t : Fin numKVBlocks) (_j : Fin 1),
      s0.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N + t.val < bounds K_scale)
    (hbV : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧
        j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      aft2Base s0 H stride_qz stride_qh
        + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
          < bounds V) :
    Stmt.TraceSafeListR R bounds
      (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
      (s.setReg "start_n" .nat [] (Tile.scalar c)) := by
  obtain ⟨hmod, hle, hoffsm, hoffsn, -, -, -, -, -, hKp, hKsp, hVp, hOp, -⟩ := hP
  have hcBN : c / BLOCK_N * BLOCK_N = c := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
  have hcdivlt : c / BLOCK_N < numKVBlocks := by
    have h1 : c / BLOCK_N * BLOCK_N < numKVBlocks * BLOCK_N := by
      rw [hcBN, Nat.mul_comm numKVBlocks BLOCK_N]
      exact hc
    exact lt_of_mul_lt_mul_right h1 (Nat.zero_le BLOCK_N)
  unfold attnLoopBody
  -- (0) start_n = start_n (identity)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 h1 => ?_)
  obtain ⟨v0, hv0, rfl⟩ := stepStmtR_assign_inv h1
  rw [evalOpR_ref, BlockState.setReg_same] at hv0
  obtain rfl := Option.some.inj hv0
  -- (1) k_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 h2 => ?_)
  obtain ⟨vkm, hvkm, rfl⟩ := stepStmtR_assign_inv h2
  rw [aft2IO_kmask_evalR R _ c (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_N BLOCK_DMODEL
    (aft2_regs_setReg_chain (by decide) (aft2_regs_setReg_chain (by decide) hoffsn))
    (BlockState.setReg_same _ _ _ _ _)] at hvkm
  obtain rfl := Option.some.inj hvkm
  -- (2) k = tl.load(K_ptrs, mask = k_mask): the `read2` window bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 h3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs idx hact
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hKp] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hidx⟩ := hact
    rw [evalOpR_ref, BlockState.setReg_same] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hP2 : idx.2.1.val < BLOCK_N * numKVBlocks - c ∧ idx.1.val < HEAD_ACTIVE := by
      simpa using hidx
    have hbound := hbK ⟨c / BLOCK_N, hcdivlt⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
      rw [hcBN]
      exact ⟨hP2.1, hP2.2⟩)
    simpa [aft2KPtrs, Lane2D.encode_div, Lane2D.encode_mod, hcBN] using hbound
  obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
  -- (3) k_scale = tl.load(K_scale_ptr): the `read5` slot bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s4 h4 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hKsp] at hptrs
    obtain rfl := Option.some.inj hptrs
    have hbound := hbKS ⟨c / BLOCK_N, hcdivlt⟩ 0
    simpa [aft2SPtr] using hbound
  obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv h4
  -- (4)–(11): register-only assigns
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s5 h5 => ?_)
  obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv h5
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s6 h6 => ?_)
  obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv h6
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s7 h7 => ?_)
  obtain ⟨v7, -, rfl⟩ := stepStmtR_assign_inv h7
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s8 h8 => ?_)
  obtain ⟨v8, -, rfl⟩ := stepStmtR_assign_inv h8
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s9 h9 => ?_)
  obtain ⟨v9, -, rfl⟩ := stepStmtR_assign_inv h9
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s10 h10 => ?_)
  obtain ⟨v10, -, rfl⟩ := stepStmtR_assign_inv h10
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s11 h11 => ?_)
  obtain ⟨v11, -, rfl⟩ := stepStmtR_assign_inv h11
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s12 h12 => ?_)
  obtain ⟨v12, -, rfl⟩ := stepStmtR_assign_inv h12
  -- (12) v = tl.load(V_ptrs, mask = …): the `read3` window bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s13 h13 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hact
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hVp] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hidx⟩ := hact
    rw [aft2IO_vmask_evalR R _ c (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_N BLOCK_DMODEL
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
          exact hoffsn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same])] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hP2 : idx.1.val < BLOCK_N * numKVBlocks - c ∧ idx.2.1.val < HEAD_ACTIVE := by
      simpa using hidx
    have hbound := hbV ⟨c / BLOCK_N, hcdivlt⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
      rw [hcBN]
      exact ⟨hP2.1, hP2.2⟩)
    simpa [aft2VPtrs, Lane2D.encode_div, Lane2D.encode_mod, hcBN] using hbound
  obtain ⟨v13, -, rfl⟩ := stepStmtR_assign_inv h13
  -- (13)–(18): register-only assigns
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s14 h14 => ?_)
  obtain ⟨v14, -, rfl⟩ := stepStmtR_assign_inv h14
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s15 h15 => ?_)
  obtain ⟨v15, -, rfl⟩ := stepStmtR_assign_inv h15
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s16 h16 => ?_)
  obtain ⟨v16, -, rfl⟩ := stepStmtR_assign_inv h16
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s17 h17 => ?_)
  obtain ⟨v17, -, rfl⟩ := stepStmtR_assign_inv h17
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s18 h18 => ?_)
  obtain ⟨v18, -, rfl⟩ := stepStmtR_assign_inv h18
  exact Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak postLoop safety**: the `acc /= l_i` divide is register-only and the
masked terminal store's active lanes are bounded by the skin's `write` window. -/
private theorem aft2IO_postSafeW (R : RoundingModel) (bounds : RegionBounds)
    (K K_scale V Out : RegionName) (s0 : BlockState)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (c : Nat) (s : BlockState)
    (hP : aft2SafeInv K K_scale V Out s0 H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s)
    (hbO : ∀ j : Fin (BLOCK_M * BLOCK_DMODEL),
      s0.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks ∧
        j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      aft2Base s0 H stride_qz stride_qh
        + (s0.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
          < bounds Out) :
    Stmt.TraceSafeListR R bounds
      (aft2PostLoop BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) s := by
  obtain ⟨-, -, hoffsm, -, -, -, -, -, -, -, -, -, hOp, -⟩ := hP
  unfold aft2PostLoop
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [attnAccAssign, Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 h1 => ?_)
  rw [attnAccAssign] at h1
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [attnStoreStmt, Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
    Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, by simp [Op.SafeAtR.eq_def], ?_⟩
  intro ptrs hptrs idx hact
  rw [evalOpR_ref] at hptrs
  simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
  rw [hOp] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨masks, hmasks, hidx⟩ := hact
  rw [aft2IO_qmask_evalR R _ (s0.pids 0) (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_M BLOCK_DMODEL
    (aft2_regs_setReg_chain (by decide) hoffsm)] at hmasks
  obtain rfl := Option.some.inj hmasks
  have hP2 : s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks ∧
      idx.2.1.val < HEAD_ACTIVE := by simpa using hidx
  have hbound := hbO (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
    simp only [Lane2D.encode_div, Lane2D.encode_mod]
    exact hP2)
  simpa [aft2RowPtr, Lane2D.encode_div, Lane2D.encode_mod] using hbound

/-- The load-free prologue prefix collapses onto the exact stepper. -/
private theorem aft2IO_pre20_castFree (R : RoundingModel)
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (t : BlockState) :
    stepStmtsR R (aft2Pre20 Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M
        BLOCK_N BLOCK_DMODEL numKVBlocks) t
      = stepStmts (aft2Pre20 Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M
        BLOCK_N BLOCK_DMODEL numKVBlocks) t :=
  aft2IO_stepStmtsR_castFree_of_stmts R _
    (fun st hst u => aft2IO_preLoop_stmt_castFree R Q K V Q_scale K_scale Out stride_qz stride_qh
      H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks st
      (by rw [aft2PreLoop]; exact List.mem_append_left _ hst) u) t

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `TraceSafeR` walk for the whole kernel**: the 20 load-free prologue
assigns are safe at every state, the `q`/`q_scale` loads are bounded by the
`read1`/`read4` windows at the walked prefix state, the KV loop runs
`Stmt.forRangeTraceSafeR_inv` over `aft2SafeInv`, and the terminal store is
bounded by the `write` window. -/
private theorem aft2IO_traceSafeR (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (bounds : RegionBounds) (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hbQ : ∀ (_t : Fin numKVBlocks) (j : Fin (BLOCK_M * BLOCK_DMODEL)),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks ∧
        j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      aft2Base s H stride_qz stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
          < bounds Q)
    (hbK : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      aft2Base s H stride_qz stride_qh
        + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM < bounds K)
    (hbV : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧
        j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      aft2Base s H stride_qz stride_qh
        + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds V)
    (hbQS : ∀ (_t : Fin numKVBlocks) (_j : Fin 1),
      s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_M + s.pids 0 < bounds Q_scale)
    (hbKS : ∀ (t : Fin numKVBlocks) (_j : Fin 1),
      s.pids 1 * cdiv (BLOCK_N * numKVBlocks) BLOCK_N + t.val < bounds K_scale)
    (hbO : ∀ j : Fin (BLOCK_M * BLOCK_DMODEL),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks ∧
        j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      aft2Base s H stride_qz stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
          < bounds Out) :
    ((attention_fwd_triton2_surface Q K V Q_scale K_scale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [aft2_body_split]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the prologue: 20 register-only assigns then the two bounded loads
    rw [aft2PreLoop]
    refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
    · exact Stmt.TraceSafeListR.of_forall _ _
        (aft2IO_pre20_stmt_safe R bounds Q K V Q_scale K_scale Out stride_qz stride_qh H
          HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks)
    · intro s2 hs2
      obtain ⟨s20, h20, hpids20, hmem20, hoffsm, hoffsn, -, -, -, hqp, hqsp, -, -, -, -⟩ :=
        aft2IO_pre20_evalW s Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M
          BLOCK_N BLOCK_DMODEL numKVBlocks
      rw [aft2IO_pre20_castFree R Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks s, h20] at hs2
      obtain rfl := Option.some.inj hs2
      rw [aft2Loads]
      -- the `q` load: the `read1` window bound
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun s21 h21 => ?_)
      · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
        refine ⟨trivial, by simp [Op.SafeAtR.eq_def], ?_⟩
        intro ptrs hptrs idx hact
        rw [evalOpR_ref, hqp] at hptrs
        obtain rfl := Option.some.inj hptrs
        obtain ⟨masks, hmasks, hidx⟩ := hact
        rw [aft2IO_qmask_evalR R _ (s.pids 0) (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_M
          BLOCK_DMODEL hoffsm] at hmasks
        obtain rfl := Option.some.inj hmasks
        have hP2 : s.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks ∧
            idx.2.1.val < HEAD_ACTIVE := by simpa using hidx
        have hbound := hbQ ⟨0, hnum⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
          simp only [Lane2D.encode_div, Lane2D.encode_mod]
          exact hP2)
        simpa [aft2RowPtr, Lane2D.encode_div, Lane2D.encode_mod] using hbound
      obtain ⟨vq, -, rfl⟩ := stepStmtR_assign_inv h21
      -- the `q_scale` load: the `read4` slot bound
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [aft2_regs_setReg_chain (by decide) hqsp] at hptrs
      obtain rfl := Option.some.inj hptrs
      have hbound := hbQS ⟨0, hnum⟩ 0
      simpa [aft2SPtr] using hbound
  · -- after the prologue: the KV loop, then the postLoop
    intro sp hsp
    obtain ⟨spW, hpreW, hinv0⟩ :=
      aft2IO_preLoop_evalW s Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM BLOCK_M
        BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
    rw [aft2IO_preLoop_castFree R Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks s, hpreW] at hsp
    obtain rfl := Option.some.inj hsp
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 hs3 => ?_)
    · show Stmt.TraceSafeR R bounds _ spW
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "start_n" (BLOCK_N * numKVBlocks) BLOCK_N
        (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (aft2SafeInv K K_scale V Out s H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks)
        ?_ 0 spW hinv0
      intro c st hc hPc
      refine ⟨aft2IO_bodySafeW R bounds K K_scale V Out s H stride_qz stride_qh HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks c st hc hPc hbK hbKS hbV, ?_⟩
      obtain ⟨st', hstep, hPc'⟩ := aft2IO_stepW K K_scale V Out s H stride_qz stride_qh
        HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc
      exact ⟨st', by
        rw [aft2IO_loopBody_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
          numKVBlocks]
        exact hstep, hPc'⟩
    · -- identify the post-loop state and finish on the terminal store
      obtain ⟨final, sfin, hLoop, hfinal, hPfin⟩ :=
        forRange_inv (idx := "start_n") (start := 0) (stop := BLOCK_N * numKVBlocks)
          (step := BLOCK_N)
          (body := attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
          (P := aft2SafeInv K K_scale V Out s H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
            BLOCK_DMODEL numKVBlocks)
          (s_init := spW) hBN.ne' hinv0
          (fun c st hc hPc => aft2IO_stepW K K_scale V Out s H stride_qz stride_qh HEAD_DIM
            BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc)
      rw [show stepStmtR R (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
            (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) spW
          = some sfin from by
        rw [aft2IO_loopStmt_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
          numKVBlocks spW]
        exact hLoop] at hs3
      obtain rfl := Option.some.inj hs3
      exact aft2IO_postSafeW R bounds K K_scale V Out s H stride_qz stride_qh HEAD_DIM BLOCK_M
        BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks final sfin hPfin hbO

/-! ## The postLoop frame

The exec stack delivers the readback values; the skin additionally needs the
per-cell frame (every cell outside the masked `Out` window is untouched). -/

/-- A `P`-masked single-region `writeMem` scatter preserves every cell no active
lane hits. -/
private theorem aft2IO_foldl_writeMem_frame_masked {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) (P : α → Prop) [DecidablePred P] :
    ∀ (l : List α) (s : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, P k → offFn k ≠ o) →
      ((l.foldl (fun acc k => if P k then acc.writeMem region (offFn k) (valFn k) else acc) s).mem
        r o = s.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, s, r, o, h => by
      rw [List.foldl_cons]
      by_cases hPk : P k
      · rw [if_pos hPk,
          aft2IO_foldl_writeMem_frame_masked region offFn valFn P rest _ r o
            (fun hr k' hk' hP' => h hr k' (List.mem_cons_of_mem _ hk') hP'),
          BlockState.writeMem_mem]
        rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hPk hro.2.symm)]
      · rw [if_neg hPk]
        exact aft2IO_foldl_writeMem_frame_masked region offFn valFn P rest s r o
          (fun hr k' hk' hP' => h hr k' (List.mem_cons_of_mem _ hk') hP')

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **PostLoop run + frame**: the two postLoop statements step (the `acc /= l_i`
divide is register-only, the terminal store scatters the masked `Out` window),
and every cell outside the write-active window is untouched. -/
private theorem aft2IO_postLoopW (Out : RegionName) (s0 : BlockState)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (s : BlockState) (aT : Tile .real [BLOCK_M, BLOCK_DMODEL]) (lT : Tile .real [BLOCK_M])
    (hoffsm : s.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)))
    (hOp : s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      = some (aft2RowPtr s0 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL))
    (hacc : s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT)
    (hli : s.regs .real [BLOCK_M] "l_i" = some lT) :
    ∃ sP, stepStmts (aft2PostLoop BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) s
        = some sP
      ∧ ∀ r o,
        (r = Out → ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          (s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks ∧
            idx.2.1.val < HEAD_ACTIVE) →
          o ≠ aft2Base s0 H stride_qz stride_qh
            + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val) →
        sP.mem r o = s.mem r o := by
  have hax : (1 : Nat) < [BLOCK_M].length.succ := by simp
  have hexpN : evalOp (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i")) s
      = some (Tile.expandDim ⟨1, hax⟩ lT) := by rw [evalOp_expandDim]; simp [hli]
  have hexp2 : @evalOp TileDType.real [BLOCK_M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i")) s
      = some (Tile.expandDim ⟨1, hax⟩ lT) := hexpN
  have hdiv : evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i"))) s
      = some (Tile.bop NumericDType.real.div
          (Broadcast.consSame (Broadcast.consR Broadcast.nil)) aT
          (Tile.expandDim ⟨1, hax⟩ lT)) := by
    rw [evalOp_div]
    simp only [evalOp_ref, hacc, hexp2, Option.bind_eq_bind, Option.bind_some]
    rfl
  set v1 : Tile .real [BLOCK_M, BLOCK_DMODEL] := Tile.bop NumericDType.real.div
    (Broadcast.consSame (Broadcast.consR Broadcast.nil)) aT
    (Tile.expandDim ⟨1, hax⟩ lT) with hv1
  set s2 : BlockState := s.setReg "acc" .real [BLOCK_M, BLOCK_DMODEL] v1 with hs2
  have hacc2 : s2.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some v1 := by
    rw [hs2]; simp only [BlockState.setReg_same]
  have hOp2 : s2.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      = some (aft2RowPtr s0 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL) := by
    rw [hs2]; exact aft2_regs_setReg_chain (by decide) hOp
  have hoffsm2 : s2.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)) := by
    rw [hs2]; exact aft2_regs_setReg_chain (by decide) hoffsm
  set oOffFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx => aft2Base s0 H stride_qz stride_qh
      + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val with hoOffFn
  set P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks ∧
      idx.2.1.val < HEAD_ACTIVE with hP
  have hopEval : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr") s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out.cast, oOffFn idx)⟩ :
          Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref, hOp2]
    refine congrArg some ?_
    ext idx
    · rfl
    · simp only [aft2RowPtr, hoOffFn]
  have hmaskEval : evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
          (Op.constNat (BLOCK_N * numKVBlocks)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => decide (P idx)⟩ :
          Tile .bool [BLOCK_M, BLOCK_DMODEL]) := by
    rw [qmask_eval s2 BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
      (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val) hoffsm2]
    refine congrArg some ?_
    ext idx
    rw [Bool.eq_iff_iff]
    simp only [hP, Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨(TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
      (fun acc idx => if P idx then acc.writeMem Out (oOffFn idx) ((v1.data idx).unbotD 0)
        else acc) s2, ?_, ?_⟩
  · rw [aft2PostLoop, attnAccAssign, attnStoreStmt,
      stepStmts.cons_some (stepStmt_assign_eq_some hdiv),
      stepStmts.cons_some (show stepStmt (Stmt.store .real [BLOCK_M, BLOCK_DMODEL]
          (.ptr (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"))
          (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
          (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
              (Op.constNat (BLOCK_N * numKVBlocks)))
            (Op.expandDim ⟨0, by simp⟩
              (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
                (Op.constNat HEAD_ACTIVE)))))) s2
          = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
              (fun acc idx => if P idx then
                acc.writeMem Out (oOffFn idx) ((v1.data idx).unbotD 0) else acc) s2) from by
        simp only [stepStmt, evalOp_ref, hacc2, hopEval, hmaskEval, Option.bind_eq_bind,
          Option.bind_some, Option.map_some, decide_eq_true_eq]
        refine congrArg some ?_
        refine List.foldl_ext _ _ s2 ?_
        intro acc idx _
        by_cases hk : P idx
        · simp only [if_pos hk, Region.cast_id, BlockState.writeMemTyped_real,
            FloatDType.real_storeValue]
        · simp only [if_neg hk]),
      stepStmts.nil]
  · intro r o hOguard
    rw [aft2IO_foldl_writeMem_frame_masked Out oOffFn (fun idx => (v1.data idx).unbotD 0) P
      (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]) s2 r o
      (fun hr k _ hPk heq => hOguard hr k hPk heq.symm)]
    rw [hs2]
    simp only [BlockState.setReg_mem]

/-! ## IO signature, stream tiles, and the closed-form spec `f`

Window transcription (strides pinned to the exec stack's contiguous layout
`stride_qm = stride_kn = stride_vk = stride_om = HEAD_DIM`, unit fastest stride;
shared plane base `p₁/H·stride_qz + p₁%H·stride_qh`):

* `read1` (`Q`, static — the window ignores `t`): lane `j = (i, e)` row-major
  over `[BLOCK_M, BLOCK_DMODEL]` reads `base + (p₀·BM + i)·HEAD_DIM + e`, masked
  `p₀·BM + i < N_CTX ∧ e < HEAD_ACTIVE` (the kernel's `q` load mask).
* `read2` (`K`, transposed, advanced `BLOCK_N` keys per step): lane `j = (e, r)`
  row-major over `[BLOCK_DMODEL, BLOCK_N]` reads `base + e + (t·BN + r)·HEAD_DIM`,
  masked `r < N_CTX ⊖ t·BN ∧ e < HEAD_ACTIVE` (`⊖` = the kernel's ℕ-truncated
  `N_CTX - start_n`, transcribed verbatim).
* `read3` (`V`, advanced `BLOCK_N` rows per step): lane `j = (r, d)` reads
  `base + (t·BN + r)·HEAD_DIM + d`, masked `r < N_CTX ⊖ t·BN ∧ d < HEAD_ACTIVE`.
* `read4` (`Q_scale`, **scalar**, static): the single lane reads
  `p₁·⌈N_CTX/BM⌉ + p₀`, unmasked.
* `read5` (`K_scale`, **scalar**, one slot per step): the single lane reads
  `p₁·⌈N_CTX/BN⌉ + t`, unmasked.
* `write` (`Out`): lane `(i, e)` writes `base + (p₀·BM + i)·HEAD_DIM + e` under
  the genuine store mask `p₀·BM + i < N_CTX ∧ e < HEAD_ACTIVE`.
* `outDType := .real`: the surface's `.to(Out.type.element_ty)` **erases at
  translation** (the DSL drops `X.to(<ptr>.type.element_ty)` unless the cast is
  to an integral type), so the lowered terminal statement is a plain
  `Stmt.store .real` — the host's `torch.bfloat16` allocation is invisible to
  the model and the terminal grid is exact. -/

/-- **Streaming IO signature** of `_attn_fwd` on the five-stream single-store
attention fold skin (`T = numKVBlocks` under `N_CTX = BLOCK_N · numKVBlocks`;
the trip count is pid-free, so there is no launch-legality `pre` analog). -/
def attentionFwdTriton2IO (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := attention_fwd_triton2_surface Q K V Q_scale K_scale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := Q_scale
  inp5 := K_scale
  out := Out
  T := numKVBlocks
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  B4 := 1
  B5 := 1
  C := BLOCK_M * BLOCK_DMODEL
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read2 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM
  read3 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read4 := fun p₀ p₁ _ _ _ => p₁ * cdiv (BLOCK_N * numKVBlocks) BLOCK_M + p₀
  read5 := fun _ p₁ _ t _ => p₁ * cdiv (BLOCK_N * numKVBlocks) BLOCK_N + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks ∧
      j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧
      j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks ∧
      j.val % BLOCK_DMODEL < HEAD_ACTIVE

/-- The **query tile** read off the static first stream (the window ignores `t`,
so the step-`0` slice carries the whole tile). -/
noncomputable def aft2IOqT (BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : 0 < T ∧ idx.1.val * BLOCK_DMODEL + idx.2.1.val < BLOCK_M * BLOCK_DMODEL then
      xs ⟨0, h.1⟩ ⟨idx.1.val * BLOCK_DMODEL + idx.2.1.val, h.2⟩
    else 0

/-- The **global key tile** read off the transposed `K` stream: global key `jg`
lives in step `jg / BLOCK_N` at block-local column `jg % BLOCK_N`. -/
noncomputable def aft2IOkT (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.1.val / BLOCK_N < T ∧
        idx.2.1.val * BLOCK_N + idx.1.val % BLOCK_N < BLOCK_DMODEL * BLOCK_N then
      ys ⟨idx.1.val / BLOCK_N, h.1⟩ ⟨idx.2.1.val * BLOCK_N + idx.1.val % BLOCK_N, h.2⟩
    else 0

/-- The **global value tile** read off the `V` stream (row-major per-step
tiles). -/
noncomputable def aft2IOvT (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.1.val / BLOCK_N < T ∧
        idx.1.val % BLOCK_N * BLOCK_DMODEL + idx.2.1.val < BLOCK_N * BLOCK_DMODEL then
      zs ⟨idx.1.val / BLOCK_N, h.1⟩ ⟨idx.1.val % BLOCK_N * BLOCK_DMODEL + idx.2.1.val, h.2⟩
    else 0

/-- The per-key score-scale carrier `q_scale · k_scale` read off the two
scalar-width streams: the static `Q_scale` slot (step `0`) times key `j`'s
`K_scale` slot (step `j / BLOCK_N`). -/
noncomputable def aft2IOkeyScale (BLOCK_N T : Nat) (ws vs : Fin T → Fin 1 → ℝ) :
    Fin (BLOCK_N * T) → ℝ :=
  fun j =>
    (if h : 0 < T then ws ⟨0, h⟩ 0 else 0)
      * (if h : j.val / BLOCK_N < T then vs ⟨j.val / BLOCK_N, h⟩ 0 else 0)

/-- **The streamed closed-form spec `f`**: base-2 per-key-scale attention of the
streamed Q/K/V tiles under the streamed per-block key scale. Head-inactive
output lanes (which the store masks off) are `0`. -/
noncomputable def attentionFwdTriton2IOOutSpec
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  if h : idx.2.1.val < HEAD_ACTIVE then
    attentionRealBase2PerKeyScale
      (aft2IOqT BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs)
      (aft2IOkT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys)
      (aft2IOvT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs)
      (aft2IOkeyScale BLOCK_N T ws vs)
      (idx.1, ⟨idx.2.1.val, h⟩, PUnit.unit)
  else 0

/-! ### Stream-pin tile bridges: under the skin's input pins the kernel-side
tiles/scales of the exec stack coincide with the stream-built ones (the query
tile only on the row the spec actually consumes — off-row lanes are outside the
kernel's `q` load mask). -/

private theorem aft2IOkT_eq (s₀ : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (hy : ∀ (t : Fin T) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * T - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      s₀.readMem K (s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
          + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM) = ys t j) :
    kTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * T) HEAD_ACTIVE
      = aft2IOkT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys := by
  funext idx
  obtain ⟨jg, e, ⟨⟩⟩ := idx
  have hjT : jg.val / BLOCK_N < T :=
    (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm T BLOCK_N]; exact jg.isLt)
  have hmod : jg.val % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
  have he : e.val < BLOCK_DMODEL := lt_of_lt_of_le e.isLt hActiveLe
  have hlane : e.val * BLOCK_N + jg.val % BLOCK_N < BLOCK_DMODEL * BLOCK_N := by
    calc e.val * BLOCK_N + jg.val % BLOCK_N < e.val * BLOCK_N + BLOCK_N := by omega
      _ = (e.val + 1) * BLOCK_N := by ring
      _ ≤ BLOCK_DMODEL * BLOCK_N := Nat.mul_le_mul_right _ he
  have hblk : jg.val / BLOCK_N * BLOCK_N + BLOCK_N ≤ BLOCK_N * T := by
    calc jg.val / BLOCK_N * BLOCK_N + BLOCK_N = (jg.val / BLOCK_N + 1) * BLOCK_N := by ring
      _ ≤ T * BLOCK_N := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hjT)
      _ = BLOCK_N * T := Nat.mul_comm _ _
  have hval : ((⟨e.val * BLOCK_N + jg.val % BLOCK_N, hlane⟩ :
      Fin (BLOCK_DMODEL * BLOCK_N)) : Nat) = e.val * BLOCK_N + jg.val % BLOCK_N := rfl
  have hvalt : ((⟨jg.val / BLOCK_N, hjT⟩ : Fin T) : Nat) = jg.val / BLOCK_N := rfl
  have hdiv1 : (e.val * BLOCK_N + jg.val % BLOCK_N) / BLOCK_N = e.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hBN, Nat.div_eq_of_lt hmod, Nat.zero_add]
  have hmod1 : (e.val * BLOCK_N + jg.val % BLOCK_N) % BLOCK_N = jg.val % BLOCK_N := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hmod]
  simp only [aft2IOkT]
  rw [dif_pos (⟨hjT, hlane⟩ : jg.val / BLOCK_N < T ∧
    e.val * BLOCK_N + jg.val % BLOCK_N < BLOCK_DMODEL * BLOCK_N)]
  rw [← hy ⟨jg.val / BLOCK_N, hjT⟩ ⟨e.val * BLOCK_N + jg.val % BLOCK_N, hlane⟩ (by
    rw [hval, hvalt, hdiv1, hmod1]
    exact ⟨by omega, e.isLt⟩)]
  simp only [kTile, baseOffset, offZ, offH, hval, hvalt, hdiv1, hmod1]
  refine congrArg (s₀.readMem K) ?_
  rw [show jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N = jg.val from
    Nat.div_add_mod' jg.val BLOCK_N]
  ring

private theorem aft2IOvT_eq (s₀ : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (hz : ∀ (t : Fin T) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * T - t.val * BLOCK_N ∧
        j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s₀.readMem V (s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
          + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL)
        = zs t j) :
    vTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * T) HEAD_ACTIVE
      = aft2IOvT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs := by
  funext idx
  obtain ⟨jg, d, ⟨⟩⟩ := idx
  have hd : d.val < BLOCK_DMODEL := lt_of_lt_of_le d.isLt hActiveLe
  have hBD : 0 < BLOCK_DMODEL := Nat.lt_of_le_of_lt (Nat.zero_le _) hd
  have hjT : jg.val / BLOCK_N < T :=
    (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm T BLOCK_N]; exact jg.isLt)
  have hmod : jg.val % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
  have hlane : jg.val % BLOCK_N * BLOCK_DMODEL + d.val < BLOCK_N * BLOCK_DMODEL := by
    calc jg.val % BLOCK_N * BLOCK_DMODEL + d.val
        < jg.val % BLOCK_N * BLOCK_DMODEL + BLOCK_DMODEL := by omega
      _ = (jg.val % BLOCK_N + 1) * BLOCK_DMODEL := by ring
      _ ≤ BLOCK_N * BLOCK_DMODEL := Nat.mul_le_mul_right _ hmod
  have hblk : jg.val / BLOCK_N * BLOCK_N + BLOCK_N ≤ BLOCK_N * T := by
    calc jg.val / BLOCK_N * BLOCK_N + BLOCK_N = (jg.val / BLOCK_N + 1) * BLOCK_N := by ring
      _ ≤ T * BLOCK_N := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hjT)
      _ = BLOCK_N * T := Nat.mul_comm _ _
  have hval : ((⟨jg.val % BLOCK_N * BLOCK_DMODEL + d.val, hlane⟩ :
      Fin (BLOCK_N * BLOCK_DMODEL)) : Nat) = jg.val % BLOCK_N * BLOCK_DMODEL + d.val := rfl
  have hvalt : ((⟨jg.val / BLOCK_N, hjT⟩ : Fin T) : Nat) = jg.val / BLOCK_N := rfl
  have hdiv1 : (jg.val % BLOCK_N * BLOCK_DMODEL + d.val) / BLOCK_DMODEL = jg.val % BLOCK_N := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hBD, Nat.div_eq_of_lt hd, Nat.zero_add]
  have hmod1 : (jg.val % BLOCK_N * BLOCK_DMODEL + d.val) % BLOCK_DMODEL = d.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]
  simp only [aft2IOvT]
  rw [dif_pos (⟨hjT, hlane⟩ : jg.val / BLOCK_N < T ∧
    jg.val % BLOCK_N * BLOCK_DMODEL + d.val < BLOCK_N * BLOCK_DMODEL)]
  rw [← hz ⟨jg.val / BLOCK_N, hjT⟩ ⟨jg.val % BLOCK_N * BLOCK_DMODEL + d.val, hlane⟩ (by
    rw [hval, hvalt, hdiv1, hmod1]
    exact ⟨by omega, d.isLt⟩)]
  simp only [vTile, baseOffset, offZ, offH, hval, hvalt, hdiv1, hmod1]
  refine congrArg (s₀.readMem V) ?_
  rw [show jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N = jg.val from
    Nat.div_add_mod' jg.val BLOCK_N]

private theorem aft2IOkeyScale_eq (s₀ : BlockState) (Q_scale K_scale : RegionName)
    (BLOCK_M BLOCK_N T : Nat) (hT : 0 < T) (hBN : 0 < BLOCK_N)
    (ws vs : Fin T → Fin 1 → ℝ)
    (hw : ∀ (t : Fin T) (j : Fin 1),
      s₀.readMem Q_scale (s₀.pids 1 * cdiv (BLOCK_N * T) BLOCK_M + s₀.pids 0) = ws t j)
    (hv : ∀ (t : Fin T) (j : Fin 1),
      s₀.readMem K_scale (s₀.pids 1 * cdiv (BLOCK_N * T) BLOCK_N + t.val) = vs t j) :
    keyScale s₀ Q_scale K_scale (BLOCK_N * T) BLOCK_M BLOCK_N (BLOCK_N * T)
      = aft2IOkeyScale BLOCK_N T ws vs := by
  funext j
  have hjT : j.val / BLOCK_N < T :=
    (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm T BLOCK_N]; exact j.isLt)
  simp only [keyScale, aft2IOkeyScale]
  rw [dif_pos hT, dif_pos hjT, ← hw ⟨0, hT⟩ 0, ← hv ⟨j.val / BLOCK_N, hjT⟩ 0]

/-- On an **active query row** the stream-built query tile agrees with the exec
stack's `qTile` (off-row lanes sit outside the kernel's `q` load mask, and the
closed form only consults the output lane's own row). -/
private theorem aft2IOqT_row_eq (s₀ : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T N_CTX : Nat)
    (hT : 0 < T) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (hx : ∀ (t : Fin T) (j : Fin (BLOCK_M * BLOCK_DMODEL)),
      s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < N_CTX ∧
        j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s₀.readMem Q (s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
          + (s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL)
        = xs t j)
    (i : Fin BLOCK_M) (hrow : s₀.pids 0 * BLOCK_M + i.val < N_CTX) (e : Fin HEAD_ACTIVE) :
    qTile s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE (i, e, PUnit.unit)
      = aft2IOqT BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs (i, e, PUnit.unit) := by
  have he : e.val < BLOCK_DMODEL := lt_of_lt_of_le e.isLt hActiveLe
  have hBD : 0 < BLOCK_DMODEL := Nat.lt_of_le_of_lt (Nat.zero_le _) he
  have hlane : i.val * BLOCK_DMODEL + e.val < BLOCK_M * BLOCK_DMODEL := by
    calc i.val * BLOCK_DMODEL + e.val < i.val * BLOCK_DMODEL + BLOCK_DMODEL := by omega
      _ = (i.val + 1) * BLOCK_DMODEL := by ring
      _ ≤ BLOCK_M * BLOCK_DMODEL := Nat.mul_le_mul_right _ i.isLt
  have hval : ((⟨i.val * BLOCK_DMODEL + e.val, hlane⟩ : Fin (BLOCK_M * BLOCK_DMODEL)) : Nat)
      = i.val * BLOCK_DMODEL + e.val := rfl
  have hdiv1 : (i.val * BLOCK_DMODEL + e.val) / BLOCK_DMODEL = i.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hBD, Nat.div_eq_of_lt he, Nat.zero_add]
  have hmod1 : (i.val * BLOCK_DMODEL + e.val) % BLOCK_DMODEL = e.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt he]
  simp only [aft2IOqT]
  rw [dif_pos (⟨hT, hlane⟩ : 0 < T ∧ i.val * BLOCK_DMODEL + e.val < BLOCK_M * BLOCK_DMODEL)]
  rw [← hx ⟨0, hT⟩ ⟨i.val * BLOCK_DMODEL + e.val, hlane⟩ (by
    rw [hval, hdiv1, hmod1]
    exact ⟨hrow, e.isLt⟩)]
  simp only [qTile, baseOffset, offZ, offH, mIndex, hval, hdiv1, hmod1]

/-- The closed form consults `Q` only at the output lane's own row, so a
row-local agreement suffices. -/
private theorem aft2IO_spec_congrQrow {M S D : Nat}
    (Q Q' : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ) (kS : Fin S → ℝ)
    (idx : TileIndex [M, D])
    (hrow : ∀ e : Fin D, Q (idx.1, e, PUnit.unit) = Q' (idx.1, e, PUnit.unit)) :
    attentionRealBase2PerKeyScale Q K V kS idx
      = attentionRealBase2PerKeyScale Q' K V kS idx := by
  obtain ⟨i, d, ⟨⟩⟩ := idx
  have hraw : ∀ j : Fin S,
      Finset.univ.sum (fun e : Fin D => Q (i, e, PUnit.unit) * K (j, e, PUnit.unit))
        = Finset.univ.sum (fun e : Fin D => Q' (i, e, PUnit.unit) * K (j, e, PUnit.unit)) :=
    fun j => Finset.sum_congr rfl (fun e _ => by rw [hrow e])
  simp only [attentionRealBase2PerKeyScale, hraw]

/-! ### ════════ ★ MAIN THEOREM (io face) ★ ════════ -/
set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `⊨[R]` io headline — on the `StreamMasked3DKernelIO₅` skin.** On its
five-stream single-store signature (three tile channels `Q`/`K`/`V` plus two
**scalar-width** `B = 1` channels `Q_scale`/`K_scale`), `_attn_fwd` implements
the genuine non-causal closed form: output lane `j = (i, e)` of `Out` holds
`attentionFwdTriton2IOOutSpec` — base-2 per-key-scale attention of the streamed
Q/K/V tiles with `keyScale j = Q_scale · K_scale[j / BLOCK_N]` assembled from the
two scalar channels — for every rounding model that is trivial on the fp16 grid.

**Hypothesis provenance**:
* `hfp16` pins `R.round .fp16 = id` — the in-loop narrowing casts
  (`p = p.to(tl.float16)`, the `v.to(tl.float16)` dot input, `out_dtype=
  tl.float16`) are *in-loop* rounding events outside the skin's
  single-boundary-round shape; this is the file's declared fp16 modeling
  boundary (the exec stack already treats these casts as the identity), now
  explicit as a headline hypothesis.
* `hBN`/`hnum` positivity shape the KV walk (`N_CTX = BLOCK_N · numKVBlocks > 0`)
  — inherited verbatim from the exact headline
  `attention_fwd_triton2_output_summary_general`.
* `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL` — the head mask selects a prefix of
  the head axis (the Python `< 96` inside `tl.arange(0, 128)`); it makes the
  streamed lane encodings land in range.
* `hBDHD : BLOCK_DMODEL ≤ HEAD_DIM` — host launches use
  `HEAD_DIM = BLOCK_DMODEL`; together with `hActiveLe` it supplies the exact
  headline's `HEAD_ACTIVE ≤ HEAD_DIM` contract.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself. The output grid is the `.real` default:
the surface's store-side `.to(Out.type.element_ty)` erases at translation, so
the lowered terminal statement is a plain `Stmt.store .real` and the host's
`torch.bfloat16` allocation is invisible to the model (no `hbf16` hypothesis is
needed, or provable). At every such `R` the terminal cells carry the exact fold
values. -/
specification attention_fwd_triton2_io_correctness (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hBDHD : BLOCK_DMODEL ≤ HEAD_DIM) :
    attentionFwdTriton2IO Q K V Q_scale K_scale Out stride_qz stride_qh Z H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨[R]
      fun _ _ _ xs ys zs ws vs j =>
        attentionFwdTriton2IOOutSpec BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
          xs ys zs ws vs (Lane2D.decode j) := by
  have hHD : HEAD_ACTIVE ≤ HEAD_DIM := le_trans hActiveLe hBDHD
  refine StreamMasked3DKernelIO₅.ImplementsR.intro _ ?_ ?_ ?_
  · exact aft2IO_flattenOk Q K V Q_scale K_scale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N
      numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
  · -- the trace-safety walk
    intro bounds st xs ys zs ws vs _hx _hy _hz _hw _hv hbr1 hbr2 hbr3 hbr4 hbr5 hbw
    simp only [attentionFwdTriton2IO] at hbr1 hbr2 hbr3 hbr4 hbr5 hbw ⊢
    exact aft2IO_traceSafeR R hfp16 bounds Q K V Q_scale K_scale Out stride_qz stride_qh Z H
      HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks st hBN hnum
      (fun t j hm => hbr1 t j hm) (fun t j hm => hbr2 t j hm) (fun t j hm => hbr3 t j hm)
      (fun t j => hbr4 t j trivial) (fun t j => hbr5 t j trivial) (fun j hm => hbw j hm)
  · -- the rounded Hoare triple: exec stack + cast-free collapse
    intro s₀ xs ys zs ws vs hu hx hy hz hw hv
    simp only [attentionFwdTriton2IO] at hx hy hz hw hv ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    -- the walk: prologue, KV loop, postLoop (the weak invariant suffices for the run)
    obtain ⟨sp, hpre, hinv0⟩ := aft2IO_preLoop_evalW s₀ Q K V Q_scale K_scale Out
      stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
    obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
      forRange_inv (idx := "start_n") (start := 0) (stop := BLOCK_N * numKVBlocks)
        (step := BLOCK_N)
        (body := attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (P := aft2SafeInv K K_scale V Out s₀ H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks)
        (s_init := sp) hBN.ne' hinv0
        (fun c stt hc hPc => aft2IO_stepW K K_scale V Out s₀ H stride_qz stride_qh HEAD_DIM
          BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c stt hc hPc)
    obtain ⟨-, -, hoffsmL, -, -, ⟨lT, hliL⟩, ⟨aT, haccL⟩, -, -, -, -, -, hOpL, hmemL⟩ := hinvL
    obtain ⟨sF, hpost, hframe⟩ := aft2IO_postLoopW Out s₀ H stride_qz stride_qh HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks sL aT lT hoffsmL hOpL haccL hliL
    have hexec : exec (attention_fwd_triton2_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE)
        s₀ = some sF := by
      unfold exec
      rw [aft2_body_split, stepStmts.append_some hpre, stepStmts.cons_some hloop]
      exact hpost
    have hexec2 : exec (attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE)
        s₀ = some sF := hexec
    -- the stream-pin tile bridges
    have hkTeq := aft2IOkT_eq s₀ K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe ys (fun t j hm => hy t j hm)
    have hvTeq := aft2IOvT_eq s₀ V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe zs (fun t j hm => hz t j hm)
    have hksEq := aft2IOkeyScale_eq s₀ Q_scale K_scale BLOCK_M BLOCK_N numKVBlocks hnum hBN
      ws vs (fun t j => hw t j trivial) (fun t j => hv t j trivial)
    refine ⟨sF, ?_, ?_, ?_⟩
    · -- termination under `execR R` (everything cast-free under `hfp16`)
      show execR R _ s₀ = some sF
      unfold execR
      rw [aft2_body_split, stepStmtsR_append,
        aft2IO_preLoop_castFree R Q K V Q_scale K_scale Out stride_qz stride_qh H HEAD_DIM
          BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks s₀,
        hpre, Option.bind_some,
        stepStmtsR_cons_some (show stepStmtR R (Stmt.forRange "start_n" 0
            (BLOCK_N * numKVBlocks) BLOCK_N
            (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) sp
            = some sL from by
          rw [aft2IO_loopStmt_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
            HEAD_DIM numKVBlocks sp]
          exact hloop),
        aft2IO_postLoop_castFree R BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks sL]
      exact hpost
    · -- `Out` readback: the streamed closed form on every write-active lane
      intro j hj
      have hact : mIndex s₀ BLOCK_M (Lane2D.decode j).1 < BLOCK_N * numKVBlocks ∧
          (Lane2D.decode j).2.1.val < HEAD_ACTIVE := by
        simp only [mIndex, Lane2D.decode_row, Lane2D.decode_col]
        exact hj
      have hmain := attention_forward_triton_closed_form_correct Q K V Q_scale K_scale Out s₀
        stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE
        STAGE hBN hActiveLe hHD hundef' (Lane2D.decode j) hact
      rw [hexec2] at hmain
      have hmain2 : sF.readMem Out
          (outOffset s₀ H stride_qz stride_qh HEAD_DIM 1 BLOCK_M (Lane2D.decode j))
          = attentionRealBase2PerKeyScale
            (qTile s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (kTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (vTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (keyScale s₀ Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            ((Lane2D.decode j).1, ⟨(Lane2D.decode j).2.1.val, hact.2⟩, PUnit.unit) := hmain
      have hspec : attentionRealBase2PerKeyScale
            (qTile s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (kTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (vTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (keyScale s₀ Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            ((Lane2D.decode j).1, ⟨(Lane2D.decode j).2.1.val, hact.2⟩, PUnit.unit)
          = attentionRealBase2PerKeyScale
            (aft2IOqT BLOCK_M BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs)
            (aft2IOkT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks ys)
            (aft2IOvT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks zs)
            (aft2IOkeyScale BLOCK_N numKVBlocks ws vs)
            ((Lane2D.decode j).1, ⟨(Lane2D.decode j).2.1.val, hact.2⟩, PUnit.unit) := by
        rw [hkTeq, hvTeq, hksEq]
        exact aft2IO_spec_congrQrow _ _ _ _ _ _
          (fun e => aft2IOqT_row_eq s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL
            HEAD_ACTIVE numKVBlocks (BLOCK_N * numKVBlocks) hnum hActiveLe xs
            (fun t jj hm => hx t jj hm) (Lane2D.decode j).1 hact.1 e)
      rw [BlockState.readMemAs_real,
        show s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
              + (s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM
              + j.val % BLOCK_DMODEL
            = outOffset s₀ H stride_qz stride_qh HEAD_DIM 1 BLOCK_M (Lane2D.decode j) from by
          simp only [outOffset, baseOffset, offZ, offH, mIndex, kIndex, Lane2D.decode_row,
            Lane2D.decode_col, Nat.mul_one],
        hmain2, hspec]
      simp only [attentionFwdTriton2IOOutSpec, Lane2D.decode_col,
        dif_pos (show j.val % BLOCK_DMODEL < HEAD_ACTIVE from hj.2), FloatDType.ofReal,
        R.round_real_apply, FloatDType.real_ofWithBot]
    · -- the frame: cells outside the write-active `Out` window are untouched
      intro r' o' hcond
      refine (hframe r' o' ?_).trans ?_
      · intro hr idx hPidx
        rcases hcond with hne | hguard
        · exact absurd hr hne
        · intro heq
          refine hguard (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) ?_ (heq.trans ?_)
          · simp only [Lane2D.encode_div, Lane2D.encode_mod]
            exact hPidx
          · simp only [aft2Base, Lane2D.encode_div, Lane2D.encode_mod]
      · rw [hmemL]

end IOFace


end VeriTile.Bench.TritonBenchG.AttentionFwdTriton2
