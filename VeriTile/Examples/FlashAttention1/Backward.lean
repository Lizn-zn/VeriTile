/-
VeriTile.Examples.FlashAttention1.Backward

FA-1 backward math surface.

This file is intentionally proof-facing and Real-only.  Compute/IEEE behavior
stays on the ComputeCorrect/testing track; the theorem surface here is the
algorithm contract consumed by FA-1 backward kernels.
-/

import VeriTile.Examples.FlashAttention1.Backward.Math
import VeriTile.Triton.Concurrency.Atomic
import VeriTile.Triton.Float

namespace VeriTile.Examples

open VeriTile.Triton
open BigOperators

namespace FA1Backward

/-- Forward LSE bridge kept in this file for the artifact theorem surface. -/
theorem streamingLSE_eq_lseReal {M D Bk : Nat} (hBk : 0 < Bk)
    (hN : 0 < numKVBlocks) (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) (i : Fin M) :
    streamingLSE Q numKVBlocks K scale i = lseReal Q K scale i :=
  streamingLSE_eq_lseReal_impl hBk hN Q K scale i

/-- Reverse-mode spec identity kept in this file for the artifact theorem surface. -/
theorem attentionBackwardReal_eq_reverseMode {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    attentionBackwardReal Q K V dO LSE scale =
      reverseModeAttentionReal Q K V dO LSE scale :=
  attentionBackwardReal_eq_reverseMode_impl Q K V dO LSE scale

/-- Mask-aware backward spec bridge kept in this file for the artifact theorem
surface.  It records that the explicit visibility-mask spec is a conservative
extension of the existing non-masked backward spec. -/
theorem attentionBackwardRealMasked_allVisible {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    attentionBackwardRealMasked (fun _ _ => Bool.true) Q K V dO LSE scale =
      attentionBackwardReal Q K V dO LSE scale :=
  attentionBackwardRealMasked_allVisible_impl Q K V dO LSE scale

/-- Softmax JVP identity kept in this file for the artifact theorem surface. -/
theorem softmax_jvp_identity {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) :
    softmaxJacobianJVP P dPRow j = softmaxJVP P dPRow j :=
  softmax_jvp_identity_impl P dPRow j

/-- Mask-aware multi-block `dQ` bridge kept in this file for the artifact
theorem surface.  It is the math identity needed by masked atomic-dQ backward:
block-local masked contributions sum to the closed-form masked backward `dQ`. -/
theorem dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked
    {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionMasked visible Q K V dO LSE scale block idx) =
      (attentionBackwardRealMasked visible Q K V dO LSE scale).dQ idx :=
  dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked_impl
    visible Q K V dO LSE scale idx

/-- Causal multi-block `dQ` corollary for FA-1 backward. -/
theorem dQBlockContributionCausal_sum_eq_attentionBackwardRealCausal
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionCausal Q K V dO LSE scale block idx) =
      (attentionBackwardRealCausal Q K V dO LSE scale).dQ idx :=
  dQBlockContributionCausal_sum_eq_attentionBackwardRealCausal_impl
    Q K V dO LSE scale idx

/-! ## Kernel surfaces

The current verified forward kernels remain available under their existing
names.  The LSE-emitting partner below is the forward-side contract needed by
backward: it is the same full-tile FA-1 loop with one additional store of
`m_i + log(l_i)` after the output store.
-/

def fa1ForwardKernelWithLSE
    (qReg kReg vReg outReg lseReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel := triton {
  pid    := tl.program_id(0)

  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)

    scores  := tl.dot(q, tl.trans(k)) * $(scale)
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  out    := o_acc / l_i[:, None]
  lse    := m_i + tl.log(l_i)
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
  tl.store($(lseReg) + offs_m, lse)
}

/-- Stripped FA-1 backward kernel: one program computes the full non-causal
Real backward slice without atomics.

This is the no-atomic issue #43 path.  It recomputes `P` from `Q`, `K`, and
stored `LSE`, forms the softmax JVP correction, then writes `dQ`, `dK`, and
`dV`.  Production FA-1 uses a different work partition and atomic `dQ`; that
path should consume the generic atomic/concurrency layer.
-/
def fa1BackwardStrippedKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) : ComputeKernel := triton {
  offs_m := tl.arange(0, $(M))
  offs_n := tl.arange(0, $(S))
  offs_d := tl.arange(0, $(D))

  q_ptrs  := offs_m[:, None] * $(D) + offs_d[None, :]
  k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
  v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
  do_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]

  q   := tl.load($(qReg) + q_ptrs)
  k   := tl.load($(kReg) + k_ptrs)
  v   := tl.load($(vReg) + v_ptrs)
  dO  := tl.load($(dOReg) + do_ptrs)
  lse := tl.load($(lseReg) + offs_m)

  scores := tl.dot(q, tl.trans(k)) * $(scale)
  p      := tl.exp(scores - lse[:, None])
  dV     := tl.dot(tl.trans(p), dO)
  dP     := tl.dot(dO, tl.trans(v))
  corr   := tl.sum(p * dP, axis = 1)
  dS     := p * (dP - corr[:, None])
  dQ     := tl.dot(dS, k) * $(scale)
  dK     := tl.dot(tl.trans(dS), q) * $(scale)

  tl.store($(dQReg) + q_ptrs, dQ)
  tl.store($(dKReg) + k_ptrs, dK)
  tl.store($(dVReg) + v_ptrs, dV)
}

/-- Strided / 4D-aware stripped FA-1 backward kernel.

This is the stride-parameterized analogue of `fa1BackwardStrippedKernel`.
Program IDs follow the forward strided convention: `program_id(0)` is the
query block, `program_id(1)` is the head, and `program_id(2)` is the batch.
The LSE tensor is viewed as `[B, H, S_q]`; all other tensors are viewed as
`[B, H, S_*, D]`. -/
def fa1BackwardStrippedKernelStrided
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- dO strides (axes [B, H, S_q, D]):
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    -- LSE strides (axes [B, H, S_q]):
    (stride_lseb stride_lseh stride_lsem : Nat)
    -- dQ strides (axes [B, H, S_q, D]):
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    -- dK strides (axes [B, H, S_k, D]):
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    -- dV strides (axes [B, H, S_k, D]):
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) : ComputeKernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off   := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off   := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off   := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  do_base_off  := pid_b * $(stride_dob) + pid_h * $(stride_doh)
  lse_base_off := pid_b * $(stride_lseb) + pid_h * $(stride_lseh)
  dQ_base_off  := pid_b * $(stride_dqb) + pid_h * $(stride_dqh)
  dK_base_off  := pid_b * $(stride_dkb) + pid_h * $(stride_dkh)
  dV_base_off  := pid_b * $(stride_dvb) + pid_h * $(stride_dvh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_n := tl.arange(0, $(S))
  offs_d := tl.arange(0, $(D))

  q_ptrs  := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
  v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
  do_ptrs := do_base_off + offs_m[:, None] * $(stride_dom) + offs_d[None, :] * $(stride_dod)
  lse_ptrs := lse_base_off + offs_m * $(stride_lsem)

  q   := tl.load($(qReg) + q_ptrs)
  k   := tl.load($(kReg) + k_ptrs)
  v   := tl.load($(vReg) + v_ptrs)
  dO  := tl.load($(dOReg) + do_ptrs)
  lse := tl.load($(lseReg) + lse_ptrs)

  scores := tl.dot(q, tl.trans(k)) * $(scale)
  p      := tl.exp(scores - lse[:, None])
  dV     := tl.dot(tl.trans(p), dO)
  dP     := tl.dot(dO, tl.trans(v))
  corr   := tl.sum(p * dP, axis = 1)
  dS     := p * (dP - corr[:, None])
  dQ     := tl.dot(dS, k) * $(scale)
  dK     := tl.dot(tl.trans(dS), q) * $(scale)

  dQ_ptrs := dQ_base_off + offs_m[:, None] * $(stride_dqs) + offs_d[None, :] * $(stride_dqd)
  dK_ptrs := dK_base_off + offs_n[:, None] * $(stride_dkn) + offs_d[None, :] * $(stride_dkd)
  dV_ptrs := dV_base_off + offs_n[:, None] * $(stride_dvn) + offs_d[None, :] * $(stride_dvd)

  tl.store($(dQReg) + dQ_ptrs, dQ)
  tl.store($(dKReg) + dK_ptrs, dK)
  tl.store($(dVReg) + dV_ptrs, dV)
}

/-- The stride-aware stripped backward kernel is algorithm-projectable: it uses
only Real/algorithm-layer operators, with no compute-only effects. -/
theorem fa1BackwardStrippedKernelStrided_projectable
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) :
    ∃ ak, (fa1BackwardStrippedKernelStrided
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh stride_dqs stride_dqd
      stride_dkb stride_dkh stride_dkn stride_dkd
      stride_dvb stride_dvh stride_dvn stride_dvd
      scale).toAlgorithm? = Except.ok ak := by
  simp [fa1BackwardStrippedKernelStrided]

theorem fa1BackwardStrippedKernelStrided_toAlgorithm_eq_toAlgKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) :
    (fa1BackwardStrippedKernelStrided
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh stride_dqs stride_dqd
      stride_dkb stride_dkh stride_dkn stride_dkd
      stride_dvb stride_dvh stride_dvn stride_dvd
      scale).toAlgorithm? =
      Except.ok (fa1BackwardStrippedKernelStrided
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
        stride_qb stride_qh stride_qs stride_qd
        stride_kb stride_kh stride_kn stride_kd
        stride_vb stride_vh stride_vn stride_vd
        stride_dob stride_doh stride_dom stride_dod
        stride_lseb stride_lseh stride_lsem
        stride_dqb stride_dqh stride_dqs stride_dqd
        stride_dkb stride_dkh stride_dkn stride_dkd
        stride_dvb stride_dvh stride_dvn stride_dvd
        scale).toAlgKernel := by
  simp [fa1BackwardStrippedKernelStrided, ComputeKernel.toAlgKernel]

/-- Block-partitioned FA-1 backward kernel with atomic `dQ` accumulation.

Each program owns one KV block of size `Bk`.  It stores that block's `dK` and
`dV` with ordinary stores, while contributing the block-local `dQ` term through
`tl.atomic_add`.  The row correction is recomputed against the full KV range
`Bk * numKVBlocks` so that each block contribution uses the same closed-form
softmax JVP correction as `attentionBackwardReal`.

This is intentionally proof-oriented rather than performance-oriented: the
full-KV recomputation keeps the first atomic correctness surface close to the
already-proved stripped backward math. -/
def fa1BackwardAtomicDQKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel := triton {
  block_n := tl.program_id(0)
  offs_m  := tl.arange(0, $(M))
  offs_b  := block_n * $(Bk) + tl.arange(0, $(Bk))
  offs_n  := tl.arange(0, $(Bk * numKVBlocks))
  offs_d  := tl.arange(0, $(D))

  q_ptrs       := offs_m[:, None] * $(D) + offs_d[None, :]
  do_ptrs      := offs_m[:, None] * $(D) + offs_d[None, :]
  k_block_ptrs := offs_b[:, None] * $(D) + offs_d[None, :]
  v_block_ptrs := offs_b[:, None] * $(D) + offs_d[None, :]
  k_all_ptrs   := offs_n[:, None] * $(D) + offs_d[None, :]
  v_all_ptrs   := offs_n[:, None] * $(D) + offs_d[None, :]

  q       := tl.load($(qReg) + q_ptrs)
  dO      := tl.load($(dOReg) + do_ptrs)
  lse     := tl.load($(lseReg) + offs_m)
  k_block := tl.load($(kReg) + k_block_ptrs)
  v_block := tl.load($(vReg) + v_block_ptrs)
  k_all   := tl.load($(kReg) + k_all_ptrs)
  v_all   := tl.load($(vReg) + v_all_ptrs)

  scores_all := tl.dot(q, tl.trans(k_all)) * $(scale)
  p_all      := tl.exp(scores_all - lse[:, None])
  dP_all     := tl.dot(dO, tl.trans(v_all))
  corr       := tl.sum(p_all * dP_all, axis = 1)

  scores_block := tl.dot(q, tl.trans(k_block)) * $(scale)
  p_block      := tl.exp(scores_block - lse[:, None])
  dV_block     := tl.dot(tl.trans(p_block), dO)
  dP_block     := tl.dot(dO, tl.trans(v_block))
  dS_block     := p_block * (dP_block - corr[:, None])
  dQ_part      := tl.dot(dS_block, k_block) * $(scale)
  dK_block     := tl.dot(tl.trans(dS_block), q) * $(scale)

  tl.atomic_add($(dQReg) + q_ptrs, dQ_part)
  tl.store($(dKReg) + k_block_ptrs, dK_block)
  tl.store($(dVReg) + v_block_ptrs, dV_block)
}

/-- Causal variant of the block-partitioned FA-1 backward kernel with atomic
`dQ` accumulation.

This keeps the same proof-oriented partitioning as `fa1BackwardAtomicDQKernel`,
but masks both the full-KV correction path and the block-local contribution
path by `offs_m >= offs_n`.  The launcher-facing theorem
`gridLaunchedAtomic_causal_dQ_correct` is the intended composition target once
the trace-extraction proof for this kernel is added. -/
def fa1BackwardAtomicDQCausalKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel := triton {
  block_n := tl.program_id(0)
  offs_m  := tl.arange(0, $(M))
  offs_b  := block_n * $(Bk) + tl.arange(0, $(Bk))
  offs_n  := tl.arange(0, $(Bk * numKVBlocks))
  offs_d  := tl.arange(0, $(D))

  q_ptrs       := offs_m[:, None] * $(D) + offs_d[None, :]
  do_ptrs      := offs_m[:, None] * $(D) + offs_d[None, :]
  k_block_ptrs := offs_b[:, None] * $(D) + offs_d[None, :]
  v_block_ptrs := offs_b[:, None] * $(D) + offs_d[None, :]
  k_all_ptrs   := offs_n[:, None] * $(D) + offs_d[None, :]
  v_all_ptrs   := offs_n[:, None] * $(D) + offs_d[None, :]

  q       := tl.load($(qReg) + q_ptrs)
  dO      := tl.load($(dOReg) + do_ptrs)
  lse     := tl.load($(lseReg) + offs_m)
  k_block := tl.load($(kReg) + k_block_ptrs)
  v_block := tl.load($(vReg) + v_block_ptrs)
  k_all   := tl.load($(kReg) + k_all_ptrs)
  v_all   := tl.load($(vReg) + v_all_ptrs)

  scores_all_raw := tl.dot(q, tl.trans(k_all)) * $(scale)
  causal_all     := offs_m[:, None] >= offs_n[None, :]
  scores_all     := tl.where(causal_all, scores_all_raw, -inf)
  p_all          := tl.exp(scores_all - lse[:, None])
  dP_all         := tl.dot(dO, tl.trans(v_all))
  corr           := tl.sum(p_all * dP_all, axis = 1)

  scores_block_raw := tl.dot(q, tl.trans(k_block)) * $(scale)
  causal_block     := offs_m[:, None] >= offs_b[None, :]
  scores_block     := tl.where(causal_block, scores_block_raw, -inf)
  p_block          := tl.exp(scores_block - lse[:, None])
  dV_block         := tl.dot(tl.trans(p_block), dO)
  dP_block         := tl.dot(dO, tl.trans(v_block))
  dS_block         := p_block * (dP_block - corr[:, None])
  dQ_part          := tl.dot(dS_block, k_block) * $(scale)
  dK_block         := tl.dot(tl.trans(dS_block), q) * $(scale)

  tl.atomic_add($(dQReg) + q_ptrs, dQ_part)
  tl.store($(dKReg) + k_block_ptrs, dK_block)
  tl.store($(dVReg) + v_block_ptrs, dV_block)
}

/-- State immediately before the `tl.atomic_add(dQ, dQ_part)` in the causal
block-partitioned backward kernel.

The causal prefix has four more assignments than the non-causal prefix:
`scores_all_raw`, `causal_all`, `scores_block_raw`, and `causal_block`, so the
atomic boundary is at statement index 33 rather than 29. -/
noncomputable def fa1BackwardAtomicDQCausalPreAtomicState
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) (s : BlockState) : BlockState :=
  (stepStmts
    ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel.body.take 33) s).getD s

private def FA1BackwardAtomicDQCausalPreAtomicFacts
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState) : Prop :=
  stepStmts
      ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale).toAlgKernel.body.take 33) s =
    some (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s) ∧
  (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .nat [M, D] "q_ptrs" =
    some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]) ∧
  (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .nat [Bk, D] "k_block_ptrs" =
    some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
      Tile .nat [Bk, D]) ∧
  (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .nat [Bk, D] "v_block_ptrs" =
    some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
      Tile .nat [Bk, D]) ∧
  (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .real [M, D] "dQ_part" =
    some (Tile.ofReal (dQBlockContributionCausal Q K V dO LSE scale block)) ∧
  (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .real [Bk, D] "dK_block" =
    some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
      (attentionBackwardRealCausal Q K V dO LSE scale).dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) ∧
  (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .real [Bk, D] "dV_block" =
    some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
      (attentionBackwardRealCausal Q K V dO LSE scale).dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))

/-- State immediately before the `tl.atomic_add(dQ, dQ_part)` in the
block-partitioned backward kernel.

This is the stateful trace boundary needed for the full atomic proof: the
atomic payload is not syntactically available from the initial state, because
`dQ_part` is computed by the preceding register program. -/
noncomputable def fa1BackwardAtomicDQPreAtomicState
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) (s : BlockState) : BlockState :=
  (stepStmts
    ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s).getD s

private def FA1BackwardAtomicDQPreAtomicFacts
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState) : Prop :=
  stepStmts
      ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
    some (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s) ∧
  (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .nat [M, D] "q_ptrs" =
    some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]) ∧
  (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .nat [Bk, D] "k_block_ptrs" =
    some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
      Tile .nat [Bk, D]) ∧
  (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .nat [Bk, D] "v_block_ptrs" =
    some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
      Tile .nat [Bk, D]) ∧
  (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .real [M, D] "dQ_part" =
    some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block)) ∧
  (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .real [Bk, D] "dK_block" =
    some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
      (attentionBackwardReal Q K V dO LSE scale).dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) ∧
  (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    M D Bk numKVBlocks scale s).regs .real [Bk, D] "dV_block" =
    some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
      (attentionBackwardReal Q K V dO LSE scale).dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
private theorem fa1BackwardAtomicDQPreAtomic_facts
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    FA1BackwardAtomicDQPreAtomicFacts qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  have hKBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem kReg ((block.val * Bk + jLocal.val) * D + d.val) =
        K (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hK0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  have hVBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem vReg ((block.val * Bk + jLocal.val) * D + d.val) =
        V (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hV0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  simp [FA1BackwardAtomicDQPreAtomicFacts, fa1BackwardAtomicDQPreAtomicState,
      fa1BackwardAtomicDQKernel, stepStmts, stepStmt, evalOp, Option.bind, hPid,
      hQ0, hK0, hV0, hdO0, hLSE, hKBlock, hVBlock, Offset.rowMajor2D,
      Offset.strided, TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, Tile.bop, Tile.uop,
      Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop, Tile.ofReal,
      probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal, WithBot.sum_someTerm_eq_some,
      exp_sum_mul_scale_eq, dS, rowCorrection, dP, probability, FA1Math.scaledScore]
  constructor
  · funext idx
    cases idx with
    | mk j rest =>
        cases rest with
        | mk d tail =>
            cases tail
            simp [Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar,
              Offset.rowMajor2D, Offset.strided]
            ring
  · constructor
    · funext i
      congr 1
      rw [mul_comm]
      rfl
    · funext i
      congr 1
      rw [mul_comm]
      rfl

set_option maxHeartbeats 3000000 in
set_option linter.unusedSimpArgs false in
private theorem fa1BackwardAtomicDQCausalPreAtomic_facts
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    FA1BackwardAtomicDQCausalPreAtomicFacts qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  have hKBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem kReg ((block.val * Bk + jLocal.val) * D + d.val) =
        K (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hK0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  have hVBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem vReg ((block.val * Bk + jLocal.val) * D + d.val) =
        V (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hV0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  simp [FA1BackwardAtomicDQCausalPreAtomicFacts, fa1BackwardAtomicDQCausalPreAtomicState,
      fa1BackwardAtomicDQCausalKernel, stepStmts, stepStmt, evalOp, Option.bind, hPid,
      hQ0, hK0, hV0, hdO0, hLSE, hKBlock, hVBlock, Offset.rowMajor2D,
      Offset.strided, TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.ge,
      Tile.bop, Tile.cop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim,
      Tile.reduceSumDrop, Tile.ofReal,
      probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq,
      WithBot.sum_someTerm_eq_some, exp_sum_mul_scale_eq, dS, rowCorrection, dP,
      probability, FA1Math.scaledScore]
  constructor
  · funext idx
    cases idx with
    | mk j rest =>
        cases rest with
        | mk d tail =>
            cases tail
            simp [Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar,
              Offset.rowMajor2D, Offset.strided]
            ring
  · constructor
    · funext i
      simpa only [dQBlockContributionCausal, dQBlockContributionMasked] using
        dQ_block_kernel_prop_tile_some_eq_dQBlockContributionCausal
          Q K V dO LSE scale block i
    · constructor
      · funext i
        simpa only [attentionBackwardRealCausal, attentionBackwardRealMasked] using
          dK_block_kernel_prop_tile_some_eq_attentionBackwardRealCausal
            Q K V dO LSE scale block i
      · funext i
        simpa only [attentionBackwardRealCausal, attentionBackwardRealMasked] using
          dV_block_kernel_prop_tile_some_eq_attentionBackwardRealCausal
            Q K V dO LSE scale block i

/- #97: the expensive FA-1 backward prefix execution proof is centralized in
`fa1BackwardAtomicDQPreAtomic_facts`; the public readback theorems below are
now cheap projections of that single proof. -/
theorem fa1BackwardAtomicDQPreAtomic_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
      some (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale s) := by
  exact (fa1BackwardAtomicDQPreAtomic_facts
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE).1

theorem fa1BackwardAtomicDQPreAtomic_qPtrs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]) := by
  exact (fa1BackwardAtomicDQPreAtomic_facts
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE).2.1

theorem fa1BackwardAtomicDQPreAtomic_kBlockPtrs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .nat [Bk, D] "k_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
        Tile .nat [Bk, D]) := by
  exact (fa1BackwardAtomicDQPreAtomic_facts
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE).2.2.1

theorem fa1BackwardAtomicDQPreAtomic_vBlockPtrs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .nat [Bk, D] "v_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
        Tile .nat [Bk, D]) := by
  exact (fa1BackwardAtomicDQPreAtomic_facts
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE).2.2.2.1

theorem fa1BackwardAtomicDQPreAtomic_dQPart
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block)) := by
  exact (fa1BackwardAtomicDQPreAtomic_facts
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE).2.2.2.2.1

theorem fa1BackwardAtomicDQPreAtomic_dKBlock
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .real [Bk, D] "dK_block" =
      some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
        (attentionBackwardReal Q K V dO LSE scale).dK
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  exact (fa1BackwardAtomicDQPreAtomic_facts
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE).2.2.2.2.2.1

theorem fa1BackwardAtomicDQPreAtomic_dVBlock
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .real [Bk, D] "dV_block" =
      some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
        (attentionBackwardReal Q K V dO LSE scale).dV
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  exact (fa1BackwardAtomicDQPreAtomic_facts
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE).2.2.2.2.2.2


/-- State immediately before the final `dQ`/`dK`/`dV` stores of the stripped
backward kernel.  The first 20 statements are the register-computation prefix;
the remaining three statements are the output stores. -/
noncomputable def fa1BackwardStrippedPreStoreState
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) (s : BlockState) : BlockState :=
  (stepStmts
    ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M S D scale).toAlgKernel.body.take 20) s).getD s

/-- State immediately before the final `dQ`/`dK`/`dV` stores of the strided
stripped backward kernel. The first 35 statements compute registers and strided
output pointer tiles; the remaining three statements are the stores. -/
noncomputable def fa1BackwardStrippedStridedPreStoreState
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) (s : BlockState) : BlockState :=
  (stepStmts
    ((fa1BackwardStrippedKernelStrided
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh stride_dqs stride_dqd
      stride_dkb stride_dkh stride_dkn stride_dkd
      stride_dvb stride_dvh stride_dvn stride_dvd
      scale).toAlgKernel.body.take 35) s).getD s

/-- State after the stride-aware stripped backward pointer setup prefix.  This is
the first 19 statements of `fa1BackwardStrippedKernelStrided`, ending at
`lse_ptrs` and before the first memory load. -/
noncomputable def fa1BackwardStrippedStridedPointerState
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh : Nat)
    (stride_dkb stride_dkh : Nat)
    (stride_dvb stride_dvh : Nat)
    (s : BlockState) : BlockState :=
  (((((((((((((((((((s
    ).setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0))
    ).setReg "pid_h" .nat [] (Tile.scalar (s.pids 1))
    ).setReg "pid_b" .nat [] (Tile.scalar (s.pids 2))
    ).setReg "q_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_qb + s.pids 1 * stride_qh))
    ).setReg "k_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_kb + s.pids 1 * stride_kh))
    ).setReg "v_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_vb + s.pids 1 * stride_vh))
    ).setReg "do_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_dob + s.pids 1 * stride_doh))
    ).setReg "lse_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_lseb + s.pids 1 * stride_lseh))
    ).setReg "dQ_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_dqb + s.pids 1 * stride_dqh))
    ).setReg "dK_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_dkb + s.pids 1 * stride_dkh))
    ).setReg "dV_base_off" .nat [] (Tile.scalar (s.pids 2 * stride_dvb + s.pids 1 * stride_dvh))
    ).setReg "offs_m" .nat [M] ⟨fun idx : TileIndex [M] => s.pids 0 * M + idx.1.val⟩
    ).setReg "offs_n" .nat [S] (Tile.vec fun i : Fin S => i.val)
    ).setReg "offs_d" .nat [D] (Tile.vec fun i : Fin D => i.val)
    ).setReg "q_ptrs" .nat [M, D] ⟨fun idx : TileIndex [M, D] =>
      s.pids 2 * stride_qb + s.pids 1 * stride_qh
        + (s.pids 0 * M + idx.1.val) * stride_qs
        + idx.2.1.val * stride_qd⟩
    ).setReg "k_ptrs" .nat [S, D] ⟨fun idx : TileIndex [S, D] =>
      s.pids 2 * stride_kb + s.pids 1 * stride_kh
        + idx.1.val * stride_kn
        + idx.2.1.val * stride_kd⟩
    ).setReg "v_ptrs" .nat [S, D] ⟨fun idx : TileIndex [S, D] =>
      s.pids 2 * stride_vb + s.pids 1 * stride_vh
        + idx.1.val * stride_vn
        + idx.2.1.val * stride_vd⟩
    ).setReg "do_ptrs" .nat [M, D] ⟨fun idx : TileIndex [M, D] =>
      s.pids 2 * stride_dob + s.pids 1 * stride_doh
        + (s.pids 0 * M + idx.1.val) * stride_dom
        + idx.2.1.val * stride_dod⟩
    ).setReg "lse_ptrs" .nat [M] ⟨fun idx : TileIndex [M] =>
      s.pids 2 * stride_lseb + s.pids 1 * stride_lseh
        + (s.pids 0 * M + idx.1.val) * stride_lsem⟩

set_option linter.unusedSimpArgs false in

/-- The first 19 statements of the stride-aware stripped backward kernel are the
pure pointer setup prefix. -/
theorem fa1BackwardStrippedKernelStrided_pointer_prefix
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) (s : BlockState) :
    stepStmts
        ((fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale).toAlgKernel.body.take 19) s =
      some
        (fa1BackwardStrippedStridedPointerState M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh
          stride_dkb stride_dkh
          stride_dvb stride_dvh
          s) := by
  simp [fa1BackwardStrippedStridedPointerState,
    fa1BackwardStrippedKernelStrided, stepStmts, stepStmt, evalOp, Option.bind,
    TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
    NumericDType.add, NumericDType.mul, Tile.bop, Tile.expandDim]

/-- Readback facts for the explicit stride-aware pointer setup state. -/
theorem fa1BackwardStrippedStridedPointerState_facts
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh : Nat)
    (stride_dkb stride_dkh : Nat)
    (stride_dvb stride_dvh : Nat)
    (s : BlockState) :
    let ps := fa1BackwardStrippedStridedPointerState M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh
      stride_dkb stride_dkh
      stride_dvb stride_dvh
      s
    ps.regs .nat [M, D] "q_ptrs" =
        some (⟨fun idx : TileIndex [M, D] =>
          s.pids 2 * stride_qb + s.pids 1 * stride_qh
            + (s.pids 0 * M + idx.1.val) * stride_qs
            + idx.2.1.val * stride_qd⟩ : Tile .nat [M, D]) ∧
    ps.regs .nat [S, D] "k_ptrs" =
        some (⟨fun idx : TileIndex [S, D] =>
          s.pids 2 * stride_kb + s.pids 1 * stride_kh
            + idx.1.val * stride_kn
            + idx.2.1.val * stride_kd⟩ : Tile .nat [S, D]) ∧
    ps.regs .nat [S, D] "v_ptrs" =
        some (⟨fun idx : TileIndex [S, D] =>
          s.pids 2 * stride_vb + s.pids 1 * stride_vh
            + idx.1.val * stride_vn
            + idx.2.1.val * stride_vd⟩ : Tile .nat [S, D]) ∧
    ps.regs .nat [M, D] "do_ptrs" =
        some (⟨fun idx : TileIndex [M, D] =>
          s.pids 2 * stride_dob + s.pids 1 * stride_doh
            + (s.pids 0 * M + idx.1.val) * stride_dom
            + idx.2.1.val * stride_dod⟩ : Tile .nat [M, D]) ∧
    ps.regs .nat [M] "lse_ptrs" =
        some (⟨fun idx : TileIndex [M] =>
          s.pids 2 * stride_lseb + s.pids 1 * stride_lseh
            + (s.pids 0 * M + idx.1.val) * stride_lsem⟩ : Tile .nat [M]) ∧
    (∀ region offset, ps.readMem region offset = s.readMem region offset) := by
  simp [fa1BackwardStrippedStridedPointerState]

set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false in

/-- State after the five strided loads in the stripped backward prefix. -/
noncomputable def fa1BackwardStrippedStridedLoadedState
    (qReg kReg vReg dOReg lseReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh : Nat)
    (stride_dkb stride_dkh : Nat)
    (stride_dvb stride_dvh : Nat)
    (s : BlockState) : BlockState :=
  (((((fa1BackwardStrippedStridedPointerState M S D
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh
    stride_dkb stride_dkh
    stride_dvb stride_dvh
    s
    ).setReg "q" .real [M, D] (⟨fun idx : TileIndex [M, D] =>
      some (s.readMem qReg
        (s.pids 2 * stride_qb + s.pids 1 * stride_qh
          + (s.pids 0 * M + idx.1.val) * stride_qs
          + idx.2.1.val * stride_qd))⟩ : Tile .real [M, D])
    ).setReg "k" .real [S, D] (⟨fun idx : TileIndex [S, D] =>
      some (s.readMem kReg
        (s.pids 2 * stride_kb + s.pids 1 * stride_kh
          + idx.1.val * stride_kn
          + idx.2.1.val * stride_kd))⟩ : Tile .real [S, D])
    ).setReg "v" .real [S, D] (⟨fun idx : TileIndex [S, D] =>
      some (s.readMem vReg
        (s.pids 2 * stride_vb + s.pids 1 * stride_vh
          + idx.1.val * stride_vn
          + idx.2.1.val * stride_vd))⟩ : Tile .real [S, D])
    ).setReg "dO" .real [M, D] (⟨fun idx : TileIndex [M, D] =>
      some (s.readMem dOReg
        (s.pids 2 * stride_dob + s.pids 1 * stride_doh
          + (s.pids 0 * M + idx.1.val) * stride_dom
          + idx.2.1.val * stride_dod))⟩ : Tile .real [M, D])
    ).setReg "lse" .real [M] (⟨fun idx : TileIndex [M] =>
      some (s.readMem lseReg
        (s.pids 2 * stride_lseb + s.pids 1 * stride_lseh
          + (s.pids 0 * M + idx.1.val) * stride_lsem))⟩ : Tile .real [M])

set_option linter.unusedSimpArgs false in

/-- Executing statements 19 through 23 of the stride-aware stripped backward
kernel loads `q`, `k`, `v`, `dO`, and `lse` from the pointer setup state. -/
theorem fa1BackwardStrippedKernelStrided_load_prefix
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) (s : BlockState) :
    stepStmts
        (((fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale).toAlgKernel.body.drop 19).take 5)
        (fa1BackwardStrippedStridedPointerState M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh
          stride_dkb stride_dkh
          stride_dvb stride_dvh
          s) =
      some
        (fa1BackwardStrippedStridedLoadedState qReg kReg vReg dOReg lseReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh
          stride_dkb stride_dkh
          stride_dvb stride_dvh
          s) := by
  have hFacts := fa1BackwardStrippedStridedPointerState_facts M S D
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh
    stride_dkb stride_dkh
    stride_dvb stride_dvh
    s
  rcases hFacts with ⟨hqPtrs, hkPtrs, hvPtrs, hdoPtrs, hlsePtrs, hRead⟩
  simp [fa1BackwardStrippedStridedLoadedState,
    fa1BackwardStrippedKernelStrided, stepStmts, stepStmt, evalOp, Option.bind,
    hqPtrs, hkPtrs, hvPtrs, hdoPtrs, hlsePtrs, hRead]

/-- The first 24 statements of the stride-aware stripped backward kernel are
the pointer setup prefix followed by the five input loads. -/
theorem fa1BackwardStrippedKernelStrided_loaded_prefix
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) (s : BlockState) :
    stepStmts
        ((fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale).toAlgKernel.body.take 24) s =
      some
        (fa1BackwardStrippedStridedLoadedState qReg kReg vReg dOReg lseReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh
          stride_dkb stride_dkh
          stride_dvb stride_dvh
          s) := by
  rw [show
      (fa1BackwardStrippedKernelStrided
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
        stride_qb stride_qh stride_qs stride_qd
        stride_kb stride_kh stride_kn stride_kd
        stride_vb stride_vh stride_vn stride_vd
        stride_dob stride_doh stride_dom stride_dod
        stride_lseb stride_lseh stride_lsem
        stride_dqb stride_dqh stride_dqs stride_dqd
        stride_dkb stride_dkh stride_dkn stride_dkd
        stride_dvb stride_dvh stride_dvn stride_dvd
        scale).toAlgKernel.body.take 24 =
      ((fa1BackwardStrippedKernelStrided
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
        stride_qb stride_qh stride_qs stride_qd
        stride_kb stride_kh stride_kn stride_kd
        stride_vb stride_vh stride_vn stride_vd
        stride_dob stride_doh stride_dom stride_dod
        stride_lseb stride_lseh stride_lsem
        stride_dqb stride_dqh stride_dqs stride_dqd
        stride_dkb stride_dkh stride_dkn stride_dkd
        stride_dvb stride_dvh stride_dvn stride_dvd
        scale).toAlgKernel.body.take 19) ++
      (((fa1BackwardStrippedKernelStrided
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
        stride_qb stride_qh stride_qs stride_qd
        stride_kb stride_kh stride_kn stride_kd
        stride_vb stride_vh stride_vn stride_vd
        stride_dob stride_doh stride_dom stride_dod
        stride_lseb stride_lseh stride_lsem
        stride_dqb stride_dqh stride_dqs stride_dqd
        stride_dkb stride_dkh stride_dkn stride_dkd
        stride_dvb stride_dvh stride_dvn stride_dvd
        scale).toAlgKernel.body.drop 19).take 5) by
    simp [fa1BackwardStrippedKernelStrided]]
  rw [stepStmts.append_some
    (fa1BackwardStrippedKernelStrided_pointer_prefix
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh stride_dqs stride_dqd
      stride_dkb stride_dkh stride_dkn stride_dkd
      stride_dvb stride_dvh stride_dvn stride_dvd
      scale s)]
  exact fa1BackwardStrippedKernelStrided_load_prefix
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh stride_dqs stride_dqd
    stride_dkb stride_dkh stride_dkn stride_dkd
    stride_dvb stride_dvh stride_dvn stride_dvd
    scale s

theorem fa1BackwardStrippedStridedLoadedState_facts
    (qReg kReg vReg dOReg lseReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh : Nat)
    (stride_dkb stride_dkh : Nat)
    (stride_dvb stride_dvh : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * stride_qb + s.pids 1 * stride_qh
            + (s.pids 0 * M + idx.1.val) * stride_qs
            + idx.2.1.val * stride_qd) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S, D] =>
          s.pids 2 * stride_kb + s.pids 1 * stride_kh
            + idx.1.val * stride_kn
            + idx.2.1.val * stride_kd) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S, D] =>
          s.pids 2 * stride_vb + s.pids 1 * stride_vh
            + idx.1.val * stride_vn
            + idx.2.1.val * stride_vd) V)
    (hdO : InputAt s dOReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * stride_dob + s.pids 1 * stride_doh
            + (s.pids 0 * M + idx.1.val) * stride_dom
            + idx.2.1.val * stride_dod) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] =>
          s.pids 2 * stride_lseb + s.pids 1 * stride_lseh
            + (s.pids 0 * M + idx.1.val) * stride_lsem)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    let ls := fa1BackwardStrippedStridedLoadedState qReg kReg vReg dOReg lseReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh
      stride_dkb stride_dkh
      stride_dvb stride_dvh
      s
    ls.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
    ls.regs .real [S, D] "k" = some (Tile.ofReal K) ∧
    ls.regs .real [S, D] "v" = some (Tile.ofReal V) ∧
    ls.regs .real [M, D] "dO" = some (Tile.ofReal dO) ∧
    ls.regs .real [M] "lse" =
      some (Tile.ofReal fun idx : TileIndex [M] => LSE idx.1) := by
  simp [InputAt] at hQ hK hV hdO hLSE
  simp [fa1BackwardStrippedStridedLoadedState, Tile.ofReal]
  constructor
  · funext idx
    simp [hQ idx.1 idx.2.1 idx.2.2]
  constructor
  · funext idx
    simp [hK idx.1 idx.2.1 idx.2.2]
  constructor
  · funext idx
    simp [hV idx.1 idx.2.1 idx.2.2]
  constructor
  · funext idx
    simp [hdO idx.1 idx.2.1 idx.2.2]
  · funext idx
    simp [hLSE idx.1]

/-- Explicit state after the strided stripped backward math suffix and output
pointer construction, assuming the five input tiles have already been loaded. -/
noncomputable def fa1BackwardStrippedStridedMathState
    (qReg kReg vReg dOReg lseReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (s : BlockState) : BlockState :=
  let bw := attentionBackwardReal Q K V dO LSE scale
  (((((((((((fa1BackwardStrippedStridedLoadedState qReg kReg vReg dOReg lseReg M S D
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh
    stride_dkb stride_dkh
    stride_dvb stride_dvh
    s
    ).setReg "scores" .real [M, S] (Tile.ofReal fun idx : TileIndex [M, S] =>
      FA1Math.scaledScore Q K scale idx.1 idx.2.1)
    ).setReg "p" .real [M, S] (Tile.ofReal fun idx : TileIndex [M, S] =>
      probability Q K LSE scale idx.1 idx.2.1)
    ).setReg "dV" .real [S, D] (Tile.ofReal bw.dV)
    ).setReg "dP" .real [M, S] (Tile.ofReal fun idx : TileIndex [M, S] =>
      dP V dO idx.1 idx.2.1)
    ).setReg "corr" .real [M] (Tile.ofReal fun idx : TileIndex [M] =>
      rowCorrection Q K V dO LSE scale idx.1)
    ).setReg "dS" .real [M, S] (Tile.ofReal fun idx : TileIndex [M, S] =>
      dS Q K V dO LSE scale idx.1 idx.2.1)
    ).setReg "dQ" .real [M, D] (Tile.ofReal bw.dQ)
    ).setReg "dK" .real [S, D] (Tile.ofReal bw.dK)
    ).setReg "dQ_ptrs" .nat [M, D] (⟨fun idx : TileIndex [M, D] =>
      s.pids 2 * stride_dqb + s.pids 1 * stride_dqh
        + (s.pids 0 * M + idx.1.val) * stride_dqs
        + idx.2.1.val * stride_dqd⟩ : Tile .nat [M, D])
    ).setReg "dK_ptrs" .nat [S, D] (⟨fun idx : TileIndex [S, D] =>
      s.pids 2 * stride_dkb + s.pids 1 * stride_dkh
        + idx.1.val * stride_dkn
        + idx.2.1.val * stride_dkd⟩ : Tile .nat [S, D])
    ).setReg "dV_ptrs" .nat [S, D] (⟨fun idx : TileIndex [S, D] =>
      s.pids 2 * stride_dvb + s.pids 1 * stride_dvh
        + idx.1.val * stride_dvn
        + idx.2.1.val * stride_dvd⟩ : Tile .nat [S, D])

theorem fa1BackwardStrippedStridedMathState_facts
    (qReg kReg vReg dOReg lseReg : RegionName)
    (M S D : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (s : BlockState) :
    let ms := fa1BackwardStrippedStridedMathState qReg kReg vReg dOReg lseReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh stride_dqs stride_dqd
      stride_dkb stride_dkh stride_dkn stride_dkd
      stride_dvb stride_dvh stride_dvn stride_dvd
      Q K V dO LSE scale s
    ms.regs .nat [M, D] "dQ_ptrs" =
      some (⟨fun idx : TileIndex [M, D] =>
        s.pids 2 * stride_dqb + s.pids 1 * stride_dqh
          + (s.pids 0 * M + idx.1.val) * stride_dqs
          + idx.2.1.val * stride_dqd⟩ : Tile .nat [M, D]) ∧
    ms.regs .nat [S, D] "dK_ptrs" =
      some (⟨fun idx : TileIndex [S, D] =>
        s.pids 2 * stride_dkb + s.pids 1 * stride_dkh
          + idx.1.val * stride_dkn
          + idx.2.1.val * stride_dkd⟩ : Tile .nat [S, D]) ∧
    ms.regs .nat [S, D] "dV_ptrs" =
      some (⟨fun idx : TileIndex [S, D] =>
        s.pids 2 * stride_dvb + s.pids 1 * stride_dvh
          + idx.1.val * stride_dvn
          + idx.2.1.val * stride_dvd⟩ : Tile .nat [S, D]) ∧
    ms.regs .real [M, D] "dQ" =
      some (Tile.ofReal (attentionBackwardReal Q K V dO LSE scale).dQ) ∧
    ms.regs .real [S, D] "dK" =
      some (Tile.ofReal (attentionBackwardReal Q K V dO LSE scale).dK) ∧
    ms.regs .real [S, D] "dV" =
      some (Tile.ofReal (attentionBackwardReal Q K V dO LSE scale).dV) := by
  simp [fa1BackwardStrippedStridedMathState]

/-- The stripped backward kernel is fully algorithm-projectable: it contains no
compute-only effect such as unsupported bit-level operations or async markers. -/
theorem fa1BackwardStrippedKernel_projectable
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) :
    ∃ ak, (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M S D scale).toAlgorithm? = Except.ok ak := by
  simp [fa1BackwardStrippedKernel]

theorem fa1BackwardStrippedKernel_toAlgorithm_eq_toAlgKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) :
    (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M S D scale).toAlgorithm? =
      Except.ok (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M S D scale).toAlgKernel := by
  simp [fa1BackwardStrippedKernel, ComputeKernel.toAlgKernel]

/-- The block-partitioned atomic-`dQ` backward kernel is algorithm-projectable:
`tl.atomic_add` is part of the AlgKernel surface, unlike async/barrier
effect markers. -/
theorem fa1BackwardAtomicDQKernel_toAlgorithm_eq_toAlgKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgorithm? =
      Except.ok (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale).toAlgKernel := by
  simp [fa1BackwardAtomicDQKernel, ComputeKernel.toAlgKernel]

/-- The causal block-partitioned atomic-`dQ` backward kernel is also
algorithm-projectable: the causal masks and `tl.where` remain in the
AlgKernel surface, and the final `tl.atomic_add` is an algorithm-level atomic
effect rather than a compute-only marker. -/
theorem fa1BackwardAtomicDQCausalKernel_toAlgorithm_eq_toAlgKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgorithm? =
      Except.ok
        (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel := by
  simp [fa1BackwardAtomicDQCausalKernel, ComputeKernel.toAlgKernel]

/-- Stateful atomic trace extraction agrees with execution for the full
atomic-`dQ` kernel.  This is the generic bridge needed before proving the
per-program contribution formula consumed by the grid atomic-add theorem. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_exec
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) (s final : BlockState) (trace : Trace)
    (hTrace :
      Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s trace final) :
    exec
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        s =
      some final :=
  Kernel.AtomicTraceStateful_final_eq_exec hTrace

/-- The pre-atomic prefix of the block-partitioned backward kernel emits no
atomic trace events.  It only computes registers; the first trace-producing
statement is the subsequent `tl.atomic_add`. -/
theorem fa1BackwardAtomicDQPreAtomic_traceEvents_empty
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) :
    ∀ st ∈
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29),
      ∀ s0, Stmt.atomicTraceEvents tid s0 st = some [] := by
  intro st hmem s0
  simp [fa1BackwardAtomicDQKernel] at hmem
  rcases hmem with hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem |
    hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem |
    hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem
  all_goals subst st; rfl

/-- The causal pre-atomic prefix also emits no atomic trace events.  Its first
trace-producing statement is the `tl.atomic_add` after the 33-statement
causal register-computation prefix. -/
theorem fa1BackwardAtomicDQCausalPreAtomic_traceEvents_empty
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) :
    ∀ st ∈
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 33),
      ∀ s0, Stmt.atomicTraceEvents tid s0 st = some [] := by
  intro st hmem s0
  simp [fa1BackwardAtomicDQCausalKernel] at hmem
  rcases hmem with hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem |
    hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem |
    hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem |
    hmem
  all_goals subst st; rfl

/-- The post-prefix tail of the block-partitioned backward kernel consists of
the atomic `dQ` contribution followed by ordinary `dK` and `dV` stores. -/
theorem fa1BackwardAtomicDQKernel_drop29
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel.body.drop 29 =
      [
        Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none,
        Stmt.store .real [Bk, D]
          (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
          (Op.ref .real [Bk, D] "dK_block") MaskOpt.none,
        Stmt.store .real [Bk, D]
          (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
          (Op.ref .real [Bk, D] "dV_block") MaskOpt.none
      ] := by
  simp [fa1BackwardAtomicDQKernel]

/-- The causal post-prefix tail has the same trace shape as the non-causal
kernel: one atomic `dQ` accumulation followed by ordinary `dK` and `dV`
stores. -/
theorem fa1BackwardAtomicDQCausalKernel_drop33
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel.body.drop 33 =
      [
        Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none,
        Stmt.store .real [Bk, D]
          (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
          (Op.ref .real [Bk, D] "dK_block") MaskOpt.none,
        Stmt.store .real [Bk, D]
          (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
          (Op.ref .real [Bk, D] "dV_block") MaskOpt.none
      ] := by
  simp [fa1BackwardAtomicDQCausalKernel]

/-- Recompose the full stateful trace from the no-atomic prefix and the
post-prefix tail. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_of_tail
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) (s sPre final : BlockState) (trace : Trace)
    (hPre :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre)
    (hTail :
      Kernel.AtomicTraceStatefulList tid
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
        some (trace, final)) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s trace final := by
  apply Kernel.AtomicTraceStateful_of_dropPrefix (n := 29)
  · exact fa1BackwardAtomicDQPreAtomic_traceEvents_empty
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale tid
  · exact hPre
  · exact hTail

/-- Recompose the full causal stateful trace from the no-atomic prefix and the
post-prefix tail. -/
theorem fa1BackwardAtomicDQCausalKernel_statefulTrace_of_tail
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) (s sPre final : BlockState) (trace : Trace)
    (hPre :
      stepStmts
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 33) s =
        some sPre)
    (hTail :
      Kernel.AtomicTraceStatefulList tid
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 33) sPre =
        some (trace, final)) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s trace final := by
  apply Kernel.AtomicTraceStateful_of_dropPrefix (n := 33)
  · exact fa1BackwardAtomicDQCausalPreAtomic_traceEvents_empty
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale tid
  · exact hPre
  · exact hTail

/-- Specialized trace surface for one block's atomic `dQ` contribution. -/
theorem fa1BackwardAtomicDQ_atomicTraceEvents_blockContribution
    {M D Bk numKVBlocks : Nat}
    (dQReg : RegionName) (tid : ThreadId) (sPre : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block))) :
    Stmt.atomicTraceEvents tid sPre
        (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) =
      some ((TileShape.allIndices [M, D]).filterMap fun i =>
        some (Stmt.atomicTraceEvent tid dQReg
          (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
          (some (dQBlockContribution Q K V dO LSE scale block i)))) := by
  simpa [Tile.ofReal] using
    Stmt.atomicTraceEvents_atomicAdd_region_none_of_reg_refs
      (hnum := NumericDType.real) (tid := tid) (s := sPre) (region := dQReg)
      (ptrName := "q_ptrs") (valueName := "dQ_part")
      (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D])
      (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block))
      hPtrs hVal

/-- Tail trace once the pre-atomic registers are known.  The two ordinary
stores after the atomic statement emit no atomic events, so the tail trace is
exactly the block contribution trace. -/
theorem fa1BackwardAtomicDQKernel_tail_trace_blockContribution
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (sPre final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block)))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
        some final) :
    Kernel.AtomicTraceStatefulList tid
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
      some
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))),
          final) := by
  rw [fa1BackwardAtomicDQKernel_drop29] at hTailStep ⊢
  have hTrace := fa1BackwardAtomicDQ_atomicTraceEvents_blockContribution
    (dQReg := dQReg) (tid := tid) (sPre := sPre)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE)
    (scale := scale) (block := block) hPtrs hVal
  conv at hTailStep => lhs; unfold stepStmts
  cases hAtomicStep :
      stepStmt
        (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) sPre with
  | none =>
      simp [hAtomicStep] at hTailStep
  | some sAfterAtomic =>
      simp [hAtomicStep] at hTailStep
      have hStoresTrace :
          Kernel.AtomicTraceStatefulList tid
              [
                Stmt.store .real [Bk, D]
                  (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
                  (Op.ref .real [Bk, D] "dK_block") MaskOpt.none,
                Stmt.store .real [Bk, D]
                  (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
                  (Op.ref .real [Bk, D] "dV_block") MaskOpt.none
              ] sAfterAtomic =
            some ([], final) := by
        apply Kernel.AtomicTraceStatefulList_empty_of_stepStmts
        · intro st hmem s0
          simp at hmem
          rcases hmem with hmem | hmem <;> subst st <;> rfl
        · exact hTailStep
      unfold Kernel.AtomicTraceStatefulList
      simp [hTrace, hAtomicStep, hStoresTrace]

/-- Specialized causal trace surface for one block's atomic `dQ`
contribution. -/
theorem fa1BackwardAtomicDQCausal_atomicTraceEvents_blockContribution
    {M D Bk numKVBlocks : Nat}
    (dQReg : RegionName) (tid : ThreadId) (sPre : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContributionCausal Q K V dO LSE scale block))) :
    Stmt.atomicTraceEvents tid sPre
        (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) =
      some ((TileShape.allIndices [M, D]).filterMap fun i =>
        some (Stmt.atomicTraceEvent tid dQReg
          (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
          (some (dQBlockContributionCausal Q K V dO LSE scale block i)))) := by
  simpa [Tile.ofReal] using
    Stmt.atomicTraceEvents_atomicAdd_region_none_of_reg_refs
      (hnum := NumericDType.real) (tid := tid) (s := sPre) (region := dQReg)
      (ptrName := "q_ptrs") (valueName := "dQ_part")
      (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D])
      (Tile.ofReal (dQBlockContributionCausal Q K V dO LSE scale block))
      hPtrs hVal

/-- Causal tail trace once the pre-atomic registers are known.  The ordinary
stores after the atomic statement emit no atomic events, so the tail trace is
exactly the causal block contribution trace. -/
theorem fa1BackwardAtomicDQCausalKernel_tail_trace_blockContribution
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (sPre final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContributionCausal Q K V dO LSE scale block)))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 33) sPre =
        some final) :
    Kernel.AtomicTraceStatefulList tid
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 33) sPre =
      some
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContributionCausal Q K V dO LSE scale block i))),
          final) := by
  rw [fa1BackwardAtomicDQCausalKernel_drop33] at hTailStep ⊢
  have hTrace := fa1BackwardAtomicDQCausal_atomicTraceEvents_blockContribution
    (dQReg := dQReg) (tid := tid) (sPre := sPre)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE)
    (scale := scale) (block := block) hPtrs hVal
  conv at hTailStep => lhs; unfold stepStmts
  cases hAtomicStep :
      stepStmt
        (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) sPre with
  | none =>
      simp [hAtomicStep] at hTailStep
  | some sAfterAtomic =>
      simp [hAtomicStep] at hTailStep
      have hStoresTrace :
          Kernel.AtomicTraceStatefulList tid
              [
                Stmt.store .real [Bk, D]
                  (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
                  (Op.ref .real [Bk, D] "dK_block") MaskOpt.none,
                Stmt.store .real [Bk, D]
                  (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
                  (Op.ref .real [Bk, D] "dV_block") MaskOpt.none
              ] sAfterAtomic =
            some ([], final) := by
        apply Kernel.AtomicTraceStatefulList_empty_of_stepStmts
        · intro st hmem s0
          simp at hmem
          rcases hmem with hmem | hmem <;> subst st <;> rfl
        · exact hTailStep
      unfold Kernel.AtomicTraceStatefulList
      simp [hTrace, hAtomicStep, hStoresTrace]

/-- Full-kernel stateful trace for one program once the pre-atomic register
facts and tail execution are available. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_blockContribution
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s sPre final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPre :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block)))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))))
        final := by
  apply fa1BackwardAtomicDQKernel_statefulTrace_of_tail
    (qReg := qReg) (kReg := kReg) (vReg := vReg) (dOReg := dOReg)
    (lseReg := lseReg) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (M := M) (D := D) (Bk := Bk) (numKVBlocks := numKVBlocks)
    (scale := scale) (tid := tid) (sPre := sPre)
  · exact hPre
  · exact fa1BackwardAtomicDQKernel_tail_trace_blockContribution
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      scale tid sPre final Q K V dO LSE block hPtrs hVal hTailStep

/-- Full causal-kernel stateful trace for one program once the pre-atomic
register facts and tail execution are available. -/
theorem fa1BackwardAtomicDQCausalKernel_statefulTrace_blockContribution
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s sPre final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPre :
      stepStmts
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 33) s =
        some sPre)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContributionCausal Q K V dO LSE scale block)))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 33) sPre =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContributionCausal Q K V dO LSE scale block i))))
        final := by
  apply fa1BackwardAtomicDQCausalKernel_statefulTrace_of_tail
    (qReg := qReg) (kReg := kReg) (vReg := vReg) (dOReg := dOReg)
    (lseReg := lseReg) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (M := M) (D := D) (Bk := Bk) (numKVBlocks := numKVBlocks)
    (scale := scale) (tid := tid) (sPre := sPre)
  · exact hPre
  · exact fa1BackwardAtomicDQCausalKernel_tail_trace_blockContribution
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      scale tid sPre final Q K V dO LSE block hPtrs hVal hTailStep

/-- Input-level stateful trace theorem for one block-program, modulo the
ordinary tail execution.  The pre-atomic prefix is discharged from the tensor
input assumptions; the remaining tail execution hypothesis is what later
store/readback composition consumes. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_blockContribution_from_inputs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29)
        (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale s) =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))))
        final := by
  apply fa1BackwardAtomicDQKernel_statefulTrace_blockContribution
    (qReg := qReg) (kReg := kReg) (vReg := vReg) (dOReg := dOReg)
    (lseReg := lseReg) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (scale := scale) (tid := tid) (sPre :=
      fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale s)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (block := block)
  · exact fa1BackwardAtomicDQPreAtomic_step
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  · exact fa1BackwardAtomicDQPreAtomic_qPtrs
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  · exact fa1BackwardAtomicDQPreAtomic_dQPart
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  · exact hTailStep

/-- Input-level causal stateful trace theorem for one block-program, modulo the
ordinary tail execution.  The concrete causal prefix is discharged from the
tensor input assumptions and produces the causal block contribution payload. -/
theorem fa1BackwardAtomicDQCausalKernel_statefulTrace_blockContribution_from_inputs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 33)
        (fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale s) =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContributionCausal Q K V dO LSE scale block i))))
        final := by
  let sPre : BlockState :=
    fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s
  have hfacts :
      FA1BackwardAtomicDQCausalPreAtomicFacts
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s :=
    fa1BackwardAtomicDQCausalPreAtomic_facts
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  apply fa1BackwardAtomicDQCausalKernel_statefulTrace_blockContribution
    (qReg := qReg) (kReg := kReg) (vReg := vReg) (dOReg := dOReg)
    (lseReg := lseReg) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (scale := scale) (tid := tid) (sPre := sPre)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (block := block)
  · exact hfacts.1
  · exact hfacts.2.1
  · exact hfacts.2.2.2.2.1
  · exact hTailStep

/-- Input-level stateful trace theorem for one block-program using the ordinary
`exec = some final` surface instead of an explicit tail-step hypothesis. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_blockContribution_from_inputs_of_exec
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hExec :
      exec
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel s =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))))
        final := by
  let sPre : BlockState :=
    fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s
  have hPre :
      stepStmts
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre := by
    exact fa1BackwardAtomicDQPreAtomic_step
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  have hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29)
        sPre =
        some final := by
    have hExec' := hExec
    rw [show
        exec
            (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel s =
          stepStmts
            ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body) s by
      rfl] at hExec'
    rw [show
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body =
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.take 29) ++
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) by
      exact (List.take_append_drop 29 _).symm] at hExec'
    rw [stepStmts.append_some hPre] at hExec'
    exact hExec'
  exact fa1BackwardAtomicDQKernel_statefulTrace_blockContribution_from_inputs
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    scale tid s final Q K V dO LSE block
    hPid hQ hK hV hdO hLSE hTailStep

/-- Readback for the ordinary `dK`/`dV` stores in the atomic-`dQ` backward
tail.  The leading atomic `dQ` update only changes memory and preserves the
registers consumed by the subsequent ordinary stores. -/
theorem atomicBackward_tailStores_readback_rowMajor2D_base
    (dQReg dKReg dVReg : RegionName)
    (M D Bk base : Nat) (s : BlockState)
    (dQPart : TileIndex [M, D] → ℝ)
    (dKFn dVFn : TileIndex [Bk, D] → ℝ)
    (hPtrsQ : s.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ :
        Tile .nat [M, D]))
    (hPtrsK : s.regs .nat [Bk, D] "k_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) base D⟩ :
        Tile .nat [Bk, D]))
    (hPtrsV : s.regs .nat [Bk, D] "v_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) base D⟩ :
        Tile .nat [Bk, D]))
    (hValQ : s.regs .real [M, D] "dQ_part" = some (Tile.ofReal dQPart))
    (hValK : s.regs .real [Bk, D] "dK_block" = some (Tile.ofReal dKFn))
    (hValV : s.regs .real [Bk, D] "dV_block" = some (Tile.ofReal dVFn))
    (hdKdV : dKReg ≠ dVReg) :
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        ((stepStmt (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [Bk, D]
            (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
            (Op.ref .real [Bk, D] "dK_block") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [Bk, D]
              (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
              (Op.ref .real [Bk, D] "dV_block") MaskOpt.none) s2)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (dKFn idx)) ∧
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        ((stepStmt (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [Bk, D]
            (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
            (Op.ref .real [Bk, D] "dK_block") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [Bk, D]
              (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
              (Op.ref .real [Bk, D] "dV_block") MaskOpt.none) s2)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (dVFn idx)) := by
  have hInj : Function.Injective
      (Offset.rowMajor2D (rows := Bk) (cols := D) base D) :=
    Offset.rowMajor2D_inj (base := base) (rowStride := D) (le_refl D)
  constructor
  · intro idx
    simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
      hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_regs]
    rw [BlockState.scatter_preserves_other_region dVReg
      (Offset.rowMajor2D (rows := Bk) (cols := D) base D) dVFn dKReg hdKdV]
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]
  · intro idx
    simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
      hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_regs]
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

set_option maxHeartbeats 1000000 in
set_option linter.unusedSimpArgs false in
/-- Input-level readback for the ordinary `dK`/`dV` stores of one atomic-`dQ`
backward block program.  The `dQ` result is handled by the atomic trace/merge
theorems; this lemma closes the ordinary tail stores for the same full kernel
execution. -/
theorem fa1BackwardAtomicDQKernel_tailStores_readback_from_inputs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hdKdV : dKReg ≠ dVReg) :
    let bw := attentionBackwardReal Q K V dO LSE scale
    let base := block.val * Bk * D
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        (exec
          (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (bw.dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        (exec
          (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (bw.dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  let sPre : BlockState :=
    fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s
  let dKFn : TileIndex [Bk, D] → ℝ := fun idx =>
    (attentionBackwardReal Q K V dO LSE scale).dK
      (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)
  let dVFn : TileIndex [Bk, D] → ℝ := fun idx =>
    (attentionBackwardReal Q K V dO LSE scale).dV
      (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)
  have hPre :
      stepStmts
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre := by
    exact fa1BackwardAtomicDQPreAtomic_step
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  have hTail := atomicBackward_tailStores_readback_rowMajor2D_base
    (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (M := M) (D := D) (Bk := Bk) (base := block.val * Bk * D)
    (s := sPre)
    (dQPart := dQBlockContribution Q K V dO LSE scale block)
    (dKFn := dKFn) (dVFn := dVFn)
    (hPtrsQ := by
      exact fa1BackwardAtomicDQPreAtomic_qPtrs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hPtrsK := by
      exact fa1BackwardAtomicDQPreAtomic_kBlockPtrs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hPtrsV := by
      exact fa1BackwardAtomicDQPreAtomic_vBlockPtrs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hValQ := by
      exact fa1BackwardAtomicDQPreAtomic_dQPart
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hValK := by
      simpa [dKFn] using
        fa1BackwardAtomicDQPreAtomic_dKBlock
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hValV := by
      simpa [dVFn] using
        fa1BackwardAtomicDQPreAtomic_dVBlock
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    hdKdV
  have hExecTail :
      exec
          (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s =
        ((stepStmt (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) sPre).bind fun s1 =>
          (stepStmt (Stmt.store .real [Bk, D]
            (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
            (Op.ref .real [Bk, D] "dK_block") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [Bk, D]
              (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
              (Op.ref .real [Bk, D] "dV_block") MaskOpt.none) s2) := by
    rw [show
        exec
            (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel s =
          stepStmts
            ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body) s by
      rfl]
    rw [show
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body =
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.take 29) ++
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) by
      exact (List.take_append_drop 29 _).symm]
    rw [stepStmts.append_some hPre]
    rw [fa1BackwardAtomicDQKernel_drop29]
    simp [stepStmts, stepStmt, evalOp]
    cases hAtomic :
        ((sPre.regs TileDType.real [M, D] "dQ_part").bind fun values =>
          (sPre.regs TileDType.nat [M, D] "q_ptrs").bind fun offsets =>
            some
              (List.foldl
                (fun acc i =>
                  acc.writeMem dQReg (offsets.data i)
                    (WithBot.unbotD 0
                      (NumericDType.real.add
                        (some (acc.readMem dQReg (offsets.data i)))
                        (values.data i))))
                sPre (TileShape.allIndices [M, D]))) with
    | none =>
        simp
    | some s1 =>
        cases hkStore :
            ((s1.regs TileDType.real [Bk, D] "dK_block").bind fun values =>
              (s1.regs TileDType.nat [Bk, D] "k_block_ptrs").bind fun offsets =>
                some
                  (List.foldl
                    (fun acc i => acc.writeMem dKReg (offsets.data i)
                      (WithBot.unbotD 0 (values.data i)))
                    s1 (TileShape.allIndices [Bk, D]))) with
        | none =>
            simp [hkStore]
        | some s2 =>
            cases hvStore :
                ((s2.regs TileDType.real [Bk, D] "dV_block").bind fun values =>
                  (s2.regs TileDType.nat [Bk, D] "v_block_ptrs").bind fun offsets =>
                    some
                      (List.foldl
                        (fun acc i => acc.writeMem dVReg (offsets.data i)
                          (WithBot.unbotD 0 (values.data i)))
                        s2 (TileShape.allIndices [Bk, D]))) with
            | none =>
                simp [hkStore, hvStore]
            | some s3 =>
                simp [hkStore, hvStore]
  constructor
  · intro idx
    rw [hExecTail]
    exact hTail.1 idx
  · intro idx
    rw [hExecTail]
    exact hTail.2 idx

set_option maxHeartbeats 1000000 in
set_option linter.unusedSimpArgs false in
/-- Input-level readback for the ordinary `dK`/`dV` stores of one causal
atomic-`dQ` backward block program.  The `dQ` result is handled by the causal
atomic trace/merge theorem; this lemma closes the ordinary tail stores for the
same full causal kernel execution. -/
theorem fa1BackwardAtomicDQCausalKernel_tailStores_readback_from_inputs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hdKdV : dKReg ≠ dVReg) :
    let bw := attentionBackwardRealCausal Q K V dO LSE scale
    let base := block.val * Bk * D
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        (exec
          (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (bw.dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        (exec
          (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (bw.dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  let sPre : BlockState :=
    fa1BackwardAtomicDQCausalPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s
  let dKFn : TileIndex [Bk, D] → ℝ := fun idx =>
    (attentionBackwardRealCausal Q K V dO LSE scale).dK
      (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)
  let dVFn : TileIndex [Bk, D] → ℝ := fun idx =>
    (attentionBackwardRealCausal Q K V dO LSE scale).dV
      (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)
  have hfacts :
      FA1BackwardAtomicDQCausalPreAtomicFacts
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s :=
    fa1BackwardAtomicDQCausalPreAtomic_facts
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  have hTail := atomicBackward_tailStores_readback_rowMajor2D_base
    (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (M := M) (D := D) (Bk := Bk) (base := block.val * Bk * D)
    (s := sPre)
    (dQPart := dQBlockContributionCausal Q K V dO LSE scale block)
    (dKFn := dKFn) (dVFn := dVFn)
    (hPtrsQ := hfacts.2.1)
    (hPtrsK := hfacts.2.2.1)
    (hPtrsV := hfacts.2.2.2.1)
    (hValQ := hfacts.2.2.2.2.1)
    (hValK := by
      simpa [dKFn] using hfacts.2.2.2.2.2.1)
    (hValV := by
      simpa [dVFn] using hfacts.2.2.2.2.2.2)
    hdKdV
  have hPre :
      stepStmts
          ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body.take 33) s =
        some sPre := hfacts.1
  have hExecTail :
      exec
          (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s =
        ((stepStmt (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) sPre).bind fun s1 =>
          (stepStmt (Stmt.store .real [Bk, D]
            (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
            (Op.ref .real [Bk, D] "dK_block") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [Bk, D]
              (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
              (Op.ref .real [Bk, D] "dV_block") MaskOpt.none) s2) := by
    rw [show
        exec
            (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel s =
          stepStmts
            ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body) s by
      rfl]
    rw [show
        (fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body =
          ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.take 33) ++
          ((fa1BackwardAtomicDQCausalKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.drop 33) by
      exact (List.take_append_drop 33 _).symm]
    rw [stepStmts.append_some hPre]
    rw [fa1BackwardAtomicDQCausalKernel_drop33]
    simp [stepStmts, stepStmt, evalOp]
    cases hAtomic :
        ((sPre.regs TileDType.real [M, D] "dQ_part").bind fun values =>
          (sPre.regs TileDType.nat [M, D] "q_ptrs").bind fun offsets =>
            some
              (List.foldl
                (fun acc i =>
                  acc.writeMem dQReg (offsets.data i)
                    (WithBot.unbotD 0
                      (NumericDType.real.add
                        (some (acc.readMem dQReg (offsets.data i)))
                        (values.data i))))
                sPre (TileShape.allIndices [M, D]))) with
    | none =>
        simp
    | some s1 =>
        cases hkStore :
            ((s1.regs TileDType.real [Bk, D] "dK_block").bind fun values =>
              (s1.regs TileDType.nat [Bk, D] "k_block_ptrs").bind fun offsets =>
                some
                  (List.foldl
                    (fun acc i => acc.writeMem dKReg (offsets.data i)
                      (WithBot.unbotD 0 (values.data i)))
                    s1 (TileShape.allIndices [Bk, D]))) with
        | none =>
            simp [hkStore]
        | some s2 =>
            cases hvStore :
                ((s2.regs TileDType.real [Bk, D] "dV_block").bind fun values =>
                  (s2.regs TileDType.nat [Bk, D] "v_block_ptrs").bind fun offsets =>
                    some
                      (List.foldl
                        (fun acc i => acc.writeMem dVReg (offsets.data i)
                          (WithBot.unbotD 0 (values.data i)))
                        s2 (TileShape.allIndices [Bk, D]))) with
            | none =>
                simp [hkStore, hvStore]
            | some s3 =>
                simp [hkStore, hvStore]
  constructor
  · intro idx
    rw [hExecTail]
    exact hTail.1 idx
  · intro idx
    rw [hExecTail]
    exact hTail.2 idx

/-- Launcher-facing grid correctness for the atomic `dQ` output cells.

This is the #88 surface over the same atomic `dQ` composition theorem above:
the caller supplies a `GridLaunchedAtomic` witness rather than raw
`frames`/`contributors`/`atomicTrace` arguments. -/
theorem fa1BackwardAtomicDQKernel_gridLaunched_dQ_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch :
      Kernel.GridLaunchedAtomic
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContribution Q K V dO LSE scale block idx)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some ((attentionBackwardReal Q K V dO LSE scale).dQ idx) := by
  intro idx
  simp [observeTileAt]
  rw [hLaunch.observeAtomicCell
    (region := dQReg) (offset := Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)
    (hNoOrdinaryWrite := hNoOrdinaryDQ idx)]
  rw [hInitialDQ idx, hAtomicContrib idx]
  rw [dQBlockContribution_sum_eq_attentionBackwardReal Q K V dO LSE scale idx]
  simp [attentionBackwardReal_dQ]

/-- Generic launcher-facing correctness for masked multi-block atomic `dQ`.

This theorem separates the concurrency composition from the kernel-specific
trace extraction: once a `GridLaunchedAtomic` witness says the atomic events
contribute the masked block-local `dQ` terms, the final memory cell is the
closed-form masked backward `dQ`. -/
theorem gridLaunchedAtomic_masked_dQ_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg : RegionName)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch : Kernel.GridLaunchedAtomic k g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContributionMasked visible Q K V dO LSE scale block idx)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some ((attentionBackwardRealMasked visible Q K V dO LSE scale).dQ idx) := by
  intro idx
  simp [observeTileAt]
  rw [hLaunch.observeAtomicCell
    (region := dQReg) (offset := Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)
    (hNoOrdinaryWrite := hNoOrdinaryDQ idx)]
  rw [hInitialDQ idx, hAtomicContrib idx]
  rw [dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked visible Q K V dO LSE scale idx]
  simp

/-- Concrete launcher-facing masked multi-block dQ theorem for the FA-1
backward atomic kernel.  This specializes the generic masked
`GridLaunchedAtomic` composition surface to `fa1BackwardAtomicDQKernel`, while
leaving the kernel-specific trace-to-masked-contribution relation explicit in
`hAtomicContrib`. -/
theorem fa1BackwardAtomicDQKernel_gridLaunched_masked_dQ_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch :
      Kernel.GridLaunchedAtomic
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContributionMasked visible Q K V dO LSE scale block idx)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some ((attentionBackwardRealMasked visible Q K V dO LSE scale).dQ idx) := by
  exact gridLaunchedAtomic_masked_dQ_correct
    (k := (fa1BackwardAtomicDQKernel
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel)
    (dQReg := dQReg) (visible := visible) (scale := scale)
    (s := s) (sFinal := sFinal)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
    hLaunch hInitialDQ hNoOrdinaryDQ hAtomicContrib

/-- Generic full launcher-facing correctness for arbitrary-mask atomic FA-1
backward.  This combines masked atomic `dQ` composition with caller-supplied
ordinary `dK`/`dV` per-block final-state readback.

The theorem is kernel-agnostic: concrete masked kernels can plug in by proving
`hDKBlock` / `hDVBlock` for each owning program frame. -/
theorem gridLaunchedAtomic_masked_backward_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg dKReg dVReg : RegionName)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch : Kernel.GridLaunchedAtomic k g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContributionMasked visible Q K V dO LSE scale block idx))
    (owner : Fin numKVBlocks → GridIndex g)
    (hDKWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dKReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hDVWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dVReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hDKBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dKReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealMasked visible Q K V dO LSE scale).dK
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))
    (hDVBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dVReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealMasked visible Q K V dO LSE scale).dV
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) :
    let bw := attentionBackwardRealMasked visible Q K V dO LSE scale
    (∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some (bw.dQ idx)) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  constructor
  · exact gridLaunchedAtomic_masked_dQ_correct
      (k := k) (dQReg := dQReg) (visible := visible) (scale := scale)
      (s := s) (sFinal := sFinal)
      (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
      hLaunch hInitialDQ hNoOrdinaryDQ hAtomicContrib
  · constructor
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dKReg off
        (hDKWrite block idx)
      have hRead :
          sFinal.readMem dKReg off =
            (hLaunch.frames (owner block)).final.readMem dKReg off := by
        unfold BlockState.readMem
        rw [hMem]
      change some (sFinal.readMem dKReg off) = _
      rw [hRead]
      simpa [observeTileAt, off] using hDKBlock block idx
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dVReg off
        (hDVWrite block idx)
      have hRead :
          sFinal.readMem dVReg off =
            (hLaunch.frames (owner block)).final.readMem dVReg off := by
        unfold BlockState.readMem
        rw [hMem]
      change some (sFinal.readMem dVReg off) = _
      rw [hRead]
      simpa [observeTileAt, off] using hDVBlock block idx

/-- Causal specialization of `gridLaunchedAtomic_masked_dQ_correct`. -/
theorem gridLaunchedAtomic_causal_dQ_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg : RegionName)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch : Kernel.GridLaunchedAtomic k g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContributionCausal Q K V dO LSE scale block idx)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dQ idx) := by
  intro idx
  have hMasked := gridLaunchedAtomic_masked_dQ_correct
    (k := k) (dQReg := dQReg)
    (visible := fun i j => decide (j.val ≤ i.val))
    (scale := scale) (s := s) (sFinal := sFinal)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
    hLaunch hInitialDQ hNoOrdinaryDQ
    (by
      intro idx
      rw [hAtomicContrib idx]
      simp [dQBlockContributionCausal])
  simpa [attentionBackwardRealCausal] using hMasked idx

/-- Concrete launcher-facing causal dQ theorem for the FA-1 backward atomic
kernel.  This specializes the generic causal `GridLaunchedAtomic` composition
surface to `fa1BackwardAtomicDQCausalKernel`, so downstream users can cite the
kernel-specific theorem instead of manually instantiating `k`. -/
theorem fa1BackwardAtomicDQCausalKernel_gridLaunched_dQ_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch :
      Kernel.GridLaunchedAtomic
        (fa1BackwardAtomicDQCausalKernel
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContributionCausal Q K V dO LSE scale block idx)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dQ idx) := by
  exact gridLaunchedAtomic_causal_dQ_correct
    (k := (fa1BackwardAtomicDQCausalKernel
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel)
    (dQReg := dQReg) (scale := scale) (s := s) (sFinal := sFinal)
      (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
      hLaunch hInitialDQ hNoOrdinaryDQ hAtomicContrib

/-- Full launcher-facing correctness for the non-causal atomic FA-1 backward
kernel.  This combines the grid-composed atomic `dQ` theorem with the ordinary
per-block `dK`/`dV` tail-store readback, so the public surface covers all three
backward outputs from the same `GridLaunchedAtomic` final state. -/
theorem fa1BackwardAtomicDQKernel_gridLaunched_backward_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch :
      Kernel.GridLaunchedAtomic
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContribution Q K V dO LSE scale block idx))
    (owner : Fin numKVBlocks → GridIndex g)
    (hOwnerPid : ∀ block, (s.withGridIndex (owner block)).pids 0 = block.val)
    (hQ : ∀ block, InputAt (s.withGridIndex (owner block)) qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : ∀ block, InputAt (s.withGridIndex (owner block)) kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : ∀ block, InputAt (s.withGridIndex (owner block)) vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : ∀ block, InputAt (s.withGridIndex (owner block)) dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : ∀ block, InputAt (shape := [M]) (s.withGridIndex (owner block)) lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hDKWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dKReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hDVWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dVReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hdKdV : dKReg ≠ dVReg) :
    let bw := attentionBackwardReal Q K V dO LSE scale
    (∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some (bw.dQ idx)) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  constructor
  · exact fa1BackwardAtomicDQKernel_gridLaunched_dQ_correct
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      scale s sFinal Q K V dO LSE g hLaunch hInitialDQ hNoOrdinaryDQ
      hAtomicContrib
  · constructor
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dKReg off
        (hDKWrite block idx)
      have hRead :
          sFinal.readMem dKReg off =
            (hLaunch.frames (owner block)).final.readMem dKReg off := by
        unfold BlockState.readMem
        rw [hMem]
      have hTail := (fa1BackwardAtomicDQKernel_tailStores_readback_from_inputs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        scale (s.withGridIndex (owner block)) Q K V dO LSE block
        (hOwnerPid block) (hQ block) (hK block) (hV block) (hdO block)
        (hLSE block) hdKdV).1 idx
      rw [(hLaunch.frames (owner block)).h_exec] at hTail
      change some (sFinal.readMem dKReg off) = _ 
      rw [hRead]
      simpa [observeTileAt, off] using hTail
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dVReg off
        (hDVWrite block idx)
      have hRead :
          sFinal.readMem dVReg off =
            (hLaunch.frames (owner block)).final.readMem dVReg off := by
        unfold BlockState.readMem
        rw [hMem]
      have hTail := (fa1BackwardAtomicDQKernel_tailStores_readback_from_inputs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        scale (s.withGridIndex (owner block)) Q K V dO LSE block
        (hOwnerPid block) (hQ block) (hK block) (hV block) (hdO block)
        (hLSE block) hdKdV).2 idx
      rw [(hLaunch.frames (owner block)).h_exec] at hTail
      change some (sFinal.readMem dVReg off) = _
      rw [hRead]
      simpa [observeTileAt, off] using hTail

/-- Full launcher-facing correctness for the causal atomic FA-1 backward
kernel.  This combines causal grid-composed atomic `dQ` with the ordinary
per-block causal `dK`/`dV` tail-store readback from the same grid final state. -/
theorem fa1BackwardAtomicDQCausalKernel_gridLaunched_backward_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch :
      Kernel.GridLaunchedAtomic
        (fa1BackwardAtomicDQCausalKernel
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContributionCausal Q K V dO LSE scale block idx))
    (owner : Fin numKVBlocks → GridIndex g)
    (hOwnerPid : ∀ block, (s.withGridIndex (owner block)).pids 0 = block.val)
    (hQ : ∀ block, InputAt (s.withGridIndex (owner block)) qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : ∀ block, InputAt (s.withGridIndex (owner block)) kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : ∀ block, InputAt (s.withGridIndex (owner block)) vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : ∀ block, InputAt (s.withGridIndex (owner block)) dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : ∀ block, InputAt (shape := [M]) (s.withGridIndex (owner block)) lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hDKWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dKReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hDVWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dVReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hdKdV : dKReg ≠ dVReg) :
    let bw := attentionBackwardRealCausal Q K V dO LSE scale
    (∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some (bw.dQ idx)) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  constructor
  · exact fa1BackwardAtomicDQCausalKernel_gridLaunched_dQ_correct
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      scale s sFinal Q K V dO LSE g hLaunch hInitialDQ hNoOrdinaryDQ
      hAtomicContrib
  · constructor
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dKReg off
        (hDKWrite block idx)
      have hRead :
          sFinal.readMem dKReg off =
            (hLaunch.frames (owner block)).final.readMem dKReg off := by
        unfold BlockState.readMem
        rw [hMem]
      have hTail := (fa1BackwardAtomicDQCausalKernel_tailStores_readback_from_inputs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        scale (s.withGridIndex (owner block)) Q K V dO LSE block
        (hOwnerPid block) (hQ block) (hK block) (hV block) (hdO block)
        (hLSE block) hdKdV).1 idx
      rw [(hLaunch.frames (owner block)).h_exec] at hTail
      change some (sFinal.readMem dKReg off) = _
      rw [hRead]
      simpa [observeTileAt, off] using hTail
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dVReg off
        (hDVWrite block idx)
      have hRead :
          sFinal.readMem dVReg off =
            (hLaunch.frames (owner block)).final.readMem dVReg off := by
        unfold BlockState.readMem
        rw [hMem]
      have hTail := (fa1BackwardAtomicDQCausalKernel_tailStores_readback_from_inputs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        scale (s.withGridIndex (owner block)) Q K V dO LSE block
        (hOwnerPid block) (hQ block) (hK block) (hV block) (hdO block)
        (hLSE block) hdKdV).2 idx
      rw [(hLaunch.frames (owner block)).h_exec] at hTail
      change some (sFinal.readMem dVReg off) = _
      rw [hRead]
      simpa [observeTileAt, off] using hTail

/-- Readback for the final three stores of the stripped backward kernel.

The computational prefix establishes the pointer/value registers; this lemma
then handles the memory-only tail and the fact that later stores to `dK`/`dV`
do not clobber `dQ`, and the later store to `dV` does not clobber `dK`. -/
theorem strippedBackward_finalStores_readback
    (dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (s : BlockState)
    (dQFn : TileIndex [M, D] → ℝ)
    (dKFn dVFn : TileIndex [S, D] → ℝ)
    (hPtrsQ : s.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ :
        Tile .nat [M, D]))
    (hPtrsK : s.regs .nat [S, D] "k_ptrs" =
      some (⟨Offset.rowMajor2D (rows := S) (cols := D) 0 D⟩ :
        Tile .nat [S, D]))
    (hPtrsV : s.regs .nat [S, D] "v_ptrs" =
      some (⟨Offset.rowMajor2D (rows := S) (cols := D) 0 D⟩ :
        Tile .nat [S, D]))
    (hValQ : s.regs .real [M, D] "dQ" = some (Tile.ofReal dQFn))
    (hValK : s.regs .real [S, D] "dK" = some (Tile.ofReal dKFn))
    (hValV : s.regs .real [S, D] "dV" = some (Tile.ofReal dVFn))
    (hdQdK : dQReg ≠ dKReg) (hdQdV : dQReg ≠ dVReg) (hdKdV : dKReg ≠ dVReg) :
    (∀ idx : TileIndex [M, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some (dQFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dKReg (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx =
      some (dKFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dVReg (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx =
      some (dVFn idx)) := by
  have hInjQ : Function.Injective
      (Offset.rowMajor2D (rows := M) (cols := D) 0 D) :=
    Offset.rowMajor2D_inj (base := 0) (rowStride := D) (le_refl D)
  have hInjK : Function.Injective
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) :=
    Offset.rowMajor2D_inj (base := 0) (rowStride := D) (le_refl D)
  have hInjV : Function.Injective
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) :=
    Offset.rowMajor2D_inj (base := 0) (rowStride := D) (le_refl D)
  constructor
  · intro idx
    simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
      hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
    rw [BlockState.scatter_preserves_other_region dVReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) dVFn dQReg hdQdV]
    rw [BlockState.scatter_preserves_other_region dKReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) dKFn dQReg hdQdK]
    rw [BlockState.scatter_readback_nd _ _ _ hInjQ idx]
  · constructor
    · intro idx
      simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
        hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
      rw [BlockState.scatter_preserves_other_region dVReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) dVFn dKReg hdKdV]
      rw [BlockState.scatter_readback_nd _ _ _ hInjK idx]
    · intro idx
      simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
        hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
      rw [BlockState.scatter_readback_nd _ _ _ hInjV idx]

/-- Offset-parametric readback for the final three stores of a stripped
backward kernel.

This is the strided counterpart of `strippedBackward_finalStores_readback`: the
computational prefix supplies arbitrary pointer tiles for `dQ`, `dK`, and `dV`,
and the caller supplies injectivity for each output address map. -/
theorem strippedBackward_finalStores_readback_of_offsets
    (dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (s : BlockState)
    (qOff : TileIndex [M, D] → Nat)
    (kOff vOff : TileIndex [S, D] → Nat)
    (dQFn : TileIndex [M, D] → ℝ)
    (dKFn dVFn : TileIndex [S, D] → ℝ)
    (hPtrsQ : s.regs .nat [M, D] "dQ_ptrs" =
      some (⟨qOff⟩ : Tile .nat [M, D]))
    (hPtrsK : s.regs .nat [S, D] "dK_ptrs" =
      some (⟨kOff⟩ : Tile .nat [S, D]))
    (hPtrsV : s.regs .nat [S, D] "dV_ptrs" =
      some (⟨vOff⟩ : Tile .nat [S, D]))
    (hValQ : s.regs .real [M, D] "dQ" = some (Tile.ofReal dQFn))
    (hValK : s.regs .real [S, D] "dK" = some (Tile.ofReal dKFn))
    (hValV : s.regs .real [S, D] "dV" = some (Tile.ofReal dVFn))
    (hInjQ : Function.Injective qOff)
    (hInjK : Function.Injective kOff)
    (hInjV : Function.Injective vOff)
    (hdQdK : dQReg ≠ dKReg) (hdQdV : dQReg ≠ dVReg) (hdKdV : dKReg ≠ dVReg) :
    (∀ idx : TileIndex [M, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "dQ_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "dK_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "dV_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dQReg qOff idx =
      some (dQFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "dQ_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "dK_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "dV_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dKReg kOff idx =
      some (dKFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "dQ_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "dK_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "dV_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dVReg vOff idx =
      some (dVFn idx)) := by
  constructor
  · intro idx
    simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
      hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
    rw [BlockState.scatter_preserves_other_region dVReg vOff dVFn dQReg hdQdV]
    rw [BlockState.scatter_preserves_other_region dKReg kOff dKFn dQReg hdQdK]
    rw [BlockState.scatter_readback_nd _ _ _ hInjQ idx]
  · constructor
    · intro idx
      simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
        hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
      rw [BlockState.scatter_preserves_other_region dVReg vOff dVFn dKReg hdKdV]
      rw [BlockState.scatter_readback_nd _ _ _ hInjK idx]
    · intro idx
      simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
        hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
      rw [BlockState.scatter_readback_nd _ _ _ hInjV idx]

/-- Execution shell for the strided stripped backward kernel after its
register-computation prefix.

Once the first 35 statements have produced the pointer tiles and value tiles,
the final three stores write exactly `dQFn`, `dKFn`, and `dVFn` through the
provided strided output maps. The remaining proof obligation for full strided
correctness is therefore the prefix fact establishing these register hypotheses
from strided inputs. -/
theorem fa1BackwardStrippedKernelStrided_correct_of_prefix
    {M S D : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) (s sPre : BlockState)
    (qOff : TileIndex [M, D] → Nat)
    (kOff vOff : TileIndex [S, D] → Nat)
    (dQFn : TileIndex [M, D] → ℝ)
    (dKFn dVFn : TileIndex [S, D] → ℝ)
    (hPre :
      stepStmts
          ((fa1BackwardStrippedKernelStrided
            qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            scale).toAlgKernel.body.take 35) s =
        some sPre)
    (hPtrsQ : sPre.regs .nat [M, D] "dQ_ptrs" =
      some (⟨qOff⟩ : Tile .nat [M, D]))
    (hPtrsK : sPre.regs .nat [S, D] "dK_ptrs" =
      some (⟨kOff⟩ : Tile .nat [S, D]))
    (hPtrsV : sPre.regs .nat [S, D] "dV_ptrs" =
      some (⟨vOff⟩ : Tile .nat [S, D]))
    (hValQ : sPre.regs .real [M, D] "dQ" = some (Tile.ofReal dQFn))
    (hValK : sPre.regs .real [S, D] "dK" = some (Tile.ofReal dKFn))
    (hValV : sPre.regs .real [S, D] "dV" = some (Tile.ofReal dVFn))
    (hInjQ : Function.Injective qOff)
    (hInjK : Function.Injective kOff)
    (hInjV : Function.Injective vOff)
    (hdQdK : dQReg ≠ dKReg) (hdQdV : dQReg ≠ dVReg) (hdKdV : dKReg ≠ dVReg) :
    (∀ idx : TileIndex [M, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale) s)
        dQReg qOff idx =
      some (dQFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale) s)
        dKReg kOff idx =
      some (dKFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale) s)
        dVReg vOff idx =
      some (dVFn idx)) := by
  have hTail := strippedBackward_finalStores_readback_of_offsets
    dQReg dKReg dVReg M S D
    (s := sPre) (qOff := qOff) (kOff := kOff) (vOff := vOff)
    (dQFn := dQFn) (dKFn := dKFn) (dVFn := dVFn)
    hPtrsQ hPtrsK hPtrsV hValQ hValK hValV hInjQ hInjK hInjV
    hdQdK hdQdV hdKdV
  have hExecTail :
      exec (fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale) s =
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "dQ_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) sPre).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "dK_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "dV_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2) := by
    rw [show
        exec (fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale) s =
          stepStmts
            ((fa1BackwardStrippedKernelStrided
              qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
              stride_qb stride_qh stride_qs stride_qd
              stride_kb stride_kh stride_kn stride_kd
              stride_vb stride_vh stride_vn stride_vd
              stride_dob stride_doh stride_dom stride_dod
              stride_lseb stride_lseh stride_lsem
              stride_dqb stride_dqh stride_dqs stride_dqd
              stride_dkb stride_dkh stride_dkn stride_dkd
              stride_dvb stride_dvh stride_dvn stride_dvd
              scale).toAlgKernel.body) s by
      rfl]
    rw [show
        (fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale).toAlgKernel.body =
          ((fa1BackwardStrippedKernelStrided
            qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            scale).toAlgKernel.body.take 35) ++
          ((fa1BackwardStrippedKernelStrided
            qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            scale).toAlgKernel.body.drop 35) by
      exact (List.take_append_drop 35 _).symm]
    rw [stepStmts.append_some hPre]
    simp [fa1BackwardStrippedKernelStrided, stepStmts, stepStmt, evalOp]
    cases hqStore :
        ((sPre.regs TileDType.real [M, D] "dQ").bind fun values =>
          (sPre.regs TileDType.nat [M, D] "dQ_ptrs").bind fun offsets =>
            some
              (List.foldl
                (fun acc i => acc.writeMem dQReg (offsets.data i)
                  (WithBot.unbotD 0 (values.data i)))
                sPre (TileShape.allIndices [M, D]))) with
    | none =>
        simp
    | some s1 =>
        cases hkStore :
            ((s1.regs TileDType.real [S, D] "dK").bind fun values =>
              (s1.regs TileDType.nat [S, D] "dK_ptrs").bind fun offsets =>
                some
                  (List.foldl
                    (fun acc i => acc.writeMem dKReg (offsets.data i)
                      (WithBot.unbotD 0 (values.data i)))
                    s1 (TileShape.allIndices [S, D]))) with
        | none =>
            simp [hkStore]
        | some s2 =>
            cases hvStore :
                ((s2.regs TileDType.real [S, D] "dV").bind fun values =>
                  (s2.regs TileDType.nat [S, D] "dV_ptrs").bind fun offsets =>
                    some
                      (List.foldl
                        (fun acc i => acc.writeMem dVReg (offsets.data i)
                          (WithBot.unbotD 0 (values.data i)))
                        s2 (TileShape.allIndices [S, D]))) with
            | none =>
                simp [hkStore, hvStore]
            | some s3 =>
                simp [hkStore, hvStore]
  constructor
  · intro idx
    rw [hExecTail]
    exact hTail.1 idx
  · constructor
    · intro idx
      rw [hExecTail]
      exact hTail.2.1 idx
    · intro idx
      rw [hExecTail]
      exact hTail.2.2 idx

/-- User-facing `Realizes` wrapper for the strided stripped backward execution
shell.

The index type is a three-way output channel:
`inl idx` observes `dQ`, `inr (inl idx)` observes `dK`, and
`inr (inr idx)` observes `dV`.  The theorem deliberately keeps the prefix
register facts as assumptions; the next layer proves those facts from strided
input tensors. -/
theorem fa1BackwardStrippedKernelStrided_realizes_of_prefix
    {M S D : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (scale : ℝ) (s sPre : BlockState)
    (qOff : TileIndex [M, D] → Nat)
    (kOff vOff : TileIndex [S, D] → Nat)
    (dQFn : TileIndex [M, D] → ℝ)
    (dKFn dVFn : TileIndex [S, D] → ℝ)
    (hPre :
      stepStmts
          ((fa1BackwardStrippedKernelStrided
            qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            scale).toAlgKernel.body.take 35) s =
        some sPre)
    (hPtrsQ : sPre.regs .nat [M, D] "dQ_ptrs" =
      some (⟨qOff⟩ : Tile .nat [M, D]))
    (hPtrsK : sPre.regs .nat [S, D] "dK_ptrs" =
      some (⟨kOff⟩ : Tile .nat [S, D]))
    (hPtrsV : sPre.regs .nat [S, D] "dV_ptrs" =
      some (⟨vOff⟩ : Tile .nat [S, D]))
    (hValQ : sPre.regs .real [M, D] "dQ" = some (Tile.ofReal dQFn))
    (hValK : sPre.regs .real [S, D] "dK" = some (Tile.ofReal dKFn))
    (hValV : sPre.regs .real [S, D] "dV" = some (Tile.ofReal dVFn))
    (hInjQ : Function.Injective qOff)
    (hInjK : Function.Injective kOff)
    (hInjV : Function.Injective vOff)
    (hdQdK : dQReg ≠ dKReg) (hdQdV : dQReg ≠ dVReg) (hdKdV : dKReg ≠ dVReg) :
    ComputeCorrect.Realizes
      (kernel := fa1BackwardStrippedKernelStrided
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
        stride_qb stride_qh stride_qs stride_qd
        stride_kb stride_kh stride_kn stride_kd
        stride_vb stride_vh stride_vn stride_vd
        stride_dob stride_doh stride_dom stride_dod
        stride_lseb stride_lseh stride_lsem
        stride_dqb stride_dqh stride_dqs stride_dqd
        stride_dkb stride_dkh stride_dkn stride_dkd
        stride_dvb stride_dvh stride_dvn stride_dvd
        scale)
      (initialState := s)
      (write :=
        fun out : Sum (TileIndex [M, D]) (Sum (TileIndex [S, D]) (TileIndex [S, D])) =>
          match out with
          | .inl idx => some (dQReg, qOff idx)
          | .inr (.inl idx) => some (dKReg, kOff idx)
          | .inr (.inr idx) => some (dVReg, vOff idx))
      (expected :=
        fun out : Sum (TileIndex [M, D]) (Sum (TileIndex [S, D]) (TileIndex [S, D])) =>
          match out with
          | .inl idx => dQFn idx
          | .inr (.inl idx) => dKFn idx
          | .inr (.inr idx) => dVFn idx) := by
  have hObs := fa1BackwardStrippedKernelStrided_correct_of_prefix
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh stride_dqs stride_dqd
    stride_dkb stride_dkh stride_dkn stride_dkd
    stride_dvb stride_dvh stride_dvn stride_dvd
    scale s sPre qOff kOff vOff dQFn dKFn dVFn hPre
    hPtrsQ hPtrsK hPtrsV hValQ hValK hValV hInjQ hInjK hInjV hdQdK hdQdV hdKdV
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0 out
  subst s0
  cases out with
  | inl idx =>
      have h := hObs.1 idx
      rw [hExec] at h
      simpa [observeTileAt] using h
  | inr rest =>
      cases rest with
      | inl idx =>
          have h := hObs.2.1 idx
          rw [hExec] at h
          simpa [observeTileAt] using h
      | inr idx =>
          have h := hObs.2.2 idx
          rw [hExec] at h
          simpa [observeTileAt] using h

theorem fa1BackwardStrippedKernelStrided_realizes_of_math_state
    {M S D : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (s : BlockState)
    (hPre :
      stepStmts
          ((fa1BackwardStrippedKernelStrided
            qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            scale).toAlgKernel.body.take 35) s =
        some
          (fa1BackwardStrippedStridedMathState qReg kReg vReg dOReg lseReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            Q K V dO LSE scale s))
    (hInjQ :
      Function.Injective (fun idx : TileIndex [M, D] =>
        s.pids 2 * stride_dqb + s.pids 1 * stride_dqh
          + (s.pids 0 * M + idx.1.val) * stride_dqs
          + idx.2.1.val * stride_dqd))
    (hInjK :
      Function.Injective (fun idx : TileIndex [S, D] =>
        s.pids 2 * stride_dkb + s.pids 1 * stride_dkh
          + idx.1.val * stride_dkn
          + idx.2.1.val * stride_dkd))
    (hInjV :
      Function.Injective (fun idx : TileIndex [S, D] =>
        s.pids 2 * stride_dvb + s.pids 1 * stride_dvh
          + idx.1.val * stride_dvn
          + idx.2.1.val * stride_dvd))
    (hdQdK : dQReg ≠ dKReg) (hdQdV : dQReg ≠ dVReg) (hdKdV : dKReg ≠ dVReg) :
    ComputeCorrect.Realizes
      (kernel := fa1BackwardStrippedKernelStrided
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
        stride_qb stride_qh stride_qs stride_qd
        stride_kb stride_kh stride_kn stride_kd
        stride_vb stride_vh stride_vn stride_vd
        stride_dob stride_doh stride_dom stride_dod
        stride_lseb stride_lseh stride_lsem
        stride_dqb stride_dqh stride_dqs stride_dqd
        stride_dkb stride_dkh stride_dkn stride_dkd
        stride_dvb stride_dvh stride_dvn stride_dvd
        scale)
      (initialState := s)
      (write :=
        fun out : Sum (TileIndex [M, D]) (Sum (TileIndex [S, D]) (TileIndex [S, D])) =>
          match out with
          | .inl idx =>
              some (dQReg,
                s.pids 2 * stride_dqb + s.pids 1 * stride_dqh
                  + (s.pids 0 * M + idx.1.val) * stride_dqs
                  + idx.2.1.val * stride_dqd)
          | .inr (.inl idx) =>
              some (dKReg,
                s.pids 2 * stride_dkb + s.pids 1 * stride_dkh
                  + idx.1.val * stride_dkn
                  + idx.2.1.val * stride_dkd)
          | .inr (.inr idx) =>
              some (dVReg,
                s.pids 2 * stride_dvb + s.pids 1 * stride_dvh
                  + idx.1.val * stride_dvn
                  + idx.2.1.val * stride_dvd))
      (expected :=
        fun out : Sum (TileIndex [M, D]) (Sum (TileIndex [S, D]) (TileIndex [S, D])) =>
          match out with
          | .inl idx => (attentionBackwardReal Q K V dO LSE scale).dQ idx
          | .inr (.inl idx) => (attentionBackwardReal Q K V dO LSE scale).dK idx
          | .inr (.inr idx) => (attentionBackwardReal Q K V dO LSE scale).dV idx) := by
  have hFacts := fa1BackwardStrippedStridedMathState_facts
    qReg kReg vReg dOReg lseReg M S D
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh stride_dqs stride_dqd
    stride_dkb stride_dkh stride_dkn stride_dkd
    stride_dvb stride_dvh stride_dvn stride_dvd
    Q K V dO LSE scale s
  rcases hFacts with ⟨hPtrsQ, hPtrsK, hPtrsV, hValQ, hValK, hValV⟩
  exact fa1BackwardStrippedKernelStrided_realizes_of_prefix
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh stride_dqs stride_dqd
    stride_dkb stride_dkh stride_dkn stride_dkd
    stride_dvb stride_dvh stride_dvn stride_dvd
    scale s
    (fa1BackwardStrippedStridedMathState qReg kReg vReg dOReg lseReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh stride_dqs stride_dqd
      stride_dkb stride_dkh stride_dkn stride_dkd
      stride_dvb stride_dvh stride_dvn stride_dvd
      Q K V dO LSE scale s)
    (fun idx : TileIndex [M, D] =>
      s.pids 2 * stride_dqb + s.pids 1 * stride_dqh
        + (s.pids 0 * M + idx.1.val) * stride_dqs
        + idx.2.1.val * stride_dqd)
    (fun idx : TileIndex [S, D] =>
      s.pids 2 * stride_dkb + s.pids 1 * stride_dkh
        + idx.1.val * stride_dkn
        + idx.2.1.val * stride_dkd)
    (fun idx : TileIndex [S, D] =>
      s.pids 2 * stride_dvb + s.pids 1 * stride_dvh
        + idx.1.val * stride_dvn
        + idx.2.1.val * stride_dvd)
    (attentionBackwardReal Q K V dO LSE scale).dQ
    (attentionBackwardReal Q K V dO LSE scale).dK
    (attentionBackwardReal Q K V dO LSE scale).dV
    hPre hPtrsQ hPtrsK hPtrsV hValQ hValK hValV
    hInjQ hInjK hInjV hdQdK hdQdV hdKdV

theorem fa1BackwardStrippedKernelStrided_realizes_of_math_suffix
    {M S D : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_dob stride_doh stride_dom stride_dod : Nat)
    (stride_lseb stride_lseh stride_lsem : Nat)
    (stride_dqb stride_dqh stride_dqs stride_dqd : Nat)
    (stride_dkb stride_dkh stride_dkn stride_dkd : Nat)
    (stride_dvb stride_dvh stride_dvn stride_dvd : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (s : BlockState)
    (hSuffix :
      stepStmts
          (((fa1BackwardStrippedKernelStrided
            qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            scale).toAlgKernel.body.drop 24).take 11)
          (fa1BackwardStrippedStridedLoadedState qReg kReg vReg dOReg lseReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh
            stride_dkb stride_dkh
            stride_dvb stride_dvh
            s) =
        some
          (fa1BackwardStrippedStridedMathState qReg kReg vReg dOReg lseReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            Q K V dO LSE scale s))
    (hInjQ :
      Function.Injective (fun idx : TileIndex [M, D] =>
        s.pids 2 * stride_dqb + s.pids 1 * stride_dqh
          + (s.pids 0 * M + idx.1.val) * stride_dqs
          + idx.2.1.val * stride_dqd))
    (hInjK :
      Function.Injective (fun idx : TileIndex [S, D] =>
        s.pids 2 * stride_dkb + s.pids 1 * stride_dkh
          + idx.1.val * stride_dkn
          + idx.2.1.val * stride_dkd))
    (hInjV :
      Function.Injective (fun idx : TileIndex [S, D] =>
        s.pids 2 * stride_dvb + s.pids 1 * stride_dvh
          + idx.1.val * stride_dvn
          + idx.2.1.val * stride_dvd))
    (hdQdK : dQReg ≠ dKReg) (hdQdV : dQReg ≠ dVReg) (hdKdV : dKReg ≠ dVReg) :
    ComputeCorrect.Realizes
      (kernel := fa1BackwardStrippedKernelStrided
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
        stride_qb stride_qh stride_qs stride_qd
        stride_kb stride_kh stride_kn stride_kd
        stride_vb stride_vh stride_vn stride_vd
        stride_dob stride_doh stride_dom stride_dod
        stride_lseb stride_lseh stride_lsem
        stride_dqb stride_dqh stride_dqs stride_dqd
        stride_dkb stride_dkh stride_dkn stride_dkd
        stride_dvb stride_dvh stride_dvn stride_dvd
        scale)
      (initialState := s)
      (write :=
        fun out : Sum (TileIndex [M, D]) (Sum (TileIndex [S, D]) (TileIndex [S, D])) =>
          match out with
          | .inl idx =>
              some (dQReg,
                s.pids 2 * stride_dqb + s.pids 1 * stride_dqh
                  + (s.pids 0 * M + idx.1.val) * stride_dqs
                  + idx.2.1.val * stride_dqd)
          | .inr (.inl idx) =>
              some (dKReg,
                s.pids 2 * stride_dkb + s.pids 1 * stride_dkh
                  + idx.1.val * stride_dkn
                  + idx.2.1.val * stride_dkd)
          | .inr (.inr idx) =>
              some (dVReg,
                s.pids 2 * stride_dvb + s.pids 1 * stride_dvh
                  + idx.1.val * stride_dvn
                  + idx.2.1.val * stride_dvd))
      (expected :=
        fun out : Sum (TileIndex [M, D]) (Sum (TileIndex [S, D]) (TileIndex [S, D])) =>
          match out with
          | .inl idx => (attentionBackwardReal Q K V dO LSE scale).dQ idx
          | .inr (.inl idx) => (attentionBackwardReal Q K V dO LSE scale).dK idx
          | .inr (.inr idx) => (attentionBackwardReal Q K V dO LSE scale).dV idx) := by
  have hLoaded :=
    fa1BackwardStrippedKernelStrided_loaded_prefix
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
      stride_qb stride_qh stride_qs stride_qd
      stride_kb stride_kh stride_kn stride_kd
      stride_vb stride_vh stride_vn stride_vd
      stride_dob stride_doh stride_dom stride_dod
      stride_lseb stride_lseh stride_lsem
      stride_dqb stride_dqh stride_dqs stride_dqd
      stride_dkb stride_dkh stride_dkn stride_dkd
      stride_dvb stride_dvh stride_dvn stride_dvd
      scale s
  have hPre :
      stepStmts
          ((fa1BackwardStrippedKernelStrided
            qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            scale).toAlgKernel.body.take 35) s =
        some
          (fa1BackwardStrippedStridedMathState qReg kReg vReg dOReg lseReg M S D
            stride_qb stride_qh stride_qs stride_qd
            stride_kb stride_kh stride_kn stride_kd
            stride_vb stride_vh stride_vn stride_vd
            stride_dob stride_doh stride_dom stride_dod
            stride_lseb stride_lseh stride_lsem
            stride_dqb stride_dqh stride_dqs stride_dqd
            stride_dkb stride_dkh stride_dkn stride_dkd
            stride_dvb stride_dvh stride_dvn stride_dvd
            Q K V dO LSE scale s) := by
    rw [show
        (fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale).toAlgKernel.body.take 35 =
        ((fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale).toAlgKernel.body.take 24) ++
        (((fa1BackwardStrippedKernelStrided
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg M S D
          stride_qb stride_qh stride_qs stride_qd
          stride_kb stride_kh stride_kn stride_kd
          stride_vb stride_vh stride_vn stride_vd
          stride_dob stride_doh stride_dom stride_dod
          stride_lseb stride_lseh stride_lsem
          stride_dqb stride_dqh stride_dqs stride_dqd
          stride_dkb stride_dkh stride_dkn stride_dkd
          stride_dvb stride_dvh stride_dvn stride_dvd
          scale).toAlgKernel.body.drop 24).take 11) by
      simp [fa1BackwardStrippedKernelStrided]]
    rw [stepStmts.append_some hLoaded]
    exact hSuffix
  exact fa1BackwardStrippedKernelStrided_realizes_of_math_state
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_dob stride_doh stride_dom stride_dod
    stride_lseb stride_lseh stride_lsem
    stride_dqb stride_dqh stride_dqs stride_dqd
    stride_dkb stride_dkh stride_dkn stride_dkd
    stride_dvb stride_dvh stride_dvn stride_dvd
    Q K V dO LSE scale s hPre hInjQ hInjK hInjV hdQdK hdQdV hdKdV

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false in

/-- Sub-2b execution wiring: the stripped FA-1 backward kernel produces final
memory whose `dQ` / `dK` / `dV` slices match `attentionBackwardReal`.

Connects:
- the existing tile-level bridges (`dV_tile_eq_attentionBackwardReal` /
  `dQ_tile_eq_attentionBackwardReal` / `dK_tile_eq_attentionBackwardReal`);
- the store-stage readback helper (`store_stage_readback_rowMajor2D`);
- through the `fa1BackwardStrippedKernel` execution path. -/
theorem fa1BackwardStrippedKernel_correct
    {M S D : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hOutDisjoint :
      dQReg ≠ dKReg ∧ dQReg ≠ dVReg ∧ dKReg ≠ dVReg) :
    let bw := attentionBackwardReal Q K V dO LSE scale
    (∀ idx : TileIndex [M, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
            dQReg dKReg dVReg M S D scale) s)
        dQReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx
      = some (bw.dQ idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
            dQReg dKReg dVReg M S D scale) s)
        dKReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx
      = some (bw.dK idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
            dQReg dKReg dVReg M S D scale) s)
        dVReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx
      = some (bw.dV idx)) := by
  rcases hOutDisjoint with ⟨hdQdK, hdQdV, hdKdV⟩
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin S) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin S) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  let sPre : BlockState :=
    fa1BackwardStrippedPreStoreState qReg kReg vReg dOReg lseReg
      dQReg dKReg dVReg M S D scale s
  have hPre :
      stepStmts
          ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M S D scale).toAlgKernel.body.take 20) s =
        some sPre := by
    simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
      Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, dQ_tile_some_eq_attentionBackwardReal, dK_tile_some_eq_attentionBackwardReal,
      dV_tile_some_eq_attentionBackwardReal, dQ_tile_eq_attentionBackwardReal,
      dK_tile_eq_attentionBackwardReal, dV_tile_eq_attentionBackwardReal,
      dS_tile_eq, rowCorrection_tile_eq, dP_tile_eq, probability_tile_eq,
      WithBot.sum_someTerm_eq_some, dS, rowCorrection, dP, probability, FA1Math.scaledScore]
  have hTail := strippedBackward_finalStores_readback
    dQReg dKReg dVReg M S D
    (s := sPre)
    (dQFn := (attentionBackwardReal Q K V dO LSE scale).dQ)
    (dKFn := (attentionBackwardReal Q K V dO LSE scale).dK)
    (dVFn := (attentionBackwardReal Q K V dO LSE scale).dV)
    (hPtrsQ := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
        dQ_tile_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dV_tile_eq_attentionBackwardReal]
      )
    (hPtrsK := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        Tile.bop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
        dQ_tile_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dV_tile_eq_attentionBackwardReal]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          rfl)
    (hPtrsV := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        Tile.bop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
        dQ_tile_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dV_tile_eq_attentionBackwardReal]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          rfl)
    (hValQ := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        Tile.bop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, dQ_tile_some_eq_attentionBackwardReal, dQ_tile_eq_attentionBackwardReal,
        dS_tile_eq, rowCorrection_tile_eq, dP_tile_eq, probability_tile_eq]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          simp [TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            hQ0, hK0, hV0, hdO0, hLSE,
            Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim,
            Tile.reduceSumDrop, Tile.ofReal,
            dQ_tile_eq_attentionBackwardReal, dS_tile_eq, rowCorrection_tile_eq,
            dP_tile_eq, probability_tile_eq, WithBot.sum_someTerm_eq_some,
            dS, rowCorrection, dP, probability, FA1Math.scaledScore]
          simp [mul_comm, mul_left_comm, mul_assoc]
          rfl)
    (hValK := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, dK_tile_some_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dS_tile_eq, rowCorrection_tile_eq, dP_tile_eq, probability_tile_eq]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          simp [TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            hQ0, hK0, hV0, hdO0, hLSE,
            Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim,
            Tile.reduceSumDrop, Tile.ofReal,
            dK_tile_eq_attentionBackwardReal, dS_tile_eq, rowCorrection_tile_eq,
            dP_tile_eq, probability_tile_eq, WithBot.sum_someTerm_eq_some,
            dS, rowCorrection, dP, probability, FA1Math.scaledScore]
          rw [mul_comm]
          apply congrArg (fun z : ℝ => scale * z)
          apply Finset.sum_congr rfl
          intro x hx
          have hExpI :
              Real.exp
                    ((∑ x_1, Q (x, x_1, PUnit.unit) * K (i, x_1, PUnit.unit))
                      * scale - LSE x) =
                Real.exp
                    (scale *
                        ∑ x_1, Q (x, x_1, PUnit.unit) * K (i, x_1, PUnit.unit)
                      - LSE x) := by
            congr 1
            ring
          rw [hExpI]
          set a : ℝ :=
            Real.exp
              (scale * ∑ x_1, Q (x, x_1, PUnit.unit) * K (i, x_1, PUnit.unit)
                - LSE x)
          set b : ℝ := ∑ x_1, dO (x, x_1, PUnit.unit) * V (i, x_1, PUnit.unit)
          set c : ℝ :=
            ∑ x_1,
              Real.exp (scale *
                    ∑ x_2, Q (x, x_2, PUnit.unit) * K (x_1, x_2, PUnit.unit)
                  - LSE x) *
                ∑ x_2, dO (x, x_2, PUnit.unit) * V (x_1, x_2, PUnit.unit)
          set c0 : ℝ :=
            ∑ x_1,
              Real.exp
                  ((∑ x_2, Q (x, x_2, PUnit.unit) * K (x_1, x_2, PUnit.unit))
                    * scale - LSE x) *
                ∑ x_2, dO (x, x_2, PUnit.unit) * V (x_1, x_2, PUnit.unit)
          have hc0 : c0 = c := by
            simp [c0, c]
            apply Finset.sum_congr rfl
            intro x_1 hx_1
            congr 1
            congr 1
            ring
          rw [← hc0]
          set q : ℝ := Q (x, d, PUnit.unit)
          change a * (b - c0) * q = a * (b - c0) * q
          rfl)
    (hValV := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, dV_tile_some_eq_attentionBackwardReal, dV_tile_eq_attentionBackwardReal,
        probability_tile_eq]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          simp [TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            hQ0, hK0, hV0, hdO0, hLSE,
            Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim,
            Tile.reduceSumDrop, Tile.ofReal,
            dV_tile_eq_attentionBackwardReal, probability_tile_eq,
            WithBot.sum_someTerm_eq_some, probability, FA1Math.scaledScore]
          apply Finset.sum_congr rfl
          intro x hx
          congr 1
          ring_nf)
    hdQdK hdQdV hdKdV
  have hExecTail :
      exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
          dQReg dKReg dVReg M S D scale) s =
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) sPre).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2) := by
    rw [show
        exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
          dQReg dKReg dVReg M S D scale) s =
          stepStmts
            ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
              dQReg dKReg dVReg M S D scale).toAlgKernel.body) s by
      rfl]
    rw [show
        (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
          dQReg dKReg dVReg M S D scale).toAlgKernel.body =
          ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
              dQReg dKReg dVReg M S D scale).toAlgKernel.body.take 20) ++
            ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
              dQReg dKReg dVReg M S D scale).toAlgKernel.body.drop 20) by
      exact (List.take_append_drop 20 _).symm]
    rw [stepStmts.append_some hPre]
    simp [fa1BackwardStrippedKernel, stepStmts, stepStmt, evalOp]
    cases hqStore :
        ((sPre.regs TileDType.real [M, D] "dQ").bind fun values =>
          (sPre.regs TileDType.nat [M, D] "q_ptrs").bind fun offsets =>
            some
              (List.foldl
                (fun acc i => acc.writeMem dQReg (offsets.data i)
                  (WithBot.unbotD 0 (values.data i)))
                sPre (TileShape.allIndices [M, D]))) with
    | none =>
        simp
    | some s1 =>
        cases hkStore :
            ((s1.regs TileDType.real [S, D] "dK").bind fun values =>
              (s1.regs TileDType.nat [S, D] "k_ptrs").bind fun offsets =>
                some
                  (List.foldl
                    (fun acc i => acc.writeMem dKReg (offsets.data i)
                      (WithBot.unbotD 0 (values.data i)))
                    s1 (TileShape.allIndices [S, D]))) with
        | none =>
            simp [hkStore]
        | some s2 =>
            cases hvStore :
                ((s2.regs TileDType.real [S, D] "dV").bind fun values =>
                  (s2.regs TileDType.nat [S, D] "v_ptrs").bind fun offsets =>
                    some
                      (List.foldl
                        (fun acc i => acc.writeMem dVReg (offsets.data i)
                          (WithBot.unbotD 0 (values.data i)))
                        s2 (TileShape.allIndices [S, D]))) with
            | none =>
                simp [hkStore, hvStore]
            | some s3 =>
                simp [hkStore, hvStore]
  constructor
  · intro idx
    cases idx with
    | mk i rest =>
      cases rest with
      | mk d tail =>
        rw [hExecTail]
        exact hTail.1 (i, d, tail)
  · constructor
    · intro idx
      cases idx with
      | mk j rest =>
        cases rest with
        | mk d tail =>
          rw [hExecTail]
          exact hTail.2.1 (j, d, tail)
    · intro idx
      cases idx with
      | mk j rest =>
        cases rest with
        | mk d tail =>
          rw [hExecTail]
          exact hTail.2.2 (j, d, tail)

end FA1Backward

end VeriTile.Examples
