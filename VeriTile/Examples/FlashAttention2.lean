/-
VeriTile.Examples.FlashAttention2

FA-2 forward proof surface.

This file starts Tier 3-B with a separate FA-2-facing module.  The first
surface is the semantically equivalent forward baseline: same 4D view contract
as FA-1 forward, exposed under FA-2 names.  The later FA-2-specific deltas
(work partitioning, delayed rescale, and block-skip control flow) should
refine this baseline rather than changing the user-facing spec.
-/

import VeriTile.Examples.FlashAttention1.Core

namespace VeriTile.Examples

open VeriTile.Triton

/-- FA-2 forward baseline kernel.

At this stage the FA-2 module exposes the same online-softmax block recurrence
as the verified FA-1 4D forward kernel.  This gives Tier 3-B an independent
theorem surface and a stable spec target before the FA-2-specific partitioning
and block-skip rewrites are introduced. -/
def fa2ForwardKernelStrided
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_ob stride_oh stride_om stride_od scale

/-- FA-2 forward baseline correctness over bundled 4D tensor views. -/
theorem fa2_forward_correct_4D_views
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ComputeCorrect.Realizes
      (kernel :=
        fa2ForwardKernelStrided views.qReg views.kReg views.vReg views.outReg
          M D Bk numKVBlocks
          views.layout.qB views.layout.qH views.layout.qS views.layout.qD
          views.layout.kB views.layout.kH views.layout.kS views.layout.kD
          views.layout.vB views.layout.vH views.layout.vS views.layout.vD
          views.layout.oB views.layout.oH views.layout.oS views.layout.oD
          scale)
      (initialState := s)
      (write := fun idx : TileIndex [M, D] =>
        some (views.outReg, (views.outBlockOffset s M) idx))
      (expected := fun idx : TileIndex [M, D] =>
        attentionReal4D Q4D K4D V4D scale
          (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
           ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit)) := by
  simpa [fa2ForwardKernelStrided, FA1Views4D.kernel]
    using fa1_forward_correct_4D_views hBk hNumKVBlocks hSk
      views Q4D K4D V4D scale s hPidB hPidH hQBnd hQ4D hK4D hV4D

end VeriTile.Examples
