/-
VeriTile.Examples.FlashAttention1.NaiveRefinement

Step 4 of issue #39: make the naive/reference refinement surface explicit.
-/

import VeriTile.Triton.Float
import VeriTile.Examples.FlashAttention1.NaiveKernel

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## Naive FA reference

`attentionReal4D` is the single-block-output, non-online softmax reference:
`softmax(QKᵀ * scale) · V` for each `(batch, head)` slice. The verified
single-pass naive boundary kernels live in `NaiveKernel.lean`; this file
keeps the reference-level refinement aliases.
-/

noncomputable def fa1NaiveReference4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  attentionReal4D Q K V scale

noncomputable def fa1NaiveCausalReference4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  attentionReal4DCausal Q K V scale

theorem fa1_naive_reference_eq_attentionReal4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) :
    fa1NaiveReference4D Q K V scale = attentionReal4D Q K V scale := rfl

theorem fa1_naive_causal_reference_eq_attentionReal4DCausal {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) :
    fa1NaiveCausalReference4D Q K V scale =
      attentionReal4DCausal Q K V scale := rfl

/-- FA-1 boundary+D-tail refines the naive direct FA reference. -/
theorem fa1_boundaryD_refines_naive_reference_exec_views
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (views.boundaryKernelD M Bd Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx
        = some (fa1NaiveReference4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [fa1NaiveReference4D]
    using fa1_forward_correct_4D_boundaryD_views hBk hSk hSkLe hDLe
      views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx

/-- Compute-facing FA-1 boundary+D-tail correctness against the naive direct
FA reference. -/
theorem fa1_boundaryD_refines_naive_reference_views
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ComputeCorrect.General
      ((views.boundaryKernelD M Bd Bk numKVBlocks scale))
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [M, Bd],
          ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
          ∀ hDIdx : idx.2.1.val < D,
          observeTileAt
              (some s')
              views.outReg (views.outBlockOffsetD s M Bd) idx
            = some (fa1NaiveReference4D Q4D K4D V4D scale
                (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                 ⟨s.pids 0 * M + idx.1.val, hLt⟩,
                 ⟨idx.2.1.val, hDIdx⟩, PUnit.unit))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hLt hDIdx
  have hview := fa1_boundaryD_refines_naive_reference_exec_views hBk hSk hSkLe hDLe
    views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx
  rw [hExec] at hview
  simpa using hview

/-- Causal FA-1 boundary+D-tail refines the naive direct causal FA reference. -/
theorem fa1_causal_boundaryD_refines_naive_reference_exec_views
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (views.causalBoundaryKernelD M Bd Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx
        = some (fa1NaiveCausalReference4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [fa1NaiveCausalReference4D]
    using fa1_forward_correct_4D_causal_boundaryD_views hBk hSk hSkLe hDLe
      views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx

/-- Compute-facing causal FA-1 boundary+D-tail correctness against the naive
direct causal FA reference. -/
theorem fa1_causal_boundaryD_refines_naive_reference_views
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ComputeCorrect.General
      ((views.causalBoundaryKernelD M Bd Bk numKVBlocks scale))
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [M, Bd],
          ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
          ∀ hDIdx : idx.2.1.val < D,
          observeTileAt
              (some s')
              views.outReg (views.outBlockOffsetD s M Bd) idx
            = some (fa1NaiveCausalReference4D Q4D K4D V4D scale
                (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                 ⟨s.pids 0 * M + idx.1.val, hLt⟩,
                 ⟨idx.2.1.val, hDIdx⟩, PUnit.unit))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hLt hDIdx
  have hview := fa1_causal_boundaryD_refines_naive_reference_exec_views hBk hSk hSkLe hDLe
    views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx
  rw [hExec] at hview
  simpa using hview

end VeriTile.Examples
