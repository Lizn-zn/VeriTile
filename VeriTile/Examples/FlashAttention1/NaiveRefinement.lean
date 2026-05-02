/-
VeriTile.Examples.FlashAttention1.NaiveRefinement

Step 4 of issue #39: make the naive/reference refinement surface explicit.
-/

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

end VeriTile.Examples
