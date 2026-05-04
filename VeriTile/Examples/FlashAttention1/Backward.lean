/-
VeriTile.Examples.FlashAttention1.Backward

FA-1 backward math surface.

This file is intentionally proof-facing and Real-only.  Compute/IEEE behavior
stays on the ComputeCorrect/testing track; the theorem surface here is the
algorithm contract consumed by FA-1 backward kernels.
-/

import VeriTile.Examples.FlashAttention1.Common

namespace VeriTile.Examples

open VeriTile.Triton
open BigOperators

namespace FA1Backward

/-- Backward outputs for one non-causal FA-1 slice. -/
structure Grads (M S D : Nat) where
  dQ : TileIndex [M, D] → ℝ
  dK : TileIndex [S, D] → ℝ
  dV : TileIndex [S, D] → ℝ

/-- Forward-side log-sum-exp reconstructed from Q/K. -/
noncomputable def lseReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) : ℝ :=
  Real.log (Finset.univ.sum fun j : Fin S =>
    Real.exp (FA1Math.scaledScore Q K scale i j))

/-- The value computed by the online-softmax registers at forward readout:
`m_i + log(l_i)`. -/
noncomputable def streamingLSE {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) (i : Fin M) : ℝ :=
  (FA1Math.mPartial Bk Q numKVBlocks K scale numKVBlocks i).unbotD 0 +
    Real.log (FA1Math.lPartial Q numKVBlocks K scale numKVBlocks i)

/-- The forward LSE store is the usual unshifted log-sum-exp.

This is the math bridge needed by a forward kernel that stores
`m_i + tl.log(l_i)` for backward. -/
theorem streamingLSE_eq_lseReal {M D Bk : Nat} (hBk : 0 < Bk)
    (hN : 0 < numKVBlocks) (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) (i : Fin M) :
    streamingLSE Q numKVBlocks K scale i = lseReal Q K scale i := by
  unfold streamingLSE lseReal
  let m : ℝ := (FA1Math.mPartial Bk Q numKVBlocks K scale numKVBlocks i).unbotD 0
  have hlf_pos : 0 < FA1Math.lFree Q K scale numKVBlocks (le_refl numKVBlocks) i :=
    FA1Math.lFree_final_pos hBk hN Q K scale i
  have hlf_ne :
      FA1Math.lFree Q K scale numKVBlocks (le_refl numKVBlocks) i ≠ 0 :=
    ne_of_gt hlf_pos
  rw [FA1Math.lPartial_eq_mShifted hBk Q numKVBlocks K scale
      numKVBlocks (le_refl _) i]
  rw [Real.log_mul (Real.exp_ne_zero _) hlf_ne]
  rw [Real.log_exp]
  rw [FA1Math.lFree_eq_flat]
  ring

/-- Softmax probabilities reconstructed from a stored LSE row. -/
noncomputable def probability {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  Real.exp (FA1Math.scaledScore Q K scale i j - LSE i)

/-- `dP = dO · Vᵀ`. -/
noncomputable def dP {M S D : Nat}
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (i : Fin M) (j : Fin S) : ℝ :=
  Finset.univ.sum fun d : Fin D =>
    dO (i, d, PUnit.unit) * V (j, d, PUnit.unit)

/-- Row correction `D_i = Σ_j P_ij · dP_ij`. -/
noncomputable def rowCorrection {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) : ℝ :=
  Finset.univ.sum fun j : Fin S =>
    probability Q K LSE scale i j * dP V dO i j

/-- Softmax JVP in the form used by FA backward:
`dS_ij = P_ij · (dP_ij - D_i)`. -/
noncomputable def dS {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  probability Q K LSE scale i j *
    (dP V dO i j - rowCorrection Q K V dO LSE scale i)

/-- Closed-form reverse-mode FA-1 backward over Real tensors. -/
noncomputable def attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    Grads M S D :=
  { dQ := fun (i, d, _) =>
      scale * Finset.univ.sum fun j : Fin S =>
        dS Q K V dO LSE scale i j * K (j, d, PUnit.unit)
    dK := fun (j, d, _) =>
      scale * Finset.univ.sum fun i : Fin M =>
        dS Q K V dO LSE scale i j * Q (i, d, PUnit.unit)
    dV := fun (j, d, _) =>
      Finset.univ.sum fun i : Fin M =>
        probability Q K LSE scale i j * dO (i, d, PUnit.unit) }

/-- Named reverse-mode spec.  Keeping this as a separate name makes the public
theorem surface state the intended equivalence explicitly while leaving the
closed form above as the executable definition. -/
noncomputable def reverseModeAttentionReal := @attentionBackwardReal

theorem attentionBackwardReal_eq_reverseMode {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    attentionBackwardReal Q K V dO LSE scale =
      reverseModeAttentionReal Q K V dO LSE scale := rfl

@[simp] theorem attentionBackwardReal_dQ {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (attentionBackwardReal Q K V dO LSE scale).dQ idx =
      scale * Finset.univ.sum (fun j : Fin S =>
        dS Q K V dO LSE scale idx.1 j * K (j, idx.2.1, PUnit.unit)) := by
  cases idx
  rfl

@[simp] theorem attentionBackwardReal_dK {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    (attentionBackwardReal Q K V dO LSE scale).dK idx =
      scale * Finset.univ.sum (fun i : Fin M =>
        dS Q K V dO LSE scale i idx.1 * Q (i, idx.2.1, PUnit.unit)) := by
  cases idx
  rfl

@[simp] theorem attentionBackwardReal_dV {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    (attentionBackwardReal Q K V dO LSE scale).dV idx =
      Finset.univ.sum (fun i : Fin M =>
        probability Q K LSE scale i idx.1 * dO (i, idx.2.1, PUnit.unit)) := by
  cases idx
  rfl

/-- Tile-level bridge for the stripped kernel's `dV = Pᵀ · dO`
register computation. -/
theorem dV_tile_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    ((Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            probability Q K LSE scale idx.1 idx.2.1))
        (Tile.ofReal dO)).data idx).unbotD 0 =
      (attentionBackwardReal Q K V dO LSE scale).dV idx := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dV]

/-- Tile-level bridge for the stripped kernel's `dQ = dS · K · scale`
register computation. -/
theorem dQ_tile_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    Option.getD
      (Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            dS Q K V dO LSE scale idx.1 idx.2.1)
          (Tile.ofReal K)).data idx)) 0 =
      (attentionBackwardReal Q K V dO LSE scale).dQ idx := by
  rw [Tile.dot_nil_data]
  simp [Tile.ofReal, attentionBackwardReal_dQ]
  ring

/-- Tile-level bridge for the stripped kernel's `dK = dSᵀ · Q · scale`
register computation. -/
theorem dK_tile_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    Option.getD
      (Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.transpose []
            (Tile.ofReal fun idx : TileIndex [M, S] =>
              dS Q K V dO LSE scale idx.1 idx.2.1))
          (Tile.ofReal Q)).data idx)) 0 =
      (attentionBackwardReal Q K V dO LSE scale).dK idx := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dK]
  ring

/-- Tile-level bridge for the stripped kernel's probability tile
`P = exp(scores - LSE[:, None])`. -/
theorem probability_tile_eq {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (idx : TileIndex [M, S]) :
    WithBot.realExp
      (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
        (Option.map (fun a : ℝ => a * scale)
          ((Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data idx))
        ((Tile.ofReal fun idx : TileIndex [M] => LSE idx.1).data (idx.1, PUnit.unit))) =
      some (probability Q K LSE scale idx.1 idx.2.1) := by
  change WithBot.realExp
      (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
        (Option.map (fun a : ℝ => a * scale)
          ((Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data
            (idx.1, idx.2.1, PUnit.unit)))
        ((Tile.ofReal fun idx : TileIndex [M] => LSE idx.1).data (idx.1, PUnit.unit))) = _
  rw [FA1Math.scaled_data_eq' Q K scale idx.1 idx.2.1]
  simp [Tile.ofReal, probability]

/-- Tile-level bridge for `dP = dO · Vᵀ`. -/
theorem dP_tile_eq {M S D : Nat}
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (idx : TileIndex [M, S]) :
    ((Tile.dot [] (Tile.ofReal dO) (Tile.transpose [] (Tile.ofReal V))).data idx).unbotD 0 =
      dP V dO idx.1 idx.2.1 := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, dP]

/-- Tile-level bridge for `corr = sum(P * dP, axis=1)`. -/
theorem rowCorrection_tile_eq {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (idx : TileIndex [M]) :
    (Tile.reduceSumDrop (shape := [M, S]) 1
      (Tile.ofReal fun idx : TileIndex [M, S] =>
        probability Q K LSE scale idx.1 idx.2.1 * dP V dO idx.1 idx.2.1)).data idx =
      some (rowCorrection Q K V dO LSE scale idx.1) := by
  simp [Tile.reduceSumDrop, Tile.ofReal, rowCorrection]
  congr 1

/-- Tile-level bridge for `dS = P * (dP - corr[:, None])`. -/
theorem dS_tile_eq {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (idx : TileIndex [M, S]) :
    Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
      ((Tile.ofReal fun idx : TileIndex [M, S] =>
        probability Q K LSE scale idx.1 idx.2.1).data idx)
      (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
        ((Tile.ofReal fun idx : TileIndex [M, S] => dP V dO idx.1 idx.2.1).data idx)
        ((Tile.ofReal fun idx : TileIndex [M] =>
          rowCorrection Q K V dO LSE scale idx.1).data (idx.1, PUnit.unit))) =
      some (dS Q K V dO LSE scale idx.1 idx.2.1) := by
  simp [Tile.ofReal, dS]

/-- Jacobian-vector product of row softmax after simplifying
`(diag(P) - P·Pᵀ) dP`. -/
noncomputable def softmaxJacobianJVP {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) : ℝ :=
  P j * (dPRow j - Finset.univ.sum fun k : Fin S => P k * dPRow k)

/-- FA-backward spelling of the same softmax JVP formula. -/
noncomputable def softmaxJVP {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) : ℝ :=
  P j * (dPRow j - Finset.univ.sum fun k : Fin S => P k * dPRow k)

theorem softmax_jvp_identity {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) :
    softmaxJacobianJVP P dPRow j = softmaxJVP P dPRow j := rfl

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

    scores  := tl.dot(q, tl.trans(k)) * $ℝ(scale)
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

  scores := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  p      := tl.exp(scores - lse[:, None])
  dV     := tl.dot(tl.trans(p), dO)
  dP     := tl.dot(dO, tl.trans(v))
  corr   := tl.sum(p * dP, axis = 1)
  dS     := p * (dP - corr[:, None])
  dQ     := tl.dot(dS, k) * $ℝ(scale)
  dK     := tl.dot(tl.trans(dS), q) * $ℝ(scale)

  tl.store($(dQReg) + q_ptrs, dQ)
  tl.store($(dKReg) + k_ptrs, dK)
  tl.store($(dVReg) + v_ptrs, dV)
}

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

/-- Store-stage readback helper for the row-major 2D stores used by the
stripped backward kernel.  It factors out the final proof obligation for each
of `dQ`, `dK`, and `dV`: once the address tile and value tile are in
registers, the corresponding unmasked store reads back exactly that value. -/
theorem store_stage_readback_rowMajor2D
    (outReg : RegionName) (ptrName valueName : RegName)
    (rows cols : Nat) (s : BlockState)
    (valueFn : TileIndex [rows, cols] → ℝ)
    (hPtrs : s.regs .nat [rows, cols] ptrName =
      some (⟨Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols⟩ :
        Tile .nat [rows, cols]))
    (hVal : s.regs .real [rows, cols] valueName = some (Tile.ofReal valueFn)) :
    ∀ idx : TileIndex [rows, cols],
      observeTileAt
        (stepStmt (Stmt.store .real [rows, cols]
          (MemAccess.region outReg (Op.ref .nat [rows, cols] ptrName))
          (Op.ref .real [rows, cols] valueName) MaskOpt.none) s)
        outReg (Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols) idx =
      some (valueFn idx) := by
  intro idx
  have h_inj : Function.Injective
      (fun i : TileIndex [rows, cols] => i.1.val * cols + i.2.1.val) := by
    intro a b h
    have hrow : Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols a =
        Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols b := by
      simpa [Offset.rowMajor2D, Offset.strided] using h
    exact Offset.rowMajor2D_inj (base := 0) (rowStride := cols) (le_refl cols) hrow
  simp [observeTileAt, stepStmt, evalOp, hPtrs, hVal, Tile.ofReal,
    BlockState.writeMemTyped_real, Offset.rowMajor2D, Offset.strided]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj idx]

end FA1Backward

end VeriTile.Examples
