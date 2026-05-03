/-
VeriTile.Examples.FlashAttention1.NaiveKernel

Executable naive FA-1 boundary kernels and correctness bridges for issue #39.
-/

import VeriTile.Examples.FlashAttention1.V1Boundary

namespace VeriTile.Examples

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! ## Direct single-block output math -/

noncomputable def fa1NaiveDirectOut {M S D Bd : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : Tile .real [M, Bd] :=
  let Qp : Tile .real [M, Bd] := Tile.ofReal (padHeadD (Bd := Bd) Q)
  let Kp : Tile .real [S, Bd] := Tile.ofReal (padHeadD (Bd := Bd) K)
  let Vp : Tile .real [S, Bd] := Tile.ofReal (padHeadD (Bd := Bd) V)
  let scores : Tile .real [M, S] :=
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.dot [] Qp (Tile.transpose [] Kp))
      (Tile.scalar ((scale : ℝ) : WithBot ℝ))
  let p : Tile .real [M, S] := Tile.uop WithBot.realExp scores
  let l : Tile .real [M] :=
    Tile.reduceSum (shape := [M, S]) ⟨1, by simp⟩ (keepDims := Bool.false) p
  let o : Tile .real [M, Bd] := Tile.dot [] p Vp
  Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil))
    o (Tile.expandDim ⟨1, by simp⟩ l)

theorem fa1NaiveDirectOut_eq_oFreeBoundary {M S D Bd : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) :
    (fa1NaiveDirectOut Q K V scale).data idx =
      some (FA1MathBoundary.oFreeBoundary S
          (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
          scale 1 idx /
        FA1MathBoundary.lFreeBoundary S
          (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale 1 idx.1) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  unfold fa1NaiveDirectOut
  simp [Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.ofReal,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.expandDim,
        NumericDType.mul, NumericDType.div,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex,
        FA1MathBoundary.oFreeBoundary, FA1MathBoundary.lFreeBoundary,
        FA1MathBoundary.blockIndex?, FA1Math.scaledScore]
  congr 1
  congr 1
  · apply Finset.sum_congr rfl
    intro x _
    congr 1
    ring_nf
  · apply Finset.sum_congr rfl
    intro x _
    congr 1
    ring_nf

theorem fa1NaiveDirectOut_eq_attentionReal {M S D Bd : Nat}
    (hS : 0 < S) (hDLe : D ≤ Bd)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    (fa1NaiveDirectOut Q K V scale).data idx =
      some (attentionReal Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  rw [fa1NaiveDirectOut_eq_oFreeBoundary]
  congr 1
  have hPos : FA1MathBoundary.lFreeBoundary S
      (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale 1 idx.1 ≠ 0 := by
    have h :=
      FA1MathBoundary.lFreeBoundary_final_pos S
        hS (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale 1
        (by omega) idx.1
    positivity
  rw [FA1MathBoundary.oFreeBoundary_div_lFreeBoundary_eq_attentionReal S
      (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
      scale 1 (by omega) idx hPos]
  exact FA1Math.attentionReal_padHeadD_eq hDLe Q K V scale idx hDIdx

private theorem sum_causal_exp_scores_eq_some {M S D : Nat}
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    (∑ x : Fin S,
      WithBot.realExp
        (if x.val ≤ qStart + i.val then
          some ((∑ d : Fin D, Q (i, d, PUnit.unit) * K (x, d, PUnit.unit)) * scale)
        else
          (⊥ : WithBot ℝ)) : WithBot ℝ)
      =
    some (∑ x : Fin S,
      if x.val ≤ qStart + i.val then
        Real.exp (scale * ∑ d : Fin D, Q (i, d, PUnit.unit) * K (x, d, PUnit.unit))
      else
        0) := by
  rw [← WithBot.sum_someTerm_eq_some]
  apply Finset.sum_congr rfl
  intro x _
  by_cases hx : x.val ≤ qStart + i.val
  · simp [hx]
    congr 1
    ring_nf
  · simp [hx]
    change (some (0 : ℝ) : WithBot ℝ) = some 0
    rfl

private theorem sum_causal_exp_scores_mul_v_eq_some {M S D Bd : Nat}
    (qStart : Nat) (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, Bd] → ℝ) (scale : ℝ) (i : Fin M) (dOut : Fin Bd) :
    (∑ x : Fin S,
      Option.map (fun a : ℝ => a * V (x, dOut, PUnit.unit))
        (WithBot.realExp
          (if x.val ≤ qStart + i.val then
            some ((∑ d : Fin D, Q (i, d, PUnit.unit) * K (x, d, PUnit.unit)) * scale)
          else
            (⊥ : WithBot ℝ))) : WithBot ℝ)
      =
    some (∑ x : Fin S,
      (if x.val ≤ qStart + i.val then
        Real.exp (scale * ∑ d : Fin D, Q (i, d, PUnit.unit) * K (x, d, PUnit.unit))
      else
        0) * V (x, dOut, PUnit.unit)) := by
  rw [← WithBot.sum_someTerm_eq_some]
  apply Finset.sum_congr rfl
  intro x _
  by_cases hx : x.val ≤ qStart + i.val
  · simp [hx]
    congr 1
    ring_nf
  · simp [hx]
    change Option.map (fun a : ℝ => a * V (x, dOut, PUnit.unit))
        (some (0 : ℝ) : WithBot ℝ) = some 0
    simp

noncomputable def fa1NaiveCausalDirectOut {M S D Bd : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : Tile .real [M, Bd] :=
  let Qp : Tile .real [M, Bd] := Tile.ofReal (padHeadD (Bd := Bd) Q)
  let Kp : Tile .real [S, Bd] := Tile.ofReal (padHeadD (Bd := Bd) K)
  let Vp : Tile .real [S, Bd] := Tile.ofReal (padHeadD (Bd := Bd) V)
  let scoresRaw : Tile .real [M, S] :=
    ⟨fun idx => Option.map (fun a : ℝ => a * scale)
      ((Tile.dot [] Qp (Tile.transpose [] Kp)).data idx)⟩
  let causal : Tile .bool [M, S] :=
    ⟨fun idx => idx.2.1.val ≤ qStart + idx.1.val⟩
  let scores : Tile .real [M, S] :=
    Tile.select causal scoresRaw ⟨fun _ => (⊥ : WithBot ℝ)⟩
  let p : Tile .real [M, S] := Tile.uop WithBot.realExp scores
  let l : Tile .real [M] :=
    Tile.reduceSum (shape := [M, S]) ⟨1, by simp⟩ (keepDims := Bool.false) p
  let o : Tile .real [M, Bd] := Tile.dot [] p Vp
  Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil))
    o (Tile.expandDim ⟨1, by simp⟩ l)

theorem fa1NaiveCausalDirectOut_eq_oFreeBoundary {M S D Bd : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) :
    (fa1NaiveCausalDirectOut qStart Q K V scale).data idx =
      some (FA1MathCausalBoundary.oFreeBoundary S qStart
          (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
          scale 1 idx /
        FA1MathCausalBoundary.lFreeBoundary S qStart
          (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale 1 idx.1) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  unfold fa1NaiveCausalDirectOut
  simp [Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.ofReal,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.expandDim, Tile.select,
        NumericDType.mul, NumericDType.div,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex]
  rw [sum_causal_exp_scores_mul_v_eq_some qStart
      (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K)
      (padHeadD (Bd := Bd) V) scale i d]
  have hDen :
      (∑ x : Fin S,
        WithBot.realExp
          (if x.val ≤ qStart + i.val then
            some ((∑ x_1 : Fin Bd,
              padHeadD (Bd := Bd) Q (i, x_1, PUnit.unit) *
                padHeadD (Bd := Bd) K (x, x_1, PUnit.unit)) * scale)
          else
            (⊥ : WithBot ℝ)) : WithBot ℝ)
        =
      some (∑ x : Fin S,
        if x.val ≤ qStart + i.val then
          Real.exp (scale * ∑ x_1 : Fin Bd,
            padHeadD (Bd := Bd) Q (i, x_1, PUnit.unit) *
              padHeadD (Bd := Bd) K (x, x_1, PUnit.unit))
        else
          0) := by
    simpa using @sum_causal_exp_scores_eq_some M S Bd qStart
      (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale i
  conv_lhs =>
    arg 3
    change (∑ x : Fin S,
        WithBot.realExp
          (if x.val ≤ qStart + i.val then
            some ((∑ x_1 : Fin Bd,
              padHeadD (Bd := Bd) Q (i, x_1, PUnit.unit) *
                padHeadD (Bd := Bd) K (x, x_1, PUnit.unit)) * scale)
          else
            (⊥ : WithBot ℝ)) : WithBot ℝ)
    rw [hDen]
  simp [FA1MathCausalBoundary.oFreeBoundary, FA1MathCausalBoundary.lFreeBoundary,
        FA1MathBoundary.blockIndex?, FA1Math.scaledScore]

theorem fa1NaiveCausalDirectOut_eq_attentionRealCausalBlock {M S D Bd : Nat}
    (hS : 0 < S) (hDLe : D ≤ Bd) (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    (fa1NaiveCausalDirectOut qStart Q K V scale).data idx =
      some (attentionRealCausalBlock qStart Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  rw [fa1NaiveCausalDirectOut_eq_oFreeBoundary]
  congr 1
  have hPos : FA1MathCausalBoundary.lFreeBoundary S qStart
      (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale 1 idx.1 ≠ 0 := by
    have h :=
      FA1MathCausalBoundary.lFreeBoundary_final_pos S
        hS qStart (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale 1
        (by omega) idx.1
    positivity
  rw [FA1MathCausalBoundary.oFreeBoundary_div_lFreeBoundary_eq_attentionRealCausalBlock S
      qStart (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
      scale 1 (by omega) idx hPos]
  exact FA1Math.attentionRealCausalBlock_padHeadD_eq hDLe qStart Q K V scale idx hDIdx

private def fa1NaiveComputeBoundaryD (M S Bd : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, S] "scores"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := Bd) (N := S)
          (Op.ref .real [M, Bd] "q")
          (Op.transpose (batch := []) (Op.ref .real [S, Bd] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, S] "p"
      (Op.exp (Op.ref .real [M, S] "scores"))
  , Stmt.assign .real [M] "l"
      (Op.reduceSum (shape := [M, S]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, S] "p"))
  , Stmt.assign .real [M, Bd] "o_acc"
      (Op.dot (batch := []) (M := M) (K := S) (N := Bd)
        (Op.ref .real [M, S] "p")
        (Op.ref .real [S, Bd] "v"))
  , Stmt.assign .real [M, Bd] "out"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, Bd] "o_acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l")))
  ]

theorem fa1_naive_compute_boundaryD_correct
    {M S D Bd : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQ : s.regs .real [M, Bd] "q" =
      some (Tile.ofReal (padHeadD (Bd := Bd) Q)))
    (hK : s.regs .real [S, Bd] "k" =
      some (Tile.ofReal (padHeadD (Bd := Bd) K)))
    (hV : s.regs .real [S, Bd] "v" =
      some (Tile.ofReal (padHeadD (Bd := Bd) V))) :
    ∃ s',
      stepStmts (fa1NaiveComputeBoundaryD M S Bd scale) s = some s' ∧
      s'.regs .real [M, Bd] "out" = some (fa1NaiveDirectOut Q K V scale) ∧
      (∀ dtype shape name,
        name ≠ "scores" → name ≠ "p" → name ≠ "l" → name ≠ "o_acc" → name ≠ "out" →
        s'.regs dtype shape name = s.regs dtype shape name) := by
  let scores : Tile .real [M, S] :=
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.dot [] (Tile.ofReal (padHeadD (Bd := Bd) Q))
        (Tile.transpose [] (Tile.ofReal (padHeadD (Bd := Bd) K))))
      (Tile.scalar ((scale : ℝ) : WithBot ℝ))
  let p : Tile .real [M, S] := Tile.uop WithBot.realExp scores
  let l : Tile .real [M] :=
    Tile.reduceSum (shape := [M, S]) ⟨1, by simp⟩ (keepDims := Bool.false) p
  let o : Tile .real [M, Bd] := Tile.dot [] p (Tile.ofReal (padHeadD (Bd := Bd) V))
  let out : Tile .real [M, Bd] :=
    Tile.bop NumericDType.real.div
      (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      o (Tile.expandDim ⟨1, by simp⟩ l)
  let s1 : BlockState := s.setReg "scores" .real [M, S] scores
  let s2 : BlockState := s1.setReg "p" .real [M, S] p
  let s3 : BlockState := s2.setReg "l" .real [M] l
  let s4 : BlockState := s3.setReg "o_acc" .real [M, Bd] o
  let s' : BlockState :=
    s4.setReg "out" .real [M, Bd] out
  refine ⟨s', ?_, ?_, ?_⟩
  · simp [fa1NaiveComputeBoundaryD, stepStmts, stepStmt, evalOp,
      hQ, hK, hV, scores, p, l, o, out, s1, s2, s3, s4, s',
      Option.bind]
    rfl
  · simp [s', s4, out, fa1NaiveDirectOut, scores, p, l, o]
  · intro dtype shape name hScores hP hL hO hOut
    simp [s', s1, s2, s3, s4, BlockState.setReg, hScores, hP, hL, hO, hOut]

private def fa1NaiveComputeCausalBoundaryD (M S Bd : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, S] "scores_raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := Bd) (N := S)
          (Op.ref .real [M, Bd] "q")
          (Op.transpose (batch := []) (Op.ref .real [S, Bd] "k")))
        (Op.const scale))
  , Stmt.assign .bool [M, S] "causal"
      (Op.ge ComparableDType.nat
        (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [S] "offs_n")))
  , Stmt.assign .real [M, S] "scores"
      (Op.where
        (Op.ref .bool [M, S] "causal")
        (Op.ref .real [M, S] "scores_raw")
        (Op.broadcast Op.negInf [M, S]))
  , Stmt.assign .real [M, S] "p"
      (Op.exp (Op.ref .real [M, S] "scores"))
  , Stmt.assign .real [M] "l"
      (Op.reduceSum (shape := [M, S]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, S] "p"))
  , Stmt.assign .real [M, Bd] "o_acc"
      (Op.dot (batch := []) (M := M) (K := S) (N := Bd)
        (Op.ref .real [M, S] "p")
        (Op.ref .real [S, Bd] "v"))
  , Stmt.assign .real [M, Bd] "out"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, Bd] "o_acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l")))
  ]

theorem fa1_naive_compute_causal_boundaryD_correct
    {M S D Bd : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hOffsM : s.regs .nat [M] "offs_m" =
      some (Tile.vec fun i : Fin M => qStart + i.val))
    (hOffsN : s.regs .nat [S] "offs_n" =
      some (Tile.vec fun n : Fin S => n.val))
    (hQ : s.regs .real [M, Bd] "q" =
      some (Tile.ofReal (padHeadD (Bd := Bd) Q)))
    (hK : s.regs .real [S, Bd] "k" =
      some (Tile.ofReal (padHeadD (Bd := Bd) K)))
    (hV : s.regs .real [S, Bd] "v" =
      some (Tile.ofReal (padHeadD (Bd := Bd) V))) :
    ∃ s',
      stepStmts (fa1NaiveComputeCausalBoundaryD M S Bd scale) s = some s' ∧
      s'.regs .real [M, Bd] "out" = some (fa1NaiveCausalDirectOut qStart Q K V scale) ∧
      (∀ dtype shape name,
        name ≠ "scores_raw" → name ≠ "causal" → name ≠ "scores" → name ≠ "p" →
        name ≠ "l" → name ≠ "o_acc" → name ≠ "out" →
        s'.regs dtype shape name = s.regs dtype shape name) := by
  let scoresRaw : Tile .real [M, S] :=
    ⟨fun idx => Option.map (fun a : ℝ => a * scale)
      ((Tile.dot [] (Tile.ofReal (padHeadD (Bd := Bd) Q))
        (Tile.transpose [] (Tile.ofReal (padHeadD (Bd := Bd) K)))).data idx)⟩
  let causal : Tile .bool [M, S] :=
    ⟨fun idx => idx.2.1.val ≤ qStart + idx.1.val⟩
  let scores : Tile .real [M, S] :=
    Tile.select causal scoresRaw ⟨fun _ => (⊥ : WithBot ℝ)⟩
  let p : Tile .real [M, S] := Tile.uop WithBot.realExp scores
  let l : Tile .real [M] :=
    Tile.reduceSum (shape := [M, S]) ⟨1, by simp⟩ (keepDims := Bool.false) p
  let o : Tile .real [M, Bd] := Tile.dot [] p (Tile.ofReal (padHeadD (Bd := Bd) V))
  let out : Tile .real [M, Bd] :=
    Tile.bop NumericDType.real.div
      (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      o (Tile.expandDim ⟨1, by simp⟩ l)
  let s1 : BlockState := s.setReg "scores_raw" .real [M, S] scoresRaw
  let s2 : BlockState := s1.setReg "causal" .bool [M, S] causal
  let s3 : BlockState := s2.setReg "scores" .real [M, S] scores
  let s4 : BlockState := s3.setReg "p" .real [M, S] p
  let s5 : BlockState := s4.setReg "l" .real [M] l
  let s6 : BlockState := s5.setReg "o_acc" .real [M, Bd] o
  let s' : BlockState :=
    s6.setReg "out" .real [M, Bd] out
  refine ⟨s', ?_, ?_, ?_⟩
  · simp [fa1NaiveComputeCausalBoundaryD, stepStmts, stepStmt, evalOp,
      hOffsM, hOffsN, hQ, hK, hV, scoresRaw, causal, scores, p, l, o, out,
      s1, s2, s3, s4, s5, s6, s',
      Tile.bop, Tile.cop, Tile.expandDim, Tile.select,
      NumericDType.add, NumericDType.mul, ComparableDType.ge,
      TileShape.dropInsertedIndex, Option.bind]
    rfl
  · simp [s', s6, out, fa1NaiveCausalDirectOut, scoresRaw, causal, scores, p, l, o]
  · intro dtype shape name hRaw hCausal hScores hP hL hO hOut
    simp [s', s1, s2, s3, s4, s5, s6, BlockState.setReg,
      hRaw, hCausal, hScores, hP, hL, hO, hOut]

private def fa1NaivePreBoundaryD (qReg kReg vReg : RegionName)
    (M S Bd S_q D : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_qb" (Op.programId 0)
  , Stmt.assign .nat [] "pid_h"  (Op.programId 1)
  , Stmt.assign .nat [] "pid_b"  (Op.programId 2)
  , Stmt.assign .nat [] "q_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat sQB))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat sQH)))
  , Stmt.assign .nat [] "k_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat sKB))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat sKH)))
  , Stmt.assign .nat [] "v_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat sVB))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat sVH)))
  , Stmt.assign .nat [] "o_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat sOB))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat sOH)))
  , Stmt.assign .nat [M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_qb") (Op.constNat M))
        (Op.arange M))
  , Stmt.assign .nat [S] "offs_n" (Op.arange S)
  , Stmt.assign .nat [Bd] "offs_d" (Op.arange Bd)
  , Stmt.assign .nat [M, Bd] "q_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "q_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat sQS)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sQD)))
  , Stmt.assign .bool [M, Bd] "q_seq_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_q))
  , Stmt.assign .bool [M, Bd] "q_d_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
        (Op.constNat D))
  , Stmt.assign .bool [M, Bd] "q_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .bool [M, Bd] "q_seq_mask")
        (Op.ref .bool [M, Bd] "q_d_mask"))
  , Stmt.assign .real [M, Bd] "q"
      (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, Bd] "q_ptrs")) (MaskOpt.maskOther (Op.ref .bool [M, Bd] "q_mask") (Op.full [M, Bd] (Op.const 0))))
  , Stmt.assign .nat [S, Bd] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "k_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [S] "offs_n"))
            (Op.constNat sKN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sKD)))
  , Stmt.assign .nat [S, Bd] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "v_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [S] "offs_n"))
            (Op.constNat sVN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sVD)))
  , Stmt.assign .bool [S, Bd] "kv_d_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [S] "offs_n"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
        (Op.constNat D))
  , Stmt.assign .real [S, Bd] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [S, Bd] "k_ptrs")) (MaskOpt.maskOther (Op.ref .bool [S, Bd] "kv_d_mask") (Op.full [S, Bd] (Op.const 0))))
  , Stmt.assign .real [S, Bd] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [S, Bd] "v_ptrs")) (MaskOpt.maskOther (Op.ref .bool [S, Bd] "kv_d_mask") (Op.full [S, Bd] (Op.const 0))))
  ]

theorem fa1_naive_pre_boundaryD_correct
    {M S D Bd S_q : Nat}
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (s : BlockState)
    (hPids0 : s.pids 0 = qb) (hPids1 : s.pids 1 = headIdx) (hPids2 : s.pids 2 = batch)
    (hQIn : ∀ idx : TileIndex [M, D],
        qb * M + idx.1.val < S_q →
        s.mem qReg
          (batch * sQB + headIdx * sQH + qb * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ qb * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S, D] =>
          batch * sKB + headIdx * sKH + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S, D] =>
          batch * sVB + headIdx * sVH + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH) s = some s0 ∧
      s0.regs .nat [M] "offs_m" = some (Tile.vec fun i : Fin M => qb * M + i.val) ∧
      s0.regs .nat [S] "offs_n" = some (Tile.vec fun n : Fin S => n.val) ∧
      s0.regs .nat [Bd] "offs_d" = some (Tile.vec fun d : Fin Bd => d.val) ∧
      s0.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
      s0.regs .real [M, Bd] "q" = some (Tile.ofReal (padHeadD (Bd := Bd) Q)) ∧
      s0.regs .real [S, Bd] "k" = some (Tile.ofReal (padHeadD (Bd := Bd) K)) ∧
      s0.regs .real [S, Bd] "v" = some (Tile.ofReal (padHeadD (Bd := Bd) V)) := by
  let qBase : Nat := batch * sQB + headIdx * sQH
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let oBase : Nat := batch * sOB + headIdx * sOH
  let qPtrs : Tile .nat [M, Bd] :=
    ⟨fun idx => qBase + (qb * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qSeqMask : Tile .bool [M, Bd] :=
    ⟨fun idx => qb * M + idx.1.val < S_q⟩
  let qDMask : Tile .bool [M, Bd] :=
    ⟨fun idx => idx.2.1.val < D⟩
  let qMask : Tile .bool [M, Bd] :=
    ⟨fun idx => (qb * M + idx.1.val < S_q) && (idx.2.1.val < D)⟩
  let qLoaded : Tile .real [M, Bd] :=
    ⟨fun idx =>
      if h : (qb * M + idx.1.val < S_q) ∧ (idx.2.1.val < D) then
        some (s.readMem qReg
          (qBase + (qb * M + idx.1.val) * sQS + idx.2.1.val * sQD))
      else some 0⟩
  let kPtrs : Tile .nat [S, Bd] :=
    ⟨fun idx => kBase + idx.1.val * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [S, Bd] :=
    ⟨fun idx => vBase + idx.1.val * sVN + idx.2.1.val * sVD⟩
  let kvDMask : Tile .bool [S, Bd] := ⟨fun idx => idx.2.1.val < D⟩
  let kLoaded : Tile .real [S, Bd] :=
    ⟨fun idx =>
      if h : idx.2.1.val < D then
        some (s.readMem kReg (kBase + idx.1.val * sKN + idx.2.1.val * sKD))
      else some 0⟩
  let vLoaded : Tile .real [S, Bd] :=
    ⟨fun idx =>
      if h : idx.2.1.val < D then
        some (s.readMem vReg (vBase + idx.1.val * sVN + idx.2.1.val * sVD))
      else some 0⟩
  let s1 : BlockState := s.setReg "pid_qb" .nat [] (Tile.scalar qb)
  let s2 : BlockState := s1.setReg "pid_h" .nat [] (Tile.scalar headIdx)
  let s3 : BlockState := s2.setReg "pid_b" .nat [] (Tile.scalar batch)
  let s4 : BlockState := s3.setReg "q_base_off" .nat [] (Tile.scalar qBase)
  let s5 : BlockState := s4.setReg "k_base_off" .nat [] (Tile.scalar kBase)
  let s6 : BlockState := s5.setReg "v_base_off" .nat [] (Tile.scalar vBase)
  let s7 : BlockState := s6.setReg "o_base_off" .nat [] (Tile.scalar oBase)
  let s8 : BlockState := s7.setReg "offs_m" .nat [M] (Tile.vec fun i : Fin M => qb * M + i.val)
  let s9 : BlockState := s8.setReg "offs_n" .nat [S] (Tile.vec fun n : Fin S => n.val)
  let s10 : BlockState := s9.setReg "offs_d" .nat [Bd] (Tile.vec fun d : Fin Bd => d.val)
  let s11 : BlockState := s10.setReg "q_ptrs" .nat [M, Bd] qPtrs
  let s12 : BlockState := s11.setReg "q_seq_mask" .bool [M, Bd] qSeqMask
  let s13 : BlockState := s12.setReg "q_d_mask" .bool [M, Bd] qDMask
  let s14 : BlockState := s13.setReg "q_mask" .bool [M, Bd] qMask
  let s15 : BlockState := s14.setReg "q" .real [M, Bd] qLoaded
  let s16 : BlockState := s15.setReg "k_ptrs" .nat [S, Bd] kPtrs
  let s17 : BlockState := s16.setReg "v_ptrs" .nat [S, Bd] vPtrs
  let s18 : BlockState := s17.setReg "kv_d_mask" .bool [S, Bd] kvDMask
  let s19 : BlockState := s18.setReg "k" .real [S, Bd] kLoaded
  let s0 : BlockState := s19.setReg "v" .real [S, Bd] vLoaded
  have hQLoaded : qLoaded = Tile.ofReal (padHeadD (Bd := Bd) Q) := by
    ext idx
    rw [Tile.ofReal_data]
    by_cases hD : idx.2.1.val < D
    · by_cases hRow : qb * M + idx.1.val < S_q
      · have hBoth : (qb * M + idx.1.val < S_q) ∧ (idx.2.1.val < D) := ⟨hRow, hD⟩
        simp [qLoaded, qBase, hBoth, padHeadD, BlockState.readMem]
        rw [show qBase + (qb * M + idx.1.val) * sQS + idx.2.1.val * sQD =
            batch * sQB + headIdx * sQH + qb * M * sQS
              + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
        exact congrArg some (hQIn (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit) hRow)
      · have hNotBoth : ¬ ((qb * M + idx.1.val < S_q) ∧ (idx.2.1.val < D)) := by
          intro h; exact hRow h.1
        rw [show qLoaded.data idx = some 0 by simp [qLoaded, hNotBoth]]
        simp [padHeadD, hD, hQOut (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit) hRow]
    · have hNotBoth : ¬ ((qb * M + idx.1.val < S_q) ∧ (idx.2.1.val < D)) := by
        intro h; exact hD h.2
      rw [show qLoaded.data idx = some 0 by simp [qLoaded, hNotBoth]]
      simp [padHeadD, hD]
  have hKLoaded : kLoaded = Tile.ofReal (padHeadD (Bd := Bd) K) := by
    ext idx
    rw [Tile.ofReal_data]
    by_cases hD : idx.2.1.val < D
    · simp [kLoaded, kBase, padHeadD, hD, BlockState.readMem]
      exact congrArg some (hK (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit))
    · simp [kLoaded, padHeadD, hD]
  have hVLoaded : vLoaded = Tile.ofReal (padHeadD (Bd := Bd) V) := by
    ext idx
    rw [Tile.ofReal_data]
    by_cases hD : idx.2.1.val < D
    · simp [vLoaded, vBase, padHeadD, hD, BlockState.readMem]
      exact congrArg some (hV (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit))
    · simp [vLoaded, padHeadD, hD]
  refine ⟨s0, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [fa1NaivePreBoundaryD, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Option.bind, TileShape.dropInsertedIndex,
      BlockState.readMem, Tile.vec, Tile.ofReal,
      hPids0, hPids1, hPids2,
      qBase, kBase, vBase, oBase, qPtrs, qSeqMask, qDMask, qMask,
      qLoaded, kPtrs, vPtrs, kvDMask, kLoaded, vLoaded, s0]
    rfl
  · simp [s0, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1]
  · simp [s0, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1]
  · simp [s0, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1]
  · simp [s0, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, oBase]
  · simp [s0, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hQLoaded]
  · simp [s0, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hKLoaded]
  · simp [s0, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hVLoaded]

private def fa1NaiveStoreBoundaryD (outReg : RegionName)
    (M Bd S_q D : Nat) (sOM sOD : Nat) : List Stmt :=
  [ Stmt.assign .nat [M, Bd] "o_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "o_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat sOM)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sOD)))
  , Stmt.assign .bool [M, Bd] "o_seq_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_q))
  , Stmt.assign .bool [M, Bd] "o_d_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
        (Op.constNat D))
  , Stmt.assign .bool [M, Bd] "o_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .bool [M, Bd] "o_seq_mask")
        (Op.ref .bool [M, Bd] "o_d_mask"))
  , Stmt.store .real [M, Bd] (MemAccess.region outReg (Op.ref .nat [M, Bd] "o_ptrs")) (Op.ref .real [M, Bd] "out") (MaskOpt.mask (Op.ref .bool [M, Bd] "o_mask"))
  ]

theorem fa1_naive_store_boundaryD_correct
    {M S D Bd S_q : Nat}
    (outReg : RegionName)
    (qb batch headIdx : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hS : 0 < S) (hDLe : D ≤ Bd)
    (hOffsM : s.regs .nat [M] "offs_m" =
      some (Tile.vec fun i : Fin M => qb * M + i.val))
    (hOffsD : s.regs .nat [Bd] "offs_d" =
      some (Tile.vec fun d : Fin Bd => d.val))
    (hOBase : s.regs .nat [] "o_base_off" =
      some (Tile.scalar (batch * sOB + headIdx * sOH)))
    (hOut : s.regs .real [M, Bd] "out" =
      some (fa1NaiveDirectOut Q K V scale))
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : qb * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (stepStmts (fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  have h_no_collision :
      ∀ k : TileIndex [M, Bd],
        (qb * M + k.1.val < S_q ∧ k.2.1.val < D) →
        batch * sOB + headIdx * sOH + (qb * M + k.1.val) * sOM
            + k.2.1.val * sOD =
          batch * sOB + headIdx * sOH + (qb * M + idx.1.val) * sOM
            + idx.2.1.val * sOD →
        k = idx := by
    intro k hk heq
    obtain ⟨ki, kd, ku⟩ := k
    obtain ⟨ii, id, iu⟩ := idx
    cases ku
    cases iu
    have hEqLogical :
        (⟨ki, ⟨kd.val, hk.2⟩, PUnit.unit⟩ : TileIndex [M, D]) =
          (⟨ii, ⟨id.val, hDIdx⟩, PUnit.unit⟩ : TileIndex [M, D]) := by
      apply hInj
      simpa [Nat.add_mul, Nat.add_assoc] using heq
    have hRowEq : ki = ii := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.1) hEqLogical
    have hDEq : kd.val = id.val := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.2.1.val) hEqLogical
    cases hRowEq
    cases Fin.ext hDEq
    rfl
  simp [observeTileAt, fa1NaiveStoreBoundaryD, stepStmts, stepStmt, evalOp,
        BlockState.setReg, hOffsM, hOffsD, hOBase, hOut,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.vec,
        NumericDType.add, NumericDType.mul,
        ComparableDType.lt, Bool.and_eq_true, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  simp only [BlockState.readMem]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
        (by exact ⟨hLt, hDIdx⟩) h_no_collision]
  rw [fa1NaiveDirectOut_eq_attentionReal hS hDLe Q K V scale idx hDIdx]
  rfl

theorem fa1_naive_store_causal_boundaryD_correct
    {M S D Bd S_q : Nat}
    (outReg : RegionName)
    (qb batch headIdx : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hS : 0 < S) (hDLe : D ≤ Bd)
    (hOffsM : s.regs .nat [M] "offs_m" =
      some (Tile.vec fun i : Fin M => qb * M + i.val))
    (hOffsD : s.regs .nat [Bd] "offs_d" =
      some (Tile.vec fun d : Fin Bd => d.val))
    (hOBase : s.regs .nat [] "o_base_off" =
      some (Tile.scalar (batch * sOB + headIdx * sOH)))
    (hOut : s.regs .real [M, Bd] "out" =
      some (fa1NaiveCausalDirectOut (qb * M) Q K V scale))
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : qb * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (stepStmts (fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (qb * M) Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  have h_no_collision :
      ∀ k : TileIndex [M, Bd],
        (qb * M + k.1.val < S_q ∧ k.2.1.val < D) →
        batch * sOB + headIdx * sOH + (qb * M + k.1.val) * sOM
            + k.2.1.val * sOD =
          batch * sOB + headIdx * sOH + (qb * M + idx.1.val) * sOM
            + idx.2.1.val * sOD →
        k = idx := by
    intro k hk heq
    obtain ⟨ki, kd, ku⟩ := k
    obtain ⟨ii, id, iu⟩ := idx
    cases ku
    cases iu
    have hEqLogical :
        (⟨ki, ⟨kd.val, hk.2⟩, PUnit.unit⟩ : TileIndex [M, D]) =
          (⟨ii, ⟨id.val, hDIdx⟩, PUnit.unit⟩ : TileIndex [M, D]) := by
      apply hInj
      simpa [Nat.add_mul, Nat.add_assoc] using heq
    have hRowEq : ki = ii := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.1) hEqLogical
    have hDEq : kd.val = id.val := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.2.1.val) hEqLogical
    cases hRowEq
    cases Fin.ext hDEq
    rfl
  simp [observeTileAt, fa1NaiveStoreBoundaryD, stepStmts, stepStmt, evalOp,
        BlockState.setReg, hOffsM, hOffsD, hOBase, hOut,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.vec,
        NumericDType.add, NumericDType.mul,
        ComparableDType.lt, Bool.and_eq_true, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  simp only [BlockState.readMem]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
        (by exact ⟨hLt, hDIdx⟩) h_no_collision]
  rw [fa1NaiveCausalDirectOut_eq_attentionRealCausalBlock hS hDLe (qb * M) Q K V scale idx hDIdx]
  rfl

def fa1NaiveVerifiedForwardKernelStridedBoundaryD
    (qReg kReg vReg outReg : RegionName)
    (M Bd S_q S D : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) : Kernel :=
  { inputs := [qReg, kReg, vReg]
    outputs := [outReg]
    body :=
      fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
      fa1NaiveComputeBoundaryD M S Bd scale ++
      fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD }

def fa1NaiveVerifiedForwardKernelStridedCausalBoundaryD
    (qReg kReg vReg outReg : RegionName)
    (M Bd S_q S D : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) : Kernel :=
  { inputs := [qReg, kReg, vReg]
    outputs := [outReg]
    body :=
      fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
      fa1NaiveComputeCausalBoundaryD M S Bd scale ++
      fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD }

theorem fa1_naive_forward_correct_strided_boundaryD
    {M S D Bd S_q : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hS : 0 < S) (hDLe : D ≤ Bd)
    (hPids0 : s.pids 0 = qb) (hPids1 : s.pids 1 = headIdx) (hPids2 : s.pids 2 = batch)
    (hQIn : ∀ idx : TileIndex [M, D],
        qb * M + idx.1.val < S_q →
        s.mem qReg
          (batch * sQB + headIdx * sQH + qb * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ qb * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S, D] =>
          batch * sKB + headIdx * sKH + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S, D] =>
          batch * sVB + headIdx * sVH + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : qb * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1NaiveVerifiedForwardKernelStridedBoundaryD qReg kReg vReg outReg
            M Bd S_q S D
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  obtain ⟨s0, hPre, hOffsM0, _hOffsN0, hOffsD0, hOBase0, hQ0, hK0, hV0⟩ :=
    fa1_naive_pre_boundaryD_correct qReg kReg vReg qb headIdx batch
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH
      Q K V s hPids0 hPids1 hPids2 hQIn hQOut hK hV
  obtain ⟨s1, hCompute, hOut1, hPreserve⟩ :=
    fa1_naive_compute_boundaryD_correct Q K V scale s0 hQ0 hK0 hV0
  have hOffsM1 : s1.regs .nat [M] "offs_m" =
      some (Tile.vec fun i : Fin M => qb * M + i.val) := by
    rw [hPreserve .nat [M] "offs_m" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hOffsM0
  have hOffsD1 : s1.regs .nat [Bd] "offs_d" =
      some (Tile.vec fun d : Fin Bd => d.val) := by
    rw [hPreserve .nat [Bd] "offs_d" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hOffsD0
  have hOBase1 : s1.regs .nat [] "o_base_off" =
      some (Tile.scalar (batch * sOB + headIdx * sOH)) := by
    rw [hPreserve .nat [] "o_base_off" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hOBase0
  rw [show exec (fa1NaiveVerifiedForwardKernelStridedBoundaryD qReg kReg vReg outReg
            M Bd S_q S D
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD scale) s =
          stepStmts
            ((fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
                sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
              fa1NaiveComputeBoundaryD M S Bd scale) ++
              fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD) s by
        simp [exec, fa1NaiveVerifiedForwardKernelStridedBoundaryD, List.append_assoc]]
  rw [show (fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
                sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
              fa1NaiveComputeBoundaryD M S Bd scale) ++
              fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD =
            fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
                sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
              (fa1NaiveComputeBoundaryD M S Bd scale ++
                fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD) by
        rw [List.append_assoc]]
  rw [stepStmts.append_some hPre]
  rw [stepStmts.append_some hCompute]
  exact fa1_naive_store_boundaryD_correct outReg qb batch headIdx sOB sOH sOM sOD
    Q K V scale s1 hS hDLe hOffsM1 hOffsD1 hOBase1 hOut1 hInj idx hLt hDIdx

theorem fa1_naive_forward_correct_strided_causal_boundaryD
    {M S D Bd S_q : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hS : 0 < S) (hDLe : D ≤ Bd)
    (hPids0 : s.pids 0 = qb) (hPids1 : s.pids 1 = headIdx) (hPids2 : s.pids 2 = batch)
    (hQIn : ∀ idx : TileIndex [M, D],
        qb * M + idx.1.val < S_q →
        s.mem qReg
          (batch * sQB + headIdx * sQH + qb * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ qb * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S, D] =>
          batch * sKB + headIdx * sKH + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S, D] =>
          batch * sVB + headIdx * sVH + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : qb * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1NaiveVerifiedForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
            M Bd S_q S D
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (qb * M) Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  obtain ⟨s0, hPre, hOffsM0, hOffsN0, hOffsD0, hOBase0, hQ0, hK0, hV0⟩ :=
    fa1_naive_pre_boundaryD_correct qReg kReg vReg qb headIdx batch
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH
      Q K V s hPids0 hPids1 hPids2 hQIn hQOut hK hV
  obtain ⟨s1, hCompute, hOut1, hPreserve⟩ :=
    fa1_naive_compute_causal_boundaryD_correct (qb * M) Q K V scale s0
      hOffsM0 hOffsN0 hQ0 hK0 hV0
  have hOffsM1 : s1.regs .nat [M] "offs_m" =
      some (Tile.vec fun i : Fin M => qb * M + i.val) := by
    rw [hPreserve .nat [M] "offs_m" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hOffsM0
  have hOffsD1 : s1.regs .nat [Bd] "offs_d" =
      some (Tile.vec fun d : Fin Bd => d.val) := by
    rw [hPreserve .nat [Bd] "offs_d" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hOffsD0
  have hOBase1 : s1.regs .nat [] "o_base_off" =
      some (Tile.scalar (batch * sOB + headIdx * sOH)) := by
    rw [hPreserve .nat [] "o_base_off" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hOBase0
  rw [show exec (fa1NaiveVerifiedForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
            M Bd S_q S D
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD scale) s =
          stepStmts
            ((fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
                sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
              fa1NaiveComputeCausalBoundaryD M S Bd scale) ++
              fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD) s by
        simp [exec, fa1NaiveVerifiedForwardKernelStridedCausalBoundaryD, List.append_assoc]]
  rw [show (fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
                sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
              fa1NaiveComputeCausalBoundaryD M S Bd scale) ++
              fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD =
            fa1NaivePreBoundaryD qReg kReg vReg M S Bd S_q D
                sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD sOB sOH ++
              (fa1NaiveComputeCausalBoundaryD M S Bd scale ++
                fa1NaiveStoreBoundaryD outReg M Bd S_q D sOM sOD) by
        rw [List.append_assoc]]
  rw [stepStmts.append_some hPre]
  rw [stepStmts.append_some hCompute]
  exact fa1_naive_store_causal_boundaryD_correct outReg qb batch headIdx sOB sOH sOM sOD
    Q K V scale s1 hS hDLe hOffsM1 hOffsD1 hOBase1 hOut1 hInj idx hLt hDIdx

theorem fa1_naive_forward_correct_4D_boundaryD
    {B H S_q S_k D Bd M : Nat}
    (hSk : 0 < S_k) (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1NaiveVerifiedForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.mem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩, d, PUnit.unit)
    show s.mem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩, d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d hIn]
    exact h
  have hQOut_inner : ∀ tileIdx : TileIndex [M, D],
      ¬ s.pids 0 * M + tileIdx.1.val < S_q →
      slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) tileIdx = 0 := by
    intro tileIdx hOut
    obtain ⟨i, d, _⟩ := tileIdx
    exact slice4DQRowsBoundary_of_not_lt M Q4D _ _ _ i d hOut
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.mem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.mem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD) := by
    have hOValidLocal : Offset.StridesValid [M, D] [sOM, sOD] :=
      ⟨hOValid.2.2.1, hOValid.2.2.2.1, trivial⟩
    have hStrInj := Offset.strided_inj
        (s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM)
        hOValidLocal
    intro a b hab
    apply hStrInj
    obtain ⟨a₁, a₂, _⟩ := a
    obtain ⟨b₁, b₂, _⟩ := b
    show Offset.strided [M, D] [sOM, sOD] _ _ =
         Offset.strided [M, D] [sOM, sOD] _ _
    simp only [Offset.strided]
    have := hab
    simp only at this
    omega
  rw [fa1_naive_forward_correct_strided_boundaryD qReg kReg vReg outReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
        (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        scale s hSk hDLe rfl rfl rfl hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hLt hDIdx]
  congr 1
  obtain ⟨i, d, u⟩ := idx
  cases u
  rw [attentionReal4D_slice]
  apply attentionReal_row_eq
  intro d'
  rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d' hLt]
  rfl

theorem fa1_naive_forward_correct_4D_causal_boundaryD
    {B H S_q S_k D Bd M : Nat}
    (hSk : 0 < S_k) (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1NaiveVerifiedForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.mem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩, d, PUnit.unit)
    show s.mem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩, d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d hIn]
    exact h
  have hQOut_inner : ∀ tileIdx : TileIndex [M, D],
      ¬ s.pids 0 * M + tileIdx.1.val < S_q →
      slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) tileIdx = 0 := by
    intro tileIdx hOut
    obtain ⟨i, d, _⟩ := tileIdx
    exact slice4DQRowsBoundary_of_not_lt M Q4D _ _ _ i d hOut
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.mem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.mem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD) := by
    have hOValidLocal : Offset.StridesValid [M, D] [sOM, sOD] :=
      ⟨hOValid.2.2.1, hOValid.2.2.2.1, trivial⟩
    have hStrInj := Offset.strided_inj
        (s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM)
        hOValidLocal
    intro a b hab
    apply hStrInj
    obtain ⟨a₁, a₂, _⟩ := a
    obtain ⟨b₁, b₂, _⟩ := b
    show Offset.strided [M, D] [sOM, sOD] _ _ =
         Offset.strided [M, D] [sOM, sOD] _ _
    simp only [Offset.strided]
    have := hab
    simp only at this
    omega
  rw [fa1_naive_forward_correct_strided_causal_boundaryD qReg kReg vReg outReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
        (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        scale s hSk hDLe rfl rfl rfl hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hLt hDIdx]
  congr 1
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hLtI : s.pids 0 * M + i.val < S_q := by
    simpa using hLt
  rw [attentionReal4DCausal_slice]
  unfold attentionRealCausalBlock attentionRealCausal
  simp [sliceBH, slice4DQRowsBoundary, hLtI]

theorem fa1_naive_forward_correct_4D_boundaryD_views
    {B H S_q S_k D Bd M : Nat}
    (hSk : 0 < S_k) (hDLe : D ≤ Bd)
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
          (exec (fa1NaiveVerifiedForwardKernelStridedBoundaryD
              views.qReg views.kReg views.vReg views.outReg
              M Bd S_q S_k D
              views.layout.qB views.layout.qH views.layout.qS views.layout.qD
              views.layout.kB views.layout.kH views.layout.kS views.layout.kD
              views.layout.vB views.layout.vH views.layout.vS views.layout.vD
              views.layout.oB views.layout.oH views.layout.oS views.layout.oD
              scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView,
         FA1Views4D.outBlockOffsetD, TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_naive_forward_correct_4D_boundaryD hSk hDLe
      views.qReg views.kReg views.vReg views.outReg
      views.layout.qB views.layout.qH views.layout.qS views.layout.qD
      views.layout.kB views.layout.kH views.layout.kS views.layout.kD
      views.layout.vB views.layout.vH views.layout.vS views.layout.vD
      views.layout.oB views.layout.oH views.layout.oS views.layout.oD
      Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D
      views.layout.hOValid idx hLt hDIdx

theorem fa1_naive_forward_correct_4D_causal_boundaryD_views
    {B H S_q S_k D Bd M : Nat}
    (hSk : 0 < S_k) (hDLe : D ≤ Bd)
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
          (exec (fa1NaiveVerifiedForwardKernelStridedCausalBoundaryD
              views.qReg views.kReg views.vReg views.outReg
              M Bd S_q S_k D
              views.layout.qB views.layout.qH views.layout.qS views.layout.qD
              views.layout.kB views.layout.kH views.layout.kS views.layout.kD
              views.layout.vB views.layout.vH views.layout.vS views.layout.vD
              views.layout.oB views.layout.oH views.layout.oS views.layout.oD
              scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView,
         FA1Views4D.outBlockOffsetD, TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_naive_forward_correct_4D_causal_boundaryD hSk hDLe
      views.qReg views.kReg views.vReg views.outReg
      views.layout.qB views.layout.qH views.layout.qS views.layout.qD
      views.layout.kB views.layout.kH views.layout.kS views.layout.kD
      views.layout.vB views.layout.vH views.layout.vS views.layout.vD
      views.layout.oB views.layout.oH views.layout.oS views.layout.oD
      Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D
      views.layout.hOValid idx hLt hDIdx

theorem fa1_boundaryD_refines_naive_kernel_views
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
          views.outReg (views.outBlockOffsetD s M Bd) idx =
      observeTileAt
          (exec (fa1NaiveVerifiedForwardKernelStridedBoundaryD
              views.qReg views.kReg views.vReg views.outReg
              M Bd S_q S_k D
              views.layout.qB views.layout.qH views.layout.qS views.layout.qD
              views.layout.kB views.layout.kH views.layout.kS views.layout.kD
              views.layout.vB views.layout.vH views.layout.vS views.layout.vD
              views.layout.oB views.layout.oH views.layout.oS views.layout.oD
              scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx := by
  intro idx hLt hDIdx
  rw [fa1_forward_correct_4D_boundaryD_views hBk hSk hSkLe hDLe
      views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx]
  rw [fa1_naive_forward_correct_4D_boundaryD_views hSk hDLe
      views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx]

theorem fa1_causal_boundaryD_refines_naive_kernel_views
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
          views.outReg (views.outBlockOffsetD s M Bd) idx =
      observeTileAt
          (exec (fa1NaiveVerifiedForwardKernelStridedCausalBoundaryD
              views.qReg views.kReg views.vReg views.outReg
              M Bd S_q S_k D
              views.layout.qB views.layout.qH views.layout.qS views.layout.qD
              views.layout.kB views.layout.kH views.layout.kS views.layout.kD
              views.layout.vB views.layout.vH views.layout.vS views.layout.vD
              views.layout.oB views.layout.oH views.layout.oS views.layout.oD
              scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx := by
  intro idx hLt hDIdx
  rw [fa1_forward_correct_4D_causal_boundaryD_views hBk hSk hSkLe hDLe
      views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx]
  rw [fa1_naive_forward_correct_4D_causal_boundaryD_views hSk hDLe
      views Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D idx hLt hDIdx]

end VeriTile.Examples
