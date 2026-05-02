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
      (Op.loadMaskOther qReg
        (Op.ref .nat [M, Bd] "q_ptrs")
        (Op.ref .bool [M, Bd] "q_mask")
        (Op.full [M, Bd] (Op.const 0)))
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
      (Op.loadMaskOther kReg
        (Op.ref .nat [S, Bd] "k_ptrs")
        (Op.ref .bool [S, Bd] "kv_d_mask")
        (Op.full [S, Bd] (Op.const 0)))
  , Stmt.assign .real [S, Bd] "v"
      (Op.loadMaskOther vReg
        (Op.ref .nat [S, Bd] "v_ptrs")
        (Op.ref .bool [S, Bd] "kv_d_mask")
        (Op.full [S, Bd] (Op.const 0)))
  ]

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
  , Stmt.storeMask outReg [M, Bd]
      (Op.ref .nat [M, Bd] "o_ptrs")
      (Op.ref .real [M, Bd] "out")
      (Op.ref .bool [M, Bd] "o_mask")
  ]

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

end VeriTile.Examples
