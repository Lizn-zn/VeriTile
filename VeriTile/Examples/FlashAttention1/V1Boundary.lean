/-
VeriTile.Examples.FlashAttention1.V1Boundary

FlashAttention-1 v1 boundary-mask and D-tail proof surface.
-/

import VeriTile.Examples.FlashAttention1.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ### Boundary strided helpers — pre-loop / loop body / post-loop

These are the v1 / boundary-mask analogues of the strided helpers above.
They deliberately expose the extra mask registers introduced by the DSL
kernel (`q_mask`, `kv_mask`, `score_mask`, `o_mask`) so the operational
proof can reason about masked loads, masked score lanes, and masked stores
without unfolding the full kernel body every time. -/

private def fa1PreLoopStridedBoundary (qReg : RegionName) (M D S_q : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH : Nat) (sVB sVH : Nat)
    (sOB sOH : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_qb" (Op.programId 0)
  , Stmt.assign .nat [] "pid_h"  (Op.programId 1)
  , Stmt.assign .nat [] "pid_b"  (Op.programId 2)
  , Stmt.assign .nat [] "q_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sQB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sQH)))
  , Stmt.assign .nat [] "k_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sKB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sKH)))
  , Stmt.assign .nat [] "v_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sVB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sVH)))
  , Stmt.assign .nat [] "o_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sOB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sOH)))
  , Stmt.assign .nat [M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_qb")
          (Op.constNat M))
        (Op.arange M))
  , Stmt.assign .nat [D] "offs_d" (Op.arange D)
  , Stmt.assign .nat [M, D] "q_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "q_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat sQS)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sQD)))
  , Stmt.assign .bool [M, D] "q_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_q))
  , Stmt.assign .real [M, D] "q"
      (Op.loadMaskOther qReg
        (Op.ref .nat [M, D] "q_ptrs")
        (Op.ref .bool [M, D] "q_mask")
        (Op.broadcast (Op.const 0) [M, D]))
  , Stmt.assign .real [M] "m_i"
      (Op.full [M] Op.negInf)
  , Stmt.assign .real [M] "l_i"
      (Op.full [M] (Op.const 0))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.full [M, D] (Op.const 0))
  ]

private def fa1LoopBodyStridedBoundary (kReg vReg : RegionName)
    (M D Bk S_k : Nat) (sKN sKD sVN sVD : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "k_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sKN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sKD)))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "v_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sVN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sVD)))
  , Stmt.assign .bool [Bk, D] "kv_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_k))
  , Stmt.assign .real [Bk, D] "k"
      (Op.loadMaskOther kReg
        (Op.ref .nat [Bk, D] "k_ptrs")
        (Op.ref .bool [Bk, D] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, D]))
  , Stmt.assign .real [Bk, D] "v"
      (Op.loadMaskOther vReg
        (Op.ref .nat [Bk, D] "v_ptrs")
        (Op.ref .bool [Bk, D] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, D]))
  , Stmt.assign .real [M, Bk] "scores_raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .bool [M, Bk] "score_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.constNat S_k))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "score_mask")
        (Op.ref .real [M, Bk] "scores_raw")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1LoopBodyStridedCausalBoundary (kReg vReg : RegionName)
    (M D Bk S_k : Nat) (sKN sKD sVN sVD : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "k_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sKN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sKD)))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "v_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sVN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sVD)))
  , Stmt.assign .bool [Bk, D] "kv_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_k))
  , Stmt.assign .real [Bk, D] "k"
      (Op.loadMaskOther kReg
        (Op.ref .nat [Bk, D] "k_ptrs")
        (Op.ref .bool [Bk, D] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, D]))
  , Stmt.assign .real [Bk, D] "v"
      (Op.loadMaskOther vReg
        (Op.ref .nat [Bk, D] "v_ptrs")
        (Op.ref .bool [Bk, D] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, D]))
  , Stmt.assign .real [M, Bk] "scores_raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .bool [M, Bk] "causal"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
  , Stmt.assign .real [M, Bk] "causal_scores"
      (Op.where
        (Op.ref .bool [M, Bk] "causal")
        (Op.ref .real [M, Bk] "scores_raw")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .bool [M, Bk] "score_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.constNat S_k))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "score_mask")
        (Op.ref .real [M, Bk] "causal_scores")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1PostLoopStridedBoundary (outReg : RegionName) (M D S_q : Nat)
    (sOM sOD : Nat) : List Stmt :=
  [ Stmt.assign .real [M, D] "out"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, D] "o_acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l_i")))
  , Stmt.assign .nat [M, D] "o_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "o_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat sOM)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sOD)))
  , Stmt.assign .bool [M, D] "o_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_q))
  , Stmt.storeMask outReg [M, D]
      (Op.ref .nat [M, D] "o_ptrs")
      (Op.ref .real [M, D] "out")
      (Op.ref .bool [M, D] "o_mask")
  ]

/-! ### D-tail boundary strided helpers

These helpers mirror the boundary helpers above, but separate the block
hidden width `Bd` from the logical head dimension `D`. Q/K/V loads and O
stores carry both sequence masks and D-tail masks, combined by
`Op.boolAnd` (DSL `tl.logical_and`). -/

private def fa1PreLoopStridedBoundaryD (qReg : RegionName)
    (M Bd S_q D : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH : Nat) (sVB sVH : Nat)
    (sOB sOH : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_qb" (Op.programId 0)
  , Stmt.assign .nat [] "pid_h"  (Op.programId 1)
  , Stmt.assign .nat [] "pid_b"  (Op.programId 2)
  , Stmt.assign .nat [] "q_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sQB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sQH)))
  , Stmt.assign .nat [] "k_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sKB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sKH)))
  , Stmt.assign .nat [] "v_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sVB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sVH)))
  , Stmt.assign .nat [] "o_base_off"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_b") (Op.constNat sOB))
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_h") (Op.constNat sOH)))
  , Stmt.assign .nat [M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid_qb")
          (Op.constNat M))
        (Op.arange M))
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
        (Op.broadcast (Op.const 0) [M, Bd]))
  , Stmt.assign .real [M] "m_i"
      (Op.full [M] Op.negInf)
  , Stmt.assign .real [M] "l_i"
      (Op.full [M] (Op.const 0))
  , Stmt.assign .real [M, Bd] "o_acc"
      (Op.full [M, Bd] (Op.const 0))
  ]

private def fa1LoopBodyStridedBoundaryD (kReg vReg : RegionName)
    (M Bd Bk S_k D : Nat) (sKN sKD sVN sVD : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, Bd] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "k_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sKN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sKD)))
  , Stmt.assign .nat [Bk, Bd] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "v_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sVN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sVD)))
  , Stmt.assign .bool [Bk, Bd] "kv_seq_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_k))
  , Stmt.assign .bool [Bk, Bd] "kv_d_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
        (Op.constNat D))
  , Stmt.assign .bool [Bk, Bd] "kv_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .bool [Bk, Bd] "kv_seq_mask")
        (Op.ref .bool [Bk, Bd] "kv_d_mask"))
  , Stmt.assign .real [Bk, Bd] "k"
      (Op.loadMaskOther kReg
        (Op.ref .nat [Bk, Bd] "k_ptrs")
        (Op.ref .bool [Bk, Bd] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, Bd]))
  , Stmt.assign .real [Bk, Bd] "v"
      (Op.loadMaskOther vReg
        (Op.ref .nat [Bk, Bd] "v_ptrs")
        (Op.ref .bool [Bk, Bd] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, Bd]))
  , Stmt.assign .real [M, Bk] "scores_raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := Bd) (N := Bk)
          (Op.ref .real [M, Bd] "q")
          (Op.transpose (batch := []) (M := Bk) (N := Bd)
            (Op.ref .real [Bk, Bd] "k")))
        (Op.const scale))
  , Stmt.assign .bool [M, Bk] "score_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.constNat S_k))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "score_mask")
        (Op.ref .real [M, Bk] "scores_raw")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, Bd] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, Bd] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := Bd)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, Bd] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1LoopBodyStridedCausalBoundaryD (kReg vReg : RegionName)
    (M Bd Bk S_k D : Nat) (sKN sKD sVN sVD : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, Bd] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "k_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sKN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sKD)))
  , Stmt.assign .nat [Bk, Bd] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "v_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sVN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sVD)))
  , Stmt.assign .bool [Bk, Bd] "kv_seq_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
            (Op.constNat 0)))
        (Op.constNat S_k))
  , Stmt.assign .bool [Bk, Bd] "kv_d_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
        (Op.constNat D))
  , Stmt.assign .bool [Bk, Bd] "kv_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .bool [Bk, Bd] "kv_seq_mask")
        (Op.ref .bool [Bk, Bd] "kv_d_mask"))
  , Stmt.assign .real [Bk, Bd] "k"
      (Op.loadMaskOther kReg
        (Op.ref .nat [Bk, Bd] "k_ptrs")
        (Op.ref .bool [Bk, Bd] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, Bd]))
  , Stmt.assign .real [Bk, Bd] "v"
      (Op.loadMaskOther vReg
        (Op.ref .nat [Bk, Bd] "v_ptrs")
        (Op.ref .bool [Bk, Bd] "kv_mask")
        (Op.broadcast (Op.const 0) [Bk, Bd]))
  , Stmt.assign .real [M, Bk] "scores_raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := Bd) (N := Bk)
          (Op.ref .real [M, Bd] "q")
          (Op.transpose (batch := []) (M := Bk) (N := Bd)
            (Op.ref .real [Bk, Bd] "k")))
        (Op.const scale))
  , Stmt.assign .bool [M, Bk] "causal"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
  , Stmt.assign .real [M, Bk] "causal_scores"
      (Op.where
        (Op.ref .bool [M, Bk] "causal")
        (Op.ref .real [M, Bk] "scores_raw")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .bool [M, Bk] "score_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
            (Op.constNat 0))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.constNat S_k))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "score_mask")
        (Op.ref .real [M, Bk] "causal_scores")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, Bd] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, Bd] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := Bd)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, Bd] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1PostLoopStridedBoundaryD (outReg : RegionName)
    (M Bd S_q D : Nat) (sOM sOD : Nat) : List Stmt :=
  [ Stmt.assign .real [M, Bd] "out"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, Bd] "o_acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l_i")))
  , Stmt.assign .nat [M, Bd] "o_ptrs"
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

end VeriTile.Examples
