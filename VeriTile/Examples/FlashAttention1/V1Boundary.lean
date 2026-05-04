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
      (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs")) (MaskOpt.maskOther (Op.ref .bool [M, D] "q_mask") (Op.broadcast (Op.const 0) [M, D])))
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
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask") (Op.broadcast (Op.const 0) [Bk, D])))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask") (Op.broadcast (Op.const 0) [Bk, D])))
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
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask") (Op.broadcast (Op.const 0) [Bk, D])))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask") (Op.broadcast (Op.const 0) [Bk, D])))
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
  , Stmt.store .real [M, D] (MemAccess.region outReg (Op.ref .nat [M, D] "o_ptrs")) (Op.ref .real [M, D] "out") (MaskOpt.mask (Op.ref .bool [M, D] "o_mask"))
  ]

@[simp] theorem fa1ForwardKernelStridedBoundary_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks S_q S_k : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) :
    (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg M D Bk numKVBlocks S_q S_k
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
      fa1PreLoopStridedBoundary qReg M D S_q
        sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
      fa1PostLoopStridedBoundary outReg M D S_q sOM sOD := by
  rfl

@[simp] theorem fa1ForwardKernelStridedCausalBoundary_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks S_q S_k : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) :
    (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg M D Bk numKVBlocks S_q S_k
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
      fa1PreLoopStridedBoundary qReg M D S_q
        sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
      fa1PostLoopStridedBoundary outReg M D S_q sOM sOD := by
  rfl

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
      (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, Bd] "q_ptrs")) (MaskOpt.maskOther (Op.ref .bool [M, Bd] "q_mask") (Op.broadcast (Op.const 0) [M, Bd])))
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
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, Bd] "k_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask") (Op.broadcast (Op.const 0) [Bk, Bd])))
  , Stmt.assign .real [Bk, Bd] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, Bd] "v_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask") (Op.broadcast (Op.const 0) [Bk, Bd])))
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
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, Bd] "k_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask") (Op.broadcast (Op.const 0) [Bk, Bd])))
  , Stmt.assign .real [Bk, Bd] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, Bd] "v_ptrs")) (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask") (Op.broadcast (Op.const 0) [Bk, Bd])))
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
  , Stmt.store .real [M, Bd] (MemAccess.region outReg (Op.ref .nat [M, Bd] "o_ptrs")) (Op.ref .real [M, Bd] "out") (MaskOpt.mask (Op.ref .bool [M, Bd] "o_mask"))
  ]

@[simp] theorem fa1ForwardKernelStridedBoundaryD_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks S_q S_k D : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) :
    (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg M Bd Bk numKVBlocks S_q S_k D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
      fa1PreLoopStridedBoundaryD qReg M Bd S_q D
        sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
      fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD := by
  rfl

@[simp] theorem fa1ForwardKernelStridedCausalBoundaryD_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks S_q S_k D : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) :
    (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg M Bd Bk numKVBlocks S_q S_k D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
      fa1PreLoopStridedBoundaryD qReg M Bd S_q D
        sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
      fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD := by
  rfl

/-- Reading the `n`-th KV block through the row-major address expression
used by the loop gives the corresponding `blockIndex` cell of a full
`[Bk * numKVBlocks, D]` input. -/
theorem fa1_block_read
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (n : Nat) (hn : n < numKVBlocks)
    (j : Fin Bk) (d : Fin D) :
    s.readMem region ((n * Bk + j.val) * D + d.val) =
      X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      (n * Bk + j.val) * D + d.val =
        Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D
          (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit) := by
    simp [Offset.rowMajor2D, Offset.strided, FA1Math.blockIndex]
  rw [haddr]
  exact hX _

/-- Tile-level version of `fa1_block_read`: the loop-local block load is the
`Tile.ofReal` view of the corresponding full K/V matrix block. -/
theorem fa1_block_load_tile_eq
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (n : Nat) (hn : n < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region ((n * Bk + idx.1.val) * D + idx.2.1.val))⟩
      : Tile .real [Bk, D])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit)) := by
  apply load_tile_eq_of_InputAt_map
    (s := s) (region := region)
    (addr := fun idx : TileIndex [Bk, D] =>
      (n * Bk + idx.1.val) * D + idx.2.1.val)
    (offsetFn := Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D)
    (embed := fun idx : TileIndex [Bk, D] =>
      (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
        idx.2.1, PUnit.unit))
    (xs := X)
  · intro idx
    simp [Offset.rowMajor2D, Offset.strided, FA1Math.blockIndex]
  · exact hX

/-- Strided variant of `fa1_block_read`: reading the `n`-th KV block at
the strided address `base + (n*Bk + j) * sN + d * sD` recovers the
`blockIndex` cell of the full input. The strided InputAt premise
matches the K / V branches of `P_fa1_strided` (with
`base = batch * sKB + headIdx * sKH`, `sN = sKN`, `sD = sKD`, etc.). -/
theorem fa1_block_read_strided
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) (hn : n < numKVBlocks)
    (j : Fin Bk) (d : Fin D) :
    s.readMem region (base + (n * Bk + j.val) * sN + d.val * sD) =
      X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      base + (n * Bk + j.val) * sN + d.val * sD =
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
            base + idx.1.val * sN + idx.2.1.val * sD)
          (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit) := by
    simp [FA1Math.blockIndex]
  rw [haddr]
  exact hX _

/-- Strided tile-level version of `fa1_block_load_tile_eq`. -/
theorem fa1_block_load_tile_eq_strided
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) (hn : n < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region
          (base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD))⟩
      : Tile .real [Bk, D])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit)) := by
  apply load_tile_eq_of_InputAt_map
    (s := s) (region := region)
    (addr := fun idx : TileIndex [Bk, D] =>
      base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD)
    (offsetFn := fun idx : TileIndex [Bk * numKVBlocks, D] =>
      base + idx.1.val * sN + idx.2.1.val * sD)
    (embed := fun idx : TileIndex [Bk, D] =>
      (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
        idx.2.1, PUnit.unit))
    (xs := X)
  · intro idx
    simp [FA1Math.blockIndex]
  · exact hX

/-- Boundary-masked strided KV block load. Valid local lanes read the logical
`[S_k, D]` tensor through `blockIndex?`; invalid padded lanes are exactly the
explicit `other = 0` value from the Triton load. -/
theorem fa1_block_load_tile_eq_strided_boundary
    {D Bk S_k : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [S_k, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [S_k, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) :
    (⟨fun idx : TileIndex [Bk, D] =>
        if _h : n * Bk + idx.1.val < S_k then
          some (s.readMem region
            (base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD))
        else
          some 0⟩
      : Tile .real [Bk, D])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk n idx.1 with
        | some j => X (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  ext idx
  rw [Tile.ofReal_data]
  by_cases h : n * Bk + idx.1.val < S_k
  · simp [h, FA1MathBoundary.blockIndex?_of_lt]
    rw [BlockState.readMem]
    have haddr :
        base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD =
          (fun idx : TileIndex [S_k, D] =>
              base + idx.1.val * sN + idx.2.1.val * sD)
            (⟨n * Bk + idx.1.val, h⟩, idx.2.1, PUnit.unit) := by
      rfl
    rw [haddr]
    exact congrArg some (hX _)
  · simp [h, FA1MathBoundary.blockIndex?_of_not_lt]

/-- D-tail boundary-masked strided KV block load. A lane reads memory only
when both the sequence lane is valid and the hidden lane is logical
(`d < D`); otherwise it is the explicit `other = 0`. The result is the
boundary block view of `padHeadD X`. -/
theorem fa1_block_load_tile_eq_strided_boundaryD
    {D Bd Bk S_k : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [S_k, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [S_k, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) :
    (⟨fun idx : TileIndex [Bk, Bd] =>
        if _h : (n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
          some (s.readMem region
            (base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD))
        else
          some 0⟩
      : Tile .real [Bk, Bd])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk n idx.1 with
        | some j => padHeadD (Bd := Bd) X (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  ext idx
  rw [Tile.ofReal_data]
  by_cases hD : idx.2.1.val < D
  · by_cases hRow : n * Bk + idx.1.val < S_k
    · have hBoth : (n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) :=
        ⟨hRow, hD⟩
      simp [hBoth, FA1MathBoundary.blockIndex?_of_lt, padHeadD]
      rw [BlockState.readMem]
      have haddr :
          base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD =
            (fun idx : TileIndex [S_k, D] =>
                base + idx.1.val * sN + idx.2.1.val * sD)
              (⟨n * Bk + idx.1.val, hRow⟩,
                ⟨idx.2.1.val, hD⟩, PUnit.unit) := by
        rfl
      rw [haddr]
      exact congrArg some (hX _)
    · have hNotBoth : ¬ ((n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D)) := by
        intro h; exact hRow h.1
      simp [hRow, FA1MathBoundary.blockIndex?_of_not_lt]
  · have hNotBoth : ¬ ((n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D)) := by
      intro h; exact hD h.2
    by_cases hRow : n * Bk + idx.1.val < S_k
    · simp [hRow, FA1MathBoundary.blockIndex?_of_lt, padHeadD, hD]
    · simp [hRow, FA1MathBoundary.blockIndex?_of_not_lt]

/-- Score lane helper for boundary/D-tail loop steps. Once the K tile has
been identified as the boundary block view of `K`, the kernel-side
`scores_raw` followed by the sequence mask is exactly
`FA1MathBoundary.maskedScore`. -/
theorem fa1_boundary_score_lane_eq
    {M D Bk S_k : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat)
    (kLoaded : Tile .real [Bk, D])
    (hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0))
    (i : Fin M) (j : Fin Bk) :
    (if k * Bk + j.val < S_k then
      Option.map (fun a : ℝ => a * scale)
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun d : Fin D => Option.map (fun b => Q (i, d, PUnit.unit) * b)
            (kLoaded.data (j, d, PUnit.unit))))
    else
      (none : WithBot ℝ))
      =
    FA1MathBoundary.maskedScore Bk k Q K scale i j := by
  by_cases h : k * Bk + j.val < S_k
  · rw [if_pos h]
    have hkLoaded : ∀ d : Fin D,
        kLoaded.data (j, d, PUnit.unit)
          = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
      intro d
      have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
        hK_loaded_eq
      simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
      exact this
    rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale i j h]
    have h_sum :
        (∑ x : Fin D, Option.map (fun b : ℝ => Q (i, x, PUnit.unit) * b)
          (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
        = some (∑ x : Fin D,
            Q (i, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
      rw [← WithBot.sum_someTerm_eq_some]
      apply Finset.sum_congr rfl
      intro x _
      rw [hkLoaded x]
      rfl
    rw [h_sum]
    show (some _ : WithBot ℝ) = some _
    unfold FA1Math.scaledScore
    congr 1
    ring
  · rw [if_neg h]
    rw [FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale i j h]
    rfl

/-- Causal score lane helper for boundary/D-tail loop steps. It is the
causal analogue of `fa1_boundary_score_lane_eq`: score computation,
causal masking, then sequence masking equals
`FA1MathCausalBoundary.maskedScore`. -/
theorem fa1_causal_boundary_score_lane_eq
    {M D Bk S_k : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat)
    (kLoaded : Tile .real [Bk, D])
    (hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0))
    (i : Fin M) (j : Fin Bk) :
    (if k * Bk + j.val < S_k then
      if k * Bk + j.val ≤ qStart + i.val then
        Option.map (fun a : ℝ => a * scale)
          (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
            (fun d : Fin D => Option.map (fun b => Q (i, d, PUnit.unit) * b)
              (kLoaded.data (j, d, PUnit.unit))))
      else
        (none : WithBot ℝ)
    else
      (none : WithBot ℝ))
      =
    FA1MathCausalBoundary.maskedScore Bk k qStart Q K scale i j := by
  by_cases hLt : k * Bk + j.val < S_k
  · rw [if_pos hLt]
    by_cases hLe : k * Bk + j.val ≤ qStart + i.val
    · rw [if_pos hLe]
      have hkLoaded : ∀ d : Fin D,
          kLoaded.data (j, d, PUnit.unit)
            = some (K (⟨k * Bk + j.val, hLt⟩, d, PUnit.unit)) := by
        intro d
        have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
          hK_loaded_eq
        simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ hLt] at this
        exact this
      rw [FA1MathCausalBoundary.maskedScore_of_lt_of_le Bk k qStart Q K scale i j hLt hLe]
      have h_sum :
          (∑ x : Fin D, Option.map (fun b : ℝ => Q (i, x, PUnit.unit) * b)
            (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
          = some (∑ x : Fin D,
              Q (i, x, PUnit.unit) * K (⟨k * Bk + j.val, hLt⟩, x, PUnit.unit)) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro x _
        rw [hkLoaded x]
        rfl
      rw [h_sum]
      show (some _ : WithBot ℝ) = some _
      unfold FA1Math.scaledScore
      congr 1
      ring
    · rw [if_neg hLe]
      rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k qStart Q K scale i j hLt hLe]
      rfl
  · rw [if_neg hLt]
    rw [FA1MathCausalBoundary.maskedScore_of_not_lt Bk k qStart Q K scale i j hLt]
    rfl

/-- Boundary-masked strided initialization stage. The Q register is loaded
with the same mask as the v1 kernel: in-bounds rows come from memory, while
out-of-bounds rows are the explicit `other=0`. After that, the loop invariant
only keeps the loaded Q tile and the K/V memory contracts. -/
theorem fa1_preLoop_correct_strided_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qMask : Tile .bool [M, D] :=
    ⟨fun idx => s.pids 0 * M + idx.1.val < S_q⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx =>
      if h : s.pids 0 * M + idx.1.val < S_q then
        some (s.readMem qReg
          (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))
      else
        some 0⟩
  let s0 :=
    (((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
      ).setReg "pid_h" .nat [] (Tile.scalar (s.pids 1))
      ).setReg "pid_b" .nat [] (Tile.scalar (s.pids 2))
      ).setReg "q_base_off" .nat []
        (Tile.scalar (s.pids 2 * sQB + s.pids 1 * sQH))
      ).setReg "k_base_off" .nat []
        (Tile.scalar (s.pids 2 * sKB + s.pids 1 * sKH))
      ).setReg "v_base_off" .nat []
        (Tile.scalar (s.pids 2 * sVB + s.pids 1 * sVH))
      ).setReg "o_base_off" .nat []
        (Tile.scalar (s.pids 2 * sOB + s.pids 1 * sOH))
      ).setReg "offs_m" .nat [M]
        (Tile.vec fun i : Fin M => s.pids 0 * M + i.val)
      ).setReg "offs_d" .nat [D]
        (Tile.vec fun d : Fin D => d.val)
      ).setReg "q_ptrs" .nat [M, D] qPtrs
      ).setReg "q_mask" .bool [M, D] qMask
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    rw [Tile.ofReal_data]
    by_cases h : s.pids 0 * M + idx.1.val < S_q
    · simp [qLoaded, qBase, h]
      rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
      exact congrArg some (hQIn idx h)
    · simp [qLoaded, h, hQOut idx h]
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStridedBoundary, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qMask, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, FA1MathBoundary.mPartial]
    · simp [s0, FA1MathBoundary.lPartial, Tile.ofReal]
    · simp [s0, FA1MathBoundary.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- D-tail boundary-masked strided initialization stage. The loaded Q tile is
the logical Q block padded to block width `Bd`; row-tail lanes come from
`hQOut`, and hidden-dimension tail lanes come from the D mask's `other=0`. -/
theorem fa1_preLoop_correct_strided_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, Bd] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qSeqMask : Tile .bool [M, Bd] :=
    ⟨fun idx => s.pids 0 * M + idx.1.val < S_q⟩
  let qDMask : Tile .bool [M, Bd] :=
    ⟨fun idx => idx.2.1.val < D⟩
  let qMask : Tile .bool [M, Bd] :=
    ⟨fun idx => (s.pids 0 * M + idx.1.val < S_q) && (idx.2.1.val < D)⟩
  let qLoaded : Tile .real [M, Bd] :=
    ⟨fun idx =>
      if h : (s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D) then
        some (s.readMem qReg
          (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))
      else
        some 0⟩
  let s0 :=
    (((((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
      ).setReg "pid_h" .nat [] (Tile.scalar (s.pids 1))
      ).setReg "pid_b" .nat [] (Tile.scalar (s.pids 2))
      ).setReg "q_base_off" .nat []
        (Tile.scalar (s.pids 2 * sQB + s.pids 1 * sQH))
      ).setReg "k_base_off" .nat []
        (Tile.scalar (s.pids 2 * sKB + s.pids 1 * sKH))
      ).setReg "v_base_off" .nat []
        (Tile.scalar (s.pids 2 * sVB + s.pids 1 * sVH))
      ).setReg "o_base_off" .nat []
        (Tile.scalar (s.pids 2 * sOB + s.pids 1 * sOH))
      ).setReg "offs_m" .nat [M]
        (Tile.vec fun i : Fin M => s.pids 0 * M + i.val)
      ).setReg "offs_d" .nat [Bd]
        (Tile.vec fun d : Fin Bd => d.val)
      ).setReg "q_ptrs" .nat [M, Bd] qPtrs
      ).setReg "q_seq_mask" .bool [M, Bd] qSeqMask
      ).setReg "q_d_mask" .bool [M, Bd] qDMask
      ).setReg "q_mask" .bool [M, Bd] qMask
      ).setReg "q" .real [M, Bd] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, Bd] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal (padHeadD (Bd := Bd) Q) := by
    ext idx
    rw [Tile.ofReal_data]
    by_cases hD : idx.2.1.val < D
    · by_cases hRow : s.pids 0 * M + idx.1.val < S_q
      · have hBoth : (s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D) :=
          ⟨hRow, hD⟩
        simp [qLoaded, qBase, hBoth, padHeadD]
        rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
            s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
              + idx.1.val * sQS + idx.2.1.val * sQD by
            simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
        exact congrArg some (hQIn (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit) hRow)
      · have hNotBoth : ¬ ((s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D)) := by
          intro h; exact hRow h.1
        rw [show qLoaded.data idx = some 0 by
          simp [qLoaded, hNotBoth]]
        simp [padHeadD, hD, hQOut (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit) hRow]
    · have hNotBoth : ¬ ((s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D)) := by
        intro h; exact hD h.2
      rw [show qLoaded.data idx = some 0 by
        simp [qLoaded, hNotBoth]]
      simp [padHeadD, hD]
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStridedBoundaryD, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal,
      qPtrs, qSeqMask, qDMask, qMask, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, FA1MathBoundary.mPartial]
    · simp [s0, FA1MathBoundary.lPartial, Tile.ofReal]
    · simp [s0, FA1MathBoundary.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- D-tail causal-boundary initialization stage. Operationally identical to
`fa1_preLoop_correct_strided_boundaryD`; only the invariant's accumulator
namespace changes at `k = 0`. -/
theorem fa1_preLoop_correct_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  rcases fa1_preLoop_correct_strided_boundaryD (Bd := Bd) (Bk := Bk)
      (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV with
    ⟨s0, hExec, hP⟩
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK0, hV0⟩
  refine ⟨s0, hExec, ?_⟩
  refine ⟨hpids0, hpids1, hpids2,
    hpid_qb, hpid_h, hpid_b, hq_base, hk_base, hv_base, ho_base,
    hoffs_m, hoffs_d, hq, ?_, ?_, ?_, hK0, hV0⟩
  · simpa [FA1MathBoundary.mPartial, FA1MathCausalBoundary.mPartial] using hm
  · simpa [FA1MathBoundary.lPartial, FA1MathCausalBoundary.lPartial] using hl
  · simpa [FA1MathBoundary.oPartial, FA1MathCausalBoundary.oPartial] using ho

/-- Causal boundary-masked strided initialization stage. Operationally this
is the same pre-loop as the non-causal boundary kernel; only the loop
invariant's accumulator interpretation changes to `FA1MathCausalBoundary`,
whose `k = 0` state is the same `(-inf, 0, 0)` initialization. -/
theorem fa1_preLoop_correct_strided_causal_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  rcases fa1_preLoop_correct_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV with
    ⟨s0, hExec, hP⟩
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK0, hV0⟩
  refine ⟨s0, hExec, ?_⟩
  refine ⟨hpids0, hpids1, hpids2,
    hpid_qb, hpid_h, hpid_b, hq_base, hk_base, hv_base, ho_base,
    hoffs_m, hoffs_d, hq, ?_, ?_, ?_, hK0, hV0⟩
  · simpa [FA1MathBoundary.mPartial, FA1MathCausalBoundary.mPartial] using hm
  · simpa [FA1MathBoundary.lPartial, FA1MathCausalBoundary.lPartial] using hl
  · simpa [FA1MathBoundary.oPartial, FA1MathCausalBoundary.oPartial] using ho
theorem fa1_postLoop_correct_strided_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      qb * M + idx.1.val < S_q →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathBoundary.lPartial Bk Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  intro idx hIdx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStridedBoundary, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_store idx]
  simp [hIdx]

/-- Causal boundary strided readout stage, raw accumulator form. For
in-bounds query rows (`qb*M + i < S_q`), the masked store writes the
normalized causal-boundary streaming accumulator. -/
theorem fa1_postLoop_correct_strided_causal_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      qb * M + idx.1.val < S_q →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  intro idx hIdx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStridedBoundary, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_store idx]
  simp [hIdx]

/-- D-tail boundary readout stage, raw accumulator form. A D-tail store
writes only lanes satisfying both the Q-row boundary and `d < D`. -/
theorem fa1_postLoop_correct_strided_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      qb * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathBoundary.lPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  intro idx hRow hD
  dsimp [P_fa1_strided_boundaryD] at hP
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
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
          (⟨ii, ⟨id.val, hD⟩, PUnit.unit⟩ : TileIndex [M, D]) := by
      apply hInj
      simpa [Nat.add_mul, Nat.add_assoc] using heq
    have hRowEq : ki = ii := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.1) hEqLogical
    have hDEq : kd.val = id.val := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.2.1.val) hEqLogical
    cases hRowEq
    cases Fin.ext hDEq
    rfl
  simp [observeTileAt, fa1PostLoopStridedBoundaryD, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
        (by exact ⟨hRow, hD⟩) h_no_collision]

/-- D-tail causal-boundary readout stage, raw accumulator form. -/
theorem fa1_postLoop_correct_strided_causal_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      qb * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (qb * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (qb * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  intro idx hRow hD
  dsimp [P_fa1_strided_causal_boundaryD] at hP
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
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
          (⟨ii, ⟨id.val, hD⟩, PUnit.unit⟩ : TileIndex [M, D]) := by
      apply hInj
      simpa [Nat.add_mul, Nat.add_assoc] using heq
    have hRowEq : ki = ii := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.1) hEqLogical
    have hDEq : kd.val = id.val := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.2.1.val) hEqLogical
    cases hRowEq
    cases Fin.ext hDEq
    rfl
  simp [observeTileAt, fa1PostLoopStridedBoundaryD, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
        (by exact ⟨hRow, hD⟩) h_no_collision]

/-- Boundary strided forward correctness in raw streaming form, parameterized
by the boundary loop-step lemma. This is the v1 analogue of
`fa1_forward_correct_strided_causal_raw_of_step`: pre-loop establishes the
boundary invariant, `forLoop_inv` consumes the supplied step theorem, and the
masked post-loop readout gives the raw `oPartial / lPartial` value for
in-bounds query lanes. -/
theorem fa1_forward_correct_strided_boundary_raw_of_step
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathBoundary.lPartial Bk Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_boundary qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hIdx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
        M D Bk numKVBlocks S_q S_k
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundary outReg M D S_q sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_boundary_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hIdx

/-- D-tail boundary strided forward correctness in raw streaming form,
parameterized by the D-tail boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_boundaryD_raw_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathBoundary.lPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_boundaryD (Bd := Bd) (Bk := Bk)
      (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hRow hD
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
        M Bd Bk numKVBlocks S_q S_k D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_boundaryD_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hRow hD

/-- Causal boundary strided forward correctness in raw streaming form,
parameterized by the causal-boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_causal_boundary_raw_of_step
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hIdx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
        M D Bk numKVBlocks S_q S_k
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundary outReg M D S_q sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_causal_boundary_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hIdx

/-- D-tail causal-boundary strided forward correctness in raw streaming form,
parameterized by the D-tail causal-boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_causal_boundaryD_raw_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
      (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hRow hD
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
        M Bd Bk numKVBlocks S_q S_k D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_causal_boundaryD_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hRow hD

/-- D-tail boundary strided forward correctness in canonical spec form,
parameterized by the D-tail boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_boundaryD_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hRow hDIdx
  rw [fa1_forward_correct_strided_boundaryD_raw_of_step
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj hStep idx hRow hDIdx]
  congr 1
  have hL := FA1MathBoundary.lPartial_final_ne_zero hBk hSk
    (padHeadD (Bd := Bd) Q) numKVBlocks (padHeadD (Bd := Bd) K)
    scale hSkLe idx.1
  rw [FA1MathBoundary.streaming_eq_attentionReal hBk
      (padHeadD (Bd := Bd) Q) hSkLe
      (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V) scale idx hL]
  exact FA1Math.attentionReal_padHeadD_eq hDLe Q K V scale idx hDIdx

/-- D-tail causal-boundary strided forward correctness in canonical spec
form, parameterized by the D-tail causal-boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_causal_boundaryD_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hRow hDIdx
  rw [fa1_forward_correct_strided_causal_boundaryD_raw_of_step
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj hStep idx hRow hDIdx]
  congr 1
  rw [FA1MathCausalBoundary.streaming_eq_attentionRealCausalBlock hBk hSk hSkLe
      (s.pids 0 * M)
      (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K)
      (padHeadD (Bd := Bd) V) scale idx]
  exact FA1Math.attentionRealCausalBlock_padHeadD_eq hDLe (s.pids 0 * M) Q K V scale idx hDIdx

set_option maxHeartbeats 5000000 in
/-- D-tail boundary strided loop step. This is the D-tail analogue of
`fa1_step_strided_boundary`: K/V loads are guarded by both sequence and
hidden-dimension masks, producing the padded K/V block used by the
boundary recurrence over block width `Bd`. -/
theorem fa1_step_strided_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  let Qp : TileIndex [M, Bd] → ℝ := padHeadD (Bd := Bd) Q
  let Kp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) K
  let Vp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) V
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] := Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kvSeqMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (k * Bk + idx.1.val < S_k)⟩
  let kvDMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (idx.2.1.val < D)⟩
  let kvMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      decide (k * Bk + idx.1.val < S_k) && decide (idx.2.1.val < D)⟩
  let kLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Kp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Kp] using
      fa1_block_load_tile_eq_strided_boundaryD kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Vp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Vp] using
      fa1_block_load_tile_eq_strided_boundaryD vReg s vBase sVN sVD V hV k
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun d : Fin Bd => Option.map (fun b => Qp (idx.1, d, PUnit.unit) * b)
            (kLoaded.data (idx.2.1, d, PUnit.unit))))⟩
  let scoreMask : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] => decide (k * Bk + idx.2.1.val < S_k)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val < S_k then scoresRaw.data idx else (none : WithBot ℝ)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j : Fin Bk => scores.data (idx.1, j, PUnit.unit))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (Option.map₂ (· - ·)
          (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (Option.map₂ (· - ·) (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathBoundary.lPartial Bk Qp numKVBlocks Kp scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, Bd] :=
    ⟨fun idx : TileIndex [M, Bd] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathBoundary.oPartial Bk Qp numKVBlocks Kp Vp scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, Bd] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, Bd] vPtrs
  let s4 := s3.setReg "kv_seq_mask" .bool [Bk, Bd] kvSeqMask
  let s5 := s4.setReg "kv_d_mask" .bool [Bk, Bd] kvDMask
  let s6 := s5.setReg "kv_mask" .bool [Bk, Bd] kvMask
  let s7 := s6.setReg "k" .real [Bk, Bd] kLoaded
  let s8 := s7.setReg "v" .real [Bk, Bd] vLoaded
  let s9 := s8.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s10 := s9.setReg "score_mask" .bool [M, Bk] scoreMask
  let s11 := s10.setReg "scores" .real [M, Bk] scores
  let s12 := s11.setReg "m_block" .real [M] mBlock
  let s13 := s12.setReg "m_new" .real [M] mNew
  let s14 := s13.setReg "alpha" .real [M] alpha
  let s15 := s14.setReg "p" .real [M, Bk] p
  let s16 := s15.setReg "l_new" .real [M] lNew
  let s17 := s16.setReg "o_acc" .real [M, Bd] oNew
  let s18 := s17.setReg "m_i" .real [M] mNew
  let s' := s18.setReg "l_i" .real [M] lNew
  have h_score_per_j : ∀ (i : Fin M) (j : Fin Bk),
      scores.data (i, j, PUnit.unit)
        = FA1MathBoundary.maskedScore Bk k Qp Kp scale i j := by
    intro i j
    exact fa1_boundary_score_lane_eq Qp Kp scale k kLoaded hK_loaded_eq i j
  refine ⟨s', ?_, ?_⟩
  · simp [fa1LoopBodyStridedBoundaryD, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact hpids0
    · exact hpids1
    · exact hpids2
    · exact hpid_qb
    · exact hpid_h
    · exact hpid_b
    · exact hq_base
    · exact hk_base
    · exact hv_base
    · exact ho_base
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_m]
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_d]
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hq]
    · have h_regs_m_i : s'.regs .real [M] "m_i" = some mNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_m_i]
      congr 1
      ext idx
      show max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1) (mBlock.data idx)
        = FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1
      rw [FA1MathBoundary.mPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      exact h_score_per_j idx.1 j
    · have h_regs_l_i : s'.regs .real [M] "l_i" = some lNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_l_i]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data idx
          = some (FA1MathBoundary.alphaPartial Bk Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathBoundary.lPartial Bk Qp numKVBlocks Kp scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathBoundary.lPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathBoundary.lPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
      rfl
    · have h_regs_o_acc : s'.regs .real [M, Bd] "o_acc" = some oNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathBoundary.alphaPartial Bk Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => Vp (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, Bd] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        intro j
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_pv_sum :
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
            = some (∑ j : Fin Bk,
                match FA1MathBoundary.blockIndex? S_k Bk k j with
                | some jGlobal =>
                    (WithBot.realExp
                      (WithBot.realSub
                        (FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
                        (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1))
                    ).unbotD 0 * Vp (jGlobal, idx.2.1, PUnit.unit)
                | none => 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        rw [h_p_per_j j, h_vLoaded j]
        by_cases h : k * Bk + j.val < S_k
        · simp [FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h]
        · simp [FA1MathBoundary.blockIndex?_of_not_lt _ _ _ _ h]
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathBoundary.oPartial Bk Qp numKVBlocks Kp Vp scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathBoundary.oPartial Bk Qp numKVBlocks Kp Vp scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathBoundary.oPartial_succ_of_lt Bk Qp numKVBlocks Kp Vp scale k hk idx]
      rfl
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- D-tail boundary strided FA-1 forward correctness, raw form. Bundles
`fa1_forward_correct_strided_boundaryD_raw_of_step` with the proven
`fa1_step_strided_boundaryD`. -/
theorem fa1_forward_correct_strided_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathBoundary.lPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_boundaryD_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- D-tail boundary strided FA-1 forward correctness in canonical spec form. -/
theorem fa1_forward_correct_strided_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  exact fa1_forward_correct_strided_boundaryD_of_step hBk hSk hSkLe hDLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

set_option maxHeartbeats 5000000 in
/-- D-tail causal-boundary strided loop step. This is the causal analogue of
`fa1_step_strided_boundaryD`: K/V loads are D-tail masked, scores are causal
masked and then sequence masked, and the recurrence is
`FA1MathCausalBoundary` over padded hidden width `Bd`. -/
theorem fa1_step_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  let Qp : TileIndex [M, Bd] → ℝ := padHeadD (Bd := Bd) Q
  let Kp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) K
  let Vp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) V
  let qStart : Nat := qb * M
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] := Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kvSeqMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (k * Bk + idx.1.val < S_k)⟩
  let kvDMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (idx.2.1.val < D)⟩
  let kvMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      decide (k * Bk + idx.1.val < S_k) && decide (idx.2.1.val < D)⟩
  let kLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Kp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Kp] using
      fa1_block_load_tile_eq_strided_boundaryD kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Vp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Vp] using
      fa1_block_load_tile_eq_strided_boundaryD vReg s vBase sVN sVD V hV k
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun d : Fin Bd => Option.map (fun b => Qp (idx.1, d, PUnit.unit) * b)
            (kLoaded.data (idx.2.1, d, PUnit.unit))))⟩
  let causal : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      decide (k * Bk + idx.2.1.val ≤ qStart + idx.1.val)⟩
  let causalScores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val ≤ qStart + idx.1.val then scoresRaw.data idx
      else (none : WithBot ℝ)⟩
  let scoreMask : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] => decide (k * Bk + idx.2.1.val < S_k)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val < S_k then causalScores.data idx else (none : WithBot ℝ)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j : Fin Bk => scores.data (idx.1, j, PUnit.unit))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (WithBot.realSub
          (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (WithBot.realSub (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathCausalBoundary.lPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, Bd] :=
    ⟨fun idx : TileIndex [M, Bd] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathCausalBoundary.oPartial Bk qStart Qp numKVBlocks Kp Vp scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, Bd] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, Bd] vPtrs
  let s4 := s3.setReg "kv_seq_mask" .bool [Bk, Bd] kvSeqMask
  let s5 := s4.setReg "kv_d_mask" .bool [Bk, Bd] kvDMask
  let s6 := s5.setReg "kv_mask" .bool [Bk, Bd] kvMask
  let s7 := s6.setReg "k" .real [Bk, Bd] kLoaded
  let s8 := s7.setReg "v" .real [Bk, Bd] vLoaded
  let s9 := s8.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s10 := s9.setReg "causal" .bool [M, Bk] causal
  let s11 := s10.setReg "causal_scores" .real [M, Bk] causalScores
  let s12 := s11.setReg "score_mask" .bool [M, Bk] scoreMask
  let s13 := s12.setReg "scores" .real [M, Bk] scores
  let s14 := s13.setReg "m_block" .real [M] mBlock
  let s15 := s14.setReg "m_new" .real [M] mNew
  let s16 := s15.setReg "alpha" .real [M] alpha
  let s17 := s16.setReg "p" .real [M, Bk] p
  let s18 := s17.setReg "l_new" .real [M] lNew
  let s19 := s18.setReg "o_acc" .real [M, Bd] oNew
  let s20 := s19.setReg "m_i" .real [M] mNew
  let s' := s20.setReg "l_i" .real [M] lNew
  have h_score_per_j : ∀ (i : Fin M) (j : Fin Bk),
      scores.data (i, j, PUnit.unit)
        = FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale i j := by
    intro i j
    exact fa1_causal_boundary_score_lane_eq qStart Qp Kp scale k kLoaded hK_loaded_eq i j
  refine ⟨s', ?_, ?_⟩
  · simp [fa1LoopBodyStridedCausalBoundaryD, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.ge, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact hpids0
    · exact hpids1
    · exact hpids2
    · exact hpid_qb
    · exact hpid_h
    · exact hpid_b
    · exact hq_base
    · exact hk_base
    · exact hv_base
    · exact ho_base
    · simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_m]
    · simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_d]
    · simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hq]
    · have h_regs_m_i : s'.regs .real [M] "m_i" = some mNew := by
        simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_m_i]
      congr 1
      ext idx
      show max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1) (mBlock.data idx)
        = FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1
      rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      exact h_score_per_j idx.1 j
    · have h_regs_l_i : s'.regs .real [M] "l_i" = some lNew := by
        simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_l_i]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data idx
          = some (FA1MathCausalBoundary.alphaPartial Bk qStart Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.lPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathCausalBoundary.lPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathCausalBoundary.lPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
      rfl
    · have h_regs_o_acc : s'.regs .real [M, Bd] "o_acc" = some oNew := by
        simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathCausalBoundary.alphaPartial Bk qStart Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => Vp (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, Bd] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        intro j
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_pv_sum :
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
            = some (∑ j : Fin Bk,
                match FA1MathBoundary.blockIndex? S_k Bk k j with
                | some jGlobal =>
                    if jGlobal.val ≤ qStart + idx.1.val then
                      (WithBot.realExp
                        (WithBot.realSub
                          (FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
                          (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1))
                      ).unbotD 0 * Vp (jGlobal, idx.2.1, PUnit.unit)
                    else
                      0
                | none => 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        rw [h_p_per_j j, h_vLoaded j]
        by_cases hLt : k * Bk + j.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k j hLt]
          by_cases hLe : k * Bk + j.val ≤ qStart + idx.1.val
          · simp [hLe]
          · rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k qStart Qp Kp scale idx.1 j hLt hLe]
            simp [hLe]
        · simp [FA1MathBoundary.blockIndex?_of_not_lt _ _ _ _ hLt]
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.oPartial Bk qStart Qp numKVBlocks Kp Vp scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathCausalBoundary.oPartial Bk qStart Qp numKVBlocks Kp Vp scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathCausalBoundary.oPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp Vp scale k hk idx]
      rfl
    · intro idx
      simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · intro idx
      simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- D-tail causal-boundary strided FA-1 forward correctness, raw form. -/
theorem fa1_forward_correct_strided_causal_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_causal_boundaryD_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- D-tail causal-boundary strided FA-1 forward correctness in canonical
causal-block spec form. -/
theorem fa1_forward_correct_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  exact fa1_forward_correct_strided_causal_boundaryD_of_step hBk hSk hSkLe hDLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- Causal boundary strided loop step. One iteration preserves
`P_fa1_strided_causal_boundary`: K/V loads use the logical boundary mask,
scores are first causal-masked and then boundary-masked, and the
online-softmax update follows `FA1MathCausalBoundary`. -/
theorem fa1_step_strided_causal_boundary
    {M D Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] := Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kvMask : Tile .bool [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => decide (k * Bk + idx.1.val < S_k)⟩
  let kLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => V (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary vReg s vBase sVN sVD V hV k
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun d : Fin D => Option.map (fun b => Q (idx.1, d, PUnit.unit) * b)
            (kLoaded.data (idx.2.1, d, PUnit.unit))))⟩
  let causal : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      decide (k * Bk + idx.2.1.val ≤ qb * M + idx.1.val)⟩
  let causalScores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val ≤ qb * M + idx.1.val then scoresRaw.data idx
      else (none : WithBot ℝ)⟩
  let scoreMask : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] => decide (k * Bk + idx.2.1.val < S_k)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val < S_k then causalScores.data idx else (none : WithBot ℝ)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j : Fin Bk => scores.data (idx.1, j, PUnit.unit))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (WithBot.realSub
          (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (WithBot.realSub (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, D] :=
    ⟨fun idx : TileIndex [M, D] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "kv_mask" .bool [Bk, D] kvMask
  let s5 := s4.setReg "k" .real [Bk, D] kLoaded
  let s6 := s5.setReg "v" .real [Bk, D] vLoaded
  let s7 := s6.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s8 := s7.setReg "causal" .bool [M, Bk] causal
  let s9 := s8.setReg "causal_scores" .real [M, Bk] causalScores
  let s10 := s9.setReg "score_mask" .bool [M, Bk] scoreMask
  let s11 := s10.setReg "scores" .real [M, Bk] scores
  let s12 := s11.setReg "m_block" .real [M] mBlock
  let s13 := s12.setReg "m_new" .real [M] mNew
  let s14 := s13.setReg "alpha" .real [M] alpha
  let s15 := s14.setReg "p" .real [M, Bk] p
  let s16 := s15.setReg "l_new" .real [M] lNew
  let s17 := s16.setReg "o_acc" .real [M, D] oNew
  let s18 := s17.setReg "m_i" .real [M] mNew
  let s' := s18.setReg "l_i" .real [M] lNew
  have h_score_per_j : ∀ (i : Fin M) (j : Fin Bk),
      scores.data (i, j, PUnit.unit)
        = FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale i j := by
    intro i j
    by_cases hLt : k * Bk + j.val < S_k
    · show (if k * Bk + ↑j < S_k then causalScores.data (i, j, PUnit.unit)
          else (none : WithBot ℝ)) = _
      rw [if_pos hLt]
      by_cases hLe : k * Bk + j.val ≤ qb * M + i.val
      · show (if k * Bk + ↑j ≤ qb * M + ↑i then scoresRaw.data (i, j, PUnit.unit)
            else (none : WithBot ℝ)) = _
        rw [if_pos hLe]
        have hkLoaded : ∀ d : Fin D,
            kLoaded.data (j, d, PUnit.unit)
              = some (K (⟨k * Bk + j.val, hLt⟩, d, PUnit.unit)) := by
          intro d
          have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
            hK_loaded_eq
          simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ hLt] at this
          exact this
        rw [FA1MathCausalBoundary.maskedScore_of_lt_of_le Bk k (qb * M) Q K scale i j hLt hLe]
        show Option.map (· * scale) _ = _
        have h_sum :
            (∑ x : Fin D, Option.map (fun b : ℝ => Q (i, x, PUnit.unit) * b)
              (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
            = some (∑ x : Fin D,
                Q (i, x, PUnit.unit) * K (⟨k * Bk + j.val, hLt⟩, x, PUnit.unit)) := by
          rw [← WithBot.sum_someTerm_eq_some]
          apply Finset.sum_congr rfl
          intro x _
          rw [hkLoaded x]
          rfl
        rw [h_sum]
        show (some _ : WithBot ℝ) = some _
        unfold FA1Math.scaledScore
        congr 1
        ring
      · show (if k * Bk + ↑j ≤ qb * M + ↑i then scoresRaw.data (i, j, PUnit.unit)
            else (none : WithBot ℝ)) = _
        rw [if_neg hLe]
        rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k (qb * M) Q K scale i j hLt hLe]
        rfl
    · show (if k * Bk + ↑j < S_k then causalScores.data (i, j, PUnit.unit)
          else (none : WithBot ℝ)) = _
      rw [if_neg hLt]
      rw [FA1MathCausalBoundary.maskedScore_of_not_lt Bk k (qb * M) Q K scale i j hLt]
      rfl
  refine ⟨s', ?_, ?_⟩
  · simp [fa1LoopBodyStridedCausalBoundary, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.ge, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact hpids0
    · exact hpids1
    · exact hpids2
    · exact hpid_qb
    · exact hpid_h
    · exact hpid_b
    · exact hq_base
    · exact hk_base
    · exact hv_base
    · exact ho_base
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_m]
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_d]
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hq]
    · have h_regs_m_i : s'.regs .real [M] "m_i" = some mNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_m_i]
      congr 1
      ext idx
      show max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) (mBlock.data idx)
        = FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1
      rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      exact h_score_per_j idx.1 j
    · have h_regs_l_i : s'.regs .real [M] "l_i" = some lNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_l_i]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data idx
          = some (FA1MathCausalBoundary.alphaPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathCausalBoundary.lPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
      rfl
    · have h_regs_o_acc : s'.regs .real [M, D] "o_acc" = some oNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathCausalBoundary.alphaPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => V (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))
              ).unbotD 0) := by
        intro j
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_pv_sum :
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
            = some (∑ j : Fin Bk,
                match FA1MathBoundary.blockIndex? S_k Bk k j with
                | some jGlobal =>
                    if jGlobal.val ≤ qb * M + idx.1.val then
                      (WithBot.realExp
                        (WithBot.realSub
                          (FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
                          (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))
                      ).unbotD 0 * V (jGlobal, idx.2.1, PUnit.unit)
                    else
                      0
                | none => 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        rw [h_p_per_j j, h_vLoaded j]
        by_cases hLt : k * Bk + j.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k j hLt]
          by_cases hLe : k * Bk + j.val ≤ qb * M + idx.1.val
          · simp [hLe]
          · rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k (qb * M) Q K scale idx.1 j hLt hLe]
            simp [hLe]
        · simp [FA1MathBoundary.blockIndex?_of_not_lt _ _ _ _ hLt]
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathCausalBoundary.oPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K V scale k hk idx]
      rfl
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- Causal boundary strided FA-1 forward correctness, raw form. Bundles
`fa1_forward_correct_strided_causal_boundary_raw_of_step` with the proven
causal-boundary loop step. -/
theorem fa1_forward_correct_strided_causal_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_causal_boundary_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal_boundary hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- Causal-boundary strided FA-1 forward correctness in canonical spec form. -/
theorem fa1_forward_correct_strided_causal_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale idx) := by
  intro idx hIdx
  rw [fa1_forward_correct_strided_causal_boundary_raw hBk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj idx hIdx]
  congr 1
  exact FA1MathCausalBoundary.streaming_eq_attentionRealCausalBlock hBk hSk hSkLe
    (s.pids 0 * M) Q K V scale idx

set_option maxHeartbeats 4000000 in
/-- Boundary strided loop step. One iteration of the v1 KV loop preserves
`P_fa1_strided_boundary`: masked K/V loads read logical K/V cells for
in-range lanes and zero for padded lanes; the score mask turns padded score
lanes into `-inf`; the online-softmax update is discharged by the boundary
block lemmas in `FA1MathBoundary`. -/
theorem fa1_step_strided_boundary
    {M D Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  -- Kernel-form witnesses for K/V (matching `tl.load(..., mask, other=0)` exactly).
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kvMask : Tile .bool [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => decide (k * Bk + idx.1.val < S_k)⟩
  -- Kernel-form K/V tiles: `if h: in-bounds then some(readMem) else some 0`.
  let kLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  -- Bridges: kLoaded / vLoaded equal the canonical match-on-blockIndex? form
  -- used by `block_scores_tile_eq` etc.
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => V (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary vReg s vBase sVN sVD V hV k
  -- Kernel-form downstream tiles. Each one is exactly what `evalOp` produces
  -- on the masked K/V load, so the operational first branch closes by `rfl`.
  -- The bridge to canonical (`maskedScore` / `mPartial` / `lPartial` / `oPartial`)
  -- happens in the invariant branch via the `FA1MathBoundary.block_*_tile_eq`
  -- lemmas applied per P-clause.
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun d : Fin D => Option.map (fun b => Q (idx.1, d, PUnit.unit) * b)
            (kLoaded.data (idx.2.1, d, PUnit.unit))))⟩
  let scoreMask : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] => decide (k * Bk + idx.2.1.val < S_k)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val < S_k then scoresRaw.data idx else (none : WithBot ℝ)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j : Fin Bk => scores.data (idx.1, j, PUnit.unit))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (Option.map₂ (· - ·)
          (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (Option.map₂ (· - ·) (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathBoundary.lPartial Bk Q numKVBlocks K scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, D] :=
    ⟨fun idx : TileIndex [M, D] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "kv_mask" .bool [Bk, D] kvMask
  let s5 := s4.setReg "k" .real [Bk, D] kLoaded
  let s6 := s5.setReg "v" .real [Bk, D] vLoaded
  let s7 := s6.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s8 := s7.setReg "score_mask" .bool [M, Bk] scoreMask
  let s9 := s8.setReg "scores" .real [M, Bk] scores
  let s10 := s9.setReg "m_block" .real [M] mBlock
  let s11 := s10.setReg "m_new" .real [M] mNew
  let s12 := s11.setReg "alpha" .real [M] alpha
  let s13 := s12.setReg "p" .real [M, Bk] p
  let s14 := s13.setReg "l_new" .real [M] lNew
  let s15 := s14.setReg "o_acc" .real [M, D] oNew
  let s16 := s15.setReg "m_i" .real [M] mNew
  let s' := s16.setReg "l_i" .real [M] lNew
  refine ⟨s', ?_, ?_⟩
  · simp [fa1LoopBodyStridedBoundary, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · show s.pids 0 = qb; exact hpids0
    · show s.pids 1 = headIdx; exact hpids1
    · show s.pids 2 = batch; exact hpids2
    · show s.regs .nat [] "pid_qb" = some (Tile.scalar qb); exact hpid_qb
    · show s.regs .nat [] "pid_h" = some (Tile.scalar headIdx); exact hpid_h
    · show s.regs .nat [] "pid_b" = some (Tile.scalar batch); exact hpid_b
    · show s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH))
      exact hq_base
    · show s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH))
      exact hk_base
    · show s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH))
      exact hv_base
    · show s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH))
      exact ho_base
    · -- offs_m
      simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_m]
    · -- offs_d
      simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_d]
    · -- q
      simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hq]
    · -- m_i
      have h_regs_m_i : s'.regs .real [M] "m_i" = some mNew := by
        simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_m_i]
      congr 1
      ext idx
      have h_score_per_j : ∀ j : Fin Bk,
          scores.data (idx.1, j, PUnit.unit)
            = FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j := by
        intro j
        by_cases h : k * Bk + j.val < S_k
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_pos h]
          have hkLoaded : ∀ d : Fin D,
              kLoaded.data (j, d, PUnit.unit)
                = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
            intro d
            have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
              hK_loaded_eq
            simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
            exact this
          rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale idx.1 j h]
          show Option.map (· * scale) _ = _
          have h_sum :
              (∑ x : Fin D, Option.map (fun b : ℝ => Q (idx.1, x, PUnit.unit) * b)
                (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
              = some (∑ x : Fin D,
                  Q (idx.1, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
            rw [← WithBot.sum_someTerm_eq_some]
            apply Finset.sum_congr rfl
            intro x _
            rw [hkLoaded x]
            rfl
          rw [h_sum]
          show (some _ : WithBot ℝ) = some _
          unfold FA1Math.scaledScore
          congr 1
          ring
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_neg h]
          rw [FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale idx.1 j h]
          rfl
      show max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1) (mBlock.data idx)
        = FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1
      rw [FA1MathBoundary.mPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      exact h_score_per_j j
    · -- l_i
      have h_regs_l_i : s'.regs .real [M] "l_i" = some lNew := by
        simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_l_i]
      apply congrArg some
      show lNew = Tile.ofReal (fun idx : TileIndex [M] =>
          FA1MathBoundary.lPartial Bk Q numKVBlocks K scale (k + 1) idx.1)
      apply Tile.ext
      intro idx
      have h_score_per_j : ∀ j : Fin Bk,
          scores.data (idx.1, j, PUnit.unit)
            = FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j := by
        intro j
        by_cases h : k * Bk + j.val < S_k
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_pos h]
          have hkLoaded : ∀ d : Fin D,
              kLoaded.data (j, d, PUnit.unit)
                = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
            intro d
            have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
              hK_loaded_eq
            simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
            exact this
          rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale idx.1 j h]
          show Option.map (· * scale) _ = _
          have h_sum :
              (∑ x : Fin D, Option.map (fun b : ℝ => Q (idx.1, x, PUnit.unit) * b)
                (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
              = some (∑ x : Fin D,
                  Q (idx.1, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
            rw [← WithBot.sum_someTerm_eq_some]
            apply Finset.sum_congr rfl
            intro x _
            rw [hkLoaded x]
            rfl
          rw [h_sum]
          show (some _ : WithBot ℝ) = some _
          unfold FA1Math.scaledScore
          congr 1
          ring
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_neg h, FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale idx.1 j h]
          rfl
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j j
      have h_alpha : alpha.data idx
          = some (FA1MathBoundary.alphaPartial Bk Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j j, h_mNew]
        show WithBot.realExp _ = _
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathBoundary.lPartial Bk Q numKVBlocks K scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathBoundary.lPartial Bk Q numKVBlocks K scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathBoundary.lPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
      rfl
    · -- o_acc
      have h_regs_o_acc : s'.regs .real [M, D] "o_acc" = some oNew := by
        simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      show oNew = Tile.ofReal (fun idx : TileIndex [M, D] =>
          FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale (k + 1) idx)
      apply Tile.ext
      intro idx
      have h_score_per_j : ∀ j : Fin Bk,
          scores.data (idx.1, j, PUnit.unit)
            = FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j := by
        intro j
        by_cases h : k * Bk + j.val < S_k
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_pos h]
          have hkLoaded : ∀ d : Fin D,
              kLoaded.data (j, d, PUnit.unit)
                = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
            intro d
            have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
              hK_loaded_eq
            simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
            exact this
          rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale idx.1 j h]
          show Option.map (· * scale) _ = _
          have h_sum :
              (∑ x : Fin D, Option.map (fun b : ℝ => Q (idx.1, x, PUnit.unit) * b)
                (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
              = some (∑ x : Fin D,
                  Q (idx.1, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
            rw [← WithBot.sum_someTerm_eq_some]
            apply Finset.sum_congr rfl
            intro x _
            rw [hkLoaded x]
            rfl
          rw [h_sum]
          show (some _ : WithBot ℝ) = some _
          unfold FA1Math.scaledScore
          congr 1
          ring
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_neg h, FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale idx.1 j h]
          rfl
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathBoundary.alphaPartial Bk Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => V (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))
              ).unbotD 0) := by
        intro j
        show WithBot.realExp _ = _
        rw [h_score_per_j j, h_mNew]
        show WithBot.realExp _ = _
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_pv_sum :
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
            = some (∑ j : Fin Bk,
                match FA1MathBoundary.blockIndex? S_k Bk k j with
                | some jGlobal =>
                    (WithBot.realExp
                      (WithBot.realSub
                        (FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
                        (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))
                    ).unbotD 0 * V (jGlobal, idx.2.1, PUnit.unit)
                | none => 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        rw [h_p_per_j j, h_vLoaded j]
        by_cases h : k * Bk + j.val < S_k
        · simp [FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h]
        · simp [FA1MathBoundary.blockIndex?_of_not_lt _ _ _ _ h]
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathBoundary.oPartial_succ_of_lt Bk Q numKVBlocks K V scale k hk idx]
      rfl
    · -- hK : InputAt s' kReg ...
      intro idx
      simpa [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · -- hV : InputAt s' vReg ...
      intro idx
      simpa [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- Boundary strided FA-1 forward correctness, raw form. Bundles
`fa1_forward_correct_strided_boundary_raw_of_step` with the proven
`fa1_step_strided_boundary` so callers no longer need to supply the
loop-step lemma explicitly. The output is observed only on in-range
Q rows (`s.pids 0 * M + idx.1.val < S_q`); out-of-range rows are
masked off by the kernel's store mask and outside this theorem's
guarantee. The conclusion is the raw streaming-accumulator ratio
`oPartial / lPartial` at `numKVBlocks`; the canonical-form bridge
to `attentionReal` over the logical `[S_k, D]` domain is left to
follow-up. -/
theorem fa1_forward_correct_strided_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathBoundary.lPartial Bk Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_boundary_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_boundary hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- Boundary strided forward correctness, stated against the canonical
`attentionReal` over the logical `[S_k, D]` KV domain (rather than the
raw streaming-accumulator ratio). Bridges `_raw` through
`FA1MathBoundary.streaming_eq_attentionReal`. The `0 < S_k` hypothesis
ensures the running normalizer is non-zero, allowing the m-shifted
algebra to cancel cleanly. -/
theorem fa1_forward_correct_strided_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale idx) := by
  intro idx hIdx
  rw [fa1_forward_correct_strided_boundary_raw hBk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj idx hIdx]
  congr 1
  have hL := FA1MathBoundary.lPartial_final_ne_zero hBk hSk Q numKVBlocks K
    scale hSkLe idx.1
  exact FA1MathBoundary.streaming_eq_attentionReal hBk Q hSkLe K V scale idx hL
theorem fa1_forward_correct_4D_boundary_slice
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
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
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal
                (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0))
                (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                scale idx) := by
  intro idx hIdxIn
  -- `hQIn`: in-bounds Q rows read the logical Q tensor.
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d hIn]
    exact h
  -- `hQOut`: out-of-bounds Q rows are zero by definition.
  have hQOut_inner : ∀ tileIdx : TileIndex [M, D],
      ¬ s.pids 0 * M + tileIdx.1.val < S_q →
      slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) tileIdx = 0 := by
    intro tileIdx hOut
    obtain ⟨i, d, _⟩ := tileIdx
    exact slice4DQRowsBoundary_of_not_lt M Q4D _ _ _ i d hOut
  -- Convert `hK4D` to inner-theorem form: K-side slice is `sliceBH K4D b h`,
  -- which has shape `[S_k, D]` directly (no `slice4DFlat` rewriting needed).
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Convert `hV4D` similarly.
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Output tile-local injectivity from `Offset.strided_inj`.
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
  exact fa1_forward_correct_strided_boundary hBk hSk hSkLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
    (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hIdxIn

theorem fa1_forward_correct_4D_boundary
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
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
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  rw [fa1_forward_correct_4D_boundary_slice hBk hSk hSkLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D hOValid idx hLt]
  congr 1
  -- Bridge: `attentionReal` over the boundary Q-row slice + `sliceBH` K/V
  -- equals `attentionReal4D` at the global `(b, h, qb*M + i, d)` index.
  -- For in-bounds rows, `slice4DQRowsBoundary` returns exactly the same
  -- value as `Q4D ∘ globalIndex`, so `attentionReal_row_eq` applies.
  obtain ⟨i, d, _⟩ := idx
  rw [attentionReal4D_slice]
  apply attentionReal_row_eq
  intro d'
  -- Goal: `slice4DQRowsBoundary M Q4D b h qb (i, d', ()) = sliceBH Q4D b h (⟨qb*M+i,_⟩, d', ())`
  rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d' hLt]
  rfl

/-- D-tail boundary FA-1 forward correctness in user-facing `attentionReal4D`
form. The theorem observes a block-width `[M, Bd]` output tile but only
asserts logical lanes `d < D`, so ordinary output stride validity over the
logical `[B,H,S_q,D]` tensor suffices for readback. -/
theorem fa1_forward_correct_4D_boundaryD
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
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
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
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
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
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
    show s.readMem kReg
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
    show s.readMem vReg
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
  rw [fa1_forward_correct_strided_boundaryD hBk hSk hSkLe hDLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
        (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hLt hDIdx]
  congr 1
  obtain ⟨i, d, u⟩ := idx
  cases u
  rw [attentionReal4D_slice]
  apply attentionReal_row_eq
  intro d'
  rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d' hLt]
  rfl

/-- D-tail causal-boundary FA-1 forward correctness in user-facing
`attentionReal4DCausal` form. The theorem observes a block-width `[M, Bd]`
output tile but only asserts logical lanes `d < D`. -/
theorem fa1_forward_correct_4D_causal_boundaryD
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
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
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
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
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
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
    show s.readMem kReg
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
    show s.readMem vReg
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
  rw [fa1_forward_correct_strided_causal_boundaryD hBk hSk hSkLe hDLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
        (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hLt hDIdx]
  congr 1
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hLtI : s.pids 0 * M + i.val < S_q := by
    simpa using hLt
  rw [attentionReal4DCausal_slice]
  unfold attentionRealCausalBlock attentionRealCausal
  simp [sliceBH, slice4DQRowsBoundary, hLtI]

/-- Causal-boundary 4D-aware corollary of
`fa1_forward_correct_strided_causal_boundary`. The result is still stated in
slice-local `attentionRealCausalBlock` form; `fa1_forward_correct_4D_causal_boundary`
below rewrites it to the user-facing 4D causal spec. -/
theorem fa1_forward_correct_4D_causal_boundary_slice
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
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
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M)
                (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0))
                (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                scale idx) := by
  intro idx hIdxIn
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
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
    show s.readMem kReg
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
    show s.readMem vReg
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
  exact fa1_forward_correct_strided_causal_boundary hBk hSk hSkLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
    (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hIdxIn

/-- Causal-boundary FA-1 forward correctness in user-facing
`attentionReal4DCausal` form. -/
theorem fa1_forward_correct_4D_causal_boundary
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
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
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  rw [fa1_forward_correct_4D_causal_boundary_slice hBk hSk hSkLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D hOValid idx hLt]
  congr 1
  obtain ⟨i, d, _⟩ := idx
  have hLtI : s.pids 0 * M + i.val < S_q := by
    simpa using hLt
  rw [attentionReal4DCausal_slice]
  unfold attentionRealCausalBlock attentionRealCausal
  simp [sliceBH, slice4DQRowsBoundary, hLtI]

theorem fa1_forward_correct_4D_boundary_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (layout.boundaryKernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Layout4D.boundaryKernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_boundary hBk hSk hSkLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt

/-- Causal-boundary FA-1 forward correctness over a bundled 4D layout. -/
theorem fa1_forward_correct_4D_causal_boundary_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (layout.causalBoundaryKernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Layout4D.causalBoundaryKernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_causal_boundary hBk hSk hSkLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt

/-- D-tail boundary FA-1 forward correctness over a bundled 4D layout.
The output tile has padded width `Bd`, but the conclusion is stated only for
logical lanes `d < D`. -/
theorem fa1_forward_correct_4D_boundaryD_layout
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (layout.boundaryKernelD qReg kReg vReg outReg M Bd Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffsetD s M Bd) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Layout4D.boundaryKernelD, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffsetD,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_boundaryD hBk hSk hSkLe hDLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt hDIdx

/-- D-tail causal-boundary FA-1 forward correctness over a bundled 4D
layout. -/
theorem fa1_forward_correct_4D_causal_boundaryD_layout
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (layout.causalBoundaryKernelD qReg kReg vReg outReg M Bd Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffsetD s M Bd) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Layout4D.causalBoundaryKernelD, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffsetD,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_causal_boundaryD hBk hSk hSkLe hDLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt hDIdx

theorem fa1_forward_correct_4D_boundary_views
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (views.boundaryKernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Views4D.boundaryKernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_boundary_layout hBk hSk hSkLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt

/-- Causal-boundary FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_causal_boundary_views
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (views.causalBoundaryKernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Views4D.causalBoundaryKernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_causal_boundary_layout hBk hSk hSkLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt

/-- D-tail boundary FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_boundaryD_views
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
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Views4D.boundaryKernelD, FA1Views4D.outBlockOffsetD,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_boundaryD_layout hBk hSk hSkLe hDLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt hDIdx

/-- D-tail causal-boundary FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_causal_boundaryD_views
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
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Views4D.causalBoundaryKernelD, FA1Views4D.outBlockOffsetD,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_causal_boundaryD_layout hBk hSk hSkLe hDLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt hDIdx


end VeriTile.Examples
