/-
VeriTile.Examples.FlashAttention1.ScoreVariants.Loop

Generic operational loop invariant proofs for score-variant FA-1 kernels.
-/

import VeriTile.Examples.FlashAttention1.ScoreVariants.Block

namespace VeriTile.Examples

open VeriTile.Triton

namespace FA1Score
/-! ## Generic score loop invariant

`P_fa1_score` is the operational invariant shape needed by score-transform
FA-1 kernels. It mirrors `P_fa1`, but the running accumulator slots are driven
by an explicit `score` function and `visible` predicate instead of being tied
to bare scaled dot-product scores.
-/

def fa1ScorePreLoop (qReg : RegionName) (M D : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0)
  , Stmt.assign .nat [M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid")
          (Op.constNat M))
        (Op.arange M))
  , Stmt.assign .nat [D] "offs_d" (Op.arange D)
  , Stmt.assign .nat [M, D] "q_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [M, D] "q"
      (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M] "m_i"
      (Op.full [M] Op.negInf)
  , Stmt.assign .real [M] "l_i"
      (Op.full [M] (Op.const 0))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.full [M, D] (Op.const 0))
  ]

def fa1ScorePostLoop (outReg : RegionName) (M D : Nat) : List Stmt :=
  [ Stmt.assign .real [M, D] "out"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, D] "o_acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l_i")))
  , Stmt.assign .nat [M, D] "o_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.store .real [M, D] (MemAccess.region outReg (Op.ref .nat [M, D] "o_ptrs")) (Op.ref .real [M, D] "out") MaskOpt.none
  ]

def fa1ScoreLoopBodySoftcap (kReg vReg : RegionName)
    (M D Bk : Nat) (scale softcap : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "raw")
            (Op.const softcap))))
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

def fa1ScoreLoopBodyAlibi (kReg vReg : RegionName)
    (M D Bk : Nat) (scale slope : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
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

def fa1ScoreLoopBodySlidingWindow (kReg vReg : RegionName)
    (M D Bk window : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "raw")
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

def fa1ScoreLoopBodyAlibiSlidingSoftcap (kReg vReg : RegionName)
    (M D Bk window : Nat) (scale slope softcap : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "biased"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
  , Stmt.assign .real [M, Bk] "capped"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "biased")
            (Op.const softcap))))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "capped")
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

def fa1ScoreLoopLoadBlock (kReg vReg : RegionName)
    (D Bk : Nat) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  ]

def fa1ScoreLoopScoreSoftcap
    (M D Bk : Nat) (scale softcap : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "raw")
            (Op.const softcap))))
  ]

def fa1ScoreLoopScoreAlibi
    (M D Bk : Nat) (scale slope : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
  ]

def fa1ScoreLoopScoreSlidingWindow
    (M D Bk window : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "raw")
        (Op.broadcast Op.negInf [M, Bk]))
  ]

def fa1ScoreLoopScoreAlibiSlidingSoftcap
    (M D Bk window : Nat) (scale slope softcap : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "biased"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
  , Stmt.assign .real [M, Bk] "capped"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "biased")
            (Op.const softcap))))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "capped")
        (Op.broadcast Op.negInf [M, Bk]))
  ]

def fa1ScoreLoopTail (M D Bk : Nat) : List Stmt :=
  [ Stmt.assign .real [M] "m_block"
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

@[simp] theorem fa1ScoreLoopBodySoftcap_parts_eq
    (kReg vReg : RegionName) (M D Bk : Nat) (scale softcap : ℝ) :
    fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreSoftcap M D Bk scale softcap ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ScoreLoopBodyAlibi_parts_eq
    (kReg vReg : RegionName) (M D Bk : Nat) (scale slope : ℝ) :
    fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreAlibi M D Bk scale slope ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ScoreLoopBodySlidingWindow_parts_eq
    (kReg vReg : RegionName) (M D Bk window : Nat) (scale : ℝ) :
    fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreSlidingWindow M D Bk window scale ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ScoreLoopBodyAlibiSlidingSoftcap_parts_eq
    (kReg vReg : RegionName) (M D Bk window : Nat)
    (scale slope softcap : ℝ) :
    fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window
        scale slope softcap =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreAlibiSlidingSoftcap M D Bk window scale slope softcap ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ForwardKernelSoftcap_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale softcap : ℝ) :
    (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk numKVBlocks
        scale softcap).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

@[simp] theorem fa1ForwardKernelAlibi_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale slope : ℝ) :
    (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk numKVBlocks
        scale slope).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

@[simp] theorem fa1ForwardKernelSlidingWindow_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks window : Nat) (scale : ℝ) :
    (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk numKVBlocks
        window scale).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

@[simp] theorem fa1ForwardKernelAlibiSlidingSoftcap_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks window : Nat) (scale slope softcap : ℝ) :
    (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
        numKVBlocks window scale slope softcap).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window
          scale slope softcap)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

def P_fa1_score
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.regs .nat [] "pid" = some (Tile.scalar origPid) ∧
  s.pid = origPid ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => origPid * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        mScoreOnline visible score k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        lScoreOnline visible score k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        oScoreOnline visible score V k idx) ∧
  InputAt s qReg
      (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) Q ∧
  InputAt s kReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) K ∧
  InputAt s vReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) V

/-- Block-indexed wrapper around `P_fa1_score`.

The generic online recurrence is indexed by the number of logical key
positions already consumed. The executable FA-1 loop is indexed by KV blocks,
so after `k` loop iterations the recurrence has consumed `Bk * k` key lanes.
-/
def P_fa1_score_blocks
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (s : BlockState) : Prop :=
  P_fa1_score qReg kReg vReg origPid Q K V visible score (Bk * k) s

theorem P_fa1_score_blocks_zero
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (s : BlockState) :
    P_fa1_score_blocks qReg kReg vReg origPid Q K V visible score 0 s =
      P_fa1_score qReg kReg vReg origPid Q K V visible score 0 s := by
  simp [P_fa1_score_blocks]

theorem P_fa1_score_blocks_final
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (s : BlockState) :
    P_fa1_score_blocks qReg kReg vReg origPid Q K V visible score numKVBlocks s =
      P_fa1_score qReg kReg vReg origPid Q K V visible score (Bk * numKVBlocks) s := by
  rfl

def P_fa1_score_blockrec
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (s : BlockState) : Prop :=
  s.regs .nat [] "pid" = some (Tile.scalar origPid) ∧
  s.pid = origPid ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => origPid * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        lScoreBlockPartial visible score k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        oScoreBlockPartial visible score V k idx) ∧
  InputAt s qReg
      (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) Q ∧
  InputAt s kReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K ∧
  InputAt s vReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V

def P_fa1_score_blockrec_loaded
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (s : BlockState) : Prop :=
  P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score k s ∧
  s.regs .nat [Bk] "offs_n" = some
      (Tile.vec fun j : Fin Bk => k * Bk + j.val) ∧
  s.regs .nat [Bk, D] "k_ptrs" = some
      (⟨fun idx : TileIndex [Bk, D] =>
        (k * Bk + idx.1.val) * D + idx.2.1.val⟩ : Tile .nat [Bk, D]) ∧
  s.regs .nat [Bk, D] "v_ptrs" = some
      (⟨fun idx : TileIndex [Bk, D] =>
        (k * Bk + idx.1.val) * D + idx.2.1.val⟩ : Tile .nat [Bk, D]) ∧
  s.regs .real [Bk, D] "k" =
      some (valueBlock K k (Nat.succ_le_iff.mpr hk)) ∧
  s.regs .real [Bk, D] "v" =
      some (valueBlock V k (Nat.succ_le_iff.mpr hk))

def P_fa1_score_blockrec_scored
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (s : BlockState) : Prop :=
  P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V visible score
    k hk s ∧
  s.regs .real [M, Bk] "scores" =
    some (scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk))

theorem fa1_score_loop_loadBlock_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score k s) :
    ∃ sLoad,
      stepStmts (fa1ScoreLoopLoadBlock kReg vReg D Bk)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some sLoad ∧
      P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V visible score
        k hk sLoad := by
  rcases hP with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let ptrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => (k * Bk + idx.1.val) * D + idx.2.1.val⟩
  let kTile : Tile .real [Bk, D] := valueBlock K k (Nat.succ_le_iff.mpr hk)
  let vTile : Tile .real [Bk, D] := valueBlock V k (Nat.succ_le_iff.mpr hk)
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] ptrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] ptrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let sLoad := s4.setReg "v" .real [Bk, D] vTile
  refine ⟨sLoad, ?_, ?_⟩
  · simp [fa1ScoreLoopLoadBlock, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
      NumericDType.mul, Option.bind, hoffs_d,
      offsN, ptrs, kTile, vTile, s0, s1, s2, s3, s4, sLoad]
    rw [fa1_score_block_mem_load_tile_eq kReg s K hK k hk]
    rw [fa1_score_block_mem_load_tile_eq vReg s V hV k hk]
    simp [Tile.vec]
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · simp [sLoad, s4, s3, s2, s1, s0, hpidReg]
      · simpa [sLoad, s4, s3, s2, s1, s0] using hpid
      · simp [sLoad, s4, s3, s2, s1, s0, hoffs_m]
      · simp [sLoad, s4, s3, s2, s1, s0, hoffs_d]
      · simp [sLoad, s4, s3, s2, s1, s0, hq]
      · simp [sLoad, s4, s3, s2, s1, s0, hm]
      · simp [sLoad, s4, s3, s2, s1, s0, hl]
      · simp [sLoad, s4, s3, s2, s1, s0, ho]
      · intro idx
        simpa [sLoad, s4, s3, s2, s1, s0] using hQ idx
      · intro idx
        simpa [sLoad, s4, s3, s2, s1, s0] using hK idx
      · intro idx
        simpa [sLoad, s4, s3, s2, s1, s0] using hV idx
    · simp [sLoad, s4, s3, s2, s1, offsN]
    · simp [sLoad, s4, s3, s2, ptrs]
    · simp [sLoad, s4, s3, ptrs]
    · simp [sLoad, s4, kTile]
    · simp [sLoad, vTile]

theorem fa1_score_loop_scoreSoftcap_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V allVisible
      (softcapDotScore softcap Q K scale) k hk sLoad) :
    ∃ sScore,
      stepStmts (fa1ScoreLoopScoreSoftcap M D Bk scale softcap) sLoad =
        some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V allVisible
        (softcapDotScore softcap Q K scale) k hk sScore := by
  have hLoaded := hP
  rcases hP with
    ⟨hCore, _hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, _hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
      (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sScore := sRaw.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreSoftcap, stepStmts, stepStmt, evalOp, Option.bind,
      hq, hkTileReg, raw, scores, sRaw, sScore]
    simp only [valueBlock]
    change (sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "scores" .real [M, Bk]
          (Tile.bop NumericDType.real.mul Broadcast.scalarL
            (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
            (Tile.uop WithBot.realTanh
              (Tile.bop NumericDType.real.div Broadcast.scalarR
                (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
                (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))) =
        (sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [softcapScoreBlock_numeric_tile_eq softcap Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sRaw, hpidReg]
        · simpa [sScore, sRaw] using hpid
        · simp [sScore, sRaw, hoffs_m]
        · simp [sScore, sRaw, hoffs_d]
        · simp [sScore, sRaw, hq]
        · simp [sScore, sRaw, hm]
        · simp [sScore, sRaw, hl]
        · simp [sScore, sRaw, ho]
        · intro idx
          simpa [sScore, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sRaw] using hK idx
        · intro idx
          simpa [sScore, sRaw] using hV idx
      · simp [sScore, sRaw, hoffs_n]
      · simp [sScore, sRaw, hk_ptrs]
      · simp [sScore, sRaw, hv_ptrs]
      · simp [sScore, sRaw, hkReg]
      · simp [sScore, sRaw, hvReg]
    · simp [sScore, scores]

theorem fa1_score_loop_scoreAlibi_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart : Nat)
    (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V allVisible
      (alibiScore qStart slope Q K scale) k hk sLoad) :
    ∃ sScore,
      stepStmts (fa1ScoreLoopScoreAlibi M D Bk scale slope) sLoad =
        some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V allVisible
        (alibiScore qStart slope Q K scale) k hk sScore := by
  subst qStart
  have hLoaded := hP
  rcases hP with
    ⟨hCore, hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let delta : Tile .real [M, Bk] :=
    Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => origPid * M + i.val))))
  let dist : Tile .real [M, Bk] :=
    distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane allVisible (alibiScore (origPid * M) slope Q K scale) k
      (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sDelta := sRaw.setReg "delta" .real [M, Bk] delta
  let sDist := sDelta.setReg "dist" .real [M, Bk] dist
  let sScore := sDist.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreAlibi, stepStmts, stepStmt, evalOp, Option.bind,
      hq, hkTileReg, hoffs_n, hoffs_m, raw, delta, dist, scores, sRaw, sDelta, sDist,
      sScore]
    simp only [valueBlock]
    change (((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "scores" .real [M, Bk]
          (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (Tile.bop NumericDType.real.sub Broadcast.scalarL
              (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
              (Tile.bop NumericDType.real.mul Broadcast.scalarL
                (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)))) =
        (((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk] (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane allVisible (alibiScore (origPid * M) slope Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [distanceBlockTileNumeric_eq (M := M) (Bk := Bk) (origPid * M) k]
    rw [alibiScoreBlock_numeric_tile_eq (origPid * M) slope Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sDist, sDelta, sRaw, hpidReg]
        · simpa [sScore, sDist, sDelta, sRaw] using hpid
        · simp [sScore, sDist, sDelta, sRaw, hoffs_m]
        · simp [sScore, sDist, sDelta, sRaw, hoffs_d]
        · simp [sScore, sDist, sDelta, sRaw, hq]
        · simp [sScore, sDist, sDelta, sRaw, hm]
        · simp [sScore, sDist, sDelta, sRaw, hl]
        · simp [sScore, sDist, sDelta, sRaw, ho]
        · intro idx
          simpa [sScore, sDist, sDelta, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sDist, sDelta, sRaw] using hK idx
        · intro idx
          simpa [sScore, sDist, sDelta, sRaw] using hV idx
      · simp [sScore, sDist, sDelta, sRaw, hoffs_n]
      · simp [sScore, sDist, sDelta, sRaw, hk_ptrs]
      · simp [sScore, sDist, sDelta, sRaw, hv_ptrs]
      · simp [sScore, sDist, sDelta, sRaw, hkReg]
      · simp [sScore, sDist, sDelta, sRaw, hvReg]
    · simp [sScore, scores]

theorem fa1_score_loop_scoreSlidingWindow_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k hk sLoad) :
    ∃ sScore,
      stepStmts (fa1ScoreLoopScoreSlidingWindow M D Bk window scale) sLoad =
        some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V
        (slidingVisible window qStart) (dotScore Q K scale) k hk sScore := by
  subst qStart
  have hLoaded := hP
  rcases hP with
    ⟨hCore, hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let delta : Tile .real [M, Bk] :=
    Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => origPid * M + i.val))))
  let dist : Tile .real [M, Bk] :=
    distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k
  let vis : Tile .bool [M, Bk] :=
    slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane (slidingVisible window (origPid * M)) (dotScore Q K scale) k
      (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sDelta := sRaw.setReg "delta" .real [M, Bk] delta
  let sDist := sDelta.setReg "dist" .real [M, Bk] dist
  let sVis := sDist.setReg "visible" .bool [M, Bk] vis
  let sScore := sVis.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreSlidingWindow, stepStmts, stepStmt, evalOp,
      Option.bind, hq, hkTileReg, hoffs_n, hoffs_m, raw, delta, dist, vis,
      scores, sRaw, sDelta, sDist, sVis, sScore]
    simp only [valueBlock]
    change ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (Tile.cop ComparableDType.real.lt Broadcast.scalarR
            (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
            (Tile.scalar (((window : ℝ) : WithBot ℝ))))).setReg
        "scores" .real [M, Bk]
          (Tile.select
            (Tile.cop ComparableDType.real.lt Broadcast.scalarR
              (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
              (Tile.scalar (((window : ℝ) : WithBot ℝ))))
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane (slidingVisible window (origPid * M)) (dotScore Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [distanceBlockTileNumeric_eq (M := M) (Bk := Bk) (origPid * M) k]
    change ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)).setReg
        "scores" .real [M, Bk]
          (Tile.select
            (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane (slidingVisible window (origPid * M)) (dotScore Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [slidingScoreBlock_numeric_tile_eq (origPid * M) window Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sVis, sDist, sDelta, sRaw, hpidReg]
        · simpa [sScore, sVis, sDist, sDelta, sRaw] using hpid
        · simp [sScore, sVis, sDist, sDelta, sRaw, hoffs_m]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hoffs_d]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hq]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hm]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hl]
        · simp [sScore, sVis, sDist, sDelta, sRaw, ho]
        · intro idx
          simpa [sScore, sVis, sDist, sDelta, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sVis, sDist, sDelta, sRaw] using hK idx
        · intro idx
          simpa [sScore, sVis, sDist, sDelta, sRaw] using hV idx
      · simp [sScore, sVis, sDist, sDelta, sRaw, hoffs_n]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hk_ptrs]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hv_ptrs]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hkReg]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hvReg]
    · simp [sScore, scores]

theorem fa1_score_loop_scoreAlibiSlidingSoftcap_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k hk sLoad) :
    ∃ sScore,
      stepStmts
        (fa1ScoreLoopScoreAlibiSlidingSoftcap M D Bk window scale slope softcap)
        sLoad = some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V
        (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k hk sScore := by
  subst qStart
  have hLoaded := hP
  rcases hP with
    ⟨hCore, hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let delta : Tile .real [M, Bk] :=
    Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => origPid * M + i.val))))
  let dist : Tile .real [M, Bk] :=
    distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k
  let biased : Tile .real [M, Bk] :=
    alibiScoreBlockTile (origPid * M) slope Q K scale k (Nat.succ_le_iff.mpr hk)
  let capped : Tile .real [M, Bk] :=
    softcapScoreTile softcap biased
  let vis : Tile .bool [M, Bk] :=
    slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane (slidingVisible window (origPid * M))
      (fun i j => softcapScore softcap (alibiScore (origPid * M) slope Q K scale i j))
      k (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sDelta := sRaw.setReg "delta" .real [M, Bk] delta
  let sDist := sDelta.setReg "dist" .real [M, Bk] dist
  let sBiased := sDist.setReg "biased" .real [M, Bk] biased
  let sCapped := sBiased.setReg "capped" .real [M, Bk] capped
  let sVis := sCapped.setReg "visible" .bool [M, Bk] vis
  let sScore := sVis.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreAlibiSlidingSoftcap, stepStmts, stepStmt, evalOp,
      Option.bind, hq, hkTileReg, hoffs_n, hoffs_m, raw, delta, dist, biased,
      capped, vis, scores, sRaw, sDelta, sDist, sBiased, sCapped, sVis, sScore]
    simp only [valueBlock]
    change ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk]
          (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (Tile.bop NumericDType.real.sub Broadcast.scalarL
              (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
              (Tile.bop NumericDType.real.mul Broadcast.scalarL
                (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k))))).setReg
        "capped" .real [M, Bk]
          (Tile.bop NumericDType.real.mul Broadcast.scalarL
            (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
            (Tile.uop WithBot.realTanh
              (Tile.bop NumericDType.real.div Broadcast.scalarR
                (Tile.bop NumericDType.real.add
                  (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
                  (Tile.bop NumericDType.real.sub Broadcast.scalarL
                    (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
                    (Tile.bop NumericDType.real.mul Broadcast.scalarL
                      (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                      (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k))))
                (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))))).setReg
        "visible" .bool [M, Bk]
          (Tile.cop ComparableDType.real.lt Broadcast.scalarR
            (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
            (Tile.scalar (((window : ℝ) : WithBot ℝ))))).setReg
        "scores" .real [M, Bk]
          (Tile.select
            (Tile.cop ComparableDType.real.lt Broadcast.scalarR
              (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
              (Tile.scalar (((window : ℝ) : WithBot ℝ))))
            (Tile.bop NumericDType.real.mul Broadcast.scalarL
              (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
              (Tile.uop WithBot.realTanh
                (Tile.bop NumericDType.real.div Broadcast.scalarR
                  (Tile.bop NumericDType.real.add
                    (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
                    (Tile.bop NumericDType.real.sub Broadcast.scalarL
                      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
                      (Tile.bop NumericDType.real.mul Broadcast.scalarL
                        (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                        (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k))))
                  (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))))
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk] biased).setReg
        "capped" .real [M, Bk] capped).setReg
        "visible" .bool [M, Bk] vis).setReg
        "scores" .real [M, Bk] scores
    rw [distanceBlockTileNumeric_eq (M := M) (Bk := Bk) (origPid * M) k]
    rw [alibiScoreBlock_numeric_eq_tile (origPid * M) slope Q K scale k hk]
    rw [softcapScoreTileNumeric_eq softcap
      (alibiScoreBlockTile (origPid * M) slope Q K scale k (Nat.succ_le_iff.mpr hk))]
    change ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk] biased).setReg
        "capped" .real [M, Bk] capped).setReg
        "visible" .bool [M, Bk] vis).setReg
        "scores" .real [M, Bk]
          (Tile.select vis capped
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk] biased).setReg
        "capped" .real [M, Bk] capped).setReg
        "visible" .bool [M, Bk] vis).setReg
        "scores" .real [M, Bk] scores
    rw [show Tile.select vis capped
        (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk]) =
        scores by
      simp [vis, capped, biased, scores]
      exact alibiSlidingSoftcapScoreBlock_numeric_tile_eq
        (origPid * M) window slope softcap Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hpidReg]
        · simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hpid
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hoffs_m]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hoffs_d]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hq]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hm]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hl]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, ho]
        · intro idx
          simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hK idx
        · intro idx
          simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hV idx
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hoffs_n]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hk_ptrs]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hv_ptrs]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hkReg]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hvReg]
    · simp [sScore, scores]

set_option maxHeartbeats 800000 in
theorem fa1_score_loop_tail_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (sScore : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V visible score
      k hk sScore) :
    ∃ s',
      stepStmts (fa1ScoreLoopTail M D Bk) sScore = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score
        (k + 1) s' := by
  rcases hP with
    ⟨hLoaded, hscores⟩
  rcases hLoaded with
    ⟨hCore, _hoffs_n, _hk_ptrs, _hv_ptrs, _hkReg, hvReg⟩
  rcases hCore with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let mOld : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      mScoreBlockPartial visible score k idx.1⟩
  let lOld : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      lScoreBlockPartial visible score k idx.1
  let oOld : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      oScoreBlockPartial visible score V k idx
  let scoresTile : Tile .real [M, Bk] :=
    scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk)
  let mBlockExec : Tile .real [M] :=
    ⟨fun outIdx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j => scoresTile.data (outIdx.1, j, outIdx.2))⟩
  let mNewExec : Tile .real [M] :=
    Tile.bop max (Broadcast.consSame Broadcast.nil) mOld mBlockExec
  let alphaExec : Tile .real [M] :=
    Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
        mOld mNewExec)
  let pExec : Tile .real [M, Bk] :=
    Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        scoresTile
        (Tile.expandDim ⟨1, by simp⟩ mNewExec))
  let lNewExec : Tile .real [M] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        alphaExec lOld)
      (Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false pExec)
  let oNewExec : Tile .real [M, D] :=
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩ alphaExec)
        oOld)
      (Tile.dot [] (M := M) (K := Bk) (N := D) pExec
        (valueBlock V k (Nat.succ_le_iff.mpr hk)))
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => scoreLane visible score idx.1
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      mScoreBlockPartial visible score (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      alphaScoreBlockPartial visible score k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score idx.1
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
          (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      lScoreBlockPartial visible score (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      oScoreBlockPartial visible score V (k + 1) idx
  let s1 := sScore.setReg "m_block" .real [M] mBlockExec
  let s2 := s1.setReg "m_new" .real [M] mNewExec
  let s3 := s2.setReg "alpha" .real [M] alphaExec
  let s4 := s3.setReg "p" .real [M, Bk] pExec
  let s5 := s4.setReg "l_new" .real [M] lNewExec
  let s6 := s5.setReg "o_acc" .real [M, D] oNewExec
  let s7 := s6.setReg "m_i" .real [M] mNewExec
  let s' := s7.setReg "l_i" .real [M] lNewExec
  have hmBlockExec : mBlockExec = mBlock := by
    exact score_block_mBlock_sup'_tile_eq
      (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) visible score k hk
  have hmNewExec : mNewExec = mNew := by
    change Tile.bop max (Broadcast.consSame Broadcast.nil) mOld mBlockExec = mNew
    rw [hmBlockExec]
    change Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
      mBlock = mNew
    rw [score_block_mNew_tile_eq visible score k hk]
  have halphaExec : alphaExec = alpha := by
    change Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
        mOld mNewExec) = alpha
    rw [hmNewExec]
    ext idx
    simp [mOld, mNew, alpha, Tile.uop, Tile.bop,
      Tile.ofReal, alphaScoreBlockPartial]
    exact FA1MathBoundary.realExp_eq_some_unbotD _
  have hpExec : pExec = p := by
    change Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        scoresTile
        (Tile.expandDim ⟨1, by simp⟩ mNewExec)) = p
    rw [hmNewExec]
    ext idx
    simp [scoresTile, mNew, p, Tile.uop, Tile.bop,
      Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
    exact FA1MathBoundary.realExp_eq_some_unbotD _
  have hpRowSumExec :
      Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false pExec =
        Tile.ofReal (fun idx : TileIndex [M] =>
          Finset.univ.sum (fun j : Fin Bk =>
            (WithBot.realExp
              (WithBot.realSub
                (scoreLane visible score idx.1
                  (scoreBlockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j))
                (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)) := by
    rw [hpExec]
    exact score_block_p_rowSum_eq visible score k hk
  have hlNewExec : lNewExec = lNew := by
    change Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        alphaExec lOld)
      (Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false pExec) = lNew
    rw [halphaExec, hpRowSumExec]
    change Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        alpha lOld)
      (Tile.ofReal (fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j))
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0))) = lNew
    rw [score_block_lNew_tile_eq visible score k hk]
  have hpvDotExec :
      Tile.dot [] (M := M) (K := Bk) (N := D) pExec
        (valueBlock V k (Nat.succ_le_iff.mpr hk)) =
        Tile.ofReal (fun idx : TileIndex [M, D] =>
          Finset.univ.sum (fun jLocal : Fin Bk =>
            let j := scoreBlockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) jLocal
            (WithBot.realExp
              (WithBot.realSub
                (scoreLane visible score idx.1 j)
                (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
              V (j, idx.2.1, PUnit.unit))) := by
    rw [hpExec]
    exact score_block_pv_dot_eq visible score V k hk
  have hoNewExec : oNewExec = oNew := by
    change Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩ alphaExec) oOld)
      (Tile.dot [] (M := M) (K := Bk) (N := D) pExec
        (valueBlock V k (Nat.succ_le_iff.mpr hk))) = oNew
    rw [halphaExec, hpvDotExec]
    change Tile.bop WithBot.realAdd
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul
        (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩ alpha) oOld)
      (Tile.ofReal (fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := scoreBlockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)))) = oNew
    rw [score_block_oAcc_tile_eq visible score V k hk]
  refine ⟨s', ?_, ?_⟩
  · simp [fa1ScoreLoopTail, stepStmts, stepStmt, evalOp, Option.bind,
      Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, hscores, hm, hl, ho, hvReg,
      mOld, lOld, oOld, scoresTile, mBlockExec, mNewExec,
      alphaExec, pExec, lNewExec, oNewExec, s1, s2, s3, s4, s5,
      s6, s7, s']
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hpidReg]
    · simpa [s', s7, s6, s5, s4, s3, s2, s1] using hpid
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hoffs_m]
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hoffs_d]
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hq]
    · simp [s', s7, hmNewExec, mNew]
    · simp [s', hlNewExec, lNew]
    · simp [s', s7, s6, hoNewExec, oNew]
    · intro idx
      simpa [s', s7, s6, s5, s4, s3, s2, s1] using hQ idx
    · intro idx
      simpa [s', s7, s6, s5, s4, s3, s2, s1] using hK idx
    · intro idx
      simpa [s', s7, s6, s5, s4, s3, s2, s1] using hV idx

theorem fa1_score_loop_stepSoftcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
      (softcapDotScore softcap Q K scale) k s) :
    ∃ s',
      stepStmts (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
        (softcapDotScore softcap Q K scale) (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      allVisible (softcapDotScore softcap Q K scale) k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreSoftcap_correct qReg kReg vReg origPid Q K V
      scale softcap k sLoad hk hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      allVisible (softcapDotScore softcap Q K scale) k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodySoftcap_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_loop_stepAlibi_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid qStart : Nat)
    (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
      (alibiScore qStart slope Q K scale) k s) :
    ∃ s',
      stepStmts (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
        (alibiScore qStart slope Q K scale) (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      allVisible (alibiScore qStart slope Q K scale) k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreAlibi_correct qReg kReg vReg origPid qStart slope Q K V
      scale k sLoad hk hqStart hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      allVisible (alibiScore qStart slope Q K scale) k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodyAlibi_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_loop_stepSlidingWindow_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k s) :
    ∃ s',
      stepStmts (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V
        (slidingVisible window qStart) (dotScore Q K scale) (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreSlidingWindow_correct qReg kReg vReg origPid qStart
      window Q K V scale k sLoad hk hqStart hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodySlidingWindow_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_loop_stepAlibiSlidingSoftcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k s) :
    ∃ s',
      stepStmts
          (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window scale
            slope softcap)
          (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V
        (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreAlibiSlidingSoftcap_correct qReg kReg vReg origPid
      qStart window slope softcap Q K V scale k sLoad hk hqStart hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodyAlibiSlidingSoftcap_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_preLoop_correct
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1ScorePreLoop qReg M D) s = some s0 ∧
      P_fa1_score qReg kReg vReg s.pid Q K V visible score 0 s0 := by
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => (s.pid * M + idx.1.val) * D + idx.2.1.val⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg ((s.pid * M + idx.1.val) * D + idx.2.1.val))⟩
  let s0 :=
    ((((((((s.setReg "pid" .nat [] (Tile.scalar s.pid))
      ).setReg "offs_m" .nat [M] (Tile.vec fun i : Fin M => s.pid * M + i.val)
      ).setReg "offs_d" .nat [D] (Tile.vec fun d : Fin D => d.val)
      ).setReg "q_ptrs" .nat [M, D] qPtrs
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, Tile.ofReal]
    rw [show (s.pid * M + idx.1.val) * D + idx.2.1.val =
        Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D idx by
          simp [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
            Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1ScorePreLoop, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qLoaded, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, mScoreOnline]
    · simp [s0, lScoreOnline, Tile.ofReal]
    · simp [s0, oScoreOnline, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

theorem fa1_score_preLoop_correct_blocks
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1ScorePreLoop qReg M D) s = some s0 ∧
      P_fa1_score_blocks qReg kReg vReg s.pid Q K V visible score 0 s0 := by
  rcases fa1_score_preLoop_correct qReg kReg vReg Q K V visible score s hQ hK hV with
    ⟨s0, hStep, hP⟩
  exact ⟨s0, hStep, by simpa [P_fa1_score_blocks] using hP⟩

theorem fa1_score_blockrec_preLoop_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1ScorePreLoop qReg M D) s = some s0 ∧
      P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score 0 s0 := by
  rcases fa1_score_preLoop_correct qReg kReg vReg Q K V visible score s hQ hK hV with
    ⟨s0, hStep, hP⟩
  rcases hP with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, _hm, _hl, _ho, hQ', hK', hV'⟩
  refine ⟨s0, hStep, ?_⟩
  refine ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, ?_, ?_, ?_, hQ', hK', hV'⟩
  · simpa [mScoreBlockPartial]
  · simpa [lScoreBlockPartial, Tile.ofReal]
  · simpa [oScoreBlockPartial, Tile.ofReal]

theorem fa1_score_postLoop_correct
    {M D S : Nat}
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (sLoop : BlockState)
    (hP : P_fa1_score qReg kReg vReg origPid Q K V visible score S sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1ScorePostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some (attentionRealMaskedScore visible score V idx) := by
  intro idx
  rcases hP with
    ⟨_hpidReg, _hpid, hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj :
      Function.Injective
        (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) :=
    Offset.rowMajor2D_inj (base := origPid * M * D) (rowStride := D) (le_refl D)
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] => (origPid * M + i.1.val) * D + i.2.1.val) := by
    intro a b h
    apply h_inj
    simpa [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
      Nat.add_assoc] using h
  simp [observeTileAt, fa1ScorePostLoop, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Offset.rowMajor2D, Offset.strided, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show origPid * M * D + idx.1.val * D + idx.2.1.val =
      (origPid * M + idx.1.val) * D + idx.2.1.val by
        rw [Nat.add_mul]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]
  simp [oScoreOnline_div_lScoreOnline_eq_attentionRealMaskedScore visible score V idx]

theorem fa1_score_postLoop_correct_blocks
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (sLoop : BlockState)
    (hP : P_fa1_score_blocks qReg kReg vReg origPid Q K V visible score
      numKVBlocks sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1ScorePostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some (attentionRealMaskedScore visible score V idx) := by
  intro idx
  exact fa1_score_postLoop_correct qReg kReg vReg outReg origPid Q K V
    visible score sLoop (by simpa [P_fa1_score_blocks] using hP) idx

theorem fa1_score_blockrec_postLoop_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (sLoop : BlockState)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score
      numKVBlocks sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1ScorePostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some
          (oScoreBlockPartial visible score V numKVBlocks idx /
            lScoreBlockPartial visible score numKVBlocks idx.1) := by
  intro idx
  rcases hP with
    ⟨_hpidReg, _hpid, hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj :
      Function.Injective
        (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) :=
    Offset.rowMajor2D_inj (base := origPid * M * D) (rowStride := D) (le_refl D)
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] => (origPid * M + i.1.val) * D + i.2.1.val) := by
    intro a b h
    apply h_inj
    simpa [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
      Nat.add_assoc] using h
  simp [observeTileAt, fa1ScorePostLoop, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Offset.rowMajor2D, Offset.strided, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show origPid * M * D + idx.1.val * D + idx.2.1.val =
      (origPid * M + idx.1.val) * D + idx.2.1.val by
        rw [Nat.add_mul]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]

end FA1Score

end VeriTile.Examples
