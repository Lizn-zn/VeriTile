/-
VeriTile.Examples.FlashAttention1.V0

FlashAttention-1 v0/full-tile proof surface.
-/

import VeriTile.Examples.FlashAttention1.Common

namespace VeriTile.Examples

open VeriTile.Triton

private def fa1PreLoop (qReg : RegionName) (M D : Nat) : List Stmt :=
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

/-- Body of the inner KV-block loop in `fa1ForwardKernel`, factored out so
the loop invariant proof can name the operational step directly. -/
private def fa1LoopBody (kReg vReg : RegionName)
    (M D Bk : Nat) (scale : ℝ) : List Stmt :=
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
  , Stmt.assign .real [M, Bk] "scores"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
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

/-- Initialization stage: the pre-loop statements establish `P_fa1 0`. -/
theorem fa1_preLoop_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1PreLoop qReg M D) s = some s0 ∧
      P_fa1 qReg kReg vReg s.pid Q K V scale 0 s0 := by
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
  · simp [fa1PreLoop, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qLoaded, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, FA1Math.mPartial]
    · simp [s0, FA1Math.lPartial, Tile.ofReal]
    · simp [s0, FA1Math.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- The three statements after the FA-1 KV-block loop: normalize the
accumulator, rebuild output pointers, and store the `[M, D]` tile. Factored
out so the readout proof can be checked independently of the loop proof. -/
private def fa1PostLoop (outReg : RegionName) (M D : Nat) : List Stmt :=
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

/-- Readout stage: once the loop invariant holds at `numKVBlocks`, the
post-loop normalization and store produce the ℝ-level attention spec. -/
theorem fa1_postLoop_correct
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1 qReg kReg vReg origPid Q K V scale numKVBlocks sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1PostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some (attentionReal Q K V scale idx) := by
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
  simp [observeTileAt, fa1PostLoop, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Offset.rowMajor2D, Offset.strided, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show origPid * M * D + idx.1.val * D + idx.2.1.val =
      (origPid * M + idx.1.val) * D + idx.2.1.val by
        rw [Nat.add_mul]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]
  simp [FA1Math.streaming_eq_attentionReal hBk Q numKVBlocks hNumKVBlocks K V scale idx
        (FA1Math.lPartial_final_ne_zero hBk Q numKVBlocks hNumKVBlocks K scale idx.1)]

/-- The DSL-expanded kernel body is exactly the factored operational shape:
pre-loop setup, one `forLoop`, then post-loop readout. -/
@[simp] theorem fa1ForwardKernel_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale).body =
      fa1PreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
      fa1PostLoop outReg M D := by
  rfl

/-! ### Strided helpers — pre-loop / loop body / post-loop

Mirror `fa1PreLoop` / `fa1LoopBody` / `fa1PostLoop` but factor out the
DSL expansion of `fa1ForwardKernelStrided`. The structural difference
versus the 2D versions is purely in the address-computation `Op` trees:
the strided kernel reads three `program_id` axes, computes four
`*_base_off` scalars from `pid_b * sQB + pid_h * sQH` etc., then builds
each pointer tile as `*_base_off + offs * stride_*S + offs_d * stride_*D`
instead of the row-major `offs * D + offs_d`. The streaming math
(scores / m_i / l_i / o_acc) is byte-identical — only the
load/store address trees differ. -/

/-- Pre-loop block of `fa1ForwardKernelStrided`: three `program_id`
reads, four base-offset assigns, two arange-based offset vectors,
strided Q-pointer tile, Q load, and accumulator initialization. -/
private def fa1PreLoopStrided (qReg : RegionName) (M D : Nat)
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
  , Stmt.assign .real [M, D] "q"
      (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M] "m_i"
      (Op.full [M] Op.negInf)
  , Stmt.assign .real [M] "l_i"
      (Op.full [M] (Op.const 0))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.full [M, D] (Op.const 0))
  ]

/-- KV-block loop body of `fa1ForwardKernelStrided`. Differs from
`fa1LoopBody` only in `k_ptrs` / `v_ptrs` (strided form using
`k_base_off` / `v_base_off`). -/
private def fa1LoopBodyStrided (kReg vReg : RegionName)
    (M D Bk : Nat) (sKN sKD sVN sVD : Nat) (scale : ℝ) : List Stmt :=
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
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "scores"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
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

/-- Causal KV-block loop body of `fa1ForwardKernelStridedCausal`.
Compared with `fa1LoopBodyStrided`, the raw scaled scores are first
written to `scores_raw`, then masked with global row/column indices:
`offs_m[:, None] >= offs_n[None, :]`. The resulting `scores` register
feeds the same online-softmax update as the non-causal body. -/
private def fa1LoopBodyStridedCausal (kReg vReg : RegionName)
    (M D Bk : Nat) (sKN sKD sVN sVD : Nat) (scale : ℝ) : List Stmt :=
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
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "scores_raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .bool [M, Bk] "causal"
      (Op.ge ComparableDType.nat
        (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "causal")
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

/-- Post-loop block of `fa1ForwardKernelStrided`: normalize `o_acc`,
build strided output pointers, store. -/
private def fa1PostLoopStrided (outReg : RegionName) (M D : Nat)
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
  , Stmt.store .real [M, D] (MemAccess.region outReg (Op.ref .nat [M, D] "o_ptrs")) (Op.ref .real [M, D] "out") MaskOpt.none
  ]

/-- The DSL-expanded body of `fa1ForwardKernelStrided` is the factored
operational shape: `fa1PreLoopStrided ++ [forLoop n fa1LoopBodyStrided]
++ fa1PostLoopStrided`. -/
@[simp] theorem fa1ForwardKernelStrided_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) :
    (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).body =
      fa1PreLoopStrided qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
      fa1PostLoopStrided outReg M D sOM sOD := by
  rfl

/-- The DSL-expanded body of `fa1ForwardKernelStridedCausal` is the
same factored operational shape as the non-causal strided kernel, but
with `fa1LoopBodyStridedCausal` in the inner `forLoop`. -/
@[simp] theorem fa1ForwardKernelStridedCausal_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (scale : ℝ) :
    (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).body =
      fa1PreLoopStrided qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
      fa1PostLoopStrided outReg M D sOM sOD := by
  rfl

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
  ext idx
  rw [Tile.ofReal_data]
  exact congrArg some (fa1_block_read region s X hX n hn idx.1 idx.2.1)

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
  ext idx
  rw [Tile.ofReal_data]
  exact congrArg some
    (fa1_block_read_strided region s base sN sD X hX n hn idx.1 idx.2.1)

/-- Boundary-masked strided KV block load. Valid local lanes read the logical
`[S_k, D]` tensor through `blockIndex?`; invalid padded lanes are exactly the
explicit `other = 0` value from the Triton load. -/
theorem fa1_preLoop_correct_strided
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg
      (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))⟩
  let s0 :=
    ((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
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
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, qBase, Tile.ofReal]
    rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStrided, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
    · simp [s0, FA1Math.mPartial]
    · simp [s0, FA1Math.lPartial, Tile.ofReal]
    · simp [s0, FA1Math.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- Causal strided initialization stage. The executable pre-loop is
identical to the non-causal strided pre-loop; only the target invariant
uses the causal streaming recurrence. At iteration `0`, both
recurrences initialize to `m_i = -inf`, `l_i = 0`, and `o_acc = 0`. -/
theorem fa1_preLoop_correct_strided_causal
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_causal qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg
      (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))⟩
  let s0 :=
    ((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
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
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, qBase, Tile.ofReal]
    rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStrided, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
    · simp [s0, FA1MathCausal.mPartial]
    · simp [s0, FA1MathCausal.lPartial, Tile.ofReal]
    · simp [s0, FA1MathCausal.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

set_option maxHeartbeats 800000 in
/-- One FA-1 KV-block iteration preserves the loop invariant. -/
theorem fa1_step
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (origPid k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1 qReg kReg vReg origPid Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBody kReg vReg M D Bk scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1 qReg kReg vReg origPid Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let ptrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => (k * Bk + idx.1.val) * D + idx.2.1.val⟩
  let kTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scores : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => ((FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)
          : ℝ) : WithBot ℝ))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.alphaPartial Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      Real.exp (FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0)
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.lPartial Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      FA1Math.oPartial Q numKVBlocks K V scale (k + 1) idx
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] ptrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] ptrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let s5 := s4.setReg "v" .real [Bk, D] vTile
  let s6 := s5.setReg "scores" .real [M, Bk] scores
  let s7 := s6.setReg "m_block" .real [M] mBlock
  let s8 := s7.setReg "m_new" .real [M] mNew
  let s9 := s8.setReg "alpha" .real [M] alpha
  let s10 := s9.setReg "p" .real [M, Bk] p
  let s11 := s10.setReg "l_new" .real [M] lNew
  let s12 := s11.setReg "o_acc" .real [M, D] oNew
  let s13 := s12.setReg "m_i" .real [M] mNew
  let s' := s13.setReg "l_i" .real [M] lNew
  apply Exists.intro
  constructor
  · simp [fa1LoopBody, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, Option.bind, hBk, hoffs_d, hq, hm, hl, ho]
    have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg ((k * Bk + j.val) * D + d.val) =
          K (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read kReg s K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg ((k * Bk + j.val) * D + d.val) =
          V (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read vReg s V hV k hk
    simp_rw [hKmem, hVmem]
    rfl
  · have hmNewData : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        (FA1Math.mPartial Bk Q numKVBlocks K scale k i) ⊔
          (some ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i H
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup' H
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataComm : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit)))))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [← hmNewData i]
      congr 2
      apply Finset.sup'_congr (H := by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) rfl
      intro j _
      rw [mul_comm]
    have hAlphaData : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        FA1Math.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPData : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j
      have hm : FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ :=
        FA1Math.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
          (Nat.succ_le_iff.mpr hk) i
      obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
      rw [← hm_eq]
      simp [FA1Math.scaledScore, mul_comm]
    have hAlphaDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      change WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i)
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i) z)) hM).trans
        (hAlphaData i)
    have hPDataOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      change WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b) z)) hM).trans
          (hPData i j)
    have hAlphaDataComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      exact hAlphaDataOf i H
    have hPDataComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      simpa [mul_comm] using hPDataOf i j H
    have hAlphaDataMaxComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxComm' : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm' : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataOf i H
    have hPDataMaxOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataOf i j H
    have max_eq_sup : ∀ (a b : WithBot ℝ), max a b = a ⊔ b := by
      intro a b
      rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hpidReg]
    · simp [hpid]
    · simp [hoffs_m]
    · simp [hoffs_d]
    · simp [hq]
    · simp
      ext idx
      exact hmNewData idx.1
    · rw [← FA1Math.block_lNew_tile_eq Q K scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
      congr 1
      · -- LHS: Option.map (· * lPartial k) (WithBot.realExp (Option.map₂ - mPartial(k) (max ...)))
        -- RHS: Option.map₂ * (some alphaPartial) (some lPartial k)
        rw [show (Option.map₂ (· * ·) (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.lPartial Q numKVBlocks K scale k idx.1)) :
              WithBot ℝ) =
            Option.map (· * FA1Math.lPartial Q numKVBlocks K scale k idx.1)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        -- Goal: WithBot.realExp (Option.map₂ - mPartial(k) (max mPartial(k) (some sup'))) =
        --       some alphaPartial.
        -- hAlphaData idx.1 says it's WithBot.realExp (Option.map₂ - mPartial(k) mPartial(k+1)).
        -- We bridge via congrArg.
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · -- LHS: ∑ x, WithBot.realExp (Option.map (... - max ...) ...)
        -- RHS: some (∑ j, Real.exp (...))
        -- Step 1: replace each summand by `some (Real.exp ...)` via the same
        -- congr-2 / hmNewData / hPData pattern used in the alpha branch.
        -- Step 2: collapse `∑ some _ = some (∑ _)` via `WithBot.sum_someTerm_eq_some`.
        refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · rw [← FA1Math.block_oAcc_tile_eq Q K V scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop, Tile.expandDim,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex]
      congr 1
      · -- LHS: Option.map (· * oPartial k (idx)) (WithBot.realExp (Option.map₂ - mPartial(k) (max ...)))
        -- RHS: Option.map₂ * (some alphaPartial) (some (oPartial k idx))
        rw [show (Option.map₂ (· * ·)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))) :
              WithBot ℝ) =
            Option.map (· * FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · -- LHS: ∑ x, Option.map (· * V (...)) (WithBot.realExp (Option.map (... - max ...)))
        -- RHS: some (∑ j, Real.exp (...) * V (...))
        refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        -- Goal at j: Option.map (· * V (...)) (WithBot.realExp (Option.map (...) (max ...)))
        --           = some (Real.exp (...) * V (...))
        -- Strategy: descend through Option.map (·*V) and WithBot.realExp via `congr 1`s
        -- (the `· * V` factor on the outside aligns with the `· * V` factor in the RHS),
        -- then use the same congr-2 + hmNewData + hPData pattern as the alpha branch.
        -- We can't use `congr 1` directly because LHS has `Option.map` head, RHS has
        -- `some` head. So we first reshape RHS to match: `some y = Option.map (·*V) (some r)`
        -- where `r * V = y`, holds by `rfl`.
        change Option.map _ _ =
            Option.map (fun a : ℝ => a *
              V (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, idx.2.1, PUnit.unit))
              (some (Real.exp (FA1Math.scaledScore Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j) -
                  WithBot.unbotD 0
                    (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))))
        congr 1
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx

set_option maxHeartbeats 800000 in
/-- Strided variant of `fa1_step`. One iteration of the strided KV-block
loop preserves `P_fa1_strided`. Same streaming-math content as
`fa1_step`; the only operational differences are the `k_ptrs` / `v_ptrs`
trees (assembled from `*_base_off` plus `offs_n[:, None] * stride_*N +
offs_d[None, :] * stride_*D`) and the `*_base_off` / `pid_*` /
`s.pids 0/1/2` registers being threaded through unchanged. -/
theorem fa1_step_strided
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
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
  let kTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scores : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => ((FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)
          : ℝ) : WithBot ℝ))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.alphaPartial Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      Real.exp (FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0)
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.lPartial Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      FA1Math.oPartial Q numKVBlocks K V scale (k + 1) idx
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let s5 := s4.setReg "v" .real [Bk, D] vTile
  let s6 := s5.setReg "scores" .real [M, Bk] scores
  let s7 := s6.setReg "m_block" .real [M] mBlock
  let s8 := s7.setReg "m_new" .real [M] mNew
  let s9 := s8.setReg "alpha" .real [M] alpha
  let s10 := s9.setReg "p" .real [M, Bk] p
  let s11 := s10.setReg "l_new" .real [M] lNew
  let s12 := s11.setReg "o_acc" .real [M, D] oNew
  let s13 := s12.setReg "m_i" .real [M] mNew
  let s' := s13.setReg "l_i" .real [M] lNew
  apply Exists.intro
  constructor
  · simp [fa1LoopBodyStrided, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, Option.bind, hBk, hoffs_d, hq, hm, hl, ho,
      hk_base, hv_base]
    have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg (batch * sKB + headIdx * sKH
            + (k * Bk + j.val) * sKN + d.val * sKD) =
          K (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided kReg s
        (batch * sKB + headIdx * sKH) sKN sKD K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg (batch * sVB + headIdx * sVH
            + (k * Bk + j.val) * sVN + d.val * sVD) =
          V (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided vReg s
        (batch * sVB + headIdx * sVH) sVN sVD V hV k hk
    simp_rw [hKmem, hVmem]
    rfl
  · have hmNewData : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        (FA1Math.mPartial Bk Q numKVBlocks K scale k i) ⊔
          (some ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i H
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup' H
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataComm : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit)))))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [← hmNewData i]
      congr 2
      apply Finset.sup'_congr (H := by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) rfl
      intro j _
      rw [mul_comm]
    have hAlphaData : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        FA1Math.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPData : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j
      have hm : FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ :=
        FA1Math.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
          (Nat.succ_le_iff.mpr hk) i
      obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
      rw [← hm_eq]
      simp [FA1Math.scaledScore, mul_comm]
    have hAlphaDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      change WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i)
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i) z)) hM).trans
        (hAlphaData i)
    have hPDataOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      change WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b) z)) hM).trans
          (hPData i j)
    have hAlphaDataComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      exact hAlphaDataOf i H
    have hPDataComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      simpa [mul_comm] using hPDataOf i j H
    have hAlphaDataMaxComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxComm' : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm' : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataOf i H
    have hPDataMaxOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataOf i j H
    have max_eq_sup : ∀ (a b : WithBot ℝ), max a b = a ⊔ b := by
      intro a b
      rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hpids0]
    · simp [hpids1]
    · simp [hpids2]
    · simp [hpid_qb]
    · simp [hpid_h]
    · simp [hpid_b]
    · simp [hq_base]
    · simp [hk_base]
    · simp [hv_base]
    · simp [ho_base]
    · simp [hoffs_m]
    · simp [hoffs_d]
    · simp [hq]
    · simp
      ext idx
      exact hmNewData idx.1
    · rw [← FA1Math.block_lNew_tile_eq Q K scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
      congr 1
      · rw [show (Option.map₂ (· * ·) (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.lPartial Q numKVBlocks K scale k idx.1)) :
              WithBot ℝ) =
            Option.map (· * FA1Math.lPartial Q numKVBlocks K scale k idx.1)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · rw [← FA1Math.block_oAcc_tile_eq Q K V scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop, Tile.expandDim,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex]
      congr 1
      · rw [show (Option.map₂ (· * ·)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))) :
              WithBot ℝ) =
            Option.map (· * FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        change Option.map _ _ =
            Option.map (fun a : ℝ => a *
              V (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, idx.2.1, PUnit.unit))
              (some (Real.exp (FA1Math.scaledScore Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j) -
                  WithBot.unbotD 0
                    (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))))
        congr 1
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx

set_option maxHeartbeats 800000 in
/-- Causal strided loop step. This is the causal analogue of
`fa1_step_strided`: the operational body additionally constructs the
causal mask and `tl.where`-masked scores before the same online-softmax
update. -/
theorem fa1_step_strided_causal
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_causal qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_causal qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
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
  let kTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scoresRaw : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let causal : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      decide (k * Bk + idx.2.1.val ≤ qb * M + idx.1.val)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun jLocal =>
          FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) jLocal))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (Option.map₂ (fun x y : ℝ => x - y)
          (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
          (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))⟩
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale (k + 1) idx
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let s5 := s4.setReg "v" .real [Bk, D] vTile
  let s6 := s5.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s7 := s6.setReg "causal" .bool [M, Bk] causal
  let s8 := s7.setReg "scores" .real [M, Bk] scores
  let s9 := s8.setReg "m_block" .real [M] mBlock
  let s10 := s9.setReg "m_new" .real [M] mNew
  let s11 := s10.setReg "alpha" .real [M] alpha
  let s12 := s11.setReg "p" .real [M, Bk] p
  let s13 := s12.setReg "l_new" .real [M] lNew
  let s14 := s13.setReg "o_acc" .real [M, D] oNew
  let s15 := s14.setReg "m_i" .real [M] mNew
  let s' := s15.setReg "l_i" .real [M] lNew
  apply Exists.intro
  constructor
  · simp [fa1LoopBodyStridedCausal, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim, Tile.transpose, Tile.dot,
      Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      NumericDType.sub, ComparableDType.ge, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg (batch * sKB + headIdx * sKH
            + (k * Bk + j.val) * sKN + d.val * sKD) =
          K (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided kReg s
        (batch * sKB + headIdx * sKH) sKN sKD K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg (batch * sVB + headIdx * sVH
            + (k * Bk + j.val) * sVN + d.val * sVD) =
          V (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided vReg s
        (batch * sVB + headIdx * sVH) sVN sVD V hV k hk
    simp_rw [hKmem, hVmem]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hpids0]
    · simp [hpids1]
    · simp [hpids2]
    · simp [hpid_qb]
    · simp [hpid_h]
    · simp [hpid_b]
    · simp [hq_base]
    · simp [hk_base]
    · simp [hv_base]
    · simp [ho_base]
    · simp [hoffs_m]
    · simp [hoffs_d]
    · simp [hq]
    · simp
      funext idx
      simp_rw [Finset.sup'_eq_sup]
      exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
    · rw [← FA1MathCausal.block_lNew_tile_eq (qb * M) Q K scale k hk]
      simp [Tile.bop, Tile.uop, Tile.ofReal,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
      funext idx
      rw [show some
            (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1 +
              ∑ x : Fin Bk,
                (WithBot.realExp
                  (Option.map₂ (fun x y : ℝ => x - y)
                    (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                      (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) x))
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                      (k + 1) idx.1))).unbotD 0)
            =
            Option.map₂ (fun x y : ℝ => x + y)
              (some
                (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                  FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1))
              (some
                (∑ x : Fin Bk,
                  (WithBot.realExp
                    (Option.map₂ (fun x y : ℝ => x - y)
                      (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                        (FA1Math.blockIndex Bk numKVBlocks k
                          (Nat.succ_le_iff.mpr hk) x))
                      (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                        (k + 1) idx.1))).unbotD 0)) from rfl]
      congr 1
      · change Option.map _ _ =
            Option.map
              (fun a : ℝ =>
                a * FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
              (some (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1))
        congr 1
        unfold FA1MathCausal.alphaCausal
        simp_rw [Finset.sup'_eq_sup]
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        simp_rw [Finset.sup'_eq_sup]
        have hscore :
            (if k * Bk + ↑j ≤ qb * M + ↑idx.1 then
              some
                ((∑ x_1 : Fin D,
                  Q (idx.1, x_1, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, x_1, PUnit.unit)) * scale)
            else none : WithBot ℝ) =
              FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j) := by
          by_cases h_mask : k * Bk + ↑j ≤ qb * M + ↑idx.1
          · rw [if_pos h_mask]
            rw [FA1MathCausal.maskedScore_of_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; exact h_mask)]
            unfold FA1Math.scaledScore
            ring_nf
            rfl
          · rw [if_neg h_mask]
            rw [FA1MathCausal.maskedScore_of_not_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; omega)]
            rfl
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j))
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        · unfold FA1Math.scaledScore
          ring_nf
          rfl
        · exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
    · rw [← FA1MathCausal.block_oAcc_tile_eq (qb * M) Q K V scale k hk]
      simp [Tile.bop, Tile.uop, Tile.ofReal, Tile.expandDim,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex]
      funext idx
      rw [show some
            (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                  (idx.1, idx.2.1, PUnit.unit) +
              ∑ x : Fin Bk,
                (WithBot.realExp
                  (Option.map₂ (fun x y : ℝ => x - y)
                    (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                      (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) x))
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                      (k + 1) idx.1))).unbotD 0 *
                  V (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) x, idx.2.1, PUnit.unit))
            =
            Option.map₂ (fun x y : ℝ => x + y)
              (some
                (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                  FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                    (idx.1, idx.2.1, PUnit.unit)))
              (some
                (∑ x : Fin Bk,
                  (WithBot.realExp
                    (Option.map₂ (fun x y : ℝ => x - y)
                      (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                        (FA1Math.blockIndex Bk numKVBlocks k
                          (Nat.succ_le_iff.mpr hk) x))
                      (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                        (k + 1) idx.1))).unbotD 0 *
                    V (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) x, idx.2.1, PUnit.unit))) from rfl]
      congr 1
      · change Option.map _ _ =
            Option.map
              (fun a : ℝ =>
                a * FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                  (idx.1, idx.2.1, PUnit.unit))
              (some (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1))
        congr 1
        unfold FA1MathCausal.alphaCausal
        simp_rw [Finset.sup'_eq_sup]
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        change Option.map _ _ =
            Option.map (fun a : ℝ => a *
              V (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, idx.2.1, PUnit.unit))
              (some ((WithBot.realExp
                (Option.map₂ (fun x y : ℝ => x - y)
                  (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                    (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j))
                  (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                    (k + 1) idx.1))).unbotD 0))
        congr 1
        simp_rw [Finset.sup'_eq_sup]
        have hscore :
            (if k * Bk + ↑j ≤ qb * M + ↑idx.1 then
              some
                ((∑ x_1 : Fin D,
                  Q (idx.1, x_1, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, x_1, PUnit.unit)) * scale)
            else none : WithBot ℝ) =
              FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j) := by
          by_cases h_mask : k * Bk + ↑j ≤ qb * M + ↑idx.1
          · rw [if_pos h_mask]
            rw [FA1MathCausal.maskedScore_of_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; exact h_mask)]
            unfold FA1Math.scaledScore
            ring_nf
            rfl
          · rw [if_neg h_mask]
            rw [FA1MathCausal.maskedScore_of_not_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; omega)]
            rfl
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j))
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        · unfold FA1Math.scaledScore
          ring_nf
          rfl
        · exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx

/-- Strided readout stage: once `P_fa1_strided numKVBlocks` holds at the
loop exit, the post-loop normalization (`out := o_acc / l_i[:, None]`)
and strided store (`tl.store(outReg + oBase + offs_m * sOM + offs_d *
sOD, out)`) realize the ℝ-level attention spec at the per-`(b, h,
q_block)` slice.

The injectivity hypothesis `hInj` is the standard tile-local
non-overlap requirement on the `[M, D]` output tile. The 4D-wrapper
corollary (issue #39 step (iv)) supplies it via `Offset.strided_inj`
applied to the global `[B, H, S_q, D]` layout — i.e. once the global
strided layout is non-overlapping, the per-instance tile-local view
inherits injectivity for free. -/
theorem fa1_postLoop_correct_strided
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1PostLoopStrided outReg M D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale idx) := by
  intro idx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStrided, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]
  simp [FA1Math.streaming_eq_attentionReal hBk Q numKVBlocks hNumKVBlocks K V scale idx
        (FA1Math.lPartial_final_ne_zero hBk Q numKVBlocks hNumKVBlocks K scale idx.1)]

/-- Causal strided readout stage, raw accumulator form. This theorem
closes the operational tail of the causal kernel: assuming the causal
loop invariant at `numKVBlocks`, the post-loop writes
`oPartial / lPartial` to the strided output tile. The final theorem
below composes this with `streaming_eq_attentionRealCausalBlock`. -/
theorem fa1_postLoop_correct_strided_causal_raw
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_causal qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1PostLoopStrided outReg M D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  intro idx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStrided, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]

theorem fa1_forward_correct
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (_hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (_hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (_hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionReal Q K V scale idx) := by
  -- Stage B: pre-loop establishes P_fa1 0.
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct qReg kReg vReg Q K V scale s _hQ _hK _hV
  -- Stage C: forLoop_inv chains fa1_step over numKVBlocks iterations.
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv (P := P_fa1 qReg kReg vReg s.pid Q K V scale) hP0
      (fun i st hi hPi => fa1_step hBk qReg kReg vReg Q K V scale s.pid i st hi hPi)
  -- Stage D: post-loop readout matches `attentionReal`.
  intro idx
  -- Reshape `exec` through body = preLoop ++ [forLoop] ++ postLoop.
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale).body =
        fa1PreLoop qReg M D ++
        [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
        fa1PostLoop outReg M D from rfl]
  -- Walk preLoop: stepStmts (preLoop ++ [forLoop] ++ postLoop) s
  --             = stepStmts ([forLoop] ++ postLoop) s0
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoop qReg M D)
        (l2 := [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
          fa1PostLoop outReg M D) hPre]
  -- Walk forLoop: stepStmts ([forLoop] ++ postLoop) s0 = stepStmts postLoop sLoop.
  rw [stepStmts.append_some
        (l1 := [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)])
        (l2 := fa1PostLoop outReg M D) ?_]
  · exact fa1_postLoop_correct hBk hNumKVBlocks qReg kReg vReg outReg s.pid Q K V scale
      sLoop hPLoop idx
  · -- stepStmts [forLoop] s0 = stepStmts [] sLoop = some sLoop
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil

/-- Strided / 4D-aware FA-1 forward correctness — single program-instance
slice. Threads `fa1_preLoop_correct_strided`, `fa1_step_strided`, and
`fa1_postLoop_correct_strided` through `forLoop_inv` exactly the way
`fa1_forward_correct` does for the 2D kernel. The output equals
`attentionReal` on the per-`(b, h, q_block)` slice; the 4D wrapper
(issue #39 step (iv)) lifts this to `attentionReal4D` via
`attentionReal4D_slice`.

The boundary / non-overlap requirement that 4D Triton FA-1 lives or
dies on (`qb*M + (M-1) < S_q`, plus `Σ (d-1)*s < next stride` along
each axis of `[B, H, S_q, D]`) is folded entirely into the readout
injectivity hypothesis `hInj` here — kept abstract so a 4D-wrapper
caller can package it via `Offset.strided_inj` + `StridesValid`, and a
2D-equivalent caller (B = H = 1, `sOB = sOH = 0`, `sOM = D`,
`sOD = 1`) can discharge it directly. -/
theorem fa1_forward_correct_strided
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale idx) := by
  -- Stage B: strided pre-loop establishes P_fa1_strided 0.
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQ hK hV
  -- Stage C: forLoop_inv chains fa1_step_strided.
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0
      (fun i st hi hPi =>
        fa1_step_strided hBk qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st hi hPi)
  -- Stage D: strided post-loop readout.
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).body =
        fa1PreLoopStrided qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
        fa1PostLoopStrided outReg M D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
          fa1PostLoopStrided outReg M D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided hBk hNumKVBlocks
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx

/-- Causal strided forward correctness in raw streaming form, parameterized
by the causal loop-step lemma. Kept as a factoring lemma for the closed
theorem `fa1_forward_correct_strided_causal_raw`, which supplies
`fa1_step_strided_causal` directly. -/
theorem fa1_forward_correct_strided_causal_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_causal qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQ hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_causal qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).body =
        fa1PreLoopStrided qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
        fa1PostLoopStrided outReg M D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
          fa1PostLoopStrided outReg M D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_causal_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx

/-- Causal strided forward correctness in raw streaming form. This is
`fa1_forward_correct_strided_causal_raw_of_step` with the loop-step
obligation discharged by `fa1_step_strided_causal`. The remaining
math bridge to the user-facing causal attention spec is handled by
the 4D theorem layer below. -/
theorem fa1_forward_correct_strided_causal_raw
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_causal_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQ hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- Causal strided forward correctness, stated against the local-block
causal attention spec rather than the raw streaming accumulator ratio. -/
theorem fa1_forward_correct_strided_causal
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale idx) := by
  intro idx
  rw [fa1_forward_correct_strided_causal_raw hBk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQ hK hV hInj idx]
  congr 1
  exact FA1MathCausal.streaming_eq_attentionRealCausalBlock hBk
    (s.pids 0 * M) Q numKVBlocks hNumKVBlocks K V scale idx

/-- 4D-aware corollary of `fa1_forward_correct_strided`. Given inputs
laid out via `Offset.strided` over `[B, H, S, D]` (with valid strides
producing a non-overlapping memory layout) and a Q-side boundary
hypothesis ensuring the M-row block fits within `S_q`, the strided
FA-1 kernel produces `attentionReal` of the per-`(b, h, q_block)`
slice.

Result is in slice form (`attentionReal` of `slice4DQRows` /
`slice4DFlat`) rather than `attentionReal4D`. This theorem is the
intermediate proof that composes directly with the strided inner
theorem; `fa1_forward_correct_4D` below packages it with
`attentionReal_slice_eq_attentionReal4D` to expose the user-facing
`attentionReal4D` statement. -/
theorem fa1_forward_correct_4D_slice
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                (slice4DFlat Bk numKVBlocks V4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale idx) := by
  intro idx
  -- Convert hQ4D's 4D Offset.strided premise to inner-theorem tile-local form.
  have hQ_inner : InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD)
      (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) hQBnd) := by
    intro tileIdx
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    exact h
  -- Convert hK4D similarly.
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (slice4DFlat Bk numKVBlocks K4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Convert hV4D similarly.
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (slice4DFlat Bk numKVBlocks V4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Output tile-local injectivity from `Offset.strided_inj`. The 4D
  -- `StridesValid` decomposes into four nested ∧'s; the third clause
  -- gives `(D-1)*sOD < sOM` and the fourth gives `0 < sOD`, which
  -- together form `Offset.StridesValid [M, D] [sOM, sOD]`.
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
  exact fa1_forward_correct_strided hBk hNumKVBlocks
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd)
    (slice4DFlat Bk numKVBlocks K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    (slice4DFlat Bk numKVBlocks V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    scale s
    hQ_inner hK_inner hV_inner hInj idx

/-- Boundary 4D-aware corollary of `fa1_forward_correct_strided_boundary`.
Mirrors `fa1_forward_correct_4D_slice` but uses the boundary kernel and the
boundary-masked Q-row slicer. K and V are sliced directly via `sliceBH`
(shape `[S_k, D]`) rather than `slice4DFlat`, since the boundary kernel
already takes K/V on the logical `[S_k, D]` domain.

Differences from the non-boundary version:
* No `hQBnd : s.pids 0 * M + M ≤ S_q`: the boundary kernel's store mask
  handles the partial Q-row tail. Instead the conclusion is per-`idx`,
  guarded by `s.pids 0 * M + idx.1.val < S_q`.
* `hSk : Bk * numKVBlocks = S_k` becomes `hSkLe : S_k ≤ Bk * numKVBlocks`
  (the boundary kernel only needs the cover-all-of-K/V condition).
* `hSk : 0 < S_k` is required by `streaming_eq_attentionReal` to ensure
  the running normalizer is non-zero. -/
theorem fa1_forward_correct_4D
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
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
  intro idx
  rw [fa1_forward_correct_4D_slice hBk hNumKVBlocks hSk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q4D K4D V4D scale s hPidB hPidH hQBnd hQ4D hK4D hV4D hOValid idx]
  congr 1
  exact attentionReal_slice_eq_attentionReal4D hSk Q4D K4D V4D scale
    ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd idx

/-- Boundary FA-1 forward correctness in user-facing `attentionReal4D`
form. Bundle of `fa1_forward_correct_4D_boundary_slice` plus an inline
`attentionReal_row_eq` bridge: the same statement, with the slice-form
result re-expressed at the corresponding 4D index of
`attentionReal4D Q4D K4D V4D scale`.

Unlike the non-boundary `_4D` theorem, the conclusion is guarded by
the per-row bound `s.pids 0 * M + idx.1.val < S_q`. The bridge cannot
reuse `attentionReal_slice_eq_attentionReal4D` directly because the
boundary K/V slice uses `sliceBH` (shape `[S_k, D]`) instead of
`slice4DFlat` (shape `[Bk * numKVBlocks, D]`), so the row-equality is
derived inline via `attentionReal_row_eq`. -/
theorem fa1_forward_correct_4D_causal_raw_of_step
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD])
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD
          (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
            (s.pids 0) hQBnd)
          (slice4DFlat Bk numKVBlocks K4D
            ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
          (slice4DFlat Bk numKVBlocks V4D
            ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
          scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD
            (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
              (s.pids 0) hQBnd)
            (slice4DFlat Bk numKVBlocks K4D
              ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
            (slice4DFlat Bk numKVBlocks V4D
              ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
            scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                (slice4DFlat Bk numKVBlocks V4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx.1) := by
  intro idx
  have hQ_inner : InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD)
      (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) hQBnd) := by
    intro tileIdx
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    exact h
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (slice4DFlat Bk numKVBlocks K4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (slice4DFlat Bk numKVBlocks V4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
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
  exact fa1_forward_correct_strided_causal_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd)
    (slice4DFlat Bk numKVBlocks K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    (slice4DFlat Bk numKVBlocks V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    scale s hQ_inner hK_inner hV_inner hInj hStep idx

/-- 4D causal FA-1 forward correctness in raw streaming form. This
discharges the causal loop-step obligation using
`fa1_step_strided_causal`, so callers no longer need to thread a
manual invariant-preservation proof. -/
theorem fa1_forward_correct_4D_causal_raw
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                (slice4DFlat Bk numKVBlocks V4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx.1) := by
  exact fa1_forward_correct_4D_causal_raw_of_step hSk
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q4D K4D V4D scale s
    hPidB hPidH hQBnd hQ4D hK4D hV4D hOValid
    (fun i st hi hPi =>
      fa1_step_strided_causal hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) hQBnd)
        (slice4DFlat Bk numKVBlocks K4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        (slice4DFlat Bk numKVBlocks V4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        scale i st hi hPi)

/-- 4D causal FA-1 forward correctness, stated directly against the
user-facing `attentionReal4DCausal` spec. This is the fully cleaned
causal theorem: no external step obligation and no raw accumulator
ratio in the conclusion. -/
theorem fa1_forward_correct_4D_causal
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
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
  intro idx
  rw [fa1_forward_correct_4D_causal_raw hBk hSk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q4D K4D V4D scale s
        hPidB hPidH hQBnd hQ4D hK4D hV4D hOValid idx]
  congr 1
  rw [FA1MathCausal.streaming_eq_attentionRealCausalBlock hBk
        (s.pids 0 * M)
        (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) hQBnd)
        numKVBlocks hNumKVBlocks
        (slice4DFlat Bk numKVBlocks K4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        (slice4DFlat Bk numKVBlocks V4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        scale idx]
  exact attentionRealCausalBlock_slice_eq_attentionReal4DCausal hSk Q4D K4D V4D scale
    ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd idx

/-! ## Layout-level theorem surface

These are the user-facing wrappers over the final 4D theorems above. They
bundle the sixteen Q/K/V/O stride arguments into `FA1Layout4D`, expose
named offset helpers for the `InputAt` premises, and keep the conclusion in
the same `attentionReal4D` / `attentionReal4DCausal` form. The input
premises are stated through `TensorView.loaded`, which is the memory-contract
surface users should normally see. -/

/-- FA-1 forward correctness over a bundled 4D layout. This is the same
statement as `fa1_forward_correct_4D`, but with the stride plumbing hidden
behind `FA1Layout4D`. -/
theorem fa1_forward_correct_4D_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (layout.kernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Layout4D.kernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D hBk hNumKVBlocks hSk
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D layout.hOValid idx

/-- Causal FA-1 forward correctness over a bundled 4D layout. This is the
layout-level version of `fa1_forward_correct_4D_causal`. -/
theorem fa1_forward_correct_4D_causal_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (layout.causalKernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Layout4D.causalKernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_causal hBk hNumKVBlocks hSk
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D layout.hOValid idx

theorem fa1_forward_correct_4D_views
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
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (views.kernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Views4D.kernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_layout hBk hNumKVBlocks hSk views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D idx

/-- Causal FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_causal_views
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
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (views.causalKernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Views4D.causalKernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_causal_layout hBk hNumKVBlocks hSk views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D idx


end VeriTile.Examples
