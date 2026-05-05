/-
VeriTile.Examples.FlashAttention1.V0.Bodies

Split-out support for FlashAttention-1 v0/full-tile proofs.
-/

import VeriTile.Examples.FlashAttention1.Common

namespace VeriTile.Examples

open VeriTile.Triton

def fa1PreLoop (qReg : RegionName) (M D : Nat) : List Stmt :=
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
def fa1LoopBody (kReg vReg : RegionName)
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
def fa1PostLoop (outReg : RegionName) (M D : Nat) : List Stmt :=
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
    (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale).toAlgKernel.body =
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
def fa1PreLoopStrided (qReg : RegionName) (M D : Nat)
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
def fa1LoopBodyStrided (kReg vReg : RegionName)
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
def fa1LoopBodyStridedCausal (kReg vReg : RegionName)
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
def fa1PostLoopStrided (outReg : RegionName) (M D : Nat)
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
        sOB sOH sOM sOD scale).toAlgKernel.body =
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
        sOB sOH sOM sOD scale).toAlgKernel.body =
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


end VeriTile.Examples
