/-
VeriTile.Examples.FlashAttention1.Core.Bodies

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

@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] =>
          v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp
  simp [Tile.expandDim]
  rfl

@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] =>
          v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp
  simp [Tile.expandDim]
  rfl

@[simp] theorem evalOp_expandDim_one_real {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] name)) s =
      (s.regs .real [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] =>
          v.data (i.1, PUnit.unit) } : Tile .real [M, 1])) := by
  unfold evalOp
  simp [Tile.expandDim]
  rfl

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
  · let s1 : BlockState :=
        BlockState.setReg s "pid" .nat [] (Tile.scalar s.pid)
    let s2 : BlockState :=
        BlockState.setReg s1 "offs_m" .nat [M]
          (Tile.vec (fun i : Fin M => s.pid * M + i.val))
    let s3 : BlockState :=
        BlockState.setReg s2 "offs_d" .nat [D]
          (Tile.vec (fun d : Fin D => d.val))
    let s4 : BlockState :=
        BlockState.setReg s3 "q_ptrs" .nat [M, D] qPtrs
    let s5 : BlockState :=
        BlockState.setReg s4 "q" .real [M, D] qLoaded
    let s6 : BlockState :=
        BlockState.setReg s5 "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
    let s7 : BlockState :=
        BlockState.setReg s6 "l_i" .real [M] (Tile.ofReal fun _ => 0)
    have hregm : s3.regs .nat [M] "offs_m" =
        some (Tile.vec (fun i : Fin M => s.pid * M + i.val)) := by
      simp [s1, s2, s3, BlockState.setReg]
    have hregd : s3.regs .nat [D] "offs_d" =
        some (Tile.vec (fun d : Fin D => d.val)) := by
      simp [s1, s2, s3, BlockState.setReg]
    have h_qptrs :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
              (Op.constNat D))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))) s3 =
          some qPtrs := by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, Option.bind_eq_bind,
        Option.bind_some]
      rw [evalOp_expandDim_one_nat, hregm]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_zero_nat, hregd]
      simp [qPtrs, Tile.bop, NumericDType.add, NumericDType.mul, BlockState.pid_eq]
    have h_load :
        evalOp
            (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs"))
              MaskOpt.none) s4 =
          some qLoaded := by
      have hmem : ∀ r o, s3.mem r o = s.mem r o := by
        intro r o
        simp [s1, s2, s3]
      simp [evalOp_load_region_none, s4, qLoaded, qPtrs, Region.cast]
      ext idx
      cases qReg
      simp [BlockState.readMem, hmem]
    have h1 : stepStmt (Stmt.assign .nat [] "pid" (Op.programId 0)) s = some s1 := by
      simp [stepStmt, s1, BlockState.pid_eq]
    have h2 :
        stepStmt
            (Stmt.assign .nat [M] "offs_m"
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil
                  (Op.ref .nat [] "pid")
                  (Op.constNat M))
                (Op.arange M))) s1 = some s2 := by
      simp [stepStmt, s1, s2, Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.pid_eq]
      congr 1
    have h3 : stepStmt (Stmt.assign .nat [D] "offs_d" (Op.arange D)) s2 = some s3 := by
      simp [stepStmt, s3]
    have h4 :
        stepStmt
            (Stmt.assign .nat [M, D] "q_ptrs"
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
                  (Op.constNat D))
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))) s3 =
          some s4 := by
      simp only [stepStmt, h_qptrs, Option.bind_some]
      rfl
    have h5 :
        stepStmt
            (Stmt.assign .real [M, D] "q"
              (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs"))
                MaskOpt.none)) s4 =
          some s5 := by
      simp [stepStmt, h_load, s5]
    have h6 :
        stepStmt (Stmt.assign .real [M] "m_i" (Op.full [M] Op.negInf)) s5 =
          some s6 := by
      simp [stepStmt, s6]
      congr 1
    have h7 :
        stepStmt (Stmt.assign .real [M] "l_i" (Op.full [M] (Op.const 0))) s6 =
          some s7 := by
      simp [stepStmt, s7, Tile.ofReal]
    have h8 :
        stepStmt (Stmt.assign .real [M, D] "o_acc" (Op.full [M, D] (Op.const 0))) s7 =
          some s0 := by
      simp [stepStmt, s0, s1, s2, s3, s4, s5, s6, s7, Tile.ofReal]
    rw [show fa1PreLoop qReg M D =
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
            (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs"))
              MaskOpt.none)
        , Stmt.assign .real [M] "m_i" (Op.full [M] Op.negInf)
        , Stmt.assign .real [M] "l_i" (Op.full [M] (Op.const 0))
        , Stmt.assign .real [M, D] "o_acc" (Op.full [M, D] (Op.const 0))
        ] by rfl]
    rw [stepStmts.cons_some h1]
    rw [stepStmts.cons_some h2]
    rw [stepStmts.cons_some h3]
    rw [stepStmts.cons_some h4]
    rw [stepStmts.cons_some h5]
    rw [stepStmts.cons_some h6]
    rw [stepStmts.cons_some h7]
    rw [stepStmts.cons_some h8]
    exact stepStmts.nil
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, StreamingAccumulator.mPartial]
    · simp [s0, StreamingAccumulator.lPartial, Tile.ofReal]
    · simp [s0, StreamingAccumulator.oPartial, Tile.ofReal]
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
  let outTile : Tile .real [M, D] :=
    ⟨fun i => some
      (StreamingAccumulator.oPartial Q numKVBlocks K V scale numKVBlocks i /
        StreamingAccumulator.lPartial Q numKVBlocks K scale numKVBlocks i.1)⟩
  let oPtrs : Tile .nat [M, D] :=
    ⟨fun i => (origPid * M + i.1.val) * D + i.2.1.val⟩
  let sOut : BlockState :=
    sLoop.setReg "out" .real [M, D] outTile
  let sPtrs : BlockState :=
    sOut.setReg "o_ptrs" .nat [M, D] oPtrs
  let sFinal : BlockState :=
    (TileShape.allIndices [M, D]).foldl
      (fun acc i => acc.writeMem outReg (oPtrs.data i) (WithBot.unbotD 0 (outTile.data i)))
      sPtrs
  have hOut :
      stepStmt
          (Stmt.assign .real [M, D] "out"
            (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [M, D] "o_acc")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l_i"))))
          sLoop = some sOut := by
    simp only [stepStmt, evalOp_div, Option.bind_eq_bind]
    rw [show evalOp (Op.ref .real [M, D] "o_acc") sLoop =
        some (Tile.ofReal fun idx : TileIndex [M, D] =>
          StreamingAccumulator.oPartial Q numKVBlocks K V scale numKVBlocks idx) by
        simp [evalOp, ho]]
    simp only [Option.bind_some]
    rw [evalOp_expandDim_one_real, hl]
    simp [sOut, outTile, Tile.bop, NumericDType.div, Tile.ofReal]
  have hPtrs :
      stepStmt
          (Stmt.assign .nat [M, D] "o_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
                (Op.constNat D))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))))
          sOut = some sPtrs := by
    have hregmOut : sOut.regs .nat [M] "offs_m" =
        some (Tile.vec fun i : Fin M => origPid * M + i.val) := by
      simp [sOut, hoffs_m]
    have hregdOut : sOut.regs .nat [D] "offs_d" =
        some (Tile.vec fun d : Fin D => d.val) := by
      simp [sOut, hoffs_d]
    simp only [stepStmt, evalOp_add, evalOp_mul, evalOp_constNat, Option.bind_eq_bind,
      Option.bind_some]
    rw [evalOp_expandDim_one_nat, hregmOut]
    simp only [Option.bind_some]
    rw [evalOp_expandDim_zero_nat, hregdOut]
    simp [sPtrs, oPtrs, Tile.bop, NumericDType.add, NumericDType.mul]
  have hStore :
      stepStmt
          (Stmt.store .real [M, D]
            (MemAccess.region outReg (Op.ref .nat [M, D] "o_ptrs"))
          (Op.ref .real [M, D] "out") MaskOpt.none)
          sPtrs = some sFinal := by
    simp [stepStmt, sOut, sPtrs, sFinal, outTile, oPtrs, BlockState.writeMemTyped_real]
  have hStep : stepStmts (fa1PostLoop outReg M D) sLoop = some sFinal := by
    rw [show fa1PostLoop outReg M D =
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
        , Stmt.store .real [M, D]
            (MemAccess.region outReg (Op.ref .nat [M, D] "o_ptrs"))
            (Op.ref .real [M, D] "out") MaskOpt.none
        ] by rfl]
    rw [stepStmts.cons_some hOut]
    rw [stepStmts.cons_some hPtrs]
    rw [stepStmts.cons_some hStore]
    exact stepStmts.nil
  have h_inj_oPtrs : Function.Injective (fun i : TileIndex [M, D] => oPtrs.data i) := by
    simpa [oPtrs] using h_inj_store
  simp [observeTileAt, hStep, sFinal]
  rw [show Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D idx =
      oPtrs.data idx by
        simp [oPtrs, Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
          Nat.add_assoc]]
  rw [BlockState.scatter_readback_nd (s := sPtrs)
    (offsetFn := fun i : TileIndex [M, D] => oPtrs.data i)
    (valueFn := fun i : TileIndex [M, D] => WithBot.unbotD 0 (outTile.data i))
    h_inj_oPtrs idx]
  simp [attentionReal, outTile,
        StreamingAccumulator.streaming_eq_attentionReal hBk Q numKVBlocks hNumKVBlocks K V scale idx
        (StreamingAccumulator.lPartial_final_ne_zero hBk Q numKVBlocks hNumKVBlocks K scale idx.1)]

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
      X (StreamingAccumulator.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      (n * Bk + j.val) * D + d.val =
        Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D
          (StreamingAccumulator.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit) := by
    simp [Offset.rowMajor2D, Offset.strided, StreamingAccumulator.blockIndex]
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
        X (StreamingAccumulator.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
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
      X (StreamingAccumulator.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      base + (n * Bk + j.val) * sN + d.val * sD =
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
            base + idx.1.val * sN + idx.2.1.val * sD)
          (StreamingAccumulator.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit) := by
    simp [StreamingAccumulator.blockIndex]
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
        X (StreamingAccumulator.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit)) := by
  ext idx
  rw [Tile.ofReal_data]
  exact congrArg some
    (fa1_block_read_strided region s base sN sD X hX n hn idx.1 idx.2.1)


end VeriTile.Examples
