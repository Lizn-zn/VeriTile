import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealIteration
import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealProjection

/-! Complete real execution with arbitrary chunk count and both constexpr
flags. The invariant records all previously emitted O chunks and pre-update
H snapshots, and preserves memory outside those two write windows. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealLoop
open VeriTile.Triton
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1600000
set_option maxRecDepth 10000

theorem earlier_chunk_disjoint (base rows cols stride c n : Nat)
    (hstride : cols ≤ stride) (hc : c < n)
    (i : Fin rows) (j : Fin cols) (i' : Fin rows) (j' : Fin cols) :
    base + (c * rows + i.val) * stride + j.val ≠
      base + (n * rows + i'.val) * stride + j'.val := by
  have hrows := Nat.mul_le_mul_right rows (show c + 1 ≤ n by omega)
  have hi := i.isLt
  have hj := j.isLt
  have hrow : c * rows + i.val + 1 ≤ n * rows + i'.val := by nlinarith
  have hm := Nat.mul_le_mul_right stride hrow
  simp only [Nat.add_mul, Nat.one_mul] at hm ⊢
  omega

structure Invariant (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD : Nat) (STORE : Bool) (s : BlockState) (n : Nat)
    (st : BlockState) : Prop where
  pids : st.pids = s.pids
  head : st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0))
  carry : st.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD n)
  query : ∀ off, st.readMem Q off = s.readMem Q off
  keys : ∀ off, st.readMem K off = s.readMem K off
  values : ∀ off, st.readMem V off = s.readMem V off
  outputCells : ∀ c, c < n → ∀ (i : Fin BT) (j : Fin BD),
    st.mem O (s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val) =
      MemCell.real (aft1OutG s Q K V s_qh scale BT BD c i j)
  snapshotCells : STORE = Bool.true → ∀ c, c < n → ∀ (i j : Fin BD),
    st.mem H (s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) =
      MemCell.real (aft1RecStateG s K V s_qh BT BD c i j)
  frame : ∀ rg off,
    (rg ≠ O ∨ ∀ c, c < n → ∀ (i : Fin BT) (j : Fin BD),
      off ≠ s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val) →
    (STORE = Bool.false ∨ rg ≠ H ∨ ∀ c, c < n → ∀ (i j : Fin BD),
      off ≠ s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) →
    st.mem rg off = s.mem rg off

namespace Invariant
variable {Q K V H O : RegionName} {s_qh s_hh s_ht BT BD n : Nat}
  {scale : ℝ} {STORE : Bool} {s st : BlockState}

theorem output (h : Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s n st)
    (c : Nat) (hc : c < n) (i : Fin BT) (j : Fin BD) :
    st.readMem O (s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val) =
      aft1OutG s Q K V s_qh scale BT BD c i j := by
  rw [BlockState.readMem, h.outputCells c hc i j]
  rfl

theorem snapshots (h : Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s n st)
    (hs : STORE = Bool.true) (c : Nat) (hc : c < n) (i j : Fin BD) :
    st.readMem H (s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) =
      aft1RecStateG s K V s_qh BT BD c i j := by
  rw [BlockState.readMem, h.snapshotCells hs c hc i j]
  rfl

theorem output_typed (h : Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s n st)
    (c : Nat) (hc : c < n) (i : Fin BT) (j : Fin BD) :
    st.readMemAs .real O (s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val) =
      some (aft1OutG s Q K V s_qh scale BT BD c i j) := by
  rw [BlockState.readMemAs, h.outputCells c hc i j]
  rfl

theorem snapshots_typed (h : Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s n st)
    (hs : STORE = Bool.true) (c : Nat) (hc : c < n) (i j : Fin BD) :
    st.readMemAs .real H (s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) =
      some (aft1RecStateG s K V s_qh BT BD c i j) := by
  rw [BlockState.readMemAs, h.snapshotCells hs c hc i j]
  rfl
end Invariant

theorem invariant_step (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState)
    (hQO : Q ≠ O) (hKO : K ≠ O) (hVO : V ≠ O)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hHO : H ≠ O)
    (hstride : BD ≤ s_ht) (n : Nat) (st : BlockState)
    (h : Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s n st) :
    ∃ st', stepStmts (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND)
      (st.setReg "i" .nat [] (Tile.scalar n)) = some st' ∧
      Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s (n + 1) st' := by
  have hp : st.pids 0 = s.pids 0 := congrFun h.pids 0
  have hcarry := aft1BhTileG_congr s st K V s_qh BT BD n hp h.keys h.values
  obtain ⟨st', hrun, hpid', hhead', hcarry', hout, hsnap, hframe⟩ :=
    RealIteration.executes Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND st
      (st.setReg "i" .nat [] (Tile.scalar n)) n hQH hKH hVH hHO hstride
      (by intro rg off; rfl) (by simpa [hp] using h.head) (by simp)
      (by simpa [hcarry] using h.carry)
  have hreads (rg : RegionName) (hrO : rg ≠ O) (hrH : rg ≠ H) (off : Nat) :
      st'.readMem rg off = st.readMem rg off := by
    simp only [BlockState.readMem, hframe rg off (Or.inl hrO) (Or.inr (Or.inl hrH)),
      BlockState.setReg_mem]
  refine ⟨st', hrun, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpid'.trans h.pids
  · simpa [hp] using hhead'
  · rw [hcarry', aft1BhTileG_congr s st K V s_qh BT BD (n + 1) hp h.keys h.values]
  · intro off; rw [hreads Q hQO hQH]; exact h.query off
  · intro off; rw [hreads K hKO hKH]; exact h.keys off
  · intro off; rw [hreads V hVO hVH]; exact h.values off
  · intro c hc i j
    rcases Nat.lt_succ_iff_lt_or_eq.mp hc with hold | rfl
    · have hcell := hframe O (s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val)
        (Or.inr (by
          intro i' j'
          rw [hp]
          exact earlier_chunk_disjoint _ BT BD BD c n (Nat.le_refl BD) hold i j i' j'))
        (Or.inr (Or.inl hHO.symm))
      rw [hcell, BlockState.setReg_mem]
      exact h.outputCells c hold i j
    · rw [← hp, hout i j]
      exact congrArg MemCell.real (aft1OutG_congr s st Q K V s_qh scale BT BD c i j hp h.query h.keys h.values)
  · intro hs c hc i j
    rcases Nat.lt_succ_iff_lt_or_eq.mp hc with hold | rfl
    · have hcell := hframe H (s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val)
        (Or.inl hHO) (Or.inr (Or.inr (by
          intro i' j'
          rw [hp]
          exact earlier_chunk_disjoint _ BD BD s_ht c n hstride hold i j i' j')))
      rw [hcell, BlockState.setReg_mem]
      exact h.snapshotCells hs c hold i j
    · rw [← hp, hsnap hs i j]
      simp only [aft1RecStateG, aft1KCellG, aft1VCellG, hp, h.keys, h.values]
  · intro rg off hO hH
    have hOcur : rg ≠ O ∨ ∀ (i : Fin BT) (j : Fin BD),
        off ≠ st.pids 0 * s_qh + (n * BT + i.val) * BD + j.val := by
      rcases hO with hr | ho
      · exact Or.inl hr
      · exact Or.inr (by simpa [hp] using ho n (Nat.lt_succ_self n))
    have hHcur : STORE = Bool.false ∨ rg ≠ H ∨ ∀ (i j : Fin BD),
        off ≠ st.pids 0 * s_hh + (n * BD + i.val) * s_ht + j.val := by
      rcases hH with hs | hr | hh
      · exact Or.inl hs
      · exact Or.inr (Or.inl hr)
      · exact Or.inr (Or.inr (by simpa [hp] using hh n (Nat.lt_succ_self n)))
    rw [hframe rg off hOcur hHcur, BlockState.setReg_mem]
    apply h.frame rg off
    · exact hO.imp_right (fun ho c hc => ho c (by omega))
    · exact hH.imp_right (fun hh => hh.imp_right (fun hv c hc => hv c (by omega)))

theorem prologue (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD : Nat) (STORE : Bool) (s : BlockState) :
    ∃ st, stepStmts (aft1PrologueG BD) s = some st ∧
      Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s 0 st := by
  unfold aft1PrologueG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BD, BD] (Op.const 0)) _ =
      some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BD, BD]) from by
        simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · simp
  · rw [BlockState.setReg_same, aft1BhTileG_zero]
  · intro off; rfl
  · intro off; rfl
  · intro off; rfl
  · intro c hc; omega
  · intro _ c hc; omega
  · intro rg off _ _; rfl

/-- Execute every chunk, retaining the output formula, optional snapshots
and the memory frame. No experiment shape restricts BT, BD or NT. -/
theorem executes (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht)
    (hQO : Q ≠ O) (hKO : K ≠ O) (hVO : V ≠ O)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hHO : H ≠ O) :
    ∃ st, exec (attention_fwd_kernel_surface Q K V H O
      s_qh BD 1 s_hh s_ht (NT * BT) scale BT BD NT STORE IFCOND).toAlgKernel s = some st ∧
      Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s NT st := by
  rw [exec, attention_fwd_triton1_body_splitG]
  obtain ⟨s0, hpre, hP0⟩ := prologue Q K V H O s_qh s_hh s_ht scale BT BD STORE s
  rw [stepStmts.append_some hpre]
  have hStop : evalOp (aft1StopOpG BT NT) s0 = some (Tile.scalar NT) := by
    simp only [aft1StopOpG, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
      Option.bind_some, Option.bind_eq_bind]
    refine congrArg some ?_
    apply Tile.ext
    intro z
    simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.div, NumericDType.sub, NumericDType.add]
    show (NT * BT + BT - 1) / BT = NT
    have hcomm : NT * BT = BT * NT := Nat.mul_comm NT BT
    have heq : NT * BT + BT - 1 = BT * NT + (BT - 1) := by rw [hcomm]; omega
    rw [heq, Nat.mul_add_div hBT]
    have : (BT - 1) / BT = 0 := Nat.div_eq_of_lt (by omega)
    omega
  obtain ⟨final, st, hloop, hfinal, hPfinal⟩ :=
    forRangeDyn_inv (idx := "i") (start := 0) (stop := NT) (step := 1)
      (startOp := Op.constNat 0) (stopOp := aft1StopOpG BT NT) (stepOp := Op.constNat 1)
      (P := fun n st => Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s n st ∧ n ≤ NT)
      (by simp [evalOp]) hStop (by simp [evalOp]) (by norm_num)
      ⟨hP0, by norm_num⟩
      (fun i st hlt hi => by
        obtain ⟨st', hstep, hinv⟩ := invariant_step Q K V H O s_qh s_hh s_ht scale
          BT BD NT STORE IFCOND s hQO hKO hVO hQH hKH hVH hHO hstride i st hi.1
        exact ⟨st', hstep, hinv, by omega⟩)
  have heq : final = NT := le_antisymm hPfinal.2 hfinal
  subst final
  rw [stepStmts.cons_some hloop, stepStmts.nil]
  exact ⟨st, rfl, hPfinal.1⟩

/-- The all-flag invariant belongs to the full typed source after real
precision erasure, including its H store and first-chunk shortcut. -/
theorem original_executes (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht)
    (hQO : Q ≠ O) (hKO : K ≠ O) (hVO : V ≠ O)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hHO : H ≠ O) :
    ∃ st, exec (TypedSource.original Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s = some st ∧
      Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s NT st := by
  rw [RealProjection.original_matches_transcription]
  exact executes Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND s
    hBT hstride hQO hKO hVO hQH hKH hVH hHO

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealLoop
