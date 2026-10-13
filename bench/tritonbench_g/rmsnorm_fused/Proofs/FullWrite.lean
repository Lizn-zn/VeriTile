import bench.tritonbench_g.rmsnorm_fused.Proofs.FullPost

namespace VeriTile.Bench.TritonBenchG.RmsnormFused.General
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

noncomputable def writeRegisters (s0 s : BlockState) (X W : RegionName)
    (stride N B c : Nat) (eps : ℝ) : BlockState :=
  ((((((s.setReg "off" .nat [] (Tile.scalar (c * B))).setReg "cols" .nat [B]
    (Tile.vec (fun j => c * B + j.val))).setReg "mask" .bool [B]
    (Tile.vec (fun j => decide (c * B + j.val < N)))).setReg "w" .real [B]
    ⟨fun i => some (if c * B + i.1.val < N then s0.readMem W (c * B + i.1.val)
      else s.undef W (c * B + i.1.val))⟩).setReg "x" .real [B]
    (inputTile s0 X stride N B c)).setReg "x_hat" .real [B]
    ⟨fun i => some ((if c * B + i.1.val < N then
      s0.readMem X (s0.pids 0 * stride + (c * B + i.1.val)) else 0) *
      inverseStd s0 X stride N eps)⟩).setReg "y" .real [B]
    ⟨fun i => some (if c * B + i.1.val < N then
      rowSpec s0 X W stride N eps (c * B + i.1.val) else 0)⟩

noncomputable def writeState (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) : BlockState :=
  (TileShape.allIndices [B]).foldl
    (fun st i => if c * B + i.1.val < N then
      st.writeMem Y (s0.pids 0 * stride + (c * B + i.1.val))
        (rowSpec s0 X W stride N eps (c * B + i.1.val)) else st)
    (writeRegisters s0 s X W stride N B c eps)

/-- The actual seven-statement writeback body executes on its source-derived context. -/
theorem write_exec (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) (h : WriteContext s0 X Y W stride N eps s) :
    stepStmts (writeBody W N B) (s.setReg "off" .nat [] (Tile.scalar (c * B))) =
      some (writeState s0 s X Y W stride N B c eps) := by
  simp (config := { contextual := Bool.true }) [writeBody, writeState, writeRegisters, stepStmts, stepStmt, evalOp.eq_def,
    h.xptr, h.yptr, h.rstd, h.readX, h.readW, inputTile,
    Tile.vec, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    BlockState.readMemValue_real, BlockState.writeMemTyped_real,
    Region.cast, FloatDType.real_storeValue, WithBot.unbotD_some,
    rowSpec, inverseStd, meanSquare, ite_mul, mul_ite, apply_ite]
  norm_num
  rfl

theorem write_context (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) (h : WriteContext s0 X Y W stride N eps s)
    (hXY : X ≠ Y) (hWY : W ≠ Y) :
    WriteContext s0 X Y W stride N eps (writeState s0 s X Y W stride N B c eps) := by
  constructor
  · simpa [writeState, BlockState.foldl_writeMem_prop_masked_pids, writeRegisters] using h.pids
  · simpa [writeState, writeRegisters] using h.xptr
  · simpa [writeState, writeRegisters] using h.yptr
  · simpa [writeState, writeRegisters] using h.rstd
  · funext o
    unfold writeState
    rw [BlockState.foldl_writeMem_prop_masked_readMem_other_region _ _ _ _ _ _ X o
      (fun _ _ _ => hXY)]
    simpa [writeRegisters] using congrFun h.readX o
  · funext o
    unfold writeState
    rw [BlockState.foldl_writeMem_prop_masked_readMem_other_region _ _ _ _ _ _ W o
      (fun _ _ _ => hWY)]
    simpa [writeRegisters] using congrFun h.readW o

theorem write_hit (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) (j : Fin B) (hj : c * B + j.val < N) :
    (writeState s0 s X Y W stride N B c eps).readMem Y
      (s0.pids 0 * stride + (c * B + j.val)) =
      rowSpec s0 X W stride N eps (c * B + j.val) := by
  unfold writeState
  have hh := BlockState.scatter_readback_prop_masked_nd (region := Y) (shape := [B])
    (writeRegisters s0 s X W stride N B c eps)
    (fun i => s0.pids 0 * stride + (c * B + i.1.val))
    (fun i => rowSpec s0 X W stride N eps (c * B + i.1.val))
    (fun i => c * B + i.1.val < N)
    (by
      intro a b hab
      dsimp only at hab
      apply Prod.ext
      · apply Fin.ext
        omega
      · exact Subsingleton.elim _ _)
    (j, PUnit.unit)
  simpa [hj] using hh

theorem write_miss (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c k : Nat) (eps : ℝ) (hk : k < c * B ∨ (c+1) * B ≤ k) :
    (writeState s0 s X Y W stride N B c eps).readMem Y
      (s0.pids 0 * stride + k) = s.readMem Y (s0.pids 0 * stride + k) := by
  unfold writeState
  rw [BlockState.scatter_prop_masked_preserves_other_offset _ _ _ _ _ (by
    intro i _
    have hi := i.1.isLt
    simp only [Nat.add_mul, Nat.one_mul] at hk
    omega)]
  simp [writeRegisters]

theorem fold_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ)
    (P : α → Prop) [DecidablePred P] (r : RegionName) (oo : Nat)
    (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hm : P hd
      · simp only [hm, if_true]
        rw [ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk),
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hm)
      · simp only [hm, if_false]
        exact ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)


/-- The full frame excludes exactly the logical output interval, not a padded tile. -/
def OutputFrame (s0 s : BlockState) (Y : RegionName) (stride N : Nat) : Prop :=
  ∀ r o, (r ≠ Y ∨ ∀ k : Fin N, o ≠ s0.pids 0 * stride + k.val) → s.mem r o = s0.mem r o

theorem write_frame (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) (h : OutputFrame s0 s Y stride N) :
    OutputFrame s0 (writeState s0 s X Y W stride N B c eps) Y stride N := by
  intro r o hno
  unfold writeState
  rw [fold_preserve_cell _ _ _ r o _ _ (by
    intro i _ hi hbad
    rcases hno with hne | hmiss
    · exact hne hbad.1
    · exact hmiss ⟨c * B + i.1.val, hi⟩ hbad.2)]
  simpa [writeRegisters] using h r o hno

structure WriteInv (s0 : BlockState) (X Y W : RegionName)
    (stride N : Nat) (eps : ℝ) (off : Nat) (s : BlockState) : Prop where
  context : WriteContext s0 X Y W stride N eps s
  output : ∀ k : Fin N, s.readMem Y (s0.pids 0 * stride + k.val) =
    if k.val < off then rowSpec s0 X W stride N eps k.val
    else s0.readMem Y (s0.pids 0 * stride + k.val)
  frame : OutputFrame s0 s Y stride N

theorem write_inv (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) (h : WriteInv s0 X Y W stride N eps (c * B) s)
    (hXY : X ≠ Y) (hWY : W ≠ Y) :
    WriteInv s0 X Y W stride N eps ((c+1) * B) (writeState s0 s X Y W stride N B c eps) := by
  refine ⟨write_context s0 s X Y W stride N B c eps h.context hXY hWY, ?_,
    write_frame s0 s X Y W stride N B c eps h.frame⟩
  intro k
  by_cases hlo : k.val < c * B
  · rw [write_miss s0 s X Y W stride N B c k.val eps (Or.inl hlo), h.output]
    have hnext : k.val < (c+1)*B := by nlinarith
    simp [hlo, hnext]
  · by_cases hhi : k.val < (c+1)*B
    · have hj : k.val - c*B < B := by
        have hsub := Nat.sub_add_cancel (Nat.le_of_not_gt hlo)
        nlinarith
      let j : Fin B := ⟨k.val-c*B, hj⟩
      have heq : c*B+j.val = k.val := by dsimp [j]; omega
      have hh := write_hit s0 s X Y W stride N B c eps j (by rw [heq]; exact k.isLt)
      rw [heq] at hh
      simpa [hhi] using hh
    · rw [write_miss s0 s X Y W stride N B c k.val eps (Or.inr (Nat.le_of_not_gt hhi)), h.output]
      simp [hlo, hhi]

theorem write_loop (s0 s : BlockState) (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) (hB : 0 < B)
    (hXY : X ≠ Y) (hWY : W ≠ Y) (h : WriteInv s0 X Y W stride N eps 0 s) :
    ∃ c t, stepStmt (.forRange "off" 0 N B (writeBody W N B)) s = some t ∧
      N ≤ c * B ∧ WriteInv s0 X Y W stride N eps (c * B) t := by
  have hinit : ∃ c, 0 = c * B ∧ WriteInv s0 X Y W stride N eps 0 s := ⟨0, by simp, h⟩
  obtain ⟨final, t, he, hn, c, hc, hi⟩ :=
    forRange_inv (idx := "off") (start := 0) (stop := N) (step := B)
      (body := writeBody W N B)
      (P := fun off st => ∃ c, off = c * B ∧ WriteInv s0 X Y W stride N eps off st)
      (Nat.ne_of_gt hB) hinit (by
        intro off st _ hst
        rcases hst with ⟨c, rfl, hst⟩
        refine ⟨writeState s0 st X Y W stride N B c eps,
          write_exec s0 st X Y W stride N B c eps hst.context, c+1, ?_, ?_⟩
        · simp [Nat.add_mul]
        · simpa [Nat.add_mul] using write_inv s0 st X Y W stride N B c eps hst hXY hWY)
  refine ⟨c, t, he, hc ▸ hn, ?_⟩
  simpa [hc] using hi

end VeriTile.Bench.TritonBenchG.RmsnormFused.General
