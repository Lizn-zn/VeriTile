import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealSplitStep

/-! Every streamed column, including partial tiles and an empty range.
The invariant retains the cached matrix tiles, actual flat-pointer advances,
initial key/gate inputs, completed column values and a full tagged frame. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitLoop
open VeriTile.Triton RealSplitMath RealSplitProgram RealSplitMemory
open RealSplitSetup (Cached inputs initial)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def footprint (p : Layout) (l : Launch) (out : RegionName) (n : Nat) : WriteFootprint :=
  fun cell => cell.1 = out ∧ ∃ j, j < n ∧ ∃ i : TileIndex [p.BC],
    rowActive p l i.1.val ∧ cell.2 = outputAddress p l i.1.val j

def start (p : Layout) (r : Regions) (s : BlockState) :=
  RealSplitSetup.result p (launch p s) r (initial p s)

structure Holds (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (n : Nat) (s : BlockState) : Prop where
  limit : n ≤ columns p (launch p s0)
  cached : Cached p (launch p s0) r s0 s
  key : ∀ off, s.readMem r.key off = s0.readMem r.key off
  gate : ∀ off, s.readMem r.gate off = s0.readMem r.gate off
  keyPtr : s.regs .ptr [p.BK] "p_k" = some (keyPointer p (launch p s0) r.key n)
  gatePtr : s.regs .ptr [p.BK] "p_gk" = some (keyPointer p (launch p s0) r.gate n)
  output : ∀ j, j < n → ∀ i : Fin p.BC, rowActive p (launch p s0) i.val →
    s.readMem r.output (outputAddress p (launch p s0) i.val j) =
      RealSplitMath.result p (launch p s0) (inputs r s0) (s0.undef r.query) (s0.undef r.gate) scale i.val j
  frame : BlockState.WriteWithin (footprint p (launch p s0) r.output n) s0 s

theorem start_holds (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState) : Holds p r scale s 0 (start p r s) := by
  refine ⟨Nat.zero_le _,RealSplitSetup.result_cached p r s,by intros; rfl,by intros; rfl,?_,?_,?_,by intro region off hn; rfl⟩
  · simp [start,RealSplitSetup.result,RealSplitSetup.pointed]
  · simp [start,RealSplitSetup.result,RealSplitSetup.pointed]
  · intro j hj
    exact (Nat.not_lt_zero j hj).elim

theorem index_holds (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (n : Nat) (s : BlockState)
    (hi : Holds p r scale s0 n s) : Holds p r scale s0 n (s.setReg "j" .nat [] (Tile.scalar n)) := by
  refine ⟨hi.limit,?_,hi.key,hi.gate,by simpa using hi.keyPtr,by simpa using hi.gatePtr,hi.output,hi.frame⟩
  constructor
  · constructor <;> simp [hi.cached.grid.chunk,hi.cached.grid.head,hi.cached.grid.row,hi.cached.grid.column,hi.cached.grid.feature,hi.cached.grid.heads]
  all_goals simp [hi.cached.rows,hi.cached.offsets,hi.cached.keyMask,hi.cached.rowMask,hi.cached.query,hi.cached.gate]

theorem iteration (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (swapped : Bool)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch p s0) ≤ p.BC)
    (n : Nat) (s : BlockState) (hn : n < columns p (launch p s0)) (hi : Holds p r scale s0 n s) :
    ∃ v, stepStmts (RealSplitProgram.iteration p r scale swapped) (s.setReg "j" .nat [] (Tile.scalar n)) = some v ∧
      Holds p r scale s0 (n+1) v := by
  let t := s.setReg "j" .nat [] (Tile.scalar n)
  have ht := index_holds p r scale s0 n s hi
  let v := RealSplitStep.result p (launch p s0) r (inputs r s0) (s0.undef r.query) (s0.undef r.gate) scale n t
  have hf : BlockState.WriteWithin (RealSplitStep.columnFootprint p (launch p s0) r.output n) t v :=
    RealSplitStep.frame _ _ _ _ _ _ _ _ _
  refine ⟨v,?_,?_⟩
  · exact RealSplitStep.executes p (launch p s0) r (inputs r s0) (s0.undef r.query) (s0.undef r.gate) scale n swapped t
      ht.cached.rows ht.cached.offsets ht.cached.rowMask ht.cached.keyMask ht.cached.query ht.cached.gate
      (by simp [t]) ht.keyPtr ht.gatePtr ht.key ht.gate
  · refine ⟨Nat.succ_le_of_lt hn,RealSplitStep.cached _ _ _ _ _ _ _ _ _ _ ht.cached,?_,?_,?_,?_,?_,?_⟩
    · intro off
      unfold BlockState.readMem
      rw [hf.mem_eq_of_not_written (by rintro ⟨h,_⟩; exact hkey h)]
      exact ht.key off
    · intro off
      unfold BlockState.readMem
      rw [hf.mem_eq_of_not_written (by rintro ⟨h,_⟩; exact hgate h)]
      exact ht.gate off
    · simp [v,RealSplitStep.result]
    · simp [v,RealSplitStep.result]
    · intro j hj i hrow
      by_cases he : j = n
      · subst j
        exact RealSplitStep.readback _ _ _ _ _ _ _ _ _ hcols hn i hrow
      · have hjn : j < n := by omega
        have hnot : ¬ RealSplitStep.columnFootprint p (launch p s0) r.output n
            (r.output,outputAddress p (launch p s0) i.val j) := by
          rintro ⟨_,e,_,haddr⟩
          have h := (output_injective p (launch p s0) hcols i.val e.1.val j n
            (Nat.lt_trans hjn hn) hn haddr).2
          exact he h
        unfold BlockState.readMem
        rw [hf.mem_eq_of_not_written hnot]
        exact ht.output j hjn i hrow
    · intro region off hnot
      exact (ht.frame region off (by
        rintro ⟨hr,j,hj,i,hrow,haddr⟩
        exact hnot ⟨hr,j,Nat.lt_trans hj (Nat.lt_succ_self n),i,hrow,haddr⟩)).trans
        (hf region off (by
          rintro ⟨hr,i,hrow,haddr⟩
          exact hnot ⟨hr,n,Nat.lt_succ_self n,i,hrow,haddr⟩))

theorem tripCount_eval (p : Layout) (l : Launch) (s : BlockState) (hg : Grid l s) :
    evalOp (tripCount p) s = some (Tile.scalar (columns p l)) := by
  simp only [tripCount,remaining,times,evalOp,evalOp_ref,hg.chunk,hg.row,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  intro i
  simp only [Tile.scalar,Tile.bop,Tile.cop,Tile.select,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub,NumericDType.mul,
    ComparableDType.lt,decide_eq_true_eq,columns]
  by_cases h : p.BC < p.T-l.chunk*p.BT-l.block*p.BC
  · simp only [h,↓reduceIte,min_eq_left (Nat.le_of_lt h)]
  · simp only [h,↓reduceIte,min_eq_right (Nat.le_of_not_gt h)]

theorem executes (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (swapped : Bool)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch p s0) ≤ p.BC)
    (s : BlockState) (hi : Holds p r scale s0 0 s) :
    ∃ v, stepStmt (loop p r scale swapped) s = some v ∧ Holds p r scale s0 (columns p (launch p s0)) v := by
  obtain ⟨n,v,he,hn,hv⟩ := forRangeDyn_inv (idx := "j") (start := 0) (stop := columns p (launch p s0)) (step := 1)
    (P := Holds p r scale s0) (body := RealSplitProgram.iteration p r scale swapped)
    (evalOp_constNat 0 s) (tripCount_eval p (launch p s0) s hi.cached.grid) (evalOp_constNat 1 s)
    (by decide) hi (iteration p r scale s0 swapped hkey hgate hcols)
  have hn' : n = columns p (launch p s0) := Nat.le_antisymm hv.limit hn
  subst n
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitLoop
