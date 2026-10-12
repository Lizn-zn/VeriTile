import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealLoop
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Source initialization and complete region-memory real correctness. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealInit
open VeriTile.Triton RealProgram RealStep RealLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def launch (u : BlockState) : Launch := ⟨u.pids 0, u.pids 1⟩

def initial (p : Layout) (u : BlockState) : BlockState :=
  let u1 := u.setReg "i_s" .nat [] (Tile.scalar (u.pids 0))
  let u2 := u1.setReg "i_bh" .nat [] (Tile.scalar (u.pids 1))
  let u3 := u2.setReg "o_i" .nat [p.BT] ⟨fun i => i.1.val⟩
  let u4 := u3.setReg "m_s" .real [p.BT, p.BT] (lowerTriTile p.BT)
  u4.setReg "b_z" .real [p.BS] (Tile.vec fun _ => some 0)

theorem triangular_eval (p : Layout) (u : BlockState)
    (h : u.regs .nat [p.BT] "o_i" = some ⟨fun i => i.1.val⟩) :
    evalOp (.where (.ge .nat (.consR (.consL .nil))
      (.expandDim ⟨1, by simp⟩ (.ref .nat [p.BT] "o_i"))
      (.expandDim ⟨0, by simp⟩ (.ref .nat [p.BT] "o_i")))
      (.broadcast (.const 1.0) [p.BT, p.BT]) (.broadcast (.const 0.0) [p.BT, p.BT])) u =
        some (lowerTriTile p.BT) := by
  simp only [evalOp, evalOp_ref, h, Option.bind_some, Option.bind_eq_bind]
  erw [evalOp_expandDim, evalOp_ref, h, evalOp_expandDim, evalOp_ref, h]
  simp only [Option.bind_some, Option.bind_eq_bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [Tile.cop, Tile.expandDim, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.ge, lowerTriTile, Tile.select]

theorem executes (p : Layout) (u : BlockState) :
    stepStmts (initCode p) u = some (initial p u) := by
  unfold initCode
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BT) _ = some
    (⟨fun i : TileIndex [p.BT] => i.1.val⟩ : Tile .nat [p.BT]) from by rw [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (triangular_eval p _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BS] (.const 0)) _ =
    some (Tile.vec fun _ => some (0 : ℝ)) from by simp [evalOp_full, evalOp_const]; rfl))]
  rw [stepStmts.nil]
  rfl

theorem invariant (p : Layout) (u : BlockState) (s o : RegionName) (a : Nat → ℝ)
    (hm : ∀ off, u.readMem s off = a off) :
    Holds u p (launch u) s o a 0 (initial p u) := by
  refine ⟨rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [initial, launch]
  · simp [initial, launch]
  · simp [initial]
  · simp [initial, RealLoop.carry, RealMath.carry]
  · intro off; simpa [initial] using hm off
  · intro t j ha ht; simp at ht
  · intro r off hn; rfl

theorem chunks_cover (p : Layout) (hBT : 0 < p.BT) : p.T ≤ numChunks p * p.BT := by
  unfold numChunks
  have hd := Nat.div_add_mod (p.T + p.BT - 1) p.BT
  have hm := Nat.mod_lt (p.T + p.BT - 1) hBT
  rw [Nat.mul_comm]
  omega

theorem original_executes (p : Layout) (u : BlockState) (s o : RegionName)
    (dtype out_dtype : FloatDType) (a : Nat → ℝ) (hm : ∀ off, u.readMem s off = a off)
    (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p (launch u)) :
    ∃ v, exec (RealSource.source Bool.false dtype out_dtype s o
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel u = some v ∧
      (∀ (t : Nat) (j : Fin p.BS), RealStep.active p (launch u) t j.val →
        v.readMem o (address p (launch u) t j.val) = RealLoop.expected p (launch u) a t j) ∧
      BlockState.WriteWithin (footprint p (launch u) o) u v := by
  obtain ⟨v, he, h⟩ := RealLoop.executes u p (launch u) s o a hSO hBT hinj
    (initial p u) (invariant p u s o a hm)
  refine ⟨v, ?_, ?_, h.frame⟩
  · rw [exec, source_body, body, ProgramRewrite.append, executes]
    change stepStmts [RealLoop.statement p s o] (initial p u) = some v
    rw [stepStmts.cons_some he, stepStmts.nil]
  · intro t j ha
    exact h.output t j ha (lt_of_lt_of_le ha.1 (chunks_cover p hBT))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealInit
