import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeStep

/-! Full NK-loop execution for both addition orders. The invariant retains the
initial memory and arbitrary unspecified values, fixed launch registers and the
independent partial sum. It includes NK=0 without a special caller premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeLoop
open VeriTile.Triton RealMergeProgram RealMergeStep
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def initial (s : BlockState) : BlockState :=
  ((s.setReg "i_t" .nat [] (Tile.scalar (s.pids 0))).setReg "i_c" .nat [] (Tile.scalar (s.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (s.pids 2))
def start (p : Layout) (s : BlockState) : BlockState :=
  ((initial s).setReg "n_bh" .nat [] (Tile.scalar (s.numPids 2))).setReg "b_A" .real [p.BC,p.BC]
    (⟨fun _ => some 0⟩ : Tile .real [p.BC,p.BC])

structure Holds (p : Layout) (s0 : BlockState) (A : RegionName) (n : Nat) (u : BlockState) : Prop where
  limit : n ≤ p.NK
  memory : u.mem = s0.mem
  padding : u.undef = s0.undef
  grid : Grid (launch s0) u
  carry : u.regs .real [p.BC,p.BC] "b_A" = some (accumulator p (launch s0) (s0.readMem A) (s0.undef A) n)

theorem initialization_executes (s : BlockState) : stepStmts initialization s = some (initial s) := by
  unfold initialization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _)),stepStmts.nil]
  rfl

theorem setup_executes (p : Layout) (s : BlockState) : stepStmts (setup p) (initial s) = some (start p s) := by
  unfold setup
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BC,p.BC] (.const 0)) _ =
      some (⟨fun _ => some 0⟩ : Tile .real [p.BC,p.BC]) from by rw [evalOp_full,evalOp_const]; rfl)),stepStmts.nil]
  rfl

theorem start_holds (p : Layout) (s : BlockState) (A : RegionName) : Holds p s A 0 (start p s) := by
  refine ⟨Nat.zero_le _,rfl,rfl,?_,?_⟩
  · constructor <;> simp [start,initial,launch]
  · simp [start,accumulator,accumulated]

theorem index_holds (p : Layout) (s0 : BlockState) (A : RegionName) (n : Nat) (u : BlockState)
    (hi : Holds p s0 A n u) : Holds p s0 A n (u.setReg "i_k" .nat [] (Tile.scalar n)) := by
  refine ⟨hi.limit,hi.memory,hi.padding,?_,by simpa using hi.carry⟩
  rcases hi.grid with ⟨ht,hc,hh,hn⟩
  constructor <;> simp [ht,hc,hh,hn]

theorem iteration (p : Layout) (s0 : BlockState) (A : RegionName) (swapped : Bool)
    (n : Nat) (u : BlockState) (hn : n < p.NK) (hi : Holds p s0 A n u) :
    ∃ v, stepStmts (RealMergeProgram.iteration p A swapped) (u.setReg "i_k" .nat [] (Tile.scalar n)) = some v ∧
      Holds p s0 A (n+1) v := by
  let t := u.setReg "i_k" .nat [] (Tile.scalar n)
  have ht : Holds p s0 A n t := index_holds p s0 A n u hi
  refine ⟨result p (launch s0) A (s0.readMem A) (s0.undef A) n t,?_,?_⟩
  · exact iteration_executes p (launch s0) A (s0.readMem A) (s0.undef A) n swapped t ht.grid
      (by simp [t]) (by intro off; simp only [BlockState.readMem,ht.memory])
      (by intro off; exact congrFun (congrFun ht.padding A) off) ht.carry
  · refine ⟨Nat.succ_le_of_lt hn,ht.memory,ht.padding,?_,?_⟩
    · rcases ht.grid with ⟨hc,hs,hh,hheads⟩
      constructor <;> simp [result,hc,hs,hh,hheads]
    · simp [result]

theorem executes (p : Layout) (s0 : BlockState) (A : RegionName) (swapped : Bool)
    (u : BlockState) (hi : Holds p s0 A 0 u) :
    ∃ v, stepStmt (loop p A swapped) u = some v ∧ Holds p s0 A p.NK v := by
  obtain ⟨n,v,he,hn,hv⟩ := forRangeDyn_inv (idx := "i_k") (start := 0) (stop := p.NK) (step := 1)
    (P := Holds p s0 A) (body := RealMergeProgram.iteration p A swapped)
    (evalOp_constNat 0 u) (evalOp_constNat p.NK u) (evalOp_constNat 1 u)
    (by decide) hi (iteration p s0 A swapped)
  have hn' : n = p.NK := Nat.le_antisymm hv.limit hn
  subst n
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeLoop
