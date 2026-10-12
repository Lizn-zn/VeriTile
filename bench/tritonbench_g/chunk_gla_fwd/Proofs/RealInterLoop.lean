import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterStep

/-! Full symbolic K-loop execution for both sources. The invariant connects
all completed iterations to the independent contraction while preserving the
initial tagged memory and zero-oracle premise. K=0 is included. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterLoop
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers RealInterAlgebra
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def inputs (r : Regions) (s : BlockState) : Inputs := ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
def initial (p : Layout) (s : BlockState) : BlockState :=
  ((((s.setReg "i_t" .nat [] (Tile.scalar (s.pids 0))).setReg "i_c" .nat [] (Tile.scalar (s.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg "i_i" .nat [] (Tile.scalar (s.pids 1 / p.NC))).setReg
    "i_j" .nat [] (Tile.scalar (s.pids 1 % p.NC))
def start (p : Layout) (s : BlockState) : BlockState :=
  (initial p s).setReg "b_A" .real [p.BC,p.BC] (⟨fun _ => some 0⟩ : Tile .real [p.BC,p.BC])

structure Holds (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (n : Nat) (u : BlockState) : Prop where
  limit : n ≤ numBlocks p
  memory : u.mem = s0.mem
  padding : u.undef = s0.undef
  grid : Grid p (launch s0) u
  carry : u.regs .real [p.BC,p.BC] "b_A" = some (accumulator p (launch s0) (inputs r s0) scale n)

theorem initialization_executes (p : Layout) (s : BlockState) :
    stepStmts (initialization p) s = some (initial p s) := by
  simp only [initialization,stepStmts,stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some]
  simp only [BlockState.setReg_same,BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide : ("i_c" : RegName) ≠ "i_bh"),
    BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide : ("i_c" : RegName) ≠ "i_i"),
    Option.bind_eq_bind,Option.bind_some]
  rfl

theorem setup_executes (p : Layout) (s : BlockState) : stepStmts (setup p) (initial p s) = some (start p s) := by
  unfold setup
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BC,p.BC] (.const 0)) _ =
      some (⟨fun _ => some 0⟩ : Tile .real [p.BC,p.BC]) from by rw [evalOp_full,evalOp_const]; rfl)),stepStmts.nil]
  rfl

theorem initial_grid (p : Layout) (s : BlockState) : Grid p (launch s) (initial p s) := by
  constructor <;> simp [initial,launch,rowBlock,columnBlock]

theorem start_holds (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState) : Holds p r scale s 0 (start p s) := by
  refine ⟨Nat.zero_le _,rfl,rfl,?_,?_⟩
  · constructor <;> simp [start,initial,launch,rowBlock,columnBlock]
  · simp [start,accumulator,partialSum,Tile.mat]

theorem index_holds (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (n : Nat) (u : BlockState)
    (hi : Holds p r scale s0 n u) : Holds p r scale s0 n (u.setReg "i_k" .nat [] (Tile.scalar n)) := by
  refine ⟨hi.limit,hi.memory,hi.padding,?_,by simpa using hi.carry⟩
  constructor <;> simp [hi.grid.chunk,hi.grid.head,hi.grid.row,hi.grid.column]

theorem iteration (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (swapped : Bool)
    (hu : ∀ region off, s0.undef region off = 0)
    (n : Nat) (u : BlockState) (hn : n < numBlocks p) (hi : Holds p r scale s0 n u) :
    ∃ v, stepStmts (RealInterProgram.iteration p r scale swapped) (u.setReg "i_k" .nat [] (Tile.scalar n)) = some v ∧
      Holds p r scale s0 (n+1) v := by
  let t := u.setReg "i_k" .nat [] (Tile.scalar n)
  have ht : Holds p r scale s0 n t := index_holds p r scale s0 n u hi
  refine ⟨RealInterStep.result p (launch s0) r (inputs r s0) scale n t,?_,?_⟩
  · exact RealInterStep.executes p (launch s0) r (inputs r s0) scale n swapped t ht.grid (by simp [t])
      (by intro off; simp only [inputs,BlockState.readMem,ht.memory])
      (by intro off; simp only [inputs,BlockState.readMem,ht.memory])
      (by intro off; simp only [inputs,BlockState.readMem,ht.memory])
      (by rw [ht.padding]; exact hu) ht.carry
  · refine ⟨Nat.succ_le_of_lt hn,ht.memory,ht.padding,
      RealInterStep.result_grid p (launch s0) r (inputs r s0) scale n t ht.grid,
      RealInterStep.result_carry p (launch s0) r (inputs r s0) scale n t⟩

theorem tripCount_eval (p : Layout) (u : BlockState) :
    evalOp (tripCount p) u = some (Tile.scalar (numBlocks p)) := by
  simp only [tripCount,evalOp_div,evalOp_sub,evalOp_add,evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem executes (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (swapped : Bool)
    (hu : ∀ region off, s0.undef region off = 0) (u : BlockState) (hi : Holds p r scale s0 0 u) :
    ∃ v, stepStmt (loop p r scale swapped) u = some v ∧ Holds p r scale s0 (numBlocks p) v := by
  obtain ⟨n,v,he,hn,hv⟩ := forRangeDyn_inv (idx := "i_k") (start := 0) (stop := numBlocks p) (step := 1)
    (P := Holds p r scale s0) (body := RealInterProgram.iteration p r scale swapped)
    (evalOp_constNat 0 u) (tripCount_eval p u) (evalOp_constNat 1 u)
    (by decide) hi (iteration p r scale s0 swapped hu)
  have hn' : n = numBlocks p := Nat.le_antisymm hv.limit hn
  subst n
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterLoop
