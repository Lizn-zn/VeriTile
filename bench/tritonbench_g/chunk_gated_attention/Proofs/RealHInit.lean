import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHLoop

/-! The original initialization, including zero-plus-seed rather than a seed
assignment. Optional seed storage is read before any output write, so it need
not be disjoint from output storage. Launch order is value, key, head. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHInit
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def registers (p : Layout) (u : BlockState) : BlockState :=
  (((u.setReg "i_v" .nat [] (Tile.scalar (u.pids 0))).setReg
    "i_k" .nat [] (Tile.scalar (u.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (u.pids 2))).setReg
    "b_h" .real [p.BK,p.BV] (stateTile p (fun _ _ => 0))

def starting (p : Layout) (l : Launch) (a padding : Nat → ℝ) (hasSeed : Bool) : StateValue p :=
  if hasSeed then seedData p l a padding else fun _ _ => 0

def initial (p : Layout) (r : Regions) (a padding : Nat → ℝ)
    (hasSeed : Bool) (u : BlockState) : BlockState :=
  if hasSeed then
    ((registers p u).setReg "p_h" .blockPtr [p.BK,p.BV] (statePointer p (launch u) r.seed)).setReg
      "b_h" .real [p.BK,p.BV] (stateTile p (seedData p (launch u) a padding))
  else registers p u

theorem if_true (body : List Stmt) (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.true) body) u = stepStmts body u := by
  simp only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,↓reduceIte]

theorem if_false (body : List Stmt) (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.false) body) u = some u := by
  simp only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,Bool.false_eq_true,↓reduceIte]

theorem add_seed (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ) (u : BlockState)
    (hu : u.undef r = padding) (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BK,p.BV] "p_h" = some (statePointer p l r))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (fun _ _ => 0))) :
    evalOp (.add .real (.consSame (.consSame .nil)) (carry p) (load2 p.BK p.BV "p_h")) u =
      some (stateTile p (seedData p l a padding)) := by
  rw [evalOp_add,carry,evalOp_ref,hh,seed_load p l r a padding "p_h" u hm hu hp]
  simp only [Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

theorem executes (p : Layout) (r : Regions) (a padding : Nat → ℝ)
    (hasSeed : Bool) (u : BlockState) (hu : u.undef r.seed = padding)
    (hm : ∀ off, u.readMem r.seed off = a off) :
    stepStmts (initialization p r hasSeed) u = some (initial p r a padding hasSeed u) := by
  unfold initialization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) _ =
    some (stateTile p (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  cases hasSeed
  · rw [stepStmts.cons_some (if_false _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (state_eval p (launch u) r.seed _
        (by simp [launch]) (by simp [launch]) (by simp [launch])))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (add_seed p (launch u) r.seed a padding _
        (by simpa only [BlockState.setReg] using hu) (by simpa using hm) (by simp) (by simp))),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem starting_expected (p : Layout) (l : Launch) (a : Inputs) (gateK hasSeed : Bool) :
    starting p l a.seed (fun _ => 0) hasSeed = expected p l a gateK hasSeed 0 := by
  funext e j
  cases hasSeed <;> simp [starting,expected,seedData,RealHMath.seed]

theorem invariant (p : Layout) (r : Regions) (a : Inputs) (gateK hasSeed : Bool) (u : BlockState)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off) :
    RealHLoop.Holds u p (launch u) r a gateK hasSeed 0 (initial p r a.seed (fun _ => 0) hasSeed u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · cases hasSeed <;> simp [initial,registers,launch]
  · cases hasSeed <;> simp [initial,registers,launch]
  · cases hasSeed <;> simp [initial,registers,launch]
  · rw [← starting_expected]
    cases hasSeed <;> simp [initial,registers,starting]
  · intro off; cases hasSeed <;> simpa [initial,registers] using hk off
  · intro off; cases hasSeed <;> simpa [initial,registers] using hv off
  · intro off; cases hasSeed <;> simpa [initial,registers] using hg off
  · intro q _ _ hq; omega
  · intro _ _ _; cases hasSeed <;> rfl
  · cases hasSeed <;> rfl
  · cases hasSeed <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHInit
