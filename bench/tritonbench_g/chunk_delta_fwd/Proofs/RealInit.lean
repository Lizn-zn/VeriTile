import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealOuterLoop

/-! Source initialization for both optional-state branches. The seed is an
independent flat input array; the mathematical initial matrix is read using the
host's contiguous K-by-V layout. Register initialization does not change memory. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealInit
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 1,u.pids 2⟩

def base (p : Layout) (l : Launch) : Nat := l.head * p.K * p.V

def seed (p : Layout) (l : Launch) (a : Nat → ℝ) (e j : Nat) : ℝ :=
  a (base p l + e * p.V + j)

def loaded (p : Layout) (l : Launch) (a : Nat → ℝ) : StateValue p := fun e j =>
  if RealStateStore.active p l e.val j.val then
    a (RealStateStore.address p l (base p l) p.V e.val j.val) else 0

def starting (p : Layout) (l : Launch) (a : Nat → ℝ) (useSeed : Bool) : StateValue p :=
  if useSeed then loaded p l a else fun _ _ => 0

def registers (p : Layout) (u : BlockState) : BlockState :=
  let u1 := u.setReg "i_k" .nat [] (Tile.scalar (u.pids 0))
  let u2 := u1.setReg "i_v" .nat [] (Tile.scalar (u.pids 1))
  let u3 := u2.setReg "i_bh" .nat [] (Tile.scalar (u.pids 2))
  u3.setReg "b_h" .real [p.BK,p.BV] (stateTile p (fun _ _ => 0))

def initial (p : Layout) (r : Regions) (a : Nat → ℝ) (useSeed : Bool) (u : BlockState) : BlockState :=
  if useSeed then
    ((registers p u).setReg "p_h0" .blockPtr [p.BK,p.BV]
      (RealStateStore.pointer p (launch u) r.initial (base p (launch u)) p.V)).setReg
      "b_h" .real [p.BK,p.BV] (stateTile p (loaded p (launch u) a))
  else registers p u

theorem load_eval (p : Layout) (l : Launch) (r : Regions) (a : Nat → ℝ) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hm : ∀ off, u.readMem r.initial off = a off)
    (hp : u.regs .blockPtr [p.BK,p.BV] "p_h0" =
      some (RealStateStore.pointer p l r.initial (base p l) p.V)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [p.BK,p.BV] "p_h0") [0,1]) .none .unspecified) u =
      some (stateTile p (loaded p l a)) := by
  rw [load_bp_2d_ref r.initial u "p_h0" (base p l) p.K p.V p.BK p.BV p.V 1
    (l.key * p.BK) (l.value * p.BV) hu hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [stateTile, Tile.mat, loaded, RealStateStore.active, keyIndex, valueIndex,
    RealStateStore.address, hm, Nat.mul_one]
  split <;> rfl

theorem executes (p : Layout) (r : Regions) (a : Nat → ℝ) (useSeed : Bool) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hm : ∀ off, u.readMem r.initial off = a off) :
    stepStmts (cdfPrologue r.initial p.K p.V p.BK p.BV useSeed) u =
      some (initial p r a useSeed u) := by
  unfold cdfPrologue
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) _ =
    some (stateTile p (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  cases useSeed
  · rw [stepStmts.cons_some (cdf_ifThen_false_noop _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [cdf_ifThen_true_run]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (makeBlockPtr_2d_eval r.initial _ _ _ _ [p.K,p.V] [p.BK,p.BV] [p.V,1]
          (base p (launch u)) ((launch u).key * p.BK) ((launch u).value * p.BV)
          (mulMulConst_eval _ "i_bh" (u.pids 2) p.K p.V (by simp))
          (mulConst_eval _ "i_k" (u.pids 0) p.BK (by simp))
          (mulConst_eval _ "i_v" (u.pids 1) p.BV (by simp))))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_eval p (launch u) r a _ (by simpa only [BlockState.setReg] using hu)
        (by intro off; simpa using hm off) (by simp [RealStateStore.pointer]))),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem represents (p : Layout) (l : Launch) (a : Inputs) (a₀ : Nat → ℝ) (useSeed : Bool)
    (hk : l.key = 0) : RealValues.Represents p l (starting p l a₀ useSeed)
      (RealWindow.expected p l a (seed p l a₀) useSeed 0) := by
  intro e j he hj
  have ha : RealStateStore.active p l e.val j.val := by
    simpa [RealStateStore.active,keyIndex,hk] using And.intro he hj
  cases useSeed <;>
    simp only [starting, Bool.false_eq_true, Bool.true_eq, ite_false, ite_true, loaded, ha,
      RealWindow.expected, RealWindow.state, RealValues.arrays, seed,
      RealStateStore.address, keyIndex, hk, Nat.zero_mul, Nat.zero_add]

theorem invariant (p : Layout) (r : Regions) (a : Inputs) (a₀ : Nat → ℝ)
    (useSeed : Bool) (u : BlockState) (hk : u.pids 0 = 0)
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hdelta : ∀ off, u.readMem r.delta off = a.delta off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off) :
    RealOuterLoop.Holds u p (launch u) r a (seed p (launch u) a₀) useSeed 0 (initial p r a₀ useSeed u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,⟨starting p (launch u) a₀ useSeed,?_,?_⟩,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · cases useSeed <;> simp [initial,registers,launch]
  · cases useSeed <;> simp [initial,registers,launch]
  · cases useSeed <;> simp [initial,registers,launch]
  · cases useSeed <;> simp [initial,registers,starting]
  · exact represents p (launch u) a a₀ useSeed hk
  · intro off; cases useSeed <;> simpa [initial,registers] using hkey off
  · intro off; cases useSeed <;> simpa [initial,registers] using hdelta off
  · intro off; cases useSeed <;> simpa [initial,registers] using hvalue off
  · intro q _ _ hq; omega
  · intro q _ _ hq; omega
  · intro _ _ _; cases useSeed <;> rfl
  · cases useSeed <;> rfl
  · cases useSeed <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealInit
