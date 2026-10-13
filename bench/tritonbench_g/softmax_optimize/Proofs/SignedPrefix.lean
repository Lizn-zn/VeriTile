import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies
import bench.tritonbench_g.softmax_optimize.Proofs.SignedIndexed

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def columns (B i : Nat) : Tile .nat [B] := ⟨fun lane ↦ i+lane.1.val⟩
def offsets (N B i p : Nat) : Tile .nat [B] := ⟨fun lane ↦ p*N+(i+lane.1.val)⟩
def pointers (X : RegionName) (N B i p : Nat) : Tile .ptr [B] :=
  ⟨fun lane ↦ (X,p*N+(i+lane.1.val))⟩

def prefixIntState (X : RegionName) (N B i p : Nat) (s : BlockState) : BlockState :=
  (((s.setReg "start_n" .int [] (Tile.scalar (i:Int))).setReg "n_offsets" .int [B]
    (castIndex (columns B i))).setReg "offset" .int [B] (castIndex (offsets N B i p))).setReg
    "input_ptrs" .ptr [B] (pointers X N B i p)

def prefixNatState (X : RegionName) (N B i p : Nat) (s : BlockState) : BlockState :=
  (((s.setReg "start_n" .nat [] (Tile.scalar i)).setReg "n_offsets" .nat [B]
    (columns B i)).setReg "offset" .nat [B] (offsets N B i p)).setReg
    "input_ptrs" .ptr [B] (pointers X N B i p)

def naturalPrefix (X : RegionName) (N B : Nat) : List Stmt := (loopAbody X N B).take 3

theorem prefix_nat_run (X : RegionName) (N B i p : Nat) (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p)) :
    stepStmts (naturalPrefix X N B) (s.setReg "start_n" .nat [] (Tile.scalar i)) =
      some (prefixNatState X N B i p s) := by
  simp only [naturalPrefix, loopAbody, List.take_succ_cons, List.take_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := columns B i) (by
    rw [evalOp_add, evalOp_ref_setReg_same, evalOp_arange]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := offsets N B i p) (by
    rw [evalOp_add, evalOp_mul, evalOp_constNat]
    simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same, hpid,
      ne_eq, String.reduceEq, not_false_eq_true]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := pointers X N B i p) (by
    simp only [evalOp.eq_def, BlockState.setReg_same, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (X,0+(p*N+(i+lane.1.val))) = (X,p*N+(i+lane.1.val))
    rw [Nat.zero_add]))]
  rw [stepStmts.nil]
  rfl

theorem prefix_int_run (X : RegionName) (N B i p : Nat) (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p)) :
    stepStmts (indexPrefix X N B) (s.setReg "start_n" .int [] (Tile.scalar (i:Int))) =
      some (prefixIntState X N B i p s) := by
  simp only [indexPrefix]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := castIndex (columns B i)) (by
    simp only [evalOp.eq_def, BlockState.setReg_same, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (i:Int)+(lane.1.val:Int) = ((i+lane.1.val:Nat):Int)
    exact (Int.natCast_add _ _).symm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := castIndex (offsets N B i p)) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same, hpid,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (p:Int)*(N:Int)+((i+lane.1.val:Nat):Int) = ((p*N+(i+lane.1.val):Nat):Int)
    simp only [Int.natCast_add, Int.natCast_mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := pointers X N B i p) (by
    simp only [evalOp.eq_def, BlockState.setReg_same, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (X,0+Int.toNat ((p*N+(i+lane.1.val):Nat):Int)) = (X,p*N+(i+lane.1.val))
    rw [Int.toNat_natCast, Nat.zero_add]))]
  rw [stepStmts.nil]
  rfl

theorem prefix_states_related (X : RegionName) (N B i p : Nat) {s t : BlockState}
    (h : LiveRelated s t) : IndexedRelated B
      (prefixIntState X N B i p s) (prefixNatState X N B i p t) := by
  have h0 := h.setScratch "start_n" .int .nat [] []
    (Tile.scalar (i:Int)) (Tile.scalar i) (Or.inl rfl)
  have h1 := h0.setScratch "n_offsets" .int .nat [B] [B]
    (castIndex (columns B i)) (columns B i) (Or.inr (Or.inl rfl))
  have h2 := h1.setScratch "offset" .int .nat [B] [B]
    (castIndex (offsets N B i p)) (offsets N B i p) (Or.inr (Or.inr rfl))
  refine ⟨h2.setShared "input_ptrs" .ptr [B] (pointers X N B i p) (by simp [Shared, Scratch]), ?_, ?_⟩
  · simp only [prefixIntState, prefixNatState, BlockState.setReg_ne_name,
      BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, Option.map_some]
  · simp only [prefixIntState, prefixNatState, BlockState.setReg_ne_name,
      BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, Option.map_some]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
