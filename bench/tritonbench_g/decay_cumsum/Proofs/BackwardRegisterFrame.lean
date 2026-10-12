import bench.tritonbench_g.decay_cumsum.Proofs.RealReference
import bench.tritonbench_g.decay_cumsum.Proofs.RegisterFrame
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardRegisterFrame
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

def setOffs (BK : Nat) (s : BlockState) : BlockState :=
  s.setReg "offs" .nat [BK] (RealReference.prepOffsTile BK)

/-- The cached arange register is dead inside the reverse loop. Adding it
before the loop only adds the same unused register to the final state. -/
theorem body (DK BT BK : Nat)
    (s : BlockState) :
    stepStmts (RealReference.bwdIterBodyG DK BT BK) (setOffs BK s) =
      (stepStmts (RealReference.bwdIterBodyG DK BT BK) s).map (setOffs BK) := by
  apply RegisterFrame.list
  intro st hs
  simp only [RealReference.bwdIterBodyG,List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first
    | apply RegisterFrame.assign
      · intro u
        simp [setOffs,ComputeDType.eraseDType,evalOp]
      · intro u v
        exact RegisterFrame.setReg_comm u "offs" _ (by decide) _ _ _ _ _ _
    | apply RegisterFrame.store_ptr_mask
      · intro u; simp [setOffs,ComputeDType.eraseDType,evalOp]
      · intro u; simp [setOffs,ComputeDType.eraseDType,evalOp]
      · intro u; simp [setOffs,ComputeDType.eraseDType,evalOp]
      · intro u r o x
        exact RegisterFrame.setReg_write u "offs" _ _ _ _ r o x

    | apply RegisterFrame.branch
      · intro u; simp [setOffs,ComputeDType.eraseDType,evalOp]
      · intro u
        apply RegisterFrame.list
        intro st hst
        simp only [List.mem_cons,List.not_mem_nil,or_false] at hst
        subst st
        apply RegisterFrame.assign
        · intro v; simp [setOffs,evalOp]
        · intro v x
          exact RegisterFrame.setReg_comm v "offs" _ (by decide) _ _ _ _ _ _

theorem loop (DK BT BK start stop : Nat) (s : BlockState) :
    stepForRangeAux "__rev_j" start stop 1 (RealReference.bwdIterBodyG DK BT BK)
      (setOffs BK s) =
    (stepForRangeAux "__rev_j" start stop 1 (RealReference.bwdIterBodyG DK BT BK) s).map (setOffs BK) := by
  apply RegisterFrame.range
  · exact body DK BT BK
  · intro u n
    exact RegisterFrame.setReg_comm u "offs" "__rev_j" (by decide) _ _ _ _ _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardRegisterFrame
