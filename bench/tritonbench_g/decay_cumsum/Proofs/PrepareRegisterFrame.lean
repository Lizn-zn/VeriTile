import bench.tritonbench_g.decay_cumsum.Proofs.RealReference
import bench.tritonbench_g.decay_cumsum.Proofs.RegisterFrame
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareRegisterFrame
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

def setOffs (BK : Nat) (s : BlockState) : BlockState :=
  s.setReg "offs" .nat [BK] (RealReference.prepOffsTile BK)

/-- The cached arange register is dead inside the preparation loop. Adding it
before the loop only adds the same unused register to the final state. -/
theorem body (Q K G QG KG : RegionName) (sH DK BT BK : Nat) (scale : ℝ)
    (s : BlockState) :
    stepStmts (RealReference.prepBody Q K G QG KG sH DK BT BK scale) (setOffs BK s) =
      (stepStmts (RealReference.prepBody Q K G QG KG sH DK BT BK scale) s).map (setOffs BK) := by
  apply RegisterFrame.list
  intro st hs
  simp only [RealReference.prepBody,List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
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

theorem loop (Q K G QG KG : RegionName) (sH DK BT BK start : Nat)
    (scale : ℝ) (s : BlockState) :
    stepForRangeAux "i" start BT 1 (RealReference.prepBody Q K G QG KG sH DK BT BK scale)
      (setOffs BK s) =
    (stepForRangeAux "i" start BT 1 (RealReference.prepBody Q K G QG KG sH DK BT BK scale) s).map (setOffs BK) := by
  apply RegisterFrame.range
  · exact body Q K G QG KG sH DK BT BK scale
  · intro u n
    exact RegisterFrame.setReg_comm u "offs" "i" (by decide) _ _ _ _ _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareRegisterFrame
