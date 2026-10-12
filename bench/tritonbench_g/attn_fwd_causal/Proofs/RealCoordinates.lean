import bench.tritonbench_g.attn_fwd_causal.Proofs.RealMiddle

/-! The eight leading coordinate/base assignments. Source integer address
arithmetic uses the DSL's Nat channel; launch bounds and address validity are
separate contracts. The unused vk_offset assignment is retained and executed. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealCoordinates
open VeriTile.Triton
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000

def ceil (N B : Nat) : Nat := (N+B-1)/B
def ceilExpr (N B : Nat) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat N) (.constNat B)) (.constNat 1)) (.constNat B)

def code (H N BM BN strideZ strideH strideM : Nat) : List Stmt :=
  [.assign .nat [] "start_m" (.programId 0),
   .assign .nat [] "off_hz" (.programId 1),
   .assign .nat [] "off_z" (.floorDiv .nat .nil (.ref .nat [] "off_hz") (.constNat H)),
   .assign .nat [] "off_h" (.mod .nat .nil (.ref .nat [] "off_hz") (.constNat H)),
   .assign .nat [] "qvk_offset" (.add .nat .nil
     (.mul .nat .nil (.ref .nat [] "off_z") (.constNat strideZ))
     (.mul .nat .nil (.ref .nat [] "off_h") (.constNat strideH))),
   .assign .nat [] "vk_offset" (.floorDiv .nat .nil (.ref .nat [] "qvk_offset") (.constNat strideM)),
   .assign .nat [] "q_scale_offset" (.mul .nat .nil (.ref .nat [] "off_hz") (ceilExpr N BM)),
   .assign .nat [] "k_scale_offset" (.mul .nat .nil (.ref .nat [] "off_hz") (ceilExpr N BN))]

def writes : List RegName := ["start_m","off_hz","off_z","off_h","qvk_offset","vk_offset","q_scale_offset","k_scale_offset"]

structure Values where
  (row headBatch batch head base valueBase queryScaleBase keyScaleBase : Nat)

def values (s : BlockState) (H N BM BN strideZ strideH strideM : Nat) : Values where
  row := s.pids 0
  headBatch := s.pids 1
  batch := s.pids 1/H
  head := s.pids 1%H
  base := (s.pids 1/H)*strideZ+(s.pids 1%H)*strideH
  valueBase := ((s.pids 1/H)*strideZ+(s.pids 1%H)*strideH)/strideM
  queryScaleBase := s.pids 1*ceil N BM
  keyScaleBase := s.pids 1*ceil N BN

structure Registers (v : Values) (s : BlockState) : Prop where
  row : s.regs .nat [] "start_m" = some (Tile.scalar v.row)
  headBatch : s.regs .nat [] "off_hz" = some (Tile.scalar v.headBatch)
  batch : s.regs .nat [] "off_z" = some (Tile.scalar v.batch)
  head : s.regs .nat [] "off_h" = some (Tile.scalar v.head)
  base : s.regs .nat [] "qvk_offset" = some (Tile.scalar v.base)
  valueBase : s.regs .nat [] "vk_offset" = some (Tile.scalar v.valueBase)
  queryScaleBase : s.regs .nat [] "q_scale_offset" = some (Tile.scalar v.queryScaleBase)
  keyScaleBase : s.regs .nat [] "k_scale_offset" = some (Tile.scalar v.keyScaleBase)

theorem executes (s : BlockState) (H N BM BN strideZ strideH strideM : Nat) :
    ∃ t, stepStmts (code H N BM BN strideZ strideH strideM) s = some t ∧
      Registers (values s H N BM BN strideZ strideH strideM) t ∧ Frame writes s t := by
  let v := values s H N BM BN strideZ strideH strideM
  let s1 := s.setReg "start_m" .nat [] (Tile.scalar v.row)
  let s2 := s1.setReg "off_hz" .nat [] (Tile.scalar v.headBatch)
  let s3 := s2.setReg "off_z" .nat [] (Tile.scalar v.batch)
  let s4 := s3.setReg "off_h" .nat [] (Tile.scalar v.head)
  let s5 := s4.setReg "qvk_offset" .nat [] (Tile.scalar v.base)
  let s6 := s5.setReg "vk_offset" .nat [] (Tile.scalar v.valueBase)
  let s7 := s6.setReg "q_scale_offset" .nat [] (Tile.scalar v.queryScaleBase)
  have hz : evalOp (.floorDiv .nat .nil (.ref .nat [] "off_hz") (.constNat H)) s2 = some (Tile.scalar v.batch) := by
    simp [evalOp.eq_def, s2, v, values]; rfl
  have hh : evalOp (.mod .nat .nil (.ref .nat [] "off_hz") (.constNat H)) s3 = some (Tile.scalar v.head) := by
    simp [evalOp.eq_def, s3, s2, v, values]; rfl
  have hb : evalOp (.add .nat .nil
      (.mul .nat .nil (.ref .nat [] "off_z") (.constNat strideZ))
      (.mul .nat .nil (.ref .nat [] "off_h") (.constNat strideH))) s4 = some (Tile.scalar v.base) := by
    simp [evalOp.eq_def, s4, s3, v, values]; rfl
  have hv : evalOp (.floorDiv .nat .nil (.ref .nat [] "qvk_offset") (.constNat strideM)) s5 = some (Tile.scalar v.valueBase) := by
    simp [evalOp.eq_def, s5, v, values]; rfl
  have hq : evalOp (.mul .nat .nil (.ref .nat [] "off_hz") (ceilExpr N BM)) s6 = some (Tile.scalar v.queryScaleBase) := by
    simp [evalOp.eq_def, ceilExpr, s6, s5, s4, s3, s2, v, values]; rfl
  have hk : evalOp (.mul .nat .nil (.ref .nat [] "off_hz") (ceilExpr N BN)) s7 = some (Tile.scalar v.keyScaleBase) := by
    simp [evalOp.eq_def, ceilExpr, s7, s6, s5, s4, s3, s2, v, values]; rfl
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.programId 0) s = some (Tile.scalar v.row) from by simp [v, values])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.programId 1) s1 = some (Tile.scalar v.headBatch) from by simp [s1, v, values])),
    stepStmts.cons_some (stepStmt_assign_eq_some hz), stepStmts.cons_some (stepStmt_assign_eq_some hh),
    stepStmts.cons_some (stepStmt_assign_eq_some hb), stepStmts.cons_some (stepStmt_assign_eq_some hv),
    stepStmts.cons_some (stepStmt_assign_eq_some hq), stepStmts.cons_some (stepStmt_assign_eq_some hk), stepStmts.nil]
  refine ⟨_, rfl, ?_, rfl, rfl, rfl, ?_⟩
  · constructor <;> simp [s7, s6, s5, s4, s3, s2, s1, v]
  · intro dt shape name hn
    simp only [writes, List.mem_cons, not_or] at hn
    simp [s7, s6, s5, s4, s3, s2, s1, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1,
      hn.2.2.2.2.1, hn.2.2.2.2.2.1, hn.2.2.2.2.2.2.1, hn.2.2.2.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealCoordinates
