import bench.tritonbench_g.attention_score.Proofs.RealProgram
import bench.tritonbench_g.attention_score.Proofs.RealLoop

/-! Initialize the query-loop invariant from actual program coordinates and
initial input memory. Grouped-head addresses are mathematical Nat addresses;
legal launch divisors and physical allocation bounds belong to the IO contract. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealPrelude
open VeriTile.Triton RealFrame
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

def config (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool) : RealLoop.Config where
  Q := Q
  K := K
  M := M
  B := B
  D := D
  N := N
  NK := NK
  baseQ := s.pids 1 / H * sqz + s.pids 1 % H * sqh
  baseK := s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh
  baseM := s.pids 1 * RN
  sqm := sqm
  sqk := sqk
  skn := skn
  skk := skk
  startN := s.pids 0
  offset := offset
  size := size
  scale := scale
  window := window
  complement := complement
  evenM := evenM
  evenN := evenN
  candidate := candidate

private theorem eval_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.floorDiv bc vx vy)) := by rw [evalOp]

private theorem eval_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.mod bc vx vy)) := by rw [evalOp]

private theorem eval_ptrAdd {a b shape} (bc : Broadcast a b shape)
    (x : Op .ptr a) (y : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.ptrAdd bc vx vy)) := by rw [evalOp]

private theorem eval_ptrBase (r : RegionName) (s : BlockState) :
    evalOp (.ptrBase r) s = some (Tile.scalar (r.cast,0)) := by rw [evalOp]

private theorem scalarBop {dtype : TileDType}
    (f : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (a b : TileCarrier dtype) : Tile.bop f Broadcast.nil (Tile.scalar a) (Tile.scalar b) =
      Tile.scalar (f a b) := rfl

private theorem makePointer (r : RegionName) (base : Op .nat [])
    (parent : List Nat) (block : TileShape) (strides : List Nat) (offsets : List Nat)
    (s : BlockState) (value : Nat) (he : evalOp base s = some (Tile.scalar value)) :
    evalOp (.makeBlockPtrDyn r base parent block strides offsets) s = some
      (⟨fun _ => { region := r, baseOffset := value, parentShape := parent, blockShape := block, strides := strides, offsets := offsets }⟩ : Tile .blockPtr block) := by
  rw [evalOp]
  simp only [he, bind, Option.bind]
  rfl

/-- Initialize every loop-carried register from program coordinates and the
actual K load. No register contents or output equations are assumed. -/
theorem correct (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool) :
    ∃ t, stepStmts (RealProgram.prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) s = some t ∧
      RealLoop.Invariant (config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
        offset size scale window complement evenM evenN candidate) s 0 t ∧
      t.regs .nat [] "lo" = some (Tile.scalar 0) ∧
      t.regs .nat [] "hi" = some (Tile.scalar RN) ∧
      t.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H)) ∧
      t.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H)) ∧
      t.regs .blockPtr [D,B] "K_block_ptr" = some (RealLoads.keyPointer K B D NK
        (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh) skn skk (s.pids 0)) ∧
      Frame RealProgram.writes s t := by
  suffices he : ∃ t, stepStmts (RealProgram.prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) s = some t ∧
      RealLoop.Invariant (config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
        offset size scale window complement evenM evenN candidate) s 0 t ∧
      t.regs .nat [] "lo" = some (Tile.scalar 0) ∧
      t.regs .nat [] "hi" = some (Tile.scalar RN) ∧
      t.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H)) ∧
      t.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H)) ∧
      t.regs .blockPtr [D,B] "K_block_ptr" = some (RealLoads.keyPointer K B D NK
        (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh) skn skk (s.pids 0)) by
    obtain ⟨t,ht,hi,hlo,hhi,hz,hh,hk⟩ := he
    exact ⟨t,ht,hi,hlo,hhi,hz,hh,hk,(RealProgram.prelude_code Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN).frame ht⟩
  unfold RealProgram.prelude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 / H)) from by
      rw [eval_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 % H)) from by
      rw [eval_mod]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat H) (Op.constNat HK))) _
        = some (Tile.scalar ((s.pids 1 % H) / (H / HK))) from by
      rw [eval_floorDiv, eval_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rw [scalarBop, scalarBop]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat sqz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat sqh))) _
        = some (Tile.scalar ((s.pids 1 / H * sqz + s.pids 1 % H * sqh))) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat skz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hkv") (Op.constNat skh))) _
        = some (Tile.scalar ((s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh))) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat RN))
          (Op.arange B))) _
        = some (⟨fun idx : TileIndex [B] => (M.cast, s.pids 1 * RN + idx.1.val)⟩ : Tile .ptr [B]) from by
      rw [eval_ptrAdd, evalOp_add, evalOp_mul]
      simp only [eval_ptrBase, evalOp_constNat, evalOp_ref, evalOp_arange,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.vec, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul, BlockState.setReg_pids]
        omega))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [B] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [B] => some (0 : ℝ)⟩ : Tile .real [B]) from by
      simp [evalOp_full]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makePointer Q (Op.ref .nat [] "q_offset") [N, D] [B, D] [sqm, sqk] [0, 0] _
      ((s.pids 1 / H * sqz + s.pids 1 % H * sqh)) (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets K (Op.ref .nat [] "k_offset") [D, NK] [D, B] [skk, skn]
        [Op.constNat 0, Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat B)]) _
        = some (⟨fun _ : TileIndex [D, B] =>
            { region := K, baseOffset := (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh), parentShape := [D, NK],
              blockShape := [D, B], strides := [skk, skn], offsets := [0, s.pids 0 * B] }⟩
            : Tile .blockPtr [D, B]) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_ref, evalOp_constNat, evalOp_mul, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind,
        Option.bind_some, List.mapM_cons, List.mapM_nil, scalarBop]
      refine congrArg some ?_; ext idx <;> rfl))]
  rw [stepStmts.cons_some (RealIteration.load_key _ Q K M B D NK
    ((s.pids 1 / H * sqz + s.pids 1 % H * sqh)) ((s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh))
    (s.pids 1 * RN) sqm sqk skn skk (s.pids 0) evenN (by simp [RealLoads.keyPointer]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat 0) _ = _ from evalOp_constNat 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat RN) _ = _ from evalOp_constNat RN _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.const scale) _ = _ from evalOp_const scale _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.ref .real [] "qk_scale")
        (Op.const 1.4426950408889634)) _
        = some (Tile.scalar (some (scale * 1.4426950408889634))) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_const, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some, scalarBop]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, by simp, by simp, by simp, by simp, by simp [RealLoads.keyPointer]⟩
  constructor
  all_goals simp [config, RealLoop.Config.inputs, RealLoop.Config.column, RealFormula.column,
    RealLoads.inputs, RealLoads.queryPointer, RealLoads.normalizerPointers, RealTiles.key]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealPrelude
