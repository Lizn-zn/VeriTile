import bench.tritonbench_g.attn_fwd_causal.Proofs.RealPointers

/-! Load the actual query and its scale and initialize the online carry.
Masked query lanes use the explicit undef=0 real execution contract. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealInit
open VeriTile.Triton RealLoads RealPointers
open AttentionFwdTriton3.RealTiles (finite)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def queryMask (BM BD N HA : Nat) : Op .bool [BM,BD] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")) (.constNat N))
    (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat HA)))

def code (BM BD N HA : Nat) : List Stmt :=
  [.assign .real [BM] "m_i" (.add .real .scalarR (.full [BM] (.const 0)) .negInf),
   .assign .real [BM] "l_i" (.add .real .scalarR (.full [BM] (.const 0)) (.const 1.0)),
   .assign .real [BM,BD] "acc" (.full [BM,BD] (.const 0)),
   .assign .real [BM,BD] "q" (.load .real (.ptr (.ref .ptr [BM,BD] "Q_ptrs")) (.mask (queryMask BM BD N HA))),
   .assign .real [] "q_scale" (.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none)]

def mask (BM BD startM N HA : Nat) : Tile .bool [BM,BD] :=
  ⟨fun i => decide (startM*BM+i.1.val < N ∧ i.2.1.val < HA)⟩

theorem eval_mask (s : BlockState) (BM BD startM N HA : Nat)
    (hr : s.regs .nat [BM] "offs_m" = some (rows BM (startM*BM))) :
    evalOp (queryMask BM BD N HA) s = some (mask BM BD startM N HA) := by
  simp only [queryMask, evalOp.eq_def, hr, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,d,⟨⟩⟩
  simp [mask, rows, Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.scalar,
    TileShape.dropInsertedIndex, ComparableDType.lt, -Tile.scalar_eta]
  rfl

theorem query_load (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hp : s.regs .ptr [BM,BD] "Q_ptrs" = some (queryPointer c BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some (rows BM (c.startM*BM)))
    (hu : ∀ r a, s.undef r a = 0) :
    evalOp (.load .real (.ptr (.ref .ptr [BM,BD] "Q_ptrs")) (.mask (queryMask BM BD c.context c.activeHead))) s =
      some (RealTiles.query (inputs s c BM BN BD) c.startM c.context c.activeHead) := by
  have hmask := eval_mask s BM BD c.startM c.context c.activeHead hr
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hmask, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,d,⟨⟩⟩
  simp [mask, queryPointer, rowPointer, RealTiles.query, inputs, finite, hu, Region.cast]
  split <;> simp_all

theorem scale_load (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hp : s.regs .ptr [] "Q_scale_ptr" = some (queryScalePointer c)) :
    evalOp (.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) s =
      some (Tile.scalar (some (inputs s c BM BN BD).queryScale)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [queryScalePointer, inputs, Tile.scalar, Region.cast, -Tile.scalar_eta]

def writes : List RegName := ["m_i","l_i","acc","q","q_scale"]

theorem executes (s : BlockState) (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (hp : RealPointers.Registers c Out BM BN BD s) (hu : ∀ r a, s.undef r a = 0) :
    ∃ t, stepStmts (code BM BD c.context c.activeHead) s = some t ∧
      RealMiddle.Entry (inputs s c BM BN BD) c t ∧
      t.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c Out BM BD) ∧ Frame writes s t := by
  let sm := s.setReg "m_i" .real [BM] (⟨fun _ => ⊥⟩ : Tile .real [BM])
  let sl := sm.setReg "l_i" .real [BM] (finite (fun _ => 1))
  let sa := sl.setReg "acc" .real [BM,BD] (finite (fun _ => 0))
  let sq := sa.setReg "q" .real [BM,BD] (RealTiles.query (inputs s c BM BN BD) c.startM c.context c.activeHead)
  have hm : evalOp (.add .real .scalarR (.full [BM] (.const 0)) .negInf) s =
      some (⟨fun _ => ⊥⟩ : Tile .real [BM]) := by
    simp only [evalOp.eq_def, bind, Option.bind]
    rfl
  have hl : evalOp (.add .real .scalarR (.full [BM] (.const 0)) (.const 1.0)) sm =
      some (finite (fun _ : TileIndex [BM] => 1)) := by
    simp only [evalOp.eq_def, bind, Option.bind]
    congr 1
    apply Tile.ext
    intro i
    norm_num [Tile.bop_data, Tile.scalar, NumericDType.add, WithBot.realAdd, finite, -Tile.scalar_eta]
  have ha : evalOp (.full [BM,BD] (.const 0)) sl = some (finite (fun _ => 0)) := by
    simp only [evalOp.eq_def, bind, Option.bind]
    rfl
  have hq := query_load sa c BM BN BD (by simp [sa, sl, sm, hp.query])
    (by simp [sa, sl, sm, hp.rows]) (by simpa [sa, sl, sm] using hu)
  rw [inputs_same_memory sa s c BM BN BD rfl] at hq
  have hqs := scale_load sq c BM BN BD (by simp [sq, sa, sl, sm, hp.queryScale])
  rw [inputs_same_memory sq s c BM BN BD rfl] at hqs
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hm), stepStmts.cons_some (stepStmt_assign_eq_some hl),
    stepStmts.cons_some (stepStmt_assign_eq_some ha), stepStmts.cons_some (stepStmt_assign_eq_some hq),
    stepStmts.cons_some (stepStmt_assign_eq_some hqs), stepStmts.nil]
  refine ⟨_, rfl, ?_, by simp [sq, sa, sl, sm, hp.output], rfl, rfl, rfl, ?_⟩
  · refine ⟨by simp [sq], by simp, by simpa [sq, sa, sl, sm] using hp.rows,
      by simp [sq, sa, sl, sm, hp.keys], ⟨?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, ?_⟩
    · simp [sq, sa, sl, sm, RealLoopFormula.initial]
    · simp [sq, sa, sl, RealLoopFormula.initial]
    · simp [sq, sa, RealLoopFormula.initial]
    · simp [sq, sa, sl, sm, hp.caller.row]
    · simp [sq, sa, sl, sm, hp.caller.key]
    · simp [sq, sa, sl, sm, hp.caller.scale]
    · simp [sq, sa, sl, sm, hp.caller.value]
    · simpa [sq, sa, sl, sm] using hu
  · intro dt shape name hn
    simp only [writes, List.mem_cons, not_or] at hn
    simp [sq, sa, sl, sm, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1, hn.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealInit
