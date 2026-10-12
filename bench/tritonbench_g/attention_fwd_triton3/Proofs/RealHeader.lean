import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealWindowSource

/-! Execute source program coordinates and grouped-head base offsets. Arithmetic
uses mathematical naturals; the DSL canonicalizes these nonnegative int64
casts into its Nat address channel; valid launch divisors and fixed-width overflow are
separate source-contract concerns. No memory or floating values are accessed. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealHeader
open VeriTile.Triton RealScores

structure Coordinates where
  startM : Nat
  headBatch : Nat
  batch : Nat
  head : Nat
  kvHead : Nat

def coordinates (s : BlockState) (heads kvHeads : Nat) : Coordinates where
  startM := s.pids 0
  headBatch := s.pids 1
  batch := s.pids 1 / heads
  head := s.pids 1 % heads
  kvHead := (s.pids 1 % heads) / (heads / kvHeads)

def coordinateCode (heads kvHeads : Nat) : List Stmt :=
  [.assign .nat [] "start_m" (.programId 0),
   .assign .nat [] "off_hz" (.programId 1),
   .assign .nat [] "off_z" (.floorDiv .nat .nil (.ref .nat [] "off_hz") (.constNat heads)),
   .assign .nat [] "off_h" (.mod .nat .nil (.ref .nat [] "off_hz") (.constNat heads)),
   .assign .nat [] "off_hkv" (.floorDiv .nat .nil (.ref .nat [] "off_h")
     (.floorDiv .nat .nil (.constNat heads) (.constNat kvHeads)))]

def coordinateWrites : List RegName := ["start_m", "off_hz", "off_z", "off_h", "off_hkv"]

structure CoordinateRegisters (c : Coordinates) (s : BlockState) : Prop where
  startM : s.regs .nat [] "start_m" = some (Tile.scalar c.startM)
  headBatch : s.regs .nat [] "off_hz" = some (Tile.scalar c.headBatch)
  batch : s.regs .nat [] "off_z" = some (Tile.scalar c.batch)
  head : s.regs .nat [] "off_h" = some (Tile.scalar c.head)
  kvHead : s.regs .nat [] "off_hkv" = some (Tile.scalar c.kvHead)

theorem coordinate_executes (s : BlockState) (heads kvHeads : Nat) :
    ∃ s', stepStmts (coordinateCode heads kvHeads) s = some s' ∧
      CoordinateRegisters (coordinates s heads kvHeads) s' ∧ Frame coordinateWrites s s' := by
  let c := coordinates s heads kvHeads
  let s1 := s.setReg "start_m" .nat [] (Tile.scalar c.startM)
  let s2 := s1.setReg "off_hz" .nat [] (Tile.scalar c.headBatch)
  let s3 := s2.setReg "off_z" .nat [] (Tile.scalar c.batch)
  let s4 := s3.setReg "off_h" .nat [] (Tile.scalar c.head)
  have ez : evalOp (.floorDiv .nat .nil (.ref .nat [] "off_hz") (.constNat heads)) s2 =
      some (Tile.scalar c.batch) := by simp [evalOp.eq_def, s2, c, coordinates]; rfl
  have eh : evalOp (.mod .nat .nil (.ref .nat [] "off_hz") (.constNat heads)) s3 =
      some (Tile.scalar c.head) := by simp [evalOp.eq_def, s3, s2, c, coordinates]; rfl
  have ek : evalOp (.floorDiv .nat .nil (.ref .nat [] "off_h")
        (.floorDiv .nat .nil (.constNat heads) (.constNat kvHeads))) s4 =
      some (Tile.scalar c.kvHead) := by simp [evalOp.eq_def, s4, c, coordinates]; rfl
  erw [coordinateCode, stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
    stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 s1)),
    stepStmts.cons_some (stepStmt_assign_eq_some ez), stepStmts.cons_some (stepStmt_assign_eq_some eh),
    stepStmts.cons_some (stepStmt_assign_eq_some ek), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · constructor <;> simp [s4, s3, s2, s1, c]
  · exact ((((Frame.setReg coordinateWrites s "start_m" _ _ _ (by simp [coordinateWrites])).trans
      (Frame.setReg coordinateWrites s1 "off_hz" _ _ _ (by simp [coordinateWrites]))).trans
      (Frame.setReg coordinateWrites s2 "off_z" _ _ _ (by simp [coordinateWrites]))).trans
      (Frame.setReg coordinateWrites s3 "off_h" _ _ _ (by simp [coordinateWrites]))).trans
      (Frame.setReg coordinateWrites s4 "off_hkv" _ _ _ (by simp [coordinateWrites]))

structure Strides where
  qBatch : Nat
  qHead : Nat
  kBatch : Nat
  kHead : Nat
  vBatch : Nat
  vHead : Nat
  oBatch : Nat
  oHead : Nat

structure Bases where
  q : Nat
  k : Nat
  v : Nat
  out : Nat

def bases (c : Coordinates) (p : Strides) : Bases where
  q := c.batch * p.qBatch + c.head * p.qHead
  k := c.batch * p.kBatch + c.kvHead * p.kHead
  v := c.batch * p.vBatch + c.kvHead * p.vHead
  out := c.batch * p.oBatch + c.head * p.oHead

def offsetExpr (head : RegName) (strideBatch strideHead : Nat) : Op .nat [] :=
  .add .nat .nil
    (.mul .nat .nil (.ref .nat [] "off_z") (.constNat strideBatch))
    (.mul .nat .nil (.ref .nat [] head) (.constNat strideHead))

def offsetCode (p : Strides) : List Stmt :=
  [.assign .nat [] "q_offset" (offsetExpr "off_h" p.qBatch p.qHead),
   .assign .nat [] "k_offset" (offsetExpr "off_hkv" p.kBatch p.kHead),
   .assign .nat [] "v_offset" (offsetExpr "off_hkv" p.vBatch p.vHead),
   .assign .nat [] "o_offset" (offsetExpr "off_h" p.oBatch p.oHead)]

def offsetWrites : List RegName := ["q_offset", "k_offset", "v_offset", "o_offset"]

structure OffsetRegisters (b : Bases) (s : BlockState) : Prop where
  q : s.regs .nat [] "q_offset" = some (Tile.scalar b.q)
  k : s.regs .nat [] "k_offset" = some (Tile.scalar b.k)
  v : s.regs .nat [] "v_offset" = some (Tile.scalar b.v)
  out : s.regs .nat [] "o_offset" = some (Tile.scalar b.out)

theorem offset_eval (s : BlockState) (head : RegName) (batch headIndex strideBatch strideHead : Nat)
    (hb : s.regs .nat [] "off_z" = some (Tile.scalar batch))
    (hh : s.regs .nat [] head = some (Tile.scalar headIndex)) :
    evalOp (offsetExpr head strideBatch strideHead) s =
      some (Tile.scalar (batch * strideBatch + headIndex * strideHead)) := by
  simp only [offsetExpr, evalOp.eq_def, hb, hh, bind, Option.bind_some]
  rfl

theorem offset_executes (s : BlockState) (c : Coordinates) (p : Strides) (hc : CoordinateRegisters c s) :
    ∃ s', stepStmts (offsetCode p) s = some s' ∧ OffsetRegisters (bases c p) s' ∧
      Frame offsetWrites s s' := by
  let b := bases c p
  let sq := s.setReg "q_offset" .nat [] (Tile.scalar b.q)
  let sk := sq.setReg "k_offset" .nat [] (Tile.scalar b.k)
  let sv := sk.setReg "v_offset" .nat [] (Tile.scalar b.v)
  have eq := offset_eval s "off_h" c.batch c.head p.qBatch p.qHead hc.batch hc.head
  have ek := offset_eval sq "off_hkv" c.batch c.kvHead p.kBatch p.kHead
    (by simp [sq, hc.batch]) (by simp [sq, hc.kvHead])
  have ev := offset_eval sk "off_hkv" c.batch c.kvHead p.vBatch p.vHead
    (by simp [sk, sq, hc.batch]) (by simp [sk, sq, hc.kvHead])
  have eo := offset_eval sv "off_h" c.batch c.head p.oBatch p.oHead
    (by simp [sv, sk, sq, hc.batch]) (by simp [sv, sk, sq, hc.head])
  erw [offsetCode, stepStmts.cons_some (stepStmt_assign_eq_some eq),
    stepStmts.cons_some (stepStmt_assign_eq_some ek), stepStmts.cons_some (stepStmt_assign_eq_some ev),
    stepStmts.cons_some (stepStmt_assign_eq_some eo), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · constructor <;> simp [sv, sk, sq, b, bases]
  · exact (((Frame.setReg offsetWrites s "q_offset" _ _ _ (by simp [offsetWrites])).trans
      (Frame.setReg offsetWrites sq "k_offset" _ _ _ (by simp [offsetWrites]))).trans
      (Frame.setReg offsetWrites sk "v_offset" _ _ _ (by simp [offsetWrites]))).trans
      (Frame.setReg offsetWrites sv "o_offset" _ _ _ (by simp [offsetWrites]))

def code (heads kvHeads : Nat) (p : Strides) : List Stmt := coordinateCode heads kvHeads ++ offsetCode p

def writes : List RegName := coordinateWrites ++ offsetWrites

theorem executes (s : BlockState) (heads kvHeads : Nat) (p : Strides) :
    ∃ s', stepStmts (code heads kvHeads p) s = some s' ∧
      CoordinateRegisters (coordinates s heads kvHeads) s' ∧
      OffsetRegisters (bases (coordinates s heads kvHeads) p) s' ∧ Frame writes s s' := by
  obtain ⟨sc, hs, hc, fc⟩ := coordinate_executes s heads kvHeads
  obtain ⟨sf, ht, hb, ff⟩ := offset_executes sc (coordinates s heads kvHeads) p hc
  refine ⟨sf, (stepStmts.append_some hs).trans ht, ?_, hb,
    (fc.mono (by simp [writes])).trans (ff.mono (by simp [writes]))⟩
  constructor
  · exact (ff.regs _ _ _ (by simp [offsetWrites])).trans hc.startM
  · exact (ff.regs _ _ _ (by simp [offsetWrites])).trans hc.headBatch
  · exact (ff.regs _ _ _ (by simp [offsetWrites])).trans hc.batch
  · exact (ff.regs _ _ _ (by simp [offsetWrites])).trans hc.head
  · exact (ff.regs _ _ _ (by simp [offsetWrites])).trans hc.kvHead

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealHeader
