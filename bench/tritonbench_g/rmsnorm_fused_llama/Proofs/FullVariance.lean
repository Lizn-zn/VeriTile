import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.FullSource

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
open VeriTile.Triton Finset
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

structure VarInv (s0 : BlockState) (X Y : RegionName)
    (stride N B c : Nat) (s : BlockState) : Prop where
  mem : s.mem = s0.mem
  pids : s.pids = s0.pids
  xptr : s.regs .ptr [] "X" = some (Tile.scalar (X, s0.pids 0 * stride))
  yptr : s.regs .ptr [] "Y" = some (Tile.scalar (Y, s0.pids 0 * stride))
  acc : s.regs .real [B] "_var" =
    some ⟨fun i => some (accumulator s0 X stride N B c i.1)⟩

noncomputable def preState (s : BlockState) (X Y : RegionName) (stride B : Nat) : BlockState :=
  (((s.setReg "row" .nat [] (Tile.scalar (s.pids 0))).setReg "Y" .ptr []
    (Tile.scalar (Y, s.pids 0 * stride))).setReg "X" .ptr []
    (Tile.scalar (X, s.pids 0 * stride))).setReg "_var" .real [B]
      ⟨fun _ => some 0⟩

theorem pre_exec (s : BlockState) (X Y : RegionName) (stride B : Nat) :
    stepStmts (pre X Y stride B) s = some (preState s X Y stride B) := by
  simp [pre, preState, stepStmts, stepStmt, evalOp.eq_def, Tile.ptrAdd,
    Tile.bop, NumericDType.mul, Tile.scalar, -Tile.scalar_eta, Function.comp_def, Region.cast]

theorem pre_inv (s : BlockState) (X Y : RegionName) (stride N B : Nat) :
    VarInv s X Y stride N B 0 (preState s X Y stride B) := by
  constructor <;> simp [preState, accumulator] <;> rfl

theorem accumulator_succ (s : BlockState) (X : RegionName)
    (stride N B c : Nat) (j : Fin B) :
    accumulator s X stride N B (c+1) j = accumulator s X stride N B c j +
      (if c * B + j.val < N then s.readMem X (s.pids 0 * stride + (c * B + j.val)) else 0) ^ 2 := by
  simp only [accumulator, Fin.sum_univ_castSucc]
  split_ifs <;> simp_all

noncomputable def inputTile (s : BlockState) (X : RegionName)
    (stride N B c : Nat) : Tile .real [B] :=
  ⟨fun i => some (if c * B + i.1.val < N then
    s.readMem X (s.pids 0 * stride + (c * B + i.1.val)) else 0)⟩

noncomputable def varState (s0 s : BlockState) (X : RegionName)
    (stride N B c : Nat) : BlockState :=
  (((s.setReg "off" .nat [] (Tile.scalar (c * B))).setReg "cols" .nat [B]
    (Tile.vec (fun j => c * B + j.val))).setReg "x" .real [B]
    (inputTile s0 X stride N B c)).setReg "_var" .real [B]
    ⟨fun i => some (accumulator s0 X stride N B (c+1) i.1)⟩

theorem var_exec (s0 s : BlockState) (X Y : RegionName)
    (stride N B c : Nat) (h : VarInv s0 X Y stride N B c s) :
    stepStmts (varBody N B) (s.setReg "off" .nat [] (Tile.scalar (c * B))) =
      some (varState s0 s X stride N B c) := by
  have hm : s.readMem = s0.readMem := by
    funext r o
    simp only [BlockState.readMem, h.mem]
  simp (config := { contextual := Bool.true }) [varBody, varState, stepStmts, stepStmt, evalOp.eq_def,
    h.xptr, h.acc, inputTile, Tile.vec, Tile.bop, Tile.cop, Tile.ptrAdd,
    Tile.scalar, -Tile.scalar_eta, Tile.select, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    BlockState.readMemValue_real, hm, accumulator_succ, pow_two, apply_ite]
  norm_num
  rfl

theorem var_inv (s0 s : BlockState) (X Y : RegionName)
    (stride N B c : Nat) (h : VarInv s0 X Y stride N B c s) :
    VarInv s0 X Y stride N B (c+1) (varState s0 s X stride N B c) := by
  constructor
  · simpa [varState] using h.mem
  · simpa [varState] using h.pids
  · simpa [varState] using h.xptr
  · simpa [varState] using h.yptr
  · simp [varState]

theorem var_loop (s0 s : BlockState) (X Y : RegionName)
    (stride N B : Nat) (hB : 0 < B) (h : VarInv s0 X Y stride N B 0 s) :
    ∃ c t, stepStmt (.forRange "off" 0 N B (varBody N B)) s = some t ∧
      N ≤ c * B ∧ VarInv s0 X Y stride N B c t := by
  have hinit : ∃ c, 0 = c * B ∧ VarInv s0 X Y stride N B c s := ⟨0, by simp, h⟩
  obtain ⟨final, t, he, hn, c, hc, hi⟩ :=
    forRange_inv (idx := "off") (start := 0) (stop := N) (step := B)
      (body := varBody N B)
      (P := fun off st => ∃ c, off = c * B ∧ VarInv s0 X Y stride N B c st)
      (Nat.ne_of_gt hB) hinit (by
        intro off st _ hst
        rcases hst with ⟨c, rfl, hst⟩
        refine ⟨varState s0 st X stride N B c, var_exec s0 st X Y stride N B c hst,
          c+1, ?_, var_inv s0 st X Y stride N B c hst⟩
        simp [Nat.add_mul])
  exact ⟨c, t, he, hc ▸ hn, hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
