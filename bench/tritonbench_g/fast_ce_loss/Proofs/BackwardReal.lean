import bench.tritonbench_g.fast_ce_loss.Proofs.Erasure
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.BackwardReal
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def gradient (cap scale : Bool) (SC SL : ℝ) (lab : Int) (col : Nat)
    (raw lse upstream : ℝ) : ℝ :=
  let scaled := if scale then raw * SL else raw
  let t := Real.tanh (scaled / SC)
  let x := if cap then SC * t else scaled
  let p := Real.exp (x - lse)
  let y := if (col : Int) = lab then p - 1 else p
  let y := if scale then y * SL else y
  let y := if cap then y * (1 - t*t) else y
  (if lab = -100 then 0 else upstream) * y

def original (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) : Kernel :=
  (backward .fp32 cap scale Bool.false X D Z L V R DR B SC SL).eraseDType.toAlgKernel

def col (u : BlockState) (B : Nat) (j : Fin B) := u.pids 1*B+j.val
def address (u : BlockState) (R B : Nat) (j : Fin B) := u.pids 0*R+col u B j

theorem readback (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) (u v : BlockState)
    (he : exec (original cap scale X D Z L V R DR B SC SL) u=some v)
    (j : Fin B) (hj : col u B j<V) :
    v.readMem X (address u R B j) =
      gradient cap scale SC SL (u.readMemValue .int (Region.cast L) (u.pids 0))
        (col u B j) (u.readMem X (address u R B j))
        (u.readMem Z (u.pids 0)) (u.readMem D (u.pids 0*DR)) := by
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi] at he
    subst v
    simp only [address,col]
    rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := X) (shape := [B]) _
      (fun k => u.pids 0*R+(u.pids 1*B+k.1.val)) _
      (fun k => u.pids 1*B+k.1.val<V) (j,PUnit.unit)
      (by simpa only [col] using hj)
      (fun k _ heq => by
        rcases k with ⟨k,ku⟩
        cases ku
        congr 1
        apply Fin.ext
        simp only at heq
        omega)]
    simp [gradient,address,col,hi,show u.pids 1*B+j.val<V from hj,
      WithBot.realTanh,WithBot.realMul,WithBot.realDiv,WithBot.realSub,WithBot.realExp]

  all_goals split_ifs <;> norm_num [WithBot.unbotD,WithBot.recBotCoe]

  all_goals first | rfl | exact Or.inl rfl | (dsimp; norm_num)

theorem frame (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) (u v : BlockState)
    (he : exec (original cap scale X D Z L V R DR B SC SL) u=some v)
    (r : RegionName) (o : Nat)
    (ho : ∀ j : Fin B, col u B j<V → ¬(X=r ∧ address u R B j=o)) :
    v.mem r o=u.mem r o := by
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi] at he
    subst v
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro i _ hi hc
    exact ho i.1 (by simpa only [col] using hi) hc

/-- Whole-kernel successful execution, all active outputs, and the tagged-cell frame. -/
theorem run (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) (u : BlockState) :
    ∃ v, exec (original cap scale X D Z L V R DR B SC SL) u=some v ∧
      (∀ j : Fin B, col u B j<V → v.readMem X (address u R B j) =
        gradient cap scale SC SL (u.readMemValue .int (Region.cast L) (u.pids 0))
          (col u B j) (u.readMem X (address u R B j))
          (u.readMem Z (u.pids 0)) (u.readMem D (u.pids 0*DR))) ∧
      (∀ r o, (∀ j : Fin B, col u B j<V → ¬(X=r ∧ address u R B j=o)) → v.mem r o=u.mem r o) := by
  cases he : exec (original cap scale X D Z L V R DR B SC SL) u with
  | none =>
    cases cap <;> cases scale
    all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
    all_goals
      simp only [original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
        ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
        Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
        Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
        VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
        IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
      simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
        Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
        NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
        ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi] at he
  | some v =>
    refine ⟨v,rfl,?_,?_⟩
    · intro j hj
      exact readback cap scale X D Z L V R DR B SC SL u v he j hj
    · intro r o ho
      exact frame cap scale X D Z L V R DR B SC SL u v he r o ho

theorem trace_safe (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) (bounds : RegionBounds) (u : BlockState)
    (hbL : u.pids 0 < bounds (Region.cast L))
    (hbZ : u.pids 0 < bounds Z)
    (hbD : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 → u.pids 0*DR < bounds D)
    (hbX : ∀ j : Fin B, col u B j<V → address u R B j < bounds X) :
    (original cap scale X D Z L V R DR B SC SL).TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
    simp [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MemAccess.SafeAt,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,
      hbL,hbZ,hbD,exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi]
    exact fun j hj => hbX j hj

theorem flatten_ok (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) :
    (original cap scale X D Z L V R DR B SC SL).FlattenOk := by
  unfold Kernel.FlattenOk
  cases cap <;> cases scale
  all_goals
    simp [original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.BackwardReal
