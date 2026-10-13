import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerOneTileMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton FullSource
set_option maxHeartbeats 3500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def body (Y X : RegionName) (M N B : Nat) : List Stmt :=
  match ((softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.true).eraseDType.toAlgKernel.body[1]? : Option Stmt) with
  | some (.ifThenElse _ yes _) => yes
  | _ => []

def tileOffset {B : Nat} (row N : Nat) (i : TileIndex [B]) : Nat := row*N+i.1.val

def tileActive {B : Nat} (N : Nat) (i : TileIndex [B]) : Bool := decide (i.1.val<N)

noncomputable def scatter {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) : BlockState :=
  (TileShape.allIndices [B]).foldl
    (fun st i ↦ if active i then st.writeMem Y (off i) (value i) else st) s

noncomputable def midState (s : BlockState) (Y X : RegionName)
    (row N B : Nat) (hB : 0<B) : BlockState :=
  let off := tileOffset (B := B) row N
  let inp := score s X row N B
  let mx := batchMax s X row N B hB
  let z := batchSum s X row N B hB
  let e := fun i : TileIndex [B] ↦ WithBot.realExp (WithBot.realSub (inp i) mx)
  let s₁ := s.setReg "n_offsets" .nat [B] ⟨fun i ↦ i.1.val⟩
  let s₂ := s₁.setReg "offset" .nat [B] ⟨off⟩
  let s₃ := s₂.setReg "input_ptrs" .ptr [B] ⟨fun i ↦ (X,off i)⟩
  let s₄ := s₃.setReg "mask" .bool [B] ⟨tileActive N⟩
  let s₅ := s₄.setReg "inp" .real [B] ⟨inp⟩
  let s₆ := s₅.setReg "m" .real [] (Tile.scalar mx)
  let s₇ := s₆.setReg "e" .real [B] ⟨e⟩
  let s₈ := s₇.setReg "z" .real [] (Tile.reduceSum 0 Bool.false ⟨e⟩)
  let s₉ := s₈.setReg "out" .real [B] ⟨fun i ↦ WithBot.realDiv (e i) (some z)⟩
  s₉.setReg "output_ptrs" .ptr [B] ⟨fun i ↦ (Y,off i)⟩

private theorem exp_finite (x : WithBot ℝ) :
    WithBot.realExp x=some (x.elim 0 Real.exp) := by cases x <;> rfl

theorem step (Y X : RegionName) (M N B row : Nat) (hB : 0<B)
    (s : BlockState) (hrow : s.regs .nat [] "pid_m"=some (Tile.scalar row)) :
    ∃ mid, stepStmts (body Y X M N B) s =
      some (scatter mid Y (tileOffset row N) (storeValue s X row N B hB) (tileActive N)) ∧
      mid.mem=s.mem ∧ mid.pids=s.pids ∧ mid.numPids=s.numPids ∧ mid.undef=s.undef := by
  simp [body, softmax_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hrow,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, List.getElem_cons_zero, hB, ↓reduceDIte]
  refine ⟨midState s Y X row N B hB, ?_, rfl, rfl, rfl, rfl⟩
  simp [midState, scatter, tileOffset, tileActive, storeValue, score, batchMax, batchSum,
    RealForwardInner.globalMax, TileShape.insertAxisIndex,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop_data,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, ComparableDType.lt,
    BlockState.writeMemTyped_real, BlockState.readMemValue_real, exp_finite, ←WithBot.coe_sum]
  simp only [Tile.reduceSumDrop, TileShape.axisDim, TileShape.insertAxisIndex, List.getElem_cons_zero]
  congr 1

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
