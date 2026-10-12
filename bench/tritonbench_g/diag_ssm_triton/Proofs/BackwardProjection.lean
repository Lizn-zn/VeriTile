import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardProjection
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardProjection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def previousOffset (batch dim block : Nat) : Op .nat [block] :=
  .castIntToNat (.sub .int .scalarR (.castNatToInt (.ref .nat [block] "offsets"))
    (.mul .int .nil (.constInt (Int.ofNat batch)) (.constInt (Int.ofNat dim))))

def previousLoad (s y : RegionName) (batch dim block : Nat) : Stmt :=
  .ifThenElse (.gt .nat .nil (.ref .nat [] "t") (.constNat 0))
    [.assign .real [block] "s" (.load .real (.region y (previousOffset batch dim block))
      (.maskOther (.ref .bool [block] "mask") ((Op.const 0).broadcast [block])))]
    [.assign .real [block] "s" (.load .real (.region s (.ref .nat [block] "col_offsets"))
      (.maskOther (.ref .bool [block] "mask") ((Op.const 0).broadcast [block])))]

def prelude (l : RegionName) (batch dim block : Nat) : List Stmt :=
  let old := RealReference.diagSsmBackwardPreLoop l batch dim block
  old.take 2 ++ [ForwardProjection.mask batch dim block] ++ old.drop 3

def loopBody (s y gx gy : RegionName) (length batch dim block : Nat) : List Stmt :=
  let old := RealReference.diagSsmBackwardLoopBody s y gx gy length batch dim block
  old.take 1 ++ [ForwardProjection.offset batch dim block] ++ (old.drop 2).take 1 ++
    [previousLoad s y batch dim block] ++ old.drop 4

theorem real_source_body (s l y gs gx gl gy : RegionName) (length batch dim block : Nat) :
    (Erasure.backward Bool.false s l y gs gx gl gy length batch dim block).toAlgKernel.body =
      prelude l batch dim block ++ [.forLoop "i" length (loopBody s y gx gy length batch dim block)] ++
        RealReference.diagSsmBackwardPostLoop gs gl block := by
  rfl

theorem previous_offset (batch dim block : Nat) (u : BlockState) :
    evalOp (previousOffset batch dim block) u =
      evalOp (.sub .nat .scalarR (.ref .nat [block] "offsets") (.constNat (batch*dim))) u := by
  simp only [previousOffset,evalOp.eq_def]
  cases h : u.regs .nat [block] "offsets" with
  | none => rfl
  | some v =>
    simp only [h,bind,Option.bind_some]
    congr 1
    apply Tile.ext
    intro i
    change ((v.data i : Int) - (batch : Int) * (dim : Int)).toNat = v.data i - batch*dim
    rw [← Int.natCast_mul]
    omega

def previousLoadReference (s y : RegionName) (batch dim block : Nat) : Stmt :=
  .ifThenElse (.gt .nat .nil (.ref .nat [] "t") (.constNat 0))
    [.assign .real [block] "s" (.load .real (.region y
      (.sub .nat .scalarR (.ref .nat [block] "offsets") (.constNat (batch*dim))))
      (.maskOther (.ref .bool [block] "mask") ((Op.const 0).broadcast [block])))]
    [.assign .real [block] "s" (.load .real (.region s (.ref .nat [block] "col_offsets"))
      (.maskOther (.ref .bool [block] "mask") ((Op.const 0).broadcast [block])))]

theorem load_congr {d : TileDType} {sh : TileShape} (r : Region d) (a b : Op .nat sh)
    (mask : MaskOpt d sh) (u : BlockState) (h : evalOp a u = evalOp b u) :
    evalOp (.load d (.region r a) mask) u = evalOp (.load d (.region r b) mask) u := by
  conv_lhs => rw [evalOp]
  conv_rhs => rw [evalOp]
  rw [h]
  cases evalOp b u <;> cases d <;> rfl

theorem previous_load_step (s y : RegionName) (batch dim block : Nat) (u : BlockState) :
    stepStmt (previousLoad s y batch dim block) u =
      stepStmt (previousLoadReference s y batch dim block) u := by
  have h := load_congr (d := .real) y (previousOffset batch dim block)
    (.sub .nat .scalarR (.ref .nat [block] "offsets") (.constNat (batch*dim)))
    (.maskOther (.ref .bool [block] "mask") ((Op.const 0).broadcast [block])) u
    (previous_offset batch dim block u)
  simp only [previousLoad,previousLoadReference,stepStmt,stepStmts,h]

theorem prelude_steps (l : RegionName) (batch dim block : Nat) (u : BlockState) :
    stepStmts (prelude l batch dim block) u =
      stepStmts (RealReference.diagSsmBackwardPreLoop l batch dim block) u := by
  simp only [prelude,RealReference.diagSsmBackwardPreLoop,List.take,List.drop,List.cons_append,List.nil_append,
    stepStmts,ForwardProjection.mask_step]

theorem loop_steps (s y gx gy : RegionName) (length batch dim block : Nat) (u : BlockState) :
    stepStmts (loopBody s y gx gy length batch dim block) u =
      stepStmts (RealReference.diagSsmBackwardLoopBody s y gx gy length batch dim block) u := by
  simp only [loopBody,RealReference.diagSsmBackwardLoopBody,List.take,List.drop,List.cons_append,List.nil_append,
    stepStmts,ForwardProjection.offset_step,previous_load_step,previousLoadReference]

/-- Constant-stride range and the zero-based for-loop execute the same states,
including failure; this is the only control representation difference. -/
theorem loop_range (idx : RegName) (start stop : Nat) (left right : List Stmt)
    (h : ∀ u, stepStmts left u = stepStmts right u) (u : BlockState) :
    stepForLoopAux idx start stop left u = stepForRangeAux idx start stop 1 right u := by
  by_cases hc : start<stop
  · rw [stepForLoopAux.step_lt hc,stepForRangeAux.step_lt (by decide) hc,h]
    cases stepStmts right (u.setReg idx .nat [] (Tile.scalar start)) with
    | none => rfl
    | some v => exact loop_range idx (start+1) stop left right h v
  · rw [stepForLoopAux.step_ge (by omega),stepForRangeAux.step_ge (by decide) (by omega)]
termination_by stop-start
decreasing_by omega

theorem flat_previous_load_step (A : FlatAlloc) (s y : RegionName)
    (batch dim block : Nat) (u : BlockState) :
    stepStmt (A.flattenStmt (previousLoad s y batch dim block)) u =
      stepStmt (A.flattenStmt (previousLoadReference s y batch dim block)) u := by
  have hoff : evalOp (.add .nat .scalarL (.constNat (A.base y)) (previousOffset batch dim block)) u =
      evalOp (.add .nat .scalarL (.constNat (A.base y))
        (.sub .nat .scalarR (.ref .nat [block] "offsets") (.constNat (batch*dim)))) u := by
    simp only [evalOp_add,previous_offset]
  have h := load_congr (d := .real) (Region.cast A.flat)
    (.add .nat .scalarL (.constNat (A.base y)) (previousOffset batch dim block))
    (.add .nat .scalarL (.constNat (A.base y))
      (.sub .nat .scalarR (.ref .nat [block] "offsets") (.constNat (batch*dim))))
    (.maskOther (.ref .bool [block] "mask") ((Op.const 0).broadcast [block])) u hoff
  simp only [previousLoad,previousLoadReference,previousOffset,FlatAlloc.flattenStmt,FlatAlloc.flattenStmts,FlatAlloc.flattenOp,
    FlatAlloc.shiftBroadcast,Region.cast,stepStmt,stepStmts] at h ⊢
  rw [h]

theorem flat_prelude_steps (A : FlatAlloc) (l : RegionName) (batch dim block : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (prelude l batch dim block)) u =
      stepStmts (A.flattenStmts (RealReference.diagSsmBackwardPreLoop l batch dim block)) u := by
  simp only [prelude,RealReference.diagSsmBackwardPreLoop,List.take,List.drop,List.cons_append,List.nil_append,
    FlatAlloc.flattenStmts,stepStmts,ForwardProjection.flat_mask_step]

theorem flat_loop_steps (A : FlatAlloc) (s y gx gy : RegionName)
    (length batch dim block : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (loopBody s y gx gy length batch dim block)) u =
      stepStmts (A.flattenStmts (RealReference.diagSsmBackwardLoopBody s y gx gy length batch dim block)) u := by
  simp only [loopBody,RealReference.diagSsmBackwardLoopBody,List.take,List.drop,List.cons_append,List.nil_append,
    FlatAlloc.flattenStmts,stepStmts,ForwardProjection.flat_offset_step,flat_previous_load_step,previousLoadReference]

theorem real_source_flat_exec (A : FlatAlloc) (s l y gs gx gl gy : RegionName)
    (length batch dim block : Nat) (u : BlockState) :
    exec (A.flattenKernel (Erasure.backward Bool.false s l y gs gx gl gy length batch dim block).toAlgKernel) u =
      exec (A.flattenKernel (RealReference.diag_ssm_backward_kernel s l y gs gx gl gy length batch dim block).toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [real_source_body,RealReference.diag_ssm_backward_kernel_toAlg_body]
  simp only [RealReference.diagSsmBackwardProjectedBody,FlatAlloc.flattenStmts_append,
    ProgramRewrite.append,flat_prelude_steps]
  congr 1
  congr 1
  funext v
  simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,stepStmts,
    stepForLoopAux.forLoop_unfold,stepForRangeAux.forRange_unfold]
  rw [loop_range "i" 0 length _ _ (flat_loop_steps A s y gx gy length batch dim block)]

theorem typed_source_flat_exec (storage : Storage) (A : FlatAlloc) (s l y gs gx gl gy : RegionName)
    (length batch dim block : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.backward storage Bool.false s l y gs gx gl gy length batch dim block).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (RealReference.diag_ssm_backward_kernel s l y gs gx gl gy length batch dim block).toAlgKernel) u := by
  have h := real_source_flat_exec A s l y gs gx gl gy length batch dim block u
  simpa only [exec,FlatAlloc.flattenKernel,Erasure.backward_body] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardProjection
