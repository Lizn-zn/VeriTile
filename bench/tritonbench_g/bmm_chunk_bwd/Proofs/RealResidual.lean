import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealSetup

/-! The complete post-loop source and real execution of its residual branch.
Load and store share the same active-lane predicate. Inactive residual loads
retain undef, and both original/candidate addition orders yield carry+residual
in the real interpretation. The final output store is recorded here but its
execution and memory frame are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealResidual
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def rowOffsets (BM : Nat) : Op .nat [BM] :=
  .add .nat .scalarL (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat BM)) (.arange BM)

def columnOffsets (BN : Nat) : Op .nat [BN] :=
  .add .nat .scalarL (.mul .nat .nil (.ref .nat [] "pid_n") (.constNat BN)) (.arange BN)

def basePointer (region : RegionName) (chunk batch row head : Nat) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase region)
    (.add .nat .nil
      (.add .nat .nil (.mul .nat .nil (.ref .nat [] "pid_b") (.constNat batch))
        (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat chunk)) (.constNat row)))
      (.mul .nat .nil (.ref .nat [] "pid_h") (.constNat head)))

def pointers (BM BN : Nat) (base : RegName) (row column : Nat) : Op .ptr [BM, BN] :=
  .ptrAdd .scalarL (.ref .ptr [] base)
    (.add .nat Broadcast.nil.consL.consR
      (.mul .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")) (.constNat row))
      (.mul .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")) (.constNat column)))

def mask (BM BN K : Nat) : Op .bool [BM, BN] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m"))
      (.ref .nat [] "chunk_size_limit"))
    (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")) (.constNat K))

def addition (swapped : Bool) (BM BN : Nat) : Op .real [BM, BN] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "res") (.ref .real [BM, BN] "acc")
  else .add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "acc") (.ref .real [BM, BN] "res")

def residual (swapped : Bool) (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .ptr [] "res_ptr" (basePointer p.Res p.chunk p.rBatch p.rRow p.rHead),
   .assign .ptr [BM, BN] "res_ptrs" (pointers BM BN "res_ptr" p.rRow p.rColumn),
   .assign .real [BM, BN] "res" (.load .real (.ptr (.ref .ptr [BM, BN] "res_ptrs")) (.mask (mask BM BN p.K))),
   .assign .real [BM, BN] "acc" (addition swapped BM BN)]

def code (swapped : Bool) (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .nat [BM] "offs_m" (rowOffsets BM),
   .assign .nat [BN] "offs_n" (columnOffsets BN),
   .ifThen (.constBool p.hasResidual) (residual swapped BM BN p),
   .assign .real [BM, BN] "db" (.ref .real [BM, BN] "acc"),
   .assign .ptr [] "db_ptr" (basePointer p.Db p.chunk p.oBatch p.oRow p.oHead),
   .assign .ptr [BM, BN] "db_ptrs" (pointers BM BN "db_ptr" p.oRow p.oColumn),
   .store .real [BM, BN] (.ptr (.ref .ptr [BM, BN] "db_ptrs"))
     (.ref .real [BM, BN] "db") (.mask (mask BM BN p.K))]

theorem matches_source (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).after = code swapped BM BN p := by
  cases swapped
  all_goals
    simp [code, residual, addition, basePointer, pointers, rowOffsets, columnOffsets, mask,
      RealProgram.realPart, RealProgram.parts, RealProgram.source,
      TypedSource.specialize, Source.dynamicRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try norm_num


def baseValue (chunk batch row head pb pc ph : Nat) : Nat :=
  pb * batch + pc * chunk * row + ph * head

def pointerValues (region : RegionName) (BM BN base row column : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) : Tile .ptr [BM, BN] :=
  Tile.mat fun i j => (region, base + (rows i * row + columns j * column))

theorem base_eval (s : BlockState) (region : RegionName) (chunk batch row head pb pc ph : Nat)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph)) :
    evalOp (basePointer region chunk batch row head) s =
      some (Tile.scalar (region, baseValue chunk batch row head pb pc ph)) := by
  simp [basePointer, evalOp.eq_def, hb, hc, hh, bind, Option.bind]
  apply Tile.ext
  intro i
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, baseValue, Nat.zero_add]

theorem pointers_eval (s : BlockState) (region : RegionName) (BM BN base row column : Nat)
    (baseReg : RegName) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hb : s.regs .ptr [] baseReg = some (Tile.scalar (region, base)))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns)) :
    evalOp (pointers BM BN baseReg row column) s =
      some (pointerValues region BM BN base row column rows columns) := by
  simp [pointers, evalOp.eq_def, hb, hr, hn, bind, Option.bind]
  rfl

theorem mask_eval (s : BlockState) (BM BN K length : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length)) :
    evalOp (mask BM BN K) s =
      some (Tile.mat fun i j => decide (rows i < length ∧ columns j < K)) := by
  simp [mask, evalOp.eq_def, hr, hn, hl, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]
  rfl

def residualValue {BM BN : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length K : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  if rows i < length ∧ columns j < K then
    s.readMem (ptrs.data (i,j,PUnit.unit)).1 (ptrs.data (i,j,PUnit.unit)).2
  else s.undef (ptrs.data (i,j,PUnit.unit)).1 (ptrs.data (i,j,PUnit.unit)).2

def residualValues {BM BN : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length K : Nat) : Tile .real [BM, BN] :=
  Tile.mat fun i j => some (residualValue s ptrs rows columns length K i j)

/-- The residual has no explicit `other`: masked values retain source undef.
The eventual store uses this same predicate, so those lanes are not written. -/
theorem residual_load (s : BlockState) (BM BN K length : Nat)
    (ptrs : Tile .ptr [BM, BN]) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [BM, BN] "res_ptrs" = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length)) :
    evalOp (.load .real (.ptr (.ref .ptr [BM, BN] "res_ptrs")) (.mask (mask BM BN K))) s =
      some (residualValues s ptrs rows columns length K) := by
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, mask_eval s BM BN K length rows columns hr hn hl,
    bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  by_cases h : rows i < length ∧ columns j < K <;>
    simp [residualValues, residualValue, Tile.mat, h]


theorem addition_eval (swapped : Bool) (s : BlockState) (BM BN : Nat)
    (carry res : Fin BM → Fin BN → ℝ)
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)))
    (hr : s.regs .real [BM, BN] "res" = some (Tile.mat fun i j => some (res i j))) :
    evalOp (addition swapped BM BN) s =
      some (Tile.mat fun i j => some (carry i j + res i j)) := by
  cases swapped
  all_goals
    simp [addition, evalOp.eq_def, ha, hr, bind, Option.bind]
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [Tile.bop, Tile.mat, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, _root_.add_comm]

def residualResult (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph length : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let base := baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph
  let ptrs := pointerValues p.Res BM BN base p.rRow p.rColumn rows columns
  let t := s.setReg "res_ptr" .ptr [] (Tile.scalar (p.Res, base))
  let t := t.setReg "res_ptrs" .ptr [BM, BN] ptrs
  let t := t.setReg "res" .real [BM, BN] (residualValues s ptrs rows columns length p.K)
  t.setReg "acc" .real [BM, BN] (Tile.mat fun i j => some
    (carry i j + residualValue s ptrs rows columns length p.K i j))

/-- Execute the source residual branch without replacing inactive loads by
zero. Only the final mathematical addition order differs between sources. -/
theorem residual_executes (swapped : Bool) (s : BlockState) (BM BN : Nat)
    (p : RealProgram.Parameters) (pb pc ph length : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (residual swapped BM BN p) s =
      some (residualResult s BM BN p pb pc ph length rows columns carry) := by
  let base := baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph
  let ptrs := pointerValues p.Res BM BN base p.rRow p.rColumn rows columns
  let s₁ := s.setReg "res_ptr" .ptr [] (Tile.scalar (p.Res, base))
  let s₂ := s₁.setReg "res_ptrs" .ptr [BM, BN] ptrs
  let s₃ := s₂.setReg "res" .real [BM, BN] (residualValues s ptrs rows columns length p.K)
  have hp := base_eval s p.Res p.chunk p.rBatch p.rRow p.rHead pb pc ph hb hc hh
  have hps : evalOp (pointers BM BN "res_ptr" p.rRow p.rColumn) s₁ = some ptrs := by
    apply pointers_eval s₁ p.Res BM BN base p.rRow p.rColumn "res_ptr" rows columns
    · simp [s₁]
    · simpa [s₁] using hr
    · simpa [s₁] using hn
  have hload : evalOp (.load .real (.ptr (.ref .ptr [BM, BN] "res_ptrs"))
      (.mask (mask BM BN p.K))) s₂ = some (residualValues s ptrs rows columns length p.K) := by
    apply residual_load s₂ BM BN p.K length ptrs rows columns
    · simp [s₂]
    · simpa [s₂, s₁] using hr
    · simpa [s₂, s₁] using hn
    · simpa [s₂, s₁] using hl
  have hadd : evalOp (addition swapped BM BN) s₃ = some (Tile.mat fun i j => some
      (carry i j + residualValue s ptrs rows columns length p.K i j)) := by
    apply addition_eval swapped s₃ BM BN carry (residualValue s ptrs rows columns length p.K)
    · simpa [s₃, s₂, s₁] using ha
    · simp [s₃, residualValues]
  dsimp only [s₃, s₂, s₁, ptrs, base] at hps hload hadd
  simp only [residual, stepStmts, stepStmt, hp, hps, hload, hadd,
    bind, Option.bind_some]
  rfl


/-- Residual evaluation changes registers only. -/
theorem residual_frame (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph length : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) :
    {residualResult s BM BN p pb pc ph length rows columns carry with regs := s.regs} = s := by
  cases s
  rfl

theorem residual_register_frame (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph length : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["res_ptr", "res_ptrs", "res", "acc"]) :
    (residualResult s BM BN p pb pc ph length rows columns carry).regs dtype shape name =
      s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  rcases hn with ⟨hb, hp, hr, ha⟩
  simp [residualResult, BlockState.setReg, hb, hp, hr, ha]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealResidual
