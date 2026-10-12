import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSetup

/-! Values of the two actual contraction loads. The K-tail predicate
is retained even in the last partial block; row and column masking are
independent. Values come from the current memory, never a supplied dot result. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoads
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def remaining (BK K : Nat) : Op .nat [] :=
  .sub .nat .nil (.constNat K)
    (.mul .nat .nil (.ref .nat [] "k") (.constNat BK))

def aMask (BM BK K : Nat) : Op .bool [BM, BK] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")) (.ref .nat [] "chunk_size_limit"))
    (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BK] "offs_k")) (remaining BK K))

def bMask (BN BK K : Nat) : Op .bool [BK, BN] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BK] "offs_k")) (remaining BK K))
    (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")) (.ref .nat [] "chunk_size_limit"))

def aLoad (BM BK K : Nat) : Op .real [BM, BK] :=
  .load .real (.ptr (.ref .ptr [BM, BK] "a_ptrs"))
    (.maskOther (aMask BM BK K) ((Op.const 0).broadcast [BM, BK]))

def bLoad (BN BK K : Nat) : Op .real [BK, BN] :=
  .load .real (.ptr (.ref .ptr [BK, BN] "b_ptrs"))
    (.maskOther (bMask BN BK K) ((Op.const 0).broadcast [BK, BN]))

theorem a_mask (s : BlockState) (BM BK K length c : Nat) (rows : Fin BM → Nat)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (aMask BM BK K) s = some (Tile.mat fun i k =>
      decide (rows i < length ∧ k.val < K - c * BK)) := by
  simp [aMask, remaining, evalOp.eq_def, hr, hk, hl, hc, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i, k, ⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, NumericDType.sub, NumericDType.mul, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex]
  rfl

theorem b_mask (s : BlockState) (BN BK K length c : Nat) (columns : Fin BN → Nat)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (bMask BN BK K) s = some (Tile.mat fun k j =>
      decide (k.val < K - c * BK ∧ columns j < length)) := by
  simp [bMask, remaining, evalOp.eq_def, hn, hk, hl, hc, bind, Option.bind]
  apply Tile.ext
  rintro ⟨k, j, ⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, NumericDType.sub, NumericDType.mul, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex]
  rfl

def aValues {BM BK : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BK])
    (rows : Fin BM → Nat) (length K c : Nat) : Tile .real [BM, BK] :=
  Tile.mat fun i k => some (if rows i < length ∧ k.val < K - c * BK then
    s.readMem (ptrs.data (i,k,PUnit.unit)).1 (ptrs.data (i,k,PUnit.unit)).2 else 0)

def bValues {BN BK : Nat} (s : BlockState) (ptrs : Tile .ptr [BK, BN])
    (columns : Fin BN → Nat) (length K c : Nat) : Tile .real [BK, BN] :=
  Tile.mat fun k j => some (if k.val < K - c * BK ∧ columns j < length then
    s.readMem (ptrs.data (k,j,PUnit.unit)).1 (ptrs.data (k,j,PUnit.unit)).2 else 0)

theorem a_load (s : BlockState) (BM BK K length c : Nat) (rows : Fin BM → Nat)
    (ptrs : Tile .ptr [BM, BK])
    (hp : s.regs .ptr [BM, BK] "a_ptrs" = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (aLoad BM BK K) s = some (aValues s ptrs rows length K c) := by
  have hm := a_mask s BM BK K length c rows hr hk hl hc
  have hz : evalOp ((Op.const 0).broadcast [BM, BK]) s =
      some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BM, BK]) := by
    simp [evalOp, evalOp_const, Option.bind]
  unfold aLoad
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hm, hz, bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i, k, ⟨⟩⟩
  by_cases h : rows i < length ∧ k.val < K - c * BK <;>
    simp [aValues, Tile.mat, h]

theorem b_load (s : BlockState) (BN BK K length c : Nat) (columns : Fin BN → Nat)
    (ptrs : Tile .ptr [BK, BN])
    (hp : s.regs .ptr [BK, BN] "b_ptrs" = some ptrs)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (bLoad BN BK K) s = some (bValues s ptrs columns length K c) := by
  have hm := b_mask s BN BK K length c columns hn hk hl hc
  have hz : evalOp ((Op.const 0).broadcast [BK, BN]) s =
      some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BK, BN]) := by
    simp [evalOp, evalOp_const, Option.bind]
  unfold bLoad
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hm, hz, bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k, j, ⟨⟩⟩
  by_cases h : k.val < K - c * BK ∧ columns j < length <;>
    simp [bValues, Tile.mat, h]

def update (swapped : Bool) (BM BN BK : Nat) : Op .real [BM, BN] :=
  let dot := Op.dot (batch := []) (.ref .real [BM, BK] "a") (.ref .real [BK, BN] "b")
  if swapped then .add .real (.consSame (.consSame .nil)) dot (.ref .real [BM, BN] "acc")
  else .add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "acc") dot

def code (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .real [BM, BK] "a" (aLoad BM BK p.K),
   .assign .real [BK, BN] "b" (bLoad BN BK p.K),
   .assign .real [BM, BN] "acc"
     (update swapped BM BN BK),
   .assign .ptr [BM, BK] "a_ptrs"
     (.ptrAdd .scalarR (.ref .ptr [BM, BK] "a_ptrs")
       (.mul .nat .nil (.constNat BK) (.constNat p.aColumn))),
   .assign .ptr [BK, BN] "b_ptrs"
     (.ptrAdd .scalarR (.ref .ptr [BK, BN] "b_ptrs")
       (.mul .nat .nil (.constNat BK) (.constNat p.bColumn)))]

set_option maxRecDepth 18000 in
set_option maxHeartbeats 3000000 in
/-- The load views belong to the entire actual five-statement loop body,
including the dot update and both pointer advances. -/
theorem matches_source (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).body = code swapped BM BN BK p := by
  cases swapped
  all_goals
    simp [code, update, aLoad, bLoad, aMask, bMask, remaining,
      RealProgram.realPart, RealProgram.parts, RealProgram.branch, RealProgram.source,
      TypedSource.specialize, Source.conditional?, Source.dynamicRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoads
