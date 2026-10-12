import bench.tritonbench_g.bmm_optimized.Proofs.RealLoop
import VeriTile.Triton.KernelLemmas.ScatterStore

/-! The complete postlude: optional broadcast mask and the actual terminal
store. Readback and frame concern tagged memory cells, including every cell
outside the active tile. Mask elision remains explicit. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealStore
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def maskCode (BM BN : Nat) (dm dn : Bool) : List Stmt :=
  match dm, dn with
  | Bool.false, Bool.false => [.assign .bool [BM,BN] "mask_c"
      (.boolAnd Broadcast.nil.consL.consR
        (.expandDim ⟨1,by simp⟩ (.ref .bool [BM] "mask_m"))
        (.expandDim ⟨0,by simp⟩ (.ref .bool [BN] "mask_n")))]
  | Bool.false, Bool.true => [.assign .bool [BM,1] "mask_c"
      (.expandDim ⟨1,by simp⟩ (.ref .bool [BM] "mask_m"))]
  | Bool.true, Bool.false => [.assign .bool [1,BN] "mask_c"
      (.expandDim ⟨0,by simp⟩ (.ref .bool [BN] "mask_n"))]
  | Bool.true, Bool.true => []

def mask (BM BN : Nat) (dm dn : Bool) : MaskOpt .real [BM,BN] :=
  match dm, dn with
  | Bool.false, Bool.false => .mask (.ref .bool [BM,BN] "mask_c")
  | Bool.false, Bool.true => .mask (.remap [BM,BN]
      (fun i => (i.1,0,PUnit.unit)) (.ref .bool [BM,1] "mask_c"))
  | Bool.true, Bool.false => .mask (.remap [BM,BN]
      (fun i => (0,i.2.1,PUnit.unit)) (.ref .bool [1,BN] "mask_c"))
  | Bool.true, Bool.true => .none

def code (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  maskCode BM BN p.divM p.divN ++ [.store .real [BM,BN]
    (.ptr (.ref .ptr [BM,BN] "o_ptrs")) (.ref .real [BM,BN] "o") (mask BM BN p.divM p.divN)]

def prepared (s : BlockState) {BM BN : Nat} (p : RealProgram.Parameters)
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) : BlockState :=
  match p.divM, p.divN with
  | Bool.false, Bool.false => s.setReg "mask_c" .bool [BM,BN]
      (Tile.mat fun i j => decide (rows i < p.M ∧ cols j < p.N))
  | Bool.false, Bool.true => s.setReg "mask_c" .bool [BM,1]
      (Tile.mat fun i _ => decide (rows i < p.M))
  | Bool.true, Bool.false => s.setReg "mask_c" .bool [1,BN]
      (Tile.mat fun _ j => decide (cols j < p.N))
  | Bool.true, Bool.true => s

def active {BM BN : Nat} (p : RealProgram.Parameters) (rows : Fin BM → Nat)
    (cols : Fin BN → Nat) (i : TileIndex [BM,BN]) : Prop :=
  RealMath.active p.divM (rows i.1) p.M ∧ RealMath.active p.divN (cols i.2.1) p.N
instance {BM BN : Nat} (p : RealProgram.Parameters) (rows : Fin BM → Nat) (cols : Fin BN → Nat) :
    DecidablePred (active p rows cols) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def offset {BM BN : Nat} (base N : Nat) (rows : Fin BM → Nat)
    (cols : Fin BN → Nat) (i : TileIndex [BM,BN]) : Nat := base + rows i.1 * N + cols i.2.1

def write (s : BlockState) {BM BN : Nat} (p : RealProgram.Parameters) (base : Nat)
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ) : BlockState :=
  (TileShape.allIndices [BM,BN]).foldl (fun t i => if active p rows cols i then
    t.writeMem p.O (offset base p.N rows cols i) (values i.1 i.2.1) else t) s

theorem prepares (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (rows : Fin BM → Nat) (cols : Fin BN → Nat)
    (hm : p.divM = Bool.false → s.regs .bool [BM] "mask_m" = some (Tile.vec fun i => decide (rows i < p.M)))
    (hn : p.divN = Bool.false → s.regs .bool [BN] "mask_n" = some (Tile.vec fun j => decide (cols j < p.N))) :
    stepStmts (maskCode BM BN p.divM p.divN) s = some (prepared s p rows cols) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn
  all_goals
    simp only [Bool.false_eq_true, false_implies, true_implies] at hm hn
    simp [maskCode, prepared, stepStmts, stepStmt, evalOp.eq_def, hm, hn, bind, Option.bind,
      Tile.mat, Tile.vec, Tile.expandDim, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex]
    try rfl

theorem executes (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (base : Nat)
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (hm : p.divM = Bool.false → s.regs .bool [BM] "mask_m" = some (Tile.vec fun i => decide (rows i < p.M)))
    (hn : p.divN = Bool.false → s.regs .bool [BN] "mask_n" = some (Tile.vec fun j => decide (cols j < p.N)))
    (hp : s.regs .ptr [BM,BN] "o_ptrs" = some (Tile.mat fun i j => (p.O, base + rows i * p.N + cols j)))
    (hv : s.regs .real [BM,BN] "o" = some (Tile.mat fun i j => some (values i j))) :
    stepStmts (code BM BN p) s = some (write (prepared s p rows cols) p base rows cols values) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn
  all_goals
    simp only [Bool.false_eq_true, false_implies, true_implies] at hm hn
    simp [code, maskCode, mask, prepared, write, active, offset, RealMath.active,
      stepStmts, stepStmt, evalOp.eq_def, hp, hv, hm, hn, bind, Option.bind,
      Tile.mat, Tile.vec, Tile.expandDim, Tile.bop, Tile.remap,
      Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
      BlockState.writeMemTyped_real]
    try rfl

theorem matches_source (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).after = code BM BN p := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> cases swapped
  all_goals
    simp [code, maskCode, mask, RealProgram.realPart, RealProgram.parts, RealProgram.source,
      TypedSource.specialize, Source.dynamicRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
      Broadcast.leftIndex, funext_iff]

theorem prepared_frame (s : BlockState) {BM BN : Nat} (p : RealProgram.Parameters)
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) :
    {prepared s p rows cols with regs := s.regs} = s := by
  cases hm : p.divM <;> cases hn : p.divN <;> simp [prepared, hm, hn, BlockState.setReg]

theorem frame (s : BlockState) {BM BN : Nat} (p : RealProgram.Parameters) (base : Nat)
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (r : RegionName) (address : Nat)
    (h : ∀ i, active p rows cols i → ¬ (p.O = r ∧ offset base p.N rows cols i = address)) :
    (write s p base rows cols values).mem r address = s.mem r address := by
  exact foldl_store_preserve_cell _ _ _ r address _ s (fun i _ hi => h i hi)

theorem readback (s : BlockState) {BM BN : Nat} (p : RealProgram.Parameters) (base : Nat)
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (i : TileIndex [BM,BN]) (hi : active p rows cols i)
    (hinj : ∀ a b, active p rows cols a → active p rows cols b →
      offset base p.N rows cols a = offset base p.N rows cols b → a = b) :
    (write s p base rows cols values).mem p.O (offset base p.N rows cols i) =
      MemCell.real (values i.1 i.2.1) := by
  obtain ⟨before,after,hsplit⟩ := List.append_of_mem (TileShape.mem_allIndices [BM,BN] i)
  have hn := TileShape.allIndices_nodup [BM,BN]
  rw [hsplit, List.nodup_append, List.nodup_cons] at hn
  have hafter : ∀ k ∈ after, active p rows cols k →
      ¬ (p.O = p.O ∧ offset base p.N rows cols k = offset base p.N rows cols i) := by
    intro k hk ha he
    have hki := hinj k i ha hi he.2
    subst k
    exact hn.2.1.1 hk
  unfold write
  rw [hsplit, List.foldl_append, List.foldl_cons]
  rw [if_pos hi, foldl_store_preserve_cell _ _ _ _ _ after _ hafter,
    BlockState.writeMem_mem, if_pos ⟨rfl,rfl⟩]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealStore
