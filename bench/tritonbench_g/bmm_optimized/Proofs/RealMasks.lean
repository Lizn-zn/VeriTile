import bench.tritonbench_g.bmm_optimized.Proofs.RealSetup

/-! Exact source mask-register setup for all eight constexpr combinations.
One-axis masks retain their source shape and are broadcast at the load site;
mask elision stays MaskOpt.none. No implicit zero default is inserted. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealMasks
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def aCode (BM BK : Nat) (dm dk : Bool) : List Stmt :=
  match dm, dk with
  | Bool.false, Bool.false => [.assign .bool [BM,BK] "mask_a"
      (.boolAnd Broadcast.nil.consL.consR
        (.expandDim ⟨1,by simp⟩ (.ref .bool [BM] "mask_m"))
        (.expandDim ⟨0,by simp⟩ (.ref .bool [BK] "mask_k")))]
  | Bool.false, Bool.true => [.assign .bool [BM,1] "mask_a"
      (.expandDim ⟨1,by simp⟩ (.ref .bool [BM] "mask_m"))]
  | Bool.true, Bool.false => [.assign .bool [1,BK] "mask_a"
      (.expandDim ⟨0,by simp⟩ (.ref .bool [BK] "mask_k"))]
  | Bool.true, Bool.true => []

def bCode (BN BK : Nat) (dn dk : Bool) : List Stmt :=
  match dn, dk with
  | Bool.false, Bool.false => [.assign .bool [BK,BN] "mask_b"
      (.boolAnd Broadcast.nil.consL.consR
        (.expandDim ⟨1,by simp⟩ (.ref .bool [BK] "mask_k"))
        (.expandDim ⟨0,by simp⟩ (.ref .bool [BN] "mask_n")))]
  | Bool.false, Bool.true => [.assign .bool [1,BN] "mask_b"
      (.expandDim ⟨0,by simp⟩ (.ref .bool [BN] "mask_n"))]
  | Bool.true, Bool.false => [.assign .bool [BK,1] "mask_b"
      (.expandDim ⟨1,by simp⟩ (.ref .bool [BK] "mask_k"))]
  | Bool.true, Bool.true => []

def code (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  (if p.divK then [] else [.assign .bool [BK] "mask_k"
    (.lt .nat .scalarR (.ref .nat [BK] "offs_k") (.constNat p.K))]) ++
    aCode BM BK p.divM p.divK ++ bCode BN BK p.divN p.divK

def aMask (BM BK : Nat) (dm dk : Bool) : MaskOpt .real [BM,BK] :=
  match dm, dk with
  | Bool.false, Bool.false => .mask (.ref .bool [BM,BK] "mask_a")
  | Bool.false, Bool.true => .mask (.remap [BM,BK]
      (fun i => (i.1,0,PUnit.unit)) (.ref .bool [BM,1] "mask_a"))
  | Bool.true, Bool.false => .mask (.remap [BM,BK]
      (fun i => (0,i.2.1,PUnit.unit)) (.ref .bool [1,BK] "mask_a"))
  | Bool.true, Bool.true => .none

def bMask (BN BK : Nat) (dn dk : Bool) : MaskOpt .real [BK,BN] :=
  match dn, dk with
  | Bool.false, Bool.false => .mask (.ref .bool [BK,BN] "mask_b")
  | Bool.false, Bool.true => .mask (.remap [BK,BN]
      (fun i => (0,i.2.1,PUnit.unit)) (.ref .bool [1,BN] "mask_b"))
  | Bool.true, Bool.false => .mask (.remap [BK,BN]
      (fun i => (i.1,0,PUnit.unit)) (.ref .bool [BK,1] "mask_b"))
  | Bool.true, Bool.true => .none

def aResult (s : BlockState) {BM BK : Nat} (dm dk : Bool)
    (row : Fin BM → Nat) (k : Fin BK → Nat) (M K : Nat) : BlockState :=
  match dm, dk with
  | Bool.false, Bool.false => s.setReg "mask_a" .bool [BM,BK]
      (Tile.mat fun i j => decide (row i < M ∧ k j < K))
  | Bool.false, Bool.true => s.setReg "mask_a" .bool [BM,1]
      (Tile.mat fun i _ => decide (row i < M))
  | Bool.true, Bool.false => s.setReg "mask_a" .bool [1,BK]
      (Tile.mat fun _ j => decide (k j < K))
  | Bool.true, Bool.true => s

def bResult (s : BlockState) {BN BK : Nat} (dn dk : Bool)
    (col : Fin BN → Nat) (k : Fin BK → Nat) (N K : Nat) : BlockState :=
  match dn, dk with
  | Bool.false, Bool.false => s.setReg "mask_b" .bool [BK,BN]
      (Tile.mat fun i j => decide (k i < K ∧ col j < N))
  | Bool.false, Bool.true => s.setReg "mask_b" .bool [1,BN]
      (Tile.mat fun _ j => decide (col j < N))
  | Bool.true, Bool.false => s.setReg "mask_b" .bool [BK,1]
      (Tile.mat fun i _ => decide (k i < K))
  | Bool.true, Bool.true => s

def result (s : BlockState) {BM BN BK : Nat} (p : RealProgram.Parameters)
    (row : Fin BM → Nat) (col : Fin BN → Nat) (k : Fin BK → Nat) : BlockState :=
  let t := if p.divK then s else s.setReg "mask_k" .bool [BK] (Tile.vec fun i => decide (k i < p.K))
  bResult (aResult t p.divM p.divK row k p.M p.K) p.divN p.divK col k p.N p.K

theorem executes (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (row : Fin BM → Nat) (col : Fin BN → Nat) (k : Fin BK → Nat)
    (hm : p.divM = Bool.false → s.regs .bool [BM] "mask_m" = some (Tile.vec fun i => decide (row i < p.M)))
    (hn : p.divN = Bool.false → s.regs .bool [BN] "mask_n" = some (Tile.vec fun j => decide (col j < p.N)))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec k)) :
    stepStmts (code BM BN BK p) s = some (result s p row col k) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk
  all_goals
    simp only [Bool.false_eq_true, false_implies, true_implies] at hm hn
    simp [code, aCode, bCode, result, aResult, bResult, stepStmts, stepStmt,
      evalOp.eq_def, bind, Option.bind, hk, hm, hn,
      Tile.cop, Tile.bop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
      ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
      -Tile.scalar_eta]
    try rfl

theorem state_frame (s : BlockState) {BM BN BK : Nat} (p : RealProgram.Parameters)
    (row : Fin BM → Nat) (col : Fin BN → Nat) (k : Fin BK → Nat) :
    {result s p row col k with regs := s.regs} = s := by
  cases hm : p.divM <;> cases hn : p.divN <;> cases hk : p.divK <;>
    simp [result, aResult, bResult, hm, hn, hk, BlockState.setReg]

theorem register_frame (s : BlockState) {BM BN BK : Nat} (p : RealProgram.Parameters)
    (row : Fin BM → Nat) (col : Fin BN → Nat) (k : Fin BK → Nat)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["mask_k","mask_a","mask_b"]) :
    (result s p row col k).regs dtype shape name = s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  cases hm : p.divM <;> cases hcol : p.divN <;> cases hk : p.divK <;>
    simp [result, aResult, bResult, hm, hcol, hk, BlockState.setReg, hn.1, hn.2.1, hn.2.2]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealMasks
