import bench.tritonbench_g.bmm_optimized.Proofs.RealMasks
import bench.tritonbench_g.bmm_optimized.Proofs.RealMath

/-! Actual loads for every constexpr mask choice. A masked lane retains the
state's unspecified load value; zero fill is not part of these execution
lemmas. The source's pointer and lane advances precede its dot/add update. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealLoads
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def aLoad (BM BK : Nat) (p : RealProgram.Parameters) : Op .real [BM,BK] :=
  .load .real (.ptr (.ref .ptr [BM,BK] "a_ptrs")) (RealMasks.aMask BM BK p.divM p.divK)
def bLoad (BN BK : Nat) (p : RealProgram.Parameters) : Op .real [BK,BN] :=
  .load .real (.ptr (.ref .ptr [BK,BN] "b_ptrs")) (RealMasks.bMask BN BK p.divN p.divK)

def aElement {BM BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BM,BK]) (rows : Fin BM → Nat) (ks : Fin BK → Nat)
    (i : Fin BM) (k : Fin BK) : ℝ :=
  let addr := ptrs.data (i,k,PUnit.unit)
  if RealMath.active p.divM (rows i) p.M ∧ RealMath.active p.divK (ks k) p.K then
    s.readMem addr.1 addr.2 else s.undef addr.1 addr.2

def bElement {BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BK,BN]) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (k : Fin BK) (j : Fin BN) : ℝ :=
  let addr := ptrs.data (k,j,PUnit.unit)
  if RealMath.active p.divK (ks k) p.K ∧ RealMath.active p.divN (cols j) p.N then
    s.readMem addr.1 addr.2 else s.undef addr.1 addr.2

def aValues {BM BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BM,BK]) (rows : Fin BM → Nat) (ks : Fin BK → Nat) : Tile .real [BM,BK] :=
  Tile.mat fun i k => some (aElement s p ptrs rows ks i k)
def bValues {BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BK,BN]) (cols : Fin BN → Nat) (ks : Fin BK → Nat) : Tile .real [BK,BN] :=
  Tile.mat fun k j => some (bElement s p ptrs cols ks k j)

theorem a_load (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BM,BK]) (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (hp : s.regs .ptr [BM,BK] "a_ptrs" = some ptrs) :
    evalOp (aLoad BM BK p) (RealMasks.result s p rows cols ks) =
      some (aValues s p ptrs rows ks) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk
  all_goals
    simp [aLoad, RealMasks.aMask, RealMasks.result, RealMasks.aResult, RealMasks.bResult,
      evalOp.eq_def, hp, bind, Option.bind]
    apply Tile.ext
    rintro ⟨i,k,⟨⟩⟩
    by_cases hr : rows i < M <;> by_cases hk : ks k < K <;>
      simp [aValues, aElement, RealMath.active, Tile.mat, Tile.remap, hr, hk]

theorem b_load (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (hp : s.regs .ptr [BK,BN] "b_ptrs" = some bp) :
    evalOp (bLoad BN BK p)
      ((RealMasks.result s p rows cols ks).setReg "a" .real [BM,BK] (aValues s p ap rows ks)) =
      some (bValues s p bp cols ks) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk
  all_goals
    simp [bLoad, RealMasks.bMask, RealMasks.result, RealMasks.aResult, RealMasks.bResult,
      evalOp.eq_def, hp, bind, Option.bind]
    apply Tile.ext
    rintro ⟨k,j,⟨⟩⟩
    by_cases hc : cols j < N <;> by_cases hk : ks k < K <;>
      simp [bValues, bElement, RealMath.active, Tile.mat, Tile.remap, hc, hk]

def code (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  RealMasks.code BM BN BK p ++ [.assign .real [BM,BK] "a" (aLoad BM BK p),
    .assign .real [BK,BN] "b" (bLoad BN BK p)]

def result {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat) : BlockState :=
  let t := (RealMasks.result s p rows cols ks).setReg "a" .real [BM,BK] (aValues s p ap rows ks)
  t.setReg "b" .real [BK,BN] (bValues s p bp cols ks)

theorem executes (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (hm : p.divM = Bool.false → s.regs .bool [BM] "mask_m" = some (Tile.vec fun i => decide (rows i < p.M)))
    (hn : p.divN = Bool.false → s.regs .bool [BN] "mask_n" = some (Tile.vec fun j => decide (cols j < p.N)))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec ks))
    (ha : s.regs .ptr [BM,BK] "a_ptrs" = some ap)
    (hb : s.regs .ptr [BK,BN] "b_ptrs" = some bp) :
    stepStmts (code BM BN BK p) s = some (result s p ap bp rows cols ks) := by
  rw [code, stepStmts.append_some (RealMasks.executes s BM BN BK p rows cols ks hm hn hk)]
  simp only [bind, Option.bind_some, stepStmts, stepStmt,
    a_load s BM BN BK p ap rows cols ks ha, b_load s BM BN BK p ap bp rows cols ks hb]
  rfl

theorem state_frame {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat) :
    {result s p ap bp rows cols ks with regs := s.regs} = s :=
  RealMasks.state_frame s p rows cols ks

theorem register_frame {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["mask_k","mask_a","mask_b","a","b"]) :
    (result s p ap bp rows cols ks).regs dtype shape name = s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  simp only [result, BlockState.setReg, hn.2.2.2.1, hn.2.2.2.2, ↓reduceIte]
  exact RealMasks.register_frame s p rows cols ks dtype shape name (by simp [hn])

def update (swapped : Bool) (BM BN BK : Nat) : Op .real [BM,BN] :=
  let dot := Op.dot (batch := []) (.ref .real [BM,BK] "a") (.ref .real [BK,BN] "b")
    { allowTF32 := some Bool.false }
  if swapped then .add .real (.consSame (.consSame .nil)) dot (.ref .real [BM,BN] "o")
  else .add .real (.consSame (.consSame .nil)) (.ref .real [BM,BN] "o") dot

def tailCode (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .nat [BK] "offs_k" (.add .nat .scalarR (.ref .nat [BK] "offs_k") (.constNat BK)),
   .assign .ptr [BM,BK] "a_ptrs" (.ptrAdd .scalarR (.ref .ptr [BM,BK] "a_ptrs") (.constNat BK)),
   .assign .ptr [BK,BN] "b_ptrs" (.ptrAdd .scalarR (.ref .ptr [BK,BN] "b_ptrs")
     (.mul .nat .nil (.constNat BK) (.constNat p.N))),
   .assign .real [BM,BN] "o" (update swapped BM BN BK)]

def body (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : List Stmt :=
  code BM BN BK p ++ tailCode swapped BM BN BK p

theorem matches_source (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).body = body swapped BM BN BK p := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> cases swapped
  all_goals
    simp [body, code, tailCode, update, aLoad, bLoad, RealMasks.code, RealMasks.aCode,
      RealMasks.bCode, RealMasks.aMask, RealMasks.bMask,
      RealProgram.realPart, RealProgram.parts, RealProgram.source,
      TypedSource.specialize, Source.dynamicRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try simp [Broadcast.leftIndex, funext_iff]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealLoads
