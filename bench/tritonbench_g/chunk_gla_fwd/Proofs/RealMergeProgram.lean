import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Exact real decomposition of both complete merge sources. The independent
formula reads each NK slice of the initial temporary array. Disabled chunks
perform no stores; all addresses retain the runtime head count. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeProgram
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

structure Layout where
  T : Nat
  BT : Nat
  BC : Nat
  NK : Nat
structure Launch where
  chunk : Nat
  subblock : Nat
  head : Nat
  heads : Nat

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2,s.numPids 2⟩
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
def inputAddress (p : Layout) (l : Launch) (kb : Nat) (i : TileIndex [p.BC,p.BC]) :=
  (l.head+kb*l.heads)*p.T*p.BC+(rowStart p l+i.1.val)*p.BC+i.2.1.val
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
def expected (p : Layout) (l : Launch) (a : Nat → ℝ) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range p.NK, a (inputAddress p l kb i)
def footprint (p : Layout) (l : Launch) (out : RegionName) : WriteFootprint :=
  fun cell => cell.1 = out ∧ enabled p l ∧ ∃ i, active p l i ∧ cell.2 = outputAddress p l i
def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ i j, active p l i → active p l j → outputAddress p l i = outputAddress p l j → i = j

def mulRef (name : RegName) (n : Nat) : Op .nat [] := .mul .nat .nil (.ref .nat [] name) (.constNat n)
def rowOp (p : Layout) : Op .nat [] := .add .nat .nil (mulRef "i_t" p.BT) (mulRef "i_c" p.BC)
def condition (p : Layout) : Op .bool [] := .boolNot (.ge .nat .nil (rowOp p) (.constNat p.T))
def inputBase (p : Layout) : Op .nat [] :=
  .mul .nat .nil (.mul .nat .nil
    (.add .nat .nil (.ref .nat [] "i_bh") (.mul .nat .nil (.ref .nat [] "i_k") (.ref .nat [] "n_bh")))
    (.constNat p.T)) (.constNat p.BC)
def outputBase (p : Layout) : Op .nat [] := .mul .nat .nil (mulRef "i_bh" p.T) (.constNat p.BT)
def inputPointer (p : Layout) (A : RegionName) : Op .blockPtr [p.BC,p.BC] :=
  .makeBlockPtrDynOffsets A (inputBase p) [p.T,p.BC] [p.BC,p.BC] [p.BC,1] [rowOp p,.constNat 0]
def outputPointer (p : Layout) (A2 : RegionName) : Op .blockPtr [p.BC,p.BC] :=
  .makeBlockPtrDynOffsets A2 (outputBase p) [p.T,p.BT] [p.BC,p.BC] [p.BT,1] [rowOp p,mulRef "i_c" p.BC]
def carry (p : Layout) : Op .real [p.BC,p.BC] := .ref .real [p.BC,p.BC] "b_A"
def loaded (p : Layout) : Op .real [p.BC,p.BC] :=
  .load .real (.blockPtr (.ref .blockPtr [p.BC,p.BC] "p_A") [0,1]) .none .unspecified

def initialization : List Stmt :=
  [.assign .nat [] "i_t" (.programId 0),.assign .nat [] "i_c" (.programId 1),.assign .nat [] "i_bh" (.programId 2)]
def setup (p : Layout) : List Stmt :=
  [.assign .nat [] "n_bh" (.numPrograms 2),.assign .real [p.BC,p.BC] "b_A" (.full [p.BC,p.BC] (.const 0))]
def iteration (p : Layout) (A : RegionName) (swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BC,p.BC] "p_A" (inputPointer p A),
   .assign .real [p.BC,p.BC] "b_A" (if swapped then .add .real (.consSame (.consSame .nil)) (loaded p) (carry p)
     else .add .real (.consSame (.consSame .nil)) (carry p) (loaded p))]
def loop (p : Layout) (A : RegionName) (swapped : Bool) : Stmt :=
  .forRangeDyn "i_k" (.constNat 0) (.constNat p.NK) (.constNat 1) (iteration p A swapped)
def finalization (p : Layout) (A2 : RegionName) : List Stmt :=
  [.assign .blockPtr [p.BC,p.BC] "p_A2" (outputPointer p A2),
   .store .real [p.BC,p.BC] (.blockPtr (.ref .blockPtr [p.BC,p.BC] "p_A2") [0,1]) (carry p) .none]
def body (p : Layout) (A A2 : RegionName) (swapped : Bool) : List Stmt :=
  initialization ++ [.ifThen (condition p) (setup p ++ [loop p A swapped] ++ finalization p A2)]
def source (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) : ComputeKernel :=
  if swapped then TypedMerge.loadFirst out A A2 p.T p.BT p.BC p.NK
  else TypedMerge.original out A A2 p.T p.BT p.BC p.NK

theorem projects (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) :
    (source p A A2 out swapped).toAlgorithm? = .ok (source p A A2 out swapped).toAlgKernel := by
  cases swapped <;> rfl

theorem source_body (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) :
    (source p A A2 out swapped).eraseDType.toAlgKernel.body = body p A A2 swapped := by
  cases swapped <;> cases out
  all_goals simp only [source,TypedMerge.original,TypedMerge.loadFirst,Bool.false_eq_true,↓reduceIte,
    ComputeKernel.eraseDType,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.map_cons,
    List.map_nil,eraseDType_float,VeriTile.Triton.eraseDType]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType,eraseDType_float,VeriTile.Triton.eraseDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeProgram
