import bench.tritonbench_g.chunk_linear_attn.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
import VeriTile.Triton

/-! Precision erasure of the complete forward state source; no padding,
store order or loop bound is changed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then forwardHDotLeft else forwardH
variable (swapped : Bool) (dot : Storage)
    (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hasSeed storeFinal : Bool)

theorem projects :
    (source swapped dot dtype out_dtype k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV NT hasSeed storeFinal).toAlgorithm? =
    .ok (source swapped dot dtype out_dtype k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV NT hasSeed storeFinal).toAlgKernel := by
  cases swapped <;> cases dot <;> rfl

theorem storage_erasure :
    (source swapped dot dtype out_dtype k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV NT hasSeed storeFinal).eraseDType.toAlgKernel.body =
    (source swapped .fp32 .real .real k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV NT hasSeed storeFinal).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, forwardH,
    forwardHDotLeft, forwardHSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases dtype <;> cases out_dtype <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHSource

/-! Independent prefix contraction from initial K/V/seed arrays.
Zero extension is justified only under the later explicit padding premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHMath
open scoped BigOperators
noncomputable section

structure Layout where
  keyHead : Nat
  keyTime : Nat
  keyFeature : Nat
  valueHead : Nat
  valueTime : Nat
  valueFeature : Nat
  historyHead : Nat
  historyKey : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  NT : Nat

structure Launch where
  value : Nat
  key : Nat
  head : Nat

structure Inputs where
  key : Nat → ℝ
  value : Nat → ℝ
  seed : Nat → ℝ

def keyIndex (p : Layout) (l : Launch) (e : Nat) := l.key * p.BK + e
def valueIndex (p : Layout) (l : Launch) (j : Nat) := l.value * p.BV + j
def timeIndex (p : Layout) (c t : Nat) := c * p.BT + t

def keyAddress (p : Layout) (l : Launch) (k t : Nat) :=
  l.head * p.keyHead + k * p.keyFeature + t * p.keyTime

def valueAddress (p : Layout) (l : Launch) (t v : Nat) :=
  l.head * p.valueHead + t * p.valueTime + v * p.valueFeature

def stateAddress (p : Layout) (l : Launch) (k v : Nat) :=
  l.head * p.K * p.V + k * p.V + v

def historyAddress (p : Layout) (l : Launch) (c k v : Nat) :=
  l.head * p.historyHead + c * p.K * p.V + k * p.historyKey + v



def keyValue (p : Layout) (l : Launch) (a : Inputs) (k t : Nat) : ℝ :=
  if k < p.K ∧ t < p.T then a.key (keyAddress p l k t) else 0

def valueValue (p : Layout) (l : Launch) (a : Inputs) (t v : Nat) : ℝ :=
  if t < p.T ∧ v < p.V then a.value (valueAddress p l t v) else 0


def seed (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (k v : Nat) : ℝ :=
  if hasSeed ∧ k < p.K ∧ v < p.V then a.seed (stateAddress p l k v) else 0


/-- The whole checked BT span contributes to each state lane. -/
def contribution (p : Layout) (l : Launch) (a : Inputs) (c k v : Nat) : ℝ :=
  ∑ lane : Fin p.BT,
    keyValue p l a k (timeIndex p c lane.val) * valueValue p l a (timeIndex p c lane.val) v

/-- State before chunk c; the optional final state uses c=NT. -/
def closed (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (c k v : Nat) : ℝ :=
  seed p l a hasSeed k v + ∑ t ∈ Finset.range c, contribution p l a t k v

@[simp] theorem closed_zero (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (k v : Nat) :
    closed p l a hasSeed 0 k v = seed p l a hasSeed k v := by simp [closed]

theorem closed_step (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (c k v : Nat) :
    closed p l a hasSeed (c+1) k v = closed p l a hasSeed c k v + contribution p l a c k v := by
  simp only [closed,Finset.sum_range_succ,add_assoc]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHMath

/-! Exact erased source decomposition, including pre-update history and optional seed/final state. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHProgram
open VeriTile.Triton RealHMath
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Regions where
  key : RegionName
  value : RegionName
  history : RegionName
  seed : RegionName
  final : RegionName

def launch (u : BlockState) : Launch := ⟨u.pids 1,u.pids 0,u.pids 2⟩
def times (name : RegName) (n : Nat) : Op .nat [] := .mul .nat .nil (.ref .nat [] name) (.constNat n)
def keyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BT] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.keyHead) [p.K,p.T] [p.BK,p.BT]
    [p.keyFeature,p.keyTime] [times "i_k" p.BK,times "i_t" p.BT]
def valuePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BV] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.valueHead) [p.T,p.V] [p.BT,p.BV]
    [p.valueTime,p.valueFeature] [times "i_t" p.BT,times "i_v" p.BV]
def stateBase (p : Layout) : Op .nat [] := .mul .nat .nil (times "i_bh" p.K) (.constNat p.V)
def statePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynOffsets r (stateBase p) [p.K,p.V] [p.BK,p.BV]
    [p.V,1] [times "i_k" p.BK,times "i_v" p.BV]
def historyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynOffsets r
    (.add .nat .nil (times "i_bh" p.historyHead)
      (.mul .nat .nil (times "i_t" p.K) (.constNat p.V)))
    [p.K,p.V] [p.BK,p.BV] [p.historyKey,1] [times "i_k" p.BK,times "i_v" p.BV]


def load2 (a b : Nat) (name : RegName) : Op .real [a,b] :=
  .load .real (.blockPtr (.ref .blockPtr [a,b] name) [0,1]) .none .unspecified

def carry (p : Layout) : Op .real [p.BK,p.BV] := .ref .real [p.BK,p.BV] "b_h"
def dot (p : Layout) : Op .real [p.BK,p.BV] :=
  .dot (batch := []) (.ref .real [p.BK,p.BT] "b_k") (.ref .real [p.BT,p.BV] "b_v")
    {allowTF32 := some Bool.false}

def initialization (p : Layout) (r : Regions) (hasSeed : Bool) : List Stmt :=
  [.assign .nat [] "i_k" (.programId 0), .assign .nat [] "i_v" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2), .assign .real [p.BK,p.BV] "b_h" (.full [p.BK,p.BV] (.const 0)),
   .ifThen (.constBool hasSeed)
     [.assign .blockPtr [p.BK,p.BV] "p_h0" (statePtr p r.seed),
      .assign .real [p.BK,p.BV] "b_h"
        (load2 p.BK p.BV "p_h0")]]



def storeCarry (p : Layout) : Stmt :=
  .store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] "p_h") [0,1]) (carry p) .none

def iteration (p : Layout) (r : Regions) (swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_k" (keyPtr p r.key),
   .assign .blockPtr [p.BT,p.BV] "p_v" (valuePtr p r.value),
   .assign .blockPtr [p.BK,p.BV] "p_h" (historyPtr p r.history),
   storeCarry p,
   .assign .real [p.BK,p.BT] "b_k" (load2 p.BK p.BT "p_k"),
   .assign .real [p.BT,p.BV] "b_v" (load2 p.BT p.BV "p_v"),
   .assign .real [p.BK,p.BV] "b_h"
     (if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
      else .add .real (.consSame (.consSame .nil)) (carry p) (dot p))]

def finalization (p : Layout) (r : Regions) (storeFinal : Bool) : List Stmt :=
  [.ifThen (.constBool storeFinal)
    [.assign .blockPtr [p.BK,p.BV] "p_ht" (statePtr p r.final),
     .store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] "p_ht") [0,1]) (carry p) .none]]

def body (p : Layout) (r : Regions) (hasSeed storeFinal swapped : Bool) : List Stmt :=
  initialization p r hasSeed ++ [.forRangeDyn "i_t" (.constNat 0) (.constNat p.NT) (.constNat 1) (iteration p r swapped)] ++ finalization p r storeFinal

def source (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool) :=
  RealHSource.source swapped dot seed final r.key r.value r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal

theorem source_body (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool) :
    (source p r dot seed final hasSeed storeFinal swapped).eraseDType.toAlgKernel.body =
      body p r hasSeed storeFinal swapped := by
  unfold source
  rw [RealHSource.storage_erasure]
  cases swapped
  all_goals simp only [RealHSource.source, Bool.false_eq_true, ↓reduceIte,
    forwardH, forwardHDotLeft, forwardHSource,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHProgram

/-! Pointer evaluation and unmasked checked loads for the complete H source.
Both row and column strides remain symbolic. Loads preserve arbitrary real
unspecified values; the zero-extension premise is applied later, separately. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHPointers
open VeriTile.Triton RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def matrixPointer (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat) : Tile .blockPtr [nr,nc] :=
  ⟨fun _ => ⟨r,base,[rows,cols],[nr,nc],[sr,sc],[offR,offC]⟩⟩

def keyPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT)
def valuePointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV)
def statePointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrixPointer r (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV)
def historyPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BK p.BV p.historyKey 1 (l.key*p.BK) (l.value*p.BV)

theorem times_eval (u : BlockState) (name : RegName) (v n : Nat)
    (h : u.regs .nat [] name = some (Tile.scalar v)) :
    evalOp (times name n) u = some (Tile.scalar (v*n)) := by
  simp only [times,evalOp_mul,evalOp_ref,evalOp_constNat,h,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem make2 (u : BlockState) (r : RegionName) (base x y : Op .nat [])
    (rows cols nr nc sr sc b i j : Nat)
    (hb : evalOp base u = some (Tile.scalar b))
    (hi : evalOp x u = some (Tile.scalar i)) (hj : evalOp y u = some (Tile.scalar j)) :
    evalOp (.makeBlockPtrDynOffsets r base [rows,cols] [nr,nc] [sr,sc] [x,y]) u =
      some (matrixPointer r b rows cols nr nc sr sc i j) := by
  simp only [makeBlockPtr2_eval,hb,hi,hj,List.mapM_cons,List.mapM_nil,Option.bind_eq_bind,
    Option.bind_some,Tile.scalar]
  rfl

theorem key_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (keyPtr p r) u = some (keyPointer p l r c) :=
  make2 u r _ _ _ _ _ _ _ _ _ _ _ _ (times_eval u _ _ _ hh) (times_eval u _ _ _ hk) (times_eval u _ _ _ hc)

theorem value_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (valuePtr p r) u = some (valuePointer p l r c) :=
  make2 u r _ _ _ _ _ _ _ _ _ _ _ _ (times_eval u _ _ _ hh) (times_eval u _ _ _ hc) (times_eval u _ _ _ hv)

theorem stateBase_eval (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    evalOp (stateBase p) u = some (Tile.scalar (l.head*p.K*p.V)) := by
  rw [stateBase,evalOp_mul,times_eval u _ _ _ hh,evalOp_constNat]
  rfl

theorem state_eval (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (statePtr p r) u = some (statePointer p l r) :=
  make2 u r _ _ _ _ _ _ _ _ _ _ _ _ (stateBase_eval p l u hh) (times_eval u _ _ _ hk) (times_eval u _ _ _ hv)

theorem history_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (historyPtr p r) u = some (historyPointer p l r c) := by
  apply make2
  · simp only [evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,times,hh,hc,Option.bind_eq_bind,Option.bind_some]
    rfl
  · exact times_eval u _ _ _ hk
  · exact times_eval u _ _ _ hv

/-- No padding value is assumed here: each inactive lane reads its oracle at
that lane's actual address. -/
theorem load2_eval (r : RegionName) (u : BlockState) (name : RegName)
    (base rows cols nr nc sr sc offR offC : Nat)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrixPointer r base rows cols nr nc sr sc offR offC)) :
    evalOp (load2 nr nc name) u = some (Tile.mat (fun (i : Fin nr) (j : Fin nc) =>
      if offR+i.val < rows ∧ offC+j.val < cols then
        u.readMem r (base+(offR+i.val)*sr+(offC+j.val)*sc)
      else u.undef r (base+(offR+i.val)*sr+(offC+j.val)*sc))) := by
  simp only [load2,evalOp,hp,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [matrixPointer,Tile.mat,TileShape.indexToList,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,BlockState.readMemValue_real]
  by_cases h : offR+i.val < rows ∧ offC+j.val < cols
  · simp only [h,decide_true,if_true]
    rfl
  · simp [h,BlockState.paddingValue,BlockState.unspecifiedValue]
    rfl

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHPointers

/-! Actual real values loaded at a chunk, including unspecified tails. These
values contain no execution state or assumed output. The zero-oracle bridge
relates their arithmetic update to the independently stated closed form. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHTiles
open VeriTile.Triton RealHMath RealHProgram RealHPointers
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

abbrev StateValue (p : Layout) := Fin p.BK → Fin p.BV → ℝ
def stateTile (p : Layout) (f : StateValue p) : Tile .real [p.BK,p.BV] := Tile.mat (fun e j => some (f e j))

def keyData (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (e : Fin p.BK) (t : Fin p.BT) : ℝ :=
  let k := keyIndex p l e.val
  let time := timeIndex p c t.val
  let off := keyAddress p l k time
  if k < p.K ∧ time < p.T then a off else padding off

def valueData (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (t : Fin p.BT) (j : Fin p.BV) : ℝ :=
  let time := timeIndex p c t.val
  let v := valueIndex p l j.val
  let off := valueAddress p l time v
  if time < p.T ∧ v < p.V then a off else padding off

structure Loaded (p : Layout) where
  key : Fin p.BK → Fin p.BT → ℝ
  value : Fin p.BT → Fin p.BV → ℝ

def loaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) : Loaded p where
  key := keyData p l a.key (padding r.key) c
  value := valueData p l a.value (padding r.value) c

def product (x : Loaded p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  ∑ t : Fin p.BT, x.key e t * x.value t j

def advance (x : Loaded p) (f : StateValue p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  f e j + product x e j

def expected (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (c : Nat) : StateValue p :=
  fun e j => closed p l a hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)

theorem product_zero_padding (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (e : Fin p.BK) (j : Fin p.BV) :
    product (loaded p l r a (fun _ _ => 0) c) e j =
      contribution p l a c (keyIndex p l e.val) (valueIndex p l j.val) := by
  rfl

theorem advance_expected (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (hasSeed : Bool) :
    advance (loaded p l r a (fun _ _ => 0) c) (expected p l a hasSeed c) = expected p l a hasSeed (c+1) := by
  funext e j
  simp only [advance,product_zero_padding,expected,closed_step]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHTiles

/-! The source loads produce the independently indexed tiles, with each
pointer's exact boundary checks. Inactive lanes retain arbitrary oracle values.
 -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHLoads
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

theorem key_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : u.undef r = padding)
    (hp : u.regs .blockPtr [p.BK,p.BT] name = some (keyPointer p l r c)) :
    evalOp (load2 p.BK p.BT name) u = some (Tile.mat (fun e t => some (keyData p l a padding c e t))) := by
  rw [load2_eval r u name (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,t,⟨⟩⟩
  simp only [Tile.mat,keyData,keyAddress,keyIndex,timeIndex,hm,hu]
  split <;> rfl

theorem value_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : u.undef r = padding)
    (hp : u.regs .blockPtr [p.BT,p.BV] name = some (valuePointer p l r c)) :
    evalOp (load2 p.BT p.BV name) u = some (Tile.mat (fun t j => some (valueData p l a padding c t j))) := by
  rw [load2_eval r u name (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨t,j,⟨⟩⟩
  simp only [Tile.mat,valueData,valueAddress,valueIndex,timeIndex,hm,hu]
  split <;> rfl

def seedData (p : Layout) (l : Launch) (a padding : Nat → ℝ) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  let k := keyIndex p l e.val
  let v := valueIndex p l j.val
  if k < p.K ∧ v < p.V then a (stateAddress p l k v) else padding (stateAddress p l k v)

theorem seed_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : u.undef r = padding)
    (hp : u.regs .blockPtr [p.BK,p.BV] name = some (statePointer p l r)) :
    evalOp (load2 p.BK p.BV name) u = some (stateTile p (seedData p l a padding)) := by
  rw [load2_eval r u name (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [stateTile,Tile.mat,seedData,stateAddress,keyIndex,valueIndex,hm,hu,Nat.mul_one]
  split <;> rfl

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHLoads

/-! Masked state-matrix stores used for both pre-update history and final state.
The base and both strides distinguish their layouts. Readback requires only
active lanes to have distinct addresses; the frame preserves complete cells. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHStore
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def active (p : Layout) (l : Launch) (e j : Nat) : Prop :=
  keyIndex p l e < p.K ∧ valueIndex p l j < p.V
instance (p : Layout) (l : Launch) (e j : Nat) : Decidable (active p l e j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def address (p : Layout) (l : Launch) (base rowStride colStride e j : Nat) : Nat :=
  base + keyIndex p l e * rowStride + valueIndex p l j * colStride

def pointer (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat) :
    Tile .blockPtr [p.BK,p.BV] :=
  ⟨fun _ => BlockPtr.mk out base [p.K,p.V] [p.BK,p.BV] [rowStride,colStride]
    [l.key * p.BK,l.value * p.BV]⟩

def scatter (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BK,p.BV]).foldl (fun st idx =>
    if active p l idx.1.val idx.2.1.val then
      st.writeMem out (address p l base rowStride colStride idx.1.val idx.2.1.val) (frozen idx.1 idx.2.1)
    else st) u

def footprint (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat) : WriteFootprint :=
  fun cell => cell.1 = out ∧ ∃ (e : Fin p.BK) (j : Fin p.BV), active p l e.val j.val ∧
    cell.2 = address p l base rowStride colStride e.val j.val

def NoCollision (p : Layout) (l : Launch) (base rowStride colStride : Nat) : Prop :=
  ∀ (e f : Fin p.BK) (i j : Fin p.BV), active p l e.val i.val → active p l f.val j.val →
    address p l base rowStride colStride e.val i.val = address p l base rowStride colStride f.val j.val → e = f ∧ i = j

theorem store_eval (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (nm : RegName) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen))
    (hp : u.regs .blockPtr [p.BK,p.BV] nm = some (pointer p l out base rowStride colStride)) :
    stepStmt (.store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] nm) [0,1])
      (.ref .real [p.BK,p.BV] "b_h") .none) u = some (scatter p l out base rowStride colStride frozen u) := by
  unfold stepStmt scatter
  simp only [evalOp_ref, hh, hp, Option.bind, Option.map]
  apply congrArg some
  apply congrArg (fun f => List.foldl f u (TileShape.allIndices [p.BK,p.BV]))
  funext st idx
  rcases idx with ⟨e,j,⟨⟩⟩
  simp only [pointer, TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, active, keyIndex, valueIndex, address]
  by_cases h : l.key * p.BK + e.val < p.K ∧ l.value * p.BV + j.val < p.V
  · simp only [h, decide_true, if_true, BlockState.writeMemTyped_real]; rfl
  · simp only [h, decide_false, Bool.false_eq_true, if_false]

theorem readback (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (u : BlockState) (hinj : NoCollision p l base rowStride colStride)
    (e : Fin p.BK) (j : Fin p.BV) (ha : active p l e.val j.val) :
    (scatter p l out base rowStride colStride frozen u).readMem out (address p l base rowStride colStride e.val j.val) = frozen e j := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true (region := out) u
    (fun idx : TileIndex [p.BK,p.BV] => address p l base rowStride colStride idx.1.val idx.2.1.val)
    (fun idx => frozen idx.1 idx.2.1) (fun idx => active p l idx.1.val idx.2.1.val)
    (e,j,PUnit.unit) ha (by
      rintro ⟨f,i,⟨⟩⟩ hi heq
      obtain ⟨hf,hj⟩ := hinj f e i j hi ha heq
      subst f; subst i; rfl)

theorem frame (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l out base rowStride colStride) u (scatter p l out base rowStride colStride frozen u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl,idx.1,idx.2.1,ha,rfl⟩

@[simp] theorem scatter_undef (p : Layout) (l : Launch) (out : RegionName)
    (base rowStride colStride : Nat) (frozen : StateValue p) (u : BlockState) :
    (scatter p l out base rowStride colStride frozen u).undef = u.undef := by
  unfold scatter
  generalize TileShape.allIndices [p.BK,p.BV] = lanes
  induction lanes generalizing u with
  | nil => rfl
  | cons i is ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHStore

/-! Real contraction and carry update, after the actual source loads. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHAlgebra
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def updateOp (p : Layout) (swapped : Bool) : Op .real [p.BK,p.BV] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
  else .add .real (.consSame (.consSame .nil)) (carry p) (dot p)

theorem product_eval (p : Layout) (x : Loaded p) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (x.key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (x.value t j)))) :
    evalOp (dot p) u = some (stateTile p (product x)) := by
  unfold dot
  erw [evalOp_dot]
  erw [evalOp_ref, hk, evalOp_ref, hv]
  simp only [Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [Tile.mat,Option.map₂_some_some,stateTile,product]
  exact (WithBot.coe_sum Finset.univ (fun t : Fin p.BT => x.key e t * x.value t j)).symm

theorem update_eval (p : Layout) (x : Loaded p) (swapped : Bool) (f : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hk : u.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (x.key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (x.value t j)))) :
    evalOp (updateOp p swapped) u = some (stateTile p (advance x f)) := by
  cases swapped
  all_goals
    simp only [updateOp,Bool.false_eq_true,↓reduceIte,evalOp_add,carry,evalOp_ref,hh,
      product_eval p x u hk hv,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨e,j,⟨⟩⟩
    simp [advance,stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.add,add_comm]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealHAlgebra

#guard_msgs (drop info) in
#auditModuleAxioms
