import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_retention.Proofs.Source

/-! Independent real decay formulas. Integer offsets are signed: no truncation
is applied to a last-chunk lane or an off-diagonal score exponent. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.Decay
open VeriTile.Triton
noncomputable section

def exp2 (x : ℝ) : ℝ := Real.exp (x * Real.log 2)
def headBase (H head : Nat) : ℝ :=
  Real.log (1 - exp2 (-5 - (head % H : Nat))) / Real.log 2

def factor (base exponent : ℝ) : ℝ := exp2 (exponent * base)
def query (base : ℝ) (row : Nat) : ℝ := factor base (row + 1)
def key (base : ℝ) (length row : Nat) : ℝ := factor base ((length : ℝ) - row - 1)
def score (base : ℝ) (row column : Nat) : ℝ :=
  if column ≤ row then factor base ((row : ℝ) - column) else 0

def chunkLength (T BT NT chunk : Nat) : Nat :=
  if chunk = NT - 1 ∧ T % BT ≠ 0 then T % BT else BT

theorem factor_pos (base exponent : ℝ) : 0 < factor base exponent :=
  Real.exp_pos _

theorem key_signed (base : ℝ) (length row : Nat) :
    key base length row = factor base (((length : Int) - row - 1 : Int) : ℝ) := by
  simp [key]

theorem score_signed (base : ℝ) (row column : Nat) :
    score base row column =
      if column ≤ row then factor base (((row : Int) - column : Int) : ℝ) else 0 := by
  simp [score]

theorem tail_offset_negative (length row : Nat) (h : length ≤ row) :
    (length : Int) - row - 1 < 0 := by omega

theorem complete_chunk (T BT NT chunk : Nat)
    (h : chunk ≠ NT - 1 ∨ T % BT = 0) : chunkLength T BT NT chunk = BT := by
  rcases h with h | h <;> simp [chunkLength,h]

theorem final_partial_chunk (T BT NT : Nat) (h : T % BT ≠ 0) :
    chunkLength T BT NT (NT - 1) = T % BT := by simp [chunkLength,h]

/-- The head rate is in (0,1) for every natural head index, so its logarithm
has a positive argument. No extra distributional assumption is needed. -/
theorem head_rate_positive (H head : Nat) : 0 < 1 - exp2 (-5 - (head % H : Nat)) := by
  have hlog : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hindex : (0 : ℝ) ≤ (head % H : Nat) := Nat.cast_nonneg _
  have he : (-5 - (head % H : Nat) : ℝ) * Real.log 2 < 0 :=
    mul_neg_of_neg_of_pos (by linarith) hlog
  exact sub_pos.mpr (by simpa [exp2] using Real.exp_lt_one_iff.mpr he)

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.Decay

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHMath
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
  H : Nat
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


def base (p : Layout) (l : Launch) : ℝ := Decay.headBase p.H l.head

def length (p : Layout) (c : Nat) : Nat := Decay.chunkLength p.T p.BT p.NT c

def carryFactor (p : Layout) (l : Launch) (c : Nat) : ℝ :=
  Decay.factor (base p l) (length p c)

/-- The independent decayed Kᵀ·V contribution, including the signed tail
exponent before zero-extended V is multiplied. -/
def contribution (p : Layout) (l : Launch) (a : Inputs) (c k v : Nat) : ℝ :=
  ∑ lane : Fin p.BT,
    keyValue p l a k (timeIndex p c lane.val) *
      (valueValue p l a (timeIndex p c lane.val) v *
        Decay.key (base p l) (length p c) lane.val)

/-- Mathematical recurrence before chunk c. It uses only initial arrays and
per-head decay; there is no implementation execution in this definition. -/
def closed (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) : Nat → Nat → Nat → ℝ
  | 0, k, v => seed p l a hasSeed k v
  | c+1, k, v => carryFactor p l c * closed p l a hasSeed c k v + contribution p l a c k v

@[simp] theorem closed_zero (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (k v : Nat) :
    closed p l a hasSeed 0 k v = seed p l a hasSeed k v := rfl

theorem closed_step (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) (c k v : Nat) :
    closed p l a hasSeed (c+1) k v =
      carryFactor p l c * closed p l a hasSeed c k v + contribution p l a c k v := rfl

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHMath

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHProgram
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
  .dot (batch := []) (.ref .real [p.BK,p.BT] "b_k") (.mul .real (.consSame (.consR .nil)) (.ref .real [p.BT,p.BV] "b_v")
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BT] "d_i")))
    {allowTF32 := some Bool.false}

def baseOp : Op .real [] :=
  .log2 (.sub .real .nil (.const 1.0) (.exp2
    (.sub .real .nil (.const (-5))
      (.mul .real .nil (.natToReal (.ref .nat [] "i_h")) (.const 1.0)))))

def decayLane (BT : Nat) (length : Op .nat []) : Op .real [BT] :=
  .exp2 (.mul .real .scalarR
    (.intToReal (.sub .int .scalarR
      (.sub .int .scalarL (.castNatToInt length) (.ref .int [BT] "o_i"))
      (.castNatToInt (.constNat 1)))) (.ref .real [] "b_b"))

def setupDecay (p : Layout) : List Stmt :=
  [.assign .nat [] "i_h" (.mod .nat .nil (.ref .nat [] "i_bh") (.constNat p.H)),
   .assign .real [] "b_b" baseOp,
   .assign .int [p.BT] "o_i" (.castNatToInt (.arange p.BT)),
   .assign .real [] "d_b" (.exp2 (.mul .real .nil (.const p.BT) (.ref .real [] "b_b"))),
   .assign .real [p.BT] "d_i" (decayLane p.BT (.constNat p.BT))]

def tailGate (p : Layout) : Stmt :=
  .ifThen (.boolAnd .nil
    (.eq .nat .nil (.ref .nat [] "i_t") (.sub .nat .nil (.constNat p.NT) (.constNat 1)))
    (.ne .nat .nil (.mod .nat .nil (.constNat p.T) (.constNat p.BT)) (.constNat 0)))
    [.assign .real [] "d_b" (.exp2 (.mul .real .nil
      (.natToReal (.mod .nat .nil (.constNat p.T) (.constNat p.BT))) (.ref .real [] "b_b"))),
     .assign .real [p.BT] "d_i" (decayLane p.BT (.mod .nat .nil (.constNat p.T) (.constNat p.BT)))]

def scaledCarry (p : Layout) : Op .real [p.BK,p.BV] :=
  .mul .real .scalarL (.ref .real [] "d_b") (carry p)

def initialization (p : Layout) (r : Regions) (hasSeed : Bool) : List Stmt :=
  [.assign .nat [] "i_k" (.programId 0), .assign .nat [] "i_v" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2)] ++ setupDecay p ++ [.assign .real [p.BK,p.BV] "b_h" (.full [p.BK,p.BV] (.const 0)),
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
   tailGate p,
   .assign .real [p.BK,p.BV] "b_h"
     (if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (scaledCarry p)
      else .add .real (.consSame (.consSame .nil)) (scaledCarry p) (dot p))]

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
    p.historyHead p.historyKey p.H p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal

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

end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHProgram

/-! Pointer evaluation and unmasked checked loads for the complete H source.
Both row and column strides remain symbolic. Loads preserve arbitrary real
unspecified values; the zero-extension premise is applied later, separately. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHPointers
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
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHPointers

/-! Actual real values loaded at a chunk, including unspecified tails. These
values contain no execution state or assumed output. The zero-oracle bridge
relates their arithmetic update to the independently stated closed form. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHTiles
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
  factor : ℝ
  weight : Fin p.BT → ℝ
  key : Fin p.BK → Fin p.BT → ℝ
  value : Fin p.BT → Fin p.BV → ℝ

def loaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) : Loaded p where
  factor := carryFactor p l c
  weight := fun t => Decay.key (base p l) (length p c) t.val
  key := keyData p l a.key (padding r.key) c
  value := valueData p l a.value (padding r.value) c

def product (x : Loaded p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  ∑ t : Fin p.BT, x.key e t * (x.value t j * x.weight t)

def advance (x : Loaded p) (f : StateValue p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  x.factor * f e j + product x e j

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
  rfl

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHTiles

/-! The source loads produce the independently indexed tiles, with each
pointer's exact boundary checks. Inactive lanes retain arbitrary oracle values.
 -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHLoads
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
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHLoads

/-! Masked state-matrix stores used for both pre-update history and final state.
The base and both strides distinguish their layouts. Readback requires only
active lanes to have distinct addresses; the frame preserves complete cells. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHStore
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
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHStore

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDecayExecution
open VeriTile.Triton RealHMath RealHProgram
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def indices (BT : Nat) : Tile .int [BT] := ⟨fun i => (i.1.val : Int)⟩
def weights (BT length : Nat) (base : ℝ) : Tile .real [BT] :=
  Tile.vec (fun i => some (Decay.key base length i.val))

private theorem exp2_eval {shape : TileShape} (x : Op .real shape) (s : BlockState) :
    evalOp (.exp2 x) s = (do
      let t ← evalOp x s
      pure (Tile.uop WithBot.realExp2 t)) := by simp only [evalOp]

private theorem log2_eval {shape : TileShape} (x : Op .real shape) (s : BlockState) :
    evalOp (.log2 x) s = (do
      let t ← evalOp x s
      pure (Tile.uop WithBot.realLog2 t)) := by simp only [evalOp]

theorem lane_eval (BT length : Nat) (base : ℝ) (lengthOp : Op .nat []) (s : BlockState)
    (hl : evalOp lengthOp s = some (Tile.scalar length))
    (hi : s.regs .int [BT] "o_i" = some (indices BT))
    (hb : s.regs .real [] "b_b" = some (Tile.scalar base)) :
    evalOp (decayLane BT lengthOp) s = some (weights BT length base) := by
  have hl' : evalOp (.castNatToInt lengthOp) s = some (Tile.scalar (length : Int)) := by
    simp only [evalOp,hl,bind,Option.bind]
    rfl
  have hone : evalOp (.castNatToInt (.constNat 1)) s = some (Tile.scalar (1 : Int)) := by
    simp only [evalOp,bind,Option.bind]
    rfl
  simp only [decayLane,exp2_eval,evalOp_mul,evalOp_intToReal,evalOp_sub,
    hl',hone,evalOp_ref,hi,hb,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,weights,indices,Tile.vec,Tile.scalar,Tile.uop,Tile.bop,Tile.intToReal,
    NumericDType.sub,NumericDType.mul,
    Broadcast.leftIndex,Broadcast.rightIndex,Decay.key,Decay.factor,Decay.exp2]

theorem base_eval (H head : Nat) (s : BlockState)
    (hi : s.regs .nat [] "i_h" = some (Tile.scalar (head % H))) :
    evalOp baseOp s = some (Tile.scalar (Decay.headBase H head)) := by
  simp only [baseOp,log2_eval,exp2_eval,evalOp_sub,evalOp_mul,evalOp_natToReal,
    evalOp_const,evalOp_ref,hi,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  intro i
  cases i
  norm_num [-Tile.scalar_eta,Tile.scalar,Tile.uop,Tile.bop,Tile.natToReal,NumericDType.sub,NumericDType.mul,
    Broadcast.leftIndex,Broadcast.rightIndex,Decay.headBase,Decay.exp2]
  rfl

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDecayExecution

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDecayState
open VeriTile.Triton RealHMath RealHProgram RealDecayExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def standardState (p : Layout) (l : Launch) (u : BlockState) : BlockState :=
  ((((u.setReg "i_h" .nat [] (Tile.scalar (l.head % p.H))).setReg "b_b" .real []
    (Tile.scalar (base p l))).setReg "o_i" .int [p.BT] (indices p.BT)).setReg
    "d_b" .real [] (Tile.scalar (Decay.factor (base p l) p.BT))).setReg
    "d_i" .real [p.BT] (weights p.BT p.BT (base p l))

theorem head_index_eval (H head : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar head)) :
    evalOp (.mod .nat .nil (.ref .nat [] "i_bh") (.constNat H)) u =
      some (Tile.scalar (head % H)) := by
  simp only [evalOp,evalOp_ref,hh,evalOp_constNat,bind,Option.bind]
  rfl

theorem factor_eval (BT : Nat) (base : ℝ) (u : BlockState)
    (hb : u.regs .real [] "b_b" = some (Tile.scalar base)) :
    evalOp (.exp2 (.mul .real .nil (.const BT) (.ref .real [] "b_b"))) u =
      some (Tile.scalar (Decay.factor base BT)) := by
  simp only [evalOp,evalOp_ref,hb,bind,Option.bind]
  rfl

theorem partial_factor_eval (T BT : Nat) (base : ℝ) (u : BlockState)
    (hb : u.regs .real [] "b_b" = some (Tile.scalar base)) :
    evalOp (.exp2 (.mul .real .nil
      (.natToReal (.mod .nat .nil (.constNat T) (.constNat BT))) (.ref .real [] "b_b"))) u =
      some (Tile.scalar (Decay.factor base (T % BT : Nat))) := by
  simp only [evalOp,evalOp_ref,hb,bind,Option.bind]
  rfl

theorem indices_eval (BT : Nat) (u : BlockState) :
    evalOp (.castNatToInt (.arange BT)) u = some (indices BT) := by
  simp only [evalOp,bind,Option.bind]
  rfl

theorem standard_executes (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    stepStmts (setupDecay p) u = some (standardState p l u) := by
  unfold setupDecay
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (head_index_eval p.H l.head u hh)),
    stepStmts.cons_some (stepStmt_assign_eq_some (base_eval p.H l.head _ (by simp))),
    stepStmts.cons_some (stepStmt_assign_eq_some (indices_eval p.BT _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (factor_eval p.BT (base p l) _ (by simp [base]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (lane_eval p.BT p.BT (base p l) (.constNat p.BT) _
      (evalOp_constNat _ _) (by simp) (by simp [base]))),stepStmts.nil]
  rfl

def partialState (p : Layout) (l : Launch) (u : BlockState) : BlockState :=
  (u.setReg "d_b" .real [] (Tile.scalar (Decay.factor (base p l) (p.T % p.BT : Nat)))).setReg
    "d_i" .real [p.BT] (weights p.BT (p.T % p.BT) (base p l))

def gateState (p : Layout) (l : Launch) (c : Nat) (u : BlockState) : BlockState :=
  if c = p.NT - 1 ∧ p.T % p.BT ≠ 0 then partialState p l u else u

theorem gate_condition (p : Layout) (c : Nat) (u : BlockState)
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (.boolAnd .nil
      (.eq .nat .nil (.ref .nat [] "i_t") (.sub .nat .nil (.constNat p.NT) (.constNat 1)))
      (.ne .nat .nil (.mod .nat .nil (.constNat p.T) (.constNat p.BT)) (.constNat 0))) u =
      some (Tile.scalar (decide (c = p.NT - 1 ∧ p.T % p.BT ≠ 0))) := by
  simp only [evalOp,evalOp_ref,ht,bind,Option.bind]
  congr 1
  apply Tile.ext
  intro i
  cases i
  simp [-Tile.scalar_eta,Tile.scalar,Tile.bop,NumericDType.sub,IntegralDType.mod,
    ComparableDType.eq,ComparableDType.ne,Broadcast.leftIndex,Broadcast.rightIndex]

theorem gate_executes (p : Layout) (l : Launch) (c : Nat) (u : BlockState)
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hi : u.regs .int [p.BT] "o_i" = some (indices p.BT))
    (hb : u.regs .real [] "b_b" = some (Tile.scalar (base p l))) :
    stepStmt (tailGate p) u = some (gateState p l c u) := by
  unfold tailGate
  rw [stepStmt]
  rw [gate_condition p c u ht]
  simp only [Option.bind_eq_bind,Option.bind_some,Tile.scalar,decide_eq_true_eq]
  by_cases h : c = p.NT - 1 ∧ p.T % p.BT ≠ 0
  · rw [if_pos h]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (partial_factor_eval p.T p.BT (base p l) u hb)),
      stepStmts.cons_some (stepStmt_assign_eq_some (lane_eval p.BT (p.T % p.BT) (base p l)
        (.mod .nat .nil (.constNat p.T) (.constNat p.BT)) _
        (by simp only [evalOp,bind,Option.bind]; rfl) (by simp [hi]) (by simp [hb]))),stepStmts.nil]
    simp only [gateState,if_pos h]
    rfl
  · simp only [if_neg h,gateState]

@[simp] theorem gate_memory (p : Layout) (l : Launch) (c : Nat) (u : BlockState) :
    (gateState p l c u).mem = u.mem := by
  unfold gateState
  split <;> rfl

@[simp] theorem gate_padding (p : Layout) (l : Launch) (c : Nat) (u : BlockState) :
    (gateState p l c u).undef = u.undef := by
  unfold gateState
  split <;> rfl

@[simp] theorem gate_pids (p : Layout) (l : Launch) (c : Nat) (u : BlockState) :
    (gateState p l c u).pids = u.pids := by
  unfold gateState
  split <;> rfl

theorem gate_regs (p : Layout) (l : Launch) (c : Nat) (u : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hb : name ≠ "d_b") (hi : name ≠ "d_i") :
    (gateState p l c u).regs dtype shape name = u.regs dtype shape name := by
  unfold gateState
  split
  · simp [partialState,hb,hi,Ne.symm hb,Ne.symm hi]
  · rfl

theorem gate_factors (p : Layout) (l : Launch) (c : Nat) (u : BlockState)
    (hb : u.regs .real [] "d_b" = some (Tile.scalar (Decay.factor (base p l) p.BT)))
    (hi : u.regs .real [p.BT] "d_i" = some (weights p.BT p.BT (base p l))) :
    (gateState p l c u).regs .real [] "d_b" = some (Tile.scalar (carryFactor p l c)) ∧
    (gateState p l c u).regs .real [p.BT] "d_i" = some (weights p.BT (length p c) (base p l)) := by
  by_cases h : c = p.NT - 1 ∧ p.T % p.BT ≠ 0
  · simp [gateState,partialState,carryFactor,length,Decay.chunkLength,h,h.1,h.2]
  · simp [gateState,carryFactor,length,Decay.chunkLength,h,hb,hi]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDecayState

/-! Arithmetic on the actually loaded tiles, using the reached signed decay
weights. These are real-correctness lemmas, independent of FP admission. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHAlgebra
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def updateOp (p : Layout) (swapped : Bool) : Op .real [p.BK,p.BV] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (scaledCarry p)
  else .add .real (.consSame (.consSame .nil)) (scaledCarry p) (dot p)

theorem weighted_eval (p : Layout) (x : Loaded p) (u : BlockState)
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (x.value t j))))
    (hi : u.regs .real [p.BT] "d_i" = some (Tile.vec (fun t => some (x.weight t)))) :
    evalOp (.mul .real (.consSame (.consR .nil)) (.ref .real [p.BT,p.BV] "b_v")
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BT] "d_i"))) u =
      some (Tile.mat (fun t j => some (x.value t j * x.weight t))) := by
  have he := evalOp_expandDim_ref_of_regs .real [p.BT] ⟨1,by simp⟩ "d_i" u _ hi
  erw [evalOp_mul,evalOp_ref,hv,he]
  rfl

theorem product_eval (p : Layout) (x : Loaded p) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (x.key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (x.value t j))))
    (hi : u.regs .real [p.BT] "d_i" = some (Tile.vec (fun t => some (x.weight t)))) :
    evalOp (dot p) u = some (stateTile p (product x)) := by
  unfold dot
  erw [evalOp_dot]
  erw [evalOp_ref,hk,weighted_eval p x u hv hi]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [Tile.mat,Option.map₂_some_some,stateTile,product]
  exact (WithBot.coe_sum Finset.univ (fun t : Fin p.BT => x.key e t * (x.value t j * x.weight t))).symm

theorem carry_eval (p : Layout) (x : Loaded p) (f : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hb : u.regs .real [] "d_b" = some (Tile.scalar x.factor)) :
    evalOp (scaledCarry p) u = some (stateTile p (fun e j => x.factor * f e j)) := by
  simp only [scaledCarry,carry,evalOp_mul,evalOp_ref,hh,hb,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem update_eval (p : Layout) (x : Loaded p) (swapped : Bool) (f : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hk : u.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (x.key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (x.value t j))))
    (hb : u.regs .real [] "d_b" = some (Tile.scalar x.factor))
    (hi : u.regs .real [p.BT] "d_i" = some (Tile.vec (fun t => some (x.weight t)))) :
    evalOp (updateOp p swapped) u = some (stateTile p (advance x f)) := by
  cases swapped
  all_goals
    simp only [updateOp,Bool.false_eq_true,↓reduceIte,evalOp_add,
      carry_eval p x f u hh hb,product_eval p x u hk hv hi,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨e,j,⟨⟩⟩
    simp [advance,stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.add,add_comm]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHAlgebra

/-! Shared signed-row decay expressions for output and backward kernels.
The score expression evaluates the signed difference before its causal mask. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDecayTiles
open VeriTile.Triton RealDecayExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def queryOp (BT : Nat) : Op .real [BT] :=
  .exp2 (.mul .real .scalarR
    (.intToReal (.add .int .scalarR (.ref .int [BT] "o_i") (.castNatToInt (.constNat 1))))
    (.ref .real [] "b_b"))

def maskOp (BT : Nat) : Op .bool [BT,BT] :=
  .ge .int (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .int [BT] "o_i"))
    (.expandDim ⟨0,by simp⟩ (.ref .int [BT] "o_i"))

def scoreOp (BT : Nat) : Op .real [BT,BT] :=
  .where (.ref .bool [BT,BT] "m_s")
    (.exp2 (.mul .real .scalarR
      (.intToReal (.sub .int (.consR (.consL .nil))
        (.expandDim ⟨1,by simp⟩ (.ref .int [BT] "o_i"))
        (.expandDim ⟨0,by simp⟩ (.ref .int [BT] "o_i"))))
      (.ref .real [] "b_b"))) (.broadcast (.const 0) [BT,BT])

def query (BT : Nat) (base : ℝ) : Tile .real [BT] :=
  Tile.vec (fun i => some (Decay.query base i.val))
def mask (BT : Nat) : Tile .bool [BT,BT] :=
  Tile.mat (fun i j => decide (j.val ≤ i.val))
def score (BT : Nat) (base : ℝ) : Tile .real [BT,BT] :=
  Tile.mat (fun i j => some (Decay.score base i.val j.val))

theorem query_eval (BT : Nat) (base : ℝ) (s : BlockState)
    (hi : s.regs .int [BT] "o_i" = some (indices BT))
    (hb : s.regs .real [] "b_b" = some (Tile.scalar base)) :
    evalOp (queryOp BT) s = some (query BT base) := by
  simp only [queryOp,evalOp,hi,hb,bind,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,query,indices,Tile.vec,Tile.scalar,Tile.uop,Tile.bop,Tile.intToReal,
    NumericDType.add,NumericDType.mul,Broadcast.leftIndex,Broadcast.rightIndex,
    Decay.query,Decay.factor,Decay.exp2]

theorem mask_eval (BT : Nat) (s : BlockState)
    (hi : s.regs .int [BT] "o_i" = some (indices BT)) :
    evalOp (maskOp BT) s = some (mask BT) := by
  have hr := evalOp_expandDim_ref_of_regs .int [BT] ⟨1,by simp⟩ "o_i" s _ hi
  have hc := evalOp_expandDim_ref_of_regs .int [BT] ⟨0,by simp⟩ "o_i" s _ hi
  simp only [maskOp,evalOp,hi,bind,Option.bind]
  erw [hr,hc]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [-Tile.scalar_eta,mask,indices,Tile.mat,Tile.expandDim,TileShape.dropInsertedIndex,
    Tile.bop,Tile.cop,ComparableDType.ge,Broadcast.leftIndex,Broadcast.rightIndex]

theorem score_eval (BT : Nat) (base : ℝ) (s : BlockState)
    (hi : s.regs .int [BT] "o_i" = some (indices BT))
    (hb : s.regs .real [] "b_b" = some (Tile.scalar base))
    (hm : s.regs .bool [BT,BT] "m_s" = some (mask BT)) :
    evalOp (scoreOp BT) s = some (score BT base) := by
  have hr := evalOp_expandDim_ref_of_regs .int [BT] ⟨1,by simp⟩ "o_i" s _ hi
  have hc := evalOp_expandDim_ref_of_regs .int [BT] ⟨0,by simp⟩ "o_i" s _ hi
  simp only [scoreOp,evalOp,hi,hb,hm,bind,Option.bind]
  erw [hr,hc]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [-Tile.scalar_eta,score,mask,indices,Tile.mat,Tile.scalar,Tile.expandDim,
    TileShape.dropInsertedIndex,Tile.uop,Tile.bop,Tile.intToReal,Tile.select,
    NumericDType.sub,NumericDType.mul,Broadcast.leftIndex,Broadcast.rightIndex,
    Decay.score,Decay.factor,Decay.exp2]
  split_ifs <;> rfl

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDecayTiles

#guard_msgs (drop info) in
#auditModuleAxioms
