import bench.tritonbench_g.chunk_linear_attn.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardStateProgram

/-! Precision erasure of the complete backward three-output source; no padding,
store order or loop bound is changed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then backwardQKVDotLeft else backwardQKV
variable (swapped : Bool) (dot : Storage)
    (q k v h do_ dh dq dk dv : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV NT : Nat)
    (scale : ℝ)

theorem projects :
    (source swapped dot q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm? =
    .ok (source swapped dot q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel := by
  cases swapped <;> cases dot <;> rfl

theorem storage_erasure :
    (source swapped dot q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel.body =
    (source swapped .fp32 q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, backwardQKV,
    backwardQKVDotLeft, backwardQKVSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVSource

/-! Initial-array formula for the three gradients. The value-axis sums include
the zero-extended final tile. History bounds use the source's combined NT*K
axis, while Q/K use K; these are deliberately separate checks. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVMath
open RealHMath (Layout)
open scoped BigOperators
noncomputable section

structure Launch where
  key : Nat
  chunk : Nat
  head : Nat
  heads : Nat
structure Inputs where
  query : Nat → ℝ
  key : Nat → ℝ
  value : Nat → ℝ
  history : Nat → ℝ
  incoming : Nat → ℝ
  historyGradient : Nat → ℝ

def numBlocks (p : Layout) : Nat := (p.V+p.BV-1)/p.BV
def row (p : Layout) (l : Launch) (t : Nat) := l.chunk*p.BT+t
def keyIndex (p : Layout) (l : Launch) (e : Nat) := l.key*p.BK+e
def valueIndex (p : Layout) (c j : Nat) := c*p.BV+j
def historyIndex (p : Layout) (l : Launch) (e : Nat) := l.chunk*p.K+keyIndex p l e

def keyAddress (p : Layout) (l : Launch) (t e : Nat) :=
  l.head*p.keyHead+row p l t*p.keyTime+keyIndex p l e*p.keyFeature
def valueAddress (p : Layout) (l : Launch) (c t j : Nat) :=
  l.head*p.valueHead+row p l t*p.valueTime+valueIndex p c j*p.valueFeature
def dvAddress (p : Layout) (l : Launch) (c t j : Nat) :=
  (l.key*l.heads+l.head)*p.valueHead+row p l t*p.valueTime+valueIndex p c j*p.valueFeature

def query (p : Layout) (l : Launch) (a : Inputs) (e t : Nat) : ℝ :=
  if keyIndex p l e < p.K ∧ row p l t < p.T then
    a.query (l.head*p.keyHead+keyIndex p l e*p.keyFeature+row p l t*p.keyTime) else 0

def key (p : Layout) (l : Launch) (a : Inputs) (t e : Nat) : ℝ :=
  if row p l t < p.T ∧ keyIndex p l e < p.K then a.key (keyAddress p l t e) else 0

def value (p : Layout) (l : Launch) (a : Inputs) (c t j : Nat) : ℝ :=
  if row p l t < p.T ∧ valueIndex p c j < p.V then a.value (valueAddress p l c t j) else 0

def incoming (p : Layout) (l : Launch) (a : Inputs) (c t j : Nat) : ℝ :=
  if row p l t < p.T ∧ valueIndex p c j < p.V then a.incoming (valueAddress p l c t j) else 0

def history (p : Layout) (l : Launch) (a : Inputs) (c j e : Nat) : ℝ :=
  if valueIndex p c j < p.V ∧ historyIndex p l e < p.NT*p.K then
    a.history (l.head*p.historyHead+valueIndex p c j+historyIndex p l e*p.historyKey) else 0

def historyGradient (p : Layout) (l : Launch) (a : Inputs) (c e j : Nat) : ℝ :=
  if historyIndex p l e < p.NT*p.K ∧ valueIndex p c j < p.V then
    a.historyGradient (l.head*p.historyHead+historyIndex p l e*p.historyKey+valueIndex p c j) else 0

def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if i ≤ j then (∑ e : Fin p.BK, key p l a i e.val*query p l a e.val j)*scale else 0

def dsSum (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ c ∈ Finset.range n, ∑ v : Fin p.BV, incoming p l a c i v.val*value p l a c j v.val

def dqSum (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n i e : Nat) : ℝ :=
  ∑ c ∈ Finset.range n, (∑ v : Fin p.BV, incoming p l a c i v.val*history p l a c v.val e)*scale

def dkSum (p : Layout) (l : Launch) (a : Inputs) (n i e : Nat) : ℝ :=
  ∑ c ∈ Finset.range n, ∑ v : Fin p.BV, value p l a c i v.val*historyGradient p l a c e v.val

def ds (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if i ≥ j then dsSum p l a (numBlocks p) i j*scale else 0

def dq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i e : Nat) : ℝ :=
  dqSum p l a scale (numBlocks p) i e + ∑ j : Fin p.BT, ds p l a scale i j.val*key p l a j.val e

def dk (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i e : Nat) : ℝ :=
  dkSum p l a (numBlocks p) i e + ∑ j : Fin p.BT, query p l a e j.val*ds p l a scale j.val i

def dv (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (c i j : Nat) : ℝ :=
  (∑ e : Fin p.BK, key p l a i e.val*historyGradient p l a c e.val j) +
    ∑ t : Fin p.BT, score p l a scale i t.val*incoming p l a c t.val j

def keyActive (p : Layout) (l : Launch) (i e : Nat) : Prop := row p l i < p.T ∧ keyIndex p l e < p.K
def valueActive (p : Layout) (l : Launch) (c i j : Nat) : Prop := row p l i < p.T ∧ valueIndex p c j < p.V
instance (p : Layout) (l : Launch) (i e : Nat) : Decidable (keyActive p l i e) := inferInstanceAs (Decidable (_ ∧ _))
instance (p : Layout) (l : Launch) (c i j : Nat) : Decidable (valueActive p l c i j) := inferInstanceAs (Decidable (_ ∧ _))

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVMath

/-! Exact erased body, including all value blocks, the three output stores,
upper/lower causal masks, transposes, and the runtime n_bh in the dV slice. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVProgram
open VeriTile.Triton
open RealHMath (Layout)
open RealHProgram (times load2)
open RealQKVMath (Launch Inputs)
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

structure Regions where
  query : RegionName
  key : RegionName
  value : RegionName
  history : RegionName
  incoming : RegionName
  historyGradient : RegionName
  dq : RegionName
  dk : RegionName
  dv : RegionName

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2,s.numPids 2⟩
def inputs (r : Regions) (s : BlockState) : Inputs :=
  ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history,
    s.readMem r.incoming,s.readMem r.historyGradient⟩

def queryPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BT] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.keyHead) [p.K,p.T] [p.BK,p.BT]
    [p.keyFeature,p.keyTime] [times "i_k" p.BK,times "i_t" p.BT]
def keyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BK] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.keyHead) [p.T,p.K] [p.BT,p.BK]
    [p.keyTime,p.keyFeature] [times "i_t" p.BT,times "i_k" p.BK]
def valuePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BV] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.valueHead) [p.T,p.V] [p.BT,p.BV]
    [p.valueTime,p.valueFeature] [times "i_t" p.BT,times "i_v" p.BV]
def historyOffset (p : Layout) : Op .nat [] :=
  .add .nat .nil (times "i_t" p.K) (times "i_k" p.BK)
def historyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BV,p.BK] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.historyHead) [p.V,p.NT*p.K] [p.BV,p.BK]
    [1,p.historyKey] [times "i_v" p.BV,historyOffset p]
def historyGradientPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.historyHead) [p.NT*p.K,p.V] [p.BK,p.BV]
    [p.historyKey,1] [historyOffset p,times "i_v" p.BV]
def dvBase (p : Layout) : Op .nat [] :=
  .mul .nat .nil (.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_k") (.ref .nat [] "n_bh"))
    (.ref .nat [] "i_bh")) (.constNat p.valueHead)
def dvPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BV] :=
  .makeBlockPtrDynOffsets r (dvBase p) [p.T,p.V] [p.BT,p.BV]
    [p.valueTime,p.valueFeature] [times "i_t" p.BT,times "i_v" p.BV]

def dot {m k n : Nat} (a : Op .real [m,k]) (b : Op .real [k,n]) : Op .real [m,n] :=
  .dot (batch := []) a b {allowTF32 := some Bool.false}
def add {m n : Nat} (a b : Op .real [m,n]) : Op .real [m,n] :=
  .add .real (.consSame (.consSame .nil)) a b
def scaleBy {m n : Nat} (a : Op .real [m,n]) (scale : ℝ) : Op .real [m,n] :=
  .mul .real .scalarR a (.const scale)
def ref (m n : Nat) (name : RegName) : Op .real [m,n] := .ref .real [m,n] name

def mask (p : Layout) (lower : Bool) : Op .bool [p.BT,p.BT] :=
  if lower then
    .ge .nat (.consR (.consL .nil)) (.expandDim ⟨1,by simp⟩ (.ref .nat [p.BT] "o_i"))
      (.expandDim ⟨0,by simp⟩ (.ref .nat [p.BT] "o_i"))
  else .le .nat (.consR (.consL .nil)) (.expandDim ⟨1,by simp⟩ (.ref .nat [p.BT] "o_i"))
      (.expandDim ⟨0,by simp⟩ (.ref .nat [p.BT] "o_i"))
def select (p : Layout) (lower : Bool) (a : Op .real [p.BT,p.BT]) : Op .real [p.BT,p.BT] :=
  .where (mask p lower) a (.broadcast (.const 0) [p.BT,p.BT])
def score (p : Layout) (scale : ℝ) : Op .real [p.BT,p.BT] :=
  scaleBy (dot (ref p.BT p.BK "b_k") (ref p.BK p.BT "b_q")) scale

def updateDS (p : Layout) : Op .real [p.BT,p.BT] :=
  add (ref p.BT p.BT "b_ds") (dot (ref p.BT p.BV "b_do") (.transpose (batch := []) (ref p.BT p.BV "b_v")))
def updateDQ (p : Layout) (scale : ℝ) : Op .real [p.BT,p.BK] :=
  add (ref p.BT p.BK "b_dq") (scaleBy (dot (ref p.BT p.BV "b_do") (ref p.BV p.BK "b_h")) scale)
def updateDK (p : Layout) : Op .real [p.BT,p.BK] :=
  add (ref p.BT p.BK "b_dk") (dot (ref p.BT p.BV "b_v") (.transpose (batch := []) (ref p.BK p.BV "b_dh")))
def outputDV (p : Layout) : Op .real [p.BT,p.BV] :=
  add (dot (ref p.BT p.BK "b_k") (ref p.BK p.BV "b_dh"))
    (dot (ref p.BT p.BT "b_s") (ref p.BT p.BV "b_do"))
def finalDS (p : Layout) (scale : ℝ) : Op .real [p.BT,p.BT] :=
  select p Bool.true (scaleBy (ref p.BT p.BT "b_ds") scale)
def finalDQ (p : Layout) (swapped : Bool) : Op .real [p.BT,p.BK] :=
  if swapped then add (dot (ref p.BT p.BT "b_ds") (ref p.BT p.BK "b_k")) (ref p.BT p.BK "b_dq")
  else add (ref p.BT p.BK "b_dq") (dot (ref p.BT p.BT "b_ds") (ref p.BT p.BK "b_k"))
def finalDK (p : Layout) : Op .real [p.BT,p.BK] :=
  add (ref p.BT p.BK "b_dk") (.transpose (batch := []) (dot (ref p.BK p.BT "b_q") (ref p.BT p.BT "b_ds")))
def store2 (m n : Nat) (ptr value : RegName) : Stmt :=
  .store .real [m,n] (.blockPtr (.ref .blockPtr [m,n] ptr) [0,1]) (ref m n value) .none

def iteration (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  [.assign .blockPtr [p.BT,p.BV] "p_v" (valuePtr p r.value),
   .assign .blockPtr [p.BV,p.BK] "p_h" (historyPtr p r.history),
   .assign .blockPtr [p.BT,p.BV] "p_do" (valuePtr p r.incoming),
   .assign .blockPtr [p.BK,p.BV] "p_dh" (historyGradientPtr p r.historyGradient),
   .assign .blockPtr [p.BT,p.BV] "p_dv" (dvPtr p r.dv),
   .assign .real [p.BT,p.BV] "b_v" (load2 p.BT p.BV "p_v"),
   .assign .real [p.BT,p.BV] "b_do" (load2 p.BT p.BV "p_do"),
   .assign .real [p.BV,p.BK] "b_h" (load2 p.BV p.BK "p_h"),
   .assign .real [p.BK,p.BV] "b_dh" (load2 p.BK p.BV "p_dh"),
   .assign .real [p.BT,p.BT] "b_ds" (updateDS p),
   .assign .real [p.BT,p.BK] "b_dq" (updateDQ p scale),
   .assign .real [p.BT,p.BK] "b_dk" (updateDK p),
   .assign .real [p.BT,p.BV] "b_dv" (outputDV p),
   store2 p.BT p.BV "p_dv" "b_dv"]

def initialization (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  [.assign .nat [] "i_k" (.programId 0),.assign .nat [] "i_t" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),.assign .nat [] "n_bh" (.numPrograms 2),
   .assign .nat [p.BT] "o_i" (.arange p.BT),
   .assign .blockPtr [p.BK,p.BT] "p_q" (queryPtr p r.query),
   .assign .blockPtr [p.BT,p.BK] "p_k" (keyPtr p r.key),
   .assign .real [p.BK,p.BT] "b_q" (load2 p.BK p.BT "p_q"),
   .assign .real [p.BT,p.BK] "b_k" (load2 p.BT p.BK "p_k"),
   .assign .real [p.BT,p.BT] "b_s" (score p scale),
   .assign .real [p.BT,p.BT] "b_s" (select p Bool.false (ref p.BT p.BT "b_s")),
   .assign .real [p.BT,p.BK] "b_dq" (.full [p.BT,p.BK] (.const 0)),
   .assign .real [p.BT,p.BK] "b_dk" (.full [p.BT,p.BK] (.const 0)),
   .assign .real [p.BT,p.BT] "b_ds" (.full [p.BT,p.BT] (.const 0))]

def finalization (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  [.assign .real [p.BT,p.BT] "b_ds" (finalDS p scale),
   .assign .real [p.BT,p.BK] "b_dq" (finalDQ p swapped),
   .assign .real [p.BT,p.BK] "b_dk" (finalDK p),
   .assign .blockPtr [p.BT,p.BK] "p_dq" (keyPtr p r.dq),
   .assign .blockPtr [p.BT,p.BK] "p_dk" (keyPtr p r.dk),
   store2 p.BT p.BK "p_dq" "b_dq",store2 p.BT p.BK "p_dk" "b_dk"]

def loop (p : Layout) (r : Regions) (scale : ℝ) : Stmt :=
  .forRangeDyn "i_v" (.constNat 0)
    (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.V) (.constNat p.BV)) (.constNat 1)) (.constNat p.BV))
    (.constNat 1) (iteration p r scale)
def body (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  initialization p r scale ++ [loop p r scale] ++ finalization p r scale swapped

def source (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool) :=
  RealQKVSource.source swapped storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.T p.K p.V p.BT p.BK p.BV p.NT

theorem source_body (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool) :
    (source p r storage scale swapped).eraseDType.toAlgKernel.body = body p r scale swapped := by
  unfold source
  rw [RealQKVSource.storage_erasure]
  cases swapped
  all_goals simp only [RealQKVSource.source,Bool.false_eq_true,↓reduceIte,
    backwardQKV,backwardQKVDotLeft,backwardQKVSource,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVProgram

/-! Rank-two real value recipes used below. These lemmas operate after
precision erasure; none is used to establish a floating-point rewrite. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealMatrix
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev Values (m n : Nat) := Fin m → Fin n → ℝ
def tile (f : Values m n) : Tile .real [m,n] := Tile.mat fun i j => some (f i j)

@[simp] theorem add_tile (a b : Values m n) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil)) (tile a) (tile b) = tile (fun i j => a i j+b i j) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rfl
@[simp] theorem scale_tile (a : Values m n) (scale : ℝ) :
    Tile.bop NumericDType.real.mul .scalarR (tile a) (Tile.scalar (some scale)) = tile (fun i j => a i j*scale) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rfl
@[simp] theorem transpose_tile (a : Values m n) :
    Tile.transpose [] (tile a) = tile (fun j i => a i j) := rfl
@[simp] theorem dot_tile (a : Values m k) (b : Values k n) :
    Tile.dot [] (tile a) (tile b) = tile (fun i j => ∑ e : Fin k, a i e*b e j) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [tile,Tile.mat,Option.map₂_some_some]
  exact (WithBot.coe_sum Finset.univ (fun e : Fin k => a i e*b e j)).symm
@[simp] theorem select_tile (m : Fin n → Fin n → Bool) (a b : Values n n) :
    Tile.select (Tile.mat m) (tile a) (tile b) = tile (fun i j => if m i j then a i j else b i j) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  cases hm : m i j <;> simp [tile,Tile.select,Tile.mat,hm]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealMatrix

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVTiles
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath RealMatrix
noncomputable section

def q (p : Layout) (l : Launch) (a : Inputs) : Values p.BK p.BT := fun e t => query p l a e.val t.val
def k (p : Layout) (l : Launch) (a : Inputs) : Values p.BT p.BK := fun t e => key p l a t.val e.val
def v (p : Layout) (l : Launch) (a : Inputs) (c : Nat) : Values p.BT p.BV := fun t j => value p l a c t.val j.val
def o (p : Layout) (l : Launch) (a : Inputs) (c : Nat) : Values p.BT p.BV := fun t j => incoming p l a c t.val j.val
def h (p : Layout) (l : Launch) (a : Inputs) (c : Nat) : Values p.BV p.BK := fun j e => history p l a c j.val e.val
def dh (p : Layout) (l : Launch) (a : Inputs) (c : Nat) : Values p.BK p.BV := fun e j => historyGradient p l a c e.val j.val

def s (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) : Values p.BT p.BT := fun i j => score p l a scale i.val j.val
def cs (p : Layout) (l : Launch) (a : Inputs) (n : Nat) : Values p.BT p.BT := fun i j => dsSum p l a n i.val j.val
def cq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) : Values p.BT p.BK := fun i e => dqSum p l a scale n i.val e.val
def ck (p : Layout) (l : Launch) (a : Inputs) (n : Nat) : Values p.BT p.BK := fun i e => dkSum p l a n i.val e.val

def ds (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) : Values p.BT p.BT := fun i j => RealQKVMath.ds p l a scale i.val j.val
def dq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) : Values p.BT p.BK := fun i e => RealQKVMath.dq p l a scale i.val e.val
def dk (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) : Values p.BT p.BK := fun i e => RealQKVMath.dk p l a scale i.val e.val
def dv (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (c : Nat) : Values p.BT p.BV := fun i j => RealQKVMath.dv p l a scale c i.val j.val

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVTiles

/-! Source pointers and loads. The history axis is NT*K, and the dV base
retains the runtime number of head programs. Inactive lanes require the explicit
zero oracle to agree with the independent zero-extended formulas. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVPointers
open VeriTile.Triton
open RealHMath (Layout)
open RealHPointers (matrixPointer make2 times_eval load2_eval)
open RealQKVMath (Launch Inputs)
open RealQKVProgram RealQKVTiles RealMatrix
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def queryPointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrixPointer r (l.head*p.keyHead) (p.K) (p.T) p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (l.chunk*p.BT)

def keyPointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrixPointer r (l.head*p.keyHead) (p.T) (p.K) p.BT p.BK p.keyTime p.keyFeature (l.chunk*p.BT) (l.key*p.BK)

def valuePointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.valueHead) (p.T) (p.V) p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (c*p.BV)

def historyPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.historyHead) (p.V) (p.NT*p.K) p.BV p.BK 1 p.historyKey (c*p.BV) (l.chunk*p.K+l.key*p.BK)

def historyGradientPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.historyHead) (p.NT*p.K) (p.V) p.BK p.BV p.historyKey 1 (l.chunk*p.K+l.key*p.BK) (c*p.BV)

def dvPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r ((l.key*l.heads+l.head)*p.valueHead) (p.T) (p.V) p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (c*p.BV)

theorem query_eval (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    : evalOp (queryPtr p r) u = some (queryPointer p l r) := by
  unfold queryPtr
  apply make2
  all_goals simp only [historyOffset,dvBase,RealHProgram.times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,
    hh,hk,ht,Option.bind_eq_bind,Option.bind_some]
  all_goals rfl

theorem key_eval (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    : evalOp (keyPtr p r) u = some (keyPointer p l r) := by
  unfold keyPtr
  apply make2
  all_goals simp only [historyOffset,dvBase,RealHProgram.times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,
    hh,hk,ht,Option.bind_eq_bind,Option.bind_some]
  all_goals rfl

theorem value_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar c))
    : evalOp (valuePtr p r) u = some (valuePointer p l r c) := by
  unfold valuePtr
  apply make2
  all_goals simp only [historyOffset,dvBase,RealHProgram.times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,
    hh,hk,ht,hv,Option.bind_eq_bind,Option.bind_some]
  all_goals rfl

theorem history_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar c))
    : evalOp (historyPtr p r) u = some (historyPointer p l r c) := by
  unfold historyPtr
  apply make2
  all_goals simp only [historyOffset,dvBase,RealHProgram.times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,
    hh,hk,ht,hv,Option.bind_eq_bind,Option.bind_some]
  all_goals rfl

theorem historyGradient_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar c))
    : evalOp (historyGradientPtr p r) u = some (historyGradientPointer p l r c) := by
  unfold historyGradientPtr
  apply make2
  all_goals simp only [historyOffset,dvBase,RealHProgram.times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,
    hh,hk,ht,hv,Option.bind_eq_bind,Option.bind_some]
  all_goals rfl

theorem dv_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar c))
    (hn : u.regs .nat [] "n_bh" = some (Tile.scalar l.heads))
    : evalOp (dvPtr p r) u = some (dvPointer p l r c) := by
  unfold dvPtr
  apply make2
  all_goals simp only [historyOffset,dvBase,RealHProgram.times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,
    hh,hk,ht,hv,hn,Option.bind_eq_bind,Option.bind_some]
  all_goals rfl

theorem query_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a.query off) (hu : u.undef r = fun _ => 0)
    (hp : u.regs .blockPtr [p.BK,p.BT] name = some (queryPointer p l r)) :
    evalOp (RealHProgram.load2 p.BK p.BT name) u = some (tile (q p l a)) := by
  rw [load2_eval r u name (l.head*p.keyHead) (p.K) (p.T) p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (l.chunk*p.BT) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [tile,Tile.mat,q,RealQKVMath.query,RealQKVMath.row,RealQKVMath.keyIndex,
    RealQKVMath.valueIndex,RealQKVMath.historyIndex,RealQKVMath.keyAddress,RealQKVMath.valueAddress,hm,hu,Nat.mul_one,
    Nat.add_assoc]
  split <;> rfl

theorem key_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a.key off) (hu : u.undef r = fun _ => 0)
    (hp : u.regs .blockPtr [p.BT,p.BK] name = some (keyPointer p l r)) :
    evalOp (RealHProgram.load2 p.BT p.BK name) u = some (tile (k p l a)) := by
  rw [load2_eval r u name (l.head*p.keyHead) (p.T) (p.K) p.BT p.BK p.keyTime p.keyFeature (l.chunk*p.BT) (l.key*p.BK) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [tile,Tile.mat,k,RealQKVMath.key,RealQKVMath.row,RealQKVMath.keyIndex,
    RealQKVMath.valueIndex,RealQKVMath.historyIndex,RealQKVMath.keyAddress,RealQKVMath.valueAddress,hm,hu,Nat.mul_one,
    Nat.add_assoc]
  split <;> rfl

theorem value_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (c : Nat)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a.value off) (hu : u.undef r = fun _ => 0)
    (hp : u.regs .blockPtr [p.BT,p.BV] name = some (valuePointer p l r c)) :
    evalOp (RealHProgram.load2 p.BT p.BV name) u = some (tile (v p l a c)) := by
  rw [load2_eval r u name (l.head*p.valueHead) (p.T) (p.V) p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (c*p.BV) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [tile,Tile.mat,v,RealQKVMath.value,RealQKVMath.row,RealQKVMath.keyIndex,
    RealQKVMath.valueIndex,RealQKVMath.historyIndex,RealQKVMath.keyAddress,RealQKVMath.valueAddress,hm,hu,Nat.mul_one,
    Nat.add_assoc]
  split <;> rfl

theorem incoming_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (c : Nat)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a.incoming off) (hu : u.undef r = fun _ => 0)
    (hp : u.regs .blockPtr [p.BT,p.BV] name = some (valuePointer p l r c)) :
    evalOp (RealHProgram.load2 p.BT p.BV name) u = some (tile (o p l a c)) := by
  rw [load2_eval r u name (l.head*p.valueHead) (p.T) (p.V) p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (c*p.BV) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [tile,Tile.mat,o,RealQKVMath.incoming,RealQKVMath.row,RealQKVMath.keyIndex,
    RealQKVMath.valueIndex,RealQKVMath.historyIndex,RealQKVMath.keyAddress,RealQKVMath.valueAddress,hm,hu,Nat.mul_one,
    Nat.add_assoc]
  split <;> rfl

theorem history_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (c : Nat)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a.history off) (hu : u.undef r = fun _ => 0)
    (hp : u.regs .blockPtr [p.BV,p.BK] name = some (historyPointer p l r c)) :
    evalOp (RealHProgram.load2 p.BV p.BK name) u = some (tile (h p l a c)) := by
  rw [load2_eval r u name (l.head*p.historyHead) (p.V) (p.NT*p.K) p.BV p.BK 1 p.historyKey (c*p.BV) (l.chunk*p.K+l.key*p.BK) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [tile,Tile.mat,h,RealQKVMath.history,RealQKVMath.row,RealQKVMath.keyIndex,
    RealQKVMath.valueIndex,RealQKVMath.historyIndex,RealQKVMath.keyAddress,RealQKVMath.valueAddress,hm,hu,Nat.mul_one,
    Nat.add_assoc]
  split <;> rfl

theorem historyGradient_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs) (c : Nat)
    (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a.historyGradient off) (hu : u.undef r = fun _ => 0)
    (hp : u.regs .blockPtr [p.BK,p.BV] name = some (historyGradientPointer p l r c)) :
    evalOp (RealHProgram.load2 p.BK p.BV name) u = some (tile (dh p l a c)) := by
  rw [load2_eval r u name (l.head*p.historyHead) (p.NT*p.K) (p.V) p.BK p.BV p.historyKey 1 (l.chunk*p.K+l.key*p.BK) (c*p.BV) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [tile,Tile.mat,dh,RealQKVMath.historyGradient,RealQKVMath.row,RealQKVMath.keyIndex,
    RealQKVMath.valueIndex,RealQKVMath.historyIndex,RealQKVMath.keyAddress,RealQKVMath.valueAddress,hm,hu,Nat.mul_one,
    Nat.add_assoc]
  split <;> rfl

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVPointers

/-! Arithmetic of the complete gradient source after real erasure. Numerical
FP equivalence is proved separately using only the admitted add_commute atom. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVAlgebra
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs)
open RealQKVProgram RealQKVTiles RealMatrix
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

theorem mask_eval (p : Layout) (lower : Bool) (u : BlockState)
    (hi : u.regs .nat [p.BT] "o_i" = some (Tile.vec fun i => i.val)) :
    evalOp (mask p lower) u = some (Tile.mat fun (i j : Fin p.BT) =>
      if lower then decide (i.val ≥ j.val) else decide (i.val ≤ j.val)) := by
  cases lower
  all_goals
    simp only [mask,Bool.false_eq_true,↓reduceIte,evalOp.eq_def,evalOp_expandDim,evalOp_ref,hi,
      List.nil_append,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    rfl

theorem select_eval (p : Layout) (lower : Bool) (op : Op .real [p.BT,p.BT])
    (f : Values p.BT p.BT) (u : BlockState)
    (hi : u.regs .nat [p.BT] "o_i" = some (Tile.vec fun i => i.val))
    (he : evalOp op u = some (tile f)) :
    evalOp (select p lower op) u = some (tile fun i j =>
      if (if lower then decide (i.val ≥ j.val) else decide (i.val ≤ j.val)) then f i j else 0) := by
  have hz : evalOp (.broadcast (.const (0 : ℝ)) [p.BT,p.BT]) u = some (tile (fun _ _ => 0)) := by
    simp only [evalOp.eq_def,evalOp_const,List.nil_append,Option.bind_eq_bind,Option.bind_some]
    rfl
  simp only [select,evalOp_where,mask_eval p lower u hi,he,hz,List.nil_append,Option.bind_eq_bind,Option.bind_some]
  rw [show (tile fun _ : Fin p.BT => fun _ : Fin p.BT => 0) = tile (fun _ _ => 0) from rfl,select_tile]

theorem score_eval (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hq : u.regs .real [p.BK,p.BT] "b_q" = some (tile (q p l a)))
    (hk : u.regs .real [p.BT,p.BK] "b_k" = some (tile (k p l a))) :
    evalOp (score p scale) u = some (tile fun i j => (∑ e : Fin p.BK, k p l a i e*q p l a e j)*scale) := by
  simp only [score,scaleBy,dot,ref,evalOp_mul,evalOp.eq_def,evalOp_ref,evalOp_const,hq,hk,
    List.nil_append,Option.bind_eq_bind,Option.bind_some,dot_tile,scale_tile]

theorem initial_score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hi : u.regs .nat [p.BT] "o_i" = some (Tile.vec fun i => i.val))
    (hs : u.regs .real [p.BT,p.BT] "b_s" =
      some (tile fun i j => (∑ e : Fin p.BK, k p l a i e*q p l a e j)*scale)) :
    evalOp (select p Bool.false (ref p.BT p.BT "b_s")) u = some (tile (s p l a scale)) := by
  rw [select_eval p Bool.false _ _ u hi (by simpa [ref] using hs)]
  apply congrArg some
  apply congrArg tile
  funext i j
  simp only [s,RealQKVMath.score,k,q,Bool.false_eq_true,↓reduceIte,decide_eq_true_eq]

theorem update_ds (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) (u : BlockState)
    (hds : u.regs .real [p.BT,p.BT] "b_ds" = some (tile (cs p l a n)))
    (ho : u.regs .real [p.BT,p.BV] "b_do" = some (tile (o p l a n)))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (tile (v p l a n)))
    : evalOp (updateDS p) u = some (tile (cs p l a (n+1))) := by
  simp only [updateDS,Bool.false_eq_true,↓reduceIte,add,scaleBy,dot,ref,evalOp_add,evalOp_mul,evalOp.eq_def,evalOp,evalOp_ref,evalOp_const,hds,ho,hv,List.nil_append,Option.bind_eq_bind,Option.bind_some,transpose_tile,dot_tile,add_tile,scale_tile]
  apply congrArg some
  apply congrArg tile
  funext i j
  simp only [cs,o,v,RealQKVMath.dsSum,Finset.sum_range_succ]

theorem update_dq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) (u : BlockState)
    (hdq : u.regs .real [p.BT,p.BK] "b_dq" = some (tile (cq p l a scale n)))
    (ho : u.regs .real [p.BT,p.BV] "b_do" = some (tile (o p l a n)))
    (hh : u.regs .real [p.BV,p.BK] "b_h" = some (tile (h p l a n)))
    : evalOp (updateDQ p scale) u = some (tile (cq p l a scale (n+1))) := by
  simp only [updateDQ,Bool.false_eq_true,↓reduceIte,add,scaleBy,dot,ref,evalOp_add,evalOp_mul,evalOp.eq_def,evalOp,evalOp_ref,evalOp_const,hdq,ho,hh,List.nil_append,Option.bind_eq_bind,Option.bind_some,transpose_tile,dot_tile,add_tile,scale_tile]
  apply congrArg some
  apply congrArg tile
  funext i j
  simp only [cq,o,h,RealQKVMath.dqSum,Finset.sum_range_succ]

theorem update_dk (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) (u : BlockState)
    (hdk : u.regs .real [p.BT,p.BK] "b_dk" = some (tile (ck p l a n)))
    (hv : u.regs .real [p.BT,p.BV] "b_v" = some (tile (v p l a n)))
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (tile (dh p l a n)))
    : evalOp (updateDK p) u = some (tile (ck p l a (n+1))) := by
  simp only [updateDK,Bool.false_eq_true,↓reduceIte,add,scaleBy,dot,ref,evalOp_add,evalOp_mul,evalOp.eq_def,evalOp,evalOp_ref,evalOp_const,hdk,hv,hh,List.nil_append,Option.bind_eq_bind,Option.bind_some,transpose_tile,dot_tile,add_tile,scale_tile]
  apply congrArg some
  apply congrArg tile
  funext i j
  simp only [ck,v,dh,RealQKVMath.dkSum,Finset.sum_range_succ]

theorem output_dv (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) (u : BlockState)
    (hk : u.regs .real [p.BT,p.BK] "b_k" = some (tile (k p l a)))
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (tile (dh p l a n)))
    (hs : u.regs .real [p.BT,p.BT] "b_s" = some (tile (s p l a scale)))
    (ho : u.regs .real [p.BT,p.BV] "b_do" = some (tile (o p l a n)))
    : evalOp (outputDV p) u = some (tile (dv p l a scale n)) := by
  simp only [outputDV,Bool.false_eq_true,↓reduceIte,add,scaleBy,dot,ref,evalOp_add,evalOp_mul,evalOp.eq_def,evalOp,evalOp_ref,evalOp_const,hk,hh,hs,ho,List.nil_append,Option.bind_eq_bind,Option.bind_some,transpose_tile,dot_tile,add_tile,scale_tile]
  apply congrArg some
  apply congrArg tile
  funext i j
  simp only [dv,k,dh,s,o,RealQKVMath.dv]

theorem final_dq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (swapped : Bool) (u : BlockState)
    (hdq : u.regs .real [p.BT,p.BK] "b_dq" = some (tile (cq p l a scale (RealQKVMath.numBlocks p))))
    (hds : u.regs .real [p.BT,p.BT] "b_ds" = some (tile (ds p l a scale)))
    (hk : u.regs .real [p.BT,p.BK] "b_k" = some (tile (k p l a)))
    : evalOp (finalDQ p swapped) u = some (tile (dq p l a scale)) := by
  cases swapped
  all_goals
    simp only [finalDQ,Bool.false_eq_true,↓reduceIte,add,scaleBy,dot,ref,evalOp_add,evalOp_mul,evalOp.eq_def,evalOp,evalOp_ref,evalOp_const,hdq,hds,hk,List.nil_append,Option.bind_eq_bind,Option.bind_some,transpose_tile,dot_tile,add_tile,scale_tile]
    apply congrArg some
    apply congrArg tile
    funext i j
    simp only [dq,cq,ds,k,RealQKVMath.dq,add_comm]

theorem final_dk (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ)  (u : BlockState)
    (hdk : u.regs .real [p.BT,p.BK] "b_dk" = some (tile (ck p l a (RealQKVMath.numBlocks p))))
    (hq : u.regs .real [p.BK,p.BT] "b_q" = some (tile (q p l a)))
    (hds : u.regs .real [p.BT,p.BT] "b_ds" = some (tile (ds p l a scale)))
    : evalOp (finalDK p) u = some (tile (dk p l a scale)) := by
  simp only [finalDK,Bool.false_eq_true,↓reduceIte,add,scaleBy,dot,ref,evalOp_add,evalOp_mul,evalOp.eq_def,evalOp,evalOp_ref,evalOp_const,hdk,hq,hds,List.nil_append,Option.bind_eq_bind,Option.bind_some,transpose_tile,dot_tile,add_tile,scale_tile]
  apply congrArg some
  apply congrArg tile
  funext i j
  simp only [dk,ck,q,ds,RealQKVMath.dk]

theorem final_ds (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hi : u.regs .nat [p.BT] "o_i" = some (Tile.vec fun i => i.val))
    (hs : u.regs .real [p.BT,p.BT] "b_ds" = some (tile (cs p l a (RealQKVMath.numBlocks p)))) :
    evalOp (finalDS p scale) u = some (tile (ds p l a scale)) := by
  unfold finalDS
  rw [select_eval p Bool.true _ _ u hi (show evalOp (scaleBy (ref p.BT p.BT "b_ds") scale) u =
    some (tile fun i j => cs p l a (RealQKVMath.numBlocks p) i j*scale) from by
      simp only [scaleBy,ref,evalOp_mul,evalOp_ref,evalOp_const,hs,List.nil_append,Option.bind_eq_bind,Option.bind_some,scale_tile])]
  apply congrArg some
  apply congrArg tile
  funext i j
  simp only [ds,cs,RealQKVMath.ds,↓reduceIte,decide_eq_true_eq]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealQKVAlgebra

/-! Checked matrix stores: exact active-lane readback and preservation of
all other tagged cells, for arbitrary bases, offsets and both strides. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealMatrixStore
open VeriTile.Triton RealMatrix
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

structure Window (m n : Nat) where
  region : RegionName
  base : Nat
  rows : Nat
  cols : Nat
  rowStride : Nat
  colStride : Nat
  rowOffset : Nat
  colOffset : Nat

def Window.active (w : Window m n) (i j : Nat) : Prop := w.rowOffset+i < w.rows ∧ w.colOffset+j < w.cols
instance (w : Window m n) (i j : Nat) : Decidable (w.active i j) := inferInstanceAs (Decidable (_ ∧ _))
def Window.address (w : Window m n) (i j : Nat) : Nat :=
  w.base+(w.rowOffset+i)*w.rowStride+(w.colOffset+j)*w.colStride
def Window.pointer (w : Window m n) : Tile .blockPtr [m,n] :=
  RealHPointers.matrixPointer w.region w.base w.rows w.cols m n w.rowStride w.colStride w.rowOffset w.colOffset

def Window.scatter (w : Window m n) (f : Values m n) (u : BlockState) : BlockState :=
  (TileShape.allIndices [m,n]).foldl (fun st idx =>
    if w.active idx.1.val idx.2.1.val then st.writeMem w.region (w.address idx.1.val idx.2.1.val) (f idx.1 idx.2.1)
    else st) u

def Window.footprint (w : Window m n) : WriteFootprint := fun cell =>
  cell.1 = w.region ∧ ∃ (i : Fin m) (j : Fin n), w.active i.val j.val ∧ cell.2 = w.address i.val j.val

def Window.NoCollision (w : Window m n) : Prop :=
  ∀ (i e : Fin m) (j f : Fin n), w.active i.val j.val → w.active e.val f.val →
    w.address i.val j.val = w.address e.val f.val → i = e ∧ j = f

theorem store_eval (w : Window m n) (f : Values m n) (ptr value : RegName) (u : BlockState)
    (hv : u.regs .real [m,n] value = some (tile f))
    (hp : u.regs .blockPtr [m,n] ptr = some w.pointer) :
    stepStmt (RealQKVProgram.store2 m n ptr value) u = some (w.scatter f u) := by
  unfold RealQKVProgram.store2 RealQKVProgram.ref stepStmt Window.scatter
  simp only [evalOp_ref,hv,hp,Option.bind,Option.map]
  apply congrArg some
  apply congrArg (fun g => List.foldl g u (TileShape.allIndices [m,n]))
  funext st idx
  rcases idx with ⟨i,j,⟨⟩⟩
  simp only [Window.pointer,RealHPointers.matrixPointer,TileShape.indexToList,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,Bool.true_and,Window.active,Window.address]
  by_cases h : w.rowOffset+i.val < w.rows ∧ w.colOffset+j.val < w.cols
  · simp only [h,decide_true,if_true,BlockState.writeMemTyped_real]; rfl
  · simp only [h,decide_false,Bool.false_eq_true,if_false]

theorem readback (w : Window m n) (f : Values m n) (u : BlockState) (hinj : w.NoCollision)
    (i : Fin m) (j : Fin n) (ha : w.active i.val j.val) :
    (w.scatter f u).readMem w.region (w.address i.val j.val) = f i j := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true (region := w.region) u
    (fun idx : TileIndex [m,n] => w.address idx.1.val idx.2.1.val)
    (fun idx => f idx.1 idx.2.1) (fun idx => w.active idx.1.val idx.2.1.val)
    (i,j,PUnit.unit) ha (by
      rintro ⟨e,g,⟨⟩⟩ hg heq
      obtain ⟨hi,hj⟩ := hinj e i g j hg ha heq
      subst e; subst g; rfl)

theorem frame (w : Window m n) (f : Values m n) (u : BlockState) :
    BlockState.WriteWithin w.footprint u (w.scatter f u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl,idx.1,idx.2.1,ha,rfl⟩

theorem other_region (w : Window m n) (f : Values m n) (u : BlockState) (r : RegionName) (h : r ≠ w.region) :
    ∀ off, (w.scatter f u).readMem r off = u.readMem r off := by
  intro off
  unfold BlockState.readMem
  rw [← frame w f u r off (fun hw => h hw.1)]

@[simp] theorem scatter_regs (w : Window m n) (f : Values m n) (u : BlockState) :
    (w.scatter f u).regs = u.regs := by
  funext dtype shape name
  simp [Window.scatter]
@[simp] theorem scatter_pids (w : Window m n) (f : Values m n) (u : BlockState) :
    (w.scatter f u).pids = u.pids := by simp [Window.scatter,BlockState.foldl_writeMem_prop_masked_pids]
@[simp] theorem scatter_numPids (w : Window m n) (f : Values m n) (u : BlockState) :
    (w.scatter f u).numPids = u.numPids := by
  unfold Window.scatter
  generalize TileShape.allIndices [m,n] = lanes
  induction lanes generalizing u with
  | nil => rfl
  | cons i is ih => simp only [List.foldl_cons,ih]; split <;> rfl
@[simp] theorem scatter_undef (w : Window m n) (f : Values m n) (u : BlockState) :
    (w.scatter f u).undef = u.undef := by
  unfold Window.scatter
  generalize TileShape.allIndices [m,n] = lanes
  induction lanes generalizing u with
  | nil => rfl
  | cons i is ih => simp only [List.foldl_cons,ih]; split <;> rfl

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealMatrixStore

#guard_msgs (drop info) in
#auditModuleAxioms
