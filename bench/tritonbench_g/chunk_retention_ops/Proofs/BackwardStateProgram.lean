import bench.tritonbench_g.chunk_retention_ops.Kernels
import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateProgram

namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then backwardHDotLeft else backwardH
variable (swapped : Bool) (dot : Storage)
    (out_dtype : FloatDType) (q do_ dh : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT BK BV NT : Nat)
    (scale : ℝ)

theorem projects :
    (source swapped dot out_dtype q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).toAlgorithm? =
    .ok (source swapped dot out_dtype q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).toAlgKernel := by
  cases swapped <;> cases dot <;> rfl

theorem storage_erasure :
    (source swapped dot out_dtype q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).eraseDType.toAlgKernel.body =
    (source swapped .fp32 .real q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, backwardH,
    backwardHDotLeft, backwardHSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases out_dtype <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhSource

namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhMath
open VeriTile.Bench.TritonBenchG.ChunkRetention
open RealHMath (Layout Launch keyIndex valueIndex timeIndex keyAddress valueAddress historyAddress)
open scoped BigOperators
noncomputable section

structure Inputs where
  query : Nat → ℝ
  incoming : Nat → ℝ

def query (p : Layout) (l : Launch) (a : Inputs) (k t : Nat) : ℝ :=
  if k < p.K ∧ t < p.T then a.query (keyAddress p l k t) else 0

def incoming (p : Layout) (l : Launch) (a : Inputs) (t v : Nat) : ℝ :=
  if t < p.T ∧ v < p.V then a.incoming (valueAddress p l t v) else 0

def contribution (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (c k v : Nat) : ℝ :=
  ∑ lane : Fin p.BT,
    (query p l a k (timeIndex p c lane.val) * scale) *
      (incoming p l a (timeIndex p c lane.val) v * Decay.query (RealHMath.base p l) lane.val)

/-- The descending carry after n completed chunks; history stores before update. -/
def carry (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) : Nat → Nat → Nat → ℝ
  | 0, _, _ => 0
  | n+1, k, v => Decay.factor (RealHMath.base p l) p.BT * carry p l a scale n k v +
      contribution p l a scale (p.NT-1-n) k v

def closed (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (c k v : Nat) : ℝ :=
  carry p l a scale (p.NT-1-c) k v

@[simp] theorem carry_zero (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (k v : Nat) :
    carry p l a scale 0 k v = 0 := rfl

theorem carry_step (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n k v : Nat) :
    carry p l a scale (n+1) k v = Decay.factor (RealHMath.base p l) p.BT * carry p l a scale n k v +
      contribution p l a scale (p.NT-1-n) k v := rfl

theorem closed_at_reverse (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n k v : Nat) (hn : n < p.NT) :
    closed p l a scale (p.NT-1-n) k v = carry p l a scale n k v := by
  unfold closed
  congr 1
  omega

theorem reverse_index (NT n : Nat) (hn : n < NT) :
    NT-1-n < NT ∧ NT-1-(NT-1-n) = n := by omega

end
end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhMath

/-! Exact descending-state body. An ascending j visits NT-1-j. The previous
carry is stored before loading/scaling the query and incoming gradient.
The loop is empty at NT=0; all source dimensions and strides remain symbolic. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhProgram
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkRetention
open RealHMath (Layout Launch)
open RealHProgram (times keyPtr valuePtr historyPtr load2)
set_option maxRecDepth 18000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

structure Regions where
  query : RegionName
  incoming : RegionName
  history : RegionName

def launch (s : BlockState) : Launch := ⟨s.pids 1,s.pids 0,s.pids 2⟩
def reverseIndex (p : Layout) : Op .nat [] :=
  .sub .nat .nil (.sub .nat .nil (.constNat p.NT) (.constNat 1)) (.ref .nat [] "j")
def carry (p : Layout) : Op .real [p.BK,p.BV] := .ref .real [p.BK,p.BV] "b_dh"
def weightedIncoming (p : Layout) : Op .real [p.BT,p.BV] :=
  .mul .real (.consSame (.consR .nil)) (.ref .real [p.BT,p.BV] "b_do")
    (.expandDim ⟨1,by simp⟩ (.ref .real [p.BT] "d_i"))
def dot (p : Layout) : Op .real [p.BK,p.BV] :=
  .dot (batch := []) (.ref .real [p.BK,p.BT] "b_q") (weightedIncoming p)
    {allowTF32 := some Bool.false}
def scaledQuery (p : Layout) (scale : ℝ) : Op .real [p.BK,p.BT] :=
  .mul .real .scalarR (.ref .real [p.BK,p.BT] "b_q") (.const scale)
def decayedCarry (p : Layout) : Op .real [p.BK,p.BV] :=
  .mul .real .scalarL (.ref .real [] "d_b") (carry p)
def update (p : Layout) (swapped : Bool) : Op .real [p.BK,p.BV] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (decayedCarry p)
  else .add .real (.consSame (.consSame .nil)) (decayedCarry p) (dot p)

def storeCarry (p : Layout) : Stmt :=
  .store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] "p_dh") [0,1]) (carry p) .none

def iterationBody (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_q" (keyPtr p r.query),
   .assign .blockPtr [p.BT,p.BV] "p_do" (valuePtr p r.incoming),
   .assign .blockPtr [p.BK,p.BV] "p_dh" (historyPtr p r.history),
   storeCarry p,
   .assign .real [p.BK,p.BT] "b_q" (load2 p.BK p.BT "p_q"),
   .assign .real [p.BK,p.BT] "b_q" (scaledQuery p scale),
   .assign .real [p.BT,p.BV] "b_do" (load2 p.BT p.BV "p_do"),
   .assign .real [p.BK,p.BV] "b_dh" (update p swapped)]

def iteration (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  .assign .nat [] "i_t" (reverseIndex p) :: iterationBody p r scale swapped

def initialization (p : Layout) : List Stmt :=
  [.assign .nat [] "i_k" (.programId 0),.assign .nat [] "i_v" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [] "i_h" (.mod .nat .nil (.ref .nat [] "i_bh") (.constNat p.H)),
   .assign .real [] "b_b" RealHProgram.baseOp,
   .assign .int [p.BT] "o_i" (.castNatToInt (.arange p.BT)),
   .assign .real [] "d_b" (.exp2 (.mul .real .nil (.const p.BT) (.ref .real [] "b_b"))),
   .assign .real [p.BT] "d_i" (RealDecayTiles.queryOp p.BT),
   .assign .real [p.BK,p.BV] "b_dh" (.full [p.BK,p.BV] (.const 0))]

def body (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  initialization p ++ [.forRange "j" 0 p.NT 1 (iteration p r scale swapped)]

def source (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :=
  RealDhSource.source swapped storage out r.query r.incoming r.history
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT

theorem source_body (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r storage out scale swapped).eraseDType.toAlgKernel.body = body p r scale swapped := by
  unfold source
  rw [RealDhSource.storage_erasure]
  cases swapped
  all_goals simp only [RealDhSource.source,Bool.false_eq_true,↓reduceIte,
    backwardH,backwardHDotLeft,backwardHSource,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhProgram

namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhTiles
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkRetention
open RealHMath (Layout Launch keyIndex valueIndex)
open RealDhMath RealDhProgram
open RealHTiles (StateValue stateTile keyData valueData)
open scoped BigOperators
noncomputable section

structure Loaded (p : Layout) where
  base : ℝ
  query : Fin p.BK → Fin p.BT → ℝ
  incoming : Fin p.BT → Fin p.BV → ℝ

def loaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (padding : RegionName → Nat → ℝ) (c : Nat) : Loaded p where
  base := RealHMath.base p l
  query := keyData p l a.query (padding r.query) c
  incoming := valueData p l a.incoming (padding r.incoming) c

def product (x : Loaded p) (scale : ℝ) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  ∑ t : Fin p.BT, (x.query e t * scale) * (x.incoming t j * Decay.query x.base t.val)

def advance (x : Loaded p) (scale : ℝ) (f : StateValue p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  Decay.factor x.base p.BT * f e j + product x scale e j

def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) : StateValue p :=
  fun e j => RealDhMath.carry p l a scale n (keyIndex p l e.val) (valueIndex p l j.val)

theorem product_zero_padding (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (c : Nat) (e : Fin p.BK) (j : Fin p.BV) :
    product (loaded p l r a (fun _ _ => 0) c) scale e j =
      contribution p l a scale c (keyIndex p l e.val) (valueIndex p l j.val) := rfl

theorem advance_expected (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (n : Nat) (hn : n < p.NT) :
    advance (loaded p l r a (fun _ _ => 0) (p.NT-1-n)) scale (expected p l a scale n) = expected p l a scale (n+1) := by
  funext e j
  rfl

end
end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhTiles

namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhStore
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkRetention RealHMath RealHProgram RealHPointers RealHTiles
export RealHStore (active address pointer scatter footprint NoCollision readback frame scatter_undef)
noncomputable section
theorem store_eval (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (nm : RegName) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p frozen))
    (hp : u.regs .blockPtr [p.BK,p.BV] nm = some (pointer p l out base rowStride colStride)) :
    stepStmt (.store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] nm) [0,1])
      (.ref .real [p.BK,p.BV] "b_dh") .none) u = some (scatter p l out base rowStride colStride frozen u) := by
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


end
end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhStore

/-! Real contraction and carry update, after the actual source loads. -/

namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhAlgebra
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkRetention
open RealHMath (Layout Launch)
open RealDhMath RealDhProgram RealDhTiles
open RealHTiles (StateValue stateTile)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem scale_eval (p : Layout) (x : Loaded p) (scale : ℝ) (u : BlockState)
    (hq : u.regs .real [p.BK,p.BT] "b_q" = some (Tile.mat (fun e t => some (x.query e t)))) :
    evalOp (scaledQuery p scale) u = some (Tile.mat (fun e t => some (x.query e t * scale))) := by
  simp only [scaledQuery,evalOp_mul,evalOp_ref,hq,evalOp_const,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,t,⟨⟩⟩
  rfl

theorem weighted_eval (p : Layout) (x : Loaded p) (u : BlockState)
    (hv : u.regs .real [p.BT,p.BV] "b_do" = some (Tile.mat (fun t j => some (x.incoming t j))))
    (hi : u.regs .real [p.BT] "d_i" = some (RealDecayTiles.query p.BT x.base)) :
    evalOp (weightedIncoming p) u = some (Tile.mat (fun t j => some (x.incoming t j * Decay.query x.base t.val))) := by
  unfold weightedIncoming
  have he := evalOp_expandDim_ref_of_regs .real [p.BT] ⟨1,by simp⟩ "d_i" u _ hi
  erw [evalOp_mul,evalOp_ref,hv,he]
  rfl

theorem product_eval (p : Layout) (x : Loaded p) (scale : ℝ) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_q" = some (Tile.mat (fun e t => some (x.query e t * scale))))
    (hv : u.regs .real [p.BT,p.BV] "b_do" = some (Tile.mat (fun t j => some (x.incoming t j))))
    (hi : u.regs .real [p.BT] "d_i" = some (RealDecayTiles.query p.BT x.base)) :
    evalOp (dot p) u = some (stateTile p (product x scale)) := by
  unfold dot
  erw [evalOp_dot]
  erw [evalOp_ref,hk,weighted_eval p x u hv hi]
  simp only [Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [Tile.mat,Option.map₂_some_some,stateTile,product]
  exact (WithBot.coe_sum Finset.univ (fun t : Fin p.BT => (x.query e t * scale) *
    (x.incoming t j * Decay.query x.base t.val))).symm

theorem carry_eval (p : Layout) (x : Loaded p) (f : StateValue p) (u : BlockState)
    (hb : u.regs .real [] "d_b" = some (Tile.scalar (Decay.factor x.base p.BT)))
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p f)) :
    evalOp (decayedCarry p) u = some (stateTile p (fun e j => Decay.factor x.base p.BT * f e j)) := by
  simp only [decayedCarry,evalOp_mul,RealDhProgram.carry,evalOp_ref,hb,hh,
    Option.bind_eq_bind,Option.bind_some]
  rfl

theorem update_eval (p : Layout) (x : Loaded p) (scale : ℝ) (swapped : Bool) (f : StateValue p) (u : BlockState)
    (hb : u.regs .real [] "d_b" = some (Tile.scalar (Decay.factor x.base p.BT)))
    (hi : u.regs .real [p.BT] "d_i" = some (RealDecayTiles.query p.BT x.base))
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p f))
    (hk : u.regs .real [p.BK,p.BT] "b_q" = some (Tile.mat (fun e t => some (x.query e t * scale))))
    (hv : u.regs .real [p.BT,p.BV] "b_do" = some (Tile.mat (fun t j => some (x.incoming t j)))) :
    evalOp (update p swapped) u = some (stateTile p (advance x scale f)) := by
  cases swapped
  all_goals
    simp only [update,Bool.false_eq_true,↓reduceIte,evalOp_add,carry_eval p x f u hb hh,
      product_eval p x scale u hk hv hi,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨e,j,⟨⟩⟩
    simp [advance,stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.add,add_comm]

end
end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhAlgebra

#guard_msgs (drop info) in
#auditModuleAxioms
