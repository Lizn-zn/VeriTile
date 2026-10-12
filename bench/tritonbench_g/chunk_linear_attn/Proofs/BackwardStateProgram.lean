import bench.tritonbench_g.chunk_linear_attn.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardStateProgram

/-! Precision erasure of the complete backward state source; no padding,
store order or loop bound is changed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then backwardHDotLeft else backwardH
variable (swapped : Bool) (dot incoming : Storage)
    (out_dtype : FloatDType) (q do_ dh : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t T K V BT BK BV NT : Nat)
    (scale : ℝ)

theorem projects :
    (source swapped dot incoming out_dtype q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm? =
    .ok (source swapped dot incoming out_dtype q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel := by
  cases swapped <;> cases dot <;> cases incoming <;> rfl

theorem storage_erasure :
    (source swapped dot incoming out_dtype q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel.body =
    (source swapped .fp32 .fp32 .real q do_ dh
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot <;> cases incoming
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

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhSource

/-! Independent backward-state formula: sum the scaled query/incoming-gradient
contractions over strictly later chunks. Both inputs refer to initial arrays;
zero extension needs the execution theorem's explicit padding premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhMath
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
    (query p l a k (timeIndex p c lane.val) * scale) * incoming p l a (timeIndex p c lane.val) v

def closed (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (c k v : Nat) : ℝ :=
  ∑ u : Fin p.NT, if c < u.val then contribution p l a scale u.val k v else 0

/-- Number of completed iterations is n; next visited chunk is NT-1-n. -/
def carry (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n k v : Nat) : ℝ :=
  ∑ u : Fin p.NT, if p.NT - n ≤ u.val then contribution p l a scale u.val k v else 0

@[simp] theorem carry_zero (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (k v : Nat) :
    carry p l a scale 0 k v = 0 := by
  refine Finset.sum_eq_zero fun u _ => ?_
  rw [if_neg]
  have := u.isLt
  omega

theorem carry_step (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n k v : Nat) (hn : n < p.NT) :
    carry p l a scale (n+1) k v = carry p l a scale n k v + contribution p l a scale (p.NT-1-n) k v := by
  classical
  unfold carry
  let B := fun u => contribution p l a scale u k v
  have hpt (u : Fin p.NT) :
      (if p.NT-(n+1) ≤ u.val then B u.val else 0) =
      (if p.NT-n ≤ u.val then B u.val else 0) +
      (if u = (⟨p.NT-1-n,by omega⟩ : Fin p.NT) then B u.val else 0) := by
    by_cases h1 : p.NT-n ≤ u.val
    · rw [if_pos (by omega),if_pos h1,if_neg (by
        intro hu
        have : u.val = p.NT-1-n := by rw [hu]
        omega),add_zero]
    · by_cases h2 : u = (⟨p.NT-1-n,by omega⟩ : Fin p.NT)
      · have hval : u.val = p.NT-1-n := by rw [h2]
        rw [if_pos (by omega),if_neg h1,if_pos h2,zero_add]
      · have hval : u.val ≠ p.NT-1-n := fun hv => h2 (Fin.ext hv)
        rw [if_neg (by omega),if_neg h1,if_neg h2,add_zero]
  rw [Finset.sum_congr rfl (fun u _ => hpt u),Finset.sum_add_distrib,
    Finset.sum_ite_eq' Finset.univ (⟨p.NT-1-n,by omega⟩ : Fin p.NT) (fun u => B u.val),
    if_pos (Finset.mem_univ _)]

theorem closed_at_reverse (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n k v : Nat) (hn : n < p.NT) :
    closed p l a scale (p.NT-1-n) k v = carry p l a scale n k v := by
  unfold closed carry
  exact Finset.sum_congr rfl fun u _ => if_congr (by omega) rfl rfl

theorem reverse_index (NT n : Nat) (hn : n < NT) :
    NT-1-n < NT ∧ NT-1-(NT-1-n) = n := by omega

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhMath

/-! Exact descending-state body. An ascending j visits NT-1-j. The previous
carry is stored before loading/scaling the query and incoming gradient.
The loop is empty at NT=0; all source dimensions and strides remain symbolic. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhProgram
open VeriTile.Triton
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
def dot (p : Layout) : Op .real [p.BK,p.BV] :=
  .dot (batch := []) (.ref .real [p.BK,p.BT] "b_q") (.ref .real [p.BT,p.BV] "b_do")
    {allowTF32 := some Bool.false}
def scaledQuery (p : Layout) (scale : ℝ) : Op .real [p.BK,p.BT] :=
  .mul .real .scalarR (.ref .real [p.BK,p.BT] "b_q") (.const scale)
def update (p : Layout) (swapped : Bool) : Op .real [p.BK,p.BV] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
  else .add .real (.consSame (.consSame .nil)) (carry p) (dot p)

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
   .assign .real [p.BK,p.BV] "b_dh" (.full [p.BK,p.BV] (.const 0))]

def body (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  initialization p ++ [.forRange "j" 0 p.NT 1 (iteration p r scale swapped)]

def source (p : Layout) (r : Regions) (query incoming : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :=
  RealDhSource.source swapped query incoming out r.query r.incoming r.history
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV p.NT

theorem source_body (p : Layout) (r : Regions) (query incoming : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r query incoming out scale swapped).eraseDType.toAlgKernel.body = body p r scale swapped := by
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

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhProgram

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhTiles
open VeriTile.Triton
open RealHMath (Layout Launch keyIndex valueIndex)
open RealDhMath RealDhProgram
open RealHTiles (StateValue stateTile keyData valueData)
open scoped BigOperators
noncomputable section

structure Loaded (p : Layout) where
  query : Fin p.BK → Fin p.BT → ℝ
  incoming : Fin p.BT → Fin p.BV → ℝ

def loaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (padding : RegionName → Nat → ℝ) (c : Nat) : Loaded p where
  query := keyData p l a.query (padding r.query) c
  incoming := valueData p l a.incoming (padding r.incoming) c

def product (x : Loaded p) (scale : ℝ) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  ∑ t : Fin p.BT, (x.query e t * scale) * x.incoming t j

def advance (x : Loaded p) (scale : ℝ) (f : StateValue p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  f e j + product x scale e j

def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) : StateValue p :=
  fun e j => RealDhMath.carry p l a scale n (keyIndex p l e.val) (valueIndex p l j.val)

theorem product_zero_padding (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (c : Nat) (e : Fin p.BK) (j : Fin p.BV) :
    product (loaded p l r a (fun _ _ => 0) c) scale e j =
      contribution p l a scale c (keyIndex p l e.val) (valueIndex p l j.val) := rfl

theorem advance_expected (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (n : Nat) (hn : n < p.NT) :
    advance (loaded p l r a (fun _ _ => 0) (p.NT-1-n)) scale (expected p l a scale n) = expected p l a scale (n+1) := by
  funext e j
  simp only [advance,product_zero_padding,expected,carry_step p l a scale n _ _ hn]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhTiles

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhStore
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles
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
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhStore

/-! Real contraction and carry update, after the actual source loads. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhAlgebra
open VeriTile.Triton
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

theorem product_eval (p : Layout) (x : Loaded p) (scale : ℝ) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_q" = some (Tile.mat (fun e t => some ((x.query e t * scale)))))
    (hv : u.regs .real [p.BT,p.BV] "b_do" = some (Tile.mat (fun t j => some (x.incoming t j)))) :
    evalOp (dot p) u = some (stateTile p (product x scale)) := by
  unfold dot
  erw [evalOp_dot]
  erw [evalOp_ref, hk, evalOp_ref, hv]
  simp only [Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [Tile.mat,Option.map₂_some_some,stateTile,product]
  exact (WithBot.coe_sum Finset.univ (fun t : Fin p.BT => (x.query e t * scale) * x.incoming t j)).symm

theorem update_eval (p : Layout) (x : Loaded p) (scale : ℝ) (swapped : Bool) (f : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p f))
    (hk : u.regs .real [p.BK,p.BT] "b_q" = some (Tile.mat (fun e t => some ((x.query e t * scale)))))
    (hv : u.regs .real [p.BT,p.BV] "b_do" = some (Tile.mat (fun t j => some (x.incoming t j)))) :
    evalOp (update p swapped) u = some (stateTile p (advance x scale f)) := by
  cases swapped
  all_goals
    simp only [update,Bool.false_eq_true,↓reduceIte,evalOp_add,RealDhProgram.carry,evalOp_ref,hh,
      product_eval p x scale u hk hv,Option.bind_eq_bind,Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨e,j,⟨⟩⟩
    simp [advance,stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      NumericDType.add,add_comm]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhAlgebra

#guard_msgs (drop info) in
#auditModuleAxioms
