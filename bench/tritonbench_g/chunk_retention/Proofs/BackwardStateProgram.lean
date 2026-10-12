import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateProgram
import bench.tritonbench_g.chunk_retention.Proofs.Source
import bench.tritonbench_g.chunk_gla_simple.Proofs.RealStep

/-! Independent formula for the supplied backward-state source. The two
operands have time/value layout and are multiplied without a transpose.
Only the final accumulated tile is stored, at the source's unusual address.
The query and scale parameters are unused by this source. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhMath
open scoped BigOperators
noncomputable section
structure Layout where
  queryHead : Nat
  queryTime : Nat
  queryFeature : Nat
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
  NT : Nat
structure Launch where
  key : Nat
  value : Nat
  head : Nat
structure Inputs where
  incoming : Nat → ℝ
  value : Nat → ℝ

def base (p : Layout) (l : Launch) := Decay.headBase p.H l.head

def address (p : Layout) (l : Launch) (t v : Nat) :=
  l.head*p.valueHead+t*p.valueTime+v*p.valueFeature

def incoming (p : Layout) (l : Launch) (a : Inputs) (c i e : Nat) : ℝ :=
  if c*p.BT+i < p.T ∧ l.value*p.BT+e < p.V then
    a.incoming (address p l (c*p.BT+i) (l.value*p.BT+e)) else 0

def value (p : Layout) (l : Launch) (a : Inputs) (c e j : Nat) : ℝ :=
  if c*p.BT+e < p.T ∧ l.value*p.BT+j < p.V then
    a.value (address p l (c*p.BT+e) (l.value*p.BT+j)) else 0

def contribution (p : Layout) (l : Launch) (a : Inputs) (c i j : Nat) : ℝ :=
  ∑ e : Fin p.BT,
    (incoming p l a c i e.val * Decay.query (base p l) i) * value p l a c e.val j

/-- Completed iterations visit NT-1, NT-2, ... . No per-iteration store or
per-iteration carry decay occurs in the supplied source. -/
def carry (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ u ∈ Finset.range n, contribution p l a (p.NT-1-u) i j

def expected (p : Layout) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  carry p l a p.NT i j * Decay.factor (base p l) p.BT

def active (p : Layout) (l : Launch) (i j : Nat) : Prop :=
  l.value*p.BT+i < p.K ∧ j < p.V
instance (p : Layout) (l : Launch) (i j : Nat) : Decidable (active p l i j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def outputAddress (p : Layout) (l : Launch) (i j : Nat) : Nat :=
  l.head*p.historyHead+l.key*p.K*p.V+(l.value*p.BT+i)*p.historyKey+j

@[simp] theorem carry_zero (p : Layout) (l : Launch) (a : Inputs) (i j : Nat) :
    carry p l a 0 i j = 0 := by simp [carry]
theorem carry_step (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) :
    carry p l a (n+1) i j = carry p l a n i j + contribution p l a (p.NT-1-n) i j := by
  simp only [carry,Finset.sum_range_succ]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhMath

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhProgram
open VeriTile.Triton RealDhMath
open RealHProgram (times load2)
set_option maxRecDepth 18000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
structure Regions where
  query : RegionName
  incoming : RegionName
  history : RegionName
  value : RegionName

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2⟩
def reverseIndex (p : Layout) : Op .nat [] :=
  .sub .nat .nil (.sub .nat .nil (.constNat p.NT) (.constNat 1)) (.ref .nat [] "j")
def valuePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BT] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.valueHead) [p.T,p.V] [p.BT,p.BT]
    [p.valueTime,p.valueFeature] [times "i_t" p.BT,times "i_v" p.BT]
def historyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BT] :=
  .makeBlockPtrDynOffsets r
    (.add .nat .nil (times "i_bh" p.historyHead) (.mul .nat .nil (times "i_t" p.K) (.constNat p.V)))
    [p.K,p.V] [p.BT,p.BT] [p.historyKey,1] [times "i_k" p.BT,times "i_v" p.BT]
def outputPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BT] :=
  .makeBlockPtrDynOffsets r
    (.add .nat .nil (times "i_bh" p.historyHead) (.mul .nat .nil (times "i_k" p.K) (.constNat p.V)))
    [p.K,p.V] [p.BT,p.BT] [p.historyKey,1] [times "i_v" p.BT,times "i_t" p.BT]
def carry (p : Layout) : Op .real [p.BT,p.BT] := .ref .real [p.BT,p.BT] "b_dh"
def dot (p : Layout) : Op .real [p.BT,p.BT] :=
  .dot (batch := [])
    (.mul .real (.consSame (.consR .nil)) (.ref .real [p.BT,p.BT] "b_o")
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BT] "d_i"))) (.ref .real [p.BT,p.BT] "b_v")
    {allowTF32 := some Bool.false}
def update (p : Layout) (swapped : Bool) : Op .real [p.BT,p.BT] :=
  if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
  else .add .real (.consSame (.consSame .nil)) (carry p) (dot p)
def iterationBody (p : Layout) (r : Regions) (swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BT,p.BT] "p_o" (valuePtr p r.incoming),
   .assign .blockPtr [p.BT,p.BT] "p_v" (valuePtr p r.value),
   .assign .blockPtr [p.BT,p.BT] "p_h" (historyPtr p r.history),
   .assign .real [p.BT,p.BT] "b_o" (load2 p.BT p.BT "p_o"),
   .assign .real [p.BT,p.BT] "b_v" (load2 p.BT p.BT "p_v"),
   .assign .real [p.BT,p.BT] "b_h" (load2 p.BT p.BT "p_h"),
   .assign .real [p.BT,p.BT] "b_dh" (update p swapped)]
def iteration (p : Layout) (r : Regions) (swapped : Bool) : List Stmt :=
  .assign .nat [] "i_t" (reverseIndex p) :: iterationBody p r swapped

def initialization (p : Layout) : List Stmt :=
  [.assign .nat [] "i_k" (.programId 0),.assign .nat [] "i_v" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [] "i_h" (.mod .nat .nil (.ref .nat [] "i_bh") (.constNat p.H)),
   .assign .real [] "b_b" RealHProgram.baseOp,
   .assign .int [p.BT] "o_i" (.castNatToInt (.arange p.BT)),
   .assign .real [] "d_b" (.exp2 (.mul .real .nil (.const p.BT) (.ref .real [] "b_b"))),
   .assign .real [p.BT] "d_i" (RealDecayTiles.queryOp p.BT),
   .assign .real [p.BT,p.BT] "b_dh" (.full [p.BT,p.BT] (.const 0)),
   .assign .nat [] "i_t" (.constNat 0)]
def scaledCarry (p : Layout) : Op .real [p.BT,p.BT] :=
  .mul .real .scalarR (carry p) (.ref .real [] "d_b")
def store (p : Layout) : Stmt :=
  .store .real [p.BT,p.BT] (.blockPtr (.ref .blockPtr [p.BT,p.BT] "p_dh") [0,1]) (carry p) .none
def finalization (p : Layout) (r : Regions) : List Stmt :=
  [.assign .real [p.BT,p.BT] "b_dh" (scaledCarry p),
   .assign .blockPtr [p.BT,p.BT] "p_dh" (outputPtr p r.history), store p]
def body (p : Layout) (r : Regions) (swapped : Bool) : List Stmt :=
  initialization p ++ .forRange "j" 0 p.NT 1 (iteration p r swapped) :: finalization p r

def source (p : Layout) (r : Regions) (dot : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :=
  RealDhSource.source swapped dot out r.query r.incoming r.history r.value
    p.queryHead p.queryTime p.queryFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey scale p.H p.T p.K p.V p.BT p.NT

theorem source_body (p : Layout) (r : Regions) (dot : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r dot out scale swapped).eraseDType.toAlgKernel.body = body p r swapped := by
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
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhProgram

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhStep
open VeriTile.Triton RealDhMath RealDhProgram
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real (makeBlockPtr_2d_eval load_bp_2d_ref mulConst_eval)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def inputs (r : Regions) (s : BlockState) : Inputs := ⟨s.readMem r.incoming,s.readMem r.value⟩
def incomingTile (p : Layout) (l : Launch) (a : Inputs) (c : Nat) : Tile .real [p.BT,p.BT] :=
  Tile.mat (fun i e => some (incoming p l a c i.val e.val))
def valueTile (p : Layout) (l : Launch) (a : Inputs) (c : Nat) : Tile .real [p.BT,p.BT] :=
  Tile.mat (fun e j => some (value p l a c e.val j.val))
def carryTile (p : Layout) (l : Launch) (a : Inputs) (n : Nat) : Tile .real [p.BT,p.BT] :=
  Tile.mat (fun i j => some (RealDhMath.carry p l a n i.val j.val))
def queryDecay (p : Layout) (l : Launch) : Tile .real [p.BT] := RealDecayTiles.query p.BT (base p l)

theorem value_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar c))
    (hv : s.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (valuePtr p r) s = some (⟨fun _ => BlockPtr.mk r (l.head*p.valueHead)
      [p.T,p.V] [p.BT,p.BT] [p.valueTime,p.valueFeature] [c*p.BT,l.value*p.BT]⟩ : Tile .blockPtr [p.BT,p.BT]) := by
  exact makeBlockPtr_2d_eval r s _ _ _ _ _ _ _ _ _
    (mulConst_eval s "i_bh" _ _ hh) (mulConst_eval s "i_t" _ _ ht) (mulConst_eval s "i_v" _ _ hv)

theorem base_eval (p : Layout) (l : Launch) (c : Nat) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (.add .nat .nil (RealHProgram.times "i_bh" p.historyHead)
      (.mul .nat .nil (RealHProgram.times "i_t" p.K) (.constNat p.V))) s =
      some (Tile.scalar (l.head*p.historyHead+c*p.K*p.V)) := by
  unfold RealHProgram.times
  rw [evalOp_add,mulConst_eval s "i_bh" _ _ hh,evalOp_mul,mulConst_eval s "i_t" _ _ ht]
  simp only [evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem history_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : s.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (historyPtr p r) s = some (⟨fun _ => BlockPtr.mk r (l.head*p.historyHead+c*p.K*p.V)
      [p.K,p.V] [p.BT,p.BT] [p.historyKey,1] [l.key*p.BT,l.value*p.BT]⟩ : Tile .blockPtr [p.BT,p.BT]) := by
  exact makeBlockPtr_2d_eval r s _ _ _ _ _ _ _ _ _
    (base_eval p l c s hh ht) (mulConst_eval s "i_k" _ _ hk) (mulConst_eval s "i_v" _ _ hv)

theorem product_eval (p : Layout) (l : Launch) (a : Inputs) (c : Nat) (s : BlockState)
    (ho : s.regs .real [p.BT,p.BT] "b_o" = some (incomingTile p l a c))
    (hv : s.regs .real [p.BT,p.BT] "b_v" = some (valueTile p l a c))
    (hd : s.regs .real [p.BT] "d_i" = some (queryDecay p l)) :
    evalOp (dot p) s = some (Tile.mat (fun i j => some (contribution p l a c i.val j.val))) := by
  have he := evalOp_expandDim_ref_of_regs .real [p.BT] ⟨1,by simp⟩ "d_i" s _ hd
  unfold dot
  erw [evalOp_dot,evalOp_mul,evalOp_ref,ho,he,evalOp_ref,hv]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [Tile.mat,incomingTile,valueTile,queryDecay,RealDecayTiles.query,Tile.vec,
    Tile.bop_data,Tile.expandDim_data,TileShape.dropInsertedIndex,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul,WithBot.realMul,Option.map₂_some_some]
  exact (WithBot.coe_sum Finset.univ (fun e : Fin p.BT =>
    (incoming p l a c i.val e.val * Decay.query (base p l) i.val) * value p l a c e.val j.val)).symm

theorem update_eval (p : Layout) (l : Launch) (a : Inputs) (n : Nat) (swapped : Bool) (s : BlockState)
    (ho : s.regs .real [p.BT,p.BT] "b_o" = some (incomingTile p l a (p.NT-1-n)))
    (hv : s.regs .real [p.BT,p.BT] "b_v" = some (valueTile p l a (p.NT-1-n)))
    (hd : s.regs .real [p.BT] "d_i" = some (queryDecay p l))
    (hc : s.regs .real [p.BT,p.BT] "b_dh" = some (carryTile p l a n)) :
    evalOp (update p swapped) s = some (carryTile p l a (n+1)) := by
  cases swapped
  all_goals simp only [update,Bool.false_eq_true,↓reduceIte,evalOp_add,
    product_eval p l a (p.NT-1-n) s ho hv hd,RealDhProgram.carry,evalOp_ref,hc,Option.bind_eq_bind,Option.bind_some]
  all_goals
    apply congrArg some
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [carryTile,Tile.mat,Tile.bop,NumericDType.add,Broadcast.leftIndex,Broadcast.rightIndex,
      RealDhMath.carry_step,add_comm]

structure Result (p : Layout) (l : Launch) (a : Inputs) (n : Nat) (s t : BlockState) : Prop where
  memory : t.mem = s.mem
  padding : t.undef = s.undef
  pids : t.pids = s.pids
  key : t.regs .nat [] "i_k" = s.regs .nat [] "i_k"
  value : t.regs .nat [] "i_v" = s.regs .nat [] "i_v"
  head : t.regs .nat [] "i_bh" = s.regs .nat [] "i_bh"
  time : t.regs .nat [] "i_t" = s.regs .nat [] "i_t"
  queryDecay : t.regs .real [p.BT] "d_i" = s.regs .real [p.BT] "d_i"
  chunkDecay : t.regs .real [] "d_b" = s.regs .real [] "d_b"
  carry : t.regs .real [p.BT,p.BT] "b_dh" = some (carryTile p l a (n+1))

theorem executes (p : Layout) (l : Launch) (r : Regions) (n : Nat) (swapped : Bool) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : s.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar (p.NT-1-n)))
    (hd : s.regs .real [p.BT] "d_i" = some (queryDecay p l))
    (hc : s.regs .real [p.BT,p.BT] "b_dh" = some (carryTile p l (inputs r s) n)) :
    ∃ t, stepStmts (iterationBody p r swapped) s = some t ∧ Result p l (inputs r s) n s t := by
  unfold iterationBody RealHProgram.load2
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.incoming _ s hh ht hv)),
    stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.value (p.NT-1-n) _
      (by simp [hh]) (by simp [ht]) (by simp [hv]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (history_eval p l r.history (p.NT-1-n) _
      (by simp [hh]) (by simp [ht]) (by simp [hk]) (by simp [hv])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref r.incoming _ "p_o" (l.head*p.valueHead) p.T p.V p.BT p.BT p.valueTime p.valueFeature
      ((p.NT-1-n)*p.BT) (l.value*p.BT) (by simp) (by simpa using hu))),
    stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref r.value _ "p_v" (l.head*p.valueHead) p.T p.V p.BT p.BT p.valueTime p.valueFeature
      ((p.NT-1-n)*p.BT) (l.value*p.BT) (by simp) (by simpa using hu))),
    stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref r.history _ "p_h" (l.head*p.historyHead+(p.NT-1-n)*p.K*p.V) p.K p.V p.BT p.BT p.historyKey 1
      (l.key*p.BT) (l.value*p.BT) (by simp) (by simpa using hu)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (update_eval p l (inputs r s) n swapped _
    (by simp [incomingTile,Tile.mat,incoming,address,inputs]; funext idx; split_ifs <;> rfl)
    (by simp [valueTile,Tile.mat,value,address,inputs]; funext idx; split_ifs <;> rfl) (by simp [hd]) (by simp [hc]))),stepStmts.nil]
  refine ⟨_,rfl,?_⟩
  constructor <;> first | rfl | simp

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhStep

#guard_msgs (drop info) in
#auditModuleAxioms
