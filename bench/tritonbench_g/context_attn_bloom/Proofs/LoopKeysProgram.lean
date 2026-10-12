import bench.tritonbench_g.context_attn_bloom.Proofs.RealPreparation
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopKeysProgram
open VeriTile.Triton Math Program RealSetup PreludeProgram
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def keyPrefix (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List ComputeStmt :=
  Source.prefixBefore (parts p r swapped scale).body "qk"
def keys (p : Layout) (r : Regions) : List Stmt := lower (keyPrefix p r Bool.false 0)

def lanes (p : Layout) : Op .nat [p.BN] := .ref .nat [p.BN] "offs_n"
def tokenIndices (p : Layout) : Op .int [p.BN] :=
  .add .int .scalarL (z "start_n") (lanes p).castNatToInt

def gather (p : Layout) (r : Regions) : Op .int [p.BN] :=
  .load .int (.region r.tokens
    (Op.add .int .scalarL
      (.mul .int .nil (Op.constNat p.requestRow).castNatToInt (z "cur_batch_req_idx"))
      (.mul .int .scalarL (Op.constNat p.requestToken).castNatToInt (tokenIndices p))).castIntToNat)
    (.maskOther (.lt .int .scalarR (tokenIndices p) (z "block_end_loc"))
      (.broadcast (.constInt 0) [p.BN]))

def tokenView (p : Layout) := Op.expandDim ⟨0,by simp⟩ (Op.ref .int [p.BN] "kv_loc")
def channelView (p : Layout) := Op.expandDim ⟨1,by simp⟩ (Op.ref .nat [p.D] "offs_d")
def keyOffsets (p : Layout) : Op .int [p.D,p.BN] :=
  .add .int Broadcast.nil.consR.consL
    (.add .int .scalarR (.mul .int .scalarR (tokenView p) (Op.constNat p.kRow).castNatToInt)
      (Op.mul .nat .nil (n "cur_kv_head") (.constNat p.kHead)).castNatToInt)
    (Op.mul .nat .scalarR (channelView p) (.constNat p.kDim)).castNatToInt

def keyMask (p : Layout) : Op .bool [p.D,p.BN] :=
  .boolAnd Broadcast.nil.consR.consL
    (.lt .int .scalarR
      (.add .int .scalarL (z "start_n") (Op.expandDim ⟨0,by simp⟩ (lanes p)).castNatToInt)
      (z "block_end_loc"))
    (.lt .nat .scalarR (channelView p) (.constNat p.headDim))

def code (p : Layout) (r : Regions) : List Stmt :=
  [.assign .int [] "start_n" (z "start_n"),
   .assign .int [p.BN] "kv_loc" (gather p r),
   .assign .int [p.D,p.BN] "off_k" (keyOffsets p),
   .assign .real [p.D,p.BN] "k"
     (.load .real (.region r.key (Op.ref .int [p.D,p.BN] "off_k").castIntToNat)
       (.maskOther (keyMask p) (.broadcast (.const (0.0 : ℝ)) [p.D,p.BN])))]

theorem shared (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    lower (keyPrefix p r swapped scale) = keys p r := by
  cases swapped <;> rfl

theorem source (p : Layout) (r : Regions) : keys p r = code p r := by
  simp only [keys,keyPrefix,lower,parts,mathematical,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,Source.prefixBefore,List.length_cons,List.length_nil,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    code,gather,tokenIndices,tokenView,channelView,keyOffsets,keyMask,lanes,n,z]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id]
  rfl
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopKeysProgram
