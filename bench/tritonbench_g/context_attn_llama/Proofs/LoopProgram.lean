import bench.tritonbench_g.context_attn_llama.Proofs.PreludeProgram
import bench.tritonbench_g.context_attn_llama.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.LoopProgram
open VeriTile.Triton Formula Program PreludeProgram
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def lanes (p : Layout) : Op .nat [p.BN] := .ref .nat [p.BN] "offs_n"
def keyView (p : Layout) := Op.expandDim ⟨0,by simp⟩ (lanes p)
def keyChannel (p : Layout) := Op.expandDim ⟨1,by simp⟩ (dimensions p)
def valueView (p : Layout) := Op.expandDim ⟨1,by simp⟩ (lanes p)
def valueChannel (p : Layout) := dimView p

def pageIndex (p : Layout) : Op .int [p.BN] :=
  .add .int .scalarL
    (.mul .int .nil (Op.constNat p.reqRow).castNatToInt (z "cur_batch_req_idx"))
    (.mul .int .scalarL (Op.constNat p.reqToken).castNatToInt
      (.add .int .scalarL (z "start_n") (lanes p).castNatToInt))
def pageMask (p : Layout) : Op .bool [p.BN] :=
  .lt .int .scalarR (.add .int .scalarL (z "start_n") (lanes p).castNatToInt)
    (z "block_end_loc")
def pageLanes (p : Layout) : Op .int [p.BN] := .ref .int [p.BN] "kv_loc"
def pageKeyView (p : Layout) := Op.expandDim ⟨0,by simp⟩ (pageLanes p)
def pageValueView (p : Layout) := Op.expandDim ⟨1,by simp⟩ (pageLanes p)

def keyOffsets (p : Layout) : Op .int [p.D,p.BN] :=
  .add .int Broadcast.nil.consR.consL
    (.add .int .scalarR
      (.mul .int .scalarR (pageKeyView p) (Op.constNat p.kRow).castNatToInt)
      (Op.mul .nat .nil (n "cur_kv_head") (.constNat p.kHead)).castNatToInt)
    (Op.mul .nat .scalarR (keyChannel p) (.constNat p.kDim)).castNatToInt

def keyMask (p : Layout) : Op .bool [p.D,p.BN] :=
  .remap [p.D,p.BN] (fun x => (⟨0,by decide⟩,x.2.1,PUnit.unit))
    (.lt .int .scalarR (.add .int .scalarL (z "start_n") (keyView p).castNatToInt)
      (z "block_end_loc"))

def keys (p : Layout) (r : Regions) : List Stmt :=
  [.assign .int [] "start_n" (z "start_n"),
   .assign .int [p.BN] "kv_loc"
     (.load .int (.region r.pages (pageIndex p).castIntToNat)
       (.maskOther (pageMask p) (.broadcast (.constInt 0) [p.BN]))),
   .assign .int [p.D,p.BN] "off_k" (keyOffsets p),
   .assign .real [p.D,p.BN] "k"
     (.load .real (.region r.key (Op.ref .int [p.D,p.BN] "off_k").castIntToNat)
       (.maskOther (keyMask p) (.broadcast (.const (0.0 : ℝ)) [p.D,p.BN])))]

def causalMask (p : Layout) : Op .bool [p.BM,p.BN] :=
  .ge .int Broadcast.nil.consL.consR
    (.add .int .scalarR (rowView p).castNatToInt (z "prompt_cache_len"))
    (.add .int .scalarL (z "start_n") (keyView p).castNatToInt)

def scaling (p : Layout) (swapped : Bool) (scale : ℝ) : Op .real [p.BM,p.BN] :=
  if swapped then .mul .real .scalarL (.const scale) (.ref .real [p.BM,p.BN] "qk")
  else .mul .real .scalarR (.ref .real [p.BM,p.BN] "qk") (.const scale)

def scores (p : Layout) (swapped : Bool) (scale : ℝ) : List Stmt :=
  [.assign .real [p.BM,p.BN] "qk" (.dot (batch := [])
     (.ref .real [p.BM,p.D] "q") (.ref .real [p.D,p.BN] "k")),
   .assign .bool [p.BM,p.BN] "mask" (causalMask p),
   .assign .real [p.BM,p.BN] "qk" (.where (.ref .bool [p.BM,p.BN] "mask")
     (scaling p swapped scale) (.broadcast (.sub .real .nil (.const (0.0 : ℝ)) (.const (10e7 : ℝ))) [p.BM,p.BN]))]

def vec (p : Layout) (name : RegName) : Op .real [p.BM] := .ref .real [p.BM] name
def mat (p : Layout) (name : RegName) : Op .real [p.BM,p.BN] := .ref .real [p.BM,p.BN] name
def column (p : Layout) (name : RegName) := Op.expandDim ⟨1,by simp⟩ (vec p name)

def stats (p : Layout) : List Stmt :=
  [.assign .real [p.BM] "m_ij" (.where
     (.gt .real Broadcast.nil.consSame (vec p "m_i") (.reduceMax ⟨1,by simp⟩ Bool.false (mat p "qk")))
     (vec p "m_i") (.reduceMax ⟨1,by simp⟩ Bool.false (mat p "qk"))),
   .assign .real [p.BM,p.BN] "qk" (.sub .real Broadcast.nil.consR.consSame (mat p "qk") (column p "m_ij")),
   .assign .real [p.BM,p.BN] "p" (.exp2 (mat p "qk")),
   .assign .real [p.BM] "l_ij" (.reduceSum ⟨1,by simp⟩ Bool.false (mat p "p")),
   .assign .real [p.BM] "alpha" (.exp2 (.sub .real Broadcast.nil.consSame (vec p "m_i") (vec p "m_ij"))),
   .assign .real [p.BM] "l_i" (.add .real Broadcast.nil.consSame
     (.mul .real Broadcast.nil.consSame (vec p "l_i") (vec p "alpha")) (vec p "l_ij")),
   .assign .real [p.BM,p.D] "acc" (.mul .real Broadcast.nil.consR.consSame
     (.ref .real [p.BM,p.D] "acc") (column p "alpha"))]

def valueOffsets (p : Layout) : Op .int [p.BN,p.D] :=
  .add .int Broadcast.nil.consL.consR
    (.add .int .scalarR
      (.mul .int .scalarR (pageValueView p) (Op.constNat p.vRow).castNatToInt)
      (Op.mul .nat .nil (n "cur_kv_head") (.constNat p.vHead)).castNatToInt)
    (Op.mul .nat .scalarR (valueChannel p) (.constNat p.vDim)).castNatToInt

def valueMask (p : Layout) : Op .bool [p.BN,p.D] :=
  .remap [p.BN,p.D] (Broadcast.leftIndex Broadcast.nil.consL.consSame)
    (.lt .int .scalarR (.add .int .scalarL (z "start_n") (valueView p).castNatToInt)
      (z "block_end_loc"))

def values (p : Layout) (r : Regions) : List Stmt :=
  [.assign .int [p.BN,p.D] "off_v" (valueOffsets p),
   .assign .real [p.BN,p.D] "v"
     (.load .real (.region r.value (Op.ref .int [p.BN,p.D] "off_v").castIntToNat)
       (.maskOther (valueMask p) (.broadcast (.const (0.0 : ℝ)) [p.BN,p.D]))),
   .assign .real [p.BM,p.BN] "p" (mat p "p"),
   .assign .real [p.BM,p.D] "acc" (.dotAcc .real .real .real (batch := [])
     (mat p "p") (.ref .real [p.BN,p.D] "v") (.ref .real [p.BM,p.D] "acc")),
   .assign .real [p.BM] "m_i" (vec p "m_ij")]

def body (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List Stmt :=
  lower (parts p r swapped scale).body

/-- The four internal segments account for every instruction in the source loop. -/
theorem source (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    body p r swapped scale = keys p r ++ scores p swapped scale ++ stats p ++ values p r := by
  cases swapped
  all_goals simp only [body,lower,parts,mathematical,kernel,VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,VeriTile.Bench.TritonBenchG.ContextAttnLlama.scaleFirst,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,Source.prefixBefore,List.length_cons,List.length_nil,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    keys,pageIndex,pageMask,pageKeyView,pageValueView,pageLanes,
    keyOffsets,keyMask,keyView,keyChannel,lanes,causalMask,scores,scaling,
    stats,vec,mat,column,values,valueOffsets,valueMask,valueView,valueChannel,
    dimensions,rowView,dimView,rows,n,z,List.cons_append,List.nil_append]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id,eraseDType_float,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.LoopProgram
