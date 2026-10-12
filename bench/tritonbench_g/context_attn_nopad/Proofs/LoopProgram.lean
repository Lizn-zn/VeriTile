import bench.tritonbench_g.context_attn_nopad.Proofs.PreludeProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.LoopProgram
open VeriTile.Triton Formula Program PreludeProgram
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def keyMask (p : Layout) : Op .bool [p.D,p.BN] :=
  .remap [p.D,p.BN] (fun x => (⟨0,by decide⟩,x.2.1,PUnit.unit))
    (.lt .int .scalarR (.add .int .scalarL (z "start_n") (keyView p).castNatToInt)
      (z "cur_batch_seq_len"))
def keyAdvance (p : Layout) : Op .int [] :=
  .mul .int .nil (.add .int .nil (z "cur_batch_in_all_start_index") (z "start_n"))
    (Op.constNat p.kRow).castNatToInt

def keys (p : Layout) (_r : Regions) : List Stmt :=
  [.assign .int [] "start_n" (z "start_n"),
   .assign .real [p.D,p.BN] "k"
     (.load .real (.ptr (.ptrAdd .scalarR (.ref .ptr [p.D,p.BN] "k_ptrs") (keyAdvance p).castIntToNat))
       (.maskOther (keyMask p) (.broadcast (.const (0.0 : ℝ)) [p.D,p.BN])))]

def causalMask (p : Layout) : Op .bool [p.BM,p.BN] :=
  .ge .int Broadcast.nil.consL.consR (rowView p).castNatToInt
    (.add .int .scalarL (z "start_n") (keyView p).castNatToInt)
def sentinel : Op .real [] := .negInf

def scaling (p : Layout) (swapped : Bool) (scale : ℝ) : Op .real [p.BM,p.BN] :=
  if swapped then .mul .real .scalarL (.const scale) (.ref .real [p.BM,p.BN] "qk")
  else .mul .real .scalarR (.ref .real [p.BM,p.BN] "qk") (.const scale)

def scores (p : Layout) (swapped : Bool) (scale : ℝ) : List Stmt :=
  [.assign .real [p.BM,p.BN] "qk" (.full [p.BM,p.BN] (.const 0)),
   .assign .real [p.BM,p.BN] "qk" (.add .real Broadcast.nil.consSame.consSame
     (.ref .real [p.BM,p.BN] "qk") (.dot (batch := [])
       (.ref .real [p.BM,p.D] "q") (.ref .real [p.D,p.BN] "k"))),
   .assign .real [p.BM,p.BN] "qk" (scaling p swapped scale),
   .assign .real [p.BM,p.BN] "qk" (.where (causalMask p)
     (.ref .real [p.BM,p.BN] "qk") (.broadcast sentinel [p.BM,p.BN]))]

def vec (p : Layout) (name : RegName) : Op .real [p.BM] := .ref .real [p.BM] name
def mat (p : Layout) (name : RegName) : Op .real [p.BM,p.BN] := .ref .real [p.BM,p.BN] name
def column (p : Layout) (name : RegName) := Op.expandDim ⟨1,by simp⟩ (vec p name)

def statsCode (p : Layout) : List Stmt :=
  [.assign .real [p.BM] "m_ij" (.reduceMax ⟨1,by simp⟩ Bool.false (mat p "qk")),
   .assign .real [p.BM,p.BN] "p" (.exp (.sub .real Broadcast.nil.consR.consSame (mat p "qk") (column p "m_ij"))),
   .assign .real [p.BM] "l_ij" (.reduceSum ⟨1,by simp⟩ Bool.false (mat p "p"))]

def normalizerCode (p : Layout) : List Stmt :=
  [.assign .real [p.BM] "m_i_new" (.where (.gt .real Broadcast.nil.consSame (vec p "m_i") (vec p "m_ij")) (vec p "m_i") (vec p "m_ij")),
   .assign .real [p.BM] "alpha" (.exp (.sub .real Broadcast.nil.consSame (vec p "m_i") (vec p "m_i_new"))),
   .assign .real [p.BM] "beta" (.exp (.sub .real Broadcast.nil.consSame (vec p "m_ij") (vec p "m_i_new"))),
   .assign .real [p.BM] "l_i_new" (.add .real Broadcast.nil.consSame
     (.mul .real Broadcast.nil.consSame (vec p "alpha") (vec p "l_i"))
     (.mul .real Broadcast.nil.consSame (vec p "beta") (vec p "l_ij")))]

def scalesCode (p : Layout) : List Stmt :=
  [.assign .real [p.BM] "p_scale" (.div .real Broadcast.nil.consSame (vec p "beta") (vec p "l_i_new")),
   .assign .real [p.BM,p.BN] "p" (.mul .real Broadcast.nil.consR.consSame (mat p "p") (column p "p_scale")),
   .assign .real [p.BM] "acc_scale" (.mul .real Broadcast.nil.consSame
     (.div .real Broadcast.nil.consSame (vec p "l_i") (vec p "l_i_new")) (vec p "alpha")),
   .assign .real [p.BM,p.D] "acc" (.mul .real Broadcast.nil.consR.consSame
     (.ref .real [p.BM,p.D] "acc") (column p "acc_scale"))]

def valueMask (p : Layout) : Op .bool [p.BN,p.D] :=
  .remap [p.BN,p.D] (Broadcast.leftIndex Broadcast.nil.consL.consSame)
    (.lt .int .scalarR (.add .int .scalarL (z "start_n") (valueView p).castNatToInt)
      (z "cur_batch_seq_len"))
def valueAdvance (p : Layout) : Op .int [] :=
  .mul .int .nil (.add .int .nil (z "cur_batch_in_all_start_index") (z "start_n"))
    (Op.constNat p.vRow).castNatToInt

def values (p : Layout) (_r : Regions) : List Stmt :=
  [.assign .real [p.BN,p.D] "v"
     (.load .real (.ptr (.ptrAdd .scalarR (.ref .ptr [p.BN,p.D] "v_ptrs") (valueAdvance p).castIntToNat))
       (.maskOther (valueMask p) (.broadcast (.const (0.0 : ℝ)) [p.BN,p.D])))]

def commitCode (p : Layout) : List Stmt :=
  [.assign .real [p.BM,p.BN] "p" (mat p "p"),
   .assign .real [p.BM,p.D] "acc" (.add .real Broadcast.nil.consSame.consSame
     (.ref .real [p.BM,p.D] "acc") (.dot (batch := []) (mat p "p") (.ref .real [p.BN,p.D] "v"))),
   .assign .real [p.BM] "l_i" (vec p "l_i_new"),
   .assign .real [p.BM] "m_i" (vec p "m_i_new")]

def body (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List Stmt :=
  lower (parts p r swapped scale).body

/-- The seven internal segments account for every instruction in the source loop. -/
theorem source (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    body p r swapped scale = keys p r ++ scores p swapped scale ++ statsCode p ++ normalizerCode p ++ scalesCode p ++ values p r ++ commitCode p := by
  cases swapped
  all_goals simp only [body,lower,parts,mathematical,kernel,VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,VeriTile.Bench.TritonBenchG.ContextAttnNopad.scaleFirst,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,Source.prefixBefore,List.length_cons,List.length_nil,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    keys,keyAdvance,keyMask,keyView,keyChannel,lanes,causalMask,sentinel,scores,scaling,
    statsCode,normalizerCode,scalesCode,commitCode,vec,mat,column,values,valueAdvance,valueMask,valueView,
    dimensions,rowView,dimView,rows,n,z,List.cons_append,List.nil_append]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id,eraseDType_float,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.LoopProgram
