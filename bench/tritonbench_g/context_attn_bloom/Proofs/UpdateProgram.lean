import bench.tritonbench_g.context_attn_bloom.Proofs.ScoresProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.ValuesProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.UpdateProgram
open VeriTile.Triton Math Program RealSetup PreludeProgram
set_option maxHeartbeats 1600000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def suffixFrom (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  body.drop (Source.prefixBefore body name occurrence).length

def stats (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :=
  lower (Source.prefixBefore (suffixFrom (parts p r swapped scale).body "m_ij") "m_i_new")
def normalizer (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :=
  lower (Source.prefixBefore (suffixFrom (parts p r swapped scale).body "m_i_new") "p_scale")
def scales (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :=
  lower (Source.prefixBefore (suffixFrom (parts p r swapped scale).body "p_scale") "off_v")
def commit (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :=
  lower (suffixFrom (parts p r swapped scale).body "p" 2)

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

def rescaleGuard (p : Layout) : Op .bool [p.BM] :=
  .ge .int .scalarR (.add .int .scalarR (Op.ref .nat [p.BM] "offs_m").castNatToInt
    (z "prompt_cache_len")) (z "start_n")

def scalesCode (p : Layout) : List Stmt :=
  [.assign .real [p.BM] "p_scale" (.div .real Broadcast.nil.consSame (vec p "beta") (vec p "l_i_new")),
   .assign .real [p.BM,p.BN] "p" (.mul .real Broadcast.nil.consR.consSame (mat p "p") (column p "p_scale")),
   .assign .real [p.BM] "acc_scale" (.mul .real Broadcast.nil.consSame
     (.div .real Broadcast.nil.consSame (vec p "l_i") (vec p "l_i_new")) (vec p "alpha")),
   .assign .real [p.BM] "acc_scale" (.where (rescaleGuard p) (vec p "acc_scale")
     (.broadcast (.const (1.0 : ℝ)) [p.BM])),
   .assign .real [p.BM,p.D] "acc" (.mul .real Broadcast.nil.consR.consSame
     (.ref .real [p.BM,p.D] "acc") (column p "acc_scale"))]

def commitCode (p : Layout) : List Stmt :=
  [.assign .real [p.BM,p.BN] "p" (mat p "p"),
   .assign .real [p.BM,p.D] "acc" (.add .real Broadcast.nil.consSame.consSame
     (.ref .real [p.BM,p.D] "acc") (.dot (batch := []) (mat p "p") (.ref .real [p.BN,p.D] "v"))),
   .assign .real [p.BM] "l_i" (vec p "l_i_new"),
   .assign .real [p.BM] "m_i" (vec p "m_i_new")]

theorem sources (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    stats p r swapped scale = statsCode p ∧
    normalizer p r swapped scale = normalizerCode p ∧
    scales p r swapped scale = scalesCode p ∧
    commit p r swapped scale = commitCode p := by
  cases swapped <;> refine ⟨?_,?_,?_,?_⟩
  all_goals simp only [stats,normalizer,scales,commit,suffixFrom,lower,parts,mathematical,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,Source.prefixBefore,List.length_cons,List.length_nil,
    List.drop_succ_cons,List.drop_zero,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    statsCode,normalizerCode,scalesCode,commitCode,vec,mat,column,rescaleGuard,n,z]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id,eraseDType_float]
  all_goals rfl
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.UpdateProgram
