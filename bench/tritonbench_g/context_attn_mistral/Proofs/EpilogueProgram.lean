import bench.tritonbench_g.context_attn_mistral.Proofs.LoopProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.EpilogueProgram
open VeriTile.Triton Formula Program PreludeProgram
set_option maxHeartbeats 1200000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def post (p : Layout) (r : Regions) : List Stmt := lower (parts p r Bool.false 0).after

def offsets (p : Layout) : Op .int [p.BM,p.D] :=
  .add .int Broadcast.nil.consL.consR
    (.add .int .scalarR
      (.mul .int .scalarR (.add .int .scalarL (z "cur_batch_in_all_start_index")
        (rowView p).castNatToInt) (Op.constNat p.outRow).castNatToInt)
      (Op.mul .nat .nil (n "cur_head") (.constNat p.outHead)).castNatToInt)
    (Op.mul .nat .scalarR (dimView p) (.constNat p.outDim)).castNatToInt

def code (p : Layout) (r : Regions) : List Stmt :=
  [.assign .int [p.BM,p.D] "off_o" (offsets p),
   .assign .ptr [p.BM,p.D] "out_ptrs"
     (.ptrAdd .scalarL (.ptrBase r.out) (Op.ref .int [p.BM,p.D] "off_o").castIntToNat),
   .store .real [p.BM,p.D] (.ptr (.ref .ptr [p.BM,p.D] "out_ptrs"))
     (.ref .real [p.BM,p.D] "acc") (.mask (queryMask p))]

theorem shared (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    lower (parts p r swapped scale).after = post p r := by
  cases swapped <;> rfl

theorem source (p : Layout) (r : Regions) : post p r = code p r := by
  simp only [post,lower,parts,mathematical,kernel,VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    code,offsets,queryMask,rowView,dimView,rows,dimensions,n,z]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id,eraseDType_float]
  rfl
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.EpilogueProgram
