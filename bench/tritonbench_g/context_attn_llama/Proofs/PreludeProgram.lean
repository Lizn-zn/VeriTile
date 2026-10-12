import bench.tritonbench_g.context_attn_llama.Proofs.Prelude
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.PreludeProgram
open VeriTile.Triton Formula Program Prelude
set_option maxHeartbeats 1500000
set_option maxRecDepth 2000
set_option pp.deepTerms false
set_option linter.unusedSimpArgs false

def n (name : RegName) : Op .nat [] := .ref .nat [] name
def z (name : RegName) : Op .int [] := .ref .int [] name
def loadMeta (r : RegionName) : Op .int [] := .load .int (.region r (n "cur_batch")) .none

def launchCode (p : Layout) : List Stmt :=
  [.assign .nat [] "start_m" (.programId 0),
   .assign .nat [] "cur_bh" (.programId 1),
   .assign .nat [] "cur_batch" (.floorDiv .nat .nil (n "cur_bh") (.constNat p.heads)),
   .assign .nat [] "cur_head" (.mod .nat .nil (n "cur_bh") (.constNat p.heads)),
   .assign .nat [] "cur_kv_head" (.floorDiv .nat .nil (n "cur_head") (.constNat p.kvGroup))]

def metadataCode (r : Regions) : List Stmt :=
  [.assign .int [] "cur_batch_in_all_start_index" (loadMeta r.start),
   .assign .int [] "prompt_cache_len" (loadMeta r.prompt),
   .assign .int [] "cur_batch_seq_len" (.sub .int .nil (loadMeta r.sequence) (z "prompt_cache_len")),
   .assign .int [] "cur_batch_req_idx" (loadMeta r.request)]

def rows (p : Layout) : Op .nat [p.BM] := .ref .nat [p.BM] "offs_m"
def dimensions (p : Layout) : Op .nat [p.D] := .ref .nat [p.D] "offs_d"
def rowView (p : Layout) := Op.expandDim ⟨1,by simp⟩ (rows p)
def dimView (p : Layout) := Op.expandDim ⟨0,by simp⟩ (dimensions p)
def offset (p : Layout) : Op .int [p.BM,p.D] :=
  .add .int Broadcast.nil.consL.consR
    (.add .int .scalarR
      (.mul .int .scalarR (.add .int .scalarL (z "cur_batch_in_all_start_index")
        (rowView p).castNatToInt) (Op.constNat p.qRow).castNatToInt)
      (Op.mul .nat .nil (n "cur_head") (.constNat p.qHead)).castNatToInt)
    (Op.mul .nat .scalarR (dimView p) (.constNat p.qDim)).castNatToInt

def offsetCode (p : Layout) : List Stmt :=
  [.assign .nat [] "block_start_loc" (Op.mul .nat .nil (.constNat p.BM) (n "start_m")),
   .assign .nat [p.BN] "offs_n" (.arange p.BN),
   .assign .nat [p.D] "offs_d" (.arange p.D),
   .assign .nat [p.BM] "offs_m"
      (.add .nat .scalarL (n "block_start_loc") (.arange p.BM)),
   .assign .int [p.BM,p.D] "off_q" (offset p)]

def queryMask (p : Layout) : Op .bool [p.BM,p.D] :=
  .remap [p.BM,p.D] (Broadcast.leftIndex Broadcast.nil.consL.consSame)
    (.lt .int .scalarR (rowView p).castNatToInt (z "cur_batch_seq_len"))
def loadQuery (p : Layout) (r : Regions) : Op .real [p.BM,p.D] :=
  .load .real (.region r.query (Op.ref .int [p.BM,p.D] "off_q").castIntToNat)
    (.maskOther (queryMask p) (.broadcast (.const (0.0 : ℝ)) [p.BM,p.D]))

def endCandidate (p : Layout) : Op .int [] :=
  .add .int .nil
    (Op.add .nat .nil (n "block_start_loc") (.constNat p.BM)).castNatToInt
    (z "prompt_cache_len")
def fullLength : Op .int [] := .add .int .nil (z "cur_batch_seq_len") (z "prompt_cache_len")
def ending (p : Layout) : Op .int [] :=
  .where (.lt .int .nil (endCandidate p) fullLength) (endCandidate p) fullLength

def queryCode (p : Layout) (r : Regions) : List Stmt :=
  [.assign .real [p.BM,p.D] "q" (loadQuery p r),
   .assign .real [p.BM] "m_i" (.add .real .scalarR (.full [p.BM] (.const 0)) .negInf),
   .assign .real [p.BM] "l_i" (.full [p.BM] (.const 0)),
   .assign .real [p.BM,p.D] "acc" (.full [p.BM,p.D] (.const 0)),
   .assign .nat [] "block_mask" (.where
     (.lt .int .nil (n "block_start_loc").castNatToInt (z "cur_batch_seq_len")) (.constNat 1) (.constNat 0)),
   .assign .int [] "block_end_loc" (ending p)]

theorem erase_view (ax : Fin (shape.length+1)) (e : Op dtype shape) :
    (Op.expandDim ax e).eraseDType = Op.expandDim ax e.eraseDType := by
  rw [Op.eraseDType.eq_def]

theorem source (p : Layout) (r : Regions) :
    prelude p r = launchCode p ++ metadataCode r ++ offsetCode p ++ queryCode p r := by
  rw [prefix_source]
  simp only [prefixSource,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
    NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,List.map_cons,List.map_nil,Region.cast_cast,
    launchCode,metadataCode,offsetCode,queryCode,n,z,loadMeta,rows,dimensions,rowView,dimView,
    offset,queryMask,loadQuery,endCandidate,fullLength,ending,List.cons_append,List.nil_append]
  simp only [erase_view,Op.eraseDType_ref,VeriTile.Triton.eraseDType,Region.cast_id]
  simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType]
  rfl
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.PreludeProgram
