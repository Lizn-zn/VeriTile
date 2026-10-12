import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHSource
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHMath

/-! Exact decomposition of both complete erased H programs. Pointer layout,
omitted padding, the seed addition, pre-update history and final store are
unchanged. `source_body` connects these proof helpers to every typed storage
combination and both candidate orders; none is a standalone replacement kernel. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHProgram
open VeriTile.Triton RealHMath
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Regions where
  key : RegionName
  value : RegionName
  gate : RegionName
  history : RegionName
  seed : RegionName
  final : RegionName

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 1,u.pids 2⟩
def times (name : RegName) (n : Nat) : Op .nat [] := .mul .nat .nil (.ref .nat [] name) (.constNat n)
def keyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BT] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.keyHead) [p.K,p.T] [p.BK,p.BT]
    [p.keyFeature,p.keyTime] [times "i_k" p.BK,times "i_t" p.BT]
def valuePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BV] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.valueHead) [p.T,p.V] [p.BT,p.BV]
    [p.valueTime,p.valueFeature] [times "i_t" p.BT,times "i_v" p.BV]
def stateBase (p : Layout) : Op .nat [] := .mul .nat .nil (times "i_bh" p.K) (.constNat p.V)
def statePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynOffsets r (stateBase p) [p.K,p.V] [p.BK,p.BV]
    [p.V,1] [times "i_k" p.BK,times "i_v" p.BV]
def historyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynOffsets r
    (.add .nat .nil (times "i_bh" p.historyHead)
      (.mul .nat .nil (times "i_t" p.K) (.constNat p.V)))
    [p.K,p.V] [p.BK,p.BV] [p.historyKey,p.historyValue] [times "i_k" p.BK,times "i_v" p.BV]
def endpointOp (p : Layout) (size block : Nat) (index : RegName) : Op .nat [] :=
  .add .nat .nil
    (.mul .nat .nil (.sub .nat .nil (.add .nat .nil (times "i_t" p.BT) (.constNat p.BT)) (.constNat 1)) (.constNat size))
    (times index block)
def endpointPtr (p : Layout) (r : RegionName) (head stride size block : Nat) (index : RegName) : Op .blockPtr [block] :=
  .makeBlockPtrDynOffsets r (times "i_bh" head) [p.T*size] [block] [stride] [endpointOp p size block index]

def load2 (a b : Nat) (name : RegName) : Op .real [a,b] :=
  .load .real (.blockPtr (.ref .blockPtr [a,b] name) [0,1]) .none .unspecified
def load1 (n : Nat) (name : RegName) : Op .real [n] :=
  .load .real (.blockPtr (.ref .blockPtr [n] name) [0]) .none .unspecified
def carry (p : Layout) : Op .real [p.BK,p.BV] := .ref .real [p.BK,p.BV] "b_h"
def dot (p : Layout) : Op .real [p.BK,p.BV] :=
  .dot (batch := []) (.ref .real [p.BK,p.BT] "b_k") (.ref .real [p.BT,p.BV] "b_v")
    {allowTF32 := some Bool.false}

def initialization (p : Layout) (r : Regions) (hasSeed : Bool) : List Stmt :=
  [.assign .nat [] "i_v" (.programId 0), .assign .nat [] "i_k" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2), .assign .real [p.BK,p.BV] "b_h" (.full [p.BK,p.BV] (.const 0)),
   .ifThen (.constBool hasSeed)
     [.assign .blockPtr [p.BK,p.BV] "p_h" (statePtr p r.seed),
      .assign .real [p.BK,p.BV] "b_h"
        (.add .real (.consSame (.consSame .nil)) (carry p) (load2 p.BK p.BV "p_h"))]]

def gateKey (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_g" (keyPtr p r.gate),
   .assign .blockPtr [p.BK] "p_gn" (endpointPtr p r.gate p.keyHead p.keyFeature p.K p.BK "i_k"),
   .assign .real [p.BK] "b_gn" (load1 p.BK "p_gn"),
   .assign .real [p.BK,p.BV] "b_h"
     (.mul .real (.consSame (.consR .nil)) (carry p)
       (.expandDim ⟨1,by simp⟩ (.exp (.ref .real [p.BK] "b_gn")))),
   .assign .real [p.BK,p.BT] "b_g" (load2 p.BK p.BT "p_g"),
   .assign .real [p.BK,p.BT] "b_k"
     (.mul .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BT] "b_k")
       (.exp (.sub .real (.consSame (.consL .nil))
         (.expandDim ⟨1,by simp⟩ (.ref .real [p.BK] "b_gn")) (.ref .real [p.BK,p.BT] "b_g"))))]

def gateValue (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BT,p.BV] "p_g" (valuePtr p r.gate),
   .assign .blockPtr [p.BV] "p_gn" (endpointPtr p r.gate p.valueHead p.valueFeature p.V p.BV "i_v"),
   .assign .real [p.BV] "b_gn" (load1 p.BV "p_gn"),
   .assign .real [p.BK,p.BV] "b_h"
     (.mul .real (.consR (.consSame .nil)) (carry p)
       (.expandDim ⟨0,by simp⟩ (.exp (.ref .real [p.BV] "b_gn")))),
   .assign .real [p.BT,p.BV] "b_g" (load2 p.BT p.BV "p_g"),
   .assign .real [p.BT,p.BV] "b_v"
     (.mul .real (.consSame (.consSame .nil)) (.ref .real [p.BT,p.BV] "b_v")
       (.exp (.sub .real (.consL (.consSame .nil))
         (.expandDim ⟨0,by simp⟩ (.ref .real [p.BV] "b_gn")) (.ref .real [p.BT,p.BV] "b_g"))))]

def storeCarry (p : Layout) : Stmt :=
  .store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] "p_h") [0,1]) (carry p) .none

def iteration (p : Layout) (r : Regions) (gateK swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_k" (keyPtr p r.key),
   .assign .blockPtr [p.BT,p.BV] "p_v" (valuePtr p r.value),
   .assign .blockPtr [p.BK,p.BV] "p_h" (historyPtr p r.history),
   storeCarry p,
   .assign .real [p.BK,p.BT] "b_k" (load2 p.BK p.BT "p_k"),
   .assign .real [p.BT,p.BV] "b_v" (load2 p.BT p.BV "p_v"),
   .ifThenElse (.constBool gateK) (gateKey p r) (gateValue p r),
   .assign .real [p.BK,p.BV] "b_h"
     (if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
      else .add .real (.consSame (.consSame .nil)) (carry p) (dot p))]

def finalization (p : Layout) (r : Regions) (storeFinal : Bool) : List Stmt :=
  [.ifThen (.constBool storeFinal)
    [.assign .blockPtr [p.BK,p.BV] "p_h" (statePtr p r.final),storeCarry p]]

def body (p : Layout) (r : Regions) (gateK hasSeed storeFinal swapped : Bool) : List Stmt :=
  initialization p r hasSeed ++ [.forRange "i_t" 0 p.NT 1 (iteration p r gateK swapped)] ++ finalization p r storeFinal

def source (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) :=
  RealHSource.source swapped dot gate history seed final r.key r.value r.gate r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT gateK hasSeed storeFinal

theorem source_body (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) :
    (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.body =
      body p r gateK hasSeed storeFinal swapped := by
  unfold source
  rw [RealHSource.storage_erasure]
  cases swapped
  all_goals simp only [RealHSource.source, Bool.false_eq_true, ↓reduceIte,
    TypedSource.hOriginal, TypedSource.hSumLeft, TypedSource.hSource,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHProgram
