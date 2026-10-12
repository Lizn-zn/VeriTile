import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Math
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Program
open VeriTile.Triton Math
set_option maxRecDepth 20000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

structure Regions where
  dt : RegionName
  rate : RegionName
  bias : RegionName
  out : RegionName
  sum : RegionName

def kernel (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :=
  source bias swapped dtype out_dtype r.dt r.rate r.bias r.out r.sum
    p.batches p.sequence p.heads p.chunk lower upper
    p.dtBatch p.dtTime p.dtHead p.rateHead p.biasHead
    p.outBatch p.outChunk p.outHead p.outColumn p.sumBatch p.sumChunk p.sumHead p.sumColumn
    useSoftplus hasBias p.BH p.BC

def headOp (p : Layout) : Op .nat [p.BH] := .ref .nat [p.BH] "offs_h"
def columnOp (p : Layout) : Op .nat [p.BC] := .ref .nat [p.BC] "offs_c"
def headMask (p : Layout) : Op .bool [p.BH] := .lt .nat .scalarR (headOp p) (.constNat p.heads)
def mask (p : Layout) (limit : Op .nat []) : Op .bool [p.BH,p.BC] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .nat .scalarR (.expandDim ⟨1,by simp⟩ (headOp p)) (.constNat p.heads))
    (.lt .nat .scalarR (.expandDim ⟨0,by simp⟩ (columnOp p)) limit)
def inputMask (p : Layout) := mask p (.ref .nat [] "chunk_size_limit")
def outputMask (p : Layout) := mask p (.constNat p.chunk)
def dtOp (p : Layout) : Op .real [p.BH,p.BC] := .ref .real [p.BH,p.BC] "dt"
def zero (p : Layout) : Op .real [p.BH,p.BC] := .broadcast (.const 0) [p.BH,p.BC]
def offset (batchStride chunkStride : Nat) : Op .nat [] :=
  .add .nat .nil (.mul .nat .nil (.ref .nat [] "pid_b") (.constNat batchStride))
    (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat chunkStride))
def dtOffset (p : Layout) : Op .nat [] :=
  .add .nat .nil (.mul .nat .nil (.ref .nat [] "pid_b") (.constNat p.dtBatch))
    (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat p.chunk)) (.constNat p.dtTime))
def pointer (p : Layout) (name : RegName) (headStride columnStride : Nat) : Op .ptr [p.BH,p.BC] :=
  .ptrAdd (.consR (.consL .nil))
    (.ptrAdd .scalarL (.ref .ptr [] name)
      (.mul .nat .scalarR (.expandDim ⟨1,by simp⟩ (headOp p)) (.constNat headStride)))
    (.mul .nat .scalarR (.expandDim ⟨0,by simp⟩ (columnOp p)) (.constNat columnStride))
def remaining (p : Layout) : Op .nat [] :=
  .sub .nat .nil (.constNat p.sequence)
    (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat p.chunk))
def limit (p : Layout) : Op .nat [] :=
  .where (.lt .nat .nil (.constNat p.chunk) (remaining p)) (.constNat p.chunk) (remaining p)
def initialization (p : Layout) (r : Regions) : List Stmt :=
  [.assign .nat [] "pid_b" (.programId 0),.assign .nat [] "pid_c" (.programId 1),
   .assign .nat [] "pid_h" (.programId 2),
   .assign .ptr [] "dt_ptr" (.ptrAdd .nil (.ptrBase r.dt) (dtOffset p)),
   .assign .ptr [] "dt_out_ptr" (.ptrAdd .nil (.ptrBase r.out) (offset p.outBatch p.outChunk)),
   .assign .ptr [] "dA_cumsum_ptr" (.ptrAdd .nil (.ptrBase r.sum) (offset p.sumBatch p.sumChunk)),
   .assign .nat [p.BH] "offs_h" (.add .nat .scalarL
     (.mul .nat .nil (.ref .nat [] "pid_h") (.constNat p.BH)) (.arange p.BH)),
   .assign .nat [p.BC] "offs_c" (.arange p.BC),
   .assign .ptr [p.BH,p.BC] "dt_ptrs" (pointer p "dt_ptr" p.dtHead p.dtTime),
   .assign .ptr [p.BH] "A_ptrs" (.ptrAdd .scalarL (.ptrBase r.rate)
     (.mul .nat .scalarR (headOp p) (.constNat p.rateHead))),
   .assign .ptr [p.BH,p.BC] "dt_out_ptrs" (pointer p "dt_out_ptr" p.outHead p.outColumn),
   .assign .ptr [p.BH,p.BC] "dA_cs_ptrs" (pointer p "dA_cumsum_ptr" p.sumHead p.sumColumn),
   .assign .nat [] "chunk_size_limit" (limit p)]
def loadDt (p : Layout) : Op .real [p.BH,p.BC] :=
  .load .real (.ptr (.ref .ptr [p.BH,p.BC] "dt_ptrs")) (.maskOther (inputMask p) (zero p))
def loadBias (p : Layout) (r : Regions) : Op .real [p.BH] :=
  .load .real (.region r.bias (.mul .nat .scalarR (headOp p) (.constNat p.biasHead)))
    (.maskOther (headMask p) (.broadcast (.const 0) [p.BH]))
def addBias (p : Layout) : Op .real [p.BH,p.BC] :=
  .add .real (.consSame (.consR .nil)) (dtOp p)
    (.expandDim ⟨1,by simp⟩ (.ref .real [p.BH] "dt_bias"))
def softplusOp (p : Layout) : Op .real [p.BH,p.BC] :=
  .where (.le .real .scalarR (dtOp p) (.const 20))
    (.log (.add .real .scalarL (.const 1) (.exp (dtOp p)))) (dtOp p)
def lowerClamp (p : Layout) (lower : ℝ) : Op .real [p.BH,p.BC] :=
  .where (.gt .real .scalarR (dtOp p) (.const lower)) (dtOp p) (.broadcast (.const lower) [p.BH,p.BC])
def clamp (p : Layout) (lower : ℝ) (upper : UpperBound) : Op .real [p.BH,p.BC] :=
  match upper with
  | .finite hi => .where (.lt .real .scalarR (lowerClamp p lower) (.const hi))
      (lowerClamp p lower) (.broadcast (.const hi) [p.BH,p.BC])
  | .posInf => .minPosInf (lowerClamp p lower)
def preparation (p : Layout) (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) : List Stmt :=
  [.assign .real [p.BH,p.BC] "dt" (loadDt p),
   .ifThen (.constBool hasBias) [.assign .real [p.BH] "dt_bias" (loadBias p r),
     .assign .real [p.BH,p.BC] "dt" (addBias p)],
   .ifThen (.constBool useSoftplus) [.assign .real [p.BH,p.BC] "dt" (softplusOp p)],
   .assign .real [p.BH,p.BC] "dt" (clamp p lower upper),
   .assign .real [p.BH,p.BC] "dt" (.where (inputMask p) (dtOp p) (zero p))]
def storeDt (p : Layout) : Stmt := .store .real [p.BH,p.BC]
  (.ptr (.ref .ptr [p.BH,p.BC] "dt_out_ptrs")) (dtOp p) (.mask (outputMask p))
def loadRate (p : Layout) : Op .real [p.BH] :=
  .load .real (.ptr (.ref .ptr [p.BH] "A_ptrs"))
    (.maskOther (headMask p) (.broadcast (.const 0) [p.BH]))
def product (p : Layout) (swapped : Bool) : Op .real [p.BH,p.BC] :=
  if swapped then .mul .real (.consSame (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .real [p.BH] "A")) (dtOp p)
  else .mul .real (.consSame (.consR .nil)) (dtOp p)
    (.expandDim ⟨1,by simp⟩ (.ref .real [p.BH] "A"))
def storeSum (p : Layout) : Stmt := .store .real [p.BH,p.BC]
  (.ptr (.ref .ptr [p.BH,p.BC] "dA_cs_ptrs")) (.ref .real [p.BH,p.BC] "dA_cs") (.mask (outputMask p))
def accumulation (p : Layout) (swapped : Bool) : List Stmt :=
  [.assign .real [p.BH] "A" (loadRate p),
   .assign .real [p.BH,p.BC] "dA" (product p swapped),
   .assign .real [p.BH,p.BC] "dA_cs" (.scan .sum ⟨1,by simp⟩ .forward (.ref .real [p.BH,p.BC] "dA")),
   storeSum p]
def body (p : Layout) (r : Regions) (hasBias useSoftplus swapped : Bool) (lower : ℝ) (upper : UpperBound) : List Stmt :=
  initialization p r ++ preparation p r hasBias useSoftplus lower upper ++ [storeDt p] ++ accumulation p swapped

section
variable (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
theorem projects :
    (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).toAlgorithm? =
    .ok (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).toAlgKernel := by
  cases upper <;> cases bias <;> cases swapped <;> rfl

theorem storage_erasure :
    (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel.body =
    (kernel p r .fp32 swapped .real .real hasBias useSoftplus lower upper).eraseDType.toAlgKernel.body := by
  cases upper <;> cases bias <;> cases swapped
  all_goals simp only [kernel, source, finiteSource, infiniteSource, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,eraseDType_float]
  all_goals cases dtype <;> cases out_dtype <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, ComparableDType.eraseDType, NumericDType.eraseDType]

theorem source_body :
    (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel.body =
    body p r hasBias useSoftplus swapped lower upper := by
  rw [storage_erasure]
  cases upper <;> cases swapped
  all_goals simp only [kernel,source,finiteSource,infiniteSource,Bool.false_eq_true,↓reduceIte,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals simp only [ComparableDType.eraseDType, eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
  all_goals simp [body,initialization,preparation,accumulation,storeDt,storeSum,
    headOp,columnOp,headMask,mask,inputMask,outputMask,dtOp,zero,offset,dtOffset,
    pointer,remaining,limit,loadDt,loadBias,addBias,softplusOp,lowerClamp,clamp,loadRate,product]
  all_goals norm_num
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Program
