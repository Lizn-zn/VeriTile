import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardProgram

/-! The reverse source split at its named loop. The checked body retains both
stores, the whole-tile contraction and every pointer decrement. The DL update
is a separate postlude; it is not one more DI/DG iteration. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardProgram
open VeriTile.Triton RealProgram
open RealForwardProgram (eraseCode stride stop)
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def parts (p : Layout) (S D DI DG DL DS : RegionName) : Source.DynamicRange :=
  (Source.dynamicRange? (chunk_gate_recurrence_bwd_surface S D DI DG DL DS 0
    p.N p.K p.V p.BK p.BV).surfaceBody "_i").getD
      ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def initCode (p : Layout) (S D DI DG DL DS : RegionName) : List Stmt :=
  eraseCode (parts p S D DI DG DL DS).before

def postCode (p : Layout) (S D DI DG DL DS : RegionName) : List Stmt :=
  eraseCode (parts p S D DI DG DL DS).after

def update (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real (.consSame (.consSame .nil))
    (.mul .real .scalarR (.ref .real [p.BK,p.BV] "Dacc") (.ref .real [] "d_i"))
    (.ref .real [p.BK,p.BV] "DS_i")

def gradient (p : Layout) : Op .real [] :=
  .reduceSum ⟨0,by simp⟩ Bool.false
    (.reshape [p.BK*p.BV]
      (.mul .real (.consSame (.consSame .nil))
        (.ref .real [p.BK,p.BV] "Dacc") (.ref .real [p.BK,p.BV] "S_i")))

def gradientStride : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] "NUM_K") (.ref .nat [] "NUM_V")

def loopBody (p : Layout) : List Stmt :=
  [.assign .real [p.BK,p.BV] "S_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "S")) .none),
   .assign .real [p.BK,p.BV] "DS_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "DS")) .none),
   .assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none),
   .assign .real [p.BK,p.BV] "Dacc" (update p),
   .assign .real [] "DG_i" (gradient p),
   .store .real [] (.ptr (.ref .ptr [] "DG")) (.ref .real [] "DG_i") .none,
   .store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DI")) (.ref .real [p.BK,p.BV] "Dacc") .none,
   .assign .ptr [p.BK,p.BV] "S" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "S") (stride p)),
   .assign .ptr [p.BK,p.BV] "DI" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "DI") (stride p)),
   .assign .ptr [p.BK,p.BV] "DS" (.ptrSub .scalarR (.ref .ptr [p.BK,p.BV] "DS") (stride p)),
   .assign .ptr [] "DG" (.ptrSub .nil (.ref .ptr [] "DG") gradientStride),
   .assign .ptr [] "D" (.ptrSub .nil (.ref .ptr [] "D") (.constNat 1))]

def statement (p : Layout) : Stmt :=
  .forRangeDyn "_i" (.constNat 0) (stop p) (.constNat 1) (loopBody p)

theorem loop_exact (p : Layout) (S D DI DG DL DS : RegionName) :
    eraseCode (parts p S D DI DG DL DS).body = loopBody p := by
  simp only [eraseCode, parts, chunk_gate_recurrence_bwd_surface,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]
  rfl

theorem source_body (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) :
    (TypedSource.backward storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel.body =
      initCode p S D DI DG DL DS ++ [statement p] ++ postCode p S D DI DG DL DS := by
  rw [RealSource.backward_body]
  simp only [initCode, postCode, eraseCode, parts, chunk_gate_recurrence_bwd_surface,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardProgram
