import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealProgram
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealSource

/-! Exact decomposition of the complete forward source for its real proof.
The prefix is extracted by the named source loop; the checked loop expansion
includes all loads, the store and all three pointer advances. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardProgram
open VeriTile.Triton RealProgram
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def parts (p : Layout) (S D O seed : RegionName) (hasSeed : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (chunk_gate_recurrence_fwd_surface S D O seed 0
    p.N p.K p.V p.BK p.BV hasSeed).surfaceBody "_i").getD
      ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def eraseCode (code : List ComputeStmt) : List Stmt :=
  (ComputeKernel.mk [] [] code).eraseDType.toAlgKernel.body

def initCode (p : Layout) (S D O seed : RegionName) (hasSeed : Bool) : List Stmt :=
  eraseCode (parts p S D O seed hasSeed).before

def stride (p : Layout) : Op .nat [] := .mul .nat .nil (.constNat p.K) (.constNat p.V)

def stop (p : Layout) : Op .nat [] := .sub .nat .nil (.constNat p.N) (.constNat 1)

def update (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real (.consSame (.consSame .nil))
    (.mul .real .scalarR (.ref .real [p.BK,p.BV] "acc") (.ref .real [] "d_i"))
    (.ref .real [p.BK,p.BV] "S_i")

def loopBody (p : Layout) : List Stmt :=
  [.assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none),
   .assign .real [p.BK,p.BV] "S_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "S")) .none),
   .assign .real [p.BK,p.BV] "acc" (update p),
   .store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "O")) (.ref .real [p.BK,p.BV] "acc") .none,
   .assign .ptr [] "D" (.ptrAdd .nil (.ref .ptr [] "D") (.constNat 1)),
   .assign .ptr [p.BK,p.BV] "S" (.ptrAdd .scalarR (.ref .ptr [p.BK,p.BV] "S") (stride p)),
   .assign .ptr [p.BK,p.BV] "O" (.ptrAdd .scalarR (.ref .ptr [p.BK,p.BV] "O") (stride p))]

def statement (p : Layout) : Stmt :=
  .forRangeDyn "_i" (.constNat 0) (stop p) (.constNat 1) (loopBody p)

theorem loop_exact (p : Layout) (S D O seed : RegionName) (hasSeed : Bool) :
    eraseCode (parts p S D O seed hasSeed).body = loopBody p := by
  simp only [eraseCode, parts, chunk_gate_recurrence_fwd_surface,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  rfl

theorem source_body (p : Layout) (S D O seed : RegionName) (hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) :
    (TypedSource.forward storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel.body =
      initCode p S D O seed hasSeed ++ [statement p] := by
  rw [RealSource.forward_body]
  simp only [initCode, eraseCode, parts, chunk_gate_recurrence_fwd_surface,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardProgram
