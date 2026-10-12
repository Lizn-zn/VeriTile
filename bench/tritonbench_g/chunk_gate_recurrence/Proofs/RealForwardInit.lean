import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardLoop
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Complete forward initialization, including the optional seed and chunk-zero
store. Combined with the loop invariant, this proves the full original source
from ordinary input-memory facts, without a carried-state hypothesis. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardInit
open VeriTile.Triton RealProgram RealForwardProgram RealForwardStep RealForwardLoop
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def headOp (p : Layout) (isSeed : Bool) : Op .nat [] :=
  .mul .nat .nil (.mul .nat .nil
    (if isSeed then .ref .nat [] "offset_bh"
      else .mul .nat .nil (.ref .nat [] "offset_bh") (.constNat p.N))
    (.constNat p.K)) (.constNat p.V)

def pointerOp (p : Layout) (r : RegionName) (isSeed : Bool) : Op .ptr [p.BK,p.BV] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarR
        (.add .nat .scalarL
          (.add .nat .nil (headOp p isSeed)
            (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "offset_d") (.constNat p.V))
              (.constNat p.BK)))
          (.mul .nat .scalarR (.expandDim ⟨1,by simp⟩ (.arange p.BK)) (.constNat p.V)))
        (.mul .nat .nil (.ref .nat [] "offset_s") (.constNat p.BV)))
      (.expandDim ⟨0,by simp⟩ (.arange p.BV)))

def gateOp (p : Layout) (D : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase D) (.mul .nat .nil (.ref .nat [] "offset_bh") (.constNat p.N))

def seedCode (p : Layout) (seed : RegionName) : List Stmt :=
  [.assign .ptr [p.BK,p.BV] "last_kv" (pointerOp p seed Bool.true),
   .assign .real [p.BK,p.BV] "acc" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "last_kv")) .none)]

def zeroCode (p : Layout) : List Stmt :=
  [.assign .real [p.BK,p.BV] "acc" (.full [p.BK,p.BV] (.const 0))]

def code (p : Layout) (S D O seed : RegionName) (hasSeed : Bool) : List Stmt :=
  [.assign .nat [] "offset_bh" (.programId 0),
   .assign .nat [] "offset_d" (.programId 1),
   .assign .nat [] "offset_s" (.programId 2),
   .assign .ptr [p.BK,p.BV] "S" (pointerOp p S Bool.false),
   .assign .ptr [p.BK,p.BV] "O" (pointerOp p O Bool.false),
   .ifThenElse (.constBool hasSeed) (seedCode p seed) (zeroCode p),
   .store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "O")) (.ref .real [p.BK,p.BV] "acc") .none,
   .assign .ptr [p.BK,p.BV] "O" (.ptrAdd .scalarR (.ref .ptr [p.BK,p.BV] "O") (stride p)),
   .assign .ptr [] "D" (gateOp p D)]

theorem code_exact (p : Layout) (S D O seed : RegionName) (hasSeed : Bool) :
    initCode p S D O seed hasSeed = code p S D O seed hasSeed := by
  simp only [initCode, eraseCode, parts, chunk_gate_recurrence_fwd_surface,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [code, seedCode, zeroCode, pointerOp, headOp, gateOp, stride,
    Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, Bool.false_eq_true, ↓reduceIte]
  rfl

def registers (p : Layout) (l : Launch) (S O : RegionName) (u : BlockState) : BlockState :=
  let u1 := u.setReg "offset_bh" .nat [] (Tile.scalar l.head)
  let u2 := u1.setReg "offset_d" .nat [] (Tile.scalar l.key)
  let u3 := u2.setReg "offset_s" .nat [] (Tile.scalar l.value)
  let u4 := u3.setReg "S" .ptr [p.BK,p.BV] (pointer p l S 0)
  u4.setReg "O" .ptr [p.BK,p.BV] (pointer p l O 0)

def seeded (p : Layout) (l : Launch) (seed : RegionName) (a : Inputs p)
    (hasSeed : Bool) (u : BlockState) : BlockState :=
  let u1 := if hasSeed then u.setReg "last_kv" .ptr [p.BK,p.BV] (seedPointer p l seed) else u
  u1.setReg "acc" .real [p.BK,p.BV] (tile p (RealMath.initial a hasSeed))

def beforeStore (p : Layout) (l : Launch) (S O seed : RegionName) (a : Inputs p)
    (hasSeed : Bool) (u : BlockState) : BlockState :=
  seeded p l seed a hasSeed (registers p l S O u)

def initial (p : Layout) (l : Launch) (S D O seed : RegionName) (a : Inputs p)
    (hasSeed : Bool) (u : BlockState) : BlockState :=
  let u1 := scatter p l O 0 (RealMath.initial a hasSeed) (beforeStore p l S O seed a hasSeed u)
  let u2 := u1.setReg "O" .ptr [p.BK,p.BV] (pointer p l O 1)
  u2.setReg "D" .ptr [] (gatePointer p l D 0)

theorem pointer_eval (p : Layout) (l : Launch) (r : RegionName) (isSeed : Bool) (u : BlockState)
    (hh : u.regs .nat [] "offset_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "offset_d" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "offset_s" = some (Tile.scalar l.value)) :
    evalOp (pointerOp p r isSeed) u =
      some (if isSeed then seedPointer p l r else pointer p l r 0) := by
  cases isSeed
  all_goals
    simp only [pointerOp, headOp, Bool.false_eq_true, ↓reduceIte, evalOp.eq_def,
      hh,hk,hv,Option.bind,bind]
    apply congrArg some
    apply Tile.ext
    rintro ⟨e,j,⟨⟩⟩
    simp [tile_elementwise, Broadcast.leftIndex, Broadcast.rightIndex,
      pointer, seedPointer, address, seedAddress]

theorem gate_eval (p : Layout) (l : Launch) (D : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "offset_bh" = some (Tile.scalar l.head)) :
    evalOp (gateOp p D) u = some (gatePointer p l D 0) := by
  simp only [gateOp, evalOp.eq_def, hh, Option.bind, bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [tile_elementwise, Broadcast.leftIndex, Broadcast.rightIndex, gatePointer, gateAddress]

theorem seed_executes (p : Layout) (l : Launch) (seed : RegionName) (a : Inputs p)
    (hasSeed : Bool) (u : BlockState)
    (hh : u.regs .nat [] "offset_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "offset_d" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "offset_s" = some (Tile.scalar l.value))
    (hm : hasSeed = Bool.true → ∀ e j, u.readMem seed (seedAddress p l e.val j.val) = a.seed e j) :
    stepStmt (.ifThenElse (.constBool hasSeed) (seedCode p seed) (zeroCode p)) u =
      some (seeded p l seed a hasSeed u) := by
  cases hasSeed
  · simp only [stepStmt, evalOp.eq_def, Option.bind_some, Option.bind_eq_bind, Tile.scalar_data,
      Bool.false_eq_true, ↓reduceIte, zeroCode, stepStmts, evalOp_full, evalOp_const]
    rfl
  · simp only [stepStmt, evalOp.eq_def, Option.bind_some, Option.bind_eq_bind, Tile.scalar_data,
      ↓reduceIte, seedCode]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l seed Bool.true u hh hk hv))]
    simp only [↓reduceIte]
    have hl : evalOp (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "last_kv")) .none)
        (u.setReg "last_kv" .ptr [p.BK,p.BV] (seedPointer p l seed)) = some (tile p a.seed) := by
      simp only [evalOp, evalOp_ref_setReg_same, Option.bind_some, Option.bind_eq_bind]
      apply congrArg some
      apply Tile.ext
      rintro ⟨e,j,⟨⟩⟩
      simp [seedPointer, tile, Tile.mat, BlockState.readMemValue_real, hm rfl]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some hl), stepStmts.nil]
    rfl

theorem executes (p : Layout) (S D O seed : RegionName) (a : Inputs p)
    (hasSeed : Bool) (u : BlockState)
    (hm : hasSeed = Bool.true → ∀ e j, u.readMem seed (seedAddress p (launch u) e.val j.val) = a.seed e j) :
    stepStmts (initCode p S D O seed hasSeed) u =
      some (initial p (launch u) S D O seed a hasSeed u) := by
  rw [code_exact]
  unfold code
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p (launch u) S Bool.false _
    (by simp [launch]) (by simp [launch]) (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p (launch u) O Bool.false _
    (by simp [launch]) (by simp [launch]) (by simp [launch])))]
  rw [stepStmts.cons_some (seed_executes p (launch u) seed a hasSeed _
    (by simp [launch]) (by simp [launch]) (by simp [launch]) (by simpa using hm))]
  rw [stepStmts.cons_some (store_tile p (launch u) O 0 "O" (RealMath.initial a hasSeed) _
    (by cases hasSeed <;> simp [seeded]) (by simp [seeded]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_tile p (launch u) O 0 "O" _
    (by cases hasSeed <;> simp [scatter,seeded])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_eval p (launch u) D _
    (by cases hasSeed <;> simp [scatter,seeded,launch])))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardInit
