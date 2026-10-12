import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardLoop
import VeriTile.Triton.Semantics.ProgramRewrite

/-! The actual backward prologue: program IDs, split counts, lane indices,
all five reverse pointers and the zero accumulator. No carry is an input. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardInit
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers RealBackwardLoop
open RealForwardProgram (eraseCode)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def headOp (p : Layout) (isSeed : Bool) : Op .nat [] :=
  .mul .nat .nil (.mul .nat .nil
    (if isSeed then .ref .nat [] "offset_bh"
      else .mul .nat .nil (.ref .nat [] "offset_bh") (.constNat p.N))
    (.constNat p.K)) (.constNat p.V)

def offsetOp (p : Layout) (isSeed : Bool) : Op .nat [p.BK,p.BV] :=
  .add .nat (.consR (.consL .nil))
    (.add .nat .scalarR
      (.add .nat .scalarL
        (.add .nat .nil (headOp p isSeed)
          (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "offset_d") (.constNat p.V)) (.constNat p.BK)))
        (.mul .nat .scalarR (.expandDim ⟨1,by simp⟩ (.ref .nat [p.BK] "offs_k")) (.constNat p.V)))
      (.mul .nat .nil (.ref .nat [] "offset_s") (.constNat p.BV)))
    (.expandDim ⟨0,by simp⟩ (.ref .nat [p.BV] "offs_v"))

def startOp (p : Layout) (back : Nat) : Op .nat [] :=
  .sub .nat .nil (.constNat p.N) (.constNat back)

def pointerOp (p : Layout) (r : RegionName) (back : Nat) : Op .ptr [p.BK,p.BV] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat .scalarR (offsetOp p Bool.false)
      (.mul .nat .nil (.mul .nat .nil (startOp p back) (.constNat p.K)) (.constNat p.V)))

def seedPointerOp (p : Layout) (r : RegionName) : Op .ptr [p.BK,p.BV] :=
  .ptrAdd .scalarL (.ptrBase r) (offsetOp p Bool.true)

def gradientOp (p : Layout) (DG : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase DG)
    (.add .nat .nil
      (.add .nat .nil
        (.add .nat .nil
          (.mul .nat .nil (.mul .nat .nil
            (.mul .nat .nil (.ref .nat [] "offset_bh") (.constNat p.N))
            (.ref .nat [] "NUM_K")) (.ref .nat [] "NUM_V"))
          (.mul .nat .nil (.ref .nat [] "offset_d") (.ref .nat [] "NUM_V")))
        (.ref .nat [] "offset_s"))
      (.mul .nat .nil (.mul .nat .nil (startOp p 2) (.ref .nat [] "NUM_K")) (.ref .nat [] "NUM_V")))

def gateOp (p : Layout) (D : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase D)
    (.add .nat .nil (.mul .nat .nil (.ref .nat [] "offset_bh") (.constNat p.N)) (startOp p 1))

def prepareCode (p : Layout) : List Stmt :=
  [.assign .nat [] "offset_bh" (.programId 0),
   .assign .nat [] "offset_d" (.programId 1),
   .assign .nat [] "offset_s" (.programId 2),
   .assign .nat [] "NUM_K" (.floorDiv .nat .nil (.constNat p.K) (.constNat p.BK)),
   .assign .nat [] "NUM_V" (.floorDiv .nat .nil (.constNat p.V) (.constNat p.BV)),
   .assign .nat [p.BK] "offs_k" (.arange p.BK),
   .assign .nat [p.BV] "offs_v" (.arange p.BV)]

def pointerCode (p : Layout) (S D DI DG DS : RegionName) : List Stmt :=
  [.assign .ptr [p.BK,p.BV] "S" (pointerOp p S 2),
   .assign .ptr [p.BK,p.BV] "DI" (pointerOp p DI 2),
   .assign .ptr [p.BK,p.BV] "DS" (pointerOp p DS 1),
   .assign .ptr [] "DG" (gradientOp p DG),
   .assign .ptr [] "D" (gateOp p D),
   .assign .real [p.BK,p.BV] "Dacc" (.full [p.BK,p.BV] (.const 0))]

theorem code_exact (p : Layout) (S D DI DG DL DS : RegionName) :
    initCode p S D DI DG DL DS = prepareCode p ++ pointerCode p S D DI DG DS := by
  simp only [initCode, eraseCode, parts, chunk_gate_recurrence_bwd_surface,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType]
  rfl

def prepared (p : Layout) (l : Launch) (u : BlockState) : BlockState :=
  let u1 := u.setReg "offset_bh" .nat [] (Tile.scalar l.head)
  let u2 := u1.setReg "offset_d" .nat [] (Tile.scalar l.key)
  let u3 := u2.setReg "offset_s" .nat [] (Tile.scalar l.value)
  let u4 := u3.setReg "NUM_K" .nat [] (Tile.scalar (numK p))
  let u5 := u4.setReg "NUM_V" .nat [] (Tile.scalar (numV p))
  let u6 := u5.setReg "offs_k" .nat [p.BK] (Tile.vec fun i => i.val)
  u6.setReg "offs_v" .nat [p.BV] (Tile.vec fun i => i.val)

def pointers (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (u : BlockState) : BlockState :=
  let u1 := u.setReg "S" .ptr [p.BK,p.BV] (backwardPointer p l S (p.N-2) 0)
  let u2 := u1.setReg "DI" .ptr [p.BK,p.BV] (backwardPointer p l DI (p.N-2) 0)
  let u3 := u2.setReg "DS" .ptr [p.BK,p.BV] (backwardPointer p l DS (p.N-1) 0)
  let u4 := u3.setReg "DG" .ptr [] (backwardGradient p l DG 0)
  let u5 := u4.setReg "D" .ptr [] (backwardGate p l D 0)
  u5.setReg "Dacc" .real [p.BK,p.BV] (tile p (fun _ _ => 0))

def initial (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (u : BlockState) : BlockState :=
  pointers p l S D DI DG DS (prepared p l u)

theorem pointer_eval (p : Layout) (l : Launch) (r : RegionName) (back : Nat) (u : BlockState)
    (h : Fixed p l u) : evalOp (pointerOp p r back) u = some (backwardPointer p l r (p.N-back) 0) := by
  simp only [pointerOp,offsetOp,headOp,startOp,Bool.false_eq_true,↓reduceIte,evalOp.eq_def,
    h.head,h.key,h.value,h.offsK,h.offsV,Option.bind,bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,backwardPointer,address]

theorem seedPointer_eval (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (h : Fixed p l u) : evalOp (seedPointerOp p r) u = some (seedPointer p l r) := by
  simp only [seedPointerOp,offsetOp,headOp,↓reduceIte,evalOp.eq_def,
    h.head,h.key,h.value,h.offsK,h.offsV,Option.bind,bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,seedPointer,seedAddress]

theorem gradient_eval (p : Layout) (l : Launch) (DG : RegionName) (u : BlockState)
    (h : Fixed p l u) : evalOp (gradientOp p DG) u = some (backwardGradient p l DG 0) := by
  simp only [gradientOp,startOp,evalOp.eq_def,h.head,h.key,h.value,h.numK,h.numV,Option.bind,bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,backwardGradient,gradientAddress]

theorem gate_eval (p : Layout) (l : Launch) (D : RegionName) (u : BlockState)
    (h : Fixed p l u) : evalOp (gateOp p D) u = some (backwardGate p l D 0) := by
  simp only [gateOp,startOp,evalOp.eq_def,h.head,Option.bind,bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,backwardGate,gateAddress]

theorem prepare_executes (p : Layout) (u : BlockState) :
    stepStmts (prepareCode p) u = some (prepared p (launch u) u) := by
  simp only [prepareCode,stepStmts,stepStmt,evalOp.eq_def,Option.bind,bind]
  rfl

theorem prepared_fixed (p : Layout) (l : Launch) (u : BlockState) : Fixed p l (prepared p l u) := by
  constructor <;> simp [prepared]

theorem pointer_executes (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (u : BlockState)
    (h : Fixed p l u) : stepStmts (pointerCode p S D DI DG DS) u = some (pointers p l S D DI DG DS u) := by
  rcases h with ⟨hh,hk,hv,hnk,hnv,hok,hov⟩
  unfold pointerCode
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l S 2 u
    (by constructor <;> assumption)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l DI 2 _
    (by constructor <;> simp [hh,hk,hv,hnk,hnv,hok,hov])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l DS 1 _
    (by constructor <;> simp [hh,hk,hv,hnk,hnv,hok,hov])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (gradient_eval p l DG _
    (by constructor <;> simp [hh,hk,hv,hnk,hnv,hok,hov])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_eval p l D _
    (by constructor <;> simp [hh,hk,hv,hnk,hnv,hok,hov])))]
  simp only [stepStmts,stepStmt,evalOp_full,evalOp_const,Option.bind,bind]
  rfl

theorem executes (p : Layout) (S D DI DG DL DS : RegionName) (u : BlockState) :
    stepStmts (initCode p S D DI DG DL DS) u = some (initial p (launch u) S D DI DG DS u) := by
  rw [code_exact,ProgramRewrite.append,prepare_executes]
  exact pointer_executes p (launch u) S D DI DG DS _ (prepared_fixed p (launch u) u)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardInit
