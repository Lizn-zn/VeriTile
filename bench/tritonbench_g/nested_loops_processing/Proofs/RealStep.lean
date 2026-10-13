import bench.tritonbench_g.nested_loops_processing.Kernels
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def source (src dst : RegionName) (sm sn : Int) : Kernel :=
  (kernel .int32 Bool.false src dst sm sn).eraseDType.toAlgKernel

def outer (src dst : RegionName) (sm sn : Int) : List Stmt :=
  match ((source src dst sm sn).body[6]? : Option Stmt) with
  | some (.forRange _ _ _ _ body) => body
  | _ => []

def middle (src dst : RegionName) (sm sn : Int) : List Stmt :=
  match ((outer src dst sm sn)[1]? : Option Stmt) with
  | some (.forRange _ _ _ _ body) => body
  | _ => []

def inner (src dst : RegionName) (sm sn : Int) : List Stmt :=
  match ((middle src dst sm sn)[2]? : Option Stmt) with
  | some (.forRange _ _ _ _ body) => body
  | _ => []

abbrev PointerTile := TileIndex [2,2] → RegionName × Nat
abbrev IntTile := TileIndex [2,2] → Int

def bump (ptr : PointerTile) (delta : Nat) : PointerTile :=
  fun lane ↦ ((ptr lane).1, (ptr lane).2 + delta)

noncomputable def loaded (s : BlockState) (ptr : PointerTile) : IntTile :=
  fun lane ↦ s.readMemValue .int (ptr lane).1 (ptr lane).2

noncomputable def scatter (s : BlockState) (ptr : PointerTile) (value : IntTile) : BlockState :=
  (TileShape.allIndices [2,2]).foldl (fun st lane ↦
    st.writeMemTyped .int (ptr lane).1 (ptr lane).2 (value lane)) s

@[simp] theorem foldl_write_regs {α : Type} (dtype : TileDType)
    (region : α → RegionName) (offset : α → Nat) (value : α → TileCarrier dtype)
    (lanes : List α) (s : BlockState) (d : TileDType) (sh : TileShape) (name : RegName) :
    (lanes.foldl (fun t i ↦ t.writeMemTyped dtype (region i) (offset i) (value i)) s).regs
      d sh name = s.regs d sh name := by
  simpa using BlockState.foldl_writeMemTypedAt_masked_regs dtype region offset value
    (fun _ ↦ Bool.true) lanes s d sh name

noncomputable def innerState (s : BlockState) (a c : PointerTile)
    (v1 v2 : IntTile) (delta : Nat) : BlockState :=
  let a' := bump a delta
  let v3 := loaded s a'
  let t := s.setReg "a_ptrs" .ptr [2,2] ⟨a'⟩
  let t := t.setReg "a3" .int [2,2] ⟨v3⟩
  let t := scatter t c v1
  let t := t.setReg "c_ptrs" .ptr [2,2] ⟨bump c delta⟩
  let t := scatter t (bump c delta) v2
  let t := t.setReg "c_ptrs" .ptr [2,2] ⟨bump (bump c delta) delta⟩
  let t := scatter t (bump (bump c delta) delta) v3
  t.setReg "c_ptrs" .ptr [2,2] ⟨bump (bump (bump c delta) delta) delta⟩

theorem inner_step (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 v2 : IntTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs .int [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs .int [2,2] "a2" = some ⟨v2⟩) :
    stepStmts (inner src dst sm sn) s = some (innerState s a c v1 v2 (2 * sn).toNat) := by
  simp [inner, middle, outer, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, ha, hc, h1, h2,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  simp [innerState, scatter, loaded, bump, Tile.bop, Tile.uop, Tile.vec, Tile.ptrAdd,
    NumericDType.mul, ha, hc, h1, h2]
  rfl

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
