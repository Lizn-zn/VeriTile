import bench.tritonbench_g.block_sparse_attn.Proofs.RealProgram
import bench.tritonbench_g.block_sparse_attn.Proofs.CSRMath
import VeriTile.Triton.Memory.TraceProofs

/-! The actual prologue CSR row-bound reads. The layout is selected from the
query head and uses the row stride, independently of the column stride.
Both loaded bounds remain signed; negative starts and empty/reversed ranges
are handled by the later signed-loop proof, not clipped during these reads. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealRows
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def pointer (rows : Region .int) (stride : Nat) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase rows)
    (.add .nat .nil (.mul .nat .nil (.ref .nat [] "layout_h") (.constNat stride))
      (.ref .nat [] "start_m"))

def load (next : Bool) : Op .int [] :=
  .load .int (.ptr (if next then .ptrAdd .nil (.ref .ptr [] "layout_ptr") (.constNat 1)
    else .ref .ptr [] "layout_ptr")) .none

def code (rows : Region .int) (stride layouts : Nat) : List Stmt :=
  [.assign .nat [] "layout_h" (.mod .nat .nil (.ref .nat [] "off_h") (.constNat layouts)),
   .assign .ptr [] "layout_ptr" (pointer rows stride),
   .assign .int [] "start_l" (load Bool.false),
   .assign .int [] "end_l" (load Bool.true)]

def source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  let before := (RealProgram.parts swapped BM BN BD p).before
  before.drop (Source.prefixBefore before "layout_h").length

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (source swapped BM BN BD p) = code p.rows p.rowStride p.layouts := by
  cases swapped
  all_goals
    simp [source, code, load, pointer, RealProgram.realPart, RealProgram.parts,
      RealProgram.source, TypedSource.specialize, Source.intRange?, Source.prefixBefore,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

def address (head layouts stride queryBlock : Nat) : Nat := head % layouts * stride + queryBlock

def bound (s : BlockState) (rows : Region .int) (head layouts stride queryBlock : Nat) (next : Bool) : Int :=
  s.readMemValue .int (Region.cast rows) (address head layouts stride queryBlock + if next then 1 else 0)

def result (s : BlockState) (rows : Region .int) (head layouts stride queryBlock : Nat) : BlockState :=
  let t := s.setReg "layout_h" .nat [] (Tile.scalar (head % layouts))
  let t := t.setReg "layout_ptr" .ptr [] (Tile.scalar (Region.cast rows, address head layouts stride queryBlock))
  let t := t.setReg "start_l" .int [] (Tile.scalar (bound s rows head layouts stride queryBlock Bool.false))
  t.setReg "end_l" .int [] (Tile.scalar (bound s rows head layouts stride queryBlock Bool.true))

theorem executes (s : BlockState) (rows : Region .int) (head layouts stride queryBlock : Nat)
    (hh : s.regs .nat [] "off_h" = some (Tile.scalar head))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar queryBlock)) :
    stepStmts (code rows stride layouts) s = some (result s rows head layouts stride queryBlock) := by
  simp [code, load, pointer, stepStmts, stepStmt, evalOp.eq_def, hh, hm,
    result, address, bound, bind, Option.bind]
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, IntegralDType.mod, Nat.zero_add]

theorem readbacks (s : BlockState) (rows : Region .int) (head layouts stride queryBlock : Nat) :
    (result s rows head layouts stride queryBlock).regs .nat [] "layout_h" =
      some (Tile.scalar (head % layouts)) ∧
    (result s rows head layouts stride queryBlock).regs .int [] "start_l" =
      some (Tile.scalar (bound s rows head layouts stride queryBlock Bool.false)) ∧
    (result s rows head layouts stride queryBlock).regs .int [] "end_l" =
      some (Tile.scalar (bound s rows head layouts stride queryBlock Bool.true)) := by
  simp [result]

theorem state_frame (s : BlockState) (rows : Region .int) (head layouts stride queryBlock : Nat) :
    {result s rows head layouts stride queryBlock with regs := s.regs} = s := rfl

def writes : List RegName := ["layout_h", "layout_ptr", "start_l", "end_l"]

theorem register_frame (s : BlockState) (rows : Region .int) (head layouts stride queryBlock : Nat)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : name ∉ writes) :
    (result s rows head layouts stride queryBlock).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  simp [result, hn]

/-- The independently specified row interval is read from its actual two
metadata cells. Its column stride is irrelevant to these row-bound reads. -/
theorem bounds_eq_input (s : BlockState) (rows : Region .int) (head layouts stride queryBlock columnStride : Nat)
    (xs : CSRMath.Inputs)
    (hfirst : s.readMemValue .int (Region.cast rows) (address head layouts stride queryBlock) =
      xs.rowIndex (address head layouts stride queryBlock))
    (hlast : s.readMemValue .int (Region.cast rows) (address head layouts stride queryBlock + 1) =
      xs.rowIndex (address head layouts stride queryBlock + 1)) :
    let schedule : CSRMath.Schedule :=
      {layout := head % layouts, rowStride := stride, columnStride := columnStride, queryBlock := queryBlock}
    bound s rows head layouts stride queryBlock Bool.false = CSRMath.first schedule xs ∧
    bound s rows head layouts stride queryBlock Bool.true = CSRMath.last schedule xs := by
  simpa [bound, address, CSRMath.first, CSRMath.last] using And.intro hfirst hlast

theorem load_safe (bounds : RegionBounds) (s : BlockState) (rows : Region .int) (offset : Nat)
    (next : Bool) (hp : s.regs .ptr [] "layout_ptr" = some (Tile.scalar (Region.cast rows, offset)))
    (hb : offset + (if next then 1 else 0) < bounds (Region.cast rows)) :
    (load next).SafeAt bounds s := by
  rw [load, Op.SafeAt.eq_def]
  refine ⟨?_, ?_, ?_⟩
  · cases next <;> simp [Op.SafeAt.eq_def]
  · trivial
  · refine ⟨?_, ?_⟩
    · cases next <;> simp [Op.MemorySafe]
    · intro tile ht
      cases next
      · simp only [Bool.false_eq_true, ↓reduceIte, evalOp_ref, hp, Option.some.injEq] at ht
        subst tile
        intro i _
        simpa only [Tile.scalar_data, Nat.add_zero] using hb
      · simp only [↓reduceIte, evalOp_ptrAdd, evalOp_ref, hp, evalOp_constNat,
          bind, Option.bind, Option.some.injEq] at ht
        subst tile
        intro i _
        exact hb

theorem code_safe (bounds : RegionBounds) (s : BlockState) (rows : Region .int)
    (head layouts stride queryBlock : Nat)
    (hh : s.regs .nat [] "off_h" = some (Tile.scalar head))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar queryBlock))
    (hb : address head layouts stride queryBlock + 1 < bounds (Region.cast rows)) :
    Stmt.TraceSafeList bounds (code rows stride layouts) s := by
  let t := s.setReg "layout_h" .nat [] (Tile.scalar (head % layouts))
  let u := t.setReg "layout_ptr" .ptr [] (Tile.scalar (Region.cast rows, address head layouts stride queryBlock))
  have hl : stepStmt (.assign .nat [] "layout_h"
      (.mod .nat .nil (.ref .nat [] "off_h") (.constNat layouts))) s = some t := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def, hh, t]
    rfl
  have hp : stepStmt (.assign .ptr [] "layout_ptr" (pointer rows stride)) t = some u := by
    apply stepStmt_assign_eq_some
    simp [pointer, evalOp.eq_def, hm, t]
    simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, NumericDType.mul, Nat.zero_add]
    rfl
  rw [code, TraceProofs.cons_of_run_iff hl]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff hp]
  refine ⟨by simp [Stmt.TraceSafe, pointer, Op.SafeAt.eq_def], ?_⟩
  apply TraceProofs.assign_cons (load_safe bounds u rows (address head layouts stride queryBlock)
    Bool.false (by simp [u]) (by simp only [Bool.false_eq_true, ↓reduceIte, Nat.add_zero]; omega))
  intro v
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simpa only [Stmt.TraceSafe] using load_safe bounds (u.setReg "start_l" .int [] v) rows
    (address head layouts stride queryBlock) Bool.true (by simp [u]) hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealRows
