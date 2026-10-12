import bench.tritonbench_g.block_sparse_attn.Proofs.RealValueFormula
import VeriTile.Triton.Semantics.Step
import VeriTile.Triton.Memory.TraceProofs

/-! The actual CSR column read and start_n assignment. The complete signed
layout-offset-plus-visit expression is evaluated before the natural address
conversion. Neither the visit nor the loaded column is assumed nonnegative.
The outer kernel must supply the layout register and signed loop counter. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealColumn
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def offset (stride : Nat) : Op .nat [] :=
  .castIntToNat (.add .int .nil
    (.castNatToInt (.mul .nat .nil (.ref .nat [] "layout_h") (.constNat stride)))
    (.ref .int [] "col_idx_idx"))

def code (BN : Nat) (cols : Region .int) (stride : Nat) : List Stmt :=
  [.assign .int [] "col_idx" (.load .int (.region cols (offset stride)) .none),
   .assign .int [] "start_n" (.mul .int .nil (.ref .int [] "col_idx") (.castNatToInt (.constNat BN)))]

def source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  let stmts := RealProgram.normalizationPrefix swapped BM BN BD p
  stmts.take ((Source.prefixBefore stmts "start_n").length + 1)

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (source swapped BM BN BD p) = code BN p.cols p.columnStride := by
  cases swapped
  all_goals
    simp [source, code, offset, RealProgram.realPart, RealProgram.normalizationPrefix,
      RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

def address (layout stride : Nat) (visit : Int) : Nat := ((layout * stride : Nat) + visit : Int).toNat

def column (s : BlockState) (cols : Region .int) (layout stride : Nat) (visit : Int) : Int :=
  s.readMemValue .int (Region.cast cols) (address layout stride visit)

def result (s : BlockState) (BN : Nat) (cols : Region .int) (layout stride : Nat) (visit : Int) : BlockState :=
  let c := column s cols layout stride visit
  (s.setReg "col_idx" .int [] (Tile.scalar c)).setReg "start_n" .int [] (Tile.scalar (c * (BN : Int)))

theorem offset_eval (s : BlockState) (layout stride : Nat) (visit : Int)
    (hl : s.regs .nat [] "layout_h" = some (Tile.scalar layout))
    (hi : s.regs .int [] "col_idx_idx" = some (Tile.scalar visit)) :
    evalOp (offset stride) s = some (Tile.scalar (address layout stride visit)) := by
  simp [offset, evalOp.eq_def, hl, hi, bind, Option.bind]
  rfl

theorem executes (s : BlockState) (BN : Nat) (cols : Region .int) (layout stride : Nat) (visit : Int)
    (hl : s.regs .nat [] "layout_h" = some (Tile.scalar layout))
    (hi : s.regs .int [] "col_idx_idx" = some (Tile.scalar visit)) :
    stepStmts (code BN cols stride) s = some (result s BN cols layout stride visit) := by
  have he := offset_eval s layout stride visit hl hi
  have hload : evalOp (.load .int (.region cols (offset stride)) .none) s =
      some (Tile.scalar (column s cols layout stride visit)) := by
    rw [evalOp.eq_def]
    simp only [he, bind, Option.bind]
    rfl
  simp [code, stepStmts, stepStmt, hload, result, evalOp.eq_def, bind, Option.bind]
  rfl

theorem state_frame (s : BlockState) (BN : Nat) (cols : Region .int) (layout stride : Nat) (visit : Int) :
    { result s BN cols layout stride visit with regs := s.regs } = s := rfl

theorem register_frame (s : BlockState) (BN : Nat) (cols : Region .int) (layout stride : Nat) (visit : Int)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hc : name ≠ "col_idx") (hs : name ≠ "start_n") :
    (result s BN cols layout stride visit).regs dtype shape name = s.regs dtype shape name := by
  simp [result, hc, hs]

theorem readbacks (s : BlockState) (BN : Nat) (cols : Region .int) (layout stride : Nat) (visit : Int) :
    (result s BN cols layout stride visit).regs .int [] "col_idx" =
        some (Tile.scalar (column s cols layout stride visit)) ∧
    (result s BN cols layout stride visit).regs .int [] "start_n" =
        some (Tile.scalar (column s cols layout stride visit * (BN : Int))) := by
  simp [result]

/-- Pinning the actual signed metadata cell recovers the independent CSR key
start. The column stride is the layout's own stride, not the number of heads. -/
theorem start_matches_input (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BN visit : Nat) (s : BlockState) (cols : Region .int)
    (hc : s.readMemValue .int (Region.cast cols)
      (((schedule.layout * schedule.columnStride : Nat) + CSRMath.first schedule xs + (visit : Int) : Int).toNat) =
      xs.columnIndex ((schedule.layout * schedule.columnStride : Nat) + CSRMath.first schedule xs + (visit : Int))) :
    column s cols schedule.layout schedule.columnStride (CSRMath.first schedule xs + (visit : Int)) * (BN : Int) =
      RealValueFormula.keyStart schedule xs BN visit := by
  simpa only [column, address, RealValueFormula.keyStart, Int.add_assoc] using congrArg (· * (BN : Int)) hc

/-- This metadata load is unmasked. Its complete address must be in bounds;
the signed visit itself may be negative. -/
theorem load_safe (bounds : RegionBounds) (s : BlockState) (cols : Region .int)
    (layout stride : Nat) (visit : Int)
    (hl : s.regs .nat [] "layout_h" = some (Tile.scalar layout))
    (hi : s.regs .int [] "col_idx_idx" = some (Tile.scalar visit))
    (hb : address layout stride visit < bounds (Region.cast cols)) :
    (Op.load .int (.region cols (offset stride)) .none).SafeAt bounds s := by
  have ha := TraceProofs.region_safe cols (offset stride) bounds s
    (Tile.scalar (address layout stride visit)) (offset_eval s layout stride visit hl hi)
    (fun _ => hb) ((MaskOpt.none (dtype := .int)).Active s)
  rw [Op.SafeAt.eq_def]
  exact ⟨by simp [offset, Op.SafeAt.eq_def], trivial, ha⟩

theorem code_safe (bounds : RegionBounds) (s : BlockState) (BN : Nat) (cols : Region .int)
    (layout stride : Nat) (visit : Int)
    (hl : s.regs .nat [] "layout_h" = some (Tile.scalar layout))
    (hi : s.regs .int [] "col_idx_idx" = some (Tile.scalar visit))
    (hb : address layout stride visit < bounds (Region.cast cols)) :
    Stmt.TraceSafeList bounds (code BN cols stride) s := by
  unfold code
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simpa only [Stmt.TraceSafe] using load_safe bounds s cols layout stride visit hl hi hb
  intro t _
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealColumn
