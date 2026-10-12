import bench.tritonbench_g.block_sparse_attn.Proofs.RealProgram
import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationValues
import VeriTile.Triton.Memory.TraceProofs

/-! The two actual K-load branches. Signed start_n is retained through the
stride product and the upper-bound mask. The cast at the pointer boundary
is exactly the frontend's source lowering, not a nonnegativity assumption.
Neither masked branch supplies an other value. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealKey
open VeriTile.Triton
set_option linter.unusedSimpArgs false

def pointer (BN BD stride : Nat) (second : Bool) : Op .ptr [BD, BN] :=
  let first := Op.ptrAdd .scalarR (.ref .ptr [BD, BN] "k_ptrs")
    (.castIntToNat (.mul .int .nil (.ref .int [] "start_n") (.castNatToInt (.constNat stride))))
  if second then .ptrAdd .scalarR first (.constNat BD) else first

def mask (BN BD context : Nat) : Op .bool [BD, BN] :=
  .remap [BD, BN] (Broadcast.consL (Broadcast.consSame Broadcast.nil)).leftIndex
    (.lt .int .scalarR
      (.add .int .scalarR (.castNatToInt (.expandDim 0 (.ref .nat [BN] "offs_n")))
        (.ref .int [] "start_n"))
      (.castNatToInt (.constNat context)))

def code (BN BD stride context : Nat) (even second : Bool) : List Stmt :=
  [.ifThenElse (.constBool even)
    [.assign .real [BD, BN] "k" (.load .real (.ptr (pointer BN BD stride second)) .none)]
    [.assign .real [BD, BN] "k" (.load .real (.ptr (pointer BN BD stride second))
      (.mask (mask BN BD context)))]]

/-- Locate K loads by their adjacent writes, not fixed statement positions. -/
def firstSource (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  let stmts := RealProgram.normalizationPrefix swapped BM BN BD p
  (Source.prefixBefore stmts "qk").drop ((Source.prefixBefore stmts "start_n").length + 1)

def writesScore (stmts : List ComputeStmt) : Bool := stmts.any fun st =>
  match st with
  | .assign _ _ name _ => name == "qk"
  | _ => Bool.false

def secondSource (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  let branch := (Source.conditional? (RealProgram.normalizationPrefix swapped BM BN BD p)
    writesScore).getD ⟨[], .constBool Bool.false, [], [], []⟩
  Source.prefixBefore branch.yes "qk"

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem first_matches (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (firstSource swapped BM BN BD p) =
      code BN BD p.kRow p.context p.evenN Bool.false := by
  cases swapped
  all_goals
    simp [code, pointer, mask, firstSource,
      RealProgram.realPart, RealProgram.accumulation, RealProgram.normalizationRest,
      RealProgram.normalizationPrefix, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    rfl

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem second_matches (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (secondSource swapped BM BN BD p) =
      code BN BD p.kRow p.context p.evenN Bool.true := by
  cases swapped
  all_goals
    simp [code, pointer, mask, secondSource,
      writesScore,
      RealProgram.realPart, RealProgram.accumulation, RealProgram.normalizationRest,
      RealProgram.normalizationPrefix, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, Source.conditional?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    rfl

open RealNormalizationValues Classical
noncomputable section

def address {BN BD : Nat} (ptrs : Tile .ptr [BD, BN]) (start : Int)
    (stride : Nat) (second : Bool) (d : Fin BD) (j : Fin BN) : RegionName × Nat :=
  let p := ptrs.data (d, j, PUnit.unit)
  (p.1, p.2 + (start * (stride : Int)).toNat + if second then BD else 0)

def active (start : Int) (context : Nat) (even : Bool) {BN : Nat} (j : Fin BN) : Prop :=
  even = Bool.true ∨ (j.val : Int) + start < (context : Int)

def loaded {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN]) (start : Int)
    (stride context : Nat) (even second : Bool) (d : Fin BD) (j : Fin BN) : ℝ :=
  let a := address ptrs start stride second d j
  if active start context even j then s.readMem a.1 a.2 else s.undef a.1 a.2

theorem pointer_eval {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride : Nat) (second : Bool)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start)) :
    evalOp (RealKey.pointer BN BD stride second) s =
      some (Tile.mat (address ptrs start stride second)) := by
  cases second <;>
    simp [RealKey.pointer, evalOp.eq_def, hp, hs, bind, Option.bind]
  all_goals rfl

theorem mask_eval {BN BD : Nat} (s : BlockState) (start : Int) (context : Nat)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) :
    evalOp (RealKey.mask BN BD context) s =
      some (Tile.mat (fun (_ : Fin BD) j => decide ((j.val : Int) + start < (context : Int)))) := by
  simp [RealKey.mask, evalOp.eq_def, hs, hn, bind, Option.bind]
  rfl

theorem load_eval {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) :
    evalOp (.load .real (.ptr (RealKey.pointer BN BD stride second))
      (if even then .none else .mask (RealKey.mask BN BD context))) s =
      some (matrix (loaded s ptrs start stride context even second)) := by
  have hp' := pointer_eval s ptrs start stride second hp hs
  have hm := mask_eval (BD := BD) s start context hs hn
  cases even
  all_goals
    rw [evalOp.eq_def]
    simp [hp', hm, bind, Option.bind]
    apply Tile.ext
    rintro ⟨d, j, ⟨⟩⟩
    by_cases h : (j.val : Int) + start < (context : Int) <;>
      simp [matrix, finite, loaded, active, h]

/-- This is a memory-derived key tile, including the actual mask behavior;
there is no premise about an already loaded k register. -/
theorem executes {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) :
    stepStmts (RealKey.code BN BD stride context even second) s =
      some (s.setReg "k" .real [BD, BN] (matrix (loaded s ptrs start stride context even second))) := by
  have he := load_eval s ptrs start stride context even second hp hs hn
  have hc (b : Bool) : evalOp (.constBool b) s = some (Tile.scalar b) := by rw [evalOp.eq_def]
  cases even <;> simp only [Bool.false_eq_true, ↓reduceIte] at he
  all_goals simp [RealKey.code, stepStmts, stepStmt, hc, he, bind, Option.bind]

@[simp] theorem loaded_setReg {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context : Nat) (even second : Bool)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    loaded (s.setReg name dtype shape v) ptrs start stride context even second =
      loaded s ptrs start stride context even second := by
  funext d j
  simp [loaded]

theorem pointer_safe (BN BD stride : Nat) (second : Bool) (bounds : RegionBounds) (s : BlockState) :
    (RealKey.pointer BN BD stride second).SafeAt bounds s := by
  cases second <;> simp [RealKey.pointer, Op.SafeAt.eq_def]

theorem pointer_memory_safe (BN BD stride : Nat) (second : Bool) (bounds : RegionBounds) :
    (RealKey.pointer BN BD stride second).MemorySafe bounds := by
  cases second <;> simp [RealKey.pointer, Op.MemorySafe]

theorem mask_safe (BN BD context : Nat) (bounds : RegionBounds) (s : BlockState) :
    (RealKey.mask BN BD context).SafeAt bounds s := by
  simp [RealKey.mask, Op.SafeAt.eq_def]

theorem mask_active {BN BD : Nat} (s : BlockState) (start : Int) (context : Nat) (even : Bool)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) (d : Fin BD) (j : Fin BN) :
    (if even then MaskOpt.none (dtype := .real) else .mask (RealKey.mask BN BD context)).Active
      s (d, j, PUnit.unit) ↔ active start context even j := by
  cases even <;> simp [MaskOpt.Active, mask_eval s start context hs hn, active]

def Bounds {BN BD : Nat} (bounds : RegionBounds) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context : Nat) (even second : Bool) : Prop :=
  ∀ d j, active start context even j →
    (address ptrs start stride second d j).2 < bounds (address ptrs start stride second d j).1

theorem load_safe {BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BD, BN]) (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hb : Bounds bounds ptrs start stride context even second) :
    (Op.load .real (.ptr (RealKey.pointer BN BD stride second))
      (if even then .none else .mask (RealKey.mask BN BD context))).SafeAt bounds s := by
  have haddr : (MemAccess.ptr (d := .real) (RealKey.pointer BN BD stride second)).ActiveAddressSafe
      bounds s ((if even then MaskOpt.none (dtype := .real) else .mask (RealKey.mask BN BD context)).Active s) := by
    refine ⟨pointer_memory_safe BN BD stride second bounds, ?_⟩
    intro tile ht
    rw [pointer_eval s ptrs start stride second hp hs] at ht
    cases ht
    rintro ⟨d, j, ⟨⟩⟩ ha
    exact hb d j ((mask_active s start context even hs hn d j).mp ha)
  rw [Op.SafeAt.eq_def]
  refine ⟨pointer_safe BN BD stride second bounds s, ?_, haddr⟩
  cases even <;> simp [mask_safe]

theorem code_safe {BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BD, BN]) (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hb : Bounds bounds ptrs start stride context even second) :
    Stmt.TraceSafeList bounds (RealKey.code BN BD stride context even second) s := by
  have hl := load_safe bounds s ptrs start stride context even second hp hs hn hb
  unfold RealKey.code
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  simp only [evalOp.eq_def, Op.SafeAt.eq_def, Tile.scalar_data, true_and]
  cases even <;> simp only [Bool.false_eq_true, ↓reduceIte] at hl ⊢
  all_goals
    exact Stmt.TraceSafeList.cons_intro (by simpa only [Stmt.TraceSafe] using hl) (fun _ _ => Stmt.TraceSafeList.nil_intro)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealKey
