import bench.tritonbench_g.bmm_optimized.Proofs.RealStore
import VeriTile.Meta.Disprove

/-! Omitted `other` is observable on an actual 32×32 dot tile. With K=33,
all input elements one and unspecified masked loads five, the full source
stores 808 instead of 33. This is a real-semantics counterexample, not a GPU
measurement or a claim about two-gates acceptance. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.UndefinedTail
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

-- These sizes and divisibility flags match a supported host configuration.
def p : RealProgram.Parameters := ⟨"A","B","O",32,32,33,1,Bool.true,Bool.true,Bool.false⟩
def initial : BlockState where
  mem := fun r _ => MemCell.real (if r = "O" then 9 else 1)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 5

def start := RealSetup.result initial 32 32 32 p
abbrev rows : Fin 32 → Nat := Fin.val
abbrev cols : Fin 32 → Nat := Fin.val
def ap (c : Nat) := RealLoop.aPointers initial 32 32 p c
def bp (c : Nat) := RealLoop.bPointers initial 32 32 p c
def ks (c : Nat) : Fin 32 → Nat := fun k => c * 32 + k.val

def entry0 := start.setReg "_i" .nat [] (Tile.scalar 0)
def first := RealStep.result entry0 p (ap 0) (bp 0) rows cols (ks 0) (fun _ _ => 0)
def entry1 := first.setReg "_i" .nat [] (Tile.scalar 1)
def second := RealStep.result entry1 p (ap 1) (bp 1) rows cols (ks 1) (fun _ _ => 32)

theorem first_executes (swapped : Bool) :
    stepStmts (RealLoads.body swapped 32 32 32 p) entry0 = some first := by
  apply RealStep.executes swapped 32 32 32 entry0 p (ap 0) (bp 0) rows cols (ks 0) (fun _ _ => 0)
  · simp [p]
  · simp [p]
  · simp [entry0, start, RealSetup.result, ks, p]
    apply Tile.ext
    rintro ⟨k,⟨⟩⟩
    simp [Tile.vec, ks]
  · simp [entry0, start, RealSetup.result, ap, RealLoop.aPointers, RealSetup.aPointers]
  · simp [entry0, start, RealSetup.result, bp, RealLoop.bPointers, RealSetup.bPointers]
  · simp [entry0, start, RealSetup.result]

theorem first_dot (i j : Fin 32) :
    RealDot.value entry0 p (ap 0) (bp 0) rows cols (ks 0) i j = 32 := by
  have hk : ∀ k : Fin 32, k.val < 33 := fun k => by omega
  simp [RealDot.value, RealLoads.aElement, RealLoads.bElement, RealMath.active,
    entry0, start, RealSetup.result, RealSetup.gridResult, initial, p, ap, bp,
    RealLoop.aPointers, RealLoop.bPointers, Tile.mat, ks, BlockState.readMem, hk]

theorem first_carry : first.regs .real [32,32] "o" = some (Tile.mat fun _ _ => some 32) := by
  simp [first, RealStep.result, first_dot]

theorem second_executes (swapped : Bool) :
    stepStmts (RealLoads.body swapped 32 32 32 p) entry1 = some second := by
  apply RealStep.executes swapped 32 32 32 entry1 p (ap 1) (bp 1) rows cols (ks 1) (fun _ _ => 32)
  · simp [p]
  · simp [p]
  · simp only [entry1, first, RealStep.result, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, BlockState.setReg_same]
    simp [ks, Nat.add_comm]
    apply Tile.ext
    rintro ⟨k,⟨⟩⟩
    simp [Tile.vec, ks, Nat.add_comm]
  · simp [entry1, first, RealStep.result, ap, RealLoop.a_advance]
  · simp only [entry1, first, RealStep.result, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, BlockState.setReg_same]
    exact congrArg some (RealLoop.b_advance initial 32 32 p 0)
  · simpa [entry1] using first_carry

theorem second_dot (i j : Fin 32) :
    RealDot.value entry1 p (ap 1) (bp 1) rows cols (ks 1) i j = 776 := by
  have hf := congrArg BlockState.mem (RealStep.state_frame entry0 p (ap 0) (bp 0) rows cols (ks 0) (fun _ _ => 0))
  have hu := congrArg BlockState.undef (RealStep.state_frame entry0 p (ap 0) (bp 0) rows cols (ks 0) (fun _ _ => 0))
  change first.mem = entry0.mem at hf
  change first.undef = entry0.undef at hu
  simp only [RealDot.value, RealLoads.aElement, RealLoads.bElement, entry1,
    BlockState.setReg, hf, hu, BlockState.readMem, ap, bp, RealLoop.aPointers,
    RealLoop.bPointers, Tile.mat, p, RealMath.active, Bool.false_eq_true,
    Bool.true_eq_false, false_or, true_or, true_and, and_true, ks, Nat.one_mul]
  norm_num [entry0, start, RealSetup.result, RealSetup.gridResult, initial, p,
    Fin.sum_univ_succ]
  simp only [show ("A" : String) ≠ "O" by decide, show ("B" : String) ≠ "O" by decide, ↓reduceIte]
  norm_num

theorem second_carry : second.regs .real [32,32] "o" = some (Tile.mat fun _ _ => some 808) := by
  norm_num [second, RealStep.result, second_dot]

theorem loop_executes (swapped : Bool) :
    stepStmt (RealLoop.statement swapped 32 32 32 p) start = some second := by
  have h0 := first_executes swapped
  have h1 := second_executes swapped
  change stepStmts (RealLoads.body swapped 32 32 32 p) (start.setReg "_i" .nat [] (Tile.scalar 0)) = some first at h0
  change stepStmts (RealLoads.body swapped 32 32 32 p) (first.setReg "_i" .nat [] (Tile.scalar 1)) = some second at h1
  simp only [RealLoop.statement, stepForRangeAux.forRangeDyn_unfold,
    evalOp_ref, evalOp_constNat, start, RealSetup.result, BlockState.setReg_ne_name,
    ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_same, p,
    BmmChunkBwd.RealMath.blocks, bind, Option.bind_some, Tile.scalar]
  change stepForRangeAux "_i" 0 2 1 (RealLoads.body swapped 32 32 32 p) start = some second
  rw [stepForRangeAux]
  simp only [Nat.reduceLT, ↓reduceIte, h0, bind, Option.bind_some]
  rw [stepForRangeAux]
  simp only [Nat.reduceAdd, Nat.reduceLT, ↓reduceIte, h1, bind, Option.bind_some]
  rw [stepForRangeAux]
  rfl

def final := RealStore.write second p 0 rows cols (fun _ _ => 808)

theorem store_executes : stepStmts (RealStore.code 32 32 p) second = some final := by
  have hp : second.regs .ptr [32,32] "o_ptrs" =
      some (Tile.mat fun i j => (p.O, 0 + rows i * p.N + cols j)) := by
    simp [second, entry1, first, RealStep.result, RealLoads.result, RealMasks.result,
      RealMasks.aResult, RealMasks.bResult, entry0, start, RealSetup.result, RealSetup.oPointers,
      RealSetup.oBase, RealSetup.rows, RealSetup.columns, RealSetup.tileM, RealSetup.tileN,
      RealGrid.rowTile, RealGrid.columnTile, p, initial]
  simpa [final, RealStore.prepared, p] using RealStore.executes second 32 32 p 0 rows cols
    (fun _ _ => 808) (by simp [p]) (by simp [p]) hp second_carry

/-- Actual complete source execution, including prefix, both K blocks and store. -/
theorem kernel_executes (swapped : Bool) :
    exec (RealProgram.kernel swapped 32 32 32 p).toAlgKernel initial = some final := by
  rw [exec, RealProgram.body_split, RealSetup.original_candidate_setup,
    List.append_assoc, stepStmts.append_some (RealSetup.executes initial 32 32 32 p),
    RealLoads.matches_source, RealStore.matches_source]
  change stepStmts ([RealLoop.statement swapped 32 32 32 p] ++ RealStore.code 32 32 p) start = some final
  rw [List.singleton_append, stepStmts.cons_some (loop_executes swapped)]
  exact store_executes

theorem stored_value : final.mem "O" 0 = MemCell.real 808 := by
  have hinj : ∀ a b : TileIndex [32,32], RealStore.active p rows cols a → RealStore.active p rows cols b →
      RealStore.offset 0 p.N rows cols a = RealStore.offset 0 p.N rows cols b → a = b := by
    rintro ⟨i,j,⟨⟩⟩ ⟨i',j',⟨⟩⟩ _ _ he
    simp only [RealStore.offset, p, rows, cols, Nat.zero_add] at he
    have hi : i = i' := Fin.ext (by omega)
    have hj : j = j' := Fin.ext (by omega)
    subst i'
    subst j'
    rfl
  exact RealStore.readback second p 0 rows cols (fun _ _ => 808) (0,0,PUnit.unit)
    (by simp [RealStore.active, RealMath.active, p]) hinj

theorem mathematical_value :
    RealMath.contraction p.K (RealLoop.aEntry initial 32 p 0) (RealLoop.bEntry initial 32 p 0) = 33 := by
  norm_num [RealMath.contraction, RealLoop.aEntry, RealLoop.bEntry, BlockState.readMem, initial, p]
  simp only [show ("A" : String) ≠ "O" by decide, show ("B" : String) ≠ "O" by decide, ↓reduceIte]

/-- Removing the unspecified-load-value condition makes the real product
claim false even with valid host tile sizes and divisible M/N. -/
refutation product_without_zero_fill (swapped : Bool) :
    ∀ s t : BlockState,
      exec (RealProgram.kernel swapped 32 32 32 p).toAlgKernel s = some t →
      t.mem "O" 0 = MemCell.real
        (RealMath.contraction p.K (RealLoop.aEntry s 32 p 0) (RealLoop.bEntry s 32 p 0)) := by
  intro h
  have he := h initial final (kernel_executes swapped)
  rw [stored_value, mathematical_value] at he
  have hne : MemCell.real (808 : ℝ) ≠ MemCell.real 33 := by
    intro hh
    have hr := congrArg (fun cell : MemCell => cell.readAs .real) hh
    have heq : (808 : ℝ) = 33 := Option.some.inj (Option.some.inj hr)
    norm_num at heq
  exact hne he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.UndefinedTail
