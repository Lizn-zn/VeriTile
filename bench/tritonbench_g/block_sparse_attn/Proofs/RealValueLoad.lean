import bench.tritonbench_g.block_sparse_attn.Proofs.RealValueSource
import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationValues
import VeriTile.Triton.Semantics.Step

/-! Execute each actual V load from the current memory and pointer registers.
Inactive lanes retain the state's arbitrary undef interpretation. Subsequent
arithmetic must show their coefficient is zero; no zero-padding assumption is
introduced here. The prologue still has to establish the base pointer tile. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueLoad
open VeriTile.Triton RealNormalizationValues Classical
noncomputable section

def address {BN BD : Nat} (ptrs : Tile .ptr [BN, BD]) (start : Int)
    (stride : Nat) (second : Bool) (j : Fin BN) (d : Fin BD) : RegionName × Nat :=
  let p := ptrs.data (j, d, PUnit.unit)
  (p.1, p.2 + (start * (stride : Int)).toNat + if second then BD else 0)

def active (start : Int) (context : Nat) (even : Bool) {BN : Nat} (j : Fin BN) : Prop :=
  even = Bool.true ∨ (j.val : Int) + start < (context : Int)

def loaded {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD]) (start : Int)
    (stride context : Nat) (even second : Bool) (j : Fin BN) (d : Fin BD) : ℝ :=
  let a := address ptrs start stride second j d
  if active start context even j then s.readMem a.1 a.2 else s.undef a.1 a.2

theorem pointer_eval {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride : Nat) (second : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start)) :
    evalOp (RealValueSource.pointer BN BD stride second) s =
      some (Tile.mat (address ptrs start stride second)) := by
  cases second <;>
    simp [RealValueSource.pointer, evalOp.eq_def, hp, hs, bind, Option.bind]
  all_goals rfl

theorem mask_eval {BN BD : Nat} (s : BlockState) (start : Int) (context : Nat)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) :
    evalOp (RealValueSource.mask BN BD context) s =
      some (Tile.mat (fun j (_ : Fin BD) => decide ((j.val : Int) + start < (context : Int)))) := by
  simp [RealValueSource.mask, evalOp.eq_def, hs, hn, bind, Option.bind]
  rfl

theorem load_eval {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) :
    evalOp (.load .real (.ptr (RealValueSource.pointer BN BD stride second))
      (if even then .none else .mask (RealValueSource.mask BN BD context))) s =
      some (matrix (loaded s ptrs start stride context even second)) := by
  have hp' := pointer_eval s ptrs start stride second hp hs
  have hm := mask_eval (BD := BD) s start context hs hn
  cases even
  all_goals
    rw [evalOp.eq_def]
    simp [hp', hm, bind, Option.bind]
    apply Tile.ext
    rintro ⟨j, d, ⟨⟩⟩
    by_cases h : (j.val : Int) + start < (context : Int) <;>
      simp [matrix, finite, loaded, active, h]

/-- This is a memory-derived value tile, including the actual mask behavior;
there is no premise about an already loaded v register. -/
theorem executes {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) :
    stepStmts (RealValueSource.code BN BD stride context even second) s =
      some (s.setReg "v" .real [BN, BD] (matrix (loaded s ptrs start stride context even second))) := by
  have he := load_eval s ptrs start stride context even second hp hs hn
  have hc (b : Bool) : evalOp (.constBool b) s = some (Tile.scalar b) := by rw [evalOp.eq_def]
  cases even <;> simp only [Bool.false_eq_true, ↓reduceIte] at he
  all_goals simp [RealValueSource.code, stepStmts, stepStmt, hc, he, bind, Option.bind]

@[simp] theorem loaded_setReg {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context : Nat) (even second : Bool)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    loaded (s.setReg name dtype shape v) ptrs start stride context even second =
      loaded s ptrs start stride context even second := by
  funext j d
  simp [loaded]

/-- The memory-derived load is irrelevant at any zero-probability lane,
including when its unspecified value differs from an independent input array. -/
theorem weighted_eq {BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context : Nat) (even second : Bool)
    (p : Fin BN → ℝ) (v : Fin BN → Fin BD → ℝ)
    (hp : ∀ j, ¬ active start context even j → p j = 0)
    (hv : ∀ j d, active start context even j →
      s.readMem (address ptrs start stride second j d).1 (address ptrs start stride second j d).2 = v j d)
    (d : Fin BD) :
    (∑ j : Fin BN, p j * loaded s ptrs start stride context even second j d) =
      ∑ j : Fin BN, p j * v j d := by
  apply Finset.sum_congr rfl
  intro j _
  by_cases h : active start context even j
  · simp [loaded, h, hv j d h]
  · simp [loaded, h, hp j h]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueLoad
