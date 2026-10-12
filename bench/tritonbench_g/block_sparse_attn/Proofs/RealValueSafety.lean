import bench.tritonbench_g.block_sparse_attn.Proofs.RealValueLoad
import VeriTile.Triton.Memory.TraceProofs

/-! Access safety for the actual V loads. Only active lanes need bounds;
the second channel uses its distinct address, including the BLOCK_D offset.
These local bounds will be derived from the complete kernel's input layout. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueSafety
open VeriTile.Triton RealValueLoad

theorem pointer_safe (BN BD stride : Nat) (second : Bool) (bounds : RegionBounds) (s : BlockState) :
    (RealValueSource.pointer BN BD stride second).SafeAt bounds s := by
  cases second <;> simp [RealValueSource.pointer, Op.SafeAt.eq_def]

theorem pointer_memory_safe (BN BD stride : Nat) (second : Bool) (bounds : RegionBounds) :
    (RealValueSource.pointer BN BD stride second).MemorySafe bounds := by
  cases second <;> simp [RealValueSource.pointer, Op.MemorySafe]

theorem mask_safe (BN BD context : Nat) (bounds : RegionBounds) (s : BlockState) :
    (RealValueSource.mask BN BD context).SafeAt bounds s := by
  simp [RealValueSource.mask, Op.SafeAt.eq_def]

theorem mask_active {BN BD : Nat} (s : BlockState) (start : Int) (context : Nat) (even : Bool)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) (j : Fin BN) (d : Fin BD) :
    (if even then MaskOpt.none (dtype := .real) else .mask (RealValueSource.mask BN BD context)).Active
      s (j, d, PUnit.unit) ↔ active start context even j := by
  cases even <;> simp [MaskOpt.Active, mask_eval s start context hs hn, active]

def Bounds {BN BD : Nat} (bounds : RegionBounds) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context : Nat) (even second : Bool) : Prop :=
  ∀ j d, active start context even j →
    (address ptrs start stride second j d).2 < bounds (address ptrs start stride second j d).1

theorem load_safe {BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hb : Bounds bounds ptrs start stride context even second) :
    (Op.load .real (.ptr (RealValueSource.pointer BN BD stride second))
      (if even then .none else .mask (RealValueSource.mask BN BD context))).SafeAt bounds s := by
  have haddr : (MemAccess.ptr (d := .real) (RealValueSource.pointer BN BD stride second)).ActiveAddressSafe
      bounds s ((if even then MaskOpt.none (dtype := .real) else .mask (RealValueSource.mask BN BD context)).Active s) := by
    refine ⟨pointer_memory_safe BN BD stride second bounds, ?_⟩
    intro tile ht
    rw [pointer_eval s ptrs start stride second hp hs] at ht
    cases ht
    rintro ⟨j, d, ⟨⟩⟩ ha
    exact hb j d ((mask_active s start context even hs hn j d).mp ha)
  rw [Op.SafeAt.eq_def]
  refine ⟨pointer_safe BN BD stride second bounds s, ?_, haddr⟩
  cases even <;> simp [mask_safe]

theorem code_safe {BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hb : Bounds bounds ptrs start stride context even second) :
    Stmt.TraceSafeList bounds (RealValueSource.code BN BD stride context even second) s := by
  have hl := load_safe bounds s ptrs start stride context even second hp hs hn hb
  unfold RealValueSource.code
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  simp only [evalOp.eq_def, Op.SafeAt.eq_def, Tile.scalar_data, true_and]
  cases even <;> simp only [Bool.false_eq_true, ↓reduceIte] at hl ⊢
  all_goals
    exact Stmt.TraceSafeList.cons_intro (by simpa only [Stmt.TraceSafe] using hl) (fun _ _ => Stmt.TraceSafeList.nil_intro)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueSafety
