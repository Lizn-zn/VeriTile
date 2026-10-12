import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroBackward
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroComplexBackward
open VeriTile.Triton TypedSource
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def window (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (region : RegionName) (component o : Nat) : Prop :=
  ∃ j : Fin block, u.pids 0*block+j.val < batch*dim ∧
    o = A.addr region ((u.pids 0*block+j.val)*2+component)

def written (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (gs gl r : RegionName) (o : Nat) : Prop :=
  r = A.flat ∧ (window A u batch dim block gs 0 o ∨ window A u batch dim block gs 1 o ∨
    window A u batch dim block gl 0 o ∨ window A u batch dim block gl 1 o)

/-- The empty complex reverse recurrence writes both components of both
terminal gradients as zero. Input/output aliasing does not change this
conclusion, since every read precedes these constant writes. -/
theorem complex_backward_memory (swapped : Bool) (A : FlatAlloc)
    (s l y gs gx gl gy : RegionName) (batch dim block : Nat) (u : BlockState) :
    ∃ v, exec (A.flattenKernel
      (TypedSource.complexBackward swapped s l y gs gx gl gy 0 batch dim block).eraseDType.toAlgKernel) u = some v ∧
      ∀ r o, v.mem r o = if written A u batch dim block gs gl r o then MemCell.real 0 else u.mem r o := by
  cases swapped
  case «true» => simpa only [← Candidate.complexBackward_flat_exec] using
    complex_backward_memory Bool.false A s l y gs gx gl gy batch dim block u
  case «false» =>
    simp only [exec,FlatAlloc.flattenKernel,Erasure.complexBackward_body]
    refine ⟨?_, ?_, ?_⟩
    rotate_left
    · simp only [Erasure.complexBackward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,ite_false,Bool.false_eq_true,↓reduceIte]
      simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
        stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
        Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
        String.reduceEq,FlatAlloc.shiftBroadcast,
        Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
        IntegralDType.mod,ComparableDType.lt,stepForLoopAux.forLoop_unfold,
        stepForLoopAux.step_ge,Region.cast,BlockState.writeMemTyped_real,BlockState.foldl_writeMem_masked_regs,
        BlockState.setReg_same,
        BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
        BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
        Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl]
      rfl
    · intro r o
      norm_num only [NumericDType.sub,WithBot.realSub_coe_coe,sub_self,FloatDType.storeValue,FloatDType.toWithBot,WithBot.some_eq_coe,WithBot.unbotD_coe]
      simp only [ZeroScatter.memory,ZeroScatter.memory_prop,TileShape.mem_allIndices [block],true_and,
        BlockState.setReg_mem,decide_eq_true_eq,Prod.exists,exists_const]
      by_cases hl1 : r = A.flat ∧ window A u batch dim block gl 1 o
      · have hc := hl1
        dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
        rw [if_pos hc,if_pos (show written A u batch dim block gs gl r o from ⟨hl1.1,Or.inr (Or.inr (Or.inr hl1.2))⟩)]
      · have hc := hl1
        dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
        rw [if_neg hc]
        by_cases hl0 : r = A.flat ∧ window A u batch dim block gl 0 o
        · have hc := hl0
          dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
          rw [if_pos hc,if_pos (show written A u batch dim block gs gl r o from ⟨hl0.1,Or.inr (Or.inr (Or.inl hl0.2))⟩)]
        · have hc := hl0
          dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
          rw [if_neg hc]
          by_cases hs1 : r = A.flat ∧ window A u batch dim block gs 1 o
          · have hc := hs1
            dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
            rw [if_pos hc,if_pos (show written A u batch dim block gs gl r o from ⟨hs1.1,Or.inr (Or.inl hs1.2)⟩)]
          · have hc := hs1
            dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
            rw [if_neg hc]
            by_cases hs0 : r = A.flat ∧ window A u batch dim block gs 0 o
            · have hc := hs0
              dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
              rw [if_pos hc,if_pos (show written A u batch dim block gs gl r o from ⟨hs0.1,Or.inl hs0.2⟩)]
            · have hc := hs0
              dsimp only [window,FlatAlloc.addr,Nat.add_zero] at hc
              rw [if_neg hc]
              rw [if_neg (show ¬ written A u batch dim block gs gl r o from by
                rintro ⟨hr,h | h | h | h⟩
                · exact hs0 ⟨hr,h⟩
                · exact hs1 ⟨hr,h⟩
                · exact hl0 ⟨hr,h⟩
                · exact hl1 ⟨hr,h⟩
              )]
termination_by if swapped then 1 else 0

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroComplexBackward
