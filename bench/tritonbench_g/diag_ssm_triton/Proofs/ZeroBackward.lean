import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroTime
import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroScatter
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroBackward
open VeriTile.Triton TypedSource
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 600000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def window (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (region : RegionName) (o : Nat) : Prop :=
  ∃ j : Fin block, u.pids 0 * block + j.val < batch * dim ∧
    o = A.addr region (u.pids 0 * block + j.val)

def written (A : FlatAlloc) (u : BlockState) (batch dim block : Nat)
    (gs gl r : RegionName) (o : Nat) : Prop :=
  r = A.flat ∧ (window A u batch dim block gs o ∨ window A u batch dim block gl o)

/-- At T=0, reverse recurrence reads the transition tile, writes zero to
the two terminal gradients, and leaves every other tagged cell unchanged.
No separation between gradient outputs is needed for these zero writes. -/
theorem backward_memory (storage : Storage) (swapped : Bool) (A : FlatAlloc)
    (s l y gs gx gl gy : RegionName) (batch dim block : Nat) (u : BlockState) :
    ∃ v, exec (A.flattenKernel
      (TypedSource.backward storage swapped s l y gs gx gl gy 0 batch dim block).eraseDType.toAlgKernel) u = some v ∧
      ∀ r o, v.mem r o = if written A u batch dim block gs gl r o then MemCell.real 0 else u.mem r o := by
  cases swapped
  case «true» => simpa only [← Candidate.backward_flat_exec] using
    backward_memory storage Bool.false A s l y gs gx gl gy batch dim block u
  case «false» =>
    simp only [exec,FlatAlloc.flattenKernel,Erasure.backward_body]
    refine ⟨?_, ?_, ?_⟩
    rotate_left
    · simp only [Erasure.backward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
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
      norm_num only [FloatDType.storeValue,FloatDType.toWithBot,WithBot.some_eq_coe,WithBot.unbotD_coe]
      simp only [ZeroScatter.memory,ZeroScatter.memory_prop,TileShape.mem_allIndices [block],true_and,
        BlockState.setReg_mem,decide_eq_true_eq,Prod.exists,exists_const]
      by_cases hl : r = A.flat ∧ window A u batch dim block gl o
      · have hl' := hl
        dsimp only [window,FlatAlloc.addr] at hl'
        rw [if_pos hl',if_pos (show written A u batch dim block gs gl r o from ⟨hl.1,Or.inr hl.2⟩)]
      · have hl' := hl
        dsimp only [window,FlatAlloc.addr] at hl'
        rw [if_neg hl']
        by_cases hs : r = A.flat ∧ window A u batch dim block gs o
        · have hs' := hs
          dsimp only [window,FlatAlloc.addr] at hs'
          rw [if_pos hs',if_pos (show written A u batch dim block gs gl r o from ⟨hs.1,Or.inl hs.2⟩)]
        · have hs' := hs
          dsimp only [window,FlatAlloc.addr] at hs'
          rw [if_neg hs',if_neg (show ¬ written A u batch dim block gs gl r o from by
            rintro ⟨hr,h | h⟩
            · exact hs ⟨hr,h⟩
            · exact hl ⟨hr,h⟩)]

termination_by if swapped then 1 else 0

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroBackward
