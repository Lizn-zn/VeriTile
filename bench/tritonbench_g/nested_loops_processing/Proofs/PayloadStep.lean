import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadAlgorithm

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton TypedSource FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

abbrev Payload (dtype : TileDType) := TileIndex [2,2] → TileCarrier dtype

noncomputable def loaded (dtype : TileDType) (s : BlockState) (ptr : PointerTile) : Payload dtype :=
  fun lane ↦ s.readMemValue dtype (ptr lane).1 (ptr lane).2

noncomputable def scatter (dtype : TileDType) (s : BlockState)
    (ptr : PointerTile) (value : Payload dtype) : BlockState :=
  (TileShape.allIndices [2,2]).foldl (fun st lane ↦
    st.writeMemTyped dtype (ptr lane).1 (ptr lane).2 (value lane)) s

noncomputable def innerState (dtype : TileDType) (s : BlockState) (a c : PointerTile)
    (v1 v2 : Payload dtype) (delta : Nat) : BlockState :=
  let a' := bump a delta
  let v3 := loaded dtype s a'
  let t := s.setReg "a_ptrs" .ptr [2,2] ⟨a'⟩
  let t := t.setReg "a3" dtype [2,2] ⟨v3⟩
  let t := scatter dtype t c v1
  let t := t.setReg "c_ptrs" .ptr [2,2] ⟨bump c delta⟩
  let t := scatter dtype t (bump c delta) v2
  let t := t.setReg "c_ptrs" .ptr [2,2] ⟨bump (bump c delta) delta⟩
  let t := scatter dtype t (bump (bump c delta) delta) v3
  t.setReg "c_ptrs" .ptr [2,2] ⟨bump (bump (bump c delta) delta) delta⟩

theorem inner_step (dtype : TileDType) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 v2 : Payload dtype)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩)
    (h2 : s.regs dtype [2,2] "a2" = some ⟨v2⟩) :
    stepStmts (PayloadAlgorithm.inner dtype src dst sm sn) s =
      some (innerState dtype s a c v1 v2 (2 * sn).toNat) := by
  simp [PayloadAlgorithm.inner, PayloadAlgorithm.inputAdvance, PayloadAlgorithm.outputAdvance,
    PayloadAlgorithm.load, PayloadAlgorithm.store, FullReal.inner, FullReal.middle, FullReal.outer,
    FullReal.source, kernel, ComputeKernel.surfaceBody,
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

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
