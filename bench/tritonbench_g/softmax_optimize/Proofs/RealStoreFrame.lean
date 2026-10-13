import bench.tritonbench_g.softmax_optimize.Proofs.LegacyReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

theorem masked_scatter_raw_frame {α : Type} (l : List α) (s : BlockState)
    (Y r : RegionName) (offset : α → Nat) (value : α → ℝ) (mask : α → Bool)
    (o : Nat) (h : ∀ i ∈ l, mask i → ¬(Y=r ∧ offset i=o)) :
    (l.foldl (fun t i ↦ if mask i then t.writeMem Y (offset i) (value i) else t) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons a rest ih =>
    rw [List.foldl_cons, ih _ (fun i hi ↦ h i (List.mem_cons_of_mem a hi))]
    by_cases ha : mask a
    · rw [if_pos ha, BlockState.writeMem_mem]
      apply if_neg
      intro he
      exact h a List.mem_cons_self ha ⟨he.1.symm,he.2.symm⟩
    · rw [if_neg ha]

theorem loopApass2_raw_frame (N TILE_N : Nat) (output_ptr input_ptr : RegionName)
    (hne : output_ptr ≠ input_ptr) (c : Nat) (cc S : ℝ) (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)))
    (hm : s.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ)))
    (hz : s.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) :
    ∃ t, stepStmts (loopApass2body output_ptr input_ptr N TILE_N)
        (s.setReg "start_n" .nat [] (Tile.scalar (c*TILE_N))) = some t ∧
      ∀ r o, (∀ i : TileIndex [TILE_N], ¬(output_ptr=r ∧ pass2Off N TILE_N c s i=o)) →
        t.mem r o=s.mem r o := by
  simp only [loopApass2body]
  -- n_offsets = c*TILE_N + arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val)) (by
      rw [evalOp_add, evalOp_ref_setReg_same, evalOp_arange]; rfl))]
  -- offset = pid*N + n_offsets
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val))) (by
      rw [evalOp_add, evalOp_mul, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same, hpid, ne_eq,
        String.reduceEq, not_false_eq_true]
      rfl))]
  -- input_ptrs = ptrBase input_ptr + offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast input_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- inp = load(input_ptrs) [unmasked]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ))) (by
      rw [evalOp_load_ptrRef_none "input_ptrs" _
        (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))
        (by simp only [evalOp_ref, BlockState.setReg_same])]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.mk]
      rw [BlockState.readMemValue_real]
      rfl))]
  -- e = exp(inp - m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realExp (WithBot.realSub
        (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ)
        ((cc : ℝ) : WithBot ℝ)))) (by
      rw [evalOp_exp, evalOp_sub]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm]
      rfl))]
  -- out = e / z
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realDiv
        (WithBot.realExp (WithBot.realSub
          (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ)
          ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) (by
      rw [evalOp_div]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hz]
      rfl))]
  -- output_ptrs = ptrBase output_ptr + offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast output_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- the pre-store state (after the seven assigns)
  set s7 := ((((((( s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N)) ).setReg "n_offsets" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val))).setReg "offset" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val)))).setReg "input_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))).setReg "inp" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ)))).setReg "e" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realExp (WithBot.realSub
        (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ))))).setReg "out" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub
        (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)))).setReg "output_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) with hs7def
  -- evaluate the store
  have houtEval : evalOp (Op.ref .real [TILE_N] "out") s7
      = some (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub
          (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
          (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) := by
    rw [evalOp_ref, hs7def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  have hptrEval : evalOp (Op.ref .ptr [TILE_N] "output_ptrs") s7
      = some (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) := by
    rw [evalOp_ref, hs7def]
    simp only [BlockState.setReg_same]
  -- the post-store state is the ptr scatter foldl
  have hstore : stepStmt (Stmt.store .real [TILE_N] (MemAccess.ptr (Op.ref .ptr [TILE_N] "output_ptrs"))
      (Op.ref .real [TILE_N] "out") MaskOpt.none) s7
      = some ((TileShape.allIndices [TILE_N]).foldl
          (fun acc i => acc.writeMemTyped .real (Region.cast output_ptr)
            (s.pid * N + (c * TILE_N + i.1.val))
            (WithBot.realDiv (WithBot.realExp (WithBot.realSub
              (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + i.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
              (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) s7) := by
    unfold stepStmt
    rw [houtEval]
    simp only [Option.bind_some, Option.bind_eq_bind, hptrEval, if_true]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  -- rewrite the typed store foldl into the plain `writeMem` library form
  set ofn : TileIndex [TILE_N] → Nat := fun i => s.pid * N + (c * TILE_N + i.1.val) with hofn
  set vfn : TileIndex [TILE_N] → ℝ := fun i => pass2Val N TILE_N c cc S s input_ptr i with hvfn
  have hfold : (TileShape.allIndices [TILE_N]).foldl
        (fun acc i => acc.writeMemTyped .real (Region.cast output_ptr)
          (s.pid * N + (c * TILE_N + i.1.val))
          (WithBot.realDiv (WithBot.realExp (WithBot.realSub
            (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + i.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
            (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) s7
      = (TileShape.allIndices [TILE_N]).foldl
          (fun acc i => acc.writeMem output_ptr (ofn i) (vfn i)) s7 := by
    refine List.foldl_ext _ _ s7 (fun acc i _ => ?_)
    rw [Region.cast_id, BlockState.writeMemTyped_real]
    congr 1
    show FloatDType.real.storeValue _ = vfn i
    rw [FloatDType.real_storeValue]
    simp only [hvfn, pass2Val]
    exact pass2_store_value _ cc S
  rw [hfold]
  set sF := (TileShape.allIndices [TILE_N]).foldl
      (fun acc i => acc.writeMem output_ptr (ofn i) (vfn i)) s7 with hsF
  refine ⟨sF,rfl,?_⟩
  intro r o h
  have hf := masked_scatter_raw_frame (TileShape.allIndices [TILE_N]) s7
    output_ptr r ofn vfn (fun _ ↦ Bool.true) o
    (fun i _ _ ↦ by simpa [hofn,pass2Off] using h i)
  simpa only [hsF, if_true, hs7def] using hf

theorem loopBpass2_raw_frame (N TILE_N : Nat) (output_ptr input_ptr : RegionName)
    (hne : output_ptr ≠ input_ptr) (c : Nat) (cc S : ℝ) (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)))
    (hm : s.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ)))
    (hz : s.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) :
    ∃ t, stepStmts (loopBpass2body output_ptr input_ptr N TILE_N)
        (s.setReg "start_n" .nat [] (Tile.scalar (c*TILE_N))) = some t ∧
      ∀ r o, (∀ i : TileIndex [TILE_N], c*TILE_N+i.1.val<N → ¬(output_ptr=r ∧ pass2Off N TILE_N c s i=o)) →
        t.mem r o=s.mem r o := by
  simp only [loopBpass2body]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val)) (by
      rw [evalOp_add, evalOp_ref_setReg_same, evalOp_arange]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val))) (by
      rw [evalOp_add, evalOp_mul, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same, hpid, ne_eq,
        String.reduceEq, not_false_eq_true]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast input_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- mask = n_offsets < N
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N))) (by
      rw [evalOp_lt, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      rfl))]
  -- masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => loopBload N TILE_N c s input_ptr idx)) (by
      rw [evalOp_load_ptrRef_maskOther "input_ptrs"
        (Op.ref .bool [TILE_N] "mask") (Op.negInf.broadcast [TILE_N]) _
        (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))
        (Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N)))
        (Tile.mk (fun _ : TileIndex [TILE_N] => (none : WithBot ℝ)))
        (by simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true])
        (by simp only [evalOp_ref, BlockState.setReg_same])
        (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.mk, loopBload]
      by_cases hin : c * TILE_N + idx.1.val < N
      · simp only [hin, decide_true, if_true]
        rw [BlockState.readMemValue_real]; rfl
      · simp only [hin, decide_false, if_false]; rfl))]
  -- e = exp(inp - m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))) (by
      rw [evalOp_exp, evalOp_sub]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm]
      rfl))]
  -- out = e / z
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realDiv (WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) (by
      rw [evalOp_div]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hz]
      rfl))]
  -- output_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast output_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- pre-store state
  set s8 := (((((((( s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N)) ).setReg "n_offsets" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val))).setReg "offset" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val)))).setReg "input_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))).setReg "mask" .bool [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N)))).setReg "inp" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => loopBload N TILE_N c s input_ptr idx))).setReg "e" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ))))).setReg "out" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)))).setReg "output_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) with hs8def
  have houtEval : evalOp (Op.ref .real [TILE_N] "out") s8
      = some (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub
          (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))
          (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) := by
    rw [evalOp_ref, hs8def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  have hptrEval : evalOp (Op.ref .ptr [TILE_N] "output_ptrs") s8
      = some (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) := by
    rw [evalOp_ref, hs8def]; simp only [BlockState.setReg_same]
  have hmaskEval : evalOp (Op.ref .bool [TILE_N] "mask") s8
      = some (Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N))) := by
    rw [evalOp_ref, hs8def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  -- masked store
  set actfn : TileIndex [TILE_N] → Bool := fun idx => decide (c * TILE_N + idx.1.val < N) with hactfn
  set ofn : TileIndex [TILE_N] → Nat := fun i => s.pid * N + (c * TILE_N + i.1.val) with hofn
  set vfn : TileIndex [TILE_N] → ℝ := fun i => pass2Val N TILE_N c cc S s input_ptr i with hvfn
  have hstore : stepStmt (Stmt.store .real [TILE_N] (MemAccess.ptr (Op.ref .ptr [TILE_N] "output_ptrs"))
      (Op.ref .real [TILE_N] "out") (MaskOpt.mask (Op.ref .bool [TILE_N] "mask"))) s8
      = some ((TileShape.allIndices [TILE_N]).foldl
          (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) s8) := by
    unfold stepStmt
    rw [houtEval]
    simp only [Option.bind_some, Option.bind_eq_bind, hmaskEval, hptrEval, Option.map_some]
    congr 1
    refine List.foldl_ext _ _ s8 (fun acc i _ => ?_)
    show (if actfn i then acc.writeMemTyped .real (Region.cast output_ptr) (ofn i)
        (WithBot.realDiv (WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr i)
          ((cc : ℝ) : WithBot ℝ))) (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)) else acc)
      = (if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc)
    cases ha : actfn i
    · simp only [Bool.false_eq_true, if_false]
    · simp only [if_true]
      rw [Region.cast_id, BlockState.writeMemTyped_real]
      congr 1
      show FloatDType.real.storeValue _ = vfn i
      rw [FloatDType.real_storeValue]
      have hin : c * TILE_N + i.1.val < N := by simpa [hactfn] using ha
      have hld : loopBload N TILE_N c s input_ptr i
          = (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + i.1.val))) : WithBot ℝ) := by
        unfold loopBload; rw [if_pos hin]
      rw [hld]
      simp only [hvfn, pass2Val]
      exact pass2_store_value _ cc S
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  set sF := (TileShape.allIndices [TILE_N]).foldl
      (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) s8 with hsF
  refine ⟨sF,rfl,?_⟩
  intro r o h
  have hf := masked_scatter_raw_frame (TileShape.allIndices [TILE_N]) s8
    output_ptr r ofn vfn actfn o
    (fun i _ hi ↦ by
      have hi' : c*TILE_N+i.1.val<N := by simpa [hactfn] using hi
      simpa [hofn,pass2Off] using h i hi')
  simpa only [hsF, hs8def] using hf

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize
