import bench.tritonbench_g.softmax_optimize.Proofs.LegacyReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton
open scoped BigOperators
set_option maxHeartbeats 6000000
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-- The historical complete four-loop real program succeeds; no execution
premise is required. Signed-source transport is established separately. -/
theorem full_surface_terminates (output_ptr input_ptr : RegionName) (M N TILE_N : Nat)
    (hN : 0<N) (hT : 0<TILE_N) (hne : output_ptr≠input_ptr) (s : BlockState) :
    ∃ t, exec (softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N).toAlgKernel s
      = some t := by
  cases hExec : exec (softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N).toAlgKernel s with
  | some t => exact ⟨t,rfl⟩
  | none =>
    exfalso
    -- abbreviations
    set pm := prevMultiple N TILE_N with hpm
    -- `exec = stepStmts body`; expose the concrete erased statement list
    rw [exec] at hExec
    -- name the seed states
    set s1 := s.setReg "pid_m" .nat [] (Tile.scalar s.pid) with hs1
    -- the body is (defeq) the concrete statement list
    have hbody : stepStmts (softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N).toAlgKernel.body s
        = stepStmts
          ([ Stmt.assign .nat [] "pid_m" (Op.programId 0),
             Stmt.assign .real [TILE_N] "m" (Op.full [TILE_N] Op.negInf),
             Stmt.assign .real [TILE_N] "z" (Op.full [TILE_N] (Op.const 0)),
             Stmt.assign .nat [] "prev_multiple"
               (Op.sub NumericDType.nat Broadcast.nil
                 (Op.mul NumericDType.nat Broadcast.nil
                   (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N))
                   (Op.constNat TILE_N))
                 (Op.constNat TILE_N)),
             Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "prev_multiple")
               (Op.constNat TILE_N) (loopAbody input_ptr N TILE_N),
             Stmt.forRangeDyn "start_n" (Op.ref .nat [] "prev_multiple") (Op.constNat N)
               (Op.constNat TILE_N) (loopBbody input_ptr N TILE_N) ]
           ++ reductionStmts TILE_N
           ++ [ Stmt.assign .nat [] "prev_multiple"
                  (Op.sub NumericDType.nat Broadcast.nil
                    (Op.mul NumericDType.nat Broadcast.nil
                      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N))
                      (Op.constNat TILE_N))
                    (Op.constNat TILE_N)),
                Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "prev_multiple")
                  (Op.constNat TILE_N) (loopApass2body output_ptr input_ptr N TILE_N),
                Stmt.forRangeDyn "start_n" (Op.ref .nat [] "prev_multiple") (Op.constNat N)
                  (Op.constNat TILE_N) (loopBpass2body output_ptr input_ptr N TILE_N) ]) s := rfl
    rw [hbody] at hExec
    -- ===== Phase 0: step the four seed assignments =====
    -- prev_multiple evaluation (used twice: seed + pass-2 reassign)
    have hpmEval : ∀ st : BlockState, evalOp
        (Op.sub NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N))
            (Op.constNat TILE_N))
          (Op.constNat TILE_N)) st = some (Tile.scalar (prevMultiple N TILE_N)) := by
      intro st
      have hfd : evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
              (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N)) st
            = some (Tile.bop (IntegralDType.nat).floorDiv Broadcast.nil
                (Tile.scalar (N + TILE_N - 1)) (Tile.scalar TILE_N)) := by
        simp only [evalOp, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
      simp only [evalOp_sub, evalOp_mul, evalOp_constNat, Option.bind_some, Option.bind_eq_bind, hfd]
      apply congrArg some
      apply Tile.ext; intro idx
      simp only [Tile.bop_data, Tile.scalar_data, prevMultiple]
      rfl
    -- flatten the `[..] ++ reductionStmts ++ [..]` literal into a cons-chain
    simp only [List.cons_append, List.nil_append, List.append_assoc] at hExec
    -- peel: pid_m, m, z, prev_multiple
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (show evalOp (Op.programId 0) s = some (Tile.scalar s.pid) from by
            simp only [evalOp_programId]))] at hExec
    set sa := s.setReg "pid_m" .nat [] (Tile.scalar s.pid) with hsa
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (show evalOp (Op.full [TILE_N] Op.negInf) sa
              = some ⟨fun _ : TileIndex [TILE_N] => (⊥ : WithBot ℝ)⟩ from by
            simp only [evalOp_full, evalOp_negInf]; rfl))] at hExec
    set sb := sa.setReg "m" .real [TILE_N] ⟨fun _ : TileIndex [TILE_N] => (⊥ : WithBot ℝ)⟩ with hsb
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (show evalOp (Op.full [TILE_N] (Op.const 0)) sb
              = some ⟨fun _ : TileIndex [TILE_N] => ((0 : ℝ) : WithBot ℝ)⟩ from by
            simp only [evalOp_full, evalOp_const]; rfl))] at hExec
    set sc := sb.setReg "z" .real [TILE_N] ⟨fun _ : TileIndex [TILE_N] => ((0 : ℝ) : WithBot ℝ)⟩ with hsc
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hpmEval sc))] at hExec
    set sSeed := sc.setReg "prev_multiple" .nat [] (Tile.scalar (prevMultiple N TILE_N)) with hsSeed
    -- ===== sSeed register facts =====
    have hSeed_pids : sSeed.pids = s.pids := by
      rw [hsSeed, hsc, hsb, hsa]
      simp only [BlockState.setReg_pids]
    have hSeed_mem : sSeed.mem = s.mem := by
      rw [hsSeed, hsc, hsb, hsa]; rfl
    have hSeed_pid : sSeed.pid = s.pid := by
      unfold BlockState.pid; rw [hSeed_pids]
    have hSeed_undef : ∀ rg o, sSeed.undef rg o = s.undef rg o := by
      intro rg o; rw [hsSeed, hsc, hsb, hsa]
      simp only [BlockState.setReg_undef]
    have hSeed_pidm : sSeed.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by
      rw [hsSeed, hsc, hsb]
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
      rw [hsa]; simp only [BlockState.setReg_same]
    have hSeed_pm : evalOp (Op.ref .nat [] "prev_multiple") sSeed
        = some (Tile.scalar (prevMultiple N TILE_N)) := by
      rw [evalOp_ref, hsSeed]; simp only [BlockState.setReg_same]
    have hSeed_m : sSeed.regs .real [TILE_N] "m" = some
        ⟨fun idx : TileIndex [TILE_N] => (laneState s input_ptr N TILE_N idx.1 (0 / TILE_N)).1⟩ := by
      rw [hsSeed, hsc]
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
      rw [hsb]; simp only [BlockState.setReg_same]
      congr 1; apply Tile.ext; intro idx
      simp only [Nat.zero_div, laneState, laneScores, List.range_zero, List.filterMap_nil,
        List.foldl_nil]
    have hSeed_z : sSeed.regs .real [TILE_N] "z" = some
        ⟨fun idx : TileIndex [TILE_N] =>
          (((laneState s input_ptr N TILE_N idx.1 (0 / TILE_N)).2 : ℝ) : WithBot ℝ)⟩ := by
      rw [hsSeed]
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
      rw [hsc]; simp only [BlockState.setReg_same]
      congr 1; apply Tile.ext; intro idx
      simp only [Nat.zero_div, laneState, laneScores, List.range_zero, List.filterMap_nil,
        List.foldl_nil]
    have hSeed_prev : sSeed.regs .nat [] "prev_multiple"
        = some (Tile.scalar ((N + TILE_N - 1) / TILE_N * TILE_N - TILE_N)) := by
      rw [hsSeed]; simp only [BlockState.setReg_same]; rfl
    have hInv0 : pass1InvA input_ptr N TILE_N s 0 sSeed :=
      ⟨⟨hSeed_pids, hSeed_mem, hSeed_undef, hSeed_pidm, hSeed_m, hSeed_z, hSeed_prev⟩, ⟨0, rfl⟩⟩
    -- ===== Phase 1: the two pass-1 loops =====
    obtain ⟨sLoopA, hstepA, hInvA⟩ :=
      pass1LoopA_run N TILE_N hT input_ptr s sSeed hSeed_pm hInv0
    rw [stepStmts.cons_some hstepA] at hExec
    -- prev_multiple still readable in sLoopA from the carried invariant clause
    have hA_pm : evalOp (Op.ref .nat [] "prev_multiple") sLoopA
        = some (Tile.scalar (prevMultiple N TILE_N)) := by
      rw [evalOp_ref, hInvA.1.2.2.2.2.2.2]; rfl
    obtain ⟨finalB, sLoopB, hstepB, hfinalB, hInvB⟩ :=
      pass1LoopB_run N TILE_N hT input_ptr s sLoopA hA_pm hInvA
    rw [stepStmts.cons_some hstepB] at hExec
    -- after both pass-1 loops, m/z = laneState over `finalB` tiles, fully covering N
    obtain ⟨⟨hB_pids, hB_mem, hB_undef, hB_pidm, hB_m, hB_z, hB_prev⟩, hB_dvd⟩ := hInvB
    -- ===== Phase 2: the reduction collapse =====
    set cfin := finalB / TILE_N with hcfin_def
    have hcfin_mul : cfin * TILE_N = finalB := Nat.div_mul_cancel hB_dvd
    have hcov : N ≤ cfin * TILE_N := by rw [hcfin_mul]; exact hfinalB
    have hcfin_pos : 0 < cfin := by
      rcases Nat.eq_zero_or_pos cfin with h | h
      · exfalso; rw [h, Nat.zero_mul] at hcov; exact absurd hcov (Nat.not_le.mpr hN)
      · exact h
    have hB_pidEq : sLoopB.pid = s.pid := by unfold BlockState.pid; rw [hB_pids]
    have hB_cong : ∀ (ℓ : Fin TILE_N) (cc : Nat),
        laneState s input_ptr N TILE_N ℓ cc = laneState sLoopB input_ptr N TILE_N ℓ cc :=
      fun ℓ cc => (laneState_congr sLoopB s input_ptr N TILE_N hB_pidEq hB_mem ℓ cc).symm
    have hB_m' : sLoopB.regs .real [TILE_N] "m" = some
        ⟨fun idx : TileIndex [TILE_N] => (laneState s input_ptr N TILE_N idx.1 cfin).1⟩ := hB_m
    have hB_z' : sLoopB.regs .real [TILE_N] "z" = some
        ⟨fun idx : TileIndex [TILE_N] =>
          (((laneState s input_ptr N TILE_N idx.1 cfin).2 : ℝ) : WithBot ℝ)⟩ := hB_z
    obtain ⟨cmax, sRed, hstepRed, hRed_pids, hRed_mem, hRed_undef, hRed_pidm,
        hRed_m, hRed_z⟩ :=
      reduction_step N TILE_N hN hT input_ptr cfin hcfin_pos hcov s sLoopB
        hB_pidEq hB_cong hB_m' hB_z'
    -- the reduced denominator `S = Σ_{j<N} exp(input[pid·N+j])`
    set S : ℝ := ∑ j ∈ Finset.range N, Real.exp (sLoopB.readMem input_ptr (sLoopB.pid * N + j)) with hS_def
    -- thread `reductionStmts` through hExec
    rw [stepStmts.append_some hstepRed] at hExec
    -- ===== Phase 3: pass-2 reassign + two store loops =====
    -- peel the pass-2 `prev_multiple` reassignment
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hpmEval sRed))] at hExec
    set sP0 := sRed.setReg "prev_multiple" .nat [] (Tile.scalar (prevMultiple N TILE_N)) with hsP0
    -- memory / pid bridges from `s` to the pass-2 seed `sP0`
    have hP0_pids : sP0.pids = s.pids := by
      rw [hsP0]; simp only [BlockState.setReg_pids]; rw [hRed_pids, hB_pids]
    have hP0_pidEq : sP0.pid = s.pid := by unfold BlockState.pid; rw [hP0_pids]
    have hP0_mem : sP0.mem = s.mem := by
      rw [hsP0]; show sRed.mem = s.mem; rw [hRed_mem, hB_mem]
    have hP0_readInp : ∀ o, sP0.readMem input_ptr o = s.readMem input_ptr o := by
      intro o; unfold BlockState.readMem; rw [hP0_mem]
    -- the reduced max value `cmax` and denominator `S` (over `s`)
    have hS_over_s : S = ∑ j ∈ Finset.range N, Real.exp (s.readMem input_ptr (s.pid * N + j)) := by
      rw [hS_def]; apply Finset.sum_congr rfl; intro j _
      have hmem : sLoopB.readMem input_ptr (sLoopB.pid * N + j) = s.readMem input_ptr (s.pid * N + j) := by
        unfold BlockState.readMem; rw [hB_mem, hB_pidEq]
      rw [hmem]
    -- pass-2 register facts on the seed `sP0`
    have hP0_pidm : sP0.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by
      rw [hsP0]
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
      rw [hRed_pidm, hB_pidm]
    have hP0_m : sP0.regs .real [] "m" = some (Tile.scalar ((cmax : ℝ) : WithBot ℝ)) := by
      rw [hsP0]
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
      exact hRed_m
    have hP0_z : sP0.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cmax) * S : ℝ)) : WithBot ℝ)) := by
      rw [hsP0]
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
      rw [hRed_z]
    have hP0_pm : evalOp (Op.ref .nat [] "prev_multiple") sP0
        = some (Tile.scalar (prevMultiple N TILE_N)) := by
      rw [evalOp_ref, hsP0]; simp only [BlockState.setReg_same]
    have hP0_prev : sP0.regs .nat [] "prev_multiple"
        = some (Tile.scalar (prevMultiple N TILE_N)) := by
      rw [hsP0]; simp only [BlockState.setReg_same]
    -- pass-2 invariant at counter 0 over seed `sP0`
    have hP2Inv0 : pass2InvA output_ptr input_ptr N TILE_N cmax S sP0 0 sP0 := by
      refine ⟨⟨rfl, ?_, hP0_m, hP0_z, fun o => rfl, ?_, rfl⟩, ⟨0, rfl⟩⟩
      · rw [hP0_pidEq]; exact hP0_pidm
      · intro j hj; rw [if_neg (Nat.not_lt.mpr (Nat.zero_le _))]
    -- pass-2 loop A (aligned → ends exactly at prevMultiple)
    obtain ⟨sP1, hstepP1, hP2InvA⟩ :=
      pass2LoopA_run N TILE_N hT output_ptr input_ptr hne cmax S sP0 sP0 hP0_pm hP2Inv0
    rw [stepStmts.cons_some hstepP1] at hExec
    -- prev_multiple still readable in sP1 from the carried invariant clause
    have hP1_pm : evalOp (Op.ref .nat [] "prev_multiple") sP1
        = some (Tile.scalar (prevMultiple N TILE_N)) := by
      rw [evalOp_ref, hP2InvA.1.2.2.2.2.2.2, hP0_prev]
    -- pass-2 loop B (single masked tail tile → ends at finalP2 ≥ N)
    obtain ⟨finalP2, sP2, hstepP2, hfinalP2, hP2InvB⟩ :=
      pass2LoopB_run N TILE_N hT output_ptr input_ptr hne cmax S sP0 sP1 hP1_pm hP2InvA
    rw [stepStmts.cons_some hstepP2, stepStmts.nil] at hExec
    cases hExec

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize
