import bench.tritonbench_g.attention_llama.Proofs.RealSafetyFragments

/-! Allocation bounds at the constructed Q and Out pointer tiles. Pure address
assignments are evaluated exactly; data registers introduce no numeric premise. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafetyPrologue
open VeriTile.Triton RealKernel RealSafetyFragments
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- Q loads require only addresses of active query rows to be allocated. -/
theorem preamble_safe (bounds : RegionBounds) (s : BlockState) (Q K V : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX BM BN BD : Nat)
    (hb : ∀ i : TileIndex [BM, BD], s.pids 0 * BM + i.1.val < H →
      alBase s N_HEAD sqz sqh + (s.pids 0 * BM + i.1.val) * sqm + i.2.1.val * sqk < bounds Q) :
    Stmt.TraceSafeList bounds (alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX BM BN BD) s := by
  unfold alPreLoopG
  -- S1: start_m = pid0
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S2: head_idx = pid1
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S3: batch_id = head_idx // N_HEAD
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.ref .nat [] "head_idx") (Op.constNat N_HEAD)) _
        = some (Tile.scalar (s.pids 1 / N_HEAD)) from by
      rw [al_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext u; rfl))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S4: off_hz = head_idx % N_HEAD
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil
        (Op.ref .nat [] "head_idx") (Op.constNat N_HEAD)) _
        = some (Tile.scalar (s.pids 1 % N_HEAD)) from by
      rw [al_evalOp_mod]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext u; rfl))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S5: offs_m = start_m * BM + arange BM
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM)) (Op.arange BM)) _
        = some (Tile.vec (fun r : Fin BM => s.pids 0 * BM + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S6: offs_n = arange BN
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_arange BN _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S7: offs_d = arange BD
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_arange BD _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S8-S10: the three flat 2D offset tiles
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat sqz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat sqh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat sqm)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat sqk))) _
        = some (⟨fun idx : TileIndex [BM, BD] =>
            alBase s N_HEAD sqz sqh + (s.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk⟩
              : Tile .nat [BM, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BM, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun r : Fin BM => s.pids 0 * BM + r.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      rfl))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat skz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat skh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat skn)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat skk))) _
        = some (⟨fun idx : TileIndex [BN, BD] =>
            alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk⟩
              : Tile .nat [BN, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BN, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => j.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      show (s.pids 1 / N_HEAD) * skz + (s.pids 1 % N_HEAD) * skh + idx.1.val * skn + idx.2.1.val * skk
        = alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk
      simp [alBase, Nat.zero_add]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat svz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat svh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat svk)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat svn))) _
        = some (⟨fun idx : TileIndex [BN, BD] =>
            alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn⟩
              : Tile .nat [BN, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BN, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => j.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      show (s.pids 1 / N_HEAD) * svz + (s.pids 1 % N_HEAD) * svh + idx.1.val * svk + idx.2.1.val * svn
        = alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn
      simp [alBase, Nat.zero_add]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S11-S13: the three pointer tiles
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q) (Op.ref .nat [BM, BD] "off_q")) _
        = some (alPtrTile Q (alBase s N_HEAD sqz sqh) sqm sqk (s.pids 0 * BM) BM BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (Q, 0 + (alBase s N_HEAD sqz sqh + (s.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk))
        = (Q, alBase s N_HEAD sqz sqh + (s.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk)
      rw [Nat.zero_add]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [BN, BD] "off_k")) _
        = some (alPtrTile K (alBase s N_HEAD skz skh) skn skk 0 BN BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (K, 0 + (alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk))
        = (K, alBase s N_HEAD skz skh + (0 + idx.1.val) * skn + idx.2.1.val * skk)
      rw [Nat.zero_add]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [BN, BD] "off_v")) _
        = some (alPtrTile V (alBase s N_HEAD svz svh) svk svn 0 BN BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (V, 0 + (alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn))
        = (V, alBase s N_HEAD svz svh + (0 + idx.1.val) * svk + idx.2.1.val * svn)
      rw [Nat.zero_add]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S14: m_prev = zeros - inf = ⊥
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BM] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BM] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, castTile_self, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, WithBot.realAdd, Option.map₂_none_right]
      rfl))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S15: l_prev = zeros
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BM] => some (0 : ℝ)⟩ : Tile .real [BM]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- S16: acc = zeros
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BD] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ : Tile .real [BM, BD]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r; rfl))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  apply TraceProofs.assign_cons
  · simp only [Op.SafeAt.eq_def, true_and, MemAccess.ActiveAddressSafe,
      memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def, true_and, MaskOpt.Active]
    intro ptrs hp i hi
    simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true] at hp
    cases hp
    rcases hi with ⟨mask, hm, hi⟩
    rw [al_rowmask_eval _ (fun r : Fin BM => s.pids 0 * BM + r.val) H (by
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true])] at hm
    cases hm
    exact hb i (of_decide_eq_true hi)
  · intro _
    apply TraceProofs.all_safe
    simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

/-- The masked store needs bounds only for active output lanes. This safety
result neither assumes a value for acc nor requires output injectivity. -/
theorem postamble_safe (bounds : RegionBounds) (Out : RegionName) (s0 : BlockState)
    (soz soh som son N_HEAD H BM BD : Nat) (sL : BlockState)
    (hpids : sL.pids = s0.pids)
    (hbi : sL.regs .nat [] "batch_id" = some (Tile.scalar (s0.pids 1 / N_HEAD)))
    (hoh : sL.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1 % N_HEAD)))
    (hb : ∀ i : TileIndex [BM, BD], alActive s0 H BM i →
      alOutOffset s0 N_HEAD soz soh som son BM i < bounds Out) :
    Stmt.TraceSafeList bounds (alPostLoopG Out soz soh som son H BM BD) sL := by
  unfold alPostLoopG
  -- W1: start_m = pid0
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.programId 0) sL = some (Tile.scalar (s0.pids 0)) from by
      rw [evalOp_programId, hpids]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- W2: offs_m = start_m * BM + arange BM
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM)) (Op.arange BM)) _
        = some (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- W3: offs_d = arange BD
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_arange BD _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- W4: off_o
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat soz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat son))) _
        = some (⟨fun idx : TileIndex [BM, BD] =>
            alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son⟩
              : Tile .nat [BM, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BM, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show evalOp (Op.ref .nat [] "batch_id") _ = some (Tile.scalar (s0.pids 1 / N_HEAD)) from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hbi]
      rw [show evalOp (Op.ref .nat [] "off_hz") _ = some (Tile.scalar (s0.pids 1 % N_HEAD)) from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hoh]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      rfl))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  -- W5: out_ptrs = Out + off_o
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BM, BD] "off_o")) _
        = some (alPtrTile Out (alBase s0 N_HEAD soz soh) som son (s0.pids 0 * BM) BM BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (Out, 0 + (alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
        = (Out, alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son)
      rw [Nat.zero_add]))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  apply Stmt.TraceSafeList.cons_intro
  · simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def,
      true_and, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def, true_and, MaskOpt.Active]
    intro ptrs hp i hi
    simp only [evalOp_ref, BlockState.setReg_same] at hp
    cases hp
    rcases hi with ⟨mask, hm, hi⟩
    rw [al_rowmask_eval _ (fun r : Fin BM => s0.pids 0 * BM + r.val) H (by
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true])] at hm
    cases hm
    exact hb i (of_decide_eq_true hi)
  · intro _ _
    exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafetyPrologue
