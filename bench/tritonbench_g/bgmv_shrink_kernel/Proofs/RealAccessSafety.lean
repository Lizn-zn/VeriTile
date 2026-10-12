import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealProgram

/-! Mask-aware access bounds for one complete split-K body. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealAccessSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- `current_k < K` mask eval (the 1-D sibling of `cmask_eval`). -/
private theorem maskA_eval (K BLOCK_K : Nat) (ck : Fin BLOCK_K → Nat)
    (s : BlockState)
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_K] "current_k")
        (Op.constNat K)) s
      = some (⟨fun e : TileIndex [BLOCK_K] => decide (ck e.1 < K)⟩
          : Tile .bool [BLOCK_K]) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hck, Option.bind_some, bind,
    Option.bind_eq_bind, Option.bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.cop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  rfl

/-- `a_ptr + current_k_c` pointer-tile eval. -/
private theorem aTilePtr_eval (input_ptr : RegionName) (BLOCK_K apo : Nat)
    (ck : Fin BLOCK_K → Nat) (s : BlockState)
    (hap : s.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, apo) : RegionName × Nat)))
    (hckc : s.regs .nat [BLOCK_K] "current_k_c" = some (Tile.vec ck)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "a_ptr")
        (Op.ref .nat [BLOCK_K] "current_k_c")) s
      = some (⟨fun e : TileIndex [BLOCK_K] =>
          ((input_ptr, apo + ck e.1) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_K]) := by
  rw [evalOp_ptrAdd']
  simp only [evalOp_ref, hap, hckc, Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro e
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

/-- `tiled_b` pointer-tile eval: cell `(n, e)` addresses
`bpo + n·lora_k_stride + ck e·lora_n_stride` in the LoRA buffer. -/
private theorem bTilePtr_eval (lora_ptr : RegionName)
    (lora_k_stride lora_n_stride BLOCK_N BLOCK_K bpo : Nat)
    (ck : Fin BLOCK_K → Nat) (s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr"
      = some (Tile.scalar ((lora_ptr, bpo) : RegionName × Nat)))
    (hon : s.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hck : s.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ck)) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "b_ptr")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n"))
            (Op.constNat lora_k_stride)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "current_k"))
          (Op.constNat lora_n_stride))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          ((lora_ptr, bpo + idx.1.val * lora_k_stride
              + ck idx.2.1 * lora_n_stride) : RegionName × Nat)⟩
            : Tile .ptr [BLOCK_N, BLOCK_K]) := by
  simp only [evalOp, evalOp.eq_def, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, evalOp_ref, evalOp_constNat, hbp, hon, hck,
    Option.bind, Option.map, Option.bind_some, bind, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, NumericDType.add]
  rfl

/-- The loop counter of an in-range iteration sits below the ceil trip count:
`i < K` with `i = (i/S)·S` forces `i/S < ⌈K/S⌉`. -/
private theorem kloop_counter_lt (K S i : Nat) (hS : 0 < S) (hilt : i < K)
    (hi : i = i / S * S) : i / S < numKIters K S := by
  by_contra h
  have h1 : numKIters K S * S ≤ i / S * S :=
    Nat.mul_le_mul_right S (Nat.le_of_not_lt h)
  have h2 : K ≤ i / S * S := le_trans (numKIters_bounds K S hS).1 h1
  rw [← hi] at h2
  exact absurd hilt (Nat.not_lt.mpr h2)

set_option maxHeartbeats 2000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two **masked**
loads' active lanes are exactly the kernel's `current_k < K` (and, for
`tiled_b`, `offset_n < N`) windows, so their addresses are covered by the
mask-gated `read1`/`read2` bound groups; the four remaining assigns are
register-only. -/
theorem body_safe (bounds : RegionBounds)
    (input_ptr lora_ptr : RegionName) (slot : Nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 : BlockState)
    (hS : 0 < BLOCK_K * SPLIT_K) (i : Nat) (hilt : i < K)
    (hi : i = i / (BLOCK_K * SPLIT_K) * (BLOCK_K * SPLIT_K))
    (sk : BlockState)
    (hk : sk.regs .nat [] "k" = some (Tile.scalar i))
    (hon : sk.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hok : sk.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)))
    (hap : sk.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)))
    (hbp : sk.regs .ptr [] "b_ptr"
      = some (Tile.scalar
          ((lora_ptr, l0_stride * slot) : RegionName × Nat)))
    (hbA : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K))) (e : Fin BLOCK_K),
      t.val * (BLOCK_K * SPLIT_K) + (e.val + s0.pids 0 * BLOCK_K) < K →
      s0.pids 1 * xm_stride
          + (t.val * (BLOCK_K * SPLIT_K) + (e.val + s0.pids 0 * BLOCK_K))
        < bounds input_ptr)
    (hbB : ∀ (t : Fin (numKIters K (BLOCK_K * SPLIT_K)))
        (l : Fin (BLOCK_N * BLOCK_K)),
      l.val / BLOCK_K < N ∧
        t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + s0.pids 0 * BLOCK_K) < K →
      l0_stride * slot + l.val / BLOCK_K * lora_k_stride
          + (t.val * (BLOCK_K * SPLIT_K)
              + (l.val % BLOCK_K + s0.pids 0 * BLOCK_K)) * lora_n_stride
        < bounds lora_ptr) :
    Stmt.TraceSafeListR .triv bounds
      (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K) sk := by
  have hcT : i / (BLOCK_K * SPLIT_K) < numKIters K (BLOCK_K * SPLIT_K) :=
    kloop_counter_lt K (BLOCK_K * SPLIT_K) i hS hilt hi
  set ckF : Fin BLOCK_K → Nat :=
    fun e => i + (e.val + s0.pids 0 * BLOCK_K) with hckF
  have hck_eq : ∀ e : Fin BLOCK_K,
      ckF e = i / (BLOCK_K * SPLIT_K) * (BLOCK_K * SPLIT_K)
        + (e.val + s0.pids 0 * BLOCK_K) := by
    intro e
    rw [hckF, ← hi]
  unfold shrinkLoopBody
  -- 1. current_k (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv h1
  rw [evalOpR_triv,
    currentK_eval BLOCK_K i (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)
      sk hk hok] at hv1
  obtain rfl := Option.some.inj hv1
  -- 2. current_k_c (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  rw [evalOpR_ref, BlockState.setReg_same] at hv2
  obtain rfl := Option.some.inj hv2
  have hck2 : ((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
      "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).regs .nat [BLOCK_K]
      "current_k" = some (Tile.vec ckF) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("current_k" : RegName) ≠ "current_k_c" by decide),
      BlockState.setReg_same]
  have hckc2 : ((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
      "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).regs .nat [BLOCK_K]
      "current_k_c" = some (Tile.vec ckF) := by
    rw [BlockState.setReg_same]
  have hap2 : ((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
      "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("a_ptr" : RegName) ≠ "current_k_c" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("a_ptr" : RegName) ≠ "current_k" by decide)]
    exact hap
  -- 3. tiled_a (masked load, mask-gated bound group)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp, by simp, ?_⟩
    intro ptrs hptrs e hactive
    rw [evalOpR_triv,
      aTilePtr_eval input_ptr BLOCK_K (s0.pids 1 * xm_stride) ckF _
        hap2 hckc2] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hme⟩ := hactive
    rw [evalOpR_triv, maskA_eval K BLOCK_K ckF _ hck2] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hkK : ckF e.1 < K := of_decide_eq_true hme
    show s0.pids 1 * xm_stride + ckF e.1 < bounds input_ptr
    have h' := hbA ⟨i / (BLOCK_K * SPLIT_K), hcT⟩ e.1
      (by rw [← hck_eq e.1]; exact hkK)
    rw [← hck_eq e.1] at h'
    exact h'
  · intro s3 h3
    obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
    -- 4. b_ptr_mask (register-only)
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
    intro s4 h4
    obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv h4
    have hon3 : (((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
        "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
        [BLOCK_K] v3).regs .nat [BLOCK_N] "offset_n"
        = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
      rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "tiled_a" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "current_k_c" by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "current_k" by decide)]
      exact hon
    have hck3 : (((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
        "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
        [BLOCK_K] v3).regs .nat [BLOCK_K] "current_k"
        = some (Tile.vec ckF) := by
      rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("current_k" : RegName) ≠ "tiled_a" by decide)]
      exact hck2
    rw [evalOpR_triv, bmask_eval N K BLOCK_N BLOCK_K ckF _ hon3 hck3] at hv4
    obtain rfl := Option.some.inj hv4
    -- 5. tiled_b (masked load, mask-gated bound group)
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨by simp, by simp, ?_⟩
      intro ptrs hptrs idx hactive
      have hbp4 : ((((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
          "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
          [BLOCK_K] v3).setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
            decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩
              : Tile .bool [BLOCK_N, BLOCK_K])).regs .ptr [] "b_ptr"
          = some (Tile.scalar
              ((lora_ptr, l0_stride * slot) : RegionName × Nat)) := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "b_ptr_mask" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "tiled_a" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "current_k_c" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("b_ptr" : RegName) ≠ "current_k" by decide)]
        exact hbp
      have hon4 : ((((sk.setReg "current_k" .nat [BLOCK_K]
          (Tile.vec ckF)).setReg "current_k_c" .nat [BLOCK_K]
          (Tile.vec ckF)).setReg "tiled_a" .real [BLOCK_K] v3).setReg
          "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
            decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩
              : Tile .bool [BLOCK_N, BLOCK_K])).regs .nat [BLOCK_N] "offset_n"
          = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offset_n" : RegName) ≠ "b_ptr_mask" by decide)]
        exact hon3
      have hck4 : ((((sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF)).setReg
          "current_k_c" .nat [BLOCK_K] (Tile.vec ckF)).setReg "tiled_a" .real
          [BLOCK_K] v3).setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K]
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
            decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩
              : Tile .bool [BLOCK_N, BLOCK_K])).regs .nat [BLOCK_K] "current_k"
          = some (Tile.vec ckF) := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("current_k" : RegName) ≠ "b_ptr_mask" by decide)]
        exact hck3
      rw [evalOpR_triv,
        bTilePtr_eval lora_ptr lora_k_stride lora_n_stride BLOCK_N BLOCK_K
          (l0_stride * slot) ckF _ hbp4 hon4 hck4] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hmasks, hmi⟩ := hactive
      rw [evalOpR_ref, BlockState.setReg_same] at hmasks
      obtain rfl := Option.some.inj hmasks
      have hmi' : idx.1.val < N ∧ ckF idx.2.1 < K := by
        have := hmi
        simp only [Bool.and_eq_true, decide_eq_true_eq] at this
        exact this
      show l0_stride * slot + idx.1.val * lora_k_stride
          + ckF idx.2.1 * lora_n_stride < bounds lora_ptr
      have h' := hbB ⟨i / (BLOCK_K * SPLIT_K), hcT⟩
        (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
        (by
          rw [Lane2D.encode_div, Lane2D.encode_mod, ← hck_eq idx.2.1]
          exact hmi')
      rw [Lane2D.encode_div, Lane2D.encode_mod, ← hck_eq idx.2.1] at h'
      exact h'
    · intro s5 h5
      obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv h5
      -- 6. accumulator (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
        (fun s6 _ => Stmt.TraceSafeListR.nil_intro)



open RealSetupSource RealSetup

/-- The complete loop is safe whenever each actually active input/weight
address is allocated. Input loads do not depend on the output N mask. -/
theorem loop_safe (bounds : RegionBounds) (p : Parameters) (initial : BlockState) (slot : Nat)
    (hS : 0 < p.BK * p.splits)
    (hbA : ∀ (c : Fin (RealLoop.numKIters p.K (p.BK * p.splits))) (e : Fin p.BK),
      RealLoop.kIdx initial p.BK p.splits c.val e.val < p.K →
      initial.pids 1 * p.inputBatchStride + RealLoop.kIdx initial p.BK p.splits c.val e.val < bounds p.input)
    (hbB : ∀ (c : Fin (RealLoop.numKIters p.K (p.BK * p.splits))) (n : Fin p.BN) (e : Fin p.BK),
      n.val < p.N → RealLoop.kIdx initial p.BK p.splits c.val e.val < p.K →
      p.slotStride * slot + n.val * p.rowStride +
        RealLoop.kIdx initial p.BK p.splits c.val e.val * p.colStride < bounds p.weights) :
    Stmt.TraceSafeR .triv bounds p.loop (afterSetup p initial slot) := by
  have hbB' : ∀ (c : Fin (numKIters p.K (p.BK * p.splits))) (l : Fin (p.BN * p.BK)),
      l.val / p.BK < p.N ∧
        c.val * (p.BK * p.splits) + (l.val % p.BK + initial.pids 0 * p.BK) < p.K →
      p.slotStride * slot + l.val / p.BK * p.rowStride +
        (c.val * (p.BK * p.splits) + (l.val % p.BK + initial.pids 0 * p.BK)) * p.colStride
        < bounds p.weights := by
    intro c l hl
    have hk : 0 < p.BK := by
      by_contra h
      have hz : p.BK = 0 := Nat.eq_zero_of_not_pos h
      simp [hz] at hS
    exact hbB c ⟨l.val / p.BK, (Nat.div_lt_iff_lt_mul hk).mpr l.isLt⟩
      ⟨l.val % p.BK, Nat.mod_lt _ hk⟩ hl.1 hl.2
  have hstep : ∀ c st, c < p.K →
      RealLoop.shrinkInv p.input p.weights slot p.N p.K p.inputBatchStride p.slotStride
        p.rowStride p.colStride p.BN p.BK p.splits initial c st →
      Stmt.TraceSafeListR .triv bounds
        (shrinkLoopBody p.N p.K p.rowStride p.colStride p.BN p.BK)
        (st.setReg "k" .nat [] (Tile.scalar c)) ∧
      ∃ next, stepStmtsR .triv (shrinkLoopBody p.N p.K p.rowStride p.colStride p.BN p.BK)
        (st.setReg "k" .nat [] (Tile.scalar c)) = some next ∧
        RealLoop.shrinkInv p.input p.weights slot p.N p.K p.inputBatchStride p.slotStride
          p.rowStride p.colStride p.BN p.BK p.splits initial (c+p.BK*p.splits) next := by
    intro c st hc hi
    obtain ⟨hpids, hic, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩ := hi
    constructor
    · refine body_safe bounds p.input p.weights slot p.N p.K p.inputBatchStride p.slotStride
        p.rowStride p.colStride p.BN p.BK p.splits initial hS c hc hic _
        (by simp) ?_ ?_ ?_ ?_ hbA hbB'
      · simpa using hon
      · simpa using hok
      · simpa using hap
      · simpa using hbp
    · simpa only [stepStmtsR_triv] using RealLoop.shrink_step p.input p.weights slot p.N p.K
        p.inputBatchStride p.slotStride p.rowStride p.colStride p.BN p.BK p.splits
        initial hS c st hc ⟨hpids, hic, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩
  simpa only [Parameters.loop, Stmt.TraceSafeR] using Stmt.forRangeTraceSafeR_inv .triv bounds "k" p.K (p.BK * p.splits)
    (shrinkLoopBody p.N p.K p.rowStride p.colStride p.BN p.BK)
    (RealLoop.shrinkInv p.input p.weights slot p.N p.K p.inputBatchStride p.slotStride
      p.rowStride p.colStride p.BN p.BK p.splits initial)
    hstep 0 (afterSetup p initial slot) (setup_invariant p initial slot hS)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealAccessSafety
