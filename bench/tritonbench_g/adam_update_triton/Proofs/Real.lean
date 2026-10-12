import VeriTile.Triton
import bench.tritonbench_g.adam_update_triton.Kernels

/-! Real execution, output-value, memory-frame and flat-pointer safety lemmas
for the two sources. The public masked IO contracts live in Correct.lean. -/

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂
set_option maxHeartbeats 5000000

/-! ## Single-program value core

The two algorithm-layer theorems below execute one program of the full
`update_fn_kernel` and characterize BOTH observable stores lane by lane —
active lanes hold the Lion oracles, masked-off lanes are unchanged — under
the assumption `p_ptr ≠ exp_avg_ptr` (no kernel callers alias those
buffers). They are the value legs of the headline's region-model triple. -/

/-- Per-lane `exp_avg` output spec: the reusable Lion momentum oracle applied
to the values this lane loads. The math lives once in `Math.Optimizer`; this
only names the memory reads. -/
noncomputable def expAvgFullSpec
    (s : BlockState) (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledOptimizer.lionMomentum
    (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) beta2

/-- Per-lane `p` output spec: the reusable Lion parameter-update oracle applied
to the values this lane loads. -/
noncomputable def pFullSpec
    (s : BlockState) (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledOptimizer.lionParam
    (s.readMem p_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) lr wd beta1

/-- Algorithm-layer correctness for the `exp_avg` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_exp_avg_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hRegions : exp_avg_ptr ≠ p_ptr)
    (hExec : exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) =
        if linearOffset s BLOCK_SIZE i < n_elements then
          expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i
        else s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.gt, ComparableDType.ne] at hExec
    subst s'
    simp only [linearOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          p_ptr _ _ _ _ _ _ _ hRegions]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · simp only [hi, if_true, expAvgFullSpec, TiledOptimizer.lionMomentum,
        linearOffset, BlockState.pid_eq, Option.map₂, Option.map, Option.bind,
        WithBot.some_eq_coe, WithBot.unbotD_coe]
      ring
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Algorithm-layer correctness for the `p` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_p_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr)
    (hExec : exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem p_ptr (linearOffset s BLOCK_SIZE i) =
        if linearOffset s BLOCK_SIZE i < n_elements then
          pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i
        else s.readMem p_ptr (linearOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.gt, ComparableDType.ne] at hExec
    subst s'
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          exp_avg_ptr _ _ _ _ _ _ _ hRegions]
    simp only [linearOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · -- The realized masked `p` store equals the Lion parameter-update oracle.
      unfold pFullSpec TiledOptimizer.lionParam TiledOptimizer.lionUpdateDir
      simp only [linearOffset, hi, if_true, BlockState.pid_eq,
        Option.map₂, Option.map, Option.bind]
      set m := s.readMem exp_avg_ptr (s.pids 0 * BLOCK_SIZE + i.val) with hm
      set g := s.readMem grad_ptr (s.pids 0 * BLOCK_SIZE + i.val) with hg
      set p := s.readMem p_ptr (s.pids 0 * BLOCK_SIZE + i.val) with hp
      have heq : (m - g) * beta1 + g = beta1 * m + (1 - beta1) * g := by ring
      split_ifs with c1 c2
      · have hd : beta1 * m + (1 - beta1) * g = 0 := heq ▸ Option.some_inj.mp c1
        simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, hd, Real.sign_zero]
        ring
      · have hd : (0 : ℝ) < beta1 * m + (1 - beta1) * g :=
          heq ▸ WithBot.coe_lt_coe.mp c2
        simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, Real.sign_of_pos hd]
        norm_num
        ring
      · have hne : (m - g) * beta1 + g ≠ 0 := fun hc => c1 (congrArg some hc)
        have hnp : ¬ (0 : ℝ) < (m - g) * beta1 + g :=
          fun hc => c2 (WithBot.coe_lt_coe.mpr hc)
        have hd : beta1 * m + (1 - beta1) * g < 0 :=
          heq ▸ lt_of_le_of_ne (not_lt.mp hnp) hne
        simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, Real.sign_of_neg hd]
        ring
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-! ## Termination and frame -/

/-- Progress: `update_fn_kernel` always executes to a defined state. -/
theorem adam_exec_isSome
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) (s : BlockState) :
    (exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE) s).isSome := by
  simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, ComparableDType.gt, ComparableDType.ne]

/-- Per-program write footprint: the two masked block stores to `p_ptr` and
`exp_avg_ptr` at offsets `pid·BLOCK_SIZE + lane`, active when `< n_elements`. -/
noncomputable def adamWritesFP
    (p_ptr exp_avg_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) : WriteFootprint :=
  WriteFootprint.union
    (WriteFootprint.activeTileImage p_ptr
      (fun i : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + i.1.val)
      (fun i : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + i.1.val < n_elements))
    (WriteFootprint.activeTileImage exp_avg_ptr
      (fun i : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + i.1.val)
      (fun i : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + i.1.val < n_elements))

/-- Frame write-footprint: a single program of `update_fn_kernel` only writes
the two masked blocks `adamWritesFP` (its `p_ptr` and `exp_avg_ptr` lanes). All
other memory — other regions, and out-of-block / masked-off cells — is
untouched. -/
theorem adam_writeWithin
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) (s : BlockState) :
    Kernel.ExecWritesWithin
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel s
      (adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE s) := by
  intro s' hExec
  simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, ComparableDType.gt, ComparableDType.ne] at hExec
  subst s'
  refine BlockState.writeWithin_trans (s₁ := ?mid1) ?hsp ?houter
  case houter =>
    refine BlockState.foldl_writeMem_prop_writeWithin (region := exp_avg_ptr)
      (active := fun i : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * BLOCK_SIZE + i.1.val < n_elements) _ _ _ _ ?_
    intro i _ hact
    exact Or.inr ⟨rfl, i, hact, rfl⟩
  case hsp =>
    refine BlockState.writeWithin_trans (s₁ := ?mid2) ?hbase ?hinner
    case hinner =>
      refine BlockState.foldl_writeMem_prop_writeWithin (region := p_ptr)
        (active := fun i : TileIndex [BLOCK_SIZE] =>
          s.pids 0 * BLOCK_SIZE + i.1.val < n_elements) _ _ _ _ ?_
      intro i _ hact
      exact Or.inl ⟨rfl, i, hact, rfl⟩
    case hbase =>
      intro r o _
      simp [BlockState.setReg_mem]

/-- Frame bridging lemma: `adam_writeWithin` pins the execution's write
footprint to `adamWritesFP` (the two active masked windows), so any cell
outside **both** active output windows — the exact frame condition shape of
`MaskedKernelIO₃ₓ₂.Implements.intro`'s `hrun` — is untouched. -/
theorem adamWritesFP_frame_cell
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s s1 : BlockState)
    (hExec : exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel s = some s1)
    (r : RegionName) (o : Nat)
    (h1 : r ≠ p_ptr ∨ ∀ j : Fin BLOCK_SIZE,
      s.pid * BLOCK_SIZE + j.val < n_elements → o ≠ s.pid * BLOCK_SIZE + j.val)
    (h2 : r ≠ exp_avg_ptr ∨ ∀ j : Fin BLOCK_SIZE,
      s.pid * BLOCK_SIZE + j.val < n_elements → o ≠ s.pid * BLOCK_SIZE + j.val) :
    s1.mem r o = s.mem r o := by
  have hw := adam_writeWithin p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
    n_elements BLOCK_SIZE s s1 hExec
  refine (hw r o ?_).symm
  rintro (⟨rfl, i, hact, ho⟩ | ⟨rfl, i, hact, ho⟩)
  · rcases h1 with hne | hno
    · exact hne rfl
    · exact hno i.1 hact ho.symm
  · rcases h2 with hne | hno
    · exact hne rfl
    · exact hno i.1 hact ho.symm

/-! ## Flat-memory bridge side conditions -/

/-- The full Lion-update surface sits inside the flat-memory bridge's
covered fragment (pointer arithmetic, masked loads/stores, `where`,
comparisons are all covered). -/
theorem update_fn_kernel_flattenOk
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ((update_fn_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [update_fn_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- Per-execution safety walk: all three masked loads and both masked
stores address the same window `pid * BLOCK_SIZE + j`, active only when
`< n_elements`, so the bounds contract is lane-wise — every *active* lane's
address is below the region bound of the buffer it touches. -/
theorem update_fn_kernel_traceSafe
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hp : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds p_ptr)
    (hg : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds grad_ptr)
    (he : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds exp_avg_ptr) :
    Kernel.TraceSafe bounds
      ((update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all 19 statements, discharging every
  -- load-free `SafeAt` and reducing the five memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [update_fn_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt, ComparableDType.gt, ComparableDType.ne,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.Active,
    BlockState.setReg]
  exact ⟨fun a ha => hp a ha, fun a ha => hg a ha, fun a ha => he a ha,
    fun a ha => hp a ha, fun a ha => he a ha⟩

/-! ## Region-model triple, IO signature, and headline -/

/-- **The region-model masked Hoare triple** — termination, active-lane
values of both in-place outputs, and frame off the two active output
windows, from any launch state whose three input windows are loaded at the
**active lanes only** (`pid * BLOCK_SIZE + j < n_elements`). This is the
`hrun` obligation of the `⊨` headline; its three legs are
`adam_exec_isSome`, `update_fn_kernel_p_correct` / `_exp_avg_correct`, and
`adamWritesFP_frame_cell`. -/
theorem update_fn_kernel_region_run
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr)
    (s₀ : BlockState) (xs ys zs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem p_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem grad_ptr (s₀.pid * BLOCK_SIZE + j.val) = ys j)
    (hz : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem exp_avg_ptr (s₀.pid * BLOCK_SIZE + j.val) = zs j) :
    ∃ s1, exec ((update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem p_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = TiledOptimizer.lionParam (xs j) (zs j) (ys j) lr wd beta1)
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem exp_avg_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = TiledOptimizer.lionMomentum (zs j) (ys j) beta2)
      ∧ (∀ r o,
          (r ≠ p_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          (r ≠ exp_avg_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := Option.isSome_iff_exists.mp
    (adam_exec_isSome p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
      n_elements BLOCK_SIZE s₀)
  have hs1' : exec ((update_fn_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel) s₀ = some s1 := hs1
  refine ⟨s1, hs1', ?_, ?_, ?_⟩
  · -- p window: the in-place parameter update realizes `lionParam`.
    intro j hj
    have h := update_fn_kernel_p_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s₀ s1 hRegions hs1 j
    rw [show linearOffset s₀ BLOCK_SIZE j = s₀.pid * BLOCK_SIZE + j.val
        from rfl] at h
    rw [h, if_pos hj]
    unfold pFullSpec
    rw [show linearOffset s₀ BLOCK_SIZE j = s₀.pid * BLOCK_SIZE + j.val
        from rfl, hx j hj, hy j hj, hz j hj]
  · -- exp_avg window: the in-place momentum update realizes `lionMomentum`.
    intro j hj
    have h := update_fn_kernel_exp_avg_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s₀ s1
      (fun hc => hRegions hc.symm) hs1 j
    rw [show linearOffset s₀ BLOCK_SIZE j = s₀.pid * BLOCK_SIZE + j.val
        from rfl] at h
    rw [h, if_pos hj]
    unfold expAvgFullSpec
    rw [show linearOffset s₀ BLOCK_SIZE j = s₀.pid * BLOCK_SIZE + j.val
        from rfl, hy j hj, hz j hj]
  · -- frame off the two active output windows.
    intro r o h1 h2
    exact adamWritesFP_frame_cell p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s₀ s1 hs1' r o h1 h2


section Commuted

theorem optional_add_comm (a b : Option ℝ) :
    Option.map₂ (fun x y => x + y) a b = Option.map₂ (fun x y => x + y) b a :=
  Option.map₂_comm add_comm

/-- Real commutativity preserves the entire state, including both stores. -/
theorem commuted_exec (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) (s : BlockState) :
    exec (update_commuted_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B).toAlgKernel s =
      exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B).toAlgKernel s := by
  simp [update_commuted_kernel, update_fn_kernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.select, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt, ComparableDType.gt, ComparableDType.ne, add_comm, optional_add_comm]

theorem commuted_flattenOk
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ((update_commuted_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [update_commuted_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

theorem commuted_traceSafe
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hp : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds p_ptr)
    (hg : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds grad_ptr)
    (he : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds exp_avg_ptr) :
    Kernel.TraceSafe bounds
      ((update_commuted_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all 19 statements, discharging every
  -- load-free `SafeAt` and reducing the five memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [update_commuted_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt, ComparableDType.gt, ComparableDType.ne,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.Active,
    BlockState.setReg]
  exact ⟨fun a ha => hp a ha, fun a ha => hg a ha, fun a ha => he a ha,
    fun a ha => hp a ha, fun a ha => he a ha⟩


end Commuted

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton
