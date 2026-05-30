import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Optimizer
import VeriTile.Triton.Launch.Composition

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Faithful transcription of `adam_update_triton.py`'s `update_fn_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def update_fn_kernel
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - $((lr : ℝ)) * $((wd : ℝ)))
  diff = exp_avg - grad
  update = diff * $(beta1) + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -$((lr : ℝ)), $((lr : ℝ)))
  p = p + update_sign * can_update
  exp_avg = diff * $(beta2) + grad
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}

/-- The full Adam update surface lowers to the algorithm layer, including both
masked stores to `p_ptr` and `exp_avg_ptr`. -/
theorem update_fn_kernel_surface_toAlgorithm_supported
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ∃ alg, (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg := by
  simp [update_fn_kernel, ComputeExpr.toAlgorithm?]

/-! ## Full-kernel output correctness

The Python test runs the full `update_fn_kernel` which writes both `p` and
`exp_avg`. Per #139's audit, slice proofs are insufficient. This theorem
characterizes the observable stores for the full kernel under the assumption
that `p_ptr ≠ exp_avg_ptr` (no kernel callers rely on aliasing those
buffers). -/

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
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- Compute-facing correctness for the `exp_avg` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_exp_avg_compute_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : exp_avg_ptr ≠ p_ptr) :
    ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i => expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [update_fn_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := update_fn_kernel_exp_avg_correct p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE s s' hRegions hExec i
  simpa [hActive] using h

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
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- Compute-facing correctness for the `p` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_p_compute_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [update_fn_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := update_fn_kernel_p_correct p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE s s' hRegions hExec i
  simpa [hActive] using h

/-- Full compute-facing output coverage for Python `update_fn`: the parameter
buffer `p_ptr` and momentum buffer `exp_avg_ptr` stores are both characterized
for the full `update_fn_kernel`. -/
theorem update_fn_kernel_all_outputs_compute_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i)) := by
  constructor
  · exact update_fn_kernel_p_compute_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s hRegions
  · exact update_fn_kernel_exp_avg_compute_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s (fun h => hRegions h.symm)

/-- Public Python `update_fn` summary: the full surface lowers and both
Python-observable stores of the full kernel, `p_ptr` and `exp_avg_ptr`, are
compute-correct under the disjoint-output-region side condition. -/
theorem update_fn_kernel_output_summary
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    (∃ alg, (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i))) := by
  constructor
  · exact update_fn_kernel_surface_toAlgorithm_supported p_ptr grad_ptr
      exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE
  · exact update_fn_kernel_all_outputs_compute_correct p_ptr grad_ptr
      exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE s hRegions

/-! ## Grid-level coverage (index layer)

The Python launch uses `grid = cdiv(n_elements, BLOCK_SIZE)` programs; program
`pid` handles block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` masked to
`< n_elements`. The per-program theorems above prove each block is locally
correct (for an arbitrary `s.pid`, i.e. for every program). This lemma supplies
the orthogonal *coverage / disjointness* fact: the `(program, lane)` offset
scheme `pid·BLOCK_SIZE + lane` hits every global index exactly once, so the
blocks partition the array with no cross-program collision.

This is the index-level half of whole-grid correctness. The remaining step —
merging every program's memory into one final `BlockState` — is the launch
framework's responsibility (`VeriTile.Triton.Launch`) and is not asserted here.
The statement is generic in `BLOCK_SIZE`/`k`; it is the cross-program
companion to `BlockState.linearOffset` and could be hoisted to
`Semantics.Offset`. -/
theorem grid_offset_covers_exactly_once
    (BLOCK_SIZE : Nat) (hB : 0 < BLOCK_SIZE) (k : Nat) :
    ∃! pi : Nat × Fin BLOCK_SIZE, pi.1 * BLOCK_SIZE + pi.2.val = k := by
  refine ⟨(k / BLOCK_SIZE, ⟨k % BLOCK_SIZE, Nat.mod_lt k hB⟩), ?_, ?_⟩
  · show k / BLOCK_SIZE * BLOCK_SIZE + k % BLOCK_SIZE = k
    rw [Nat.mul_comm]; exact Nat.div_add_mod k BLOCK_SIZE
  · rintro ⟨p, i⟩ heq
    have heq2 : p * BLOCK_SIZE + i.val = k := heq
    have heq' : i.val + BLOCK_SIZE * p = k := by rw [Nat.mul_comm]; omega
    have hu := (Nat.div_mod_unique hB).mpr ⟨heq', i.isLt⟩
    exact Prod.ext hu.1.symm (Fin.ext hu.2.symm)

/-! ## Whole-grid single-memory correctness (relational)

The Python launch runs `cdiv(n_elements, BLOCK_SIZE)` programs over a 1-D grid;
program `pid` updates block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` masked to
`< n_elements`. This section composes the per-program results into ONE merged
final memory and characterizes every global cell of it.

Following the codebase convention for whole-grid statements over real kernels
(`VeriTile.Examples.FlashAttention*`), the concurrent launch is taken as a
`Kernel.GridLaunchedOrdinary` witness: the framework supplies one framed
execution per program plus pairwise write-disjointness, and `mergeFrames`
exposes the single merged memory `sFinal` without committing to an execution
order. `hFrames` pins each program's write footprint to the masked two-region
block store `adamWritesFP`; under it, every in-bounds global index of the
merged memory realizes the Lion oracle. -/

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

/-- 1-D launch grid of `m` programs. -/
abbrev adamGrid (m : Nat) : Grid := { dims := [m] }

/-- The Python launch uses `m = cdiv n_elements BLOCK_SIZE` programs, which is
exactly enough to cover every in-bounds index: `k < n_elements → k/BLOCK < m`. -/
theorem adam_cdiv_covers (n_elements BLOCK_SIZE : Nat) (hB : 0 < BLOCK_SIZE)
    {k : Nat} (hk : k < n_elements) :
    k / BLOCK_SIZE < (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE := by
  rw [Nat.div_lt_iff_lt_mul hB]
  have hmod := Nat.div_add_mod (n_elements + BLOCK_SIZE - 1) BLOCK_SIZE
  have hlt := Nat.mod_lt (n_elements + BLOCK_SIZE - 1) hB
  have hcomm : (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE
      = BLOCK_SIZE * ((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE) := Nat.mul_comm _ _
  omega

/-- Program index owning global cell `k`: `pid = k / BLOCK_SIZE`, valid as long
as the grid has at least `m` programs covering `k`. -/
def adamOwner {m : Nat} (BLOCK_SIZE : Nat) (k : Nat)
    (hcov : k / BLOCK_SIZE < m) : GridIndex (adamGrid m) :=
  fun ax => ⟨k / BLOCK_SIZE, by
    have h0 : ax.val = 0 := by
      have h := ax.isLt
      have hr : (adamGrid m).rank = 1 := rfl
      omega
    simpa [adamGrid, Grid.dim, h0] using hcov⟩

@[simp] theorem adamOwner_pid {m : Nat} (BLOCK_SIZE : Nat) (k : Nat)
    (hcov : k / BLOCK_SIZE < m) (s : BlockState) :
    (s.withGridIndex (adamOwner BLOCK_SIZE k hcov)).pid = k / BLOCK_SIZE := by
  rw [BlockState.withGridIndex_pid _ _ (by simp [Grid.rank])]
  rfl

/-- The owning program's block offset for global cell `k` is `k` itself:
`(k/BLOCK_SIZE)·BLOCK_SIZE + k%BLOCK_SIZE = k`. -/
theorem adamOwner_linearOffset {m : Nat} (BLOCK_SIZE : Nat) (hB : 0 < BLOCK_SIZE)
    (k : Nat) (hcov : k / BLOCK_SIZE < m) (s : BlockState) :
    linearOffset (s.withGridIndex (adamOwner BLOCK_SIZE k hcov)) BLOCK_SIZE
        (⟨k % BLOCK_SIZE, Nat.mod_lt k hB⟩ : Fin BLOCK_SIZE) = k := by
  show (s.withGridIndex (adamOwner BLOCK_SIZE k hcov)).pid * BLOCK_SIZE
      + (k % BLOCK_SIZE) = k
  rw [adamOwner_pid]
  have hmod := Nat.div_add_mod k BLOCK_SIZE
  have hcomm : k / BLOCK_SIZE * BLOCK_SIZE = BLOCK_SIZE * (k / BLOCK_SIZE) :=
    Nat.mul_comm _ _
  omega

/-- Whole-grid single-memory correctness for Python `update_fn`.

Given a disjoint ordinary launch (`hLaunch`) of `update_fn_kernel` over an
`m`-program 1-D grid that covers every in-bounds index (`hcover`), with each
program's write footprint pinned to the masked two-region block store
(`hFrames`), the single merged final memory `sFinal` realizes the Lion oracle
at every global index `k < n_elements`: `p_ptr` holds `lionParam` and
`exp_avg_ptr` holds `lionMomentum`, applied to the originally-loaded values. -/
theorem update_fn_kernel_grid_merged_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE m : Nat)
    (hB : 0 < BLOCK_SIZE) (hRegions : p_ptr ≠ exp_avg_ptr)
    (s sFinal : BlockState)
    (hcover : ∀ k, k < n_elements → k / BLOCK_SIZE < m)
    (hLaunch : Kernel.GridLaunchedOrdinary
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel (adamGrid m) s sFinal)
    (hFrames : ∀ idx : GridIndex (adamGrid m),
      (hLaunch.frames idx).writes =
        adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE (s.withGridIndex idx)) :
    ∀ k, k < n_elements →
      sFinal.readMem p_ptr k =
        TiledOptimizer.lionParam (s.readMem p_ptr k)
          (s.readMem exp_avg_ptr k) (s.readMem grad_ptr k) lr wd beta1 ∧
      sFinal.readMem exp_avg_ptr k =
        TiledOptimizer.lionMomentum (s.readMem exp_avg_ptr k)
          (s.readMem grad_ptr k) beta2 := by
  intro k hk
  -- The owning program and its lane for global cell k.
  set owner : GridIndex (adamGrid m) := adamOwner BLOCK_SIZE k (hcover k hk) with hOwner
  set lane : Fin BLOCK_SIZE := ⟨k % BLOCK_SIZE, Nat.mod_lt k hB⟩ with hLane
  have hoff : linearOffset (s.withGridIndex owner) BLOCK_SIZE lane = k :=
    adamOwner_linearOffset BLOCK_SIZE hB k (hcover k hk) s
  -- `s.withGridIndex owner` reads the same memory as `s`.
  have hmemEq : (s.withGridIndex owner).mem = s.mem := BlockState.withPids_mem _ _
  have hread : ∀ (X : RegionName) (o : Nat),
      (s.withGridIndex owner).readMem X o = s.readMem X o := by
    intro X o; unfold BlockState.readMem; rw [hmemEq]
  -- The owner frame writes both cells (p_ptr, k) and (exp_avg_ptr, k).
  have hWriteP : (hLaunch.frames owner).writes (p_ptr, k) := by
    rw [hFrames owner]
    refine Or.inl ?_
    refine ⟨rfl, (lane, PUnit.unit), ?_, ?_⟩
    · show (s.withGridIndex owner).pid * BLOCK_SIZE + lane.val < n_elements
      rw [show (s.withGridIndex owner).pid * BLOCK_SIZE + lane.val = k from hoff]
      exact hk
    · show (s.withGridIndex owner).pid * BLOCK_SIZE + lane.val = k
      exact hoff
  have hWriteE : (hLaunch.frames owner).writes (exp_avg_ptr, k) := by
    rw [hFrames owner]
    refine Or.inr ?_
    refine ⟨rfl, (lane, PUnit.unit), ?_, ?_⟩
    · show (s.withGridIndex owner).pid * BLOCK_SIZE + lane.val < n_elements
      rw [show (s.withGridIndex owner).pid * BLOCK_SIZE + lane.val = k from hoff]
      exact hk
    · show (s.withGridIndex owner).pid * BLOCK_SIZE + lane.val = k
      exact hoff
  -- Per-program correctness applied at the owner program.
  have hPcorr := update_fn_kernel_p_correct p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE (s.withGridIndex owner)
    (hLaunch.frames owner).final hRegions (hLaunch.frames owner).h_exec lane
  have hEcorr := update_fn_kernel_exp_avg_correct p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE (s.withGridIndex owner)
    (hLaunch.frames owner).final (fun h => hRegions h.symm)
    (hLaunch.frames owner).h_exec lane
  rw [hoff] at hPcorr hEcorr
  rw [if_pos hk] at hPcorr hEcorr
  constructor
  · -- p_ptr
    have hobs := hLaunch.observeOrdinaryCell owner p_ptr k hWriteP
    have hsf : sFinal.readMem p_ptr k =
        (hLaunch.frames owner).final.readMem p_ptr k := by
      unfold BlockState.readMem; rw [hobs]
    rw [hsf, hPcorr, pFullSpec, hoff]
    simp only [hread]
  · -- exp_avg_ptr
    have hobs := hLaunch.observeOrdinaryCell owner exp_avg_ptr k hWriteE
    have hsf : sFinal.readMem exp_avg_ptr k =
        (hLaunch.frames owner).final.readMem exp_avg_ptr k := by
      unfold BlockState.readMem; rw [hobs]
    rw [hsf, hEcorr, expAvgFullSpec, hoff]
    simp only [hread]

/-- Reusable packaging of whole-grid correctness as `Kernel.LaunchCorrect`: both
Python-observable outputs of `update_fn` realize the Lion oracle in the single
merged memory. `p_ptr` realizes `lionParam` and `exp_avg_ptr` realizes
`lionMomentum` at every in-bounds global index, for any disjoint launch whose
per-program footprints match `adamWritesFP`. -/
theorem update_fn_kernel_launchCorrect
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE m : Nat)
    (hB : 0 < BLOCK_SIZE) (hRegions : p_ptr ≠ exp_avg_ptr)
    (s : BlockState)
    (hcover : ∀ k, k < n_elements → k / BLOCK_SIZE < m) :
    Kernel.LaunchCorrect
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel (adamGrid m) s
      (fun idx => adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE
        (s.withGridIndex idx))
      (fun k : Nat => if k < n_elements then some (p_ptr, k) else none)
      (fun k => TiledOptimizer.lionParam (s.readMem p_ptr k)
        (s.readMem exp_avg_ptr k) (s.readMem grad_ptr k) lr wd beta1) ∧
    Kernel.LaunchCorrect
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel (adamGrid m) s
      (fun idx => adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE
        (s.withGridIndex idx))
      (fun k : Nat => if k < n_elements then some (exp_avg_ptr, k) else none)
      (fun k => TiledOptimizer.lionMomentum (s.readMem exp_avg_ptr k)
        (s.readMem grad_ptr k) beta2) := by
  refine ⟨?_, ?_⟩
  · intro sFinal hL hFrames k addr hwrite
    dsimp only at hwrite
    by_cases hk : k < n_elements
    · rw [if_pos hk, Option.some.injEq] at hwrite
      subst hwrite
      exact (update_fn_kernel_grid_merged_correct p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE m hB hRegions s sFinal
        hcover hL hFrames k hk).1
    · rw [if_neg hk] at hwrite; exact absurd hwrite (by simp)
  · intro sFinal hL hFrames k addr hwrite
    dsimp only at hwrite
    by_cases hk : k < n_elements
    · rw [if_pos hk, Option.some.injEq] at hwrite
      subst hwrite
      exact (update_fn_kernel_grid_merged_correct p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE m hB hRegions s sFinal
        hcover hL hFrames k hk).2
    · rw [if_neg hk] at hwrite; exact absurd hwrite (by simp)

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton
