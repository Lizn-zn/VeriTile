import VeriTile.Examples.Common
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Optimizer
import VeriTile.Triton.Launch.Composition
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-!
# `adam_update_triton` — proof architecture

The Python kernel is named "adam" but implements **Lion** (Chen et al.): a
sign-based parameter update driven by a single momentum buffer, with decoupled
weight decay. Per launch, program `pid` updates the block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of two buffers — `p_ptr` (parameters) and
`exp_avg_ptr` (momentum) — masked to `< n_elements`.

The verification is layered. Each layer is built from the one below it; the
"single program" layers are then composed into a whole-grid statement. A
second, complementary headline — the single-program **KernelIO `⊨`
specification** over flat pointer memory — sits beside the grid-launch stack
and reuses its per-program core:

```
adam_update_correctness                              ← KernelIO ⊨ HEADLINE
  · adamIO ⊨ (lionParam, lionMomentum): the masked in-place Hoare triple
    (out1 = in1 = p_ptr, out2 = in3 = exp_avg_ptr; MaskedKernelIO₃ₓ₂)
    transported to flat pointer memory, ∀ base pointers.
  ├─ update_fn_kernel_flattenOk / update_fn_kernel_traceSafe   (bridge side
  │     conditions: fragment membership + per-execution safety walk)
  └─ update_fn_kernel_region_run                     ← region-model triple
      ├─ adam_exec_isSome                                  (termination)
      ├─ update_fn_kernel_p_correct / _exp_avg_correct     (values)
      └─ adam_writeWithin (via adamWritesFP_frame_cell)    (frame)

update_fn_kernel_launchCorrect                       ← FINAL THEOREM
  · unconditional; packages the result as two reusable `Kernel.LaunchCorrect`
    instances (p_ptr ⇒ lionParam, exp_avg_ptr ⇒ lionMomentum).
  · witnessed by `adamLaunch`; correctness from `…_unconditional`.
  │
  ├─ adamLaunch : GridLaunchedOrdinary                ← the CONSTRUCTED launch
  │   · frames + pairwise write-disjointness + merged memory (h_final := rfl).
  │   ├─ adamFrame  (one ExecFrame per program)
  │   │   ├─ adam_exec_isSome      — progress: `exec` always returns `some`.
  │   │   └─ adam_writeWithin      — each program writes ONLY its two masked
  │   │       blocks; everything else is untouched.  ⟵ the hard footprint proof
  │   │       (writeMem foldl `WriteWithin` helpers + 3-leg `writeWithin_trans`:
  │   │        setReg base → p_ptr store → exp_avg_ptr store).
  │   └─ adamFrames_disjoint       — distinct programs write distinct cells.
  │       ├─ block_offset_inj      — pid·B+i = pid'·B+j (i,j<B) ⇒ pid = pid'.
  │       └─ adamGrid_idx_ext      — a 1-D grid index is its axis-0 value.
  │
  └─ update_fn_kernel_grid_correct_unconditional      ← raw `∀ k < n` form
      · discharges the workhorse's launch/footprint/coverage hypotheses using
        `adamLaunch`, `adam_cdiv_covers`, and `fun idx => rfl`.
      │
      └─ update_fn_kernel_grid_merged_correct          ← RELATIONAL workhorse
          · GIVEN a disjoint launch + per-program footprints = `adamWritesFP`,
            every in-bounds global cell of the merged memory = the Lion oracle.
          ├─ update_fn_kernel_p_correct / _exp_avg_correct   (per-program)
          │     · one program's masked store realizes pFullSpec / expAvgFullSpec,
          │       which are thin wrappers over `lionParam` / `lionMomentum`
          │       (the math lives once in `VeriTile.Triton.Math.Optimizer`).
          ├─ GridLaunchedOrdinary.observeOrdinaryCell        (read merged cell)
          └─ adamOwner / adamOwner_linearOffset              (cell k's owner:
                pid = k / B, whose lane k % B addresses exactly k)
```

## Why the layers

* **Single-program correctness** (`…_p_correct`, `…_exp_avg_correct`) is the
  arithmetic core: execute the kernel on one `BlockState` and read back the two
  masked stores. The specs `pFullSpec` / `expAvgFullSpec` are *oracle wrappers*
  — they apply the layout-free Lion oracle to the values this lane loads, so the
  formula is stated once, in `Math.Optimizer`, and never duplicated here.

* **Relational whole-grid** (`…_grid_merged_correct`) composes programs. It does
  NOT execute the grid; the concurrent launch is supplied as a
  `GridLaunchedOrdinary` witness (the codebase convention for real kernels). It
  characterizes the *single merged memory* `sFinal` at every global index `k`,
  using coverage (each `k` is owned by exactly one program, `pid = k / BLOCK`)
  and disjointness (no two programs write the same cell).

* **Unconditional** (`…_grid_correct_unconditional`, `…_launchCorrect`)
  *constructs* the launch witness — proving the grid execution succeeds
  (`adam_exec_isSome`), bounds each program's writes (`adam_writeWithin`), and
  shows programs do not collide (`adamFrames_disjoint`) — so the relational
  hypotheses disappear. The final theorem has only the basic side conditions
  `0 < BLOCK_SIZE` and `p_ptr ≠ exp_avg_ptr`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled; the grid is composed by `mergeFrames` rather than executed
concurrently. These are the standard VeriTile conventions, shared with the
FlashAttention examples.
-/

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch

open VeriTile.Examples
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

/-! ## Single-program value core

The two algorithm-layer theorems below execute one program of the full
`update_fn_kernel` and characterize BOTH observable stores lane by lane —
active lanes hold the Lion oracles, masked-off lanes are unchanged — under
the assumption `p_ptr ≠ exp_avg_ptr` (no kernel callers alias those
buffers). They are the shared value core of the file's two headlines: the
whole-grid `update_fn_kernel_launchCorrect` and the KernelIO `⊨`
specification. -/

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

/-- Frame write-footprint: a single program of `update_fn_kernel` only writes
the two masked blocks `adamWritesFP` (its `p_ptr` and `exp_avg_ptr` lanes). All
other memory — other regions, and out-of-block / masked-off cells — is
untouched. This discharges the `h_writeWithin` field of each grid frame. -/
theorem adam_writeWithin
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr) (s : BlockState) :
    Kernel.ExecWritesWithin
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel s
      (adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE s) := by
  intro s' hExec
  simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
  have hmemEq : (s.withGridIndex owner).mem = s.mem := BlockState.withGridIndex_mem _ _
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

/-! ## Unconditional whole-grid launch

The relational results above take the launch witness as a hypothesis. This
section *constructs* it: progress (`exec` always succeeds), the per-program
frame, and pairwise disjointness, yielding an unconditional whole-grid theorem
with no launch/footprint side hypotheses. -/

/-- Progress: `update_fn_kernel` always executes to a defined state. -/
theorem adam_exec_isSome
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) (s : BlockState) :
    (exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE) s).isSome := by
  simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, ComparableDType.gt, ComparableDType.ne]

/-- Distinct program blocks address distinct cells: `p₁·B + i = p₂·B + j` with
`i, j < B` forces `p₁ = p₂`. -/
theorem block_offset_inj (B : Nat) (hB : 0 < B) {p1 p2 i j : Nat}
    (hi : i < B) (hj : j < B) (h : p1 * B + i = p2 * B + j) : p1 = p2 := by
  have hd : (p1 * B + i) / B = (p2 * B + j) / B := by rw [h]
  rwa [Nat.add_comm (p1 * B) i, Nat.mul_comm p1 B, Nat.add_mul_div_left _ _ hB,
       Nat.div_eq_of_lt hi, Nat.zero_add, Nat.add_comm (p2 * B) j,
       Nat.mul_comm p2 B, Nat.add_mul_div_left _ _ hB, Nat.div_eq_of_lt hj,
       Nat.zero_add] at hd

/-- The per-program execution frame for `update_fn_kernel`. -/
noncomputable def adamFrame
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE m : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr) (s : BlockState)
    (idx : GridIndex (adamGrid m)) :
    Kernel.ExecFrame
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel
      (s.withGridIndex idx) :=
  { final := (exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE) (s.withGridIndex idx)).get
        (adam_exec_isSome p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
          n_elements BLOCK_SIZE (s.withGridIndex idx))
    writes := adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE
        (s.withGridIndex idx)
    h_exec := (Option.some_get _).symm
    h_writeWithin :=
      adam_writeWithin p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
        n_elements BLOCK_SIZE hRegions (s.withGridIndex idx) _
        (Option.some_get _).symm }

/-- A 1-D grid index is determined by its single axis-0 value. -/
theorem adamGrid_idx_ext {m : Nat} {idx₁ idx₂ : GridIndex (adamGrid m)}
    (h : (idx₁ ⟨0, by simp [Grid.rank]⟩) = (idx₂ ⟨0, by simp [Grid.rank]⟩)) :
    idx₁ = idx₂ := by
  funext a
  have ha : a = ⟨0, by simp [Grid.rank]⟩ := by
    apply Fin.ext
    have hlt := a.isLt
    have hr : (adamGrid m).rank = 1 := rfl
    omega
  subst ha; exact h

/-- Two distinct programs write disjoint cells. -/
theorem adamWritesFP_disjoint
    (p_ptr exp_avg_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr) (hB : 0 < BLOCK_SIZE)
    {s₁ s₂ : BlockState} (hpid : s₁.pid ≠ s₂.pid) :
    WriteFootprint.disjoint
      (adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE s₁)
      (adamWritesFP p_ptr exp_avg_ptr n_elements BLOCK_SIZE s₂) := by
  rintro ⟨r, o⟩ h₁ h₂
  unfold adamWritesFP WriteFootprint.union WriteFootprint.activeTileImage at h₁ h₂
  rcases h₁ with ⟨hr₁, i₁, _, ho₁⟩ | ⟨hr₁, i₁, _, ho₁⟩ <;>
    rcases h₂ with ⟨hr₂, i₂, _, ho₂⟩ | ⟨hr₂, i₂, _, ho₂⟩
  · exact hpid (block_offset_inj BLOCK_SIZE hB i₁.1.isLt i₂.1.isLt (ho₁.trans ho₂.symm))
  · exact hRegions (hr₁.symm.trans hr₂)
  · exact hRegions (hr₂.symm.trans hr₁)
  · exact hpid (block_offset_inj BLOCK_SIZE hB i₁.1.isLt i₂.1.isLt (ho₁.trans ho₂.symm))

/-- Pairwise write-disjointness of all program frames. -/
theorem adamFrames_disjoint
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE m : Nat)
    (hB : 0 < BLOCK_SIZE) (hRegions : p_ptr ≠ exp_avg_ptr) (s : BlockState) :
    Kernel.GridWritesDisjoint
      (fun idx => adamFrame p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
        n_elements BLOCK_SIZE m hRegions s idx) := by
  intro idx₁ idx₂ hne
  apply adamWritesFP_disjoint p_ptr exp_avg_ptr n_elements BLOCK_SIZE hRegions hB
  -- distinct grid indices have distinct program ids
  rw [BlockState.withGridIndex_pid _ _ (by simp [Grid.rank]),
      BlockState.withGridIndex_pid _ _ (by simp [Grid.rank])]
  intro hval
  exact hne (adamGrid_idx_ext (Fin.ext hval))

/-- The constructed disjoint ordinary launch of `update_fn_kernel`. -/
noncomputable def adamLaunch
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE m : Nat)
    (hB : 0 < BLOCK_SIZE) (hRegions : p_ptr ≠ exp_avg_ptr) (s : BlockState) :
    Kernel.GridLaunchedOrdinary
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel (adamGrid m) s
      (Kernel.mergeFrames (adamGrid m) s
        (fun idx => adamFrame p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
          n_elements BLOCK_SIZE m hRegions s idx)) :=
  { frames := fun idx => adamFrame p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
      n_elements BLOCK_SIZE m hRegions s idx
    h_disjoint := adamFrames_disjoint p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE m hB hRegions s
    h_final := rfl }

/-- **Unconditional whole-grid single-memory correctness.** Launching
`update_fn_kernel` over the `cdiv n_elements BLOCK_SIZE`-program grid produces a
merged final memory in which every in-bounds global index realizes the Lion
oracle — `p_ptr` holds `lionParam`, `exp_avg_ptr` holds `lionMomentum` — with no
launch or footprint side hypotheses. -/
theorem update_fn_kernel_grid_correct_unconditional
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (hRegions : p_ptr ≠ exp_avg_ptr) (s : BlockState) :
    let m := (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE
    let sFinal := Kernel.mergeFrames (adamGrid m) s
      (fun idx => adamFrame p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
        n_elements BLOCK_SIZE m hRegions s idx)
    ∀ k, k < n_elements →
      sFinal.readMem p_ptr k =
        TiledOptimizer.lionParam (s.readMem p_ptr k)
          (s.readMem exp_avg_ptr k) (s.readMem grad_ptr k) lr wd beta1 ∧
      sFinal.readMem exp_avg_ptr k =
        TiledOptimizer.lionMomentum (s.readMem exp_avg_ptr k)
          (s.readMem grad_ptr k) beta2 := by
  intro m sFinal k hk
  exact update_fn_kernel_grid_merged_correct p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE m hB hRegions s sFinal
    (fun k hk => adam_cdiv_covers n_elements BLOCK_SIZE hB hk)
    (adamLaunch p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
      n_elements BLOCK_SIZE m hB hRegions s)
    (fun idx => rfl) k hk

/-- **Unconditional `LaunchCorrect` packaging.** Both Python-observable outputs
of `update_fn` satisfy the reusable `Kernel.LaunchCorrect` contract outright (no
launch/footprint hypotheses): launching `update_fn_kernel` over the
`cdiv n_elements BLOCK_SIZE`-program grid yields a merged memory in which
`p_ptr` realizes `lionParam` and `exp_avg_ptr` realizes `lionMomentum` at every
in-bounds global index. The launch witness is the constructed `adamLaunch`. -/
theorem update_fn_kernel_launchCorrect
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (hRegions : p_ptr ≠ exp_avg_ptr) (s : BlockState) :
    Kernel.LaunchCorrect
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel
      (adamGrid ((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE)) s
      (fun k : Nat => if k < n_elements then some (p_ptr, k) else none)
      (fun k => TiledOptimizer.lionParam (s.readMem p_ptr k)
        (s.readMem exp_avg_ptr k) (s.readMem grad_ptr k) lr wd beta1) ∧
    Kernel.LaunchCorrect
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel
      (adamGrid ((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE)) s
      (fun k : Nat => if k < n_elements then some (exp_avg_ptr, k) else none)
      (fun k => TiledOptimizer.lionMomentum (s.readMem exp_avg_ptr k)
        (s.readMem grad_ptr k) beta2) := by
  set m := (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE with hm
  have hcorrect := update_fn_kernel_grid_correct_unconditional
    p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE hB hRegions s
  refine ⟨⟨_, adamLaunch p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
      n_elements BLOCK_SIZE m hB hRegions s, ?_⟩,
    ⟨_, adamLaunch p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
      n_elements BLOCK_SIZE m hB hRegions s, ?_⟩⟩
  · intro k addr hwrite
    dsimp only at hwrite
    by_cases hk : k < n_elements
    · rw [if_pos hk, Option.some.injEq] at hwrite
      subst hwrite
      exact (hcorrect k hk).1
    · rw [if_neg hk] at hwrite; exact absurd hwrite (by simp)
  · intro k addr hwrite
    dsimp only at hwrite
    by_cases hk : k < n_elements
    · rw [if_pos hk, Option.some.injEq] at hwrite
      subst hwrite
      exact (hcorrect k hk).2
    · rw [if_neg hk] at hwrite; exact absurd hwrite (by simp)

/-! ## KernelIO `⊨` specification (single program, flat memory)

The grid-launch stack above characterizes the *merged whole-grid* memory in
the region model. This section states the complementary single-program
headline on the **KernelIO spec surface**: `adamIO ⊨ (lionParam,
lionMomentum)`, a full masked Hoare triple over *flat pointer memory* —
∀ disjoint base-pointer placements of the three buffers, ∀ program ids whose
active lanes are in bounds, ∀ launch states whose input windows are loaded
at the active lanes (everything else junk), the translated pointer kernel
terminates, both active output windows hold the Lion oracles, and every
other flat cell is untouched.

The kernel is **in-place**: `p_ptr` and `exp_avg_ptr` are inputs *and*
outputs. `MaskedKernelIO₃ₓ₂` supports this by decoupling the allocation
list (`bufs`, each buffer exactly once) from the argument roles, so
`out1 = in1 = p` and `out2 = in3 = exp_avg`; the triple reads the *old*
window contents into `f` and asserts the *new* ones — the standard
before/after reading.

Three obligations feed `MaskedKernelIO₃ₓ₂.Implements.intro`: `FlattenOk`
(bridge fragment membership), the per-execution `TraceSafe` walk, and the
region-model masked triple `update_fn_kernel_region_run`, whose three legs
are exactly the grid stack's per-program core (`adam_exec_isSome`,
`update_fn_kernel_p_correct`/`_exp_avg_correct`, `adam_writeWithin`). -/

open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

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

/-- Frame bridging lemma: `adam_writeWithin` pins the execution's write
footprint to `adamWritesFP` (the two active masked windows), so any cell
outside **both** active output windows — the exact frame condition shape of
`MaskedKernelIO₃ₓ₂.Implements.intro`'s `hrun` — is untouched. -/
theorem adamWritesFP_frame_cell
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr) (s s1 : BlockState)
    (hExec : exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel s = some s1)
    (r : RegionName) (o : Nat)
    (h1 : r ≠ p_ptr ∨ ∀ j : Fin BLOCK_SIZE,
      s.pid * BLOCK_SIZE + j.val < n_elements → o ≠ s.pid * BLOCK_SIZE + j.val)
    (h2 : r ≠ exp_avg_ptr ∨ ∀ j : Fin BLOCK_SIZE,
      s.pid * BLOCK_SIZE + j.val < n_elements → o ≠ s.pid * BLOCK_SIZE + j.val) :
    s1.mem r o = s.mem r o := by
  have hw := adam_writeWithin p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2
    n_elements BLOCK_SIZE hRegions s s1 hExec
  refine (hw r o ?_).symm
  rintro (⟨rfl, i, hact, ho⟩ | ⟨rfl, i, hact, ho⟩)
  · rcases h1 with hne | hno
    · exact hne rfl
    · exact hno i.1 hact ho.symm
  · rcases h2 with hne | hno
    · exact hne rfl
    · exact hno i.1 hact ho.symm

/-- **The region-model masked Hoare triple** — termination, active-lane
values of both in-place outputs, and frame off the two active output
windows, from any launch state whose three input windows are loaded at the
**active lanes only** (`pid * B + j < n`). This is the `hrun` obligation of
the `⊨` headline; its three legs are the grid stack's per-program core. -/
theorem update_fn_kernel_region_run
    (lr wd beta1 beta2 : ℝ) (n B : Nat)
    (s₀ : BlockState) (xs ys zs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.pid * B + j.val < n →
      s₀.readMem ⟨"p"⟩ (s₀.pid * B + j.val) = xs j)
    (hy : ∀ j : Fin B, s₀.pid * B + j.val < n →
      s₀.readMem ⟨"grad"⟩ (s₀.pid * B + j.val) = ys j)
    (hz : ∀ j : Fin B, s₀.pid * B + j.val < n →
      s₀.readMem ⟨"exp_avg"⟩ (s₀.pid * B + j.val) = zs j) :
    ∃ s1, exec ((update_fn_kernel ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
        lr wd beta1 beta2 n B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B, s₀.pid * B + j.val < n →
          s1.readMem ⟨"p"⟩ (s₀.pid * B + j.val)
            = TiledOptimizer.lionParam (xs j) (zs j) (ys j) lr wd beta1)
      ∧ (∀ j : Fin B, s₀.pid * B + j.val < n →
          s1.readMem ⟨"exp_avg"⟩ (s₀.pid * B + j.val)
            = TiledOptimizer.lionMomentum (zs j) (ys j) beta2)
      ∧ (∀ r o,
          (r ≠ ⟨"p"⟩ ∨ ∀ j : Fin B, s₀.pid * B + j.val < n →
            o ≠ s₀.pid * B + j.val) →
          (r ≠ ⟨"exp_avg"⟩ ∨ ∀ j : Fin B, s₀.pid * B + j.val < n →
            o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hne : (⟨"p"⟩ : RegionName) ≠ ⟨"exp_avg"⟩ := by decide
  obtain ⟨s1, hs1⟩ := Option.isSome_iff_exists.mp
    (adam_exec_isSome ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩ lr wd beta1 beta2 n B s₀)
  have hs1' : exec ((update_fn_kernel ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B).toAlgKernel) s₀ = some s1 := hs1
  refine ⟨s1, hs1', ?_, ?_, ?_⟩
  · -- p window: the in-place parameter update realizes `lionParam`.
    intro j hj
    have h := update_fn_kernel_p_correct ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B s₀ s1 hne hs1 j
    rw [show linearOffset s₀ B j = s₀.pid * B + j.val from rfl] at h
    rw [h, if_pos hj]
    unfold pFullSpec
    rw [show linearOffset s₀ B j = s₀.pid * B + j.val from rfl,
      hx j hj, hy j hj, hz j hj]
  · -- exp_avg window: the in-place momentum update realizes `lionMomentum`.
    intro j hj
    have h := update_fn_kernel_exp_avg_correct ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B s₀ s1 (fun hc => hne hc.symm) hs1 j
    rw [show linearOffset s₀ B j = s₀.pid * B + j.val from rfl] at h
    rw [h, if_pos hj]
    unfold expAvgFullSpec
    rw [show linearOffset s₀ B j = s₀.pid * B + j.val from rfl,
      hy j hj, hz j hj]
  · -- frame off the two active output windows.
    intro r o h1 h2
    exact adamWritesFP_frame_cell ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B hne s₀ s1 hs1' r o h1 h2

/-- `update_fn_kernel`'s masked in-place **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `bufs` — the allocation list: three buffers, each exactly once;
* `in1`/`in2`/`in3` — parameters, gradient, momentum (the wiring);
* `out1 = in1`, `out2 = in3` — the **in-place** roles: the kernel rewrites
  the parameter and momentum buffers it read;
* `read1..3`/`write1..2` — every window is the same block
  `[pid·B, pid·B + B)` (the launch convention
  `offsets = pid * BLOCK_SIZE + arange`);
* `mask` — program `pid`'s active lanes, `pid * B + j < n`. Inactive lanes
  (the overhang of the last partial block) carry no obligations.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
def adamIO (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := update_fn_kernel ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩ lr wd beta1 beta2 n B
  bufs := [⟨"p"⟩, ⟨"grad"⟩, ⟨"exp_avg"⟩]  -- p and exp_avg are updated in place
  in1 := ⟨"p"⟩
  in2 := ⟨"grad"⟩
  in3 := ⟨"exp_avg"⟩
  out1 := ⟨"p"⟩          -- = in1: in-place parameter update
  out2 := ⟨"exp_avg"⟩    -- = in3: in-place momentum update
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write1 := fun pid => pid * B
  write2 := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < n

/-- **The headline**: `update_fn_kernel` implements the Lion step on its
masked in-place IO signature — every active lane of the parameter buffer
ends up holding `lionParam` and of the momentum buffer `lionMomentum`,
applied to the *originally loaded* windows; every other flat cell is
untouched. Proof: `Implements.intro` assembles the region-model triple with
the bridge side conditions. -/
specification adam_update_correctness (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    adamIO lr wd beta1 beta2 n B ⊨ fun p grad expAvg =>
      (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
       fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [adamIO]) (by simp [adamIO]) ?_ ?_ ?_
  · exact update_fn_kernel_flattenOk ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B
  · intro bounds s h1 h2 h3 _ _
    exact update_fn_kernel_traceSafe ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B bounds s h1 h2 h3
  · intro s₀ xs ys zs hx hy hz
    exact update_fn_kernel_region_run lr wd beta1 beta2 n B s₀ xs ys zs hx hy hz

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof.
#axiomsClean adam_update_correctness

/- The headline's statement surface is the masked IO signature, the
audit-once Hoare-triple combinator, and the two Lion oracles — no other
project constant. -/
#stmtSurfaceSubset adam_update_correctness ⊆
  [adamIO, VeriTile.Triton.MaskedKernelIO₃ₓ₂.Implements,
   TiledOptimizer.lionParam, TiledOptimizer.lionMomentum,
   VeriTile.Triton.MaskedKernelIO₃ₓ₂.B]

end VeriTile.Bench.Examples.AdamUpdateGridLaunch
