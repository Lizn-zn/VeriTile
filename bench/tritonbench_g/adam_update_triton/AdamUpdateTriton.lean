import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Optimizer

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

/-- Proof-oriented `exp_avg` update slice of `adam_update_triton.py`'s
`update_fn_kernel`.

The full surface above covers the `p` update. This proof slice focuses on the
per-element momentum update and masked `exp_avg` store. -/
def update_fn_kernel_exp_avg_slice
    (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  diff = exp_avg - grad
  exp_avg = diff * $(beta2) + grad
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}

noncomputable def expAvgSpec
    (s : BlockState) (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) -
      s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) *
    beta2 + s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)

/-- Algorithm-layer correctness for the `exp_avg` update slice. -/
theorem update_fn_kernel_exp_avg_slice_correct
    (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (update_fn_kernel_exp_avg_slice grad_ptr exp_avg_ptr
        beta2 n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) =
        if linearOffset s BLOCK_SIZE i < n_elements then
          expAvgSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i
        else s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel_exp_avg_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
          NumericDType.sub, ComparableDType.lt] at hExec
    subst s'
    simp only [linearOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · simp [hi, expAvgSpec, linearOffset]
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `exp_avg` update slice. -/
theorem update_fn_kernel_exp_avg_slice_compute_correct
    (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := update_fn_kernel_exp_avg_slice grad_ptr exp_avg_ptr
        beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i => expAvgSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [update_fn_kernel_exp_avg_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := update_fn_kernel_exp_avg_slice_correct grad_ptr exp_avg_ptr beta2
    n_elements BLOCK_SIZE s s' hExec i
  simpa [hActive] using h

/-! ## Full-kernel output correctness

The Python test runs the full `update_fn_kernel` which writes both `p` and
`exp_avg`. Per #139's audit, slice proofs are insufficient. This theorem
characterizes the observable stores for the full kernel under the assumption
that `p_ptr ≠ exp_avg_ptr` (no kernel callers rely on aliasing those
buffers). -/

noncomputable def expAvgFullSpec
    (s : BlockState) (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) -
      s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) *
    beta2 + s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)

noncomputable def pFullSpec
    (s : BlockState) (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  by
    classical
    let off := linearOffset s BLOCK_SIZE i
    let pTile : TileCarrier TileDType.real := some (s.readMem p_ptr off)
    let gradTile : TileCarrier TileDType.real := some (s.readMem grad_ptr off)
    let expAvgTile : TileCarrier TileDType.real := some (s.readMem exp_avg_ptr off)
    let diff : TileCarrier TileDType.real :=
      Option.map₂ (fun x y => x - y) expAvgTile gradTile
    let update : TileCarrier TileDType.real := Option.map₂ (fun x y => x + y)
      (Option.map (fun x => x * beta1) diff) gradTile
    exact
      WithBot.unbotD 0
        (Option.map₂ (fun x y => x + y)
          (Option.map (fun x => x * (1 - lr * wd)) pTile)
          (if update = some 0 then
            some 0
          else if @LT.lt (TileCarrier TileDType.real) WithBot.instPreorder.toLT
              (some 0) update then
            some (0.0 - lr)
          else
            some lr))

/-! ### Bridge to the reusable Lion oracle (`VeriTile.Triton.Math.Optimizer`)

The transcription specs above mirror the kernel's realized arithmetic
(`β·(m − g) + g` momentum, branch-based sign step). These bridges connect them
to the layout-free Lion definitions so the public correctness statement is
against the standard optimizer oracle, not a kernel-shaped restatement. -/

/-- The `exp_avg` transcription spec is the Lion momentum EMA. -/
theorem expAvgFullSpec_eq_lionMomentum
    (s : BlockState) (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) :
    expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i =
      TiledOptimizer.lionMomentum
        (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) beta2 := by
  unfold expAvgFullSpec TiledOptimizer.lionMomentum
  ring

/-- The `p` transcription spec is the Lion decoupled-weight-decay parameter
update. -/
theorem pFullSpec_eq_lionParam
    (s : BlockState) (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) :
    pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i =
      TiledOptimizer.lionParam
        (s.readMem p_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) lr wd beta1 := by
  classical
  unfold pFullSpec TiledOptimizer.lionParam TiledOptimizer.lionUpdateDir
  simp only [Option.map₂, Option.map, Option.bind]
  set m := s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) with hm
  set g := s.readMem grad_ptr (linearOffset s BLOCK_SIZE i) with hg
  set p := s.readMem p_ptr (linearOffset s BLOCK_SIZE i) with hp
  -- the kernel's factored direction `(m − g)·β₁ + g` is the oracle direction.
  have heq : (m - g) * beta1 + g = beta1 * m + (1 - beta1) * g := by ring
  split_ifs with c1 c2
  · -- direction = 0
    have hd : beta1 * m + (1 - beta1) * g = 0 := heq ▸ Option.some_inj.mp c1
    simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, hd, Real.sign_zero]
    ring
  · -- direction > 0
    have hd : (0 : ℝ) < beta1 * m + (1 - beta1) * g := heq ▸ WithBot.coe_lt_coe.mp c2
    simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, Real.sign_of_pos hd]
    norm_num
    ring
  · -- direction < 0
    have hne : (m - g) * beta1 + g ≠ 0 := fun hc => c1 (congrArg some hc)
    have hnp : ¬ (0 : ℝ) < (m - g) * beta1 + g := fun hc => c2 (WithBot.coe_lt_coe.mpr hc)
    have hd : beta1 * m + (1 - beta1) * g < 0 :=
      heq ▸ lt_of_le_of_ne (not_lt.mp hnp) hne
    simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, Real.sign_of_neg hd]
    ring

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
    · simp [hi, expAvgFullSpec, linearOffset]
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
    · unfold pFullSpec
      simp only [linearOffset, hi, if_true, Option.map, Option.map₂,
        BlockState.pid_eq]
      rfl
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

/-- Oracle-facing summary: both Python-observable stores of `update_fn` are
compute-correct against the reusable Lion optimizer oracle in
`VeriTile.Triton.Math.Optimizer` — `p_ptr` realizes `lionParam` and
`exp_avg_ptr` realizes `lionMomentum` — under the disjoint-output-region side
condition. -/
theorem update_fn_kernel_lion_outputs_compute_correct
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
      (expected := fun i => TiledOptimizer.lionParam
        (s.readMem p_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) lr wd beta1)) ∧
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i => TiledOptimizer.lionMomentum
        (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) beta2)) := by
  have h := update_fn_kernel_all_outputs_compute_correct p_ptr grad_ptr
    exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE s hRegions
  have hp : (fun i => pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1
        BLOCK_SIZE i) =
      (fun i => TiledOptimizer.lionParam
        (s.readMem p_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) lr wd beta1) := by
    funext i
    exact pFullSpec_eq_lionParam s p_ptr grad_ptr exp_avg_ptr lr wd beta1
      BLOCK_SIZE i
  have he : (fun i => expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i) =
      (fun i => TiledOptimizer.lionMomentum
        (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
        (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) beta2) := by
    funext i
    exact expAvgFullSpec_eq_lionMomentum s grad_ptr exp_avg_ptr beta2
      BLOCK_SIZE i
  exact ⟨hp ▸ h.1, he ▸ h.2⟩

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton
