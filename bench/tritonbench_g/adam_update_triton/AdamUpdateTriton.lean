import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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

def outOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

noncomputable def expAvgSpec
    (s : BlockState) (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  (s.readMem exp_avg_ptr (outOffset s BLOCK_SIZE i) -
      s.readMem grad_ptr (outOffset s BLOCK_SIZE i)) *
    beta2 + s.readMem grad_ptr (outOffset s BLOCK_SIZE i)

/-- Algorithm-layer correctness for the `exp_avg` update slice. -/
theorem update_fn_kernel_exp_avg_slice_correct
    (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (update_fn_kernel_exp_avg_slice grad_ptr exp_avg_ptr
        beta2 n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem exp_avg_ptr (outOffset s BLOCK_SIZE i) =
        if outOffset s BLOCK_SIZE i < n_elements then
          expAvgSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i
        else s.readMem exp_avg_ptr (outOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel_exp_avg_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
          NumericDType.sub, ComparableDType.lt] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · simp [hi, expAvgSpec, outOffset]
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
        (fun i : Fin BLOCK_SIZE => outOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, outOffset s BLOCK_SIZE i)))
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
  (s.readMem exp_avg_ptr (outOffset s BLOCK_SIZE i) -
      s.readMem grad_ptr (outOffset s BLOCK_SIZE i)) *
    beta2 + s.readMem grad_ptr (outOffset s BLOCK_SIZE i)

noncomputable def pFullSpec
    (s : BlockState) (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  by
    classical
    let off := outOffset s BLOCK_SIZE i
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
      s'.readMem exp_avg_ptr (outOffset s BLOCK_SIZE i) =
        if outOffset s BLOCK_SIZE i < n_elements then
          expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i
        else s.readMem exp_avg_ptr (outOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.gt, ComparableDType.ne] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          p_ptr _ _ _ _ _ _ _ hRegions]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · simp [hi, expAvgFullSpec, outOffset]
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
        (fun i : Fin BLOCK_SIZE => outOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, outOffset s BLOCK_SIZE i)))
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
      s'.readMem p_ptr (outOffset s BLOCK_SIZE i) =
        if outOffset s BLOCK_SIZE i < n_elements then
          pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i
        else s.readMem p_ptr (outOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.gt, ComparableDType.ne] at hExec
    subst s'
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          exp_avg_ptr _ _ _ _ _ _ _ hRegions]
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · unfold pFullSpec
      simp only [outOffset, hi, if_true, Option.map, Option.map₂,
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
        (fun i : Fin BLOCK_SIZE => outOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, outOffset s BLOCK_SIZE i)))
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
        (fun i : Fin BLOCK_SIZE => outOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, outOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => outOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, outOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i)) := by
  constructor
  · exact update_fn_kernel_p_compute_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s hRegions
  · exact update_fn_kernel_exp_avg_compute_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s (fun h => hRegions h.symm)

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton
