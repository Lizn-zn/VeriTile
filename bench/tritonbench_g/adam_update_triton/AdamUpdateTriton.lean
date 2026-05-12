import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Surface transcription of `adam_update_triton.py`'s `update_fn_kernel`.

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

The full kernel also updates `p` through the Python/Triton idiom
`update_sign * can_update`, where `can_update` is boolean. The current DSL
does not coerce bool tiles as numeric masks in arithmetic, so this first slice
captures the per-element momentum update and masked store surface. -/
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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * BLOCK_SIZE + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply Fin.ext
      exact Nat.add_left_cancel h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel_exp_avg_slice, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
          NumericDType.sub, ComparableDType.lt] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
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

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton
