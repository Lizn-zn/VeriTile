import VeriTile.Triton.DSL

/-!
AdamUpdateGridLaunch: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels
open VeriTile Triton

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

def originalKernel (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let p_ptr : RegionName := ⟨"p"⟩
  let grad_ptr : RegionName := ⟨"grad"⟩
  let exp_avg_ptr : RegionName := ⟨"exp_avg"⟩
  triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask, dtype=tl.float32)
  grad = tl.load(offset_grad_ptr, mask=mask, dtype=tl.float32)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask, dtype=tl.float32)
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

def optimizedKernel (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let p_ptr : RegionName := ⟨"p"⟩
  let grad_ptr : RegionName := ⟨"grad"⟩
  let exp_avg_ptr : RegionName := ⟨"exp_avg"⟩
  triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask, dtype=tl.float32)
  grad = tl.load(offset_grad_ptr, mask=mask, dtype=tl.float32)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask, dtype=tl.float32)
  p = p * (1 - $((lr : ℝ)) * $((wd : ℝ)))
  diff = exp_avg - grad
  update = diff * $(beta1) + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -$((lr : ℝ)), $((lr : ℝ)))
  p = p + update_sign * can_update
  exp_avg = grad + diff * $(beta2)
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}

end VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels
