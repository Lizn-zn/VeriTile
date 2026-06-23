import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `masked_add_cuda` — strict per-kernel correctness

`masked_add_kernel` is a masked fused multiply-add: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)`, narrows its bounds mask by the boolean
`p_mask` (`mask = mask & ~p_mask`), and for each surviving lane writes
`grad += p * alpha` back to `grad_ptr` (equivalent to
`grad.add_(p.data * (1 - p.mask), alpha)`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`masked_add_kernel[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `pid` is universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
masked_add_kernel_output_summary              ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ masked_add_kernel_compute_correct        ← ComputeCorrect over the masked store
       └─ masked_add_kernel_correct           ← algorithm-layer readback per lane
```

The active predicate `maskedAddActive` is the conjunction of the bounds mask
`offset < n_elements` and the loaded `p_mask` being `false`; the spec
`maskedAddSpec` is the in-place fused update `grad + p * alpha`. The values each
lane reads are taken directly from the initial `BlockState` memory.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The kernel reads `grad`/`p`
into registers before the in-place scatter
back to `grad_ptr`, so the masked store is correct in place; no extra
disjointness side condition is required.
-/

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda

open VeriTile.Triton

/-- Faithful transcription of `masked_add_cuda.py`'s `masked_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def masked_add_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad += p * $(alpha)
  tl.store(grad_ptr + offsets, grad, mask=mask)
}

def maskedAddOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

def maskedAddActive
    (s : BlockState) (p_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedAddOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool p_mask_ptr (maskedAddOffset s BLOCK_SIZE i) = Bool.false

instance maskedAddActiveDecidable
    (s : BlockState) (p_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold maskedAddActive
  infer_instance

noncomputable def maskedAddSpec
    (s : BlockState) (grad_ptr p_ptr : RegionName)
    (alpha : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem grad_ptr (maskedAddOffset s BLOCK_SIZE i) +
    s.readMem p_ptr (maskedAddOffset s BLOCK_SIZE i) * alpha

/-- Algorithm-layer correctness for `masked_add_kernel`. -/
theorem masked_add_kernel_correct
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (masked_add_kernel grad_ptr p_ptr p_mask_ptr
          n_elements alpha BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := maskedAddOffset s BLOCK_SIZE i
      s'.readMem grad_ptr outAddr =
        if maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i then
          maskedAddSpec s grad_ptr p_ptr alpha BLOCK_SIZE i
        else s.readMem grad_ptr outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, masked_add_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        tile_elementwise, Bool.and_eq_true] at hExec
  subst s'
  simp only [maskedAddOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hActive : maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i
  · rcases hActive with ⟨hBounds, hMask⟩
    have hCond :
        s.pids 0 * BLOCK_SIZE + i.val < n_elements ∧
          (if s.pids 0 * BLOCK_SIZE + i.val < n_elements then
            s.readMemValue .bool p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val) = Bool.false
          else
            BlockState.defaultCarrier .bool = Bool.false) := by
      simp [maskedAddOffset] at hBounds hMask
      simp [hBounds, hMask]
    have hMask' :
        s.readMemValue .bool p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val) = Bool.false := by
      simpa [maskedAddOffset] using hMask
    simp [maskedAddActive, maskedAddSpec, maskedAddOffset, hMask', hCond]
  · simp [maskedAddActive, maskedAddOffset] at hActive ⊢
    by_cases hBounds : s.pid * BLOCK_SIZE + i.val < n_elements
    · simp [hBounds] at hActive ⊢
      cases hMask : s.readMemValue .bool p_mask_ptr (s.pid * BLOCK_SIZE + i.val) <;>
        simp [hMask] at hActive ⊢
    · simp [hBounds]

/-- Compute-facing correctness for `masked_add_kernel`. -/
theorem masked_add_kernel_compute_correct
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := masked_add_kernel grad_ptr p_ptr p_mask_ptr
        n_elements alpha BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i)
        (fun i => (grad_ptr, maskedAddOffset s BLOCK_SIZE i)))
      (expected := fun i => maskedAddSpec s grad_ptr p_ptr alpha BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [masked_add_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := masked_add_kernel_correct grad_ptr p_ptr p_mask_ptr
    n_elements alpha BLOCK_SIZE s s' hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `masked_add_kernel`: the DSL surface lowers to
the algorithm layer, and the masked in-place store to `grad_ptr` is
compute-correct — every active lane (in-bounds and `p_mask` false) holds
`grad + p * alpha`, all other lanes are preserved. -/
theorem masked_add_kernel_output_summary
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    (∃ alg, (masked_add_kernel grad_ptr p_ptr p_mask_ptr
      n_elements alpha BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := masked_add_kernel grad_ptr p_ptr p_mask_ptr
        n_elements alpha BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i)
        (fun i => (grad_ptr, maskedAddOffset s BLOCK_SIZE i)))
      (expected := fun i => maskedAddSpec s grad_ptr p_ptr alpha BLOCK_SIZE i) := by
  refine ⟨?_, ?_⟩
  · simp [masked_add_kernel, ComputeExpr.toAlgorithm?]
  · exact masked_add_kernel_compute_correct grad_ptr p_ptr p_mask_ptr
      n_elements alpha BLOCK_SIZE s

end VeriTile.Bench.TritonBenchG.MaskedAddCuda
