import bench.tritonbench_g.add_value.Kernels
import bench.tritonbench_g.add_value.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Historical abstract boundary-rounding theorem. Its source uses the .real
carrier without casts, so execR reduces to exact real execution and the output
rounding is identity. This theorem is preserved as a legacy lemma; it does not
establish floating-point program equivalence. Current real correctness and
admitted FP equivalence are the two separate entry files. -/

namespace VeriTile.Bench.TritonBenchG.AddValue

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

namespace LegacyRounding

/-- The historical IO is the same masked window as the current real contract. -/
def io (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) : MaskedKernelIO₁ where
  kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
  inp := x_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N

end LegacyRounding

/-- The kernel is **cast-free**: every statement steps identically under
`stepStmtsR R` and `stepStmts`, so `execR R` is the exact stepper. -/
private theorem puzzle1_kernel_castFree (R : RoundingModel)
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (s : BlockState) :
    execR R ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s
      = exec ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s := by
  simp [execR, exec, puzzle1_kernel, ComputeKernel.toAlgKernel,
    stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
    BlockState.writeMemTypedR,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk **under the rounding model** — the `hts`
obligation of `MaskedKernelIO₁.ImplementsR.intro`. Same lane-wise bounds
contract as `puzzle1_kernel_traceSafe`: the masked load and the masked store
address the window `pid * BLOCK_SIZE + j`, active only when `< N`. -/
theorem puzzle1_kernel_traceSafeR (R : RoundingModel)
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafeR R bounds
      ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAtR` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [puzzle1_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.SafeAtR,
    stepStmtR, evalOpR.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MemAccess.SafeAtR, MaskOpt.ActiveR, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- **The `⊨[R]` headline**: for every rounding model `R`, `puzzle1_kernel`
implements lane-wise add-a-scalar `xs i + value` on its masked IO signature
at the `.real` output grid. Same full masked Hoare triple as
`puzzle1_kernel_correctness` — ∀ disjoint flat placement, ∀ program id whose
active lanes are in bounds, ∀ launch state whose input window holds `xs` at
the active lanes — but the run is `execR R` and every active output lane is
read back as an `.real`-typed cell holding `R.round .real (xs i + value)`.

The store is untyped (`tl.store(output_ptr + offsets, output, mask=mask)` —
no `.to(...)`), so the honest grid is `.real` and the boundary round
degenerates (`R.round .real = id`): this kernel's rounding face carries the
exact value contract for every `R`, with the *modeling* claim that the
kernel introduces no rounding event of its own. No hypotheses: the block
geometry is universally quantified, `BLOCK_SIZE = 0` is the vacuous
zero-lane launch, and the window is injective outright. -/
theorem puzzle1_kernel_io_correctness (R : RoundingModel)
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    LegacyRounding.io x_ptr output_ptr N BLOCK_SIZE value
      ⊨[R, .real] fun xs i => xs i + value := by
  refine MaskedKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact puzzle1_kernel_flattenOk x_ptr output_ptr N BLOCK_SIZE value
  · intro bounds s h1 h2 _
    exact puzzle1_kernel_traceSafeR R x_ptr output_ptr N BLOCK_SIZE value
      bounds s h1 h2
  · intro s₀ xs hx
    simp only [LegacyRounding.io] at hx ⊢
    obtain ⟨s1, hexec, hval, hframe⟩ := puzzle1_kernel_region_run
      x_ptr output_ptr N BLOCK_SIZE value s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    refine ⟨s1, ?_, ?_, fun r o hcond _ => hframe r o hcond⟩
    · rw [puzzle1_kernel_castFree R x_ptr output_ptr N BLOCK_SIZE value s₀]
      exact hexec
    · intro j hj
      rw [BlockState.readMemAs_real, hval j hj]
      simp

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AddValue
