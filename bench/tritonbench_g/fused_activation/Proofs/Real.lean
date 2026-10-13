import bench.tritonbench_g.fused_activation.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.FusedActivation
open VeriTile.Triton

noncomputable def fusedActivationInput
    (x bias input multiplier : ℝ) : ℝ :=
  multiplier * input + x + bias

/-- Algorithm-layer branch form of the activation selector. `false` is the
`tl.maximum(0, x)` ReLU branch. -/
noncomputable def fusedActivationSpec
    (ACTIVATION_SIGMOID : Bool) (x bias input multiplier : ℝ) : ℝ :=
  let z := fusedActivationInput x bias input multiplier
  if ACTIVATION_SIGMOID then
    Real.sigmoid z
  else
    WithBot.unbotD 0
      (if ComparableDType.real.gt (some 0) (some z) then
        (some 0 : WithBot ℝ)
      else
        (some z : WithBot ℝ))

def fusedActivationOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

/-- Algorithm-layer correctness for `fused_add_mul_activation_kernel`: at every
in-bounds lane the in-place store leaves `fusedActivationSpec` of the loaded
values, and every out-of-bounds lane of the window is preserved. The read
hypotheses are needed only at the active (in-bounds) lanes — exactly the lanes
the kernel's masked loads touch. -/
theorem fused_add_mul_activation_kernel_correct
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s s' : BlockState)
    (xs inputs : Fin BLOCK_SIZE → ℝ)
    (biases : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ i : Fin BLOCK_SIZE, fusedActivationOffset s BLOCK_SIZE i < xnumel →
      s.readMem x_ptr (fusedActivationOffset s BLOCK_SIZE i) = xs i)
    (h_in : ∀ i : Fin BLOCK_SIZE, fusedActivationOffset s BLOCK_SIZE i < xnumel →
      s.readMem in_ptr (fusedActivationOffset s BLOCK_SIZE i) = inputs i)
    (h_bias : ∀ i : Fin BLOCK_SIZE, fusedActivationOffset s BLOCK_SIZE i < xnumel →
      s.readMem bias_ptr ((fusedActivationOffset s BLOCK_SIZE i) % num_weights) = biases i)
    (hExec : exec (fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
          num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := fusedActivationOffset s BLOCK_SIZE i
      s'.readMem x_ptr outAddr =
        if outAddr < xnumel then
          fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier
        else s.readMem x_ptr outAddr := by
  intro i
  cases ACTIVATION_SIGMOID
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp.eq_def,
          tile_elementwise] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i (by simpa [fusedActivationOffset] using hi)
      have hin := h_in i (by simpa [fusedActivationOffset] using hi)
      have hb := h_bias i (by simpa [fusedActivationOffset] using hi)
      simp [fusedActivationOffset] at hx hin hb
      by_cases hlt : multiplier * inputs i + xs i + biases i < 0
      · simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb,
          ComparableDType.gt]
      · simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb,
          ComparableDType.gt]
    · simp [hi]
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp.eq_def,
          tile_elementwise] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i (by simpa [fusedActivationOffset] using hi)
      have hin := h_in i (by simpa [fusedActivationOffset] using hi)
      have hb := h_bias i (by simpa [fusedActivationOffset] using hi)
      simp [fusedActivationOffset] at hx hin hb
      simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb]
    · simp [hi]

/-- Frame half: every memory cell not actively written by the masked in-place
store — every cell of every region other than `x_ptr`, and the out-of-bounds
lanes of the output window itself — is preserved by the run. -/
private theorem fused_add_mul_activation_kernel_frame
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s s1 : BlockState)
    (hExec : exec ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE,
      s.pids 0 * BLOCK_SIZE + i.val < xnumel →
      ¬(x_ptr = r ∧ s.pids 0 * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  cases ACTIVATION_SIGMOID <;>
  · simp [exec, fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel, stepStmts,
      stepStmt, evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
    subst hExec
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hmk hc
    exact hmiss k.1 hmk hc

/-- Termination: the kernel executes to completion from any state (straight-line
elementwise body in both activation branches). -/
private theorem fused_add_mul_activation_kernel_exec_isSome
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s : BlockState) :
    ∃ s1, exec ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s
      = some s1 := by
  cases ACTIVATION_SIGMOID <;>
  simp [exec, fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel, stepStmts,
    stepStmt, evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?]

/-- **The region-model masked in-place Hoare triple** — termination,
active-lane output values, and frame off the active output lanes, from any
launch state whose three read windows hold `xs`/`biases`/`inputs` at the
in-bounds lanes. This is the `hrun` obligation of the `⊨` headline; the value
half reuses `fused_add_mul_activation_kernel_correct` (the kernel reads
`x_ptr` before the in-place scatter, so the before/after reading is sound). -/
theorem fused_add_mul_activation_kernel_region_run
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s₀ : BlockState) (xs biases inputs : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s₀.readMem x_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = xs j)
    (h_bias : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s₀.readMem bias_ptr ((s₀.pids 0 * BLOCK_SIZE + j.val) % num_weights) = biases j)
    (h_in : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s₀.readMem in_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = inputs j) :
    ∃ s1, exec ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
          num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
          s1.readMem x_ptr (s₀.pids 0 * BLOCK_SIZE + j.val)
            = fusedActivationSpec ACTIVATION_SIGMOID (xs j) (biases j) (inputs j)
                multiplier)
      ∧ (∀ r o,
          (∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
            r ≠ x_ptr ∨ o ≠ s₀.pids 0 * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := fused_add_mul_activation_kernel_exec_isSome x_ptr bias_ptr
    in_ptr num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀
  have hs1' : exec (fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID) s₀ = some s1 := by
    simpa [ComputeKernel.toAlgKernel, fused_add_mul_activation_kernel,
      ComputeExpr.toAlgorithm?] using hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := fused_add_mul_activation_kernel_correct x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀ s1 xs inputs
      biases (by simpa [fusedActivationOffset, BlockState.pid] using h_x)
      (by simpa [fusedActivationOffset, BlockState.pid] using h_in)
      (by simpa [fusedActivationOffset, BlockState.pid] using h_bias) hs1' j
    simp only [fusedActivationOffset, BlockState.pid] at h
    rw [h, if_pos hj]
  · refine fused_add_mul_activation_kernel_frame x_ptr bias_ptr in_ptr num_weights
      xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀ s1 hs1 r o
      (fun i hbounds ⟨hr, ho⟩ => ?_)
    rcases hcond i hbounds with hne | hno
    · exact hne hr.symm
    · exact hno ho.symm

/-- Per-execution safety walk: one computational unfold walks the whole body —
the pid/arange/mask/bias-index arithmetic is memory-silent, and the three
masked loads (`x_ptr` and `in_ptr` at `pid·BLOCK_SIZE + j`, `bias_ptr` at
`(pid·BLOCK_SIZE + j) % num_weights`) plus the masked in-place store to
`x_ptr` reduce to the lane-wise bounds hypotheses at the load mask
`index < xnumel`. Both activation branches are walked. -/
theorem fused_add_mul_activation_kernel_traceSafe
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s.pids 0 * BLOCK_SIZE + j.val < bounds x_ptr)
    (h2 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < xnumel →
      (s.pids 0 * BLOCK_SIZE + j.val) % num_weights < bounds bias_ptr)
    (h3 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s.pids 0 * BLOCK_SIZE + j.val < bounds in_ptr) :
    Kernel.TraceSafe bounds
      ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights xnumel
        BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  cases ACTIVATION_SIGMOID <;>
  · simp [fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, BlockState.setReg, tile_elementwise,
      Tile.bop, Tile.cop, Tile.uop, Tile.select,
      NumericDType.add, NumericDType.mul, ComparableDType.lt]
    exact ⟨fun a ha => h1 a ha, fun a ha => h2 a ha, fun a ha => h3 a ha,
      fun a ha => h1 a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, three masked loads, the activation branch, and the masked in-place
store). -/
theorem fused_add_mul_activation_kernel_flattenOk
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) :
    ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights xnumel
        BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases ACTIVATION_SIGMOID <;>
  simp [fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 3000000

private theorem optional_add_comm (a b : Option ℝ) :
    Option.map₂ (fun x y => x + y) a b = Option.map₂ (fun x y => x + y) b a :=
  Option.map₂_comm add_comm

/-- Real execution equality follows from real addition commutativity. -/
theorem commuted_exec (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool)
    (s : BlockState) :
    exec (fused_add_mul_activation_commuted_kernel x b i W N B m a).toAlgKernel s =
      exec (fused_add_mul_activation_kernel x b i W N B m a).toAlgKernel s := by
  cases a <;>
    simp [fused_add_mul_activation_commuted_kernel, fused_add_mul_activation_kernel,
      exec, stepStmts, stepStmt, evalOp.eq_def, tile_elementwise, add_comm, optional_add_comm]

theorem commuted_flattenOk (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) :
    (fused_add_mul_activation_commuted_kernel x b i W N B m a).toAlgKernel.FlattenOk := by
  cases a <;>
    simp [fused_add_mul_activation_commuted_kernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem commuted_traceSafe (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pids 0 * B + j.val < N → s.pids 0 * B + j.val < bounds x)
    (hb : ∀ j : Fin B, s.pids 0 * B + j.val < N →
      (s.pids 0 * B + j.val) % W < bounds b)
    (hi : ∀ j : Fin B, s.pids 0 * B + j.val < N → s.pids 0 * B + j.val < bounds i) :
    Kernel.TraceSafe bounds
      (fused_add_mul_activation_commuted_kernel x b i W N B m a).toAlgKernel s := by
  unfold Kernel.TraceSafe
  cases a <;>
  · simp [fused_add_mul_activation_commuted_kernel, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
      BlockState.setReg, tile_elementwise, Tile.bop, Tile.cop, Tile.uop, Tile.select,
      NumericDType.add, NumericDType.mul, ComparableDType.lt]
    exact ⟨fun j hj => hx j hj, fun j hj => hb j hj, fun j hj => hi j hj,
      fun j hj => hx j hj⟩

end VeriTile.Bench.TritonBenchG.FusedActivation
