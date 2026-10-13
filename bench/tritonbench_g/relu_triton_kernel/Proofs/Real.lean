import bench.tritonbench_g.relu_triton_kernel.Kernels
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel
open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁
/-- Algorithm-layer correctness for `relu_kernel`, from a launch state whose
input window is loaded at the **active lanes only** (`pid * block_size + j < N`;
the masked load reads the undef channel elsewhere, so inactive lanes are never
consulted).

Only `pid = 0` writes. In that case, active lanes write the conditional-form
ReLU and inactive tail lanes are preserved; nonzero pids preserve the observed
output cells. -/
theorem relu_kernel_correct
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (_hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : ∀ j : Fin block_size, s.pid * block_size + j.val < N →
      s.readMem x_ptr (s.pid * block_size + j.val) = xs j) :
    ∀ i : Fin block_size,
      let addr := s.pid * block_size + i.val
      observeAt (exec (relu_kernel x_ptr out_ptr N block_size) s)
          out_ptr block_size s.pid i
        = some (if s.pid = 0 ∧ addr < N then TiledActivation.relu (xs i)
                else s.readMem out_ptr addr) := by
  intro i
  by_cases hpid : s.pid = 0
  · simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          BlockState.tileIndex1d_offset_injective (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hx0 : s.readMem x_ptr i.val = xs i := by
        simpa [hpid] using h_x i (by simpa [hpid] using hi)
      simp [hi, hx0, TiledActivation.relu]
      split_ifs with hge
      · -- hge : (some 0 : WithBot ℝ) ≤ some (xs i)
        have h : (0 : ℝ) ≤ xs i :=
          WithBot.coe_le_coe.mp (hge : (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        change xs i = max 0 (xs i)
        exact (max_eq_right h).symm
      · have h : ¬ ((0 : ℝ) ≤ xs i) := fun h' =>
          hge (WithBot.coe_le_coe.mpr h' :
            (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        rw [not_le] at h
        change (0.0 : ℝ) = max 0 (xs i)
        rw [max_eq_left h.le]
        norm_num
    · simp [hi]
  · simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the `pid == 0`-gated
masked output store is preserved by the run — in particular every cell of
every region other than `out_ptr`, the *inactive* lanes of the output window,
and (for `pid ≠ 0`) everything. -/
private theorem relu_kernel_frame (x_ptr out_ptr : RegionName)
    (N B : Nat) (s s1 : BlockState)
    (hExec : exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, s.pid = 0 ∧ s.pid * B + i.val < N →
      ¬(out_ptr = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  by_cases hpid : s.pid = 0
  · simp [exec, relu_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
      NumericDType.add, NumericDType.mul,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid] at hExec
    subst hExec
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hmk hc
    refine hmiss k.1 ⟨hpid, ?_⟩ ⟨hc.1, ?_⟩
    · simpa [hpid] using hmk
    · simpa [hpid] using hc.2
  · simp [exec, relu_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
      NumericDType.add, NumericDType.mul,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid] at hExec
    subst hExec
    rfl

/-- **The region-model masked Hoare triple** — termination, write-active-lane
output values, and frame off the write-active output lanes, from any launch
state whose input window is loaded at the **read-active lanes only**. The
write-active set is the kernel's actual store gate
`pid = 0 ∧ pid * B + j < N`, so the triple is fully general over `N`:
programs with `pid ≠ 0` have no write-active lanes and preserve all of
memory. This is what the `⊨` headline transports to flat memory. -/
theorem relu_kernel_region_run
    (x_ptr out_ptr : RegionName)
    (N B : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.pid * B + j.val < N →
      s₀.readMem x_ptr (s₀.pid * B + j.val) = xs j) :
    ∃ s1, exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B, s₀.pid = 0 ∧ s₀.pid * B + j.val < N →
          s1.readMem out_ptr (s₀.pid * B + j.val) = TiledActivation.relu (xs j))
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin B, s₀.pid = 0 ∧ s₀.pid * B + j.val < N →
            o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := relu_kernel_correct x_ptr out_ptr N B hB s₀ xs hx
  rw [show exec (relu_kernel x_ptr out_ptr N B) s₀
      = exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s₀ from rfl] at hobs
  cases hsrc : exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s₀ with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · obtain ⟨hpid0, hj⟩ := hj
        have hj' : j.val < N := by simpa [hpid0] using hj
        have := hobs j
        rw [hsrc] at this
        simpa [observeAt, hpid0, hj'] using this
      · refine relu_kernel_frame x_ptr out_ptr N B s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

/-- The ReLU surface sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked load/store, `where`, comparison, and the
`if pid == 0` gate are all covered). -/
theorem relu_kernel_flattenOk (x_ptr out_ptr : RegionName) (N B : Nat) :
    ((relu_kernel x_ptr out_ptr N B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [relu_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load touches only its read-active
lanes (`pid * B + j < N`), and the store — reached only when the `pid == 0`
gate is taken — only its write-active lanes, so the bounds contract is
lane-wise: every *active* lane's address is below the region bound of the
buffer it touches. -/
theorem relu_kernel_traceSafe (x_ptr out_ptr : RegionName)
    (N B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < N →
      s.pid * B + j.val < bounds x_ptr)
    (hout : ∀ j : Fin B, s.pid = 0 ∧ s.pid * B + j.val < N →
      s.pid * B + j.val < bounds out_ptr) :
    Kernel.TraceSafe bounds ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [relu_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt, ComparableDType.eq, ComparableDType.ge,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  refine ⟨fun a ha => hx a ha, ?_⟩
  split <;> exact ⟨fun h a ha => hout a ⟨h, ha⟩, trivial⟩

/-- Algorithm-layer correctness for `relu_fused_kernel`, from a launch state whose
input window is loaded at the **active lanes only** (`pid * block_size + j < N`;
the masked load reads the undef channel elsewhere, so inactive lanes are never
consulted).

Only `pid = 0` writes. In that case, active lanes write the conditional-form
ReLU and inactive tail lanes are preserved; nonzero pids preserve the observed
output cells. -/
theorem relu_fused_kernel_correct
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (_hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : ∀ j : Fin block_size, s.pid * block_size + j.val < N →
      s.readMem x_ptr (s.pid * block_size + j.val) = xs j) :
    ∀ i : Fin block_size,
      let addr := s.pid * block_size + i.val
      observeAt (exec (relu_fused_kernel x_ptr out_ptr N block_size) s)
          out_ptr block_size s.pid i
        = some (if s.pid = 0 ∧ addr < N then TiledActivation.relu (xs i)
                else s.readMem out_ptr addr) := by
  intro i
  by_cases hpid : s.pid = 0
  · simp [observeAt, exec, relu_fused_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          BlockState.tileIndex1d_offset_injective (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hx0 : s.readMem x_ptr i.val = xs i := by
        simpa [hpid] using h_x i (by simpa [hpid] using hi)
      simp [hi, hx0, TiledActivation.relu]
      split_ifs with hge
      · -- hge : (some 0 : WithBot ℝ) ≤ some (xs i)
        have h : (0 : ℝ) ≤ xs i :=
          WithBot.coe_le_coe.mp (hge : (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        change xs i = max 0 (xs i)
        exact (max_eq_right h).symm
      · have h : ¬ ((0 : ℝ) ≤ xs i) := fun h' =>
          hge (WithBot.coe_le_coe.mpr h' :
            (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        rw [not_le] at h
        change (0.0 : ℝ) = max 0 (xs i)
        rw [max_eq_left h.le]
        norm_num
    · simp [hi]
  · simp [observeAt, exec, relu_fused_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the `pid == 0`-gated
masked output store is preserved by the run — in particular every cell of
every region other than `out_ptr`, the *inactive* lanes of the output window,
and (for `pid ≠ 0`) everything. -/
private theorem relu_fused_kernel_frame (x_ptr out_ptr : RegionName)
    (N B : Nat) (s s1 : BlockState)
    (hExec : exec ((relu_fused_kernel x_ptr out_ptr N B).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, s.pid = 0 ∧ s.pid * B + i.val < N →
      ¬(out_ptr = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  by_cases hpid : s.pid = 0
  · simp [exec, relu_fused_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
      NumericDType.add, NumericDType.mul,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid] at hExec
    subst hExec
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hmk hc
    refine hmiss k.1 ⟨hpid, ?_⟩ ⟨hc.1, ?_⟩
    · simpa [hpid] using hmk
    · simpa [hpid] using hc.2
  · simp [exec, relu_fused_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
      NumericDType.add, NumericDType.mul,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid] at hExec
    subst hExec
    rfl

/-- **The region-model masked Hoare triple** — termination, write-active-lane
output values, and frame off the write-active output lanes, from any launch
state whose input window is loaded at the **read-active lanes only**. The
write-active set is the kernel's actual store gate
`pid = 0 ∧ pid * B + j < N`, so the triple is fully general over `N`:
programs with `pid ≠ 0` have no write-active lanes and preserve all of
memory. This is what the `⊨` headline transports to flat memory. -/
theorem relu_fused_kernel_region_run
    (x_ptr out_ptr : RegionName)
    (N B : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.pid * B + j.val < N →
      s₀.readMem x_ptr (s₀.pid * B + j.val) = xs j) :
    ∃ s1, exec ((relu_fused_kernel x_ptr out_ptr N B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B, s₀.pid = 0 ∧ s₀.pid * B + j.val < N →
          s1.readMem out_ptr (s₀.pid * B + j.val) = TiledActivation.relu (xs j))
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin B, s₀.pid = 0 ∧ s₀.pid * B + j.val < N →
            o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := relu_fused_kernel_correct x_ptr out_ptr N B hB s₀ xs hx
  rw [show exec (relu_fused_kernel x_ptr out_ptr N B) s₀
      = exec ((relu_fused_kernel x_ptr out_ptr N B).toAlgKernel) s₀ from rfl] at hobs
  cases hsrc : exec ((relu_fused_kernel x_ptr out_ptr N B).toAlgKernel) s₀ with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · obtain ⟨hpid0, hj⟩ := hj
        have hj' : j.val < N := by simpa [hpid0] using hj
        have := hobs j
        rw [hsrc] at this
        simpa [observeAt, hpid0, hj'] using this
      · refine relu_fused_kernel_frame x_ptr out_ptr N B s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

/-- The ReLU surface sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked load/store, `where`, comparison, and the
`if pid == 0` gate are all covered). -/
theorem relu_fused_kernel_flattenOk (x_ptr out_ptr : RegionName) (N B : Nat) :
    ((relu_fused_kernel x_ptr out_ptr N B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [relu_fused_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load touches only its read-active
lanes (`pid * B + j < N`), and the store — reached only when the `pid == 0`
gate is taken — only its write-active lanes, so the bounds contract is
lane-wise: every *active* lane's address is below the region bound of the
buffer it touches. -/
theorem relu_fused_kernel_traceSafe (x_ptr out_ptr : RegionName)
    (N B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < N →
      s.pid * B + j.val < bounds x_ptr)
    (hout : ∀ j : Fin B, s.pid = 0 ∧ s.pid * B + j.val < N →
      s.pid * B + j.val < bounds out_ptr) :
    Kernel.TraceSafe bounds ((relu_fused_kernel x_ptr out_ptr N B).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [relu_fused_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt, ComparableDType.eq, ComparableDType.ge,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  refine ⟨fun a ha => hx a ha, ?_⟩
  split <;> exact ⟨fun h a ha => hout a ⟨h, ha⟩, trivial⟩

end VeriTile.Bench.TritonBenchG.ReluTritonKernel
