import bench.tritonbench_g.swiglu_triton.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluTriton
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

def swigluOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val

/-! ## Forward kernel -/

/-- Algorithm-layer correctness for `_swiglu_forward_kernel`. -/
theorem swiglu_forward_kernel_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i)
    (hExec : exec (swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem C outAddr =
        if i.val < n_cols then
          TiledActivation.swiglu (as i) (bs i)
        else s.readMem C outAddr := by
  intro i
  simp [exec, swiglu_forward_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [swigluOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · have ha := h_a i
    have hb := h_b i
    simp [swigluOffset] at ha hb
    simp [hi, TiledActivation.swiglu, TiledActivation.silu, ha, hb]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `C`, and the *inactive* lanes of
the output row itself — is preserved by the run. -/
private theorem swiglu_forward_kernel_frame
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(C = r ∧ s.pid * stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, swiglu_forward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  rw [← Int.natCast_mul, Int.toNat_natCast] at hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the forward kernel executes to completion from any state (all
statements are elementwise, so no `0 < BLOCK_SIZE` side condition is needed). -/
private theorem swiglu_forward_kernel_exec_isSome
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, swiglu_forward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input rows are loaded at the **active lanes only** (`j < n_cols`). This is the
`hrun` obligation of the `⊨` headline; the value half reuses
`swiglu_forward_kernel_correct` (instantiated at the tiles the state actually
holds). -/
theorem swiglu_forward_kernel_region_run
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s₀ : BlockState) (as bs : Fin BLOCK_SIZE → ℝ)
    (ha : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem A (s₀.pid * stride + j.val) = as j)
    (hb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem B (s₀.pid * stride + j.val) = bs j) :
    ∃ s1, exec ((swiglu_forward_kernel A B C stride n_cols
          BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem C (s₀.pid * stride + j.val)
            = TiledActivation.swiglu (as j) (bs j))
      ∧ (∀ r o,
          (r ≠ C ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := swiglu_forward_kernel_exec_isSome A B C stride n_cols
    BLOCK_SIZE s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := swiglu_forward_kernel_correct A B C stride n_cols BLOCK_SIZE
      s₀ s1
      (fun i => s₀.readMem A (swigluOffset s₀ stride i))
      (fun i => s₀.readMem B (swigluOffset s₀ stride i))
      (fun _ => rfl) (fun _ => rfl) hs1 j
    simp only [hj, if_pos] at h
    rw [show swigluOffset s₀ stride j = s₀.pid * stride + j.val from rfl] at h
    rw [h, ha j hj, hb j hj]
  · refine swiglu_forward_kernel_frame A B C stride n_cols BLOCK_SIZE
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The forward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads with `other`, `.to(...)` casts, elementwise
math, masked store). -/
theorem swiglu_forward_kernel_flattenOk
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [swiglu_forward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same row window `pid * stride + j`, active only when `j < n_cols`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem swiglu_forward_kernel_traceSafe
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hina : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds A)
    (hinb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds B)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds C) :
    Kernel.TraceSafe bounds
      ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hina hinb hout
  simp [swiglu_forward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, tile_elementwise,
    NumericDType.mul, ComparableDType.lt]
  simp only [← Int.natCast_mul, Int.toNat_natCast]
  exact ⟨fun a ha => hina a ha, fun a ha => hinb a ha, fun a ha => hout a ha⟩

/-- `_swiglu_forward_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: gate `A`,
  value `B`, output `C`);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` (the row) reads and writes its row at
  `pid * stride` in all three buffers (the host-side one-program-per-row launch
  convention `x_ptr += program_id * stride`);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def swigluFwdIO (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE
  in1 := A
  in2 := B
  out := C
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols

/-! ## Backward kernel -/

/-- Algorithm-layer correctness for `_swiglu_backward_kernel`.

The two stores write `A` first and `B` second. We assume `A ≠ B` so the second
store cannot overwrite the first output channel. -/
theorem swiglu_backward_kernel_correct
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (swigluOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i)
    (hExec : exec (swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE) s = some s') :
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem A outAddr =
        if i.val < n_cols then
          TiledActivation.swigluBwdA (dcs i) (as i) (bs i)
        else s.readMem A outAddr) ∧
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem B outAddr =
        if i.val < n_cols then
          TiledActivation.swigluBwdB (dcs i) (as i)
        else s.readMem B outAddr) := by
  simp [exec, swiglu_backward_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  constructor
  · intro i
    simp only [swigluOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
    rw [BlockState.scatter_prop_masked_preserves_other_region
      (region := B) (R := A) (h_ne := hAB)
      (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
      (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      have hb := h_b i
      simp [swigluOffset] at hdc ha hb
      simp [hi, TiledActivation.swigluBwdA, TiledActivation.silu, hdc, ha, hb]
    · simp [hi]
  · intro i
    simp only [swigluOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      simp [swigluOffset] at hdc ha
      simp [hi, TiledActivation.swigluBwdB, TiledActivation.silu, hdc, ha]
    · rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := A) (R := B) (h_ne := Ne.symm hAB)
        (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
        (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
      simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell outside **both** active output windows —
every cell of every region other than `A`/`B`, and the *inactive* lanes of the
two output rows themselves — is preserved by the run. -/
private theorem swiglu_backward_kernel_frame
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissA : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(A = r ∧ s.pid * stride + i.val = o))
    (hmissB : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(B = r ∧ s.pid * stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, swiglu_backward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
  · intro k _ hmk hc
    rw [← Int.natCast_mul, Int.toNat_natCast] at hc
    exact hmissB k.1 (by simpa using hmk) hc
  · intro k _ hmk hc
    rw [← Int.natCast_mul, Int.toNat_natCast] at hc
    exact hmissA k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the backward kernel executes to completion from any state (all
statements are elementwise, so no `0 < BLOCK_SIZE` side condition is needed). -/
private theorem swiglu_backward_kernel_exec_isSome
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, swiglu_backward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked in-place Hoare triple** — termination,
active-lane values of both in-place outputs, and frame off the two active
output windows, from any launch state whose three input rows are loaded at the
**active lanes only** (`j < n_cols`). This is the `hrun` obligation of the `⊨`
headline; the value half reuses `swiglu_backward_kernel_correct` (instantiated
at the tiles the state actually holds). -/
theorem swiglu_backward_kernel_region_run
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (hAB : A ≠ B)
    (s₀ : BlockState) (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hdc : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem DC (s₀.pid * stride + j.val) = dcs j)
    (ha : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem A (s₀.pid * stride + j.val) = as j)
    (hb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem B (s₀.pid * stride + j.val) = bs j) :
    ∃ s1, exec ((swiglu_backward_kernel DC A B stride n_cols
          BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem A (s₀.pid * stride + j.val)
            = TiledActivation.swigluBwdA (dcs j) (as j) (bs j))
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem B (s₀.pid * stride + j.val)
            = TiledActivation.swigluBwdB (dcs j) (as j))
      ∧ (∀ r o,
          (r ≠ A ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          (r ≠ B ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := swiglu_backward_kernel_exec_isSome DC A B stride n_cols
    BLOCK_SIZE s₀
  have hcorr := swiglu_backward_kernel_correct DC A B stride n_cols BLOCK_SIZE
    s₀ s1
    (fun i => s₀.readMem DC (swigluOffset s₀ stride i))
    (fun i => s₀.readMem A (swigluOffset s₀ stride i))
    (fun i => s₀.readMem B (swigluOffset s₀ stride i))
    hAB (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun j hj => ?_, fun r o h1 h2 => ?_⟩
  · -- `A` window: the in-place gate gradient realizes `swigluBwdA`.
    have h := hcorr.1 j
    simp only [hj, if_pos] at h
    rw [show swigluOffset s₀ stride j = s₀.pid * stride + j.val from rfl] at h
    rw [h, hdc j hj, ha j hj, hb j hj]
  · -- `B` window: the in-place value gradient realizes `swigluBwdB`.
    have h := hcorr.2 j
    simp only [hj, if_pos] at h
    rw [show swigluOffset s₀ stride j = s₀.pid * stride + j.val from rfl] at h
    rw [h, hdc j hj, ha j hj]
  · -- frame off the two active output windows.
    refine swiglu_backward_kernel_frame DC A B stride n_cols BLOCK_SIZE
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_) (fun i hi ⟨hr, ho⟩ => ?_)
    · rcases h1 with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm
    · rcases h2 with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm

/-- The backward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads with `other`, `.to(...)` casts, elementwise
math, two masked stores). -/
theorem swiglu_backward_kernel_flattenOk
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [swiglu_backward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: all three masked loads and both masked stores
address the same row window `pid * stride + j`, active only when `j < n_cols`,
so the bounds contract is **lane-wise** — every *active* lane's address is
below the region bound of the buffer it touches. -/
theorem swiglu_backward_kernel_traceSafe
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hdc : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds DC)
    (hina : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds A)
    (hinb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds B) :
    Kernel.TraceSafe bounds
      ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hdc hina hinb
  simp [swiglu_backward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, tile_elementwise,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt]
  simp only [← Int.natCast_mul, Int.toNat_natCast]
  exact ⟨fun a ha => hdc a ha, fun a ha => hina a ha, fun a ha => hinb a ha,
    fun a ha => hina a ha, fun a ha => hinb a ha⟩

/-- `_swiglu_backward_kernel`'s masked in-place **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `bufs` — the allocation list: three buffers, each exactly once;
* `in1`/`in2`/`in3` — upstream gradient `DC`, gate `A`, value `B` (the wiring);
* `out1 = in2`, `out2 = in3` — the **in-place** roles: the kernel overwrites
  the gate and value buffers it read with `da` and `db`;
* `read1..3`/`write1..2` — every window is the same row `pid * stride` (the
  host-side one-program-per-row launch convention
  `x_ptr += program_id * stride`);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def swigluBwdIO (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE
  bufs := [DC, A, B]  -- A and B are updated in place
  in1 := DC
  in2 := A
  in3 := B
  out1 := A    -- = in2: `da` overwrites the gate input in place
  out2 := B    -- = in3: `db` overwrites the value input in place
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  read3 := fun pid => pid * stride
  write1 := fun pid => pid * stride
  write2 := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols

end VeriTile.Bench.TritonBenchG.SwigluTriton
