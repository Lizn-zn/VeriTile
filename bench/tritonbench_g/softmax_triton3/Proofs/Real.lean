import bench.tritonbench_g.softmax_triton3.Proofs.Source
import VeriTile.Triton
import VeriTile.Triton.Semantics.MaskedReduction
import bench.tritonbench_g.softmax_triton3.Proofs.Spec

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3
open VeriTile.Triton

/-- Algorithm-layer cellwise correctness for `softmax_kernel`: in-bounds lanes
hold `softmaxSpec` of the loaded row (and, under `HAS_MASK`, the loaded mask
row), out-of-bounds lanes are preserved. -/
theorem softmax_kernel_correct
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel output_ptr input_ptr mask_ptr row_stride
          n_cols BLOCK_SIZE HAS_MASK) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * row_stride + i.val
      s'.readMem output_ptr outAddr =
        if i.val < n_cols then
          softmaxSpec n_cols BLOCK_SIZE
            (cond HAS_MASK
              (some (fun j => s.readMem mask_ptr (s.pid * row_stride + j.val)))
              none)
            (fun j => s.readMem input_ptr (s.pid * row_stride + j.val)) i
        else s.readMem output_ptr outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · cases HAS_MASK <;>
      simp [exec, softmax_kernel, stepStmts, stepStmt, evalOp.eq_def,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
            FloatDType.cast, hB] at hExec
    · subst s'
      simp [BlockState.pid_eq]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
            (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
      by_cases hi : i.val < n_cols
      · simp [hi, softmaxSpec, softmaxInputTile,
              Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
              TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
              NumericDType.sub, hB]
        congr
      · simp [hi]
    · subst s'
      simp [BlockState.pid_eq]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
            (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
      by_cases hi : i.val < n_cols
      · simp [hi, softmaxSpec, softmaxInputTile, softmaxMaskTile,
              Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
              TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
              NumericDType.add, NumericDType.sub, hB]
        congr
      · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `output_ptr`, and the *inactive*
lanes of the output row itself — is preserved by the run (both `HAS_MASK`
branches; the mask load writes no memory). -/
private theorem softmax_kernel_frame
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (hB : 0 < BLOCK_SIZE) (s s1 : BlockState)
    (hExec : exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(output_ptr = r ∧ s.pid * row_stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  cases HAS_MASK <;>
    (simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          FloatDType.cast,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hB] at hExec
     subst s1
     refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
     intro k _ hmk hc
     exact hmiss k.1 (by simpa using hmk) hc)

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (both
`HAS_MASK` branches). `0 < BLOCK_SIZE` is required because the `max` reduce
(like `Finset.sup'`) is only defined on non-empty axes. -/
private theorem softmax_kernel_exec_isSome
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (hB : 0 < BLOCK_SIZE) (s : BlockState) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK).toAlgKernel) s = some s1 := by
  cases HAS_MASK <;>
    simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          FloatDType.cast,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hB]

/-- **The region-model masked Hoare triple, `HAS_MASK = false`** — termination,
active-lane output values, and frame off the active output lanes, from any
launch state whose input row is loaded at the **active lanes only**
(`j < n_cols`). This is the `hrun` obligation of the unmasked-branch `⊨`
headline. -/
theorem softmax_kernel_region_run
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem input_ptr (s₀.pid * row_stride + j.val) = xs j) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
          n_cols BLOCK_SIZE Bool.false).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem output_ptr (s₀.pid * row_stride + j.val)
            = softmaxSpec n_cols BLOCK_SIZE none xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * row_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := softmax_kernel_exec_isSome output_ptr input_ptr mask_ptr
    row_stride n_cols BLOCK_SIZE Bool.false hB s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := softmax_kernel_correct output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.false s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact softmaxSpec_congr n_cols BLOCK_SIZE none _ xs (fun k hk => hx k hk) j
  · refine softmax_kernel_frame output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.false hB s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- **The region-model masked Hoare triple, `HAS_MASK = true`** — same as
`softmax_kernel_region_run`, from any launch state whose input row **and**
additive-mask row are loaded at the active lanes only. This is the `hrun`
obligation of the masked-branch `⊨` headline. -/
theorem softmax_kernel_masked_region_run
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s₀ : BlockState) (xs ms : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem input_ptr (s₀.pid * row_stride + j.val) = xs j)
    (hm : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem mask_ptr (s₀.pid * row_stride + j.val) = ms j) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
          n_cols BLOCK_SIZE Bool.true).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem output_ptr (s₀.pid * row_stride + j.val)
            = softmaxSpec n_cols BLOCK_SIZE (some ms) xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * row_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := softmax_kernel_exec_isSome output_ptr input_ptr mask_ptr
    row_stride n_cols BLOCK_SIZE Bool.true hB s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := softmax_kernel_correct output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.true s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact (softmaxSpec_congr n_cols BLOCK_SIZE _ _ xs (fun k hk => hx k hk) j).trans
      (softmaxSpec_mask_congr n_cols BLOCK_SIZE xs _ ms (fun k hk => hm k hk) j)
  · refine softmax_kernel_frame output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.true hB s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, dtype casts, the constexpr `if`,
reductions, masked store) — both `HAS_MASK` branches. -/
theorem softmax_kernel_flattenOk
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ((softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE HAS_MASK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases HAS_MASK <;>
    simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
      Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = false`: one computational unfold
walks all the statements — the pointer/index staging, the reductions, and the
register arithmetic are memory-silent, the constexpr `if` is inert — and
reduces the two masked accesses (row load, row store) to the **lane-wise**
bounds hypotheses: every *active* lane's address is below the region bound. -/
theorem softmax_kernel_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.false).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = true`: as
`softmax_kernel_traceSafe`, plus the mask row load — three masked accesses
(row load, mask load, row store), each reduced to its lane-wise bounds
hypothesis. -/
theorem softmax_kernel_masked_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hmask : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds mask_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.true).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hmask hout
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hmask a ha, fun a ha => hout a ha⟩

private theorem mul_recip (a b : Option ℝ) :
    Option.map₂ (· * ·) a (b.map (fun x => x⁻¹)) = Option.map₂ (· / ·) a b := by
  cases a <;> cases b <;> simp [div_eq_mul_inv]

set_option maxHeartbeats 1800000 in
theorem reciprocal_exec (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols B : Nat) (hasMask : Bool) (hB : 0 < B) (s : BlockState) :
    exec (softmax_reciprocal_kernel output_ptr input_ptr mask_ptr row_stride n_cols B hasMask).toAlgKernel s =
    exec (softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols B hasMask).toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  cases hasMask <;> simp [softmax_kernel, softmax_reciprocal_kernel, exec, stepStmts, stepStmt,
    evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    NumericDType.mul, NumericDType.div, mul_recip, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast]

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, dtype casts, the constexpr `if`,
reductions, masked store) — both `HAS_MASK` branches. -/
theorem softmax_reciprocal_kernel_flattenOk
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ((softmax_reciprocal_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE HAS_MASK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases HAS_MASK <;>
    simp [softmax_reciprocal_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
      Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = false`: one computational unfold
walks all the statements — the pointer/index staging, the reductions, and the
register arithmetic are memory-silent, the constexpr `if` is inert — and
reduces the two masked accesses (row load, row store) to the **lane-wise**
bounds hypotheses: every *active* lane's address is below the region bound. -/
theorem softmax_reciprocal_kernel_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_reciprocal_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.false).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [softmax_reciprocal_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = true`: as
`softmax_reciprocal_kernel_traceSafe`, plus the mask row load — three masked accesses
(row load, mask load, row store), each reduced to its lane-wise bounds
hypothesis. -/
theorem softmax_reciprocal_kernel_masked_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hmask : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds mask_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_reciprocal_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.true).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hmask hout
  simp [softmax_reciprocal_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hmask a ha, fun a ha => hout a ha⟩

set_option maxHeartbeats 2000000 in
/-- Removing the explicit casts changes no operation in the real interpretation. -/
theorem math_exec_eq (out inp mask : RegionName) (stride n B : Nat)
    (hasMask : Bool) (hB : 0 < B) (s : BlockState) :
    exec (math_kernel out inp mask stride n B hasMask).toAlgKernel s =
      exec (softmax_kernel out inp mask stride n B hasMask).toAlgKernel s := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  cases hasMask <;> simp [math_kernel, softmax_kernel, exec, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, FloatDType.cast, Tile.reduceMax, Tile.reduceMaxDrop,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex]

set_option maxHeartbeats 2000000 in
theorem math_reciprocal_exec_eq (out inp mask : RegionName) (stride n B : Nat)
    (hasMask : Bool) (hB : 0 < B) (s : BlockState) :
    exec (math_reciprocal_kernel out inp mask stride n B hasMask).toAlgKernel s =
      exec (softmax_reciprocal_kernel out inp mask stride n B hasMask).toAlgKernel s := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  cases hasMask <;> simp [math_reciprocal_kernel, softmax_reciprocal_kernel,
    exec, stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, FloatDType.cast, Tile.reduceMax, Tile.reduceMaxDrop,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex]

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, dtype casts, the constexpr `if`,
reductions, masked store) — both `HAS_MASK` branches. -/
theorem math_kernel_flattenOk
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ((math_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE HAS_MASK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases HAS_MASK <;>
    simp [math_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
      Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = false`: one computational unfold
walks all the statements — the pointer/index staging, the reductions, and the
register arithmetic are memory-silent, the constexpr `if` is inert — and
reduces the two masked accesses (row load, row store) to the **lane-wise**
bounds hypotheses: every *active* lane's address is below the region bound. -/
theorem math_kernel_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((math_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.false).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [math_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = true`: as
`math_kernel_traceSafe`, plus the mask row load — three masked accesses
(row load, mask load, row store), each reduced to its lane-wise bounds
hypothesis. -/
theorem math_kernel_masked_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hmask : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds mask_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((math_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.true).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hmask hout
  simp [math_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hmask a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, dtype casts, the constexpr `if`,
reductions, masked store) — both `HAS_MASK` branches. -/
theorem math_reciprocal_kernel_flattenOk
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ((math_reciprocal_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE HAS_MASK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases HAS_MASK <;>
    simp [math_reciprocal_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
      Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = false`: one computational unfold
walks all the statements — the pointer/index staging, the reductions, and the
register arithmetic are memory-silent, the constexpr `if` is inert — and
reduces the two masked accesses (row load, row store) to the **lane-wise**
bounds hypotheses: every *active* lane's address is below the region bound. -/
theorem math_reciprocal_kernel_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((math_reciprocal_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.false).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [math_reciprocal_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = true`: as
`math_reciprocal_kernel_traceSafe`, plus the mask row load — three masked accesses
(row load, mask load, row store), each reduced to its lane-wise bounds
hypothesis. -/
theorem math_reciprocal_kernel_masked_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hmask : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds mask_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((math_reciprocal_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.true).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hmask hout
  simp [math_reciprocal_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hmask a ha, fun a ha => hout a ha⟩

theorem typed_exec_eq (inputType maskType : Storage) (v : Bool) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) (hB : 0 < B) (s : BlockState) :
    exec (typedKernel inputType maskType v out inp mask stride n B hasMask).eraseDType.toAlgKernel s =
      exec (softmax_kernel out inp mask stride n B hasMask).toAlgKernel s := by
  cases v
  · rw [typed_projection]
    exact math_exec_eq out inp mask stride n B hasMask hB s
  · rw [reciprocal_projection, math_reciprocal_exec_eq out inp mask stride n B hasMask hB]
    exact reciprocal_exec out inp mask stride n B hasMask hB s

theorem typed_flattenOk (inputType maskType : Storage) (v : Bool) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType v out inp mask stride n B hasMask).eraseDType.toAlgKernel.FlattenOk := by
  cases v
  · rw [typed_projection]; exact math_kernel_flattenOk out inp mask stride n B hasMask
  · rw [reciprocal_projection]
    exact math_reciprocal_kernel_flattenOk out inp mask stride n B hasMask

theorem typed_traceSafe (inputType maskType : Storage) (v : Bool) (out inp mask : RegionName)
    (stride n B : Nat) (hB : 0 < B) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin B, j.val < n → s.pid * stride + j.val < bounds inp)
    (hout : ∀ j : Fin B, j.val < n → s.pid * stride + j.val < bounds out) :
    Kernel.TraceSafe bounds
      (typedKernel inputType maskType v out inp mask stride n B Bool.false).eraseDType.toAlgKernel s := by
  cases v
  · rw [typed_projection]
    exact math_kernel_traceSafe out inp mask stride n B hB bounds s hin hout
  · rw [reciprocal_projection]
    exact math_reciprocal_kernel_traceSafe out inp mask stride n B hB bounds s hin hout

theorem typed_masked_traceSafe (inputType maskType : Storage) (v : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hB : 0 < B) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin B, j.val < n → s.pid * stride + j.val < bounds inp)
    (hm : ∀ j : Fin B, j.val < n → s.pid * stride + j.val < bounds mask)
    (hout : ∀ j : Fin B, j.val < n → s.pid * stride + j.val < bounds out) :
    Kernel.TraceSafe bounds
      (typedKernel inputType maskType v out inp mask stride n B Bool.true).eraseDType.toAlgKernel s := by
  cases v
  · rw [typed_projection]
    exact math_kernel_masked_traceSafe out inp mask stride n B hB bounds s hin hm hout
  · rw [reciprocal_projection]
    exact math_reciprocal_kernel_masked_traceSafe out inp mask stride n B hB bounds s hin hm hout

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
