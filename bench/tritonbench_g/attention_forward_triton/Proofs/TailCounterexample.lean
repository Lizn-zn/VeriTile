import bench.tritonbench_g.attention_forward_triton.Proofs.LegacyRounding
import bench.tritonbench_g.attention_forward_triton.Proofs.RealNormalization
import VeriTile.Meta.Disprove
import VeriTile.Meta.StatementAudit

/-! A real-arithmetic counterexample to unqualified length-N attention.
The original masks padded K/V loads but includes their scores in the softmax
denominator. This file erases the complete typed source, retaining its 128 head lanes and
96 active lanes. It is not a floating-point experiment or a numerical atom. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.TailCounterexample
open VeriTile.Triton

set_option maxHeartbeats 3200000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

/-- One real key in a two-key block, with the source's actual head layout. -/
def kernel : ComputeKernel :=
  (TypedSource.original "Q" "K" "V" "QS" "KS" "Out"
    128 128 128 1 128 128 128 1 128 128 128 1 128 128 128 1
    1 1 1 128 1 2 128 96 1).eraseDType

/-- Zero queries/keys and unit active values; masked loads use zero. -/
noncomputable def initial : BlockState where
  mem := fun r _ => MemCell.real (if r = "V" then 1 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  undef := fun _ _ => 0

/-- The addition monoid here is the one used by `Tile.dot`'s finite sum. -/
private theorem repeated_value (n : Nat) (x : ℝ) :
    @HSMul.hSMul Nat (WithBot ℝ) (WithBot ℝ)
      (@instHSMul Nat (WithBot ℝ)
        (@AddMonoid.toNSMul (WithBot ℝ)
          (@WithBot.addCommMonoid ℝ Real.instAddCommMonoid).toAddMonoid))
      n (some x) = some (n * x) := by
  change some (n • x) = some (n * x)
  simp [nsmul_eq_mul]

@[simp] private theorem bot_gt_value (x : ℝ) :
    ComparableDType.gt .real none (some x) = Bool.false := by
  change decide ((x : WithBot ℝ) < ⊥) = Bool.false
  simp

private theorem readback_first_lane (s : BlockState)
    (f : BlockState → TileIndex [1, 128] → BlockState)
    (hf : ∀ s i, f s i = if i.2.1.val < 96
      then s.writeMem "Out" i.2.1.val ((2 : ℝ)⁻¹) else s) :
    (List.foldl f s (TileShape.allIndices [1, 128])).readMem "Out" 0 = (2 : ℝ)⁻¹ := by
  have hinj : Function.Injective (fun i : TileIndex [1, 128] => i.2.1.val) := by
    rintro ⟨a, b, u⟩ ⟨c, d, v⟩ h
    have hbd : b = d := Fin.ext h
    subst d
    have hac : a = c := Subsingleton.elim _ _
    subst c
    cases u
    cases v
    rfl
  rw [show f = (fun s i => if i.2.1.val < 96
    then s.writeMem "Out" i.2.1.val ((2 : ℝ)⁻¹) else s) from funext fun s => funext (hf s)]
  simpa using BlockState.scatter_readback_prop_masked_nd (region := "Out") s
    (fun i : TileIndex [1, 128] => i.2.1.val) (fun _ => (1 / 2 : ℝ))
    (fun i => i.2.1.val < 96) hinj (0, 0, PUnit.unit)

theorem stored_output :
    (exec kernel initial).map (·.readMem "Out" 0) = some (1 / 2 : ℝ) := by
  conv_lhs =>
    arg 2
    simp only [kernel, TypedSource.original, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind]
    simp [kernel, attention_forward_triton_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, exec, stepStmts, stepStmt,
      stepForRangeAux, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
      Tile.cop, Tile.ptrAdd,
      Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
      NumericDType.sub, NumericDType.mul, NumericDType.div, IntegralDType.floorDiv,
      IntegralDType.mod, ComparableDType.lt, initial, Tile.dot,
      Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.uop, WithBot.realExp2, repeated_value, WithBot.lt_def,
      TileShape.axisDim,
      TileShape.insertAxisIndex, FloatDType.cast, BlockState.readMem, BlockState.readMemTyped,
      MemCell.real, MemCell.of, MemCell.readAs]
  simp only [Option.map_some, Option.some.injEq]
  simp (config := { contextual := Bool.true }) [WithBot.add, WithTop.add, WithBot.unbotD]
  apply readback_first_lane
  intro acc i
  by_cases hi : i.2.1.val < 96
  · simp only [hi, ↓reduceIte]
    apply congrArg (acc.writeMem "Out" i.2.1.val)
    change (1 + 0) / (Fintype.card (Fin 2) : ℝ) = (2 : ℝ)⁻¹
    norm_num
  · simp [hi]

/-- Standard attention over exactly the one real key, using the original
input memory and source scale/address maps, with no padded score term. -/
noncomputable def standardOutput : ℝ :=
  attentionRealBase2PerKeyScale
    (qTile initial "Q" 1 128 128 128 1 96)
    (kTile initial "K" 1 128 128 128 1 96)
    (vTile initial "V" 1 128 128 128 1 96)
    (keyScale initial "QS" "KS" 1 1 2 1) (0, 0, PUnit.unit)

theorem standard_output : standardOutput = 1 := by
  simp [standardOutput, attentionRealBase2PerKeyScale, qTile, kTile, vTile,
    keyScale, baseOffset, offZ, offH, mIndex, initial, BlockState.readMem,
    BlockState.readMemTyped, MemCell.real, MemCell.of, MemCell.readAs]

/-- Success is part of the observed counterexample, not an assumption. -/
theorem full_kernel_executes : ∃ t, exec kernel initial = some t := by
  have h := stored_output
  cases he : exec kernel initial with
  | none => simp [he] at h
  | some t => exact ⟨t, rfl⟩

/-- The full original kernel does not compute length-N attention for all
tail sizes. This refutes the displayed real specification, not an FP atom. -/
refutation standard_attention_on_tail :
    (exec kernel initial).map (·.readMem "Out" 0) = some standardOutput := by
  rw [stored_output, standard_output]
  norm_num

/-- Reciprocal normalization leaves the same padded score in the denominator. -/
def reciprocalKernel : ComputeKernel :=
  (TypedSource.reciprocal "Q" "K" "V" "QS" "KS" "Out"
    128 128 128 1 128 128 128 1 128 128 128 1 128 128 128 1
    1 1 1 128 1 2 128 96 1).eraseDType

theorem reciprocal_stored_output :
    (exec reciprocalKernel initial).map (·.readMem "Out" 0) = some (1 / 2 : ℝ) := by
  have h := RealNormalization.executions_agree "Q" "K" "V" "QS" "KS" "Out"
    128 128 128 1 128 128 128 1 128 128 128 1 128 128 128 1
    1 1 1 128 1 2 128 96 1 initial
  change exec kernel initial = exec reciprocalKernel initial at h
  rw [← h]
  exact stored_output

theorem reciprocal_kernel_executes : ∃ t, exec reciprocalKernel initial = some t := by
  have h := reciprocal_stored_output
  cases he : exec reciprocalKernel initial with
  | none => simp [he] at h
  | some t => exact ⟨t, rfl⟩

refutation reciprocal_standard_attention_on_tail :
    (exec reciprocalKernel initial).map (·.readMem "Out" 0) = some standardOutput := by
  rw [reciprocal_stored_output, standard_output]
  norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.TailCounterexample
