import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealSetup

/-! Complete real block-state execution of the actual source. The independent
formula sums the full rank K and optionally adds initial output memory. This
covers arbitrary valid sizes/strides and both versions, not just the loop body.
Allocation bounds and transport to flat memory are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealProgram
open VeriTile.Triton RealLoop RealSetupSource RealSetup
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Independent rank-K GEMV formula, including the source's exact output
window and optional initial output. The split contribution has no output
stride multiplier, matching the actual kernel. -/
def formula (p : Parameters) (initial : BlockState) (slot : Nat) (add : Bool) (g : Nat) : ℝ :=
  let v := ∑ k : Fin p.K,
    initial.readMem p.input (initial.pids 1 * p.inputBatchStride + k.val * p.inputStride) *
    initial.readMem p.weights
      (p.slotStride * slot + initial.pids 0 * p.length * p.rowStride + g * p.rowStride + k.val * p.colStride)
  if add then v + initial.readMem p.output
    (initial.pids 1 * p.outputBatchStride + initial.pids 0 * p.length + p.slice * p.outputStride + g * p.outputStride)
  else v

private theorem masked_rank_sum (x w : Nat → ℝ) (K BK : Nat) (hK : K ≤ BK) :
    (∑ k : Fin BK, x k.val * (if k.val < K then w k.val else 0)) =
      ∑ k : Fin K, x k.val * w k.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => x k * (if k < K then w k else 0)) BK,
    Fin.sum_univ_eq_sum_range (fun k => x k * w k) K]
  simp only [mul_ite, mul_zero]
  calc
    (∑ k ∈ Finset.range BK, if k < K then x k * w k else 0) =
        ∑ k ∈ Finset.range K, if k < K then x k * w k else 0 := by
      symm
      apply Finset.sum_subset (Finset.range_mono hK)
      intro k _ hk
      exact if_neg (by simpa only [Finset.mem_range] using hk)
    _ = ∑ k ∈ Finset.range K, x k * w k := by
      apply Finset.sum_congr rfl
      intro k hk
      simp only [if_pos (Finset.mem_range.mp hk)]

theorem expected_formula (p : Parameters) (initial : BlockState) (slot : Nat)
    (cast even add : Bool) (g : Nat) (hK : p.K ≤ p.BK) :
    (p.geometry initial slot).expected (inputValues p initial even)
      (afterSetup p initial slot cast even) add g = formula p initial slot add g := by
  simp only [Geometry.expected, setup_read]
  have ha : inputValues p initial even =
      (fun k => if even then inputValues p initial Bool.true k
        else if k.val < p.K then inputValues p initial Bool.true k else 0) := by
    funext k
    cases even <;> by_cases hk : k.val < p.K <;> simp [inputValues, hk]
  rw [ha]
  dsimp only [Parameters.geometry]
  rw [RealBlock.dot_input_mask]
  have hd := masked_rank_sum
    (fun k => initial.readMem p.input (initial.pids 1 * p.inputBatchStride + k * p.inputStride))
    (fun k => initial.readMem p.weights
      (p.slotStride * slot + initial.pids 0 * p.length * p.rowStride + g * p.rowStride + k * p.colStride))
    p.K p.BK hK
  simpa only [RealBlock.dot, inputValues, Bool.true_or, if_true, formula,
    Geometry.outputAddress, Parameters.geometry] using congrArg
    (fun v => if add then v + initial.readMem p.output
      ((p.geometry initial slot).outputAddress g) else v) hd

theorem body_active (p : Parameters) (initial : BlockState) (slot : Nat)
    (cast swapped even add : Bool)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = Int.ofNat slot)
    (hBN : 0 < p.BN) (hstride : 0 < p.outputStride) (hw : p.weights ≠ p.output) :
    ∃ final, stepStmts (p.body cast swapped even add) initial = some final ∧
      (∀ g, g < p.length → final.readMem p.output ((p.geometry initial slot).outputAddress g) =
        (p.geometry initial slot).expected (inputValues p initial even)
          (afterSetup p initial slot cast even) add g) ∧
      (∀ r o, (p.geometry initial slot).Misses r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := (p.geometry initial slot).executes
    (inputValues p initial even) (afterSetup p initial slot cast even) add swapped
    hBN hstride hw (setup_registers p initial slot cast even)
  refine ⟨final, ?_, hv, ?_⟩
  · unfold Parameters.body
    rw [stepStmts.append_some (prelude_exec p initial (Int.ofNat slot) hm)]
    have hg : evalOp Parameters.guard (afterPrelude initial (Int.ofNat slot)) =
        some (Tile.scalar Bool.true) := by
      have h := guard_eval (afterPrelude initial (Int.ofNat slot)) (Int.ofNat slot)
        (by simp [afterPrelude])
      have hn : Int.ofNat slot ≠ -1 := by
        change (slot : ℤ) ≠ -1
        omega
      simpa only [decide_eq_true hn] using h
    have hb : stepStmt (.ifThen Parameters.guard (p.setup cast even ++ [p.loop add swapped]))
        (afterPrelude initial (Int.ofNat slot)) = some final := by
      simp only [stepStmt, hg, Option.bind_some, Tile.scalar_data, if_true]
      rw [stepStmts.append_some (setup_exec p initial slot cast even)]
      change stepStmts [(p.geometry initial slot).statement add swapped] _ = some final
      rw [stepStmts.cons_some he, stepStmts.nil]
    rw [stepStmts.cons_some hb, stepStmts.nil]
  · intro r o hmiss
    simpa only [setup_mem] using hf r o hmiss

/-- Full actual source execution for a nonnegative active slot. No equation
about output memory is a premise. The full K sum requires K ≤ BLOCK_K;
positive output stride prevents repeated writes within this program. -/
theorem source_executes (p : Parameters) (input : TypedSource.InputStorage)
    (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast swapped even add : Bool) (initial : BlockState) (slot : Nat)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = Int.ofNat slot)
    (hBN : 0 < p.BN) (hstride : 0 < p.outputStride) (hK : p.K ≤ p.BK)
    (_hsplits : 0 < p.splits) (hw : p.weights ≠ p.output) :
    ∃ final, exec (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel initial = some final ∧
      (∀ g, g < p.length → final.readMem p.output ((p.geometry initial slot).outputAddress g) =
        formula p initial slot add g) ∧
      (∀ r o, (p.geometry initial slot).Misses r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := body_active p initial slot cast swapped even add hm hBN hstride hw
  refine ⟨final, ?_, ?_, hf⟩
  · simpa only [exec, Parameters.source_body] using he
  · intro g hg
    exact (hv g hg).trans (expected_formula p initial slot cast even add g hK)

/-- The signed -1 sentinel skips the complete active branch. No input or
weight load, loop step, old-output load or store is executed. All memory stays
unchanged, independently of dtype, flags, shape and stride values. -/
theorem sentinel_skips (p : Parameters) (input : TypedSource.InputStorage)
    (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast swapped even add : Bool) (initial : BlockState)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = -1) :
    ∃ final, exec (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel initial = some final ∧
      final.mem = initial.mem := by
  refine ⟨afterPrelude initial (-1), ?_, rfl⟩
  simp only [exec, Parameters.source_body, Parameters.body]
  rw [stepStmts.append_some (prelude_exec p initial (-1) hm)]
  have hg : evalOp Parameters.guard (afterPrelude initial (-1)) = some (Tile.scalar Bool.false) :=
    guard_eval _ (-1) (by simp [afterPrelude])
  have hb : stepStmt (.ifThen Parameters.guard (p.setup cast even ++ [p.loop add swapped]))
      (afterPrelude initial (-1)) = some (afterPrelude initial (-1)) := by
    simp only [stepStmt, hg, Option.bind_some, Tile.scalar_data, Bool.false_eq_true, if_false]
    rfl
  rw [stepStmts.cons_some hb, stepStmts.nil]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealProgram
