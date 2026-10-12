import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealFlat

/-! Real IO for one indexed GEMV program. The vector window is independent of
the output length; weights and initial output are independent input values.
The output allocation is listed once even though it is also read when adding.
Signed -1 skips the active windows; other legal indices are nonnegative. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealIO
open VeriTile.Triton RealSetupSource RealLoop
noncomputable section
set_option maxHeartbeats 2200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Inputs (p : Parameters) where
  vector : Fin p.BK → ℝ
  weights : Fin p.length → Fin p.K → ℝ
  initialOutput : Fin p.length → ℝ

/-- The read/write windows of the complete source, not its execution results.
Weights are indexed by output row and rank K; the input vector additionally
covers BK padding when EVEN_K selects an unmasked load. -/
structure IO (p : Parameters) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  even : Bool
  add : Bool
  readIndex : Nat → Nat
  readInput : Nat → Fin p.BK → Nat
  readWeight : Nat → Int → Fin p.length → Fin p.K → Nat
  write : Nat → Nat → Fin p.length → Nat

def IO.InputActive {p : Parameters} (io : IO p) (k : Fin p.BK) : Prop :=
  io.even = Bool.true ∨ k.val < p.K

structure IO.Bounds {p : Parameters} (io : IO p) (bounds : RegionBounds)
    (split batch : Nat) (slot : Int) : Prop where
  metadata : io.readIndex batch < bounds (Region.cast p.indices)
  vector : slot ≠ -1 → ∀ k, io.InputActive k → io.readInput batch k < bounds p.input
  weights : slot ≠ -1 → ∀ g k, io.readWeight split slot g k < bounds p.weights
  output : slot ≠ -1 → ∀ g, io.write split batch g < bounds p.output

structure IO.Pins {p : Parameters} (io : IO p) (s : BlockState) (slot : Int) (xs : Inputs p) : Prop where
  metadata : s.readMemValue .int (Region.cast p.indices) (io.readIndex (s.pids 1)) = slot
  vector : slot ≠ -1 → ∀ k, io.InputActive k → s.readMem p.input (io.readInput (s.pids 1) k) = xs.vector k
  weights : slot ≠ -1 → ∀ g k, s.readMem p.weights (io.readWeight (s.pids 0) slot g k) = xs.weights g k
  initialOutput : slot ≠ -1 → io.add = Bool.true → ∀ g,
    s.readMem p.output (io.write (s.pids 0) (s.pids 1) g) = xs.initialOutput g

/-- Successful flat execution, active output values and the full-cell frame.
Initial output is a read input. No premise constrains a computed output or
requires an intermediate program execution. The allocation list has four
regions: vector, weights, output, signed metadata; output is not duplicated. -/
def IO.Implements {p : Parameters} (io : IO p) (f : Inputs p → Fin p.length → ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint →
    A.regions = [p.input, p.weights, p.output, Region.cast p.indices] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (slot : Int) (xs : Inputs p),
    (slot = -1 ∨ 0 ≤ slot) →
    s.undef = (fun _ _ => 0) →
    io.Bounds A.extent (s.pids 0) (s.pids 1) slot → io.Pins s slot xs →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ g, slot ≠ -1 → final.readMem A.flat (A.addr p.output (io.write (s.pids 0) (s.pids 1) g)) = f xs g) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ g, slot ≠ -1 → o ≠ A.addr p.output (io.write (s.pids 0) (s.pids 1) g)) →
        final.mem r o = (A.flattenState s).mem r o)

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Parameters) (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) :
    (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgorithm? =
      .ok (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel := by
  simp only [Parameters.source, ComputeKernel.eraseDType, RealSource.source_projection,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Parameters) (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) : IO p where
  kernel := (p.source input weight out_dtype cast swapped even add).eraseDType
  projection := projection p input weight out_dtype cast swapped even add
  even := even
  add := add
  readIndex := fun batch => batch
  readInput := fun batch k => batch*p.inputBatchStride+k.val*p.inputStride
  readWeight := fun split slot g k => p.slotStride*slot.toNat+split*p.length*p.rowStride+g.val*p.rowStride+k.val*p.colStride
  write := fun split batch g => batch*p.outputBatchStride+split*p.length+p.slice*p.outputStride+g.val*p.outputStride

/-- Independent mathematical formula over full rank K and initial output.
The inequality embeds each mathematical rank index in the loaded BK vector. -/
def expected (p : Parameters) (hK : p.K ≤ p.BK) (add : Bool) (xs : Inputs p) (g : Fin p.length) : ℝ :=
  let v := ∑ k : Fin p.K, xs.vector ⟨k.val, lt_of_lt_of_le k.isLt hK⟩ * xs.weights g k
  if add then v+xs.initialOutput g else v

theorem formula_pins (p : Parameters) (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) (s : BlockState) (slot : Nat)
    (hK : p.K ≤ p.BK) (xs : Inputs p)
    (hp : (sourceIO p input weight out_dtype cast swapped even add).Pins s (Int.ofNat slot) xs)
    (g : Fin p.length) :
    RealProgram.formula p s slot add g.val = expected p hK add xs g := by
  have hn : Int.ofNat slot ≠ -1 := by change (slot : ℤ) ≠ -1; omega
  have hdot : (∑ k : Fin p.K,
      s.readMem p.input (s.pids 1*p.inputBatchStride+k.val*p.inputStride) *
      s.readMem p.weights (p.slotStride*slot+s.pids 0*p.length*p.rowStride+g.val*p.rowStride+k.val*p.colStride)) =
      ∑ k : Fin p.K, xs.vector ⟨k.val, lt_of_lt_of_le k.isLt hK⟩ * xs.weights g k := by
    apply Finset.sum_congr rfl
    intro k _
    have hx := hp.vector hn ⟨k.val, lt_of_lt_of_le k.isLt hK⟩ (Or.inr k.isLt)
    have hw := hp.weights hn g k
    change s.readMem p.input (s.pids 1*p.inputBatchStride+k.val*p.inputStride) = _ at hx
    change s.readMem p.weights (p.slotStride*slot+s.pids 0*p.length*p.rowStride+g.val*p.rowStride+k.val*p.colStride) = _ at hw
    rw [hx, hw]
  simp only [RealProgram.formula, expected, hdot]
  cases ha : add
  · rfl
  · have ho := hp.initialOutput hn (by exact ha) g
    change s.readMem p.output
      (s.pids 1*p.outputBatchStride+s.pids 0*p.length+p.slice*p.outputStride+g.val*p.outputStride) = _ at ho
    simp only [if_true, ho]

/-- Assemble the independent input formula with the complete source's flat
execution theorem. The active and sentinel branches both retain full frames. -/
theorem implements (p : Parameters) (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool)
    (hBN : 0 < p.BN) (hstride : 0 < p.outputStride) (hK : p.K ≤ p.BK)
    (hsplits : 0 < p.splits) (hw : p.weights ≠ p.output) :
    sourceIO p input weight out_dtype cast swapped even add ⊨ expected p hK add := by
  intro A hd _ hcov s slot xs hvalid hu hb hp
  rcases hvalid with rfl | hpositive
  · obtain ⟨final, he, hf⟩ := RealFlat.sentinel_skips A hd hcov p input weight out_dtype
      cast swapped even add s hp.metadata hu hb.metadata
    refine ⟨final, he, ?_, ?_⟩
    · intro g hbad; exact False.elim (hbad rfl)
    · intro r o _; exact congrFun (congrFun hf r) o
  · lift slot to Nat using hpositive
    have hn : (slot : ℤ) ≠ -1 := by omega
    have hab : RealSafety.ActiveBounds A.extent p s slot even := by
      constructor
      · exact hb.vector hn
      · intro g hg k hk
        exact hb.weights hn ⟨g,hg⟩ ⟨k.val,hk⟩
      · intro g hg
        exact hb.output hn ⟨g,hg⟩
    obtain ⟨final, he, hv, hf⟩ := RealFlat.source_executes A hd hcov p input weight out_dtype
      cast swapped even add s slot hp.metadata hu hb.metadata hab hBN hstride hK hsplits hw
    refine ⟨final, he, ?_, ?_⟩
    · intro g _
      exact (hv g.val g.isLt).trans (formula_pins p input weight out_dtype cast swapped even add s slot hK xs hp g)
    · intro r o hmiss
      apply hf r o
      rcases hmiss with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun g hg => ho ⟨g,hg⟩ hn)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealIO
