import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealFlat

/-! Real IO for one indexed split-K GEMV program. Input pins describe the
selected rank slice of independent X/W arrays and, for atomic addition, the
initial output. The output allocation appears once; signed -1 skips all data
windows. No precondition constrains an intermediate or computed output. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIO
open VeriTile.Triton RealSetupSource
noncomputable section
set_option maxHeartbeats 2400000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Inputs (p : Parameters) where
  vector : Fin p.K → ℝ
  weights : Fin p.N → Fin p.K → ℝ
  initialOutput : Fin p.N → ℝ

def rank (p : Parameters) (split c e : Nat) : Nat :=
  c * (p.BK * p.splits) + (e + split * p.BK)

/-- Rank indices actually read by this program's split. K-tail lanes do not
enter the input windows because their indices cannot inhabit Fin K. -/
def RankActive (p : Parameters) (split : Nat) (k : Fin p.K) : Prop :=
  ∃ c : Fin (RealLoop.numKIters p.K (p.BK * p.splits)), ∃ e : Fin p.BK,
    rank p split c.val e.val = k.val

structure IO (p : Parameters) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  readIndex : Nat → Nat
  readInput : Nat → Fin p.K → Nat
  readWeight : Int → Fin p.N → Fin p.K → Nat
  write : Nat → Fin p.N → Nat

structure IO.Bounds {p : Parameters} (io : IO p) (bounds : RegionBounds)
    (split batch : Nat) (slot : Int) : Prop where
  metadata : io.readIndex batch < bounds (Region.cast p.indices)
  vector : slot ≠ -1 → ∀ k, RankActive p split k → io.readInput batch k < bounds p.input
  weights : slot ≠ -1 → ∀ n k, RankActive p split k → io.readWeight slot n k < bounds p.weights
  output : slot ≠ -1 → ∀ n, io.write batch n < bounds p.output

structure IO.Pins {p : Parameters} (io : IO p) (s : BlockState) (slot : Int) (xs : Inputs p) : Prop where
  metadata : s.readMemValue .int (Region.cast p.indices) (io.readIndex (s.pids 1)) = slot
  vector : slot ≠ -1 → ∀ k, RankActive p (s.pids 0) k →
    s.readMem p.input (io.readInput (s.pids 1) k) = xs.vector k
  weights : slot ≠ -1 → ∀ n k, RankActive p (s.pids 0) k →
    s.readMem p.weights (io.readWeight slot n k) = xs.weights n k
  initialOutput : slot ≠ -1 → p.splits ≠ 1 → ∀ n,
    s.readMem p.output (io.write (s.pids 1) n) = xs.initialOutput n

/-- Successful flat execution, active output values and full-cell frame.
Output is a single allocation, even when it supplies the initial atomic value.
The independent formula may depend on the program's split ID. -/
def IO.Implements {p : Parameters} (io : IO p) (f : Nat → Inputs p → Fin p.N → ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint →
    A.regions = [p.input, p.weights, p.output, Region.cast p.indices] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (slot : Int) (xs : Inputs p),
    (slot = -1 ∨ 0 ≤ slot) → s.undef = (fun _ _ => 0) →
    io.Bounds A.extent (s.pids 0) (s.pids 1) slot → io.Pins s slot xs →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ n, slot ≠ -1 → final.readMem A.flat
        (A.addr p.output (io.write (s.pids 1) n)) = f (s.pids 0) xs n) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ n, slot ≠ -1 → o ≠ A.addr p.output (io.write (s.pids 1) n)) →
        final.mem r o = (A.flattenState s).mem r o)

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) :
    (p.source storage out_dtype swapped).eraseDType.toAlgorithm? =
      .ok (p.source storage out_dtype swapped).eraseDType.toAlgKernel := by
  simp only [Parameters.source, ComputeKernel.eraseDType, RealSource.source_projection,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) : IO p where
  kernel := (p.source storage out_dtype swapped).eraseDType
  projection := projection p storage out_dtype swapped
  readIndex := fun batch => batch
  readInput := fun batch k => batch * p.inputBatchStride + k.val
  readWeight := fun slot n k => p.slotStride * slot.toNat + n.val * p.rowStride + k.val * p.colStride
  write := fun batch n => batch * p.outputBatchStride + n.val * p.outputStride

/-- Independent masked rank-slice contraction, scaled and optionally added
to initial output. Precision erasure gives this a real-valued interpretation. -/
def expected (p : Parameters) (split : Nat) (xs : Inputs p) (n : Fin p.N) : ℝ :=
  let v := p.scaling * ∑ c ∈ Finset.range (RealLoop.numKIters p.K (p.BK * p.splits)),
    ∑ e : Fin p.BK,
      if hk : rank p split c e.val < p.K then
        xs.vector ⟨rank p split c e.val, hk⟩ * xs.weights n ⟨rank p split c e.val, hk⟩
      else 0
  if p.splits = 1 then v else xs.initialOutput n + v

theorem formula_pins (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (swapped : Bool) (s : BlockState) (slot : Nat) (xs : Inputs p)
    (hp : (sourceIO p storage out_dtype swapped).Pins s (Int.ofNat slot) xs) (n : Fin p.N) :
    RealProgram.formula p s slot n.val = expected p (s.pids 0) xs n := by
  have hn : Int.ofNat slot ≠ -1 := by change (slot : ℤ) ≠ -1; omega
  have hv : RealProgram.contribution p s slot n.val =
      p.scaling * ∑ c ∈ Finset.range (RealLoop.numKIters p.K (p.BK * p.splits)),
        ∑ e : Fin p.BK,
          if hk : rank p (s.pids 0) c e.val < p.K then
            xs.vector ⟨rank p (s.pids 0) c e.val, hk⟩ * xs.weights n ⟨rank p (s.pids 0) c e.val, hk⟩
          else 0 := by
    unfold RealProgram.contribution
    apply congrArg (p.scaling * ·)
    apply Finset.sum_congr rfl
    intro c hc
    apply Finset.sum_congr rfl
    intro e _
    by_cases hk : rank p (s.pids 0) c e.val < p.K
    · have ha : RankActive p (s.pids 0) ⟨rank p (s.pids 0) c e.val, hk⟩ :=
        ⟨⟨c, Finset.mem_range.mp hc⟩, e, rfl⟩
      have hx := hp.vector hn _ ha
      have hw := hp.weights hn n _ ha
      simp only [rank] at hk
      simp only [rank, if_pos hk, dif_pos hk]
      exact congrArg₂ (· * ·) hx hw
    · simp only [rank] at hk
      simp only [rank, if_neg hk, dif_neg hk]
  simp only [RealProgram.formula, expected, hv]
  by_cases hs : p.splits = 1
  · simp only [hs, if_true]
  · simp only [hs, if_false]
    exact congrArg (· + _) (hp.initialOutput hn hs n)

/-- Both signed branches satisfy the same explicit IO contract. No success
assumption or intermediate/output equation is supplied by the caller. -/
theorem implements (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (swapped : Bool) (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) (hN : p.N ≤ p.BN) :
    sourceIO p storage out_dtype swapped ⊨ expected p := by
  intro A hd _ hcov s slot xs hvalid hu hb hp
  rcases hvalid with rfl | hpositive
  · obtain ⟨final, he, hf⟩ := RealFlat.sentinel_skips A hd hcov p storage out_dtype swapped
      s hp.metadata hu hb.metadata
    refine ⟨final, he, ?_, ?_⟩
    · intro n hbad; exact False.elim (hbad rfl)
    · intro r o _; exact congrFun (congrFun hf r) o
  · lift slot to Nat using hpositive
    have hn : (slot : ℤ) ≠ -1 := by omega
    have hab : RealSafety.ActiveBounds A.extent p s slot := by
      constructor
      · intro c e hk
        exact hb.vector hn ⟨_, hk⟩ ⟨c, e, rfl⟩
      · intro c n e hn' hk
        exact hb.weights hn ⟨n.val, hn'⟩ ⟨_, hk⟩ ⟨c, e, rfl⟩
      · intro n hn'
        exact hb.output hn ⟨n.val, hn'⟩
    obtain ⟨final, he, hv, hf⟩ := RealFlat.source_executes A hd hcov p storage out_dtype swapped
      s slot hp.metadata hu hb.metadata hab hS hstride hN
    refine ⟨final, he, ?_, ?_⟩
    · intro n _
      exact (hv n.val n.isLt).trans (formula_pins p storage out_dtype swapped s slot xs hp n)
    · intro r o hmiss
      apply hf r o
      rcases hmiss with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun n hn' => ho ⟨n.val, hn'⟩ hn)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIO
