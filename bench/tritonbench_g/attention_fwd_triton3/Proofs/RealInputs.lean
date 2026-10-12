import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealAddressBounds

/-! Six mathematical input views and their relation to initial memory.
Only active Q rows are pinned. M/L/Out initial values are read only when
INIT=0; initializing ignores those arrays. No output equation is a premise. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInputs
open VeriTile.Triton RealPrologue
open scoped BigOperators
noncomputable section

structure Inputs (BM BD NK : Nat) where
  query : Fin BM → Fin BD → ℝ
  key : Fin NK → Fin BD → ℝ
  value : Fin NK → Fin BD → ℝ
  center : Fin BM → ℝ
  denominator : Fin BM → ℝ
  output : Fin BM → Fin BD → ℝ

structure Pins (BM BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) : Prop where
  query : ∀ row lane, (p.evenM ≠ 0 ∨ s.pids 0 * BM + row.val < p.layout.queryContext) →
    s.readMem p.layout.Q (RealAddressBounds.queryAddress BM p s row.val lane.val) = xs.query row lane
  key : ∀ key lane,
    s.readMem p.layout.K (RealAddressBounds.keyAddress p s key.val lane.val) = xs.key key lane
  value : ∀ key lane,
    s.readMem p.layout.V (RealAddressBounds.valueAddress p s key.val lane.val) = xs.value key lane
  center : p.init = 0 → ∀ row,
    s.readMem p.layout.M (RealExecution.rowAddress BM p s (row, PUnit.unit)) = xs.center row
  denominator : p.init = 0 → ∀ row,
    s.readMem p.layout.L (RealExecution.rowAddress BM p s (row, PUnit.unit)) = xs.denominator row
  output : p.init = 0 → ∀ row lane,
    s.readMem p.layout.Out (RealExecution.outputAddress BM BD p s (row, lane, PUnit.unit)) = xs.output row lane

/-- Every initial state supplies these six views, including in-place carry.
The pinning contract therefore imposes no circular condition on final outputs. -/
def read (BM BD : Nat) (p : Parameters) (s : BlockState) : Inputs BM BD p.layout.kvContext where
  query := fun row lane => s.readMem p.layout.Q (RealAddressBounds.queryAddress BM p s row.val lane.val)
  key := fun key lane => s.readMem p.layout.K (RealAddressBounds.keyAddress p s key.val lane.val)
  value := fun key lane => s.readMem p.layout.V (RealAddressBounds.valueAddress p s key.val lane.val)
  center := fun row => s.readMem p.layout.M (RealExecution.rowAddress BM p s (row, PUnit.unit))
  denominator := fun row => s.readMem p.layout.L (RealExecution.rowAddress BM p s (row, PUnit.unit))
  output := fun row lane => s.readMem p.layout.Out (RealExecution.outputAddress BM BD p s (row, lane, PUnit.unit))

theorem read_pins (BM BD : Nat) (p : Parameters) (s : BlockState) : Pins BM BD p s (read BM BD p s) :=
  ⟨fun _ _ _ => rfl, fun _ _ => rfl, fun _ _ => rfl,
    fun _ _ => rfl, fun _ _ => rfl, fun _ _ _ => rfl⟩

def query (BM BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (row : Fin BM) (lane : Fin BD) : ℝ :=
  if p.evenM ≠ 0 ∨ start * BM + row.val < p.layout.queryContext then xs.query row lane else 0

def initial (BM BD : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) :
    RealLoopFormula.Carry BM BD where
  center := fun row => if p.init ≠ 0 then ⊥ else some (xs.center row)
  denominator := fun row => if p.init ≠ 0 then 1 else xs.denominator row
  numerator := fun i => if p.init ≠ 0 then 0 else xs.output i.1 i.2.1

def score (BM BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (row : Fin BM) (key : Fin p.layout.kvContext) : ℝ :=
  (∑ lane : Fin BD, query BM BD start p xs row lane * xs.key key lane) *
    (p.scale * 1.4426950408889634)

theorem query_pinned (BM BN BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs)
    (row : Fin BM) (lane : Fin BD) :
    (loopParameters BM BN BD p s).q (row, lane, PUnit.unit) = query BM BD (s.pids 0) p xs row lane := by
  by_cases he : p.evenM ≠ 0
  · simpa [loopParameters, RealAccumulation.values, RealAccumulation.checks, RealPointers.queryPointer,
      RealHeader.coordinates, query, he, RealAddressBounds.queryAddress] using hp.query row lane (Or.inl he)
  · by_cases hr : s.pids 0 * BM + row.val < p.layout.queryContext
    · simpa [loopParameters, RealAccumulation.values, RealAccumulation.checks, RealPointers.queryPointer,
        RealHeader.coordinates, query, he, hr, RealAddressBounds.queryAddress] using hp.query row lane (Or.inr hr)
    · simp [loopParameters, RealAccumulation.values, RealAccumulation.checks, RealPointers.queryPointer,
        RealHeader.coordinates, query, he, hr]

theorem key_pinned (BM BN BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs)
    (key : Fin p.layout.kvContext) (lane : Fin BD) :
    RealLoopContext.keyInput s (loopParameters BM BN BD p s) key.val lane = xs.key key lane :=
  hp.key key lane

theorem value_pinned (BM BN BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs)
    (key : Fin p.layout.kvContext) (lane : Fin BD) :
    RealLoopContext.valueInput s (loopParameters BM BN BD p s) key.val lane = xs.value key lane :=
  hp.value key lane

theorem seed_pinned (BM BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs) :
    seed BM BD p s = initial BM BD p xs := by
  by_cases hi : p.init ≠ 0
  · simp [seed, RealInit.seed, initial, hi]
  · have he : p.init = 0 := by omega
    apply RealLoopFormula.Carry.ext
    · funext row
      simp only [seed, RealInit.seed, initial, if_neg hi]
      exact congrArg (fun x : ℝ => (some x : WithBot ℝ)) (hp.center he row)
    · funext row
      simpa [seed, RealInit.seed, initial, hi, RealInit.pointerValues, RealPointers.rowPointer,
        RealExecution.rowAddress] using hp.denominator he row
    · funext i
      rcases i with ⟨row, lane, ⟨⟩⟩
      simpa [seed, RealInit.seed, initial, hi, RealAccumulation.values, RealPointers.outputPointer,
        RealExecution.outputAddress, RealEpilogue.blockAddress, RealExecution.outputPointer] using hp.output he row lane

theorem score_pinned (BM BN BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs)
    (row : Fin BM) (key : Fin p.layout.kvContext) :
    RealLoopContext.fullScore s (loopParameters BM BN BD p s) row key.val =
      score BM BD (s.pids 0) p xs row key := by
  unfold RealLoopContext.fullScore score
  congr 1
  apply Finset.sum_congr rfl
  intro lane _
  rw [query_pinned BM BN BD p s xs hp, key_pinned BM BN BD p s xs hp]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInputs
