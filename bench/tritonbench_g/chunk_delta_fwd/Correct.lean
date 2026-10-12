import bench.tritonbench_g.chunk_delta_fwd.Kernels
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Real correctness of both complete kernels in Kernels. History stores H
before each update; corrected V uses that frozen H; the optional final store
writes H after all outer chunks. Both state switches and all storage choices
are retained. This is a per-program real contract with mathematical addresses.

The IO relation requires zero unspecified padding values, disjoint physical
allocations, active address bounds and the valid-launch conditions below.
Omitted Triton padding itself does not guarantee zero. Proofs/UndefinedTail.lean
refutes removing that premise: T=1 and 16×16 tiles with all-zero inputs can
store final H[0,0]=15 when unspecified K/V tail lanes are one. No claim about concrete
IEEE arithmetic or a composed CUDA grid is made here. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.Correctness
open VeriTile.Triton RealProgram
open scoped BigOperators VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealIO
noncomputable section

/-- Dimensions, tiles, iteration count and all eight source strides. -/
abbrev Parameters := Layout

/-- Bounds check only active cells. The two input matrices use the original
K×T / T×K views. History uses historyTime as its row stride, while seed and
final state are contiguous K×V. The valid launch owns the whole K dimension;
active corrected/history/final addresses must not collide within each output.
All input arrays are read from initial memory, before any output stores. -/
def originalIO (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) : RealIO.IO p where
  kernel := (TypedSource.original kd value initial final r.key r.value r.delta r.corrected r.history r.initial r.final
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal).eraseDType
  projection := RealIO.projection p r kd value initial final Bool.false useSeed storeFinal
  read := fun s => ⟨s.readMem r.key,s.readMem r.delta,s.readMem r.value⟩
  seed := fun s => s.readMem r.initial
  valid := fun s => s.pids 0 = 0 ∧ RealInnerLoop.NoCollision p (RealInit.launch s) ∧
    RealOuterLoop.HistoryNoCollision p (RealInit.launch s) ∧
    (storeFinal = Bool.true → RealStateStore.NoCollision p (RealInit.launch s) (RealInit.base p (RealInit.launch s)) p.V)
  bounds := fun s => RealSafety.Bounds p (RealInit.launch s) r useSeed storeFinal
  history := r.history
  corrected := r.corrected
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => RealOuterStep.historyAddress p l c e.val j.val
  valueWindow := fun l t j => valueAddress p l t j.val
  finalWindow := fun l e j => RealStateStore.address p l (RealInit.base p l) p.V e.val j.val
  stateMask := fun l e j => RealStateStore.active p l e.val j.val
  valueMask := fun l t j => activeValue p l t j.val

def sumLeftIO (p : Parameters) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) : RealIO.IO p :=
  { originalIO p r kd value initial final useSeed storeFinal with
    kernel := (TypedSource.sumLeft kd value initial final r.key r.value r.delta r.corrected r.history r.initial r.final
      p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
      p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal).eraseDType
    projection := RealIO.projection p r kd value initial final Bool.true useSeed storeFinal }

/-- Independent matrix recurrence (RealWindow.state):
H₀ = seed when enabled, otherwise zero;
C_c(t,j) = V(t,j) - Σ_{e<K} D(t,e) H_c(e,j);
H_{c+1}(e,j) = H_c(e,j) + Σ_{t in [c*BT,c*BT+width), t<T} K(t,e) C_c(t,j).

width = ceil(BT/BC)*BC, so adjacent windows may overlap. The final corrected
value at row t uses H_(t/BT), because the last visiting chunk overwrites it.
There is no assumption that BC divides BT. -/
def expected (p : Parameters) (useSeed : Bool) (l : Launch) (a : Inputs) (a₀ : Nat → ℝ) : RealIO.Outputs p :=
  let xs : RealMath.Inputs :=
    { key := fun t e => a.key (l.head*p.keyHead + e*p.keyDim + t*p.keyTime)
      delta := fun t e => a.delta (l.head*p.keyHead + t*p.keyTime + e*p.keyDim)
      value := fun t j => a.value (l.head*p.valueHead + t*p.valueTime + j*p.valueDim)
      seed := fun e j => a₀ (l.head*p.K*p.V + e*p.V + j) }
  let width := ((p.BT+p.BC-1)/p.BC)*p.BC
  let state := RealWindow.state xs p.T p.K p.BT width useSeed
  { history := fun c e j => state c e.val (l.value*p.BV+j.val)
    corrected := fun t j => xs.value t (l.value*p.BV+j.val) -
      ∑ e : Fin p.K, xs.delta t e.val * state (t/p.BT) e.val (l.value*p.BV+j.val)
    final := fun e j => state p.NT e.val (l.value*p.BV+j.val) }

/-- Inputs K/D/V remain unchanged through every loop iteration. History and
corrected V are distinct; when final storage is enabled it is distinct from
both. Seed may alias outputs because initialization reads it before all stores. -/
specification chunk_delta_correctness (p : Parameters) (r : Regions)
    (kd value initial final : TypedSource.Storage) (useSeed storeFinal : Bool)
    (hK : p.K ≤ p.BK) (hBT : 0 < p.BT) (hBC : 0 < p.BC) (hcover : p.T ≤ p.NT*p.BT)
    (hs : RealOuterStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final) :
    Spec.Real (originalIO p r kd value initial final useSeed storeFinal ⊨ expected p useSeed) := by
  exact RealIO.implements p r kd value initial final Bool.false useSeed storeFinal hK hBT hBC hcover hs hfsep

specification chunk_sum_left_correctness (p : Parameters) (r : Regions)
    (kd value initial final : TypedSource.Storage) (useSeed storeFinal : Bool)
    (hK : p.K ≤ p.BK) (hBT : 0 < p.BT) (hBC : 0 < p.BC) (hcover : p.T ≤ p.NT*p.BT)
    (hs : RealOuterStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final) :
    Spec.Real (sumLeftIO p r kd value initial final useSeed storeFinal ⊨ expected p useSeed) := by
  exact RealIO.implements p r kd value initial final Bool.true useSeed storeFinal hK hBT hBC hcover hs hfsep

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.Correctness
