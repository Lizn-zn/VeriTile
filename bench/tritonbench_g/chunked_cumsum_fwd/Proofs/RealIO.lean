import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Flat
import VeriTile.Meta.Specification

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RealIO
open VeriTile.Triton Math Program Initialization
noncomputable section

structure Values (p : Layout) where
  prepared : Fin p.BH → Fin p.BC → ℝ
  cumulative : Fin p.BH → Fin p.BC → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  preparedOutput : RegionName
  cumulativeOutput : RegionName
  preparedWindow : Launch → Fin p.BH → Fin p.BC → Nat
  cumulativeWindow : Launch → Fin p.BH → Fin p.BC → Nat
  mask : Launch → Fin p.BH → Fin p.BC → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) : WriteFootprint := fun cell =>
  (cell.1 = io.preparedOutput ∧ ∃ i j, io.mask l i j ∧ cell.2 = io.preparedWindow l i j) ∨
  (cell.1 = io.cumulativeOutput ∧ ∃ i j, io.mask l i j ∧ cell.2 = io.cumulativeWindow l i j)

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, io.regionWrites l (rg,off) ∧ A.addr rg off = cell.2

/-- Successful full execution, both initial-array formulas, and preservation
of every other tagged cell. No unspecified-value-oracle premise is needed:
every source load provides an explicit `other=0`. -/
def IO.Implements {p : Layout} (io : IO p) (f : Launch → Inputs → Values p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, io.valid s → io.bounds s A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some v ∧
      (∀ i j, io.mask (launch s) i j →
        v.readMem A.flat (A.addr io.preparedOutput (io.preparedWindow (launch s) i j)) =
          (f (launch s) (io.read s)).prepared i j ∧
        v.readMem A.flat (A.addr io.cumulativeOutput (io.cumulativeWindow (launch s) i j)) =
          (f (launch s) (io.read s)).cumulative i j) ∧
      BlockState.WriteWithin (io.writes A (launch s)) (A.flattenState s) v

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgorithm? =
      .ok (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,Program.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) : IO p where
  kernel := (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType
  projection := projection p r bias swapped dtype out_dtype hasBias useSoftplus lower upper
  read := fun s => ⟨s.readMem r.dt,s.readMem r.rate,s.readMem r.bias⟩
  valid := fun s => (launch s).chunk*p.chunk ≤ p.sequence ∧
    Stores.NoCollision p (launch s) (Execution.outOffsets p (launch s)) ∧
    Stores.NoCollision p (launch s) (Execution.sumOffsets p (launch s))
  bounds := fun s => Safety.Bounds p (launch s) r hasBias
  preparedOutput := r.out
  cumulativeOutput := r.sum
  preparedWindow := fun l i j => outAddress p l i.val j.val
  cumulativeWindow := fun l i j => sumAddress p l i.val j.val
  mask := fun l i j => activeOutput p l i.val j.val

def expected (p : Layout) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (l : Launch) (a : Inputs) : Values p where
  prepared := fun i j => Math.prepared p l a hasBias useSoftplus lower upper i.val j.val
  cumulative := Math.cumulative p l a hasBias useSoftplus lower upper

theorem implements (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (hRA : r.rate ≠ r.out) (hOS : r.out ≠ r.sum) :
    sourceIO p r bias swapped dtype out_dtype hasBias useSoftplus lower upper ⊨
      expected p hasBias useSoftplus lower upper := by
  intro A hd hcov s hv hb
  obtain ⟨v,he,ho,hf⟩ := Flat.correct A hd hcov p r bias swapped dtype out_dtype
    ⟨s.readMem r.dt,s.readMem r.rate,s.readMem r.bias⟩ hasBias useSoftplus lower upper s
    (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hRA hOS hv.2.1 hv.2.2 hb
  exact ⟨v,he,ho,hf⟩

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RealIO
