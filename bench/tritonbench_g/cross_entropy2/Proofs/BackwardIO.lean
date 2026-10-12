import bench.tritonbench_g.cross_entropy2.Proofs.BackwardFlat
import VeriTile.Meta.Specification
namespace VeriTile.Bench.TritonBenchG.CrossEntropy2.BackwardIO
open VeriTile.Triton BackwardFormula BackwardProgram
noncomputable section

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  bounds : Launch → Inputs → RegionBounds → Prop
  out : RegionName
  write : Launch → Fin p.B → Nat
  mask : Launch → Fin p.B → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) : WriteFootprint := fun cell =>
  ∃ i, io.mask l i ∧ io.out=cell.1 ∧ io.write l i=cell.2
def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint := fun cell =>
  cell.1=A.flat ∧ ∃ region offset, io.regionWrites l (region,offset) ∧ A.addr region offset=cell.2

/-- Independent initial-input formula, successful finite flat execution and
the complete tagged-cell frame, under exactly the declared access bounds. -/
def IO.Implements {p : Layout} (io : IO p) (f : Launch → Inputs → Fin p.B → ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, io.bounds (launch u) (io.read u) A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ i, io.mask (launch u) i →
        v.readMem A.flat (A.addr io.out (io.write (launch u) i))=f (launch u) (io.read u) i) ∧
      BlockState.WriteWithin (io.writes A (launch u)) (A.flattenState u) v

scoped notation:25 io " ⊨ " f => IO.Implements io f

def sourceIO (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) : IO p where
  kernel := (source p r storage swapped).eraseDType
  projection := BackwardProgram.projection p r storage swapped
  read := fun u => inputs u r
  bounds := BackwardSafety.Bounds p r
  out := r.gradient
  write := fun l i => gradientAddress p l i.val
  mask := fun l i => active p l i.val

theorem implements (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) :
    sourceIO p r storage swapped ⊨ fun l a i => expected p l a i.val := by
  intro A hd hcov u hb
  exact BackwardFlat.correct A hd hcov p r storage swapped u hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropy2.BackwardIO
