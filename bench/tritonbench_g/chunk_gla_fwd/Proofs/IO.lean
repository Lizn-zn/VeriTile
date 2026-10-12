import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Single-output IO shared by this case's five kernels. The index type can
be a matrix lane or a streamed column/lane pair. Specifications supply their
own input arrays, grid, output addresses, masks and active-access bounds.
No final memory or output equation is assumed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO
open VeriTile.Triton
noncomputable section

structure IO (Grid Input Index : Type) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  grid : BlockState → Grid
  read : BlockState → Input
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  out : RegionName
  write : Grid → Index → Nat
  mask : Grid → Index → Prop

def IO.writes {G I X : Type} (io : IO G I X) (A : FlatAlloc) (g : G) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ i, io.mask g i ∧ cell.2 = A.addr io.out (io.write g i)

/-- The flat adapter requires zero unspecified real values, disjoint allocation
intervals and bounds for actual active accesses. Successful execution, the
independent formula and preservation of all other tagged cells are conclusions.
The original source does not itself request zero padding. -/
def IO.Implements {G I X : Type} (io : IO G I X) (f : G → I → X → ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, s.undef = (fun _ _ => 0) → io.valid s → io.bounds s A.extent →
    ∃ t, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some t ∧
      (∀ i, io.mask (io.grid s) i →
        t.readMem A.flat (A.addr io.out (io.write (io.grid s) i)) = f (io.grid s) (io.read s) i) ∧
      BlockState.WriteWithin (io.writes A (io.grid s)) (A.flattenState s) t

scoped notation:25 io " ⊨ " f => IO.Implements io f

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO
