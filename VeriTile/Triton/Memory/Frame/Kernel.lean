/-
VeriTile.Triton.Memory.Frame.Kernel

Per-program kernel frame contracts.
-/

import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Triton

namespace Kernel

/-- Successful single-program execution with an explicit write footprint. -/
structure ExecFrame (k : Kernel) (s : BlockState) where
  final : BlockState
  writes : WriteFootprint
  h_exec : exec k s = some final
  h_writeWithin : BlockState.WriteWithin writes s final

namespace ExecFrame

theorem mem_eq_of_not_written {k : Kernel} {s : BlockState}
    (frame : Kernel.ExecFrame k s)
    {region : RegionName} {offset : Nat}
    (hnot : ¬ frame.writes (region, offset)) :
    frame.final.mem region offset = s.mem region offset :=
  BlockState.WriteWithin.mem_eq_of_not_written frame.h_writeWithin hnot

theorem mem_eq_of_region_not_written {k : Kernel} {s : BlockState}
    (frame : Kernel.ExecFrame k s)
    {region : RegionName}
    (hnot : ∀ offset, ¬ frame.writes (region, offset)) :
    frame.final.mem region = s.mem region :=
  BlockState.WriteWithin.mem_eq_of_region_not_written frame.h_writeWithin hnot

end ExecFrame

/-- Reusable single-kernel frame predicate. If execution fails, this is vacuous;
use `ExecFrame` when a concrete successful final state is needed. -/
def ExecWritesWithin (k : Kernel) (s : BlockState) (P : WriteFootprint) : Prop :=
  ∀ s', exec k s = some s' → BlockState.WriteWithin P s s'

namespace ExecWritesWithin

theorem mem_eq_of_not_written {k : Kernel} {s s' : BlockState}
    {P : WriteFootprint}
    (hFrame : Kernel.ExecWritesWithin k s P)
    (hExec : exec k s = some s')
    {region : RegionName} {offset : Nat}
    (hnot : ¬ P (region, offset)) :
    s'.mem region offset = s.mem region offset :=
  BlockState.WriteWithin.mem_eq_of_not_written (hFrame s' hExec) hnot

theorem mem_eq_of_region_not_written {k : Kernel} {s s' : BlockState}
    {P : WriteFootprint}
    (hFrame : Kernel.ExecWritesWithin k s P)
    (hExec : exec k s = some s')
    {region : RegionName}
    (hnot : ∀ offset, ¬ P (region, offset)) :
    s'.mem region = s.mem region :=
  BlockState.WriteWithin.mem_eq_of_region_not_written (hFrame s' hExec) hnot

end ExecWritesWithin

end Kernel

namespace ComputeKernel

/-- Compute-kernel frame predicate through the algorithm projection. -/
def ExecWritesWithin (ck : ComputeKernel) (s : BlockState) (P : WriteFootprint) : Prop :=
  match ck.toAlgorithm? with
  | Except.ok k => k.ExecWritesWithin s P
  | Except.error _ => False

end ComputeKernel

end VeriTile.Triton
