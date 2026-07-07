/-
VeriTile.Launch.Grid

ND grid-domain theorem surface for Triton program instances.
-/

import VeriTile.Semantics
import Mathlib.Data.Fintype.Pi

namespace VeriTile

/-- A Triton launch grid, represented by its per-axis dimensions. -/
structure Grid where
  dims : List Nat
  deriving DecidableEq, Repr

namespace Grid

/-- Number of in-rank grid axes. -/
def rank (g : Grid) : Nat :=
  g.dims.length

/-- Grid dimension at `axis`, defaulting to `0` outside the rank. -/
def dim (g : Grid) (axis : Nat) : Nat :=
  g.dims.getD axis 0

@[simp] theorem rank_mk (dims : List Nat) :
    ({ dims := dims } : Grid).rank = dims.length := rfl

@[simp] theorem dim_mk (dims : List Nat) (axis : Nat) :
    ({ dims := dims } : Grid).dim axis = dims.getD axis 0 := rfl

/-- Total launch-grid-dimension function for `BlockState.numPids` (#92):
in-rank axes give the grid dimension, out-of-rank axes default to `1`
(a launch runs exactly one program along any axis it does not span). -/
def toNumPids (g : Grid) : Nat → Nat :=
  fun axis => if axis < g.rank then g.dim axis else 1

@[simp] theorem toNumPids_in_rank (g : Grid) {axis : Nat} (h : axis < g.rank) :
    g.toNumPids axis = g.dim axis := by
  simp [toNumPids, h]

@[simp] theorem toNumPids_out_of_rank (g : Grid) {axis : Nat} (h : g.rank ≤ axis) :
    g.toNumPids axis = 1 := by
  simp [toNumPids, Nat.not_lt.mpr h]

end Grid

/-- A typed program index for every in-rank axis of an ND grid. -/
def GridIndex (g : Grid) : Type :=
  (axis : Fin g.rank) → Fin (g.dim axis.val)

namespace GridIndex

noncomputable instance fintype (g : Grid) : Fintype (GridIndex g) := by
  classical
  dsimp [GridIndex]
  infer_instance

noncomputable instance decidableEq (g : Grid) : DecidableEq (GridIndex g) := by
  classical
  infer_instance

/-- Convert a typed grid index to the total `Nat → Nat` PID function used by
`BlockState`. Out-of-rank axes default to `0`, matching the existing
`BlockState.pids` convention. -/
def toPids {g : Grid} (idx : GridIndex g) : Nat → Nat :=
  fun axis =>
    if h : axis < g.rank then
      (idx ⟨axis, h⟩).val
    else
      0

@[simp] theorem toPids_in_rank {g : Grid} (idx : GridIndex g)
    (axis : Fin g.rank) :
    idx.toPids axis.val = (idx axis).val := by
  simp [toPids, axis.isLt]

@[simp] theorem toPids_out_of_rank {g : Grid} (idx : GridIndex g)
    {axis : Nat} (h : g.rank ≤ axis) :
    idx.toPids axis = 0 := by
  simp [toPids, Nat.not_lt.mpr h]

end GridIndex

namespace BlockState

/-- Replace the total program-id function in a block state. -/
def withPids (s : BlockState) (pids : Nat → Nat) : BlockState :=
  { s with pids := pids }

/-- Replace the total launch-grid-dimension function in a block state (#92). -/
def withNumPids (s : BlockState) (numPids : Nat → Nat) : BlockState :=
  { s with numPids := numPids }

/-- Instantiate a block state with a typed ND grid index: sets the per-axis
program ids from the index and the per-axis grid dimensions
(`tl.num_programs`) from the grid itself. -/
def withGridIndex {g : Grid} (s : BlockState) (idx : GridIndex g) : BlockState :=
  (s.withPids idx.toPids).withNumPids g.toNumPids

@[simp] theorem withPids_pids (s : BlockState) (pids : Nat → Nat) :
    (s.withPids pids).pids = pids := rfl

@[simp] theorem withPids_pid (s : BlockState) (pids : Nat → Nat) :
    (s.withPids pids).pid = pids 0 := rfl

@[simp] theorem withPids_mem (s : BlockState) (pids : Nat → Nat) :
    (s.withPids pids).mem = s.mem := rfl

@[simp] theorem withPids_regs (s : BlockState) (pids : Nat → Nat) :
    (s.withPids pids).regs = s.regs := rfl

@[simp] theorem withPids_undef (s : BlockState) (pids : Nat → Nat) :
    (s.withPids pids).undef = s.undef := rfl

@[simp] theorem withPids_numPids (s : BlockState) (pids : Nat → Nat) :
    (s.withPids pids).numPids = s.numPids := rfl

@[simp] theorem withNumPids_numPids (s : BlockState) (numPids : Nat → Nat) :
    (s.withNumPids numPids).numPids = numPids := rfl

@[simp] theorem withNumPids_pids (s : BlockState) (numPids : Nat → Nat) :
    (s.withNumPids numPids).pids = s.pids := rfl

@[simp] theorem withNumPids_pid (s : BlockState) (numPids : Nat → Nat) :
    (s.withNumPids numPids).pid = s.pid := rfl

@[simp] theorem withNumPids_mem (s : BlockState) (numPids : Nat → Nat) :
    (s.withNumPids numPids).mem = s.mem := rfl

@[simp] theorem withNumPids_regs (s : BlockState) (numPids : Nat → Nat) :
    (s.withNumPids numPids).regs = s.regs := rfl

@[simp] theorem withNumPids_undef (s : BlockState) (numPids : Nat → Nat) :
    (s.withNumPids numPids).undef = s.undef := rfl

@[simp] theorem withGridIndex_pids_in_rank {g : Grid}
    (s : BlockState) (idx : GridIndex g) (axis : Fin g.rank) :
    (s.withGridIndex idx).pids axis.val = (idx axis).val := by
  simp [withGridIndex]

@[simp] theorem withGridIndex_pids_out_of_rank {g : Grid}
    (s : BlockState) (idx : GridIndex g) {axis : Nat} (h : g.rank ≤ axis) :
    (s.withGridIndex idx).pids axis = 0 := by
  simp [withGridIndex, h]

@[simp] theorem withGridIndex_pid {g : Grid}
    (s : BlockState) (idx : GridIndex g) (h : 0 < g.rank) :
    (s.withGridIndex idx).pid = (idx ⟨0, h⟩).val := by
  simpa using GridIndex.toPids_in_rank idx ⟨0, h⟩

@[simp] theorem withGridIndex_numPids_in_rank {g : Grid}
    (s : BlockState) (idx : GridIndex g) {axis : Nat} (h : axis < g.rank) :
    (s.withGridIndex idx).numPids axis = g.dim axis := by
  simp [withGridIndex, h]

@[simp] theorem withGridIndex_numPids_out_of_rank {g : Grid}
    (s : BlockState) (idx : GridIndex g) {axis : Nat} (h : g.rank ≤ axis) :
    (s.withGridIndex idx).numPids axis = 1 := by
  simp [withGridIndex, h]

@[simp] theorem withGridIndex_mem {g : Grid}
    (s : BlockState) (idx : GridIndex g) :
    (s.withGridIndex idx).mem = s.mem := rfl

@[simp] theorem withGridIndex_regs {g : Grid}
    (s : BlockState) (idx : GridIndex g) :
    (s.withGridIndex idx).regs = s.regs := rfl

@[simp] theorem withGridIndex_undef {g : Grid}
    (s : BlockState) (idx : GridIndex g) :
    (s.withGridIndex idx).undef = s.undef := rfl

end BlockState

namespace Kernel

/-- Raw ND-grid theorem surface: the postcondition observes the optional
result of executing the kernel at each typed program index. -/
def ForAllPrograms (k : Kernel) (g : Grid) (s : BlockState)
    (post : GridIndex g → Option BlockState → Prop) : Prop :=
  ∀ idx : GridIndex g, post idx (exec k (s.withGridIndex idx))

/-- Successful-execution ND-grid theorem surface. Most correctness wrappers use
this shape: every program instance executes and satisfies a state postcondition. -/
def ForAllProgramsSome (k : Kernel) (g : Grid) (s : BlockState)
    (post : GridIndex g → BlockState → Prop) : Prop :=
  ∀ idx : GridIndex g,
    ∃ s', exec k (s.withGridIndex idx) = some s' ∧ post idx s'

end Kernel

/-- In-rank `tl.program_id(axis)` reads the corresponding typed grid index. -/
theorem program_id_under_grid_index {g : Grid}
    (s : BlockState) (idx : GridIndex g) (axis : Fin g.rank) :
    evalOp (Op.programId axis.val) (s.withGridIndex idx) =
      some (Tile.scalar (idx axis).val) := by
  simp [evalOp]

/-- Out-of-rank `tl.program_id(axis)` reads `0`. -/
theorem program_id_out_of_grid_rank {g : Grid}
    (s : BlockState) (idx : GridIndex g) {axis : Nat} (h : g.rank ≤ axis) :
    evalOp (Op.programId axis) (s.withGridIndex idx) =
      some (Tile.scalar 0) := by
  simp [evalOp, h]

/-- In-rank `tl.num_programs(axis)` reads the launch-grid dimension (#92). -/
theorem num_programs_under_grid_index {g : Grid}
    (s : BlockState) (idx : GridIndex g) (axis : Fin g.rank) :
    evalOp (Op.numPrograms axis.val) (s.withGridIndex idx) =
      some (Tile.scalar (g.dim axis.val)) := by
  simp [axis.isLt]

/-- Out-of-rank `tl.num_programs(axis)` reads `1`: the launch runs exactly one
program along any axis the grid does not span. -/
theorem num_programs_out_of_grid_rank {g : Grid}
    (s : BlockState) (idx : GridIndex g) {axis : Nat} (h : g.rank ≤ axis) :
    evalOp (Op.numPrograms axis) (s.withGridIndex idx) =
      some (Tile.scalar 1) := by
  simp [h]

end VeriTile
