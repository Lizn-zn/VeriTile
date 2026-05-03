/-
VeriTile.Triton.Launch.Grid

ND grid-domain theorem surface for Triton program instances.
-/

import VeriTile.Triton.Semantics

namespace VeriTile.Triton

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

end Grid

/-- A typed program index for every in-rank axis of an ND grid. -/
def GridIndex (g : Grid) : Type :=
  (axis : Fin g.rank) → Fin (g.dim axis.val)

namespace GridIndex

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

/-- Instantiate a block state with a typed ND grid index. -/
def withGridIndex {g : Grid} (s : BlockState) (idx : GridIndex g) : BlockState :=
  s.withPids idx.toPids

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
  exact GridIndex.toPids_in_rank idx ⟨0, h⟩

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

end VeriTile.Triton
