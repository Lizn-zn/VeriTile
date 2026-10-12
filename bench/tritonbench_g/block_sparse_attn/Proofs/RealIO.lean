import bench.tritonbench_g.block_sparse_attn.Proofs.RealInputs

/-! A real IO contract for the complete CSR attention source. Input values
are mathematical arrays; windows and valid CSR indices are explicit fields.
The postcondition includes successful flat execution, tagged output cells and
the full memory frame. No trace or computed-output assumption is an input. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealIO
open VeriTile.Triton
noncomputable section

abbrev Inputs := CSRMath.Inputs

structure IO (BM : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  Q : RegionName
  K : RegionName
  V : RegionName
  rows : Region .int
  columns : Region .int
  output : RegionName
  context : Nat
  width : Nat
  evenM : Bool
  evenN : Bool
  rowSlot : Nat → Nat → Bool → Nat
  columnSlot : Nat → Nat → Inputs → Nat → Int
  queryWindow : Nat → Nat → Nat → Nat
  keyWindow : Nat → Int → Nat → Nat
  valueWindow : Nat → Int → Nat → Nat
  outputWindow : Nat → Nat → Fin BM → Nat → Nat
  domain : Nat → Nat → Inputs → Prop

def IO.count {BM : Nat} (io : IO BM) (pid₀ pid₁ : Nat) (xs : Inputs) : Nat :=
  (xs.rowIndex (io.rowSlot pid₀ pid₁ Bool.true) - xs.rowIndex (io.rowSlot pid₀ pid₁ Bool.false)).toNat

def IO.selectedKey {BM : Nat} (io : IO BM) (BN pid₀ pid₁ : Nat) (xs : Inputs)
    (visit : Nat) (lane : Fin BN) : Int :=
  xs.columnIndex (io.columnSlot pid₀ pid₁ xs visit) * (BN : Int) + (lane.val : Int)

/-- Pins only initial arrays and metadata. Undefined/padded cells beyond the
mathematical context do not need specified values. -/
structure IO.Pins {BM : Nat} (io : IO BM) (s : BlockState) (xs : Inputs) : Prop where
  rows : ∀ next, s.readMemValue .int (Region.cast io.rows) (io.rowSlot (s.pids 0) (s.pids 1) next) =
    xs.rowIndex (io.rowSlot (s.pids 0) (s.pids 1) next)
  columns : ∀ visit, visit < io.count (s.pids 0) (s.pids 1) xs →
    s.readMemValue .int (Region.cast io.columns) (io.columnSlot (s.pids 0) (s.pids 1) xs visit).toNat =
      xs.columnIndex (io.columnSlot (s.pids 0) (s.pids 1) xs visit)
  query : ∀ r d, r < io.context → d < io.width →
    s.readMem io.Q (io.queryWindow (s.pids 1) r d) = xs.query r d
  key : ∀ r d, 0 ≤ r → r < (io.context : Int) → d < io.width →
    s.readMem io.K (io.keyWindow (s.pids 1) r d) = xs.key r d
  value : ∀ r d, 0 ≤ r → r < (io.context : Int) → d < io.width →
    s.readMem io.V (io.valueWindow (s.pids 1) r d) = xs.value r d

/-- Allocation covers visited metadata and active physical loads, including
padding in unchecked branches. Stores always keep the query-row mask. -/
structure IO.Bounds {BM : Nat} (io : IO BM) (BN : Nat) (bounds : RegionBounds)
    (pid₀ pid₁ : Nat) (xs : Inputs) : Prop where
  rows : ∀ next, io.rowSlot pid₀ pid₁ next < bounds (Region.cast io.rows)
  columns : ∀ visit, visit < io.count pid₀ pid₁ xs → (io.columnSlot pid₀ pid₁ xs visit).toNat < bounds (Region.cast io.columns)
  query : ∀ (r : Fin BM) d, d < io.width → (io.evenM = Bool.true ∨ pid₀ * BM + r.val < io.context) →
    io.queryWindow pid₁ (pid₀ * BM + r.val) d < bounds io.Q
  key : ∀ visit, visit < io.count pid₀ pid₁ xs → ∀ (j : Fin BN) d, d < io.width →
    (io.evenN = Bool.true ∨ io.selectedKey BN pid₀ pid₁ xs visit j < (io.context : Int)) →
    io.keyWindow pid₁ (io.selectedKey BN pid₀ pid₁ xs visit j) d < bounds io.K
  value : ∀ visit, visit < io.count pid₀ pid₁ xs → ∀ (j : Fin BN) d, d < io.width →
    (io.evenN = Bool.true ∨ io.selectedKey BN pid₀ pid₁ xs visit j < (io.context : Int)) →
    io.valueWindow pid₁ (io.selectedKey BN pid₀ pid₁ xs visit j) d < bounds io.V
  output : ∀ r d, d < io.width → pid₀ * BM + r.val < io.context →
    io.outputWindow pid₀ pid₁ r d < bounds io.output

/-- One program's real execution from independent arrays and a valid CSR
schedule. The same displayed windows govern allocation, values and frame. -/
def IO.Implements {BM : Nat} (io : IO BM) (BN : Nat)
    (f : Nat → Nat → Inputs → Fin BM → Nat → ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (xs : Inputs),
    io.domain (s.pids 0) (s.pids 1) xs → s.undef = (fun _ _ => 0) →
    io.Bounds BN A.extent (s.pids 0) (s.pids 1) xs → io.Pins s xs →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ r d, d < io.width → s.pids 0 * BM + r.val < io.context →
        final.mem A.flat (A.addr io.output (io.outputWindow (s.pids 0) (s.pids 1) r d)) =
          MemCell.real (f (s.pids 0) (s.pids 1) xs r d)) ∧
      (∀ region address,
        (∀ r d, d < io.width → s.pids 0 * BM + r.val < io.context →
          ¬(A.flat = region ∧ A.addr io.output (io.outputWindow (s.pids 0) (s.pids 1) r d) = address)) →
        final.mem region address = (A.flattenState s).mem region address)

scoped notation:25 io " ⊨[" BN "] " f => IO.Implements io BN f

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealIO
