import bench.tritonbench_g.bmm_optimized.Proofs.RealMemory
import VeriTile.Triton.Memory.KernelCore
import VeriTile.Triton.Memory.KernelSpec.TileIndex

/-! Two independent initial matrices and one output tile. Both runtime grid
dimensions are pinned before invoking the shared flat-memory contract. All
three program IDs index windows; the source still executes tl.num_programs.
The contract includes zero unspecified loads, disjoint flat allocations,
active-address bounds, successful execution and the full memory frame. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealIO
open VeriTile.Triton
noncomputable section

structure Inputs (BM BN K : Nat) where
  a : Fin BM → Fin K → ℝ
  b : Fin K → Fin BN → ℝ

def expected {BM BN K : Nat} (xs : Inputs BM BN K) (i : TileIndex [BM,BN]) : ℝ :=
  ∑ k : Fin K, xs.a i.1 k * xs.b k i.2.1

structure IO (BM BN K : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  A : RegionName
  B : RegionName
  O : RegionName
  gridX : Nat
  gridY : Nat
  aWindow : Nat → Nat → Nat → TileIndex [BM,K] → Nat
  bWindow : Nat → Nat → Nat → TileIndex [K,BN] → Nat
  outputWindow : Nat → Nat → Nat → TileIndex [BM,BN] → Nat
  aMask : Nat → Nat → Nat → TileIndex [BM,K] → Prop
  bMask : Nat → Nat → Nat → TileIndex [K,BN] → Prop
  outputMask : Nat → Nat → Nat → TileIndex [BM,BN] → Prop

def inputShape (BM BN K : Nat) : Fin 2 → TileShape := ![[BM,K], [K,BN]]
def IO.inputRegion {BM BN K : Nat} (io : IO BM BN K) : Fin 2 → RegionName := ![io.A,io.B]
def IO.inputWindow {BM BN K : Nat} (io : IO BM BN K) :
    (ch : Fin 2) → Nat → Nat → Nat → TileIndex (inputShape BM BN K ch) → Nat
  | 0 => io.aWindow
  | 1 => io.bWindow

def IO.inputMask {BM BN K : Nat} (io : IO BM BN K) :
    (ch : Fin 2) → Nat → Nat → Nat → TileIndex (inputShape BM BN K ch) → Prop
  | 0 => io.aMask
  | 1 => io.bMask

def IO.toU {BM BN K : Nat} (io : IO BM BN K) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [io.A,io.B,io.O]
  ity := fun _ => .float
  iarity := fun ch => (TileShape.allIndices (inputShape BM BN K ch)).length
  ibuf := io.inputRegion
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices [BM,BN]).length
  obuf := fun _ => io.O
  obuf_mem := by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun ch _ p₀ p₁ p₂ j => io.inputWindow ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  imask := fun ch _ p₀ p₁ p₂ j => io.inputMask ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  owin := fun _ _ p₀ p₁ p₂ j => io.outputWindow p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  omask := fun _ _ p₀ p₁ p₂ j => io.outputMask p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  swin := fun ch => Fin.elim0 ch
  smask := fun ch => Fin.elim0 ch

def IO.inputs {BM BN K : Nat} (io : IO BM BN K) (vals : io.toU.Ctx) : Inputs BM BN K where
  a := fun i k => vals (0 : Fin 2) (tilePos [BM,K] (i,k,PUnit.unit))
  b := fun k j => vals (1 : Fin 2) (tilePos [K,BN] (k,j,PUnit.unit))

structure IO.Launch {BM BN K : Nat} (io : IO BM BN K) (s : BlockState) : Prop where
  gridX : s.numPids 0 = io.gridX
  gridY : s.numPids 1 = io.gridY
  x : s.pids 0 < io.gridX
  y : s.pids 1 < io.gridY

/-- Per-program flat IO at every legal host-grid coordinate. The batch ID is
unrestricted except by allocation/input-window bounds. -/
def IO.Implements {BM BN K : Nat} (io : IO BM BN K)
    (f : Inputs BM BN K → TileIndex [BM,BN] → ℝ) : Prop :=
  ∀ (s : BlockState), io.Launch s → ∀ vals : io.toU.Ctx,
    io.toU.ImplementsAt s vals (fun _ j => f (io.inputs vals) ((TileShape.allIndices _).get j))

scoped notation:25 io " ⊨ " f => IO.Implements io f

structure Pins {BM BN K : Nat} (io : IO BM BN K) (s : BlockState) (xs : Inputs BM BN K) : Prop where
  a : ∀ i k, io.aMask (s.pids 0) (s.pids 1) (s.pids 2) (i,k,PUnit.unit) →
    s.readMem io.A (io.aWindow (s.pids 0) (s.pids 1) (s.pids 2) (i,k,PUnit.unit)) = xs.a i k
  b : ∀ k j, io.bMask (s.pids 0) (s.pids 1) (s.pids 2) (k,j,PUnit.unit) →
    s.readMem io.B (io.bWindow (s.pids 0) (s.pids 1) (s.pids 2) (k,j,PUnit.unit)) = xs.b k j

structure Bounds {BM BN K : Nat} (io : IO BM BN K) (s : BlockState) (bounds : RegionBounds) : Prop where
  a : ∀ i, io.aMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.aWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.A
  b : ∀ i, io.bMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.bWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.B
  output : ∀ i, io.outputMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.O

structure Results {BM BN K : Nat} (io : IO BM BN K) (initial current : BlockState)
    (ys : TileIndex [BM,BN] → ℝ) : Prop where
  output : ∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
    current.readMem io.O (io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) = ys i
  frame : ∀ r a, (∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
    r ≠ io.O ∨ a ≠ io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) →
    current.mem r a = initial.mem r a

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealIO
