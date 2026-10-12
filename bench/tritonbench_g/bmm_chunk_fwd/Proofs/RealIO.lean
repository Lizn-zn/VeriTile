import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealInputs
import VeriTile.Triton.Memory.KernelCore
import VeriTile.Triton.Memory.KernelSpec.TileIndex

/-! Named real inputs A/B and signed row/column sequence IDs, with a single
output tile. All windows use three program IDs. Metadata channels share one
allocation, are pinned only when requested, and retain signed integer values.
The shared core supplies flat execution and the frame, with no scratch. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealIO
open VeriTile.Triton
noncomputable section
structure IO (BM BN K : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  A : RegionName
  B : RegionName
  Seq : RegionName
  Out : RegionName
  aWindow : Nat → Nat → Nat → TileIndex [BM, K] → Nat
  bWindow : Nat → Nat → Nat → TileIndex [K, BN] → Nat
  rowIDWindow : Nat → Nat → Nat → TileIndex [BM] → Nat
  columnIDWindow : Nat → Nat → Nat → TileIndex [BN] → Nat
  outputWindow : Nat → Nat → Nat → TileIndex [BM, BN] → Nat
  aMask : Nat → Nat → Nat → TileIndex [BM, K] → Prop
  bMask : Nat → Nat → Nat → TileIndex [K, BN] → Prop
  rowIDMask : Nat → Nat → Nat → TileIndex [BM] → Prop
  columnIDMask : Nat → Nat → Nat → TileIndex [BN] → Prop
  outputMask : Nat → Nat → Nat → TileIndex [BM, BN] → Prop
  hasSeq : Bool

def inputShape (BM BN K : Nat) : Fin 4 → TileShape := ![[BM, K], [K, BN], [BM], [BN]]
def inputType : Fin 4 → ChanTy := ![.float, .float, .int, .int]
def IO.inputRegion {BM BN K : Nat} (io : IO BM BN K) : Fin 4 → RegionName := ![io.A, io.B, io.Seq, io.Seq]
def IO.inputWindow {BM BN K : Nat} (io : IO BM BN K) :
    (ch : Fin 4) → Nat → Nat → Nat → TileIndex (inputShape BM BN K ch) → Nat
  | 0 => io.aWindow
  | 1 => io.bWindow
  | 2 => io.rowIDWindow
  | 3 => io.columnIDWindow

def IO.inputMask {BM BN K : Nat} (io : IO BM BN K) :
    (ch : Fin 4) → Nat → Nat → Nat → TileIndex (inputShape BM BN K ch) → Prop
  | 0 => io.aMask
  | 1 => io.bMask
  | 2 => fun p₀ p₁ p₂ i => io.hasSeq ∧ io.rowIDMask p₀ p₁ p₂ i
  | 3 => fun p₀ p₁ p₂ i => io.hasSeq ∧ io.columnIDMask p₀ p₁ p₂ i

def IO.toU {BM BN K : Nat} (io : IO BM BN K) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 4
  nOut := 1
  nScr := 0
  bufs := [io.A, io.B, io.Seq, io.Out]
  ity := inputType
  iarity := fun ch => (TileShape.allIndices (inputShape BM BN K ch)).length
  ibuf := io.inputRegion
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices [BM, BN]).length
  obuf := fun _ => io.Out
  obuf_mem := by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun ch _ p₀ p₁ p₂ j => io.inputWindow ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  imask := fun ch _ p₀ p₁ p₂ j => io.inputMask ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  owin := fun _ _ p₀ p₁ p₂ j => io.outputWindow p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  omask := fun _ _ p₀ p₁ p₂ j => io.outputMask p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  swin := fun ch => Fin.elim0 ch
  smask := fun ch => Fin.elim0 ch

def IO.inputs {BM BN K : Nat} (io : IO BM BN K) (vals : io.toU.Ctx) : RealInputs.Inputs BM BN K where
  a := fun i k => vals (0 : Fin 4) (tilePos [BM, K] (i,k,PUnit.unit))
  b := fun k j => vals (1 : Fin 4) (tilePos [K, BN] (k,j,PUnit.unit))
  rowIDs := fun i => vals (2 : Fin 4) (tilePos [BM] (i,PUnit.unit))
  columnIDs := fun j => vals (3 : Fin 4) (tilePos [BN] (j,PUnit.unit))

def IO.Implements {BM BN K : Nat} (io : IO BM BN K)
    (f : Nat → Nat → Nat → RealInputs.Inputs BM BN K → TileIndex [BM, BN] → ℝ) : Prop :=
  io.toU.Implements3D (fun p₀ p₁ p₂ vals _ j => f p₀ p₁ p₂ (io.inputs vals) ((TileShape.allIndices _).get j))
scoped notation:25 io " ⊨ " f => IO.Implements io f

structure Pins {BM BN K : Nat} (io : IO BM BN K) (s : BlockState)
    (xs : RealInputs.Inputs BM BN K) : Prop where
  a : ∀ i k, io.aMask (s.pids 0) (s.pids 1) (s.pids 2) (i,k,PUnit.unit) →
    s.readMem io.A (io.aWindow (s.pids 0) (s.pids 1) (s.pids 2) (i,k,PUnit.unit)) = xs.a i k
  b : ∀ k j, io.bMask (s.pids 0) (s.pids 1) (s.pids 2) (k,j,PUnit.unit) →
    s.readMem io.B (io.bWindow (s.pids 0) (s.pids 1) (s.pids 2) (k,j,PUnit.unit)) = xs.b k j
  rowIDs : io.hasSeq → ∀ i, io.rowIDMask (s.pids 0) (s.pids 1) (s.pids 2) (i,PUnit.unit) →
    s.readMemValue .int io.Seq (io.rowIDWindow (s.pids 0) (s.pids 1) (s.pids 2) (i,PUnit.unit)) = xs.rowIDs i
  columnIDs : io.hasSeq → ∀ j, io.columnIDMask (s.pids 0) (s.pids 1) (s.pids 2) (j,PUnit.unit) →
    s.readMemValue .int io.Seq (io.columnIDWindow (s.pids 0) (s.pids 1) (s.pids 2) (j,PUnit.unit)) = xs.columnIDs j

structure Bounds {BM BN K : Nat} (io : IO BM BN K) (s : BlockState) (bounds : RegionBounds) : Prop where
  a : ∀ i, io.aMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.aWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.A
  b : ∀ i, io.bMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.bWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.B
  rowIDs : io.hasSeq → ∀ i, io.rowIDMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.rowIDWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Seq
  columnIDs : io.hasSeq → ∀ i, io.columnIDMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.columnIDWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Seq
  output : ∀ i, io.outputMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Out

structure Results {BM BN K : Nat} (io : IO BM BN K) (initial current : BlockState)
    (ys : TileIndex [BM, BN] → ℝ) : Prop where
  output : ∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
    current.readMem io.Out (io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) = ys i
  frame : ∀ r a, (∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
    r ≠ io.Out ∨ a ≠ io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) →
    current.mem r a = initial.mem r a

end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealIO
