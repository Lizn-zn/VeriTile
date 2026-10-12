import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealInputs
import VeriTile.Triton.Memory.KernelCore
import VeriTile.Triton.Memory.KernelSpec.TileIndex

/-! Named real IO for initial Dout/A/residual arrays and one gradient output.
All three program IDs index the windows and the formula. The shared UKernelIO
contract provides successful flat execution and preservation outside active
output cells; this adapter has no scratch cells or separate interpreter. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealIO
open VeriTile.Triton
noncomputable section

structure IO (BM BN chunk : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  Dout : RegionName
  A : RegionName
  Res : RegionName
  Db : RegionName
  doutWindow : Nat → Nat → Nat → TileIndex [BM, chunk] → Nat
  aWindow : Nat → Nat → Nat → TileIndex [chunk, BN] → Nat
  residualWindow : Nat → Nat → Nat → TileIndex [BM, BN] → Nat
  outputWindow : Nat → Nat → Nat → TileIndex [BM, BN] → Nat
  doutMask : Nat → Nat → Nat → TileIndex [BM, chunk] → Prop
  aMask : Nat → Nat → Nat → TileIndex [chunk, BN] → Prop
  outputMask : Nat → Nat → Nat → TileIndex [BM, BN] → Prop
  hasResidual : Bool

def inputShape (BM BN chunk : Nat) : Fin 3 → TileShape := ![[BM, chunk], [chunk, BN], [BM, BN]]

def IO.inputRegion {BM BN chunk : Nat} (io : IO BM BN chunk) : Fin 3 → RegionName := ![io.Dout, io.A, io.Res]

def IO.inputWindow {BM BN chunk : Nat} (io : IO BM BN chunk) :
    (ch : Fin 3) → Nat → Nat → Nat → TileIndex (inputShape BM BN chunk ch) → Nat
  | 0 => io.doutWindow
  | 1 => io.aWindow
  | 2 => io.residualWindow

def IO.inputMask {BM BN chunk : Nat} (io : IO BM BN chunk) :
    (ch : Fin 3) → Nat → Nat → Nat → TileIndex (inputShape BM BN chunk ch) → Prop
  | 0 => io.doutMask
  | 1 => io.aMask
  | 2 => fun p₀ p₁ p₂ i => io.hasResidual ∧ io.outputMask p₀ p₁ p₂ i

def IO.toU {BM BN chunk : Nat} (io : IO BM BN chunk) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := [io.Dout, io.A, io.Res, io.Db]
  ity := fun _ => .float
  iarity := fun ch => (TileShape.allIndices (inputShape BM BN chunk ch)).length
  ibuf := io.inputRegion
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices [BM, BN]).length
  obuf := fun _ => io.Db
  obuf_mem := by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun ch _ p₀ p₁ p₂ j => io.inputWindow ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  imask := fun ch _ p₀ p₁ p₂ j => io.inputMask ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  owin := fun _ _ p₀ p₁ p₂ j => io.outputWindow p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  omask := fun _ _ p₀ p₁ p₂ j => io.outputMask p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  swin := fun ch => Fin.elim0 ch
  smask := fun ch => Fin.elim0 ch

def IO.inputs {BM BN chunk : Nat} (io : IO BM BN chunk) (vals : io.toU.Ctx) : RealInputs.Inputs BM BN chunk where
  dout := fun i k => vals (0 : Fin 3) (tilePos [BM, chunk] (i, k, PUnit.unit))
  a := fun k j => vals (1 : Fin 3) (tilePos [chunk, BN] (k, j, PUnit.unit))
  residual := fun i j => vals (2 : Fin 3) (tilePos [BM, BN] (i, j, PUnit.unit))

/-- The mathematical function may use the third program ID, which determines
BMM's final partial chunk. Masked array padding is unconstrained. -/
def IO.Implements {BM BN chunk : Nat} (io : IO BM BN chunk)
    (f : Nat → Nat → Nat → RealInputs.Inputs BM BN chunk → TileIndex [BM, BN] → ℝ) : Prop :=
  io.toU.Implements3D (fun p₀ p₁ p₂ vals _ j => f p₀ p₁ p₂ (io.inputs vals) ((TileShape.allIndices _).get j))

scoped notation:25 io " ⊨ " f => IO.Implements io f

structure Pins {BM BN chunk : Nat} (io : IO BM BN chunk) (s : BlockState)
    (xs : RealInputs.Inputs BM BN chunk) : Prop where
  dout : ∀ i k, io.doutMask (s.pids 0) (s.pids 1) (s.pids 2) (i,k,PUnit.unit) →
    s.readMem io.Dout (io.doutWindow (s.pids 0) (s.pids 1) (s.pids 2) (i,k,PUnit.unit)) = xs.dout i k
  a : ∀ k j, io.aMask (s.pids 0) (s.pids 1) (s.pids 2) (k,j,PUnit.unit) →
    s.readMem io.A (io.aWindow (s.pids 0) (s.pids 1) (s.pids 2) (k,j,PUnit.unit)) = xs.a k j
  residual : io.hasResidual → ∀ i j, io.outputMask (s.pids 0) (s.pids 1) (s.pids 2) (i,j,PUnit.unit) →
    s.readMem io.Res (io.residualWindow (s.pids 0) (s.pids 1) (s.pids 2) (i,j,PUnit.unit)) = xs.residual i j

structure Bounds {BM BN chunk : Nat} (io : IO BM BN chunk) (s : BlockState) (bounds : RegionBounds) : Prop where
  dout : ∀ i, io.doutMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.doutWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Dout
  a : ∀ i, io.aMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.aWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.A
  residual : io.hasResidual → ∀ i, io.outputMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.residualWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Res
  output : ∀ i, io.outputMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Db

structure Results {BM BN chunk : Nat} (io : IO BM BN chunk) (initial current : BlockState)
    (ys : TileIndex [BM, BN] → ℝ) : Prop where
  output : ∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
    current.readMem io.Db (io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) = ys i
  frame : ∀ r a, (∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
    r ≠ io.Db ∨ a ≠ io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) →
    current.mem r a = initial.mem r a

end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealIO
