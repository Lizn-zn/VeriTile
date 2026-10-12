import bench.tritonbench_g.attention_score.Proofs.RealInputs
import VeriTile.Triton.Memory.KernelCore
import VeriTile.Triton.Memory.KernelSpec.TileIndex

/-! Named Q/K/M input views and one masked output view. The shared UKernelIO
contract supplies physical allocation/disjointness, successful flat execution,
active output readback and preservation outside the output window. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealIO
open VeriTile.Triton
noncomputable section

structure IO (rows B D : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  (Q K M Out : RegionName)
  queryWindow : Nat → Nat → Nat → TileIndex [rows,D] → Nat
  keyWindow : Nat → Nat → Nat → TileIndex [D,B] → Nat
  normalizerWindow : Nat → Nat → Nat → TileIndex [rows] → Nat
  outputWindow : Nat → Nat → Nat → Fin B → Nat
  queryMask : Nat → Nat → Nat → TileIndex [rows,D] → Prop
  keyMask : Nat → Nat → Nat → TileIndex [D,B] → Prop
  outputMask : Nat → Nat → Nat → Fin B → Prop

def inputShape (rows B D : Nat) : Fin 3 → TileShape := ![[rows,D],[D,B],[rows]]

def IO.inputRegion {rows B D : Nat} (io : IO rows B D) : Fin 3 → RegionName := ![io.Q,io.K,io.M]

def IO.inputWindow {rows B D : Nat} (io : IO rows B D) :
    (ch : Fin 3) → Nat → Nat → Nat → TileIndex (inputShape rows B D ch) → Nat
  | 0 => io.queryWindow
  | 1 => io.keyWindow
  | 2 => io.normalizerWindow

def IO.inputMask {rows B D : Nat} (io : IO rows B D) :
    (ch : Fin 3) → Nat → Nat → Nat → TileIndex (inputShape rows B D ch) → Prop
  | 0 => io.queryMask
  | 1 => io.keyMask
  | 2 => fun _ _ _ _ => True

def IO.toU {rows B D : Nat} (io : IO rows B D) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := [io.Q,io.K,io.M,io.Out]
  ity := fun _ => .float
  iarity := fun ch => (TileShape.allIndices (inputShape rows B D ch)).length
  ibuf := io.inputRegion
  oty := fun _ => .float
  oarity := fun _ => B
  obuf := fun _ => io.Out
  obuf_mem := by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun ch _ p₀ p₁ p₂ j => io.inputWindow ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  imask := fun ch _ p₀ p₁ p₂ j => io.inputMask ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  owin := fun _ _ p₀ p₁ p₂ j => io.outputWindow p₀ p₁ p₂ j
  omask := fun _ _ p₀ p₁ p₂ j => io.outputMask p₀ p₁ p₂ j
  swin := fun ch => Fin.elim0 ch
  smask := fun ch => Fin.elim0 ch

def IO.inputs {rows B D : Nat} (io : IO rows B D) (vals : io.toU.Ctx) : RealInputs.Inputs rows B D where
  query := fun r d => vals (0 : Fin 3) (tilePos [rows,D] (r,d,PUnit.unit))
  key := fun k d => vals (1 : Fin 3) (tilePos [D,B] (d,k,PUnit.unit))
  normalizer := fun r => vals (2 : Fin 3) (tilePos [rows] (r,PUnit.unit))

/-- Successful flat execution and all active outputs, with the full frame.
Input masks and allocation windows are supplied visibly by each concrete IO. -/
def IO.Implements {rows B D : Nat} (io : IO rows B D)
    (f : Nat → Nat → RealInputs.Inputs rows B D → Fin B → ℝ) : Prop :=
  io.toU.Implements (fun p₀ p₁ vals _ j => f p₀ p₁ (io.inputs vals) j)

scoped notation:25 io " ⊨ " f => IO.Implements io f

structure Pins {rows B D : Nat} (io : IO rows B D) (s : BlockState)
    (xs : RealInputs.Inputs rows B D) : Prop where
  query : ∀ r d, io.queryMask (s.pids 0) (s.pids 1) (s.pids 2) (r,d,PUnit.unit) →
    s.readMem io.Q (io.queryWindow (s.pids 0) (s.pids 1) (s.pids 2) (r,d,PUnit.unit)) = xs.query r d
  key : ∀ k d, io.keyMask (s.pids 0) (s.pids 1) (s.pids 2) (d,k,PUnit.unit) →
    s.readMem io.K (io.keyWindow (s.pids 0) (s.pids 1) (s.pids 2) (d,k,PUnit.unit)) = xs.key k d
  normalizer : ∀ r,
    s.readMem io.M (io.normalizerWindow (s.pids 0) (s.pids 1) (s.pids 2) (r,PUnit.unit)) = xs.normalizer r

structure Bounds {rows B D : Nat} (io : IO rows B D) (s : BlockState) (bounds : RegionBounds) : Prop where
  query : ∀ i, io.queryMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.queryWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Q
  key : ∀ i, io.keyMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.keyWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.K
  normalizer : ∀ i, io.normalizerWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.M
  output : ∀ i, io.outputMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Out

structure Results {rows B D : Nat} (io : IO rows B D) (initial current : BlockState)
    (ys : Fin B → ℝ) : Prop where
  output : ∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
    current.readMem io.Out (io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) = ys i
  frame : ∀ r a,
    (r ≠ io.Out ∨ ∀ i, io.outputMask (initial.pids 0) (initial.pids 1) (initial.pids 2) i →
      a ≠ io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) →
    current.mem r a = initial.mem r a

end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealIO
