import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealInputs
import VeriTile.Triton.Memory.KernelCore
import VeriTile.Triton.Memory.KernelSpec.TileIndex

/-! Named real IO for Q/K/V and in-place M/L/Out carry. The six input views
and three output views share six allocations. The implementation contract
reuses UKernelIO's existing flat-memory bridge; there is no second interpreter
or separate flattening proof here. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIO
open VeriTile.Triton
noncomputable section

structure IO (BM BD NK : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  Q : RegionName
  K : RegionName
  V : RegionName
  M : RegionName
  L : RegionName
  Out : RegionName
  queryWindow : Nat → Nat → Nat → TileIndex [BM, BD] → Nat
  keyWindow : Nat → Nat → Nat → TileIndex [NK, BD] → Nat
  valueWindow : Nat → Nat → Nat → TileIndex [NK, BD] → Nat
  centerWindow : Nat → Nat → Nat → TileIndex [BM] → Nat
  denominatorWindow : Nat → Nat → Nat → TileIndex [BM] → Nat
  outputWindow : Nat → Nat → Nat → TileIndex [BM, BD] → Nat
  queryMask : Nat → Nat → Nat → TileIndex [BM, BD] → Prop
  readCarry : Prop
  writeDenominator : Prop

structure Outputs (BM BD : Nat) where
  center : Fin BM → ℝ
  denominator : Fin BM → ℝ
  output : TileIndex [BM, BD] → ℝ

def inputShape (BM BD NK : Nat) : Fin 6 → TileShape :=
  ![[BM, BD], [NK, BD], [NK, BD], [BM], [BM], [BM, BD]]

def outputShape (BM BD : Nat) : Fin 3 → TileShape := ![[BM], [BM], [BM, BD]]

def IO.inputRegion {BM BD NK : Nat} (io : IO BM BD NK) : Fin 6 → RegionName :=
  ![io.Q, io.K, io.V, io.M, io.L, io.Out]

def IO.outputRegion {BM BD NK : Nat} (io : IO BM BD NK) : Fin 3 → RegionName :=
  ![io.M, io.L, io.Out]

def IO.inputWindow {BM BD NK : Nat} (io : IO BM BD NK) :
    (ch : Fin 6) → Nat → Nat → Nat → TileIndex (inputShape BM BD NK ch) → Nat
  | 0 => io.queryWindow
  | 1 => io.keyWindow
  | 2 => io.valueWindow
  | 3 => io.centerWindow
  | 4 => io.denominatorWindow
  | 5 => io.outputWindow

def IO.inputMask {BM BD NK : Nat} (io : IO BM BD NK) :
    (ch : Fin 6) → Nat → Nat → Nat → TileIndex (inputShape BM BD NK ch) → Prop
  | 0 => io.queryMask
  | 1 => fun _ _ _ _ => True
  | 2 => fun _ _ _ _ => True
  | 3 => fun _ _ _ _ => io.readCarry
  | 4 => fun _ _ _ _ => io.readCarry
  | 5 => fun _ _ _ _ => io.readCarry

def IO.outputWindowFor {BM BD NK : Nat} (io : IO BM BD NK) :
    (ch : Fin 3) → Nat → Nat → Nat → TileIndex (outputShape BM BD ch) → Nat
  | 0 => io.centerWindow
  | 1 => io.denominatorWindow
  | 2 => io.outputWindow

def IO.outputMask {BM BD NK : Nat} (io : IO BM BD NK) : Fin 3 → Prop
  | 0 => True
  | 1 => io.writeDenominator
  | 2 => True

/-- The numbered channels are internal adapters for the shared IO core.
All windows are independent of input values, so pinning is non-circular. -/
def IO.toU {BM BD NK : Nat} (io : IO BM BD NK) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 6
  nOut := 3
  nScr := 0
  bufs := [io.Q, io.K, io.V, io.M, io.L, io.Out]
  ity := fun _ => .float
  iarity := fun ch => (TileShape.allIndices (inputShape BM BD NK ch)).length
  ibuf := io.inputRegion
  oty := fun _ => .float
  oarity := fun ch => (TileShape.allIndices (outputShape BM BD ch)).length
  obuf := io.outputRegion
  obuf_mem := by intro ch; fin_cases ch <;> simp [IO.outputRegion]
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun ch _ p₀ p₁ p₂ j => io.inputWindow ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  imask := fun ch _ p₀ p₁ p₂ j => io.inputMask ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  owin := fun ch _ p₀ p₁ p₂ j => io.outputWindowFor ch p₀ p₁ p₂ ((TileShape.allIndices _).get j)
  omask := fun ch _ _ _ _ _ => io.outputMask ch
  swin := fun ch => Fin.elim0 ch
  smask := fun ch => Fin.elim0 ch

def IO.inputs {BM BD NK : Nat} (io : IO BM BD NK) (vals : io.toU.Ctx) : RealInputs.Inputs BM BD NK where
  query := fun r d => vals (0 : Fin 6) (tilePos [BM, BD] (r, d, PUnit.unit))
  key := fun k d => vals (1 : Fin 6) (tilePos [NK, BD] (k, d, PUnit.unit))
  value := fun k d => vals (2 : Fin 6) (tilePos [NK, BD] (k, d, PUnit.unit))
  center := fun r => vals (3 : Fin 6) (tilePos [BM] (r, PUnit.unit))
  denominator := fun r => vals (4 : Fin 6) (tilePos [BM] (r, PUnit.unit))
  output := fun r d => vals (5 : Fin 6) (tilePos [BM, BD] (r, d, PUnit.unit))

def Outputs.channel {BM BD : Nat} (ys : Outputs BM BD) :
    (ch : Fin 3) → TileIndex (outputShape BM BD ch) → ℝ
  | 0 => fun i => ys.center i.1
  | 1 => fun i => ys.denominator i.1
  | 2 => ys.output

/-- Successful flat execution, all active output values and preservation of
every cell outside M/Out and the optionally written L window. There is no
scratch region. Dimensions and all six named windows are supplied by the IO. -/
def IO.Implements {BM BD NK : Nat} (io : IO BM BD NK)
    (f : Nat → Nat → RealInputs.Inputs BM BD NK → Outputs BM BD) : Prop :=
  io.toU.Implements (fun p₀ p₁ vals ch j =>
    (f p₀ p₁ (io.inputs vals)).channel ch ((TileShape.allIndices _).get j))

scoped notation:25 io " ⊨ " f => IO.Implements io f

structure Pins {BM BD NK : Nat} (io : IO BM BD NK) (s : BlockState)
    (xs : RealInputs.Inputs BM BD NK) : Prop where
  query : ∀ r d, io.queryMask (s.pids 0) (s.pids 1) (s.pids 2) (r, d, PUnit.unit) →
    s.readMem io.Q (io.queryWindow (s.pids 0) (s.pids 1) (s.pids 2) (r, d, PUnit.unit)) = xs.query r d
  key : ∀ k d, s.readMem io.K (io.keyWindow (s.pids 0) (s.pids 1) (s.pids 2) (k, d, PUnit.unit)) = xs.key k d
  value : ∀ k d, s.readMem io.V (io.valueWindow (s.pids 0) (s.pids 1) (s.pids 2) (k, d, PUnit.unit)) = xs.value k d
  center : io.readCarry → ∀ r, s.readMem io.M (io.centerWindow (s.pids 0) (s.pids 1) (s.pids 2) (r, PUnit.unit)) = xs.center r
  denominator : io.readCarry → ∀ r,
    s.readMem io.L (io.denominatorWindow (s.pids 0) (s.pids 1) (s.pids 2) (r, PUnit.unit)) = xs.denominator r
  output : io.readCarry → ∀ r d,
    s.readMem io.Out (io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2) (r, d, PUnit.unit)) = xs.output r d

structure Bounds {BM BD NK : Nat} (io : IO BM BD NK) (s : BlockState) (bounds : RegionBounds) : Prop where
  query : ∀ i, io.queryMask (s.pids 0) (s.pids 1) (s.pids 2) i →
    io.queryWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Q
  key : ∀ i, io.keyWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.K
  value : ∀ i, io.valueWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.V
  center : ∀ i, io.centerWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.M
  denominator : io.readCarry ∨ io.writeDenominator → ∀ i,
    io.denominatorWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.L
  output : ∀ i, io.outputWindow (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.Out

structure Results {BM BD NK : Nat} (io : IO BM BD NK) (initial current : BlockState)
    (ys : Outputs BM BD) : Prop where
  center : ∀ i, current.readMem io.M (io.centerWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) = ys.center i.1
  denominator : io.writeDenominator → ∀ i,
    current.readMem io.L (io.denominatorWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) = ys.denominator i.1
  output : ∀ i, current.readMem io.Out (io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) = ys.output i
  frame : ∀ r a,
    (r ≠ io.M ∨ ∀ i, a ≠ io.centerWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) →
    (¬io.writeDenominator ∨ r ≠ io.L ∨ ∀ i, a ≠ io.denominatorWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) →
    (r ≠ io.Out ∨ ∀ i, a ≠ io.outputWindow (initial.pids 0) (initial.pids 1) (initial.pids 2) i) →
    current.mem r a = initial.mem r a

end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIO
