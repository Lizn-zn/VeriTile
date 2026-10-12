import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardFlat
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardFlat

/-! Complete per-program IO contracts. Inputs are read from initial memory;
outputs are independent real formulas. Successful flat execution and preservation
of every other tagged memory cell are conclusions. No supplied carry register,
FP rule or unspecified-padding oracle is an input premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealIO
open VeriTile.Triton RealProgram
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000

structure ForwardInputs (p : Layout) where
  source : Nat → Values p
  gate : Nat → ℝ
  seed : Values p

structure BackwardInputs (p : Layout) where
  source : Nat → Values p
  gate : Nat → ℝ
  incoming : Nat → Values p

structure BackwardOutputs (p : Layout) where
  inputGradient : Nat → Values p
  gateGradient : Nat → ℝ
  seedGradient : Values p

structure ForwardIO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → ForwardInputs p
  bounds : BlockState → RegionBounds → Prop
  output : RegionName
  write : Launch → Nat → Fin p.BK → Fin p.BV → Nat

structure BackwardIO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → BackwardInputs p
  bounds : BlockState → RegionBounds → Prop
  inputGradient : RegionName
  gateGradient : RegionName
  seedGradient : RegionName
  writeInput : Launch → Nat → Fin p.BK → Fin p.BV → Nat
  writeGate : Launch → Nat → Nat
  writeSeed : Launch → Fin p.BK → Fin p.BV → Nat

def ForwardIO.writes {p : Layout} (io : ForwardIO p) (A : FlatAlloc) (l : Launch) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ (t : Nat) (e : Fin p.BK) (j : Fin p.BV),
    t < p.N ∧ A.addr io.output (io.write l t e j) = cell.2

def BackwardIO.regionWrites {p : Layout} (io : BackwardIO p) (l : Launch) : WriteFootprint :=
  fun cell =>
    ((cell.1 = io.inputGradient ∧ ∃ t e j, t < p.N-1 ∧ cell.2 = io.writeInput l t e j) ∨
     (cell.1 = io.gateGradient ∧ ∃ t, t < p.N-1 ∧ cell.2 = io.writeGate l t)) ∨
    (cell.1 = io.seedGradient ∧ ∃ e j, cell.2 = io.writeSeed l e j)

def BackwardIO.writes {p : Layout} (io : BackwardIO p) (A : FlatAlloc) (l : Launch) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ r off, io.regionWrites l (r,off) ∧ A.addr r off = cell.2

def ForwardIO.Implements {p : Layout} (io : ForwardIO p) (f : ForwardInputs p → Nat → Values p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, io.bounds s A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some v ∧
      (∀ t e j, t < p.N → v.readMem A.flat (A.addr io.output (io.write (launch s) t e j)) =
        f (io.read s) t e j) ∧
      BlockState.WriteWithin (io.writes A (launch s)) (A.flattenState s) v

def BackwardIO.Implements {p : Layout} (io : BackwardIO p) (f : BackwardInputs p → BackwardOutputs p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, io.bounds s A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some v ∧
      (∀ t e j, t < p.N-1 → v.readMem A.flat (A.addr io.inputGradient (io.writeInput (launch s) t e j)) =
        (f (io.read s)).inputGradient t e j) ∧
      (∀ t, t < p.N-1 → v.readMem A.flat (A.addr io.gateGradient (io.writeGate (launch s) t)) =
        (f (io.read s)).gateGradient t) ∧
      (∀ e j, v.readMem A.flat (A.addr io.seedGradient (io.writeSeed (launch s) e j)) =
        (f (io.read s)).seedGradient e j) ∧
      BlockState.WriteWithin (io.writes A (launch s)) (A.flattenState s) v

scoped notation:25 io " ⊨ " f => ForwardIO.Implements io f
scoped notation:25 io " ⊨ " f => BackwardIO.Implements io f

def forwardInputs (p : Layout) (S D seed : RegionName) (s : BlockState) : ForwardInputs p :=
  ⟨fun t e j => s.readMem S (address p (launch s) t e.val j.val),
   fun t => s.readMem D (gateAddress p (launch s) t),
   fun e j => s.readMem seed (seedAddress p (launch s) e.val j.val)⟩

def backwardInputs (p : Layout) (S D DS : RegionName) (s : BlockState) : BackwardInputs p :=
  ⟨fun t e j => s.readMem S (address p (launch s) t e.val j.val),
   fun t => s.readMem D (gateAddress p (launch s) t),
   fun t e j => s.readMem DS (address p (launch s) t e.val j.val)⟩

def ForwardInputs.arrays {p : Layout} (a : ForwardInputs p) : Inputs p :=
  ⟨a.source,a.gate,a.seed,fun _ _ _ => 0⟩

def BackwardInputs.arrays {p : Layout} (a : BackwardInputs p) : Inputs p :=
  ⟨a.source,a.gate,fun _ _ => 0,a.incoming⟩

theorem forward_projection (p : Layout) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (swapped hasSeed : Bool) :
    (RealForwardFlat.source swapped p S D O seed hasSeed storage gate seedStorage H).eraseDType.toAlgorithm? =
      .ok (RealForwardFlat.source swapped p S D O seed hasSeed storage gate seedStorage H).eraseDType.toAlgKernel := by
  have hp : (RealForwardFlat.source swapped p S D O seed hasSeed storage gate seedStorage H).toAlgorithm? =
      .ok (RealForwardFlat.source swapped p S D O seed hasSeed storage gate seedStorage H).toAlgKernel := by
    cases storage <;> cases gate <;> cases swapped <;> rfl
  simp only [ComputeKernel.eraseDType,hp,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

theorem backward_projection (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (swapped : Bool) :
    (RealBackwardFlat.source swapped p S D DI DG DL DS storage gate incoming H).eraseDType.toAlgorithm? =
      .ok (RealBackwardFlat.source swapped p S D DI DG DL DS storage gate incoming H).eraseDType.toAlgKernel := by
  have hp : (RealBackwardFlat.source swapped p S D DI DG DL DS storage gate incoming H).toAlgorithm? =
      .ok (RealBackwardFlat.source swapped p S D DI DG DL DS storage gate incoming H).toAlgKernel := by
    cases gate <;> cases incoming <;> cases swapped <;> rfl
  simp only [ComputeKernel.eraseDType,hp,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def forwardIO (p : Layout) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (swapped hasSeed : Bool) : ForwardIO p where
  kernel := (RealForwardFlat.source swapped p S D O seed hasSeed storage gate seedStorage H).eraseDType
  projection := forward_projection p S D O seed storage gate seedStorage H swapped hasSeed
  read := forwardInputs p S D seed
  bounds := fun s => RealForwardSafety.Bounds p (launch s) S D O seed hasSeed
  output := O
  write := fun l t e j => address p l t e.val j.val

def backwardIO (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (swapped : Bool) : BackwardIO p where
  kernel := (RealBackwardFlat.source swapped p S D DI DG DL DS storage gate incoming H).eraseDType
  projection := backward_projection p S D DI DG DL DS storage gate incoming H swapped
  read := backwardInputs p S D DS
  bounds := fun s => RealBackwardSafety.Bounds p (launch s) S D DI DG DL DS
  inputGradient := DI
  gateGradient := DG
  seedGradient := DL
  writeInput := fun l t e j => address p l t e.val j.val
  writeGate := RealBackwardPointers.gradientAddress p
  writeSeed := fun l e j => seedAddress p l e.val j.val

def forwardExpected (p : Layout) (hasSeed : Bool) (a : ForwardInputs p) : Nat → Values p :=
  RealMath.forward a.arrays hasSeed

def backwardExpected (p : Layout) (a : BackwardInputs p) : BackwardOutputs p :=
  ⟨fun t => RealMath.reverse a.arrays p.N (t+1),
   RealMath.gateGradient a.arrays p.N,RealMath.reverse a.arrays p.N 0⟩

theorem forward_implements (p : Layout) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (swapped hasSeed : Bool)
    (hN : 0 < p.N) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V) (hSO : S ≠ O) (hDO : D ≠ O) :
    forwardIO p S D O seed storage gate seedStorage H swapped hasSeed ⊨ forwardExpected p hasSeed := by
  intro A hd hcov s hb
  exact RealForwardFlat.source_executes A hd hcov p S D O seed (forwardInputs p S D seed s).arrays
    swapped hasSeed storage gate seedStorage H s hN hBK hBV hSO hDO
    (fun _ _ _ _ => rfl) (fun _ _ => rfl) (fun _ _ _ => rfl) hb

theorem backward_implements (p : Layout) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (swapped : Bool)
    (hN : 0 < p.N) (hBKpos : 0 < p.BK) (hBVpos : 0 < p.BV) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V)
    (sep : RealBackwardLoop.Separation S D DI DG DS) (hIDL : DI ≠ DL) (hGDL : DG ≠ DL) :
    backwardIO p S D DI DG DL DS storage gate incoming H swapped ⊨ backwardExpected p := by
  intro A hd hcov s hb
  exact RealBackwardFlat.source_executes A hd hcov p S D DI DG DL DS (backwardInputs p S D DS s).arrays
    swapped storage gate incoming H s hN hBKpos hBVpos hBK hBV sep hIDL hGDL
    (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl) (fun _ _ => rfl) hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealIO
