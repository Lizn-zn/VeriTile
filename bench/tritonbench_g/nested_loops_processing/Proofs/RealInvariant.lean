import bench.tritonbench_g.nested_loops_processing.Proofs.RealMemory

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

noncomputable def inputValues (s : BlockState) (src : RegionName) (sm sn shift : Nat) : IntTile :=
  fun i ↦ s.readMemValue .int src ((tilePointer src sm sn shift i).2)

/-- The execution invariant records the independent ordered-write prefix and
the complete raw-cell frame outside the destination region. -/
structure CopyInvariant (initial : BlockState) (src dst : RegionName)
    (sm sn count : Nat) (current : BlockState) : Prop where
  output : current.mem dst = CopyMath.expectedPrefix
    (fun address ↦ MemCell.of .int (initial.readMemValue .int src address))
    (initial.mem dst) sm sn count
  frame : ∀ r, r ≠ dst → current.mem r = initial.mem r

theorem CopyInvariant.initial (s : BlockState) (src dst : RegionName) (sm sn : Nat) :
    CopyInvariant s src dst sm sn 0 s := ⟨rfl, fun _ _ ↦ rfl⟩

theorem CopyInvariant.setReg {initial current : BlockState} {src dst : RegionName}
    {sm sn count : Nat} (h : CopyInvariant initial src dst sm sn count current)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (value : Tile dtype shape) :
    CopyInvariant initial src dst sm sn count (current.setReg name dtype shape value) := by
  exact ⟨h.output, h.frame⟩

theorem CopyInvariant.loaded {initial current : BlockState} {src dst : RegionName}
    {sm sn count : Nat} (h : CopyInvariant initial src dst sm sn count current)
    (hsep : src ≠ dst) (shift : Nat) :
    loaded current (tilePointer src sm sn shift) = inputValues initial src sm sn shift := by
  have hm := h.frame src hsep
  funext lane
  simp [FullReal.loaded, inputValues, tilePointer, BlockState.readMemValue, BlockState.readMemTyped, hm]

theorem CopyInvariant.scatter {initial current : BlockState} {src dst : RegionName}
    {sm sn count : Nat} (h : CopyInvariant initial src dst sm sn count current) :
    CopyInvariant initial src dst sm sn (count + 1)
      (scatter current (tilePointer dst sm sn (2 * count))
        (inputValues initial src sm sn (CopyMath.inputShift count))) := by
  constructor
  · have hc := scatter_copyTile current (fun address ↦ initial.readMemValue .int src address)
      dst sm sn count
    change (FullReal.scatter current (tilePointer dst sm sn (2 * count))
      (fun i ↦ initial.readMemValue .int src ((tilePointer dst sm sn (CopyMath.inputShift count) i).2))).mem dst = _
    rw [hc, h.output, CopyMath.expectedPrefix_succ]
  · intro r hr
    rw [scatter_other_region current (tilePointer dst sm sn (2 * count)) _ r (fun _ ↦ hr)]
    exact h.frame r hr

/-- Three actual inner-body stores advance the independent write prefix by three. -/
theorem CopyInvariant.innerState {initial current : BlockState} {src dst : RegionName}
    {sm sn count shift : Nat} (h : CopyInvariant initial src dst sm sn count current)
    (hsep : src ≠ dst) (hs : shift + 2 = CopyMath.inputShift (count + 2)) :
    CopyInvariant initial src dst sm sn (count + 3)
      (innerState current (tilePointer src sm sn shift) (tilePointer dst sm sn (2 * count))
        (inputValues initial src sm sn (CopyMath.inputShift count))
        (inputValues initial src sm sn (CopyMath.inputShift (count + 1))) (2 * sn)) := by
  let a := bump (tilePointer src sm sn shift) (2 * sn)
  let v3 := inputValues initial src sm sn (CopyMath.inputShift (count + 2))
  have hv : FullReal.loaded current a = v3 := by
    dsimp [a, v3]
    rw [tilePointer_bump, hs]
    exact h.loaded hsep _
  have h0 := (h.setReg "a_ptrs" .ptr [2,2] ⟨a⟩).setReg "a3" .int [2,2] ⟨v3⟩
  have h1 := h0.scatter
  have h2 := (h1.setReg "c_ptrs" .ptr [2,2]
    ⟨tilePointer dst sm sn (2 * (count + 1))⟩).scatter
  have h3 := (h2.setReg "c_ptrs" .ptr [2,2]
    ⟨tilePointer dst sm sn (2 * (count + 2))⟩).scatter
  have hf := h3.setReg "c_ptrs" .ptr [2,2] ⟨tilePointer dst sm sn (2 * (count + 3))⟩
  dsimp only [a, v3] at hv hf
  change CopyInvariant initial src dst sm sn (count + 3) _ at hf
  convert hf using 1
  simp only [FullReal.innerState, hv, tilePointer_bump_count]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
