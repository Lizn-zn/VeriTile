import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHPointers

/-! Actual real values loaded at a chunk, including unspecified tails. These
values contain no execution state or assumed output. The zero-oracle bridge
relates their arithmetic update to the independently stated closed form. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHTiles
open VeriTile.Triton RealHMath RealHProgram RealHPointers
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

abbrev StateValue (p : Layout) := Fin p.BK → Fin p.BV → ℝ
def stateTile (p : Layout) (f : StateValue p) : Tile .real [p.BK,p.BV] := Tile.mat (fun e j => some (f e j))

def keyData (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (e : Fin p.BK) (t : Fin p.BT) : ℝ :=
  let k := keyIndex p l e.val
  let time := timeIndex p c t.val
  let off := keyAddress p l k time
  if k < p.K ∧ time < p.T then a off else padding off

def valueData (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (t : Fin p.BT) (j : Fin p.BV) : ℝ :=
  let time := timeIndex p c t.val
  let v := valueIndex p l j.val
  let off := valueAddress p l time v
  if time < p.T ∧ v < p.V then a off else padding off

def endpointData (p : Layout) (l : Launch) (a padding : Nat → ℝ)
    (c head stride size start i : Nat) : ℝ :=
  let n := (c*p.BT+p.BT-1)*size+start+i
  if n < p.T*size then a (l.head*head+n*stride) else padding (l.head*head+n*stride)

structure Loaded (p : Layout) where
  key : Fin p.BK → Fin p.BT → ℝ
  value : Fin p.BT → Fin p.BV → ℝ
  keyGate : Fin p.BK → Fin p.BT → ℝ
  valueGate : Fin p.BT → Fin p.BV → ℝ
  keyEnd : Fin p.BK → ℝ
  valueEnd : Fin p.BV → ℝ

def loaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) : Loaded p where
  key := keyData p l a.key (padding r.key) c
  value := valueData p l a.value (padding r.value) c
  keyGate := keyData p l a.gate (padding r.gate) c
  valueGate := valueData p l a.gate (padding r.gate) c
  keyEnd := fun e => endpointData p l a.gate (padding r.gate) c p.keyHead p.keyFeature p.K (l.key*p.BK) e.val
  valueEnd := fun j => endpointData p l a.gate (padding r.gate) c p.valueHead p.valueFeature p.V (l.value*p.BV) j.val

def gatedKey (x : Loaded p) (gateK : Bool) (e : Fin p.BK) (t : Fin p.BT) : ℝ :=
  if gateK then x.key e t * Real.exp (x.keyEnd e - x.keyGate e t) else x.key e t

def gatedValue (x : Loaded p) (gateK : Bool) (t : Fin p.BT) (j : Fin p.BV) : ℝ :=
  if gateK then x.value t j else x.value t j * Real.exp (x.valueEnd j - x.valueGate t j)

def scaled (x : Loaded p) (gateK : Bool) (f : StateValue p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  f e j * Real.exp (if gateK then x.keyEnd e else x.valueEnd j)

def product (x : Loaded p) (gateK : Bool) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  ∑ t : Fin p.BT, gatedKey x gateK e t * gatedValue x gateK t j

def advance (x : Loaded p) (gateK : Bool) (f : StateValue p) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  scaled x gateK f e j + product x gateK e j

def expected (p : Layout) (l : Launch) (a : Inputs) (gateK hasSeed : Bool) (c : Nat) : StateValue p :=
  fun e j => closed p l a gateK hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)

theorem product_zero_padding (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (e : Fin p.BK) (j : Fin p.BV) :
    product (loaded p l r a (fun _ _ => 0) c) gateK e j =
      contribution p l a gateK c (keyIndex p l e.val) (valueIndex p l j.val) := by
  cases gateK <;>
    simp [product,gatedKey,gatedValue,loaded,keyData,valueData,endpointData,
      contribution,keyValue,valueValue,RealHMath.gateValue,endpoint,endpointIndex,
      keyIndex,valueIndex,Nat.add_assoc]

theorem advance_expected (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (gateK hasSeed : Bool) :
    advance (loaded p l r a (fun _ _ => 0) c) gateK (expected p l a gateK hasSeed c) =
      expected p l a gateK hasSeed (c+1) := by
  funext e j
  rw [advance,product_zero_padding,expected,closed_step]
  cases gateK <;>
    simp [scaled,expected,gateFactor,endpoint,loaded,endpointData,endpointIndex,
      keyIndex,valueIndex,Nat.add_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHTiles
