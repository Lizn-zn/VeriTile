import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealSource

/-! The real inner step, with arbitrary launch indices, inner-block index and
incoming accumulator. These definitions use the original block-pointer layouts
and both-axis masks. They do not assume that BC divides BT or that a tile fits
inside K, V or T. The independent inputs are flat real arrays. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealProgram
open VeriTile.Triton
open scoped BigOperators
noncomputable section

structure Layout where
  keyHead : Nat
  keyTime : Nat
  keyDim : Nat
  valueHead : Nat
  valueTime : Nat
  valueDim : Nat
  historyHead : Nat
  historyTime : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BC : Nat
  BK : Nat
  BV : Nat
  NT : Nat

structure Regions where
  key : RegionName
  delta : RegionName
  value : RegionName
  corrected : RegionName
  history : RegionName
  initial : RegionName
  final : RegionName

structure Launch where
  key : Nat
  value : Nat
  head : Nat

structure Inputs where
  key : Nat → ℝ
  delta : Nat → ℝ
  value : Nat → ℝ

abbrev StateValue (p : Layout) := Fin p.BK → Fin p.BV → ℝ

def timeBase (p : Layout) (outer inner : Nat) : Nat := outer * p.BT + inner * p.BC

def keyIndex (p : Layout) (l : Launch) (e : Nat) : Nat := l.key * p.BK + e

def valueIndex (p : Layout) (l : Launch) (j : Nat) : Nat := l.value * p.BV + j

def keyAddress (p : Layout) (l : Launch) (t e : Nat) : Nat :=
  l.head * p.keyHead + keyIndex p l e * p.keyDim + t * p.keyTime

def deltaAddress (p : Layout) (l : Launch) (t e : Nat) : Nat :=
  l.head * p.keyHead + t * p.keyTime + keyIndex p l e * p.keyDim

def valueAddress (p : Layout) (l : Launch) (t j : Nat) : Nat :=
  l.head * p.valueHead + t * p.valueTime + valueIndex p l j * p.valueDim

def activeValue (p : Layout) (l : Launch) (t j : Nat) : Prop :=
  t < p.T ∧ valueIndex p l j < p.V
instance (p : Layout) (l : Launch) (t j : Nat) : Decidable (activeValue p l t j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def keyValue (p : Layout) (l : Launch) (a : Inputs) (t e : Nat) : ℝ :=
  if keyIndex p l e < p.K ∧ t < p.T then a.key (keyAddress p l t e) else 0

def deltaValue (p : Layout) (l : Launch) (a : Inputs) (t e : Nat) : ℝ :=
  if t < p.T ∧ keyIndex p l e < p.K then a.delta (deltaAddress p l t e) else 0

def inputValue (p : Layout) (l : Launch) (a : Inputs) (t j : Nat) : ℝ :=
  if activeValue p l t j then a.value (valueAddress p l t j) else 0

/-- The frozen state is used for every corrected row in the same outer chunk. -/
def correctedValue (p : Layout) (l : Launch) (a : Inputs) (frozen : StateValue p)
    (t : Nat) (j : Fin p.BV) : ℝ :=
  inputValue p l a t j.val - ∑ e : Fin p.BK, deltaValue p l a t e.val * frozen e j

/-- One inner block adds to the existing sum; it does not reset that sum. -/
def nextAccum (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen acc : StateValue p) : StateValue p := fun e j =>
  acc e j + ∑ c : Fin p.BC, keyValue p l a (timeBase p outer inner + c.val) e.val *
    correctedValue p l a frozen (timeBase p outer inner + c.val) j

def stateTile (p : Layout) (f : StateValue p) : Tile .real [p.BK, p.BV] :=
  Tile.mat fun e j => some (f e j)

def keyTile (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat) : Tile .real [p.BK, p.BC] :=
  Tile.mat fun e c => some (keyValue p l a (timeBase p outer inner + c.val) e.val)

def deltaTile (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat) : Tile .real [p.BC, p.BK] :=
  Tile.mat fun c e => some (deltaValue p l a (timeBase p outer inner + c.val) e.val)

def valueTile (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat) : Tile .real [p.BC, p.BV] :=
  Tile.mat fun c j => some (inputValue p l a (timeBase p outer inner + c.val) j.val)

def correctedTile (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen : StateValue p) : Tile .real [p.BC, p.BV] :=
  Tile.mat fun c j => some (correctedValue p l a frozen (timeBase p outer inner + c.val) j)

def keyPointer (p : Layout) (l : Launch) (r : RegionName) (outer inner : Nat) :
    Tile .blockPtr [p.BK, p.BC] :=
  ⟨fun _ => BlockPtr.mk r (l.head * p.keyHead) [p.K, p.T] [p.BK, p.BC]
    [p.keyDim, p.keyTime] [l.key * p.BK, timeBase p outer inner]⟩

def deltaPointer (p : Layout) (l : Launch) (r : RegionName) (outer inner : Nat) :
    Tile .blockPtr [p.BC, p.BK] :=
  ⟨fun _ => BlockPtr.mk r (l.head * p.keyHead) [p.T, p.K] [p.BC, p.BK]
    [p.keyTime, p.keyDim] [timeBase p outer inner, l.key * p.BK]⟩

def valuePointer (p : Layout) (l : Launch) (r : RegionName) (outer inner : Nat) :
    Tile .blockPtr [p.BC, p.BV] :=
  ⟨fun _ => BlockPtr.mk r (l.head * p.valueHead) [p.T, p.V] [p.BC, p.BV]
    [p.valueTime, p.valueDim] [timeBase p outer inner, l.value * p.BV]⟩

def innerBody (p : Layout) (r : Regions) : List Stmt :=
  chunkDeltaInnerBody r.key r.value r.delta r.corrected
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim
    p.T p.K p.V p.BT p.BC p.BK p.BV

def beforeStore (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (outer inner : Nat)
    (frozen : StateValue p) (u : BlockState) : BlockState :=
  let u1 := u.setReg "p_k" .blockPtr [p.BK,p.BC] (keyPointer p l r.key outer inner)
  let u2 := u1.setReg "p_d" .blockPtr [p.BC,p.BK] (deltaPointer p l r.delta outer inner)
  let u3 := u2.setReg "p_v" .blockPtr [p.BC,p.BV] (valuePointer p l r.value outer inner)
  let u4 := u3.setReg "p_v_new" .blockPtr [p.BC,p.BV] (valuePointer p l r.corrected outer inner)
  let u5 := u4.setReg "b_k" .real [p.BK,p.BC] (keyTile p l a outer inner)
  let u6 := u5.setReg "b_d" .real [p.BC,p.BK] (deltaTile p l a outer inner)
  let u7 := u6.setReg "b_v" .real [p.BC,p.BV] (valueTile p l a outer inner)
  u7.setReg "b_v" .real [p.BC,p.BV] (correctedTile p l a outer inner frozen)

def scatter (p : Layout) (l : Launch) (out : RegionName) (a : Inputs) (outer inner : Nat)
    (frozen : StateValue p) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BC,p.BV]).foldl (fun st idx =>
    if activeValue p l (timeBase p outer inner + idx.1.val) idx.2.1.val then
      st.writeMem out (valueAddress p l (timeBase p outer inner + idx.1.val) idx.2.1.val)
        (correctedValue p l a frozen (timeBase p outer inner + idx.1.val) idx.2.1)
    else st) u

def result (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (outer inner : Nat)
    (frozen acc : StateValue p) (u : BlockState) : BlockState :=
  (scatter p l r.corrected a outer inner frozen (beforeStore p l r a outer inner frozen u)).setReg
    "b_h_cumsum" .real [p.BK,p.BV] (stateTile p (nextAccum p l a outer inner frozen acc))

end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealProgram
