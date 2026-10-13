import VeriTile.Triton.Memory.Frame.Basic
import VeriTile.Triton.Semantics.State
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.Scatter
open VeriTile.Triton

noncomputable def one {shape : TileShape} (X : RegionName)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState) : BlockState := by
  classical
  exact (TileShape.allIndices shape).foldl
    (fun st i ↦ if active i then st.writeMem X (offset i) (value i) else st) s

noncomputable def pair {shape : TileShape} (X : RegionName)
    (offset : Bool → TileIndex shape → Nat) (value : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState) : BlockState :=
  one X (offset Bool.true) (value Bool.true) active
    (one X (offset Bool.false) (value Bool.false) active s)

def PairDisjoint {shape : TileShape} (offset : Bool → TileIndex shape → Nat)
    (active : TileIndex shape → Prop) : Prop :=
  ∀ a b i j, active i → active j → offset a i=offset b j → a=b ∧ i=j

theorem one_value_congr {shape : TileShape} (X : RegionName)
    (offset : TileIndex shape → Nat) (f g : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState)
    (h : ∀ i, active i → f i=g i) :
    one X offset f active s = one X offset g active s := by
  classical
  unfold one
  congr 1
  funext st i
  by_cases hi : active i
  · simp only [hi, ↓reduceIte, h i hi]
  · simp only [hi, ↓reduceIte]

theorem pair_value_congr {shape : TileShape} (X : RegionName)
    (offset : Bool → TileIndex shape → Nat) (f g : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState)
    (h : ∀ odd i, active i → f odd i=g odd i) :
    pair X offset f active s = pair X offset g active s := by
  unfold pair
  rw [one_value_congr X _ (f Bool.false) (g Bool.false) active s (h Bool.false)]
  exact one_value_congr X _ _ _ active _ (h Bool.true)

/-- The source computes both values before either write. Cross-family
noncollision keeps the even value intact after the following odd store. -/
theorem pair_readback {shape : TileShape} (X : RegionName)
    (offset : Bool → TileIndex shape → Nat) (value : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState)
    (hdisjoint : PairDisjoint offset active) (odd : Bool) (i : TileIndex shape)
    (hi : active i) :
    (pair X offset value active s).readMem X (offset odd i) = value odd i := by
  classical
  cases odd
  · unfold pair one
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem]
    · exact BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ active i hi
        (fun j hj he ↦ (hdisjoint Bool.false Bool.false j i hj hi he).2)
    · intro j _ hj he
      have hf := (hdisjoint Bool.false Bool.true i j hi hj he).1
      cases hf
  · unfold pair one
    exact BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ active i hi
      (fun j hj he ↦ (hdisjoint Bool.true Bool.true j i hj hi he).2)

theorem one_frame {shape : TileShape} {P : WriteFootprint} (X : RegionName)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState)
    (hmem : ∀ i, active i → P (X,offset i)) :
    BlockState.WriteWithin P s (one X offset value active s) := by
  classical
  exact BlockState.foldl_writeMem_prop_writeWithin X offset value active _ s
    (fun i _ hi ↦ hmem i hi)

theorem pair_frame {shape : TileShape} {P : WriteFootprint} (X : RegionName)
    (offset : Bool → TileIndex shape → Nat) (value : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState)
    (hmem : ∀ odd i, active i → P (X,offset odd i)) :
    BlockState.WriteWithin P s (pair X offset value active s) :=
  BlockState.writeWithin_trans (one_frame X _ _ active s (hmem Bool.false))
    (one_frame X _ _ active _ (hmem Bool.true))

theorem pair_read_other {shape : TileShape} (X R : RegionName) (hRX : R≠X)
    (offset : Bool → TileIndex shape → Nat) (value : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState) (o : Nat) :
    (pair X offset value active s).readMem R o = s.readMem R o := by
  classical
  unfold pair one
  rw [BlockState.scatter_prop_masked_preserves_other_region X _ _ active R hRX,
    BlockState.scatter_prop_masked_preserves_other_region X _ _ active R hRX]

theorem one_mem_congr {shape : TileShape} (X : RegionName)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s t : BlockState) (hmem : s.mem=t.mem) :
    (one X offset value active s).mem = (one X offset value active t).mem := by
  classical
  unfold one
  generalize TileShape.allIndices shape = indices
  induction indices generalizing s t with
  | nil => exact hmem
  | cons i rest ih =>
      simp only [List.foldl_cons]
      by_cases hi : active i
      · simp only [hi, ↓reduceIte]
        apply ih
        funext R o
        simp only [BlockState.writeMem_mem, hmem]
      · simpa only [hi, ↓reduceIte] using ih s t hmem

theorem pair_mem_congr {shape : TileShape} (X : RegionName)
    (offset : Bool → TileIndex shape → Nat) (value : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s t : BlockState) (hmem : s.mem=t.mem) :
    (pair X offset value active s).mem = (pair X offset value active t).mem :=
  one_mem_congr X _ _ active _ _ (one_mem_congr X _ _ active s t hmem)

@[simp] theorem one_regs {shape : TileShape} (X : RegionName)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState) :
    (one X offset value active s).regs = s.regs := by
  classical
  funext dtype shape name
  exact BlockState.foldl_writeMem_prop_masked_regs X offset value active _ s dtype shape name

@[simp] theorem one_pids {shape : TileShape} (X : RegionName)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState) :
    (one X offset value active s).pids = s.pids := by
  classical
  exact BlockState.foldl_writeMem_prop_masked_pids X offset value active _ s

@[simp] theorem pair_regs {shape : TileShape} (X : RegionName)
    (offset : Bool → TileIndex shape → Nat) (value : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState) :
    (pair X offset value active s).regs = s.regs := by
  classical
  simp only [pair, one_regs]

@[simp] theorem pair_pids {shape : TileShape} (X : RegionName)
    (offset : Bool → TileIndex shape → Nat) (value : Bool → TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) (s : BlockState) :
    (pair X offset value active s).pids = s.pids := by
  classical
  simp only [pair, one, BlockState.foldl_writeMem_prop_masked_pids]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.Scatter
