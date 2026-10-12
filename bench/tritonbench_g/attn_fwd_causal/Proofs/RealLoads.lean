import bench.tritonbench_g.attn_fwd_causal.Proofs.RealBody
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealTiles

/-! Exact input-memory views and masked pointer loads for both helpers.
Independent source strides and partial tails are retained. Address bounds are
separate from these value lemmas; masked-off reads use the explicit undef=0
contract, while every active value comes from the actual input memory. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoads
open VeriTile.Triton
open AttentionFwdTriton3.RealTiles (finite)
noncomputable section

structure Config where
  (Q K V QS KS : RegionName)
  (base queryScaleBase keyScaleBase : Nat)
  (strideM strideD strideKN : Nat)
  (headDim context activeHead startM : Nat)

def inputs (s : BlockState) (c : Config) (BM BN BD : Nat) : RealFormula.Inputs BM BN BD where
  query := fun row d => s.readMem c.Q (c.base+(c.startM*BM+row.val)*c.strideM+d.val*c.strideD)
  key := fun start j d => s.readMem c.K (c.base+d.val+j.val*c.strideKN+c.headDim*start)
  value := fun start j d => s.readMem c.V (c.base+j.val*c.strideM+d.val*c.strideD+c.headDim*start)
  queryScale := s.readMem c.QS (c.queryScaleBase+c.startM)
  keyScale := fun block => s.readMem c.KS (c.keyScaleBase+block)

def keyPointer (c : Config) (BD BN start : Nat) : Tile .ptr [BD,BN] :=
  ⟨fun i => (c.K.cast,c.base+i.1.val+i.2.1.val*c.strideKN+c.headDim*start)⟩
def valuePointer (c : Config) (BN BD start : Nat) : Tile .ptr [BN,BD] :=
  ⟨fun i => (c.V.cast,c.base+i.1.val*c.strideM+i.2.1.val*c.strideD+c.headDim*start)⟩
def keyScalePointer (c : Config) (BN start : Nat) : Tile .ptr [] :=
  Tile.scalar (c.KS.cast,c.keyScaleBase+start/BN)

def keyMask (BD BN N HA start : Nat) : Tile .bool [BD,BN] :=
  ⟨fun i => decide (start+i.2.1.val < N ∧ i.1.val < HA)⟩
def valueMask (BN BD N HA start : Nat) : Tile .bool [BN,BD] :=
  ⟨fun i => decide (start+i.1.val < N ∧ i.2.1.val < HA)⟩

theorem inputs_same_memory (s t : BlockState) (c : Config) (BM BN BD : Nat) (hm : s.mem = t.mem) :
    inputs s c BM BN BD = inputs t c BM BN BD := by simp only [inputs, BlockState.readMem, hm]

theorem eval_key_mask (s : BlockState) (BD BN N HA start : Nat)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start)) :
    evalOp (RealBody.keyMask BD BN N HA) s = some (keyMask BD BN N HA start) := by
  simp only [RealBody.keyMask, evalOp.eq_def, hn, hs, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyMask, Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    TileShape.dropInsertedIndex, ComparableDType.lt, NumericDType.sub,
    RealExpressions.tail_mask_iff, -Tile.scalar_eta]
  rfl

theorem eval_value_mask (s : BlockState) (BN BD N HA start : Nat)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start)) :
    evalOp (RealBody.valueMask BN BD N HA) s = some (valueMask BN BD N HA start) := by
  simp only [RealBody.valueMask, evalOp.eq_def, hn, hs, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp [valueMask, Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    TileShape.dropInsertedIndex, ComparableDType.lt, NumericDType.sub,
    RealExpressions.tail_mask_iff, -Tile.scalar_eta]
  rfl

theorem key_load (s : BlockState) (c : Config) (BM BN BD start : Nat) (pointer : RegName)
    (hp : s.regs .ptr [BD,BN] pointer = some (keyPointer c BD BN start))
    (hm : s.regs .bool [BD,BN] "k_mask" = some (keyMask BD BN c.context c.activeHead start))
    (hu : ∀ r a, s.undef r a = 0) :
    evalOp (.load .real (.ptr (.ref .ptr [BD,BN] pointer)) (.mask (.ref .bool [BD,BN] "k_mask"))) s =
      some (RealTiles.key (inputs s c BM BN BD) start c.context c.activeHead) := by
  simp only [evalOp.eq_def, hp, hm, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyMask, keyPointer, RealTiles.key, inputs, finite, hu, Region.cast]
  split <;> simp_all

theorem value_load (s : BlockState) (c : Config) (BM BN BD start : Nat) (diagonal : Bool) (pointer : RegName)
    (hp : s.regs .ptr [BN,BD] pointer = some (valuePointer c BN BD start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hu : ∀ r a, s.undef r a = 0) :
    evalOp (.load .real (.ptr (.ref .ptr [BN,BD] pointer))
      (.mask (RealBody.valueMask BN BD c.context c.activeHead))) s =
      some (RealTiles.value (inputs s c BM BN BD) ⟨start,diagonal⟩ c.context c.activeHead) := by
  have hm := eval_value_mask s BN BD c.context c.activeHead start hn hs
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hm, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp [valueMask, valuePointer, RealTiles.value, RealFormula.loadedValue, inputs, finite, hu, Region.cast]
  split <;> simp_all

theorem key_scale_load (s : BlockState) (c : Config) (BM BN BD start : Nat) (pointer : RegName)
    (hp : s.regs .ptr [] pointer = some (keyScalePointer c BN start)) :
    evalOp (.load .real (.ptr (.ref .ptr [] pointer)) .none) s =
      some (Tile.scalar (some ((inputs s c BM BN BD).keyScale (start/BN)))) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  cases i
  simp [keyScalePointer, inputs, Tile.scalar, Region.cast, -Tile.scalar_eta]

theorem key_advance (s : BlockState) (c : Config) (BN BD start : Nat) (name : RegName)
    (hp : s.regs .ptr [BD,BN] name = some (keyPointer c BD BN start)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BD,BN] name)
      (.mul .nat .nil (.constNat BN) (.constNat c.headDim))) s =
      some (keyPointer c BD BN (start+BN)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  simp [keyPointer, Tile.scalar, NumericDType.mul, Nat.mul_add, Nat.mul_comm, Nat.add_assoc, -Tile.scalar_eta]

theorem value_advance (s : BlockState) (c : Config) (BN BD start : Nat) (name : RegName)
    (hp : s.regs .ptr [BN,BD] name = some (valuePointer c BN BD start)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BN,BD] name)
      (.mul .nat .nil (.constNat BN) (.constNat c.headDim))) s =
      some (valuePointer c BN BD (start+BN)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  simp [valuePointer, Tile.scalar, NumericDType.mul, Nat.mul_add, Nat.mul_comm, Nat.add_assoc, -Tile.scalar_eta]

theorem scale_advance (s : BlockState) (c : Config) (BN start : Nat) (hN : 0 < BN) (name : RegName)
    (hp : s.regs .ptr [] name = some (keyScalePointer c BN start)) :
    evalOp (.ptrAdd .nil (.ref .ptr [] name) (.constNat 1)) s =
      some (keyScalePointer c BN (start+BN)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  cases i
  simp [keyScalePointer, Tile.scalar, Nat.add_div_right start hN, Nat.add_assoc, -Tile.scalar_eta]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoads
