import bench.tritonbench_g.attention_score.Proofs.RealTiles
import VeriTile.Triton.Semantics.BlockPtrEval

/-! Actual Q/K/M memory reads after precision erasure. The input view refers
only to initial input cells. Checked blocks yield zero outside N/NK; unchecked
blocks read their actual addresses, with access bounds proved separately. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealLoads
open VeriTile.Triton
noncomputable section

/-- The three independent input streams at this program's grouped-head bases. -/
def inputs (s : BlockState) (Q K M : RegionName)
    (D baseQ baseK baseM sqm sqk skn skk : Nat) : RealFormula.Inputs D where
  query := fun row d => s.readMem Q (baseQ + row * sqm + d.val * sqk)
  key := fun key d => s.readMem K (baseK + d.val * skk + key * skn)
  normalizer := fun row => s.readMem M (baseM + row)

def queryPointer (Q : RegionName) (B D N base sqm sqk row : Nat) : Tile .blockPtr [B,D] :=
  ⟨fun _ => { region := Q, baseOffset := base, parentShape := [N,D], blockShape := [B,D], strides := [sqm,sqk], offsets := [row,0] }⟩

def keyPointer (K : RegionName) (B D NK base skn skk startN : Nat) : Tile .blockPtr [D,B] :=
  ⟨fun _ => { region := K, baseOffset := base, parentShape := [D,NK], blockShape := [D,B], strides := [skk,skn], offsets := [0,startN*B] }⟩

def normalizerPointers (M : RegionName) (B base row : Nat) : Tile .ptr [B] :=
  ⟨fun i => (M.cast, base + row + i.1.val)⟩

def checks (even : Bool) : List Nat := if even then [] else [0,1]

theorem inputs_same_memory (s t : BlockState) (Q K M : RegionName)
    (D baseQ baseK baseM sqm sqk skn skk : Nat) (hm : s.mem = t.mem) :
    inputs s Q K M D baseQ baseK baseM sqm sqk skn skk =
      inputs t Q K M D baseQ baseK baseM sqm sqk skn skk := by
  simp only [inputs, BlockState.readMem, hm]

theorem query_load (s : BlockState) (Q K M : RegionName)
    (B D N baseQ baseK baseM sqm sqk skn skk row : Nat) (evenM : Bool)
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (queryPointer Q B D N baseQ sqm sqk row)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [B,D] "Q_block_ptr") (checks evenM)) .none) s =
      some (RealTiles.query (inputs s Q K M D baseQ baseK baseM sqm sqk skn skk) B N row evenM) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨i,d,⟨⟩⟩
  cases evenM
  all_goals simp [checks, queryPointer, RealTiles.query, inputs, d.isLt,
    BlockState.readMemValue_real, BlockState.defaultCarrier]
  all_goals split_ifs <;> rfl

theorem key_load (s : BlockState) (Q K M : RegionName)
    (B D NK baseQ baseK baseM sqm sqk skn skk startN : Nat) (evenN : Bool)
    (hp : s.regs .blockPtr [D,B] "K_block_ptr" = some (keyPointer K B D NK baseK skn skk startN)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [D,B] "K_block_ptr") (checks evenN)) .none) s =
      some (RealTiles.key (inputs s Q K M D baseQ baseK baseM sqm sqk skn skk) B NK startN evenN) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  cases evenN
  all_goals simp [checks, keyPointer, RealTiles.key, inputs, d.isLt,
    BlockState.readMemValue_real, BlockState.defaultCarrier]
  all_goals split_ifs <;> rfl

theorem normalizer_load (s : BlockState) (Q K M : RegionName)
    (B D baseQ baseK baseM sqm sqk skn skk row : Nat)
    (hp : s.regs .ptr [B] "m_ptrs" = some (normalizerPointers M B baseM row)) :
    evalOp (.load .real (.ptr (.ref .ptr [B] "m_ptrs")) .none) s =
      some (RealTiles.normalizer (inputs s Q K M D baseQ baseK baseM sqm sqk skn skk) B row) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [normalizerPointers, RealTiles.normalizer, inputs, Nat.add_assoc, Region.cast]

/-- The source advances Q by one query block, leaving its shape and strides. -/
theorem query_advance (s : BlockState) (Q : RegionName)
    (B D N base sqm sqk row : Nat)
    (hp : s.regs .blockPtr [B,D] "Q_block_ptr" = some (queryPointer Q B D N base sqm sqk row)) :
    evalOp (.advanceBlockPtr (.ref .blockPtr [B,D] "Q_block_ptr") [B,0]) s =
      some (queryPointer Q B D N base sqm sqk (row+B)) := by
  simp only [advanceBlockPtr_eval, evalOp_ref, hp, bind, Option.bind]
  rfl

theorem normalizer_advance (s : BlockState) (M : RegionName) (B base row : Nat)
    (hp : s.regs .ptr [B] "m_ptrs" = some (normalizerPointers M B base row)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [B] "m_ptrs") (.constNat B)) s =
      some (normalizerPointers M B base (row+B)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [normalizerPointers, Tile.scalar, Nat.add_left_comm, Nat.add_comm, -Tile.scalar_eta]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealLoads
