import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealPrologueSource

/-! Real stores used by the attention epilogue. Successful statement execution,
readback at injective addresses and preservation of every other memory cell
are separate facts. Address bounds and allocation transport come later. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealStores
open VeriTile.Triton
noncomputable section

def writeTile {shape : TileShape} (s : BlockState) (region : RegionName)
    (address : TileIndex shape → Nat) (value : TileIndex shape → ℝ) : BlockState :=
  (TileShape.allIndices shape).foldl (fun st i => st.writeMem region (address i) (value i)) s

@[simp] theorem write_tile_regs {shape : TileShape} (s : BlockState) (region : RegionName)
    (address : TileIndex shape → Nat) (value : TileIndex shape → ℝ) (dt sh name) :
    (writeTile s region address value).regs dt sh name = s.regs dt sh name := by
  exact BlockState.foldl_writeMem_regs _ _ _ _ _ _ _ _

theorem write_tile_control {shape : TileShape} (s : BlockState) (region : RegionName)
    (address : TileIndex shape → Nat) (value : TileIndex shape → ℝ) :
    (writeTile s region address value).pids = s.pids ∧
      (writeTile s region address value).undef = s.undef := by
  unfold writeTile
  generalize TileShape.allIndices shape = indices
  induction indices generalizing s with
  | nil => exact ⟨rfl, rfl⟩
  | cons i rest ih => simpa only [List.foldl_cons, BlockState.writeMem_pids] using ih (s.writeMem region (address i) (value i))

theorem write_tile_readback {shape : TileShape} (s : BlockState) (region : RegionName)
    (address : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (hi : Function.Injective address) (i : TileIndex shape) :
    (writeTile s region address value).readMem region (address i) = value i :=
  BlockState.scatter_readback_nd s address value hi i

theorem write_tile_frame {shape : TileShape} (s : BlockState) (region : RegionName)
    (address : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (r : RegionName) (o : Nat) (h : r ≠ region ∨ ∀ i, o ≠ address i) :
    (writeTile s region address value).mem r o = s.mem r o := by
  unfold writeTile
  generalize TileShape.allIndices shape = indices
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons, ih]
    have hn : ¬ (r = region ∧ o = address i) := by
      intro he
      rcases h with hr | ho
      · exact hr he.1
      · exact ho i he.2
    simp [BlockState.writeMem, hn]

theorem write_tile_read_other {shape : TileShape} (s : BlockState) (region : RegionName)
    (address : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (r : RegionName) (o : Nat) (h : r ≠ region ∨ ∀ i, o ≠ address i) :
    (writeTile s region address value).readMem r o = s.readMem r o := by
  simp only [BlockState.readMem, write_tile_frame s region address value r o h]

theorem pointer_store (s : BlockState) (shape : TileShape) (pointer value : RegName)
    (region : RegionName) (address : TileIndex shape → Nat) (v : Tile .real shape)
    (hp : s.regs .ptr shape pointer = some (⟨fun i => (region, address i)⟩ : Tile .ptr shape))
    (hv : s.regs .real shape value = some v) :
    stepStmt (.store .real shape (.ptr (.ref .ptr shape pointer)) (.ref .real shape value) .none) s =
      some (writeTile s region address (fun i => FloatDType.real.storeValue (v.data i))) := by
  conv_lhs => unfold stepStmt
  simp only [evalOp_ref, hp, hv, bind, Option.bind_some, if_true, BlockState.writeMemTyped_real]
  rfl

theorem block_store (s : BlockState) (shape : TileShape) (pointer value : RegName)
    (region : RegionName) (bp : BlockPtr) (v : Tile .real shape)
    (hr : bp.region = region)
    (hp : s.regs .blockPtr shape pointer = some (⟨fun _ => bp⟩ : Tile .blockPtr shape))
    (hv : s.regs .real shape value = some v) :
    stepStmt (.store .real shape (.blockPtr (.ref .blockPtr shape pointer) []) (.ref .real shape value) .none) s =
      some (writeTile s region (fun i => bp.address (TileShape.indexToList shape i))
        (fun i => FloatDType.real.storeValue (v.data i))) := by
  conv_lhs => unfold stepStmt
  simp only [evalOp_ref, hp, hv, bind, Option.bind_some, BlockPtr.inBounds_nil_checkedAxes,
    Bool.and_self, if_true, BlockState.writeMemTyped_real, hr]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealStores
