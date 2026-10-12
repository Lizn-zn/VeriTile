import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealResidual

/-! The actual terminal masked store, with tagged-cell readback and a full
memory frame. Output strides remain arbitrary; only active writes need be
injective. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealStore
open VeriTile.Triton
noncomputable section

def offset (BM BN base row column : Nat) (rows : Fin BM → Nat)
    (columns : Fin BN → Nat) (i : TileIndex [BM, BN]) : Nat :=
  base + (rows i.1 * row + columns i.2.1 * column)

def active (BM BN length K : Nat) (rows : Fin BM → Nat)
    (columns : Fin BN → Nat) (i : TileIndex [BM, BN]) : Prop :=
  rows i.1 < length ∧ columns i.2.1 < K

instance (BM BN length K : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat) :
    DecidablePred (active BM BN length K rows columns) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def code (BM BN K : Nat) : Stmt :=
  .store .real [BM, BN] (.ptr (.ref .ptr [BM, BN] "db_ptrs"))
    (.ref .real [BM, BN] "db") (.mask (RealResidual.mask BM BN K))

theorem final_source_statement (swapped : Bool) (BM BN : Nat) (p : RealProgram.Parameters) :
    (RealResidual.code swapped BM BN p).getLast? = some (code BM BN p.K) := by
  cases swapped <;> rfl

def write (s : BlockState) (region : RegionName) (BM BN base row column length K : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl (fun t i =>
    if active BM BN length K rows columns i then
      t.writeMem region (offset BM BN base row column rows columns i) (values i.1 i.2.1) else t) s

theorem executes (s : BlockState) (region : RegionName) (BM BN base row column length K : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (hp : s.regs .ptr [BM, BN] "db_ptrs" =
      some (RealResidual.pointerValues region BM BN base row column rows columns))
    (hv : s.regs .real [BM, BN] "db" = some (Tile.mat fun i j => some (values i j)))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length)) :
    stepStmt (code BM BN K) s =
      some (write s region BM BN base row column length K rows columns values) := by
  have hm := RealResidual.mask_eval s BM BN K length rows columns hr hn hl
  rw [code, stepStmt]
  simp only [evalOp_ref, hp, hv, hm, bind, Option.bind_some,
    RealResidual.pointerValues, Tile.mat, Option.map_some, decide_eq_true_eq, BlockState.writeMemTyped_real]
  rfl

theorem frame (s : BlockState) (region : RegionName) (BM BN base row column length K : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (r : RegionName) (address : Nat)
    (h : ∀ i, active BM BN length K rows columns i →
      ¬ (region = r ∧ offset BM BN base row column rows columns i = address)) :
    (write s region BM BN base row column length K rows columns values).mem r address = s.mem r address := by
  exact foldl_store_preserve_cell _ _ _ r address _ s (fun i _ hi => h i hi)

theorem readback (s : BlockState) (region : RegionName) (BM BN base row column length K : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (i : TileIndex [BM, BN]) (hi : active BM BN length K rows columns i)
    (hinj : ∀ a b, active BM BN length K rows columns a → active BM BN length K rows columns b →
      offset BM BN base row column rows columns a = offset BM BN base row column rows columns b → a = b) :
    (write s region BM BN base row column length K rows columns values).mem region
      (offset BM BN base row column rows columns i) = MemCell.real (values i.1 i.2.1) := by
  obtain ⟨before, after, hsplit⟩ := List.append_of_mem (TileShape.mem_allIndices [BM, BN] i)
  have hn := TileShape.allIndices_nodup [BM, BN]
  rw [hsplit, List.nodup_append, List.nodup_cons] at hn
  have hafter : ∀ k ∈ after, active BM BN length K rows columns k →
      ¬(region = region ∧ offset BM BN base row column rows columns k =
        offset BM BN base row column rows columns i) := by
    intro k hk ha he
    have hki := hinj k i ha hi he.2
    subst k
    exact hn.2.1.1 hk
  unfold write
  rw [hsplit, List.foldl_append, List.foldl_cons]
  rw [if_pos hi, foldl_store_preserve_cell _ _ _ _ _ after _ hafter,
    BlockState.writeMem_mem, if_pos ⟨rfl, rfl⟩]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealStore
