import bench.tritonbench_g.index_select_cat.Proofs.IO
import bench.tritonbench_g.index_select_cat.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat
open VeriTile.Triton
open scoped VeriTile.Triton.GatherMasked2DKernelIO₁

/-- **The headline**: `index_select_cat_fwd_kernel` implements the pure gather
`output[i, col] = source[index[i], col]` on its gather IO signature — for every
disjoint flat placement of `source_ptr`/`index_ptr`/`output_ptr`, every pair of
program ids whose active lanes are in bounds, and every launch state whose
windows hold the row-index tile `ids` (`.nat` channel) and the gathered data
tile `xs`, the translated pointer kernel terminates; every write-active lane
`j` (`indices < num_indices ∧ cols < num_cols`, the kernel's 2D `mask`) has
`xs j` at its contiguous destination cell; and every memory cell off the store
window is unchanged — unconditionally.

The skin's readback leg carries a per-context `WriteInj` antecedent because a
`GatherMasked2DKernelIO₁` `write` window may in general eat the loaded index
tile. On this pure-gather side the `write` window is *static*, so `WriteInj` is
just the old `hOutInj` no-duplicate-destination side condition on
`i·stride0 + col·stride1` at the write-active lanes — a fact about the strides,
not about the index data, and it stays inside `⊨` rather than narrowing the
headline.

`hBC : 0 < BLOCK_SIZE_COL` is genuinely forced: the index-tile load runs over
the `[BLOCK_SIZE_INDEX]` tile regardless of `BLOCK_SIZE_COL`, so when
`BLOCK_SIZE_COL = 0` the flattened lane space `Fin (BI * BC)` is empty and
carries no in-bounds witness for it.

Proof: `GatherMasked2DKernelIO₁.Implements.intro` assembles the region-model
gather triple with the flat-memory bridge side conditions. -/
specification index_select_cat_fwd_kernel_correctness
    (output_ptr source_ptr index_ptr : RegionName)
    (num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (hBC : 0 < BLOCK_SIZE_COL) :
    indexSelectCatIO output_ptr source_ptr index_ptr num_indices num_cols
      stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL ⊨
        fun _ _ _ xs j => xs j := by
  refine GatherMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact index_select_cat_fwd_kernel_flattenOk output_ptr source_ptr index_ptr
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  · intro bounds s ids hidsPin hbx hbr hbw
    simp only [indexSelectCatIO] at hidsPin hbx hbr hbw ⊢
    -- the lane of tile cell `(i, c)` — well-formed exactly because `BC > 0`
    have hlane : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
        ((Lane2D.encode ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              / BLOCK_SIZE_COL = i.val ∧
          (Lane2D.encode ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              % BLOCK_SIZE_COL = c.val) :=
      fun i c => ⟨Lane2D.encode_div _, Lane2D.encode_mod _⟩
    refine index_select_cat_fwd_kernel_traceSafe output_ptr source_ptr index_ptr
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
      bounds s ?_ ?_ ?_
    · intro i hi
      have hc : Fin BLOCK_SIZE_COL := ⟨0, hBC⟩
      obtain ⟨hd, hm⟩ := hlane i hc
      have h := hbx (Lane2D.encode ((i, hc, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at h
      exact h
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbr (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hm] at h
      have hpin := hidsPin (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at hpin
      show s.readMemValue TileDType.nat index_ptr
            (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1 < bounds source_ptr
      rw [hpin]
      exact h
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbw (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hd, hm] at h
      exact h
  · intro s₀ ids xs hi hx
    exact index_select_cat_fwd_kernel_region_run output_ptr source_ptr index_ptr
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
      s₀ ids xs hi hx

/-- Both complete source variants implement the independent gather cell formula
from the original signed-index and source input windows. Successful flat
execution and the full frame are included. Duplicate source rows, overlapping
output strides and input/output aliasing require no injectivity premise. -/
specification source_real_correct (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    VeriTile.Spec.Real (FullGatherIO input output index v out idx inp n c s0 s1 BI BC) := by
  exact fullGatherIO_correct input output index v out idx inp n c s0 s1 BI BC

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.IndexSelectCat
