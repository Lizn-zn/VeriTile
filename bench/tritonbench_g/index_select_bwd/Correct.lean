import bench.tritonbench_g.index_select_bwd.Proofs.Real
import bench.tritonbench_g.index_select_bwd.Proofs.IO

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd

open VeriTile.Triton
open scoped VeriTile.Triton.GatherMasked2DKernelIO₁

/-- **The headline**: `index_select_cat_bwd_kernel` implements the indexed
gradient scatter `grad_source[index[i], col] = grad_output[i, col]` on its
scatter IO signature — for every disjoint flat placement of
`grad_output_ptr`/`index_ptr`/`grad_source_ptr`, every pair of program ids
whose active lanes are in bounds (the write bound at the *ungated* scatter
cells), and every launch state whose windows hold the row-index tile `ids`
(`.nat` channel) and the gradient tile `xs`, the translated pointer kernel
terminates; under the per-context `WriteInj` antecedent every write-active lane
`j` (`grad_output_indices < num_indices ∧ cols < num_cols`, the kernel's
`grad_output_mask`) has `xs j` at its scatter cell
`ids j * stride0 + col * stride1`; and every memory cell off the raw scatter
cells is unchanged — unconditionally.

`WriteInj` here is **genuinely non-vacuous and genuinely required**: the
benchmark's own test cases pass repeated indices (`[0, 0, 0, 0, 0]`,
`[9, 9, 9, 9, 9]`), two lanes of one column block then scatter to the same cell,
and no pointwise readback can hold. It is the honest no-duplicate-destination
precondition and it lives inside `⊨`, so the headline itself stays free of
scope-narrowing hypotheses.

`hBC : 0 < BLOCK_SIZE_COL` is genuinely forced: the index-tile load runs over
the `[BLOCK_SIZE_INDEX]` tile regardless of `BLOCK_SIZE_COL`, so when
`BLOCK_SIZE_COL = 0` the flattened lane space is empty and carries no in-bounds
witness for it.

Proof: `GatherMasked2DKernelIO₁.Implements.intro` assembles the region-model
scatter triple with the flat-memory bridge side conditions. -/
specification index_select_cat_bwd_kernel_correctness
    (grad_source_ptr index_ptr grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (hBC : 0 < BLOCK_SIZE_COL) :
    indexSelectBwdIO grad_source_ptr index_ptr grad_output_ptr num_rows
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL ⊨
        fun _ _ _ xs j => xs j := by
  refine GatherMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact index_select_cat_bwd_kernel_flattenOk grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  · intro bounds s ids hidsPin hbx hbr hbw
    simp only [indexSelectBwdIO] at hidsPin hbx hbr hbw ⊢
    have hlane : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
        ((Lane2D.encode ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              / BLOCK_SIZE_COL = i.val ∧
          (Lane2D.encode ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              % BLOCK_SIZE_COL = c.val) :=
      fun i c => ⟨Lane2D.encode_div _, Lane2D.encode_mod _⟩
    refine index_select_cat_bwd_kernel_traceSafe grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL bounds s ?_ ?_ ?_
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbr (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]))
        (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hd, hm] at h
      exact h
    · intro i hi
      have hc : Fin BLOCK_SIZE_COL := ⟨0, hBC⟩
      obtain ⟨hd, hm⟩ := hlane i hc
      have h := hbx (Lane2D.encode ((i, hc, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at h
      exact h
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbw (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]))
        (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hm] at h
      have hpin := hidsPin (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at hpin
      show s.readMemValue TileDType.nat index_ptr
            (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1 < bounds grad_source_ptr
      rw [hpin]
      exact h
  · intro s₀ ids xs hi hx
    exact index_select_cat_bwd_kernel_region_run grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL s₀ ids xs hi hx


/-- Both complete source variants independently implement the last-active-write
scatter formula from the original signed-index and gradient input windows.
Successful flat execution and every untouched memory cell are included;
duplicate destinations and input/output aliasing require no injectivity premise. -/
specification source_real_correct (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) :
    VeriTile.Spec.Real (FullScatterIO input output index v out idx inp rows n c s0 s1 BI BC) := by
  exact fullScatterIO_correct input output index v out idx inp rows n c s0 s1 BI BC

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.IndexSelectBwd
