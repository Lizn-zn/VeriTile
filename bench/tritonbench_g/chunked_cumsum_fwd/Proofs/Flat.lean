import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Safety

/-! Complete finite flat-memory execution. The generic adapter initializes
its flat unspecified-value oracle to zero. We apply it to a region state with
that same oracle: all source loads specify other=0 and the region correctness
proof accepts arbitrary oracles. Inputs, program IDs and the entire initial
flat state are unchanged, so no oracle premise is added to this contract. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Flat
open VeriTile.Triton Math Program Initialization Execution
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, Execution.footprint p l r (rg,off) ∧ A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (u v : BlockState)
    (hf : BlockState.WriteWithin (Execution.footprint p l r) u v) :
    BlockState.WriteWithin (footprint A p l r) (A.flattenState u) (A.flattenState v) := by
  intro rg off hn
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨rg',off'⟩ := cell
      obtain ⟨_,heq,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro hh
      exact hn ⟨rfl,rg',off',hh,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem correct (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool) (dtype out_dtype : FloatDType)
    (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState)
    (hdt : ∀ off, u.readMem r.dt off = a.dt off)
    (hbi : ∀ off, u.readMem r.bias off = a.bias off)
    (hra : ∀ off, u.readMem r.rate off = a.rate off)
    (hRA : r.rate ≠ r.out) (hOS : r.out ≠ r.sum)
    (hiO : Stores.NoCollision p (launch u) (outOffsets p (launch u)))
    (hiS : Stores.NoCollision p (launch u) (sumOffsets p (launch u)))
    (hb : Safety.Bounds p (launch u) r hasBias A.extent) :
    ∃ v, exec (A.flattenKernel (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
      (∀ (i : Fin p.BH) (j : Fin p.BC), activeOutput p (launch u) i.val j.val →
        v.readMem A.flat (A.addr r.out (outAddress p (launch u) i.val j.val)) =
          prepared p (launch u) a hasBias useSoftplus lower upper i.val j.val ∧
        v.readMem A.flat (A.addr r.sum (sumAddress p (launch u) i.val j.val)) =
          cumulative p (launch u) a hasBias useSoftplus lower upper i j) ∧
      BlockState.WriteWithin (footprint A p (launch u) r) (A.flattenState u) v := by
  let s : BlockState := { u with undef := fun _ _ => 0 }
  obtain ⟨v,he,ho,hf⟩ := Execution.correct p r bias swapped dtype out_dtype a hasBias useSoftplus lower upper s
    hdt hbi hra hRA hOS hiO hiS
  have hflat := A.exec_flatten hd hcov _ s
    (Safety.source_safe p r bias swapped dtype out_dtype hasBias useSoftplus lower upper s A.extent hb)
    (Safety.source_flatten_ok p r bias swapped dtype out_dtype hasBias useSoftplus lower upper) rfl
  rw [he,Option.map_some] at hflat
  have hmem (rg : RegionName) (off : Nat) (hbound : off < A.extent rg) :
      (A.flattenState v).readMem A.flat (A.addr rg off) = v.readMem rg off := by
    have hr : rg ∈ A.regions := by
      by_contra hn
      rw [hcov rg hn] at hbound
      exact Nat.not_lt_zero _ hbound
    exact A.flattenState_readMem hd v hr hbound
  refine ⟨A.flattenState v,hflat,?_,flatten_frame A p (launch u) r s v hf⟩
  intro i j hi
  rw [hmem _ _ (hb.out i j hi),hmem _ _ (hb.sum i j hi)]
  exact ho i j hi

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Flat
