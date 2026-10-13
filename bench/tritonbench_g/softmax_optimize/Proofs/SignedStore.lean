import bench.tritonbench_g.softmax_optimize.Proofs.SignedExpression

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

namespace IndexedRelated

theorem write_real {B : Nat} {s t : BlockState} (h : IndexedRelated B s t)
    (r : RegionName) (o : Nat) (v : TileCarrier .real) :
    IndexedRelated B (s.writeMemTyped .real r o v) (t.writeMemTyped .real r o v) := by
  refine ⟨?_, h.columns_eq, h.offset_eq⟩
  exact ⟨h.pids_eq, h.numPids_eq, by
    simp only [BlockState.writeMemTyped, BlockState.writeMemAs, h.mem_eq],
    h.undef_eq, h.regs_eq, h.prev_eq⟩

theorem scatter_real {B : Nat} {shape : TileShape} {s t : BlockState}
    (h : IndexedRelated B s t) (ptrs : Tile .ptr shape) (values : Tile .real shape)
    (active : TileIndex shape → Bool) (indices : List (TileIndex shape)) :
    IndexedRelated B
      (indices.foldl (fun acc i ↦ if active i then
        acc.writeMemTyped .real (ptrs.data i).1 (ptrs.data i).2 (values.data i) else acc) s)
      (indices.foldl (fun acc i ↦ if active i then
        acc.writeMemTyped .real (ptrs.data i).1 (ptrs.data i).2 (values.data i) else acc) t) := by
  induction indices generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    split
    · exact h.write_real _ _ _
    · exact h

end IndexedRelated

theorem store_full_related (B : Nat) : ∀ s t, IndexedRelated B s t →
    ResultsRelated (IndexedRelated B)
      (stepStmt (.store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
        (.ref .real [B] "out") .none) s)
      (stepStmt (.store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
        (.ref .real [B] "out") .none) t) := by
  intro s t h
  have hp := h.regs_eq .ptr [B] "output_ptrs" (by simp [Shared, Scratch])
  have hv := h.regs_eq .real [B] "out" (by simp [Shared, Scratch])
  simp only [stepStmt, evalOp_ref, hp, hv]
  cases t.regs .real [B] "out" <;> cases t.regs .ptr [B] "output_ptrs" <;>
    simp only [Option.bind_some, Option.bind_none, ResultsRelated]
  · exact True.intro
  · exact True.intro
  · exact True.intro
  · exact h.scatter_real _ _ _ _

theorem store_tail_related (B : Nat) : ∀ s t, IndexedRelated B s t →
    ResultsRelated (IndexedRelated B)
      (stepStmt (.store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
        (.ref .real [B] "out") (.mask (.ref .bool [B] "mask"))) s)
      (stepStmt (.store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
        (.ref .real [B] "out") (.mask (.ref .bool [B] "mask"))) t) := by
  intro s t h
  have hp := h.regs_eq .ptr [B] "output_ptrs" (by simp [Shared, Scratch])
  have hv := h.regs_eq .real [B] "out" (by simp [Shared, Scratch])
  have hm := h.regs_eq .bool [B] "mask" (by simp [Shared, Scratch])
  simp only [stepStmt, evalOp_ref, hp, hv, hm]
  cases t.regs .real [B] "out" <;> cases t.regs .bool [B] "mask" <;>
    cases t.regs .ptr [B] "output_ptrs" <;>
    simp only [Option.bind_some, Option.bind_none, Option.map_some, Option.map_none, ResultsRelated]
  all_goals first | exact True.intro | exact h.scatter_real _ _ _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
