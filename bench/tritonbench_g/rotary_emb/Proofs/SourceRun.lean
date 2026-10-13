import bench.tritonbench_g.rotary_emb.Proofs.KStore
import bench.tritonbench_g.rotary_emb.Proofs.KMemory
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- The complete 34-statement source succeeds. All source loads explicitly use
`other=0`, and no precondition on initial registers or undefined memory is needed. -/
theorem source_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    exec (mathKernel p Q K Cos Sin) s = some (kStoreState p s Q K Cos Sin) := by
  unfold exec
  rw [source_parts]
  simp only [ProgramRewrite.append, prefix_run, q_index_run, q_load_run,
    q_arithmetic_run, q_store_run, k_index_run, k_load_run, k_arithmetic_run,
    k_store_run, Option.bind_some]

/-- Full source IO against the initial-memory rotation formula, including all
four store families and the complete raw-cell frame. -/
theorem source_io (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState)
    (hsep : RegionsSeparate Q K Cos Sin) (hGeometry : p.Geometry s) :
    ∃ t, exec (mathKernel p Q K Cos Sin) s = some t ∧
      (∀ odd i, p.activeQ s i →
        t.readMem Q (p.qAddress s odd i) = p.expectedQ s Q Cos Sin odd i) ∧
      (∀ odd i, p.activeK s i →
        t.readMem K (p.kAddress s odd i) = p.expectedK s K Cos Sin odd i) ∧
      BlockState.WriteWithin (p.footprint s Q K) s t := by
  refine ⟨kStoreState p s Q K Cos Sin, source_run p Q K Cos Sin s, ?_, ?_, ?_⟩
  · exact final_state_q p s Q K Cos Sin hsep hGeometry
  · exact final_state_k p s Q K Cos Sin hsep hGeometry
  · exact final_state_frame p s Q K Cos Sin hsep

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
