import bench.tritonbench_g.index_select_bwd.Proofs.RealExecution
import bench.tritonbench_g.index_select_bwd.Proofs.Safety
import bench.tritonbench_g.index_select_bwd.Proofs.Candidate
import bench.tritonbench_g.index_select_bwd.Proofs.Precision

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- Complete signed-index scatter IO. Every allocated output cell satisfies the
independent last-active-write formula, including duplicate destinations; every
other flat cell is unchanged. Input/output names may alias. Nonnegative signed
rows make the source's signed pointer arithmetic faithful to natural addresses. -/
def FullScatterIO (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) : Prop :=
  ∀ (A : FlatAlloc), A.Disjoint → A.regions = [inp,idx,out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (ids : Fin BI → Int) (xs : Fin BI → Fin BC → ℝ),
    s.undef = (fun _ _ => 0) →
    (∀ (i : Fin BI), s.pids 0 * BI + i.val < n → 0 ≤ ids i) →
    (∀ (i : Fin BI), s.pids 0 * BI + i.val < n →
      s.pids 0 * BI + i.val < A.extent idx) →
    (∀ (i : Fin BI) (j : Fin BC), s.pids 0 * BI + i.val < n →
      s.pids 1 * BC + j.val < c →
      (s.pids 0 * BI + i.val) * s0 + (s.pids 1 * BC + j.val) * s1 < A.extent inp) →
    (∀ (i : Fin BI) (j : Fin BC), s.pids 0 * BI + i.val < n →
      s.pids 1 * BC + j.val < c →
      (ids i).toNat * s0 + (s.pids 1 * BC + j.val) * s1 < A.extent out) →
    (∀ (i : Fin BI), s.pids 0 * BI + i.val < n →
      s.readMemValue .int idx (s.pids 0 * BI + i.val) = ids i) →
    (∀ (i : Fin BI) (j : Fin BC), s.pids 0 * BI + i.val < n →
      s.pids 1 * BC + j.val < c →
      s.readMem inp ((s.pids 0 * BI + i.val) * s0 + (s.pids 1 * BC + j.val) * s1) = xs i j) →
    ∃ t, exec (A.flattenKernel
        (source input output index v out idx inp rows n c s0 s1 BI BC).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧
      (∀ r ∈ A.regions, ∀ o, o < A.extent r →
        t.mem A.flat (A.addr r o) = A.trCell
          (scatterCell BI BC (s.pids 0) (s.pids 1) n c s0 s1 ids xs out r o (s.mem r o))) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ (i : Fin BI) (j : Fin BC),
          s.pids 0 * BI + i.val < n → s.pids 1 * BC + j.val < c →
          o ≠ A.addr out ((ids i).toNat * s0 + (s.pids 1 * BC + j.val) * s1)) →
        t.mem r o = (A.flattenState s).mem r o)

theorem fullScatterIO_correct (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) :
    FullScatterIO input output index v out idx inp rows n c s0 s1 BI BC := by
  intro A hd hregions hclosed s ids xs hu hn hidx hr hw hi hx
  have hs := source_traceSafe out idx inp rows n c s0 s1 BI BC A.extent s ids hi hn hr hidx hw
  obtain ⟨t,he⟩ := source_terminates out idx inp rows n c s0 s1 BI BC s
  have hm := source_memory out idx inp rows n c s0 s1 BI BC s t ids xs hi hx hn he
  refine ⟨A.flattenState t,?_,?_,?_⟩
  · rw [source_flat_exec, A.exec_flatten hd hclosed _ s hs
      (source_flattenOk out idx inp rows n c s0 s1 BI BC) hu, he, Option.map_some]
  · intro r hr o ho
    rw [A.flattenState_mem_addr hd t hr ho,hm r o]
  · intro r o hmiss
    by_cases hrf : r = A.flat
    · subst r
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o with
      | none => rfl
      | some pair =>
        obtain ⟨r',o'⟩ := pair
        obtain ⟨_,heq,_⟩ := A.decode_sound hdec
        apply congrArg A.trCell
        refine (hm r' o').trans ?_
        apply scatterCell_frame
        intro i j hir hjc
        by_cases hro : r' = out
        · subst r'
          right
          intro hoff
          rcases hmiss with hf | hn
          · exact hf rfl
          · exact hn i j hir hjc (by rw [heq,hoff])
        · exact Or.inl hro
    · simp only [FlatAlloc.flattenState,if_neg hrf]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.IndexSelectBwd
