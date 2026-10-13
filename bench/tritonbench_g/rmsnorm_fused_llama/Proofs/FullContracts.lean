import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.FullSafety
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Safety

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
open VeriTile.Triton
set_option maxHeartbeats 3000000

/-- Full logical-row flat IO, including successful execution and every untouched cell. -/
def FullExactIO (kernel : ComputeKernel) (X Y W : RegionName)
    (stride N : Nat) (eps : ℝ) : Prop :=
  ∀ (A : FlatAlloc) (s0 : BlockState), A.Disjoint → A.regions = [X, W, Y] →
    (∀ r, r ∉ A.regions → A.extent r = 0) → s0.undef = (fun _ _ => 0) →
    (∀ k : Fin N, s0.pids 0 * stride + k.val < A.extent X) →
    (∀ k : Fin N, k.val < A.extent W) →
    (∀ k : Fin N, s0.pids 0 * stride + k.val < A.extent Y) →
    ∃ t, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState s0) = some t ∧
      (∀ k : Fin N, t.readMem A.flat (A.addr Y (s0.pids 0 * stride + k.val)) =
        rowSpec s0 X W stride N eps k.val) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ k : Fin N, o ≠ A.addr Y (s0.pids 0 * stride + k.val)) →
        t.mem r o = (A.flattenState s0).mem r o)

theorem flatten_frame (A : FlatAlloc) (s0 s : BlockState) (Y : RegionName)
    (stride N : Nat) (h : OutputFrame s0 s Y stride N)
    (r : RegionName) (o : Nat)
    (hno : r ≠ A.flat ∨ ∀ k : Fin N, o ≠ A.addr Y (s0.pids 0 * stride + k.val)) :
    (A.flattenState s).mem r o = (A.flattenState s0).mem r o := by
  change (if r = A.flat then A.readFlat s o else MemCell.real 0) =
    (if r = A.flat then A.readFlat s0 o else MemCell.real 0)
  by_cases hr : r = A.flat
  · subst r
    simp only [if_pos rfl]
    unfold FlatAlloc.readFlat
    cases hc : A.decode o with
    | none => rfl
    | some p =>
      apply congrArg A.trCell
      apply h p.1 p.2
      by_cases hp : p.1 = Y
      · right
        intro k hk
        rcases hno with hne | hmiss
        · exact hne rfl
        · apply hmiss k
          have haddr := (A.decode_sound hc).2.1
          simpa [hp, hk] using haddr
      · exact Or.inl hp
  · simp [hr]

theorem typed_full_io (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXY : X ≠ Y) (hWY : W ≠ Y) :
    FullExactIO (typedKernel input weight output v X Y W stride N B eps).eraseDType
      X Y W stride N eps := by
  intro A s0 hd hr hc hu hbx hbw hby
  obtain ⟨t, he, hv, hf⟩ := real_execution s0 X Y W stride N B eps hB hXY hWY
  have hs := real_trace_safe A.extent s0 X Y W stride N B eps hB hXY hWY hbx hbw hby
  have hflat := A.execR_flatten hd hc .triv
    (mathSource X Y W stride N B eps).toAlgKernel s0 hs
    (mathSource_flattenOk X Y W stride N B eps) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState t, ?_, ?_, flatten_frame A s0 t Y stride N hf⟩
  · exact (typed_flat_exec A input weight output v X Y W stride N B eps
      (A.flattenState s0)).trans hflat
  · intro k
    have hY : Y ∈ A.regions := by rw [hr]; simp
    have hh := A.flattenState_readMemAs hd t hY (hby k) .real
    have hm : (A.flattenState t).readMem A.flat (A.addr Y (s0.pids 0 * stride + k.val)) =
        t.readMem Y (s0.pids 0 * stride + k.val) := by
      simp only [BlockState.readMemAs_real] at hh
      exact Option.some.inj hh
    exact hm.trans (hv k)

theorem typed_full_trace_safe (input weight output : Storage) (v : Bool)
    (bounds : RegionBounds) (s0 : BlockState) (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) (hB : 0 < B) (hXY : X ≠ Y) (hWY : W ≠ Y)
    (hbx : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds X)
    (hbw : ∀ k : Fin N, k.val < bounds W)
    (hby : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds Y) :
    (typedKernel input weight output v X Y W stride N B eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s0 :=
  (Safety.typed_safe_iff bounds input weight output v X Y W stride N B eps s0).mpr
    (real_trace_safe bounds s0 X Y W stride N B eps hB hXY hWY hbx hbw hby)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
