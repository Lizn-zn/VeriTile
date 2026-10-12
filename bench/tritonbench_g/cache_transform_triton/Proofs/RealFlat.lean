import bench.tritonbench_g.cache_transform_triton.Proofs.RealSafety
import bench.tritonbench_g.cache_transform_triton.Proofs.RealValues
import bench.tritonbench_g.cache_transform_triton.Proofs.DecodeValues
import bench.tritonbench_g.cache_transform_triton.Proofs.RealFrame
import bench.tritonbench_g.cache_transform_triton.Proofs.RealCandidate
import VeriTile.Triton.Memory.Denotation

/-! Successful full executions, both output windows, and preservation of
every other typed cell in one flat address space. The candidate shares the
original's complete final state, including both stores. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealFlat
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

private theorem flat_frame {ι : Type} (A : FlatAlloc) (cosOut sinOut : RegionName)
    (active : ι → Prop) (off : ι → Nat) (initial final : BlockState)
    (hf : ∀ r o, (∀ i, active i → (r = cosOut ∨ r = sinOut) → off i ≠ o) →
      final.mem r o = initial.mem r o) :
    ∀ r o, (r ≠ A.flat ∨ ∀ i, active i →
      o ≠ A.addr cosOut (off i) ∧ o ≠ A.addr sinOut (off i)) →
      (A.flattenState final).mem r o = (A.flattenState initial).mem r o := by
  intro r o hmiss
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState, FlatAlloc.readFlat]
    cases hdecode : A.decode o with
    | none => rfl
    | some cell =>
      obtain ⟨r', o'⟩ := cell
      obtain ⟨_, heq, _⟩ := A.decode_sound hdecode
      apply congrArg A.trCell
      apply hf
      intro i ha hout hoff
      rcases hmiss with hbad | hmiss
      · exact hbad rfl
      · rcases hout with rfl | rfl
        · exact (hmiss i ha).1 (by rw [heq, hoff])
        · exact (hmiss i ha).2 (by rw [heq, hoff])
  · simp only [FlatAlloc.flattenState, if_neg hr]

private theorem in_regions (A : FlatAlloc)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (r : RegionName) (o : Nat)
    (hb : o < A.extent r) : r ∈ A.regions := by
  by_contra hnot
  rw [hcov r hnot] at hb
  exact Nat.not_lt_zero _ hb

section
variable (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat)

include hd hcov

theorem prefill_executes (commuted : Bool) (s : BlockState) (hN : 0 < N)
    (hu : s.undef = (fun _ _ => 0)) (hRegion : cosOut ≠ sinOut)
    (hInj : Function.Injective (fun i : Fin H =>
      (s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride))
    (hb : RealSafety.PrefillBounds A.extent cos sin metadata cosOut sinOut
      rowStride hiddenStride total H N B s hN) :
    ∃ final, exec (A.flattenKernel (RealSource.prefill cos sin metadata cosOut sinOut
        rowStride hiddenStride total H N B commuted)) (A.flattenState s) = some final ∧
      (∀ i : Fin H, s.pids 0 * B + s.pids 1 < total →
        final.readMem A.flat (A.addr cosOut
          ((s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride)) =
            s.readMem cos (RealSafety.prefillRead metadata rowStride hiddenStride N B s hN i.val) ∧
        final.readMem A.flat (A.addr sinOut
          ((s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride)) =
            s.readMem sin (RealSafety.prefillRead metadata rowStride hiddenStride N B s hN i.val)) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i : Fin H, s.pids 0 * B + s.pids 1 < total →
        o ≠ A.addr cosOut ((s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride) ∧
        o ≠ A.addr sinOut ((s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride)) →
        final.mem r o = (A.flattenState s).mem r o) := by
  let result := (RealExecution.prefillFinish cos sin metadata cosOut sinOut
    rowStride hiddenStride total H N B Bool.false s hN).val
  have hflat := A.exec_flatten hd hcov _ s
    (RealSafety.prefill_traceSafe _ _ _ _ _ _ _ _ _ _ _ _ commuted s hN hb)
    (RealSafety.prefill_flattenOk _ _ _ _ _ _ _ _ _ _ _ commuted) hu
  rw [RealCandidate.prefill_executes _ _ _ _ _ _ _ _ _ _ _ commuted s hN, Option.map_some] at hflat
  refine ⟨A.flattenState result, hflat, ?_, ?_⟩
  · intro i ha
    rw [A.flattenState_readMem hd result (in_regions A hcov _ _ (hb.cosOut i ha)) (hb.cosOut i ha),
      A.flattenState_readMem hd result (in_regions A hcov _ _ (hb.sinOut i ha)) (hb.sinOut i ha)]
    exact ⟨RealValues.prefill_cos _ _ _ _ _ _ _ _ _ _ _ s hN hRegion hInj i ha,
      RealValues.prefill_sin _ _ _ _ _ _ _ _ _ _ _ s hN hInj i ha⟩
  · exact flat_frame A cosOut sinOut
      (fun _ : Fin H => s.pids 0 * B + s.pids 1 < total)
      (fun i : Fin H => (s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride)
      s result (RealFrame.prefill_frame _ _ _ _ _ _ _ _ _ _ _ s hN)

theorem decode_executes (commuted : Bool) (s : BlockState)
    (hu : s.undef = (fun _ _ => 0)) (hRegion : cosOut ≠ sinOut)
    (hInj : Function.Injective (fun i : TileIndex [B,H] =>
      (s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride))
    (hb : RealSafety.DecodeBounds A.extent cos sin metadata cosOut sinOut
      rowStride hiddenStride H N B s) :
    ∃ final, exec (A.flattenKernel (RealSource.decode cos sin metadata cosOut sinOut
        rowStride hiddenStride H N B commuted)) (A.flattenState s) = some final ∧
      (∀ i : TileIndex [B,H], s.pids 0 * B + i.1.val < N →
        final.readMem A.flat (A.addr cosOut
          ((s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride)) =
            s.readMem cos (RealSafety.decodeRead metadata rowStride hiddenStride B s i.1.val i.2.1.val) ∧
        final.readMem A.flat (A.addr sinOut
          ((s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride)) =
            s.readMem sin (RealSafety.decodeRead metadata rowStride hiddenStride B s i.1.val i.2.1.val)) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i : TileIndex [B,H], s.pids 0 * B + i.1.val < N →
        o ≠ A.addr cosOut ((s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride) ∧
        o ≠ A.addr sinOut ((s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride)) →
        final.mem r o = (A.flattenState s).mem r o) := by
  let result := (RealExecution.decodeFinish cos sin metadata cosOut sinOut
    rowStride hiddenStride H N B Bool.false s).val
  have hflat := A.exec_flatten hd hcov _ s
    (RealSafety.decode_traceSafe _ _ _ _ _ _ _ _ _ _ _ commuted s hb)
    (RealSafety.decode_flattenOk _ _ _ _ _ _ _ _ _ _ commuted) hu
  rw [RealCandidate.decode_executes, Option.map_some] at hflat
  refine ⟨A.flattenState result, hflat, ?_, ?_⟩
  · intro i ha
    rw [A.flattenState_readMem hd result (in_regions A hcov _ _ (hb.cosOut i ha)) (hb.cosOut i ha),
      A.flattenState_readMem hd result (in_regions A hcov _ _ (hb.sinOut i ha)) (hb.sinOut i ha)]
    exact ⟨DecodeValues.cos_output _ _ _ _ _ _ _ _ _ _ s hRegion hInj i ha,
      DecodeValues.sin_output _ _ _ _ _ _ _ _ _ _ s hInj i ha⟩
  · exact flat_frame A cosOut sinOut (fun i : TileIndex [B,H] => s.pids 0 * B + i.1.val < N)
      (fun i : TileIndex [B,H] => (s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride)
      s result (RealFrame.decode_frame _ _ _ _ _ _ _ _ _ _ s)
end

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealFlat
