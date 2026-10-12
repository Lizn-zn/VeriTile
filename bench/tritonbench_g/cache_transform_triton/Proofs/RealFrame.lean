import bench.tritonbench_g.cache_transform_triton.Proofs.RealExecution

/-! Complete typed-cell framing outside the actual masked writes. This is an
equality of MemCell values, including their dtype tags, rather than merely an
equality of their real projections. No injectivity is needed for framing. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealFrame
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

private theorem scatter_frame {ι : Type} (region : RegionName)
    (off : ι → Nat) (value : ι → ℝ) (active : ι → Prop) [DecidablePred active] :
    ∀ (lanes : List ι) (s : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ i ∈ lanes, active i → off i ≠ o) →
      (lanes.foldl (fun t i => if active i then t.writeMem region (off i) (value i) else t) s).mem r o =
        s.mem r o
  | [], _, _, _, _ => rfl
  | i :: rest, s, r, o, h => by
      rw [List.foldl_cons]
      by_cases hi : active i
      · rw [if_pos hi,
          scatter_frame region off value active rest _ r o
            (fun hr j hj ha => h hr j (List.mem_cons_of_mem _ hj) ha),
          BlockState.writeMem_mem]
        rw [if_neg (fun bad => h bad.1 i List.mem_cons_self hi bad.2.symm)]
      · rw [if_neg hi]
        exact scatter_frame region off value active rest s r o
          (fun hr j hj ha => h hr j (List.mem_cons_of_mem _ hj) ha)

def PrefillMisses (cosOut sinOut : RegionName) (rowStride hiddenStride total H B : Nat)
    (s : BlockState) (r : RegionName) (o : Nat) : Prop :=
  ∀ i : Fin H, s.pids 0 * B + s.pids 1 < total →
    (r = cosOut ∨ r = sinOut) → (s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride ≠ o

def DecodeMisses (cosOut sinOut : RegionName) (rowStride hiddenStride H N B : Nat)
    (s : BlockState) (r : RegionName) (o : Nat) : Prop :=
  ∀ i : TileIndex [B,H], s.pids 0 * B + i.1.val < N →
    (r = cosOut ∨ r = sinOut) → (s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride ≠ o

theorem prefill_frame (cos sin : RegionName) (metadata : Region .int)
    (cosOut sinOut : RegionName) (rowStride hiddenStride total H N B : Nat)
    (s : BlockState) (hN : 0 < N) (r : RegionName) (o : Nat)
    (hmiss : PrefillMisses cosOut sinOut rowStride hiddenStride total H B s r o) :
    (RealExecution.prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B
      Bool.false s hN).val.mem r o = s.mem r o := by
  dsimp [RealExecution.prefillFinish]
  rw [scatter_frame sinOut _ _ _ _ _ r o
    (fun hr i _ ha => hmiss i.1 ha (Or.inr hr))]
  rw [scatter_frame cosOut _ _ _ _ _ r o
    (fun hr i _ ha => hmiss i.1 ha (Or.inl hr))]
  rfl

theorem decode_frame (cos sin : RegionName) (metadata : Region .int)
    (cosOut sinOut : RegionName) (rowStride hiddenStride H N B : Nat)
    (s : BlockState) (r : RegionName) (o : Nat)
    (hmiss : DecodeMisses cosOut sinOut rowStride hiddenStride H N B s r o) :
    (RealExecution.decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B
      Bool.false s).val.mem r o = s.mem r o := by
  dsimp [RealExecution.decodeFinish]
  simp only [Tile.remap, Broadcast.leftIndex, decide_eq_true_eq]
  rw [scatter_frame sinOut _ _ _ _ _ r o
    (fun hr i _ ha => hmiss i ha (Or.inr hr))]
  rw [scatter_frame cosOut _ _ _ _ _ r o
    (fun hr i _ ha => hmiss i ha (Or.inl hr))]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealFrame
