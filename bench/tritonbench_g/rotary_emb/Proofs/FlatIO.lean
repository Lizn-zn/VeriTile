import bench.tritonbench_g.rotary_emb.Proofs.TypedIO
import bench.tritonbench_g.softmax_flaggems.Proofs.RealFlatFrame
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- Every operation in the actual source admits the region-to-flat bridge. -/
theorem math_flattenOk (p : Parameters) (Q K Cos Sin : RegionName) :
    (mathKernel p Q K Cos Sin).FlattenOk := by
  simp [Kernel.FlattenOk, mathKernel, rotary_kernel_surface,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem typed_flattenOk (q k cos sin : FullSource.Storage) (v : Bool)
    (p : Parameters) (Q K Cos Sin : RegionName) :
    (FullSource.kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim
      p.kBatch p.kHead p.kDim p.cosBatch p.cosDim p.sinBatch p.sinDim p.length
      p.headsQ p.headsK p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  rw [FullSource.kernel_math_body]
  cases v
  · rw [math_original]
    exact math_flattenOk p Q K Cos Sin
  · simp [mathVariant, mathKernel, rotary_kernel_surface, headStatement,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The existing flatten-state bridge requires zero undefined values explicitly;
region IO and region safety impose no such restriction. Both outputs share the
same final flat state and the frame excludes exactly the actual write footprint. -/
theorem typed_flat_io (q k cos sin : FullSource.Storage) (v : Bool)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState)
    (hsep : RegionsSeparate Q K Cos Sin) (hGeometry : p.Geometry s)
    (hb : p.Bounds s A.extent Q K Cos Sin)
    (hQr : Q ∈ A.regions) (hKr : K ∈ A.regions)
    (hu : s.undef = fun _ _ ↦ 0) :
    ∃ t, exec (A.flattenKernel
      (FullSource.kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim
        p.kBatch p.kHead p.kDim p.cosBatch p.cosDim p.sinBatch p.sinDim p.length
        p.headsQ p.headsK p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel)
      (A.flattenState s) = some t ∧
      (∀ odd i, p.activeQ s i →
        t.readMem A.flat (A.addr Q (p.qAddress s odd i)) = p.expectedQ s Q Cos Sin odd i) ∧
      (∀ odd i, p.activeK s i →
        t.readMem A.flat (A.addr K (p.kAddress s odd i)) = p.expectedK s K Cos Sin odd i) ∧
      BlockState.WriteWithin
        (SoftmaxFlaggems.RealFlatFrame.footprint A (p.footprint s Q K)) (A.flattenState s) t := by
  obtain ⟨t, ht, hq, hk, hf⟩ := typed_io q k cos sin v p Q K Cos Sin s hsep hGeometry
  refine ⟨A.flattenState t, ?_, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (typed_safe q k cos sin v A.extent p Q K Cos Sin s hb)
      (typed_flattenOk q k cos sin v p Q K Cos Sin) hu, ht, Option.map_some]
  · intro odd i hi
    rw [A.flattenState_readMem hd t hQr (hb.1 odd i hi)]
    exact hq odd i hi
  · intro odd i hi
    rw [A.flattenState_readMem hd t hKr (hb.2.1 odd i hi)]
    exact hk odd i hi
  · exact SoftmaxFlaggems.RealFlatFrame.transport A _ s t hf

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
