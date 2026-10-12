import bench.tritonbench_g.cache_transform_triton.Proofs.RealFlat
import bench.tritonbench_g.cache_transform_triton.Proofs.Layout

/-! Independent initial arrays and a complete two-output prefill IO contract.
Every metadata entry is read even when the token is inactive. Payload pins
refer only to initial cache values. Output values are solely conclusions. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.PrefillIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

structure Inputs (H N : Nat) where
  ends : Fin N → Int
  cos : Nat → Fin H → ℝ
  sin : Nat → Fin H → ℝ

def row {H N : Nat} (hN : 0 < N) (token : Nat) (xs : Inputs H N) : Nat :=
  (RealMath.relativeRow hN token xs.ends).toNat

def expected {H N : Nat} (hN : 0 < N) (token : Nat) (xs : Inputs H N) (i : Fin H) : ℝ × ℝ :=
  (xs.cos (row hN token xs) i, xs.sin (row hN token xs) i)

structure IO (p : Layout) (N : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  readMetadata : Fin N → Nat
  read : Nat → Fin p.H → Nat
  write : Nat → Fin p.H → Nat

structure IO.Bounds {p : Layout} {N : Nat} (io : IO p N) (bounds : RegionBounds)
    (total token selected : Nat) : Prop where
  metadata : ∀ i, io.readMetadata i < bounds p.metadata.cast
  cos : ∀ i, token < total → io.read selected i < bounds p.cos
  sin : ∀ i, token < total → io.read selected i < bounds p.sin
  cosOut : ∀ i, token < total → io.write token i < bounds p.cosOut
  sinOut : ∀ i, token < total → io.write token i < bounds p.sinOut

structure IO.Pins {p : Layout} {N : Nat} (io : IO p N) (s : BlockState)
    (total token : Nat) (hN : 0 < N) (xs : Inputs p.H N) : Prop where
  metadata : ∀ j, s.readMemValue .int p.metadata.cast (io.readMetadata j) = xs.ends j
  cos : ∀ i, token < total → s.readMem p.cos (io.read (row hN token xs) i) = (expected hN token xs i).1
  sin : ∀ i, token < total → s.readMem p.sin (io.read (row hN token xs) i) = (expected hN token xs i).2

/-- One complete launch program in flat memory. Metadata is unmasked; both
stores share the token predicate. The full-cell frame includes inactive lanes
and unrelated allocations. `undef = 0` is the current flat-bridge premise;
the typed FP execution theorem itself permits arbitrary unspecified values. -/
def IO.Implements {p : Layout} {N : Nat} (io : IO p N) (total : Nat) (hN : 0 < N)
    (f : Nat → Inputs p.H N → Fin p.H → ℝ × ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint →
    A.regions = [p.cos, p.sin, p.metadata.cast, p.cosOut, p.sinOut] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (xs : Inputs p.H N),
    s.undef = (fun _ _ => 0) →
    io.Bounds A.extent total (s.pids 0 * p.B + s.pids 1) (row hN (s.pids 0 * p.B + s.pids 1) xs) →
    io.Pins s total (s.pids 0 * p.B + s.pids 1) hN xs →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ i, s.pids 0 * p.B + s.pids 1 < total →
        final.readMem A.flat (A.addr p.cosOut (io.write (s.pids 0 * p.B + s.pids 1) i)) =
          (f (s.pids 0 * p.B + s.pids 1) xs i).1 ∧
        final.readMem A.flat (A.addr p.sinOut (io.write (s.pids 0 * p.B + s.pids 1) i)) =
          (f (s.pids 0 * p.B + s.pids 1) xs i).2) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i, s.pids 0 * p.B + s.pids 1 < total →
        o ≠ A.addr p.cosOut (io.write (s.pids 0 * p.B + s.pids 1) i) ∧
        o ≠ A.addr p.sinOut (io.write (s.pids 0 * p.B + s.pids 1) i)) →
        final.mem r o = (A.flattenState s).mem r o)

scoped notation:25 io " ⊨[" total "," hN "] " f => IO.Implements io total hN f

def sourceIO (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (total N : Nat) : IO p N where
  kernel := (p.prefill cosType sinType commuted total N).eraseDType
  projection := by
    simp only [Layout.prefill, ComputeKernel.toAlgKernel, RealSource.prefill_projection]
  readMetadata := fun i => i.val
  read := fun r i => r * p.rowStride + i.val * p.hiddenStride
  write := fun token i => token * p.rowStride + i.val * p.hiddenStride

theorem read_pins (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (total N : Nat) (s : BlockState) (hN : 0 < N) (xs : Inputs p.H N)
    (hp : (sourceIO p cosType sinType commuted total N).Pins s total
      (s.pids 0 * p.B + s.pids 1) hN xs) (i : Fin p.H) :
    RealSafety.prefillRead p.metadata p.rowStride p.hiddenStride N p.B s hN i.val =
      (sourceIO p cosType sinType commuted total N).read (row hN (s.pids 0 * p.B + s.pids 1) xs) i := by
  have hm : (fun j : Fin N => s.readMemValue .int p.metadata.cast j.val) = xs.ends :=
    funext hp.metadata
  simp only [RealSafety.prefillRead, hm, sourceIO, row]
  have hr := RealMath.relativeRow_nonnegative hN (s.pids 0 * p.B + s.pids 1) xs.ends
  rw [Int.toNat_add (mul_nonneg hr (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_mul hr (Int.natCast_nonneg _), Int.toNat_natCast, Int.toNat_natCast]

theorem implements (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (total N : Nat) (hN : 0 < N)
    (hRegion : p.cosOut ≠ p.sinOut) (hStride : 0 < p.hiddenStride) :
    sourceIO p cosType sinType commuted total N ⊨[total,hN] expected hN := by
  intro A hd _ hcov s xs hu hb hp
  have hRead := read_pins p cosType sinType commuted total N s hN xs hp
  have hbounds : RealSafety.PrefillBounds A.extent p.cos p.sin p.metadata p.cosOut p.sinOut
      p.rowStride p.hiddenStride total p.H N p.B s hN := by
    refine ⟨hb.metadata, ?_, ?_, hb.cosOut, hb.sinOut⟩
    · intro i ha; rw [hRead i]; exact hb.cos i ha
    · intro i ha; rw [hRead i]; exact hb.sin i ha
  have hInj : Function.Injective (fun i : Fin p.H =>
      (s.pids 0 * p.B + s.pids 1) * p.rowStride + i.val * p.hiddenStride) := by
    intro i j h
    apply Fin.ext
    exact Nat.eq_of_mul_eq_mul_right hStride (Nat.add_left_cancel h)
  obtain ⟨final, he, hv, hf⟩ := RealFlat.prefill_executes A hd hcov p.cos p.sin p.metadata p.cosOut p.sinOut
    p.rowStride p.hiddenStride total p.H N p.B commuted s hN hu hRegion hInj hbounds
  refine ⟨final, ?_, ?_, hf⟩
  · simpa only [sourceIO, Layout.prefill, ComputeKernel.toAlgKernel, RealSource.prefill_projection] using he
  · intro i ha
    obtain ⟨hc, hs⟩ := hv i ha
    rw [hRead i] at hc hs
    exact ⟨hc.trans (hp.cos i ha), hs.trans (hp.sin i ha)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.PrefillIO
