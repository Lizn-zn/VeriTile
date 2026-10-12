import bench.tritonbench_g.cache_transform_triton.Proofs.RealFlat
import bench.tritonbench_g.cache_transform_triton.Proofs.Layout

/-! Full BLOCK_SIZE × HIDDEN_DIM decoding IO over independent initial arrays.
Rows are signed indices already adjusted by the host's lengths-minus-one.
Active rows must be nonnegative, so the address conversion never clips a
negative index into a valid cache access. Inactive metadata is unrestricted. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.DecodeIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

structure Inputs (H N : Nat) where
  rows : Fin N → Int
  cos : Nat → Fin H → ℝ
  sin : Nat → Fin H → ℝ

def expected {H N : Nat} (xs : Inputs H N) (seq : Fin N) (i : Fin H) : ℝ × ℝ :=
  (xs.cos (xs.rows seq).toNat i, xs.sin (xs.rows seq).toNat i)

def token (p : Layout) (pid : Nat) (i : Fin p.B) : Nat := pid * p.B + i.val

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  readMetadata : Nat → Nat
  read : Nat → Fin p.H → Nat
  write : Nat → Fin p.H → Nat

structure IO.Bounds {p : Layout} (io : IO p) (bounds : RegionBounds)
    (N pid : Nat) (xs : Inputs p.H N) : Prop where
  metadata : ∀ i : Fin p.B, token p pid i < N → io.readMetadata (token p pid i) < bounds p.metadata.cast
  cos : ∀ i : TileIndex [p.B,p.H], ∀ ha : token p pid i.1 < N,
    io.read (xs.rows ⟨token p pid i.1, ha⟩).toNat i.2.1 < bounds p.cos
  sin : ∀ i : TileIndex [p.B,p.H], ∀ ha : token p pid i.1 < N,
    io.read (xs.rows ⟨token p pid i.1, ha⟩).toNat i.2.1 < bounds p.sin
  cosOut : ∀ i : TileIndex [p.B,p.H], token p pid i.1 < N →
    io.write (token p pid i.1) i.2.1 < bounds p.cosOut
  sinOut : ∀ i : TileIndex [p.B,p.H], token p pid i.1 < N →
    io.write (token p pid i.1) i.2.1 < bounds p.sinOut

structure IO.Pins {p : Layout} (io : IO p) (s : BlockState) (N : Nat) (xs : Inputs p.H N) : Prop where
  metadata : ∀ i : Fin p.B, ∀ ha : token p (s.pids 0) i < N,
    s.readMemValue .int p.metadata.cast (io.readMetadata (token p (s.pids 0) i)) =
      xs.rows ⟨token p (s.pids 0) i, ha⟩
  cos : ∀ i : TileIndex [p.B,p.H], ∀ ha : token p (s.pids 0) i.1 < N,
    s.readMem p.cos (io.read (xs.rows ⟨token p (s.pids 0) i.1, ha⟩).toNat i.2.1) =
      (expected xs ⟨token p (s.pids 0) i.1, ha⟩ i.2.1).1
  sin : ∀ i : TileIndex [p.B,p.H], ∀ ha : token p (s.pids 0) i.1 < N,
    s.readMem p.sin (io.read (xs.rows ⟨token p (s.pids 0) i.1, ha⟩).toNat i.2.1) =
      (expected xs ⟨token p (s.pids 0) i.1, ha⟩ i.2.1).2

/-- Successful flat execution, both active output tiles and preservation of
all other typed cells. Metadata loads remain active even if H = 0. The current
flat bridge requires zero unspecified values; no output equation is assumed. -/
def IO.Implements {p : Layout} (io : IO p) (N : Nat)
    (f : Inputs p.H N → Fin N → Fin p.H → ℝ × ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint →
    A.regions = [p.cos, p.sin, p.metadata.cast, p.cosOut, p.sinOut] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (xs : Inputs p.H N),
    (∀ i : Fin p.B, ∀ ha : token p (s.pids 0) i < N, 0 ≤ xs.rows ⟨token p (s.pids 0) i, ha⟩) →
    s.undef = (fun _ _ => 0) → io.Bounds A.extent N (s.pids 0) xs → io.Pins s N xs →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ i : TileIndex [p.B,p.H], ∀ ha : token p (s.pids 0) i.1 < N,
        final.readMem A.flat (A.addr p.cosOut (io.write (token p (s.pids 0) i.1) i.2.1)) =
          (f xs ⟨token p (s.pids 0) i.1, ha⟩ i.2.1).1 ∧
        final.readMem A.flat (A.addr p.sinOut (io.write (token p (s.pids 0) i.1) i.2.1)) =
          (f xs ⟨token p (s.pids 0) i.1, ha⟩ i.2.1).2) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i : TileIndex [p.B,p.H], token p (s.pids 0) i.1 < N →
        o ≠ A.addr p.cosOut (io.write (token p (s.pids 0) i.1) i.2.1) ∧
        o ≠ A.addr p.sinOut (io.write (token p (s.pids 0) i.1) i.2.1)) →
        final.mem r o = (A.flattenState s).mem r o)

scoped notation:25 io " ⊨[" N "] " f => IO.Implements io N f

def sourceIO (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (N : Nat) : IO p where
  kernel := (p.decode cosType sinType commuted N).eraseDType
  projection := by
    simp only [Layout.decode, ComputeKernel.toAlgKernel, RealSource.decode_projection]
  readMetadata := id
  read := fun r i => r * p.rowStride + i.val * p.hiddenStride
  write := fun seq i => seq * p.rowStride + i.val * p.hiddenStride

theorem read_pins (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (N : Nat) (s : BlockState) (xs : Inputs p.H N)
    (hp : (sourceIO p cosType sinType commuted N).Pins s N xs)
    (i : TileIndex [p.B,p.H]) (ha : token p (s.pids 0) i.1 < N)
    (hr : 0 ≤ xs.rows ⟨token p (s.pids 0) i.1, ha⟩) :
    RealSafety.decodeRead p.metadata p.rowStride p.hiddenStride p.B s i.1.val i.2.1.val =
      (sourceIO p cosType sinType commuted N).read
        (xs.rows ⟨token p (s.pids 0) i.1, ha⟩).toNat i.2.1 := by
  have hm := hp.metadata i.1 ha
  change s.readMemValue .int p.metadata.cast (s.pids 0 * p.B + i.1.val) = _ at hm
  simp only [RealSafety.decodeRead, hm, sourceIO]
  rw [Int.toNat_add (mul_nonneg hr (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_mul hr (Int.natCast_nonneg _), Int.toNat_natCast, Int.toNat_natCast]

theorem implements (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (N : Nat) (hRegion : p.cosOut ≠ p.sinOut)
    (hLayout : Function.Injective (fun i : TileIndex [p.B,p.H] =>
      i.1.val * p.rowStride + i.2.1.val * p.hiddenStride)) :
    sourceIO p cosType sinType commuted N ⊨[N] expected := by
  intro A hd _ hcov s xs hr hu hb hp
  have hRead := fun i ha => read_pins p cosType sinType commuted N s xs hp i ha (hr i.1 ha)
  have hbounds : RealSafety.DecodeBounds A.extent p.cos p.sin p.metadata p.cosOut p.sinOut
      p.rowStride p.hiddenStride p.H N p.B s := by
    refine ⟨hb.metadata, ?_, ?_, hb.cosOut, hb.sinOut⟩
    · intro i ha; rw [hRead i ha]; exact hb.cos i ha
    · intro i ha; rw [hRead i ha]; exact hb.sin i ha
  have hInj : Function.Injective (fun i : TileIndex [p.B,p.H] =>
      (s.pids 0 * p.B + i.1.val) * p.rowStride + i.2.1.val * p.hiddenStride) := by
    intro i j h
    apply hLayout
    simp only [Nat.add_mul, Nat.add_assoc] at h
    exact Nat.add_left_cancel h
  obtain ⟨final, he, hv, hf⟩ := RealFlat.decode_executes A hd hcov p.cos p.sin p.metadata p.cosOut p.sinOut
    p.rowStride p.hiddenStride p.H N p.B commuted s hu hRegion hInj hbounds
  refine ⟨final, ?_, ?_, hf⟩
  · simpa only [sourceIO, Layout.decode, ComputeKernel.toAlgKernel, RealSource.decode_projection] using he
  · intro i ha
    obtain ⟨hc, hs⟩ := hv i ha
    rw [hRead i ha] at hc hs
    exact ⟨hc.trans (hp.cos i ha), hs.trans (hp.sin i ha)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.DecodeIO
