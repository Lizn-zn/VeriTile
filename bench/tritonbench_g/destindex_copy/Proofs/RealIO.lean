import bench.tritonbench_g.destindex_copy.Proofs.Projection
import bench.tritonbench_g.destindex_copy.Proofs.Layout
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.RealIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

structure Inputs (N R : Nat) where
  row : Int
  nope : Fin N → ℝ
  rope : Fin R → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  readMetadata : Nat → Nat
  readNope : Nat → Fin p.BLOCK_DMODEL_NOPE → Nat
  readRope : Nat → Fin p.BLOCK_DMODEL_ROPE → Nat
  writeNope : Int → Fin p.BLOCK_DMODEL_NOPE → Nat
  writeRope : Int → Fin p.BLOCK_DMODEL_ROPE → Nat

structure IO.Bounds {p : Layout} (io : IO p) (bounds : RegionBounds) (pid : Nat)
    (xs : Inputs p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE) : Prop where
  metadata : io.readMetadata pid < bounds p.Dest_loc.cast
  nope : ∀ j, io.readNope pid j < bounds p.KV_nope
  rope : ∀ j, io.readRope pid j < bounds p.KV_rope
  outNope : ∀ j, io.writeNope xs.row j < bounds p.O_nope
  outRope : ∀ j, io.writeRope xs.row j < bounds p.O_rope

structure IO.Pins {p : Layout} (io : IO p) (s : BlockState)
    (xs : Inputs p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE) : Prop where
  metadata : s.readMemValue .int p.Dest_loc.cast (io.readMetadata (s.pids 0)) = xs.row
  nope : ∀ j, s.readMem p.KV_nope (io.readNope (s.pids 0) j) = xs.nope j
  rope : ∀ j, s.readMem p.KV_rope (io.readRope (s.pids 0) j) = xs.rope j

/-- Per-program successful finite flat IO, both unmasked output tiles and all
other tagged cells. Destination rows must be nonnegative: the signed source
index is not clipped into a valid row. Every padding lane needs real storage.
All source and metadata reads precede stores, so input/output aliasing is
allowed; the two output regions must be distinct in the correctness theorem. -/
def IO.Implements {p : Layout} (io : IO p)
    (f : Inputs p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE →
      (Fin p.BLOCK_DMODEL_NOPE → ℝ) × (Fin p.BLOCK_DMODEL_ROPE → ℝ)) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint →
    A.regions = [p.KV_nope,p.KV_rope,p.Dest_loc.cast,p.O_nope,p.O_rope] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (xs : Inputs p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE),
    0 ≤ xs.row → s.undef = (fun _ _ => 0) →
    io.Bounds A.extent (s.pids 0) xs → io.Pins s xs →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ j, final.readMem A.flat (A.addr p.O_nope (io.writeNope xs.row j)) = (f xs).1 j) ∧
      (∀ j, final.readMem A.flat (A.addr p.O_rope (io.writeRope xs.row j)) = (f xs).2 j) ∧
      (∀ r o, (r ≠ A.flat ∨
        (∀ j, o ≠ A.addr p.O_nope (io.writeNope xs.row j)) ∧
        (∀ j, o ≠ A.addr p.O_rope (io.writeRope xs.row j))) →
        final.mem r o = (A.flattenState s).mem r o)

scoped infix:25 " ⊨ " => IO.Implements

def referenceIO (p : Layout) : IO p where
  kernel := RealReference.fwd_kernel_destindex_copy_kv p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE
  projection := by rfl
  readMetadata := id
  readNope := fun pid j => pid*p.stride_kv_nope_bs + p.stride_kv_nope_d*j.val
  readRope := fun pid j => pid*p.stride_kv_rope_bs + p.stride_kv_rope_d*j.val
  writeNope := fun row j => row.toNat*p.stride_o_nope_bs + p.stride_o_nope_d*j.val
  writeRope := fun row j => row.toNat*p.stride_o_rope_bs + p.stride_o_rope_d*j.val

theorem address_nonnegative (row : Int) (h : 0 ≤ row) (bs ds j : Nat) :
    RealReference.writeAddress row bs ds j = row.toNat * bs + ds*j := by
  unfold RealReference.writeAddress
  rw [Int.toNat_add (mul_nonneg h (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_mul h (Int.natCast_nonneg _), Int.toNat_natCast, Int.toNat_natCast]

theorem address_injective {B : Nat} (row : Int) (hr : 0 ≤ row) (bs ds : Nat)
    (h : Function.Injective (fun j : Fin B => ds*j.val)) :
    Function.Injective (fun j : TileIndex [1,B] => RealReference.writeAddress row bs ds j.2.1.val) := by
  intro a b hab
  simp only [address_nonnegative row hr] at hab
  have hj : a.2.1 = b.2.1 := h (Nat.add_left_cancel hab)
  obtain ⟨a1,a2,a3⟩ := a
  obtain ⟨b1,b2,b3⟩ := b
  simp only at hj
  subst hj
  rw [Subsingleton.elim a1 b1,Subsingleton.elim a3 b3]

private theorem in_regions (A : FlatAlloc)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (r : RegionName) (o : Nat)
    (hb : o < A.extent r) : r ∈ A.regions := by
  by_contra hn
  rw [hc r hn] at hb
  exact Nat.not_lt_zero _ hb

private theorem flat_frame {N R : Nat} (A : FlatAlloc) (outNope outRope : RegionName)
    (nope : Fin N → Nat) (rope : Fin R → Nat) (s t : BlockState)
    (hf : ∀ r o, (∀ j, ¬(outNope = r ∧ nope j = o)) →
      (∀ j, ¬(outRope = r ∧ rope j = o)) → t.mem r o = s.mem r o) :
    ∀ r o, (r ≠ A.flat ∨
      (∀ j, o ≠ A.addr outNope (nope j)) ∧ (∀ j, o ≠ A.addr outRope (rope j))) →
      (A.flattenState t).mem r o = (A.flattenState s).mem r o := by
  intro r o hm
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState,FlatAlloc.readFlat]
    cases hd : A.decode o with
    | none => rfl
    | some cell =>
      obtain ⟨r',o'⟩ := cell
      obtain ⟨_,he,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      · intro j h
        rcases hm with hn | hm
        · exact hn rfl
        · exact hm.1 j (by rw [he,←h.1,←h.2])
      · intro j h
        rcases hm with hn | hm
        · exact hn rfl
        · exact hm.2 j (by rw [he,←h.1,←h.2])
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem reference_correct (p : Layout) (hOut : p.O_nope ≠ p.O_rope)
    (hNope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_NOPE => p.stride_o_nope_d*j.val))
    (hRope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_ROPE => p.stride_o_rope_d*j.val)) :
    referenceIO p ⊨ fun xs => (xs.nope,xs.rope) := by
  intro A hd _ hc s xs hr hu hb hp
  have hm := hp.metadata
  change s.readMemValue .int p.Dest_loc.cast (s.pids 0) = xs.row at hm
  have address (bs ds j : Nat) :
      RealReference.writeAddress (s.readMemValue .int p.Dest_loc.cast (s.pids 0)) bs ds j =
        xs.row.toNat*bs + ds*j := by rw [hm,address_nonnegative xs.row hr]
  have hs := RealReference.fwd_kernel_destindex_copy_kv_traceSafe p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE
    A.extent s hb.metadata hb.nope hb.rope
    (by intro j; rw [address]; exact hb.outNope j)
    (by intro j; rw [address]; exact hb.outRope j)
  obtain ⟨t,he⟩ := RealReference.fwd_kernel_destindex_copy_kv_exec_isSome p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE s
  have hflat := A.exec_flatten hd hc _ s hs
    (RealReference.fwd_kernel_destindex_copy_kv_flattenOk p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE) hu
  rw [he,Option.map_some] at hflat
  have hbase : RealReference.destBase s p.Dest_loc.cast = xs.row := by
    simpa [RealReference.destBase,BlockState.pid_eq] using hm
  have hiNope : Function.Injective (fun j : TileIndex [1,p.BLOCK_DMODEL_NOPE] =>
      RealReference.outNopeAddr s p.Dest_loc.cast p.stride_o_nope_bs p.stride_o_nope_d j) := by
    simp only [RealReference.outNopeAddr,hbase,RealReference.dimNope]
    exact address_injective xs.row hr _ _ hNope
  have hiRope : Function.Injective (fun j : TileIndex [1,p.BLOCK_DMODEL_ROPE] =>
      RealReference.outRopeAddr s p.Dest_loc.cast p.stride_o_rope_bs p.stride_o_rope_d j) := by
    simp only [RealReference.outRopeAddr,hbase,RealReference.dimRope]
    exact address_injective xs.row hr _ _ hRope
  refine ⟨A.flattenState t,hflat,?_,?_,?_⟩
  · intro j
    rw [A.flattenState_readMem hd t (in_regions A hc _ _ (hb.outNope j)) (hb.outNope j)]
    have hv := RealReference.fwd_kernel_destindex_copy_kv_nope_correct_of_exec p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE
      s t hOut hiNope he ((0:Fin 1),j,PUnit.unit)
    simpa only [RealReference.outNopeAddr,hbase,RealReference.dimNope,address_nonnegative xs.row hr,
      RealReference.sourceNopeAddr,BlockState.pid_eq] using hv.trans (hp.nope j)
  · intro j
    rw [A.flattenState_readMem hd t (in_regions A hc _ _ (hb.outRope j)) (hb.outRope j)]
    have hv := RealReference.fwd_kernel_destindex_copy_kv_rope_correct_of_exec p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE
      s t hiRope he ((0:Fin 1),j,PUnit.unit)
    simpa only [RealReference.outRopeAddr,hbase,RealReference.dimRope,address_nonnegative xs.row hr,
      RealReference.sourceRopeAddr,BlockState.pid_eq] using hv.trans (hp.rope j)
  · apply flat_frame A p.O_nope p.O_rope _ _ s t
    intro r o hn hr'
    apply RealReference.fwd_kernel_destindex_copy_kv_frame p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE s t he r o
    · intro j
      rw [address]
      exact hn j.2.1
    · intro j
      rw [address]
      exact hr' j.2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.RealIO
