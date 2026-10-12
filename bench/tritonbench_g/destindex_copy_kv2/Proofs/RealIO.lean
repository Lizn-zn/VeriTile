import bench.tritonbench_g.destindex_copy_kv2.Proofs.Projection
import bench.tritonbench_g.destindex_copy_kv2.Proofs.Layout
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2.RealIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

structure Inputs (H D : Nat) where
  row : Int
  values : TileIndex [H,D] → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  readMetadata : Nat → Nat
  read : Nat → TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] → Nat
  write : Int → TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] → Nat
  active : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] → Prop

structure IO.Bounds {p : Layout} (io : IO p) (bounds : RegionBounds) (pid : Nat)
    (xs : Inputs p.BLOCK_HEAD p.BLOCK_DMODEL) : Prop where
  metadata : io.readMetadata pid < bounds p.Dest_loc.cast
  input : ∀ j, io.active j → io.read pid j < bounds p.K
  output : ∀ j, io.active j → io.write xs.row j < bounds p.Out

structure IO.Pins {p : Layout} (io : IO p) (s : BlockState)
    (xs : Inputs p.BLOCK_HEAD p.BLOCK_DMODEL) : Prop where
  metadata : s.readMemValue .int p.Dest_loc.cast (io.readMetadata (s.pids 0)) = xs.row
  input : ∀ j, io.active j → s.readMem p.K (io.read (s.pids 0) j) = xs.values j

/-- Successful finite flat IO and all tagged cells outside active stores.
The metadata read is unmasked even when every payload lane is inactive.
Nonnegative destination metadata prevents clipping at the pointer boundary. -/
def IO.Implements {p : Layout} (io : IO p)
    (f : Inputs p.BLOCK_HEAD p.BLOCK_DMODEL → TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] → ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → A.regions = [p.K,p.Dest_loc.cast,p.Out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ (s : BlockState) (xs : Inputs p.BLOCK_HEAD p.BLOCK_DMODEL),
    0 ≤ xs.row → s.undef = (fun _ _ => 0) →
    io.Bounds A.extent (s.pids 0) xs → io.Pins s xs →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ j, io.active j → final.readMem A.flat (A.addr p.Out (io.write xs.row j)) = f xs j) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ j, io.active j → o ≠ A.addr p.Out (io.write xs.row j)) →
        final.mem r o = (A.flattenState s).mem r o)

scoped infix:25 " ⊨ " => IO.Implements

def referenceIO (p : Layout) : IO p where
  kernel := RealReference.fwd_kernel_destindex_copy_kv p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD
  projection := by rfl
  readMetadata := id
  read := fun pid j => pid*p.stride_k_bs + p.stride_k_h*j.1.val + p.stride_k_d*j.2.1.val
  write := fun row j => row.toNat*p.stride_o_bs + p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val
  active := fun j => j.1.val<p.head_num

theorem address_nonnegative (row : Int) (hr : 0 ≤ row) (bs hs ds h d : Nat) :
    RealReference.writeAddress row bs hs ds h d = row.toNat*bs + hs*h + ds*d := by
  unfold RealReference.writeAddress
  rw [Int.toNat_add (add_nonneg (mul_nonneg hr (Int.natCast_nonneg _)) (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_add (mul_nonneg hr (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_mul hr (Int.natCast_nonneg _),Int.toNat_natCast,Int.toNat_natCast,Int.toNat_natCast]

private theorem in_regions (A : FlatAlloc)
    (hc : ∀ r, r ∉ A.regions → A.extent r=0) (r : RegionName) (o : Nat)
    (hb : o < A.extent r) : r ∈ A.regions := by
  by_contra hn
  rw [hc r hn] at hb
  exact Nat.not_lt_zero _ hb

private theorem flat_frame {ι : Type} (A : FlatAlloc) (out : RegionName)
    (active : ι → Prop) (off : ι → Nat) (s t : BlockState)
    (hf : ∀ r o, (∀ j, active j → ¬(out=r ∧ off j=o)) → t.mem r o=s.mem r o) :
    ∀ r o, (r≠A.flat ∨ ∀ j, active j → o≠A.addr out (off j)) →
      (A.flattenState t).mem r o = (A.flattenState s).mem r o := by
  intro r o hm
  by_cases hr : r=A.flat
  · subst r
    simp only [FlatAlloc.flattenState,FlatAlloc.readFlat]
    cases hd : A.decode o with
    | none => rfl
    | some cell =>
      obtain ⟨r',o'⟩ := cell
      obtain ⟨_,he,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro j ha h
      rcases hm with hn | hm
      · exact hn rfl
      · exact hm j ha (by rw [he,←h.1,←h.2])
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem reference_correct (p : Layout)
    (hLayout : Function.Injective (fun j : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] =>
      p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val)) :
    referenceIO p ⊨ fun xs => xs.values := by
  intro A hd _ hc s xs hr hu hb hp
  have hm := hp.metadata
  change s.readMemValue .int p.Dest_loc.cast (s.pids 0) = xs.row at hm
  have address (h d : Nat) :
      RealReference.writeAddress (s.readMemValue .int p.Dest_loc.cast (s.pids 0))
        p.stride_o_bs p.stride_o_h p.stride_o_d h d =
        xs.row.toNat*p.stride_o_bs + p.stride_o_h*h + p.stride_o_d*d := by
    rw [hm,address_nonnegative xs.row hr]
  have hs := RealReference.fwd_kernel_destindex_copy_kv_traceSafe p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD
    A.extent s hb.metadata (fun j hh => hb.input j hh)
    (by intro j hh; rw [address]; exact hb.output j hh)
  obtain ⟨t,he⟩ := RealReference.fwd_kernel_destindex_copy_kv_exec_isSome p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD s
  have hflat := A.exec_flatten hd hc _ s hs
    (RealReference.fwd_kernel_destindex_copy_kv_flattenOk p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD) hu
  rw [he,Option.map_some] at hflat
  have hbase : RealReference.destBase s p.Dest_loc.cast = xs.row := by
    simpa [RealReference.destBase,BlockState.pid_eq] using hm
  have hi : Function.Injective (fun j : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] =>
      RealReference.outAddr s p.Dest_loc.cast p.stride_o_bs p.stride_o_h p.stride_o_d j) := by
    intro a b h
    apply hLayout
    simp only [RealReference.outAddr,hbase,RealReference.headIndex,RealReference.dimIndex,
      address_nonnegative xs.row hr,Nat.add_assoc] at h
    exact Nat.add_left_cancel h
  refine ⟨A.flattenState t,hflat,?_,?_⟩
  · intro j hj
    rw [A.flattenState_readMem hd t (in_regions A hc _ _ (hb.output j hj)) (hb.output j hj)]
    have hv := RealReference.fwd_kernel_destindex_copy_kv_correct_of_exec p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD s hi t he j
    have ha : RealReference.active p.head_num j := hj
    rw [if_pos ha] at hv
    simpa only [RealReference.outAddr,hbase,RealReference.headIndex,RealReference.dimIndex,
      address_nonnegative xs.row hr,RealReference.sourceAddr,BlockState.pid_eq] using hv.trans (hp.input j hj)
  · apply flat_frame A p.Out _ _ s t
    intro r o hn
    apply RealReference.fwd_kernel_destindex_copy_kv_frame p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD s t he r o
    intro j hh
    rw [address]
    exact hn j hh

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopyKv2.RealIO
