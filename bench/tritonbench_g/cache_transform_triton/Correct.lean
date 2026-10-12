import bench.tritonbench_g.cache_transform_triton.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.PrefillIO
import bench.tritonbench_g.cache_transform_triton.Proofs.DecodeIO
import VeriTile.Meta.StatementAudit

/-! Real correctness of both complete cache gathers and their index-commuted
candidates. Cosine and sine independently use fp32/fp16/bf16 storage before
precision erasure. The formulas below use independent initial arrays.

Each contract proves successful execution in flat memory, both active output
windows and preservation of every other typed cell. All access bounds are
explicit: prefill reads N metadata entries without a mask; decode loads a
full B × H tile with its original row mask. Active decode indices are
nonnegative and already include the host's lengths-minus-one adjustment.

The source reuses cosine-cache strides for sine and both fresh outputs. The
contracts retain those actual strides and require collision-free stores.
The host's unpadded next_power_of_2 metadata allocation is not repaired.
The current flat bridge requires zero unspecified values. Address arithmetic
is mathematical, without fixed-width overflow or launch-wide race claims. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.Correctness
open VeriTile.Triton
noncomputable section

abbrev Parameters := Layout
abbrev PrefillInputs := PrefillIO.Inputs
abbrev DecodeInputs := DecodeIO.Inputs

def prefillOriginalIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (total N : Nat) : PrefillIO.IO p N where
  kernel := (p.prefill cosType sinType Bool.false total N).eraseDType
  projection := (PrefillIO.sourceIO p cosType sinType Bool.false total N).projection
  readMetadata := fun i => i.val
  read := fun row i => row * p.rowStride + i.val * p.hiddenStride
  write := fun token i => token * p.rowStride + i.val * p.hiddenStride

def prefillIndexCommutedIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (total N : Nat) : PrefillIO.IO p N :=
  { prefillOriginalIO p cosType sinType total N with
    kernel := (p.prefill cosType sinType Bool.true total N).eraseDType
    projection := (PrefillIO.sourceIO p cosType sinType Bool.true total N).projection }

/-- Select the greatest cumulative boundary at or before the token, exactly as
the source's signed maximum does, then gather both cache rows. -/
def prefillExpected {H N : Nat} (hN : 0 < N) (token : Nat)
    (xs : PrefillInputs H N) (i : Fin H) : ℝ × ℝ :=
  let boundary := (Finset.univ : Finset (Fin N)).sup' ⟨⟨0,hN⟩, Finset.mem_univ _⟩
    (fun j => if xs.ends j ≤ (token : Int) then xs.ends j else 0)
  let row := ((token : Int) - boundary).toNat
  (xs.cos row i, xs.sin row i)

def decodeOriginalIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (N : Nat) : DecodeIO.IO p where
  kernel := (p.decode cosType sinType Bool.false N).eraseDType
  projection := (DecodeIO.sourceIO p cosType sinType Bool.false N).projection
  readMetadata := id
  read := fun row i => row * p.rowStride + i.val * p.hiddenStride
  write := fun seq i => seq * p.rowStride + i.val * p.hiddenStride

def decodeIndexCommutedIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (N : Nat) : DecodeIO.IO p :=
  { decodeOriginalIO p cosType sinType N with
    kernel := (p.decode cosType sinType Bool.true N).eraseDType
    projection := (DecodeIO.sourceIO p cosType sinType Bool.true N).projection }

/-- Gather the row selected by each active sequence's signed metadata. -/
def decodeExpected {H N : Nat} (xs : DecodeInputs H N) (seq : Fin N) (i : Fin H) : ℝ × ℝ :=
  (xs.cos (xs.rows seq).toNat i, xs.sin (xs.rows seq).toNat i)

section Prefill
open scoped PrefillIO
variable (p : Parameters) (cosType sinType : TypedSource.Storage) (total N : Nat)
    (hN : 0 < N) (hRegion : p.cosOut ≠ p.sinOut) (hStride : 0 < p.hiddenStride)
include hRegion hStride

specification prefill_cache_correctness :
    Spec.Real (prefillOriginalIO p cosType sinType total N ⊨[total,hN] prefillExpected hN) := by
  exact PrefillIO.implements p cosType sinType Bool.false total N hN hRegion hStride

specification prefill_index_commuted_correctness :
    Spec.Real (prefillIndexCommutedIO p cosType sinType total N ⊨[total,hN] prefillExpected hN) := by
  exact PrefillIO.implements p cosType sinType Bool.true total N hN hRegion hStride
end Prefill

section Decode
open scoped DecodeIO
variable (p : Parameters) (cosType sinType : TypedSource.Storage) (N : Nat)
    (hRegion : p.cosOut ≠ p.sinOut)
    (hLayout : Function.Injective (fun i : TileIndex [p.B,p.H] =>
      i.1.val * p.rowStride + i.2.1.val * p.hiddenStride))
include hRegion hLayout

specification decoding_cache_correctness :
    Spec.Real (decodeOriginalIO p cosType sinType N ⊨[N] decodeExpected) := by
  exact DecodeIO.implements p cosType sinType Bool.false N hRegion hLayout

specification decode_index_commuted_correctness :
    Spec.Real (decodeIndexCommutedIO p cosType sinType N ⊨[N] decodeExpected) := by
  exact DecodeIO.implements p cosType sinType Bool.true N hRegion hLayout
end Decode

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.Correctness
