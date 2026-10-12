import bench.tritonbench_g.bgmv_expand_slice.Kernels
import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Real correctness of both complete sources in Kernels. Each active split
row contains a full rank-K GEMV sum, optionally added to INITIAL output.
All source dtypes, cast choices, input-load choices and accumulation modes
are covered after precision erasure. The -1 index skips all input/output work.

The contract includes successful flat-memory execution, active read/write
bounds and preservation of every other cell. It requires four disjoint
allocations, signed metadata either -1 or nonnegative, initial input pins and
zero undef. Dimensions and strides remain symbolic; K≤BK, BN>0, SPLIT_N>0 and
positive output stride are stated below. The source has no global N mask:
all computed split rows need valid weight/output storage. Launch composition,
fixed-width address overflow and concrete IEEE behavior are outside this
per-program real specification. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.Correctness
open VeriTile.Triton
open scoped VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealIO
noncomputable section

abbrev Parameters := RealSetupSource.Parameters
abbrev Inputs := RealIO.Inputs

section
variable (p : Parameters) (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast even add : Bool)

/-- Input has BK lanes, active when EVEN_K or k<K, even for an empty output
window. Weights cover every split row and K rank entries; initial output is
pinned at the write window only when ADD_INPUTS. Allocation roles are vector,
weights, output and signed metadata: the read/write output is allocated once.
`length = (N + SPLIT_N - 1) / SPLIT_N`. The split term in the output address
is NOT multiplied by outputStride, matching the source. -/
def originalIO : RealIO.IO p where
  kernel := (TypedSource.original input weight cast out_dtype p.input p.weights p.output p.N p.K p.indices
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
    p.outputStride p.slice p.BN p.BK p.splits even add).eraseDType
  projection := RealIO.projection p input weight out_dtype cast Bool.false even add
  even := even
  add := add
  readIndex := fun batch => batch
  readInput := fun batch k => batch*p.inputBatchStride+k.val*p.inputStride
  readWeight := fun split slot g k => p.slotStride*slot.toNat+split*p.length*p.rowStride+g.val*p.rowStride+k.val*p.colStride
  write := fun split batch g => batch*p.outputBatchStride+split*p.length+p.slice*p.outputStride+g.val*p.outputStride

def outputFirstIO : RealIO.IO p :=
  { originalIO p input weight out_dtype cast even add with
    kernel := (TypedSource.outputFirst input weight cast out_dtype p.input p.weights p.output p.N p.K p.indices
      p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
      p.outputStride p.slice p.BN p.BK p.splits even add).eraseDType
    projection := RealIO.projection p input weight out_dtype cast Bool.true even add }

end

/-- Independent mathematical sum. All three fields of xs are INITIAL input
values, including initialOutput; no computed output is assumed. -/
def expected (p : Parameters) (hK : p.K ≤ p.BK) (add : Bool) (xs : Inputs p) (g : Fin p.length) : ℝ :=
  let v := ∑ k : Fin p.K, xs.vector ⟨k.val, lt_of_lt_of_le k.isLt hK⟩ * xs.weights g k
  if add then v+xs.initialOutput g else v

specification bgmv_expand_slice_correctness (p : Parameters)
    (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast even add : Bool) (hBN : 0 < p.BN) (hstride : 0 < p.outputStride)
    (hK : p.K ≤ p.BK) (hsplits : 0 < p.splits) (hw : p.weights ≠ p.output) :
    Spec.Real (originalIO p input weight out_dtype cast even add ⊨ expected p hK add) := by
  exact RealIO.implements p input weight out_dtype cast Bool.false even add hBN hstride hK hsplits hw

specification bgmv_output_first_correctness (p : Parameters)
    (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast even add : Bool) (hBN : 0 < p.BN) (hstride : 0 < p.outputStride)
    (hK : p.K ≤ p.BK) (hsplits : 0 < p.splits) (hw : p.weights ≠ p.output) :
    Spec.Real (outputFirstIO p input weight out_dtype cast even add ⊨ expected p hK add) := by
  exact RealIO.implements p input weight out_dtype cast Bool.true even add hBN hstride hK hsplits hw

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.Correctness
