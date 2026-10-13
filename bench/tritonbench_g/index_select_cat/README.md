# index_select_cat

The Python kernel performs a signed-index gather followed by a masked store:
`output[i * stride0 + col * stride1] = source[index[i] * stride0 + col * stride1]`.
Both strides come from the source tensor. The wrapper does not validate the
output strides or dtype; its shape checks, index-length truncation, launch grid,
and BI=1 / BC=512 launch configuration remain host-side boundaries. Kernel
proofs retain arbitrary BI/BC, strides, masks, and both program axes.

`Kernels.lean` preserves the legacy mathematical/natural-index transcription
and adds complete typed sources for six input and six output floating formats
(fp32, fp16, bf16, fp8e4nv, fp8e5, fp64), independently, with int32 or int64
index storage. Each variant covers all 72 profiles. Integer/boolean data and
other index formats are unsupported profiles; the Python wrapper does not
reject them. The candidate commutes only `pid0 * BI` to `BI * pid0` in the
initial index-window calculation.

The gathered values retain their source storage type, without arithmetic or
fp32 widening. When output storage differs, the source transcription makes
Triton's implicit store conversion explicit. This is a translation-surface
exception. Fp64 conversion boundaries retain the required inner precision
scope. `Proofs/Precision.lean` checks exact load/store ASTs for every profile,
same-storage stores without conversion, actual signed `.int` index loads,
original masks, and missing `other` arguments.

The independent real formula enumerates row/column pairs and updates each
memory cell with the last active lane targeting it. Its values are pinned only
to original gathered input cells. The public contract includes successful
full flat execution, all allocated-cell values, and a frame for every untouched
flat cell. It requires no source-index injectivity or output assumptions.
Overlapping output strides are covered by the stated per-program ordered-store
semantics. Input/output names may alias because all reads precede the store;
arbitrary byte reinterpretation between differently sized aliased views is
outside the typed-cell model.

Actual signed rows remain signed through address multiplication and addition.
The natural-address model requires nonnegative active rows and bounds on all
active reads and writes. These are contract assumptions, not extra Python
checks. Negative pointer displacements are not modeled as Python negative-index
wrapping. Int32/int64 declarations share the exact `Int` carrier; fixed-width
integer overflow validity remains a frontend obligation. Independent index-read
bounds remain explicit even when BC=0. No positive tile-size premise is added
to the new contract.

Host scheduling, cross-program races from overlapping strides or aliasing, and
GPU race resolution are outside this per-program theorem. Target-specific fp8
availability is not asserted. The old natural-index contract and all sixteen
legacy declaration headers/docstrings are preserved separately.

Full-kernel FP equivalence uses no floating arithmetic atoms and preserves
success/failure and complete state for every backend. Concrete finite-payload
backend witnesses execute all profiles with repeated signed indices and masked
padding; same-storage aliased row swaps verify that loads precede writes.
These witnesses test typed execution, not hardware numerical accuracy.
External comparator replay and GPU execution remain pending.

The integrated correctness/FP dependency build passes (3447 jobs). Three
public contracts have validated source/support fingerprints and `test-gap` status.
