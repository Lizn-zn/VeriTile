# attention_fwd_triton1

- Source file: `attention_fwd_triton1.py`
- Corpus: TritonBench-G v1
- Status: verified per-program real correctness and admitted FP equivalence.

This is chunked **linear attention**, without a softmax. Each chunk computes
`O_c = ((scale * Q_c) K_c^T) V_c + (scale * Q_c) H_c`, with
`H_c = sum_{j < c} K_j^T V_j`. `STORE` writes `H_c` before updating it;
`IFCOND` skips the zero carry contribution in the first chunk. Both switches
are parameters of all current specifications.

## Public entries

- `Kernels.lean` contains both complete fp32 sources: the original and a
  candidate commuting `b_q * scale` to `scale * b_q`. The supplied Python
  tests select fp32 storage, supplied through region metadata and inferred
  block-pointer load types. Both switches, five block pointers, explicit
  casts, `allow_tf32=False` dots and both stores remain visible. These are
  manually transcribed sources; the dtype-alias resolution is documented below.
- `Correct.lean` displays three input streams, both output windows and masks,
  buffer separation and independent input-only formulas. Both specifications
  prove successful flat-memory execution, active-address safety and the frame
  outside the active outputs. Disabled H stores preserve all H memory.
- `FPEquiv.lean` compares the complete typed programs, including failed
  executions. Its sole numerical atom is `mul_commute`. Finite checks apply
  to the actual loaded Q tile and interpreted scalar in every reachable
  iteration, after the local prefix and any preceding H store. No dot,
  carry-update, reassociation or cast-identity assumption is used.

The real scope is contiguous head lanes (`s_qt=BD`, `s_qd=1`), `T=NT*BT`,
`BT>0`, and `BD≤s_ht`. O and H are distinct buffers and do not alias Q/K/V.
The allocation must cover all input and active output addresses; H bounds
are required only when STORE is enabled. These are source-layout and memory
conditions, not experiment shapes. The unmasked block pointers do not justify
partially allocated tail chunks. The FP equivalence retains all symbolic
source dimensions and strides. Both layers are per-program; host grid
composition, concrete IEEE equality and performance are outside the claims.

## Supporting proofs

`Proofs/RealProjection.lean` connects both precision-erased sources to their
real bodies. `RealBranch`, `RealStore` and `RealIteration` handle the
first-chunk branch, matrix stores and complete iteration. `RealLoop` proves
the arbitrary-length invariant, actual output-cell values and frame;
`RealCandidate` proves the scalar-left execution and safety connections.
`RealSafety` follows every actual access, and `RealIO` assembles both flat
three-input/two-output contracts. Historical default-branch rounding and
mathematical helpers remain in `Proofs/Legacy.lean`.

`Proofs/FP.lean` uses `equiv_decompose` and `fp_prove`. `FPModel.lean` and
`TypedExecution.lean` give a model of the admitted scalar rules and successful
two-iteration executions with satisfied domains for both sources and all four
flag combinations. Opaque zero dots suffice for this model-consistency check.
`RealExecution.lean` additionally gives a nonzero real witness: O=[1,2],
H=[0,1] when STORE is enabled, and H unchanged otherwise. This distinguishes
pre-update snapshots from post-update ones. These small Lean witnesses check
semantics; they do not constrain the general specifications or claim GPU
launchability.

## Dtype-alias transcription

The source has six casts in order: `p_h.dtype.element_ty`, four uses of
`b_q.dtype`, and `p_o.dtype.element_ty`. With the supplied fp32 Q/K/V and the
wrapper's `q.new_empty` H and `empty_like(q)` O, all six targets are fp32.
Kernels.lean spells each one `tl.float32`, retaining its location and operand.
The legacy dtype-alias syntax currently drops floating casts; using that
shorthand here would lose FP operations. Region metadata preserves all three
fp32 loads without adding a Python-absent load argument.

This explicit alias resolution is registered as a transcription exception.
The generic lexical source gate cannot prove it. The mapping above was checked
against the Python source, and the Lean source/projection/execution proofs
check the resulting typed programs, not Python translation or GPU execution.

## Validation

Both public entries and all helpers compile with `TritonBenchGVerified`.
Independent comparator replay and trust checks pass for all 16 files in this
directory and the exact streamed IO adapter; 55 named targets were checked.
The coverage/support fingerprints, proof-gap manifest and specification sheets
are current. The source gate passes with the documented alias-resolution
exception; separately resolving just those three aliases matches all six cast
targets and every load/store expression against the Python source.
