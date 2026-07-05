# layer_norm_welfold

- Source file: `layer_norm_welfold.py`
- Corpus: TritonBench-G v1
- Status: DONE — DSL port (`LayerNormWelfold.lean`), genuine closed-form
  mean / rstd / affine-output specs, and `ComputeCorrect.Realizes` summary
  (`layer_norm_welfold_output_summary_general`) proven sorry-free.

Despite the directory name, `triton_red_fused_native_layer_norm_no_welford`
is the **two-pass (no-Welford)** torch-inductor LayerNorm forward: a first
tiled loop accumulates the plain row sum (mean → `in_out_ptr0`, after a
`tl.debug_barrier()`), a second tiled loop accumulates the squared
deviations (`libdevice.rsqrt(var + 1e-05)` → `in_out_ptr1`, after a second
barrier), and a third tiled loop stores the normalized affine output
`((x − mean)·rstd)·w + b` to `out_ptr0`.

Proof architecture mirrors the Welford sibling `fused_layernorm_triton`
(same specs, simpler reduction algebra — no moment-combine identity
needed): the full faithful surface (all three loops, both barriers, `tl.full`
accumulators) lowers to the algorithm layer; the `XBLOCK = 1` reduction
phases are genuine end-to-end from `in_ptr0` for a single reduction block
(`rnumel ≤ RBLOCK`), with the second pass consuming the *register* mean;
the normalize loop is covered for **all** chunk indices under the honest
mean/rstd-cell hypotheses (exactly the values the reduction phases store).
The multi-iteration accumulator recurrence is the pure step face
`sum_accumulate_step_closed`; the cross-iteration `_tmp3`/`_tmp12` register
scheduling is the documented trusted runtime boundary (#290-style carried
register).
