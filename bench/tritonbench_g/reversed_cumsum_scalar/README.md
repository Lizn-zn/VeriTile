# reversed_cumsum_scalar

- Source file: `reversed_cumsum_scalar.py` (pinned upstream `603e28a`)
- Corpus: TritonBench-G v1
- Status: DONE — DSL port (`ReversedCumsumScalar.lean`), genuine suffix-sum
  closed-form spec, and `ComputeCorrect.Realizes` summary
  (`reversed_cumsum_scalar_output_summary_general`) proven sorry-free.

`chunk_global_reversed_cumsum_scalar_kernel` walks each batch·head row in
chunks **from the last chunk down to the first** (`range(cdiv(T,BT)−1,−1,−1)`)
and emits `b_s − tl.cumsum(b_s) + b_z` — the within-chunk suffix sum plus the
carried suffix total `b_z` (updated with the current chunk total before the
store). Note the kernel realizes the reversed cumsum through the subtraction
identity with a **forward** `tl.cumsum`; the reverse-range loop (not
`reverse=True`) was the DSL surface this port waited on.

Proof architecture mirrors the forward twin `chunk_cumsum_kernel` (within-chunk
scan identity + carry-fold) and the packaging of the sibling `reversed_cumsum`:
the single-chunk path (`T ≤ BT`, covering the Python benchmark `T = 4`,
`BT = 16`) is genuine end-to-end; the per-chunk face is stated under the
explicit carry-buffer hypothesis, with the cross-chunk `b_z` scheduling as the
documented trusted runtime boundary (#290-style carried register). A full
loop-invariant proof in the style of `chunk_cumsum_kernel`'s
`surface_loop_correct` is the natural upgrade.
