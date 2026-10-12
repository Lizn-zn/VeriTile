# bmm_chunk_fwd

- Source: `bmm_chunk_fwd.py`, TritonBench-G v1; `_bmm_chunk_fwd_kernel` at line 22.
- Public source: `Kernels.lean`.
- Real correctness: `Correct.lean`.
- Admitted FP equivalence: `FPEquiv.lean`.
- Status: complete real/FP contracts, regressions and independent replay accepted.

The original computes chunked `A · Bᵀ`, optionally filters unequal signed
sequence IDs, converts the output and stores a chunk tile. The candidate
reverses only `acc + dot(a,b)` to `dot(a,b) + acc` in every K iteration.

Both real specifications use an independent initial-array contraction. The
actual shortened chunk length, partial K blocks, three program IDs and
independent strides remain symbolic. Causal mode skips a whole tile when
`pid_n * BLOCK_N >= (pid_m + 1) * BLOCK_M`; retained diagonal tiles have no
additional elementwise triangular mask. Data and metadata loads use the
shortened length, but stores use `chunk_size`. Padded final-chunk output
positions are therefore written as zero. Sequence IDs are signed and compared
literally; the inactive load sentinels are distinct values -1 and -2.

The real contract includes successful flat execution, actual masked-address
safety and every-cell memory frame. The value proof requires positive K-block
size and injective active output addresses. Distinct region allocations are disjoint;
metadata windows are required only in the sequence-filter arm. Host launch
validity, fixed-width overflow and concurrent grid composition remain outside
this per-program real claim.

A and B independently use fp32/fp16/bf16 storage. All nine pairs retain the
host's bf16-then-fp16-then-fp32 dot priority, source load casts, fp32 accumulation
and output conversion. Real correctness erases precision; FP equivalence keeps
it. `#print_fp_assumptions bmm_dot_first_equiv` prints only `add_commute`.
The domain checks the actual carry and dot result after each pair of loads in
every reached K iteration. No dot/reduction reassociation or additional GPU
experiment is used.

`BmmChunkFwdLoop` computes 336 and 1025 through two partial K blocks with
nonzero padding and undefined values. `BmmChunkFwdComplete` runs both variants
and both flags, in region and flat memory: negative IDs retain 336 and filter
1025, padded output cells are zero, and untouched cells retain fp16 tags.
A separate causal tile skip preserves every cell. `FPBmmChunkFwdSource` checks
storage-before-cast, dot priority, source context and domain presence.

`Proofs/` contains the full source execution, loop induction, independent
formula, signed metadata branch, masked stores, safety and flat-IO assembly.
`Proofs/Legacy.lean` retains the older noncausal/no-sequence specialization
with whole K blocks and fully active output tiles. It is not the current
public correctness or admitted FP specification.

The consistent abstract FP model also supplies complete typed original and
candidate executions through two K blocks and both metadata arms. This is
a nonempty-domain witness, not GPU emulation. The integrated build passes
(4,033 jobs). All three focused regressions, the 16 layout/coverage tests, source gates and
the 371-declaration/196-file reviewed inventory pass. Independent comparator
and trust replay accept all 34 files: every source/proof module in this
directory and the three regressions. The verifier checks 189 named declarations,
1,299 kernel theorem targets and all 1,023 snapshot input hashes. Receipt:
`Logs/comparator-check-gs9xcq3i`.
