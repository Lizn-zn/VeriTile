# f8_conversion_utils

`Kernels.lean` retains both complete conversion kernels and their integer-index
commuted candidates. `kernel_f8_to_f16` loads fp8 e4nv/e5 and performs both
original fp16 stores. `kernel_f16_to_f8` converts fp16/fp32 input to the chosen
fp8 format. Region declarations retain input/output precision; explicit casts
represent Triton's implicit conversion to destination storage. All masks and
omitted load defaults remain.

`Correct.lean` displays four real specifications with the independent formula
`output[i] = initial_input[i]` after precision erasure. They include finite
allocation/coverage, actual-access bounds, successful full execution, every
active output and the complete tagged-cell frame. N, block size and program
index remain symbolic. Logical input/output regions may alias because the
load precedes both stores. Zero unspecified values remain an explicit premise
of the flat-memory adapter.

`FPEquiv.lean` proves both complete typed original/candidate equivalences. The
only edit commutes the natural-number product used in offsets; every floating
conversion and store remains. No numerical atom or cast-idempotence assumption
is used. This does not claim fp8 conversion is lossless or a round-trip identity.

`bench/tests/FPF8ConversionSource.lean` checks the actual typed loads/stores,
casts, masks, omitted defaults and duplicate store. The public finite witnesses
in `bench/tests/F8ConversionExecution.lean` copy -3 and 2 on a partial tile,
preserve an integer-tagged tail and run fully masked cases with empty
allocations for every precision/candidate variant. `Proofs/TypedExecution.lean`
inhabits all six precision-direction combinations from empty registers in the
abstract model, with the masked tail and original/candidate conversions.

Scope is one program. Host reinterpret/launches, grid races, fixed-width index
overflow, physical IEEE quantization and rounding certification are excluded.

Validation passed: complete repository closure, public finite witnesses,
typed model executions, precision/cast/mask regressions, full source
correspondence for both JITs, source gates, 32 audit/layout tests,
coverage/manifest checks and fresh independent comparator/trust replay of
11 files. Current fingerprints and theorem inventories match `Logs/comparator-check-eauavdpw`.
The initial disk-space failure was discarded and replayed from a fresh copy.
