# Source boundaries under active verification

The wrapper allocates homogeneous `int32` input/output and passes contiguous
nonnegative strides. The low-level kernel has no program ID: every launched
program repeats the same accesses. Per-program proofs do not establish race
freedom or justify the host grid.

Each program performs 24 unmasked 2×2 stores. Their column shifts are
`0, 2, …, 46`. The largest loaded column shift is 26. For nonnegative strides,
the maximum accessed offsets are `stride_m + 27 * stride_n` for input and
`stride_m + 47 * stride_n` for output. The 4×4 wrapper example is out of bounds;
the 2×2 example launches zero programs. Neither is a successful kernel witness.

The independent mathematical result uses ordered writes, preserving overlap
between different output tiles. Per-tile address injectivity remains a separate
condition for interpreting concurrent lane stores. Initial-source specifications
require distinct source and destination regions. Full structural equivalence
does not require this separation and retains alias-dependent execution.

Signed stride expressions use mathematical integers, but the pointer model uses
natural offsets. Source-facing contracts therefore require nonnegative strides:
negative increments would otherwise be truncated at each pointer addition.
Finite-width stride arithmetic and address overflow are outside this model.
There is no payload arithmetic in the homogeneous `int32` wrapper, so copying
does not itself need an integer overflow assumption.

The new source family contains 15 homogeneous storage profiles: signed and
unsigned 8/16/32/64-bit labels, boolean, and six floating formats. Signed widths
share the Int channel and unsigned widths share Nat; these labels do not prove
machine-width conversion behavior. Mixed source/destination conversion remains
an explicit generic low-level gap, as recorded for `var_len_copy`; it is absent
from this wrapper. The complete real contracts and typed successful-execution witnesses have
passed local Lean checks. The public FP theorem and all eight headline axiom checks also passed.
Dependency-aware integration and independent replay are tracked separately;
these boundaries remain in force after validation.

The carrier-parametric ideal-real oracle explicitly applies `storedCell` to
loaded payloads. Floating storage uses the library's finite-value representation
map (`ofReal (storeValue value)`), including its bottom-value fallback; this is
not an IEEE conversion claim. The separately typed FP execution keeps backend
carriers and source precision operations intact.
