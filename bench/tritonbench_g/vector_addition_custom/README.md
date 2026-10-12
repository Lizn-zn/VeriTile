# vector_addition_custom

Source: [vector_addition_custom.py](vector_addition_custom.py), TritonBench-G v1.

- [Kernels.lean](Kernels.lean): the complete original `_add_kernel` and a
  candidate changing the value inside `tl.store` from `a + b` to `b + a`.
  Both keep the original loads, addresses and masks. No extra output-register
  assignment is inserted. Size and block size remain symbolic.
- [Correct.lean](Correct.lean): both sources implement the independent real
  formula `as i + bs i`, including termination, active-address safety and
  preservation of all cells outside the active output window.
- [FPEquiv.lean](FPEquiv.lean): complete fp32 equivalence using `add_commute`.
  Both actual loaded operands must be finite immediately before the store,
  including masked lanes whose arithmetic is evaluated. The proof changes
  only the store value; its address, mask and dtype stay identical. All states
  and execution failures agree.
- [Proofs/Real.lean](Proofs/Real.lean): real execution, arithmetic, safety and
  memory-frame lemmas. The short FP composition uses `equiv_decompose` and
  `fp_prove` in its public entry.

`Source.prefixBeforeStore` locates the store without hard-coded statement
positions. Inserted non-store statements do not change its occurrence number.
The contextual FP lemma still checks the complete selected prefix and operands;
it does not allow arbitrary changes to a store.

The two proof layers share Kernels.lean and do not import each other. The
results cover each program, with the host launch outside the contract. They
make no bitwise IEEE or performance claim. Experimental dimensions select the
atom and do not restrict the symbolic dimensions in these proofs.

Current scopes and source fingerprints: [coverage_review.json](../coverage_review.json).
