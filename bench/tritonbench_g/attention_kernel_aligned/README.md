# attention_kernel_aligned

- Source: `attention_kernel_aligned.py`, `_fwd_kernel_aligned`.
- Status: correctness/FP migration deferred at the user's request; not completed.

The Python host accepts fp16 and bf16, but the score dot explicitly uses
`out_dtype=OUT_DTYPE`. For bf16, the current upstream Triton frontend rejects
this option: [semantic.py](https://github.com/triton-lang/triton/blob/main/python/triton/language/semantic.py#L1445-L1448)
(checked 2026-10-08). The existing Lean DSL rejects it as well. This is a source
compilation issue, not a two-gates rejection or a numerical inequivalence proof.
Whether to retain this failure or add a separately labelled fp32-dot/cast repair
will be decided when this directory is revisited. No Python source was changed.

The historical Lean port also fails the current typed-dot frontend with its
generic OUT_DTYPE. Its old theorem text is not evidence of a successful current
build. The new real/FP entries and independent proof replay remain pending.
The fp16 source's score dot outputs fp16; its bias is added without log2 scaling.
These differences from `attention_kernel` must be retained when work resumes.
