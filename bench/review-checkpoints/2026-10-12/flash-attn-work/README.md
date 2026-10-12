# flash_attn review in progress

The benchmark is still pending. No GPU experiment has been run and no repository acceptance or independent comparator/trust pass is claimed.

The staging source includes the complete `_fwd_kernel` and its candidate, which only commutes the natural head-base product. Q and O independently support fp16, bf16 and fp32; both dot left operands are explicitly cast to fp16, so K and V remain fp16. L is fp32. Full Python/Lean AST comparison and precision checks pass.

The public FP statement compares whole typed states, including execution failure. It has an empty input domain and uses no numerical FP atoms. Complete nonempty typed executions pass for all nine Q/O storage combinations and both causal modes. These witnesses use a singleton numerical algebra and retain types, natural addresses and masks. They do not claim physical IEEE values.

The independent real formulas retain the exact decimal scale `sm_scale * 1.44269504`, the common Q/K/V/O base `pid1 * stride_q_head`, and V's use of `stride_k_seqlen`. Every visited KV tile is loaded in full, since the source has no block boundary checks. The reference therefore includes `ceil(hi / BLOCK_N) * BLOCK_N` keys before causal filtering, with actual bounds required for all loads even when a key is causally masked.

Pure online-softmax algebra proves the output ratio and LSE formulas and positive denominators for nonempty key lists. Full typed-source erasure and both original/candidate flat execution equality pass. Symbolic source initialization, one loop iteration, pointer advancement, unchanged loop memory, and the two-store postlude have local Lean proofs. The remaining central obligation is the numerical loop invariant connecting these pieces to the independent formulas for arbitrary sizes, strides and both modes; full trace safety and tagged-cell output frames are also pending.

`WitnessData.lean` provides a disjoint finite allocation, actual read/write bounds, and a tail-sensitive reference case: the first three V keys are zero and the fourth is four. At SEQLEN=3 and BLOCK_N=2 the noncausal independent formula gives output 1 and LSE 2; causal rows give outputs 0 and LSEs 0/1. These data/formula checks still need the complete finite execution theorem.

`review.json` lists only local checks whose source hashes match their successful receipts. The source and all logs are in this persistent workspace directory; scratch probes are excluded from migration. Root integration, public real IO contracts, final finite execution, repository gates and official independent replay remain required before this case is counted complete.
