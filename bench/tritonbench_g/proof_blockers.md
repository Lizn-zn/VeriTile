# TritonBench-G Proof Blockers

No current TritonBench-G port exposes an explicit algorithm-layer `hAlg`
correctness blocker.

The mechanical audit still remains a translation-consistency gate, not a
substitute for future human line review against `review_criteria.md`.

The stronger proof-status inventory for #146 lives in
`proof_gap_manifest.tsv` and is checked by
`bench/check_proof_gap_manifest.py`. It classifies every current
`output_summary` declaration and links each remaining non-full proof gap to a
specific follow-up issue plus a blocker family. The manifest is intentionally conservative:
`full_value_candidate` means no local proof-gap marker was found in the summary
context, not that future human review is forbidden from downgrading it.
The broad #147 quantization bucket has been split. Real-to-int8 cast semantics
track under #154 and now have an executable DSL/AST semantics path. The
now-discharged #158 quantization scale/value coupling rows connect the checked
int8, quantize-copy-kv, grouped quantize-kv-copy, and quantize-kv-transform
outputs directly to their full Python-shape surfaces. Rows whose local blocker
is primarily attention, matmul,
recurrence, reduction, or explicit blocked-summary work track under the
corresponding family issue.
The broad #150 recurrent/cumsum bucket has been split by mechanism:
the now-discharged `chunk-cumsum-carry-fold` (#185) tracks chunk cumsum
summaries now connected to full scalar, vector, and chunked forward surfaces,
the now-discharged
`decay-cumsum-scan-fold` (#186) tracks `decay_cumsum.py` summaries now
connected to full prepare, forward cumsum, and backward global-cumsum surfaces,
the now-discharged `recurrent-state-loop-fold` (#187) connects chunk-gate,
HGRN, and RWKV recurrent state summaries to their full producer surfaces,
the now-discharged `gla-output-tile-producer` (#188) tracks
`chunk_gla_simple.py` summaries whose proof used to start from a precomputed
output tile, and
`reverse-cumsum-directional-scan` (#94) tracks reverse cumsum direction
semantics.
The broad #149 attention/softmax bucket has been split as well:
`attention-final-store-lift` tracks summaries that still connect a faithful
surface to a final-store/proof-oriented writeback from precomputed Acc/Score/Prob
tiles under #161, while the #162 rows now split further into
`attention-forward-online-softmax-recurrence`,
`attention-score-probability-reduction`, `attention-context-decode-reduction`,
`token-attention-reduction`, and `attention-backward-score-reduction`. The
LLaMA and Bloom token-softmax case-1 rows plus the reduce-V and Mistral
token-attention case-1 rows are now discharged by connecting the checked
probability/output directly to their full Python-shape surfaces.
The broad #151 reduction/layernorm aggregation bucket has been split into
narrower follow-ups: the now-discharged
`chunk-delta-forward-recurrence-store` (#190) connects the two chunk-delta
forward summaries to the full producer surface for `h`, `v_new`, and optional
`final_state`, while the now-discharged
`layernorm-backward-residual-recompute-aggregation` (#191) covers the LayerNorm
backward residual/recompute summaries that used to compose row-level C1/C2
reductions, DX/Y writebacks, and partial DW/DB slices.
The #161 final-store bucket has been split again into kernel-specific producer
obligations: the now-discharged
`attention-fwd-triton1-bo-bhpre-producers` (#165),
the now-discharged `dense-attention-acc-store` (#166), whose remaining
Q/K/V streaming-softmax `Acc`/`L` producer proof is now discharged by
`dense-attention-online-softmax-recurrence` (#199),
`context-attention-mistral-sliding-window-acc-store` (#167),
`context-attention-nopad-varlen-acc-store` (#167), and
`flash-decode-normalized-vector-store` (#168).
The #168 flash-decode normalized-vector bucket has been split into
kernel-specific stage2 recurrence obligations:
the now-discharged `flash-decode-llama-stage2-normalization` (#171) and
`flash-decode-phi-stage2-normalization` (#172).
The #171 LLaMA stage2 normalization bucket now connects loop-produced `Acc` and
`SumExp` values to the final `O` writeback as a full-value candidate, and the
#181 LLaMA running-max recurrence step over `Mid_O_LogExpSum` is now also a
full-value candidate.
The #172 Phi stage2 recurrence bucket has been split into narrower
obligations: the now-discharged `flash-decode-phi-running-max-recurrence`
(#175), the now-discharged `flash-decode-phi-masked-accumulator-recurrence`
(#176), and the now-discharged `flash-decode-phi-normalization-store` (#177).
The #175 path states and proves the Python test-shape running `max_logic`
recurrence, #176 carries the Python test-shape masked `Mid_O` load through the
`AccOut` and `SumExpOut` recurrence step, and #177 connects the Python
test-shape `Acc` and `SumExp` outputs to the final masked `Out` writeback; all
three are full-value candidates in `proof_gap_manifest.tsv`.
The broad #148 matmul/dot bucket is now discharged: GEMV, BMM, dequantization,
IV-dependent matmul, plain matmul, activation-tail, and TMA summaries connect
their checked outputs directly to full Python-shape surfaces and are
full-value candidates in `proof_gap_manifest.tsv`.
The #191 layer-norm backward residual/recompute paths now connect the checked
Python test-shape outputs to the full backward surface for DX, recomputed Y,
and partial DW/DB, so the affected `layer_norm_ops.py` summaries are
full-value candidates.
The broad #153 rotary/cache bucket has also been split into narrower value
proof follow-ups: `rope-head-slice-lift` covers RoPE summaries whose Python
surface is faithful but whose value proof is still stated over Q/K head slices,
and `rotary-2d-tile-value-lift` covers rotary summaries with row-level `o0` /
`o1` value proofs that still need the full `[BLOCK_M, BLOCK_HALF]` tile lift.
The explicit blocked-output summaries formerly tracked by the broad #152
`semantic-blocker` bucket are all quantization `llrint` / int8-cast blockers
and now track under the open #154 `fixed-width-int8-cast-semantics` family.

## Translation-Surface Blockers

No current TritonBench-G port has an active documented translation-surface
blocker.

### Required VeriTile surface extensions

The attention helper-call surfaces are currently modeled as opaque outer-kernel
statements for translation coverage. Future proof work may add executable
semantics for projecting those helper calls, but that is no longer a
translation-surface blocker.
