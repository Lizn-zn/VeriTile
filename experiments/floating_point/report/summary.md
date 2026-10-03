# Numerical rule results

42 instances; 38 replayed; 34 accepted.

Each replicate contributes one mean across its IID scalar instances; R counts replicates.
z = |mean| / SE across replicate means (diagnostic only). U uses the configured magnitude gate.
B = abs(mean) + se_multiplier * SE; bias PASS requires B <= tau, in local ULPs.
Bias FAIL means an interval lies outside tolerance; INCONCLUSIVE means a boundary is crossed.
The SE bands are engineering criteria, not calibrated simultaneous or optional-stopping confidence guarantees.
Bias differences are normalized by each rounded golden value's output-format ULP before averaging.
The magnitude gate uses K=Ec/Er, the ratio of peak absolute oracle errors, without ULP normalization or an additive allowance.
Both errors zero gives K=0; a zero reference error with positive candidate error gives infinity.
K follows the FlashAttention maximum-error metric; tail extrapolation and the bias gate are additional criteria.
`empirical_max` means an observed maximum, not a fitted tail confidence bound.
Acceptance is statistical under the configured profile, not proof of strict floating-point equivalence.
Accept is pending until CPU replay. Missing statistics are shown as —, never zero.

| Rule | Format | R | z | B (ULP) | tau (ULP) | U | U type | Bias | Vars | Accept | State |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| ADD-COMMUTE | bf16 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-COMMUTE | fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | bf16 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| ADD-ASSOC | bf16 | 4096 | 0.5262654 | 5.747947e-05 | 0.05 | 1.156941 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ADD-ASSOC | bf16_fp32 | 4096 | 0.06535288 | 2.142983e-08 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ASSOC | fp32 | 4096 | 0.9255956 | 0.01525995 | 0.05 | 1.163875 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | bf16 | 4096 | 0.03629942 | 1.147891e-05 | 0.05 | 3.802424 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | fp32 | 4096 | 0.6556256 | 1.312948e-05 | 0.05 | 3.850519 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | bf16 | 4096 | 122.4037 | 0.001239349 | 0.05 | 2.338281 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | bf16_fp32 | 4096 | 0.4471699 | 1.772632e-10 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | fp32 | 4096 | 0.0719694 | 0.01296174 | 0.05 | 2.353754 | pot_pwm | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | bf16 | 4096 | 23.85633 | 0.0004666169 | 0.05 | 0.7544843 | pot_pwm | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | fp32 | 4096 | 1.114051 | 0.0267935 | 0.05 | 0.8064917 | pot_pwm | PASS | PASS | yes | COMPLETE |
| CANCEL | bf16 | 4096 | 74.3824 | 0.002713128 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | bf16_fp32 | 4096 | 1.593619 | 8.745921e-07 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | fp32 | 4096 | 0.5932181 | 0.0496093 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | bf16 | 4096 | 4330.745 | 0.008101662 | 0.05 | 19.99234 | pot_pwm | PASS | WARN | no | COMPLETE |
| DIV-RCP | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | fp32 | 4096 | 23.69138 | 5.621148e-05 | 0.05 | 20.84146 | pot_pwm | PASS | WARN | no | COMPLETE |
| SQRT-RSQRT | bf16 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | bf16_fp32 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | fp32 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| CAST-MOVE | bf16 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | fp32 | 4096 | 10.65762 | 53.82958 | 0.05 | 0.8497966 | pot_pwm | FAIL | PASS | no | COMPLETE |
| CAST-REMOVE | bf16 | 4096 | 1618.834 | 0.00295496 | 0.05 | 0.9288439 | pot_pwm | PASS | PASS | yes | COMPLETE |
| CAST-REMOVE | bf16_fp32 | 4096 | 1591.302 | 0.002957028 | 0.05 | 0.9447803 | pot_pwm | PASS | PASS | yes | COMPLETE |
| CAST-REMOVE | fp32 | 4096 | 246.5784 | 19.59246 | 0.05 | 6.372863e-05 | pot_pwm | FAIL | PASS | no | COMPLETE |
| ACC-WIDEN | bf16 | 4096 | 46.74694 | 0.0004120227 | 0.05 | 0.5824564 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ACC-WIDEN | bf16_fp32 | 4096 | 44.90049 | 0.0004025739 | 0.05 | 0.5706934 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ACC-WIDEN | fp32 | 4096 | 0 | 0 | 0.05 | 1 | empirical_max | PASS | PASS | yes | COMPLETE |

## Notes

- BF16-WIDEN-RETURN / fp32: requires bf16 input and output
- SQRT-RSQRT / bf16: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / bf16_fp32: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / fp32: sqrt domain violated; the requested normal distribution was not conditioned
