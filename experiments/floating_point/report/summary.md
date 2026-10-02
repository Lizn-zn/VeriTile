# Numerical rule results

42 instances; 38 replayed; 32 accepted.

Each replicate contributes one mean across its IID scalar instances; R counts replicates.
z = |mean| / SE across replicate means (diagnostic only). U uses the configured magnitude gate.
B = abs(mean) + se_multiplier * SE; bias PASS requires B <= tau, in local ULPs.
Bias FAIL means an interval lies outside tolerance; INCONCLUSIVE means a boundary is crossed.
The SE bands are engineering criteria, not calibrated simultaneous or optional-stopping confidence guarantees.
Errors are normalized per element by the output-format ULP at the rounded golden value before aggregation.
The magnitude gate uses peak normalized oracle errors and an additive allowance of 1 local ULP.
`empirical_max` means an observed maximum, not a fitted tail confidence bound.
Acceptance is statistical under the configured profile, not proof of strict floating-point equivalence.
Accept is pending until CPU replay. Missing statistics are shown as —, never zero.

| Rule | Format | R | z | B (ULP) | tau (ULP) | U | U type | Bias | Vars | Accept | State |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| ADD-COMMUTE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-COMMUTE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | fp32 | — | — | — | — | — | — | — | — | no | UNSUPPORTED |
| ADD-ASSOC | bf16 | 4096 | 0.5262654 | 5.747947e-05 | 0.05 | 1.032816 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ADD-ASSOC | bf16_fp32 | 50176 | 0.3619857 | 4.997314e-09 | 0.05 | 300.3429 | pot_pwm | PASS | FAIL | no | COMPLETE |
| ADD-ASSOC | fp32 | 4096 | 0.9255956 | 0.01525995 | 0.05 | 120.7185 | pot_pwm | PASS | FAIL | no | COMPLETE |
| MUL-ASSOC | bf16 | 4096 | 0.03629942 | 1.147891e-05 | 0.05 | 0.297186 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | fp32 | 4096 | 0.6556256 | 1.312948e-05 | 0.05 | 0.3442812 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | bf16 | 4096 | 122.4037 | 0.001239349 | 0.05 | 176 | empirical_max | PASS | FAIL | no | COMPLETE |
| MUL-DISTRIB | bf16_fp32 | 4096 | 0.4471699 | 1.772632e-10 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | fp32 | 4096 | 0.0719694 | 0.01296174 | 0.05 | 2.640162e+07 | pot_pwm | PASS | FAIL | no | COMPLETE |
| FMA-CONTRACT | bf16 | 4096 | 23.85633 | 0.0004666169 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | fp32 | 4096 | 1.114051 | 0.0267935 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | bf16 | 4096 | 74.3824 | 0.002713128 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | bf16_fp32 | 4096 | 1.593619 | 8.745921e-07 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | fp32 | 4096 | 0.5932181 | 0.0496093 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | bf16 | 4096 | 4330.745 | 0.008101662 | 0.05 | 0.6904547 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | fp32 | 4096 | 23.69138 | 5.621148e-05 | 0.05 | 1.016309 | pot_pwm | PASS | PASS | yes | COMPLETE |
| SQRT-RSQRT | bf16 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | bf16_fp32 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | fp32 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| CAST-MOVE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | fp32 | 4096 | 10.65762 | 53.82958 | 0.05 | 6.74171e-05 | pot_pwm | FAIL | PASS | no | COMPLETE |
| CAST-REMOVE | bf16 | 4096 | 1618.834 | 0.00295496 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-REMOVE | bf16_fp32 | 4096 | 1591.302 | 0.002957028 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-REMOVE | fp32 | 4096 | 246.5784 | 19.59246 | 0.05 | 7.744817e-06 | pot_pwm | FAIL | PASS | no | COMPLETE |
| ACC-WIDEN | bf16 | 4096 | 46.74694 | 0.0004120227 | 0.05 | 0.1285883 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ACC-WIDEN | bf16_fp32 | 4096 | 44.90049 | 0.0004025739 | 0.05 | 0.2910845 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ACC-WIDEN | fp32 | 4096 | 0 | 0 | 0.05 | 0.9999999 | pot_pwm | PASS | PASS | yes | COMPLETE |

## Notes

- BF16-WIDEN-RETURN / fp32: requires bf16 input and output
- SQRT-RSQRT / bf16: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / bf16_fp32: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / fp32: sqrt domain violated; the requested normal distribution was not conditioned
