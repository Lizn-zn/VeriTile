# Numerical rule results

42 instances; 38 replayed; 30 accepted.

z = max |z| over output buckets (diagnostic only). U uses the configured magnitude gate.
B = max(abs(mean) + se_multiplier * SE); bias PASS requires B <= tau, in local ULPs.
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
| ADD-ASSOC | bf16 | 4096 | 3.658139 | 0.005621907 | 0.05 | 1.058843 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ADD-ASSOC | bf16_fp32 | 8704 | 1.73225 | 2.23833e-05 | 0.05 | 164.1267 | pot_pwm | PASS | FAIL | no | COMPLETE |
| ADD-ASSOC | fp32 | 4096 | 3.126887 | 6.516323 | 0.05 | 132.7016 | pot_pwm | INCONCLUSIVE | FAIL | no | COMPLETE |
| MUL-ASSOC | bf16 | 4096 | 3.993513 | 0.001294595 | 0.05 | 0.297186 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | fp32 | 4096 | 3.412681 | 0.001215082 | 0.05 | 0.3437202 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | bf16 | 4096 | 5.723825 | 0.00680902 | 0.05 | 176 | empirical_max | PASS | FAIL | no | COMPLETE |
| MUL-DISTRIB | bf16_fp32 | 4096 | 1 | 3.576279e-07 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | fp32 | 4096 | 3.149106 | 5.780505 | 0.05 | 2.060443e+07 | pot_pwm | INCONCLUSIVE | FAIL | no | COMPLETE |
| FMA-CONTRACT | bf16 | 4096 | 4.052241 | 0.009310908 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | fp32 | 4096 | 3.136422 | 8.428577 | 0.05 | 0 | empirical_max | INCONCLUSIVE | PASS | no | COMPLETE |
| CANCEL | bf16 | 4096 | 4.893518 | 0.02136486 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | bf16_fp32 | 4096 | 3.117238 | 0.0001815225 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | fp32 | 4096 | 2.741671 | 10.2166 | 0.05 | 0 | empirical_max | INCONCLUSIVE | PASS | no | COMPLETE |
| DIV-RCP | bf16 | 4096 | 71.11831 | 0.009062381 | 0.05 | 0.6904547 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | fp32 | 4096 | 4.80541 | 0.001224256 | 0.05 | 1.017085 | pot_pwm | PASS | PASS | yes | COMPLETE |
| SQRT-RSQRT | bf16 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | bf16_fp32 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | fp32 | 0 | — | — | — | — | — | — | — | no | NUMERIC_EVENT |
| CAST-MOVE | bf16 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | bf16_fp32 | 4096 | 0 | 0 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | fp32 | 4096 | 3.62554 | 24746.97 | 0.05 | 6.67413e-05 | pot_pwm | INCONCLUSIVE | PASS | no | COMPLETE |
| CAST-REMOVE | bf16 | 4096 | 29.03541 | 0.00396336 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-REMOVE | bf16_fp32 | 4096 | 28.69514 | 0.003929621 | 0.05 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-REMOVE | fp32 | 4096 | 7.330954 | 60.71056 | 0.05 | 7.731313e-06 | pot_pwm | FAIL | PASS | no | COMPLETE |
| ACC-WIDEN | bf16 | 4096 | 4.667298 | 0.004971794 | 0.05 | 0.1484949 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ACC-WIDEN | bf16_fp32 | 4096 | 4.320873 | 0.00481253 | 0.05 | 0.2133777 | pot_pwm | PASS | PASS | yes | COMPLETE |
| ACC-WIDEN | fp32 | 4096 | 0 | 0 | 0.05 | 0.9999999 | pot_pwm | PASS | PASS | yes | COMPLETE |

## Notes

- BF16-WIDEN-RETURN / fp32: requires bf16 input and output
- SQRT-RSQRT / bf16: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / bf16_fp32: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / fp32: sqrt domain violated; the requested normal distribution was not conditioned
