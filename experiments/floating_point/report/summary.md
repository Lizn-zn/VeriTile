# Numerical rule results

42 instances; 38 replayed; 30 accepted.

z = max |z| over output buckets. U uses the configured magnitude gate.
`empirical_max` means an observed maximum, not a fitted tail confidence bound.
Acceptance is statistical under the configured profile, not proof of strict floating-point equivalence.
Accept is pending until CPU replay. Missing statistics are shown as —, never zero.

| Rule | Format | R | z | U | U type | Bias | Vars | Accept | State |
|---|---|---:|---:|---:|---|---|---|---|---|
| ADD-COMMUTE | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-COMMUTE | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-COMMUTE | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ROUND-IDEM | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| BF16-WIDEN-RETURN | fp32 | — | — | — | — | — | — | no | UNSUPPORTED |
| ADD-ASSOC | bf16 | 4096 | 3.982262 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ASSOC | bf16_fp32 | 4096 | 1.414386 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| ADD-ASSOC | fp32 | 4096 | 4.148575 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | bf16 | 4096 | 3.496949 | 0.67445 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-ASSOC | fp32 | 4096 | 3.54922 | 0.6334643 | pot_pwm | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | bf16 | 4096 | 8.162913 | 0.3690348 | pot_pwm | WARN | PASS | no | COMPLETE |
| MUL-DISTRIB | bf16_fp32 | 4096 | 1 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| MUL-DISTRIB | fp32 | 4096 | 3.828405 | 0.3063636 | pot_pwm | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | bf16 | 4096 | 32.8901 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| FMA-CONTRACT | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| FMA-CONTRACT | fp32 | 4096 | 4.203631 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | bf16 | 4096 | 4.333449 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | bf16_fp32 | 4096 | 3.290487 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CANCEL | fp32 | 4096 | 3.924846 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | bf16 | 6144 | 2.867842 | 1.900391 | pot_pwm | PASS | PASS | yes | COMPLETE |
| DIV-RCP | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| DIV-RCP | fp32 | 6656 | 3.173428 | 1.763261 | pot_pwm | PASS | PASS | yes | COMPLETE |
| SQRT-RSQRT | bf16 | 0 | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | bf16_fp32 | 0 | — | — | — | — | — | no | NUMERIC_EVENT |
| SQRT-RSQRT | fp32 | 0 | — | — | — | — | — | no | NUMERIC_EVENT |
| CAST-MOVE | bf16 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | bf16_fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |
| CAST-MOVE | fp32 | 4096 | 9.274597 | 0.8099674 | pot_pwm | FAIL | PASS | no | COMPLETE |
| CAST-REMOVE | bf16 | 4096 | 33.38985 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| CAST-REMOVE | bf16_fp32 | 4096 | 32.98056 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| CAST-REMOVE | fp32 | 4096 | 8.5717 | 1.849728e-05 | pot_pwm | FAIL | PASS | no | COMPLETE |
| ACC-WIDEN | bf16 | 4096 | 8.887007 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| ACC-WIDEN | bf16_fp32 | 4096 | 8.49242 | 0 | empirical_max | WARN | PASS | no | COMPLETE |
| ACC-WIDEN | fp32 | 4096 | 0 | 0 | empirical_max | PASS | PASS | yes | COMPLETE |

## Notes

- BF16-WIDEN-RETURN / fp32: requires bf16 input and output
- SQRT-RSQRT / bf16: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / bf16_fp32: sqrt domain violated; the requested normal distribution was not conditioned
- SQRT-RSQRT / fp32: sqrt domain violated; the requested normal distribution was not conditioned
