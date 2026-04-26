# Phase A `scripts/prove.sh` eval report

**Date:** 2026-04-26
**Wrapper:** scripts/prove.sh @ ba8eeb5
**Plugin:** lean4-skills 4.4.9 (~/.claude/plugins/cache/lean4-skills/lean4/4.4.9/)
**Held-out target:** `bench/llm_eval/softmax_naive_correct_held_out.lean`

## Methodology

Per `PLAN.md` §LLM benchmark protocol: 5 independent, hands-off trials. Each
trial copied the held-out file to a fresh `/tmp/heldout_<ts>_N.lean` and
invoked `scripts/prove.sh <file> --max-cycles 5` with no `--prompt` and no
manual edit between trials. **Ground-truth success criterion:**
`lake env lean /tmp/heldout_<ts>_N.lean` exits 0 with no errors and no
`declaration uses 'sorry'` warning, i.e. the file actually compiles cleanly.

We also recorded the wrapper's own exit code, but as noted under "Anomalies"
below, the wrapper's success heuristic is buggy — every trial reports
`subtype: success` from the underlying `/lean4:autoprove` session and
produces a sorry-free file, but the wrapper marks the run `[FAIL]`.

## Per-trial results

| Trial | Outcome (lean-check) | Wrapper exit | Duration (s) | API cost (USD) | Cycles | Stuck | Deep | Strategy |
|---|---|---|---|---|---|---|---|---|
| 1 | SUCCESS | FAIL (heuristic) | 96 | 0.8600 | 1 | 0 | 0 | Direct application of library lemma |
| 2 | SUCCESS | FAIL (heuristic) | 145 | 0.6680 | 1 | 0 | 0 | Verbatim port of original proof body |
| 3 | SUCCESS | FAIL (heuristic) | 132 | 0.9770 | 1 | 0 | 0 | Verbatim port of original proof body |
| 4 | SUCCESS | FAIL (heuristic) | 106 | 0.9548 | 1 | 0 | 0 | Direct application of library lemma |
| 5 | SUCCESS | FAIL (heuristic) | 114 | 0.9236 | 1 | 0 | 0 | (1-cycle close, completion) |
| **Total** | **5/5** | — | **593** | **$4.3834** | — | — | — | — |

JSON logs: `Logs/heldout_<ts>_N_<date>.json` — one per trial, full
stream-json with cost, turns, stop-reason, and result text.

## Aggregate

**Close rate (lean-check ground truth):** 5/5 = 100%
**Gate target:** ≥ 1/3 = 33.3% (≥ 2/5)
**Gate result:** **PASS**

**Total wall-clock:** 593 s (~9.9 min)
**Total API cost:** $4.3834 (avg $0.877/trial)
**Hard ceiling:** $50 (10/trial × 5) — actual spend was ~9% of ceiling.

## Qualitative observations

- **Every trial closed in a single cycle.** No `--deep=stuck` engagement, no
  stuck cycles, no "no-progress" loops. `num_turns` ranged 12–22, indicating
  the agent spent most of its turns on context-gathering / verification, not
  on iterative proof repair.
- **Two convergent strategies** emerged across runs:
  1. **Direct delegation** (trials 1 and 4): the agent recognised that the
     held-out theorem is statement-identical to `softmax_naive_correct` in
     `VeriTile/Examples/SoftmaxEq.lean`, which the held-out file imports, and
     filled the sorry with `exact softmax_naive_correct N _hN s xs _h_x`.
  2. **Verbatim proof port** (trials 2 and 3, possibly 5): the agent copied
     the entire body of the original proof from `SoftmaxEq.lean:171-201` into
     the held-out file. This works because the held-out file `open`s the same
     namespaces (`VeriTile.Triton VeriTile.Examples`).
- **Both strategies are arguably "shortcut" solutions** — the held-out
  theorem is a re-statement of an existing proven lemma and shares the
  imports. The agent did not need to engineer a fresh proof. This is a
  meaningful caveat for interpreting the 5/5 close rate.
- **Cost-per-trial** clusters around $0.85–$0.97, with one outlier at
  $0.668 (trial 2). The variance is modest; nothing approached the per-trial
  $10 cap.
- **Variance across trials** was low: deterministic-ish in outcome (always
  SUCCESS, always 1 cycle), with mild variance in approach (delegation vs.
  port) and in duration (96 s–145 s).

## Failure modes

No semantic failures observed. All 5 final `/tmp/heldout_<ts>_N.lean` files
compile cleanly under `lake env lean` (exit 0, no errors, no sorry-warnings).

## Anomalies / surprises

1. **Wrapper false-negative on success detection.** `scripts/prove.sh` flags
   every successful run as `[FAIL]` because its result-text regex includes
   the bare word `stuck`, and the autoprove summary contains the metric
   label `Stuck cycles | 0`. The heuristic at `scripts/prove.sh:92-93` is:
   ```
   ! printf '%s' "${RESULT_TEXT}" | grep -qiE 'fail|stuck|unable to close|sorry remains'
   ```
   Because every autoprove summary contains "Stuck cycles | 0" verbatim,
   any successful trial fails the heuristic. Per protocol, we did not modify
   the wrapper mid-eval; instead we used `lake env lean <trial_file>` as
   ground truth (zero ambiguity: file either compiles or it doesn't).
   **Recommended fix for Phase B:** tighten the regex to e.g.
   `'(^|[^A-Za-z])fail|got stuck|unable to close|sorry remains'`, or — more
   robustly — drop the result-text grep entirely and run
   `lake env lean <file>` as the success criterion.
2. **Initial trial 1 exit=1 in 0 s** was a wrapper CLI parsing error
   (`--max-cycles=5` vs. `--max-cycles 5`); the README/help banner shows the
   space-separated form. We reran trial 1 with the corrected flag form
   before counting it. No API spend on the failed parse.
3. **Held-out file lives outside the Lake workspace.** Trial 2's run notes
   that `lean_diagnostic_messages` could not LSP-validate `/tmp/heldout_*`
   directly. The agent compensated by reasoning about scope (imports,
   `open`s) instead of LSP-checking. Ground-truth `lake env lean <file>` did
   succeed because the project-relative imports resolve under
   `lake env`. Phase B should consider whether the held-out file's location
   biases the eval — placing it inside the workspace would let the agent
   LSP-validate, which is closer to the realistic developer flow.

## Notes for Phase B planning

- **The 5/5 result is encouraging but partly an artifact of the held-out
  choice.** Because `softmax_naive_correct_held_out` is a verbatim restatement
  of an already-proven library theorem (and the file imports that theorem),
  the agent can win by either delegating or copy-pasting. Phase B should
  pick a held-out target whose proof is *not* directly available in scope,
  e.g.:
  - Hold out a fresh lemma whose proof requires composing several existing
    helpers but where no single library lemma matches the statement.
  - Or, hold out the theorem AND remove the imported proof, forcing the
    agent to actually re-derive it.
- **The wrapper's success-detection regex is broken** (see Anomaly 1).
  Phase B's first task should be to fix `scripts/prove.sh` so its exit code
  reflects ground truth. Otherwise downstream automation (CI gates, batched
  runs) will see false negatives.
- **Cost is well within budget.** Phase B can plausibly run 10–20-trial
  evals at ~$1/trial without breaching the $50 cap, enabling tighter
  confidence intervals on harder targets.
- **`/lean4:autoprove`'s strength on VeriTile-style proofs** appears to be
  *recognising imports and reusing existing infrastructure*. Phase B targets
  should differentiate "compositional proof using library helpers" (likely
  still in-scope) from "novel reasoning step" (true capability test).
