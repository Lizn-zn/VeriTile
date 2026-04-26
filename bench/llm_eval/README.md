# LLM benchmark held-out set

These files are held-out evaluation targets for the LLM proof-drafting tool
(`scripts/prove.sh`, wrapping `/lean4:autoprove`).

Held-out files are NOT part of the main library build (`VeriTile.lean`
does not import them). The main library theorem stays intact and proven;
the held-out copy has its proof body replaced by `sorry` for the LLM tool
to attempt.

## Phase A held-out

- `softmax_naive_correct_held_out.lean` — held-out copy of `softmax_naive_correct`
  from `VeriTile/Examples/SoftmaxEq.lean`.

## Eval protocol

See `PLAN.md` §LLM benchmark protocol. Summary:
- 5 independent runs per held-out lemma
- No human edit to the LLM output before re-running
- No prompt iteration during eval
- Cost / wall-clock / token counts reported

## Running the eval

```bash
# One-off run
scripts/prove.sh bench/llm_eval/softmax_naive_correct_held_out.lean

# Full Phase A eval (5 trials, see Workstream 5 in implementation plan)
for trial in 1 2 3 4 5; do
  cp bench/llm_eval/softmax_naive_correct_held_out.lean /tmp/heldout_${trial}.lean
  scripts/prove.sh /tmp/heldout_${trial}.lean --max-cycles=5
done
```

Per-trial JSON logs land in `Logs/heldout_*_*.json`; aggregate eval results land in `bench/llm_eval/results/`.
