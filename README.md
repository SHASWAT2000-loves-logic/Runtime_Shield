# Answer-Revealing Reprompt — Shielded Cooking Experiments

This document describes the `answer-revealing-reprompt` branch of the MuJoCo Franka cooking benchmark.

The branch is an **incremental change on top of the fixed `automaton-liveness-v2` implementation**. The six experiments, MuJoCo scenes, robot executor, automaton, synthetic shield distribution, repeated-self-loop decay/pruning, model interface, models, parameters, logging, and trial launcher remain the same. The deliberate experimental change is the rejection reprompt.

The model still returns **one JSON action per request**. The shield never requests or depends on token-derived VLM action probabilities.

See [`RESULTS_ANSWER_REVEALING_REPROMPT.md`](RESULTS_ANSWER_REVEALING_REPROMPT.md) for the complete 120-run evaluation.

## Research question

The previous fixed `automaton-liveness-v2` experiment showed that the runtime shield could prune and reject repeated state-preserving actions, but a non-answer-revealing rejection message did not reliably make the VLM recover.

This branch asks:

If the shield reveals one exact currently live progress action in the rejection reprompt, will the VLM follow that correction and recover from repeated or unsafe proposals?

The experiment intentionally changes **only the rejection-reprompt behavior** so that this recovery mechanism can be tested while preserving the rest of the setup.

## Recipe

The normal recipe is a partial order:

```text
onion + garlic
        ↓
      tomato
        ↓
turmeric + coriander + salt
        ↓
      cream
        ↓
      finish
```

Onion and garlic may be added in either order. The three core spices may be added in any order. At least one salt increment is required, with at most three permitted salt additions.

Experiments 08 and 10 dynamically add cumin as a required spice after tomato.

## Shield semantics

At each automaton state, actions are classified as:

- **Unsafe (`S`)**: not permitted in the current recipe state; probability is zero and the action is rejected.
- **Co-live (`D`)**: temporarily permitted, but repeated use is penalized and can be pruned. In this benchmark, `stir` is co-live.
- **Live progress (`H_l`)**: currently required actions that advance the recipe.
- **Unconstrained (`U`)**: permitted actions outside the active live obligation.
- **Finish**: accepted only in the terminal recipe state.

The shield does not directly execute or substitute an action. A robot action is executed only after the VLM returns an action and that action passes the runtime check.

## Inherited repeated-self-loop behavior

The repeated-state-preserving-action fix from `automaton-liveness-v2` is unchanged.

For a state-preserving unconstrained action:

1. The first accepted self-loop starts an action-specific counter in the current automaton state.
2. Its shield-side synthetic probability is multiplied by:

```text
(1 - gamma) ^ count
```

3. Once its normalized pre-prune probability is at or below `theta_prune`, it is assigned probability zero.
4. A later proposal of that action is rejected and reprompted.
5. Counters reset when the automaton/cooking state changes.
6. State-changing unconstrained actions, such as still-valid extra salt increments, are not treated as repeated self-loops.

With `gamma=0.20` and `theta_prune=0.10`, the deterministic Experiment 07 smoke-test state produces the same repeated-garlic trajectory as the previous fixed branch:

```text
0.2500 → 0.1667 → 0.1087 → 0.0699 → pruned to 0
```

The fourth redundant garlic proposal is therefore rejected after three accepted no-motion self-loops.

## Branch-specific answer-revealing reprompt

This is the experimental change.

When a model proposal is rejected and the current automaton state has at least one live progress action, the shield chooses the **first live action in the existing automaton/action ordering**, converts it to the normal planner JSON schema, and includes that exact JSON object in the rejection message.

For the Experiment 07 state in which onion and garlic are already complete and tomato is the live obligation:

```text
ACTION REJECTED.
The proposed action has been repeated without making any progress and is now
blocked in the current cooking state. You must not propose it again.

The runtime shield has identified the correct next action. Return this exact
JSON object now:
{"reasoning":"The runtime shield identified the required progress action.","action":"add","item":"tomato"}

Do not choose a different action and do not output any additional text.
```

Unsafe rejected actions use the same answer-revealing mechanism when a live progress action exists.

### Important interface property

The revealed JSON is **prompt guidance**, not a direct shield action:

```text
model proposes action
        ↓
shield checks proposal
        ↓
proposal rejected
        ↓
shield reveals one live action in the next prompt
        ↓
model must return a new JSON action
        ↓
new model action is checked again
        ↓
only an accepted model-returned action can execute
```

The one-action-per-request interface therefore remains intact.

### Generic progress directive

The existing generic non-answer-revealing progress directive is unchanged for ordinary requests.

On an answer-revealing rejection retry, the generic directive is suppressed so the prompt does not simultaneously say both:

```text
"The runtime shield is not providing the action for you."
```

and provide an exact action.

## Experiments

| ID | Script | Scenario |
|---|---|---|
| 01 | `cooking_stars.py` | Baseline partial-order recipe |
| 05 | `cooking_exp05_chili_distractor.py` | An irrelevant chili container appears after the first action |
| 07 | `cooking_exp07_human_completed_garlic.py` | The environment moves garlic into the bowl without adding it to command history |
| 08 | `cooking_exp08_cumin_appears_required_minimal_prompt.py` | Cumin appears after tomato and becomes required |
| 09 | `cooking_exp09_onion_rollback.py` | Onion is moved back to the table after the robot adds it |
| 10 | `cooking_exp10_onion_rollback_plus_cumin.py` | Onion rollback and later dynamic cumin requirement |

## Models and batch configuration

The default launcher runs:

- `qwen2.5vl:3b`, with `num_predict=1024`;
- `qwen3-vl:32b`, with `num_predict=4096`;
- shielded and unshielded modes;
- six experiments;
- five trials per condition.

Total:

```text
6 experiments × 2 models × 2 modes × 5 trials = 120 runs
```

Shielded defaults:

```text
gamma                    = 0.20
theta_prune              = 0.10
guidance_threshold       = 0.95
shield_reprompt_limit    = 12
max_steps                = 50
salt_max_additions       = 3
seed                     = 0
```

Unshielded runs use a 50-decision budget.

## Requirements

The repository must already contain the clone-and-run MuJoCo/Franka assets used by the existing cooking benchmark, including the Franka model and cooking scene.

Runtime requirements include:

- Python;
- MuJoCo Python bindings;
- NumPy;
- Pillow;
- an Ollama-compatible server reachable from the experiment machine;
- the configured Qwen model identifiers available on that server.

The launcher defaults to:

```text
http://volta13:11434
```

Override it using `OLLAMA_URL`.

## Key files

```text
cooking/
├── cooking_automaton.py
├── cooking_strategy_template_guide.py
├── cooking_unshielded_policy.py
├── cooking_stars.py
├── cooking_exp05_chili_distractor.py
├── cooking_exp07_human_completed_garlic.py
├── cooking_exp08_cumin_appears_required_minimal_prompt.py
├── cooking_exp09_onion_rollback.py
├── cooking_exp10_onion_rollback_plus_cumin.py
├── run_automaton_v2_trials.sh
├── test_automaton_v2.py
└── automaton_v2_runs/
```

The answer-revealing behavior is implemented in `cooking_strategy_template_guide.py`, while each experiment runner suppresses the generic progress directive during a rejection retry.

## Validate the branch

From `cooking/`:

```bash
python -m py_compile *.py
bash -n run_automaton_v2_trials.sh
python test_automaton_v2.py
```

Expected smoke-test result:

```text
AUTOMATON V2 ANSWER-REVEALING REPROMPT SMOKE TEST PASSED
```

The deterministic smoke test also verifies that the revealed action is generated from the current automaton live-action set.

## Run the complete batch

```bash
cd cooking
TRIALS=5 ./run_automaton_v2_trials.sh
```

The launcher is sequential and writes results to `automaton_v2_runs/`.

## Run selected conditions

The launcher accepts comma-separated environment variables.

Examples:

```bash
# Only Qwen3, shielded, Experiment 07, five trials
MODELS=qwen3 MODES=shielded EXPERIMENTS=07 TRIALS=5 \
  ./run_automaton_v2_trials.sh

# Both modes for Experiments 09 and 10
MODELS=qwen3 MODES=shielded,unshielded EXPERIMENTS=09,10 TRIALS=5 \
  ./run_automaton_v2_trials.sh

# Alternate server and output directory
OLLAMA_URL=http://localhost:11434 OUTPUT_ROOT=my_runs TRIALS=1 \
  ./run_automaton_v2_trials.sh
```

Supported values:

```text
MODELS:      qwen25,qwen3
MODES:       shielded,unshielded
EXPERIMENTS: 01,05,07,08,09,10
```

## Output structure

```text
automaton_v2_runs/
├── manifest_<timestamp>.tsv
├── logs/
│   └── <run_id>.log
├── reasoning/
│   ├── <run_id>.jsonl
│   └── <run_id>_shield.jsonl   # shielded runs only
└── images/
    └── <run_id>/
        └── *.png
```

The manifest is the authoritative run index and records model, mode, experiment, trial, exit code, status, and log path.

The main reasoning JSONL traces preserve each complete prompt, including any answer-revealing rejection message and the model response. Shield JSONL traces preserve action classification, pruning, and acceptance/rejection decisions.

## Success criteria

A run passes only when:

- every required ingredient is complete in the current MuJoCo state;
- the model returns a valid `finish` action;
- no hard recipe violation invalidates the run;
- no model/output failure terminates the run first.

A shielded rejection is not itself a failure. A run fails if the model does not recover within the configured reprompt limit or another terminal failure occurs.

## Known limitations

- The answer-revealing reprompt intentionally gives the VLM information that was withheld in the previous branch; this is a recovery experiment, not a test of autonomous inference after rejection.
- When more than one live action is valid, the shield deterministically reveals the first action in the existing automaton/action ordering. It does not optimize among multiple live choices.
- Five trials per condition are useful for diagnosis but are too few for strong statistical conclusions.
- Model generations are stochastic, so differences from the previous 120-run batch are descriptive rather than a controlled paired estimate of causal effect.
- Qwen3 frequently consumes the full `num_predict=4096` budget before returning final JSON.
- The shield distribution is synthetic and must not be interpreted as the VLM's internal action-probability distribution.
