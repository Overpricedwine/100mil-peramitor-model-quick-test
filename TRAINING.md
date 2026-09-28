# Training data and grading plan

The 99.2M model is an **untrained generator** until it sees many valid next
function examples. The model architecture is complete; these are the missing
training assets:

1. Known-correct construction traces, split into train, validation, and held-out
   evaluation sets.
2. Natural-language requests describing those traces.
3. Multi-view renders and a fixed visual-grading rubric.
4. A curriculum: one part, property edits, multi-part models, deletion and
   restoration, then longer conversations with existing scenes.
5. Checkpoints, metrics, and a human-reviewed holdout set.

The scene catalogue must deliberately balance buildings, objects/props,
vehicles, furniture, tools, and environment pieces. Character or creature
examples are optional categories, never the default. Every seed request and
grade uses a low-poly constraint: simple primitives and readable silhouettes.

## Recommended data flow

```text
deterministic scene specification
  -> valid canonical function trace (the supervised truth)
  -> local request-author prompt produces a natural-language request
  -> one JSONL example for each next function
  -> train the Transformer
  -> render its predicted scene
  -> local visual grader scores request vs. renders + metadata
  -> conversation director reads the complete saved scene + grade and writes
     one small next improvement request, or ends the conversation
  -> use grader scores for ranking/repair/RL after supervised training
```

The seed generator is intentionally deterministic, but its templates vary
across buildings, vehicles, furniture, props, tools, environment objects, and
simple creature models (robot, dog, bird, and crab).
It includes rotations, transparency, a `finish` action, and a short revision
conversation with edit/delete/restore targets. Function IDs are stable
(`fn_000001`, etc.) and appear in the context before they can be targeted.

Whole template families are held out: fountains are validation and rock
clusters are test. Generate separate files rather than training on all data:

```bash
python make_training_data.py --scenes 10000 --split train --output data/train.jsonl
python make_training_data.py --scenes 10000 --split validation --output data/validation.jsonl
python make_training_data.py --scenes 10000 --split test --output data/test.jsonl
python train_transformer.py --data data/train.jsonl --output runs/transformer --steps 10000
```

On the Vast GPU, use the local Qwen model to write varied training requests
from the verified seed specifications:

```bash
python make_training_data.py --scenes 10000 --split train --request-author qwen --seed 42 \
  --output data/train.jsonl
```

The request author alternates `minimal` and `specific` requests. Its seed and
style hint change wording only; the known scene specification and canonical
function trace remain the verified target.

## Grader variation and continuing conversations

The local VLM grader is deterministic by default so reward scores and final
evaluation can be compared fairly. For training exploration, call
`grade(..., suggestion_seed=123)`; it uses low-temperature sampling but the
same seed reproduces the same repair priorities. The conversation director
supports `decide(..., selection_seed=123)` in the same way. The grader receives
the complete saved conversation, function history (including deleted functions)
and the previous grade, then returns `next_objective` plus an ordered
three-item `repair_queue`. The director selects only the first uncompleted
repair per turn and sees the queue again after the scene is rebuilt and graded.
Use `continue_after_grade(...)` to persist that one selected objective as a
system turn before asking the function generator for its next JSON call.

Do not use sampled grades for held-out model comparisons. Use deterministic
grading there, and reserve seeded variation for generating diverse repair
conversations during training.

## Reward fine-tuning (first RL stage)

`train.py` is only a small-policy REINFORCE smoke test. Do not use it to claim
the 100M text Transformer has been RL trained. The practical first stage for
the real model is **offline reward fine-tuning**: run `run_scene_cycle.py`,
review its high-scoring `trajectory.jsonl` samples in Studio, then train with a
reward-weighted token loss and a KL penalty that keeps the model close to its
supervised checkpoint:

```bash
python train_rft.py --trajectories runs/my_bridge/trajectory.jsonl \
  --checkpoint runs/transformer/checkpoint-10000.pt \
  --tokenizer runs/transformer/scene_tokenizer.json \
  --output runs/rft --min-score 80 --steps 1000
```

The trainer defaults to only `candidate_for_review` trajectories. That flag is
grader-based, not a substitute for checking the Studio result; use
`--include-unreviewed` only for an experiment. It optimizes high final grade
and non-negative score change, and clips exponential reward weights so one
unreliable VLM grade cannot dominate. The KL penalty is the safety leash that
reduces grammar/language drift.

Online PPO/GRPO is a later step. It needs parallel isolated sessions, saved
old-token log-probabilities, a frozen reference policy, multi-sample advantage
normalization, and periodic human/Studio evaluation. Add it only after the
offline RFT checkpoint improves the held-out scenes.

`scene_agent/prompt_templates.py` contains two versioned prompts. The request
author receives the known scene specification and must only describe it. The
grader sees the request, metadata and three renders, then returns a structured
rubric score. Save the prompt version, generated request, render paths, grade,
and model version alongside each run.

## Do not make the grader the only teacher

Having one VLM invent requests *and* grade its own results creates a feedback
loop: it can reward its own blind spots. Keep the canonical function trace as
the supervised truth, hold out some scene templates and prompt wording, and use
a separate grader checkpoint or at least a different fixed prompt for the final
evaluation. Review a sample of grades in Studio before using them as RL reward.

## Grading rubric

Grade requested components/counts, layout/scale/orientation, appearance
(colour/material/transparency), and structural plausibility separately. The
grader must identify missing elements and suggest a concrete next repair, not
just provide a score. It must not infer hidden geometry from a flattering camera
angle; this is why it receives front, side and top renders plus scene metadata.
