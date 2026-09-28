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

The seed generator is intentionally deterministic. Start with it:

```bash
python make_training_data.py --scenes 10000 --output data/scene_functions.jsonl
python train_transformer.py --data data/scene_functions.jsonl --output runs/transformer --steps 10000
```

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
