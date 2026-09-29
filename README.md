# Roblox Scene Agent

This is a CUDA-ready prototype for *render-in-the-loop* Roblox character and
creature construction.
The policy proposes exactly one constrained scene function per step. Every
function and every conversation turn are stored on disk, so a scene can be
rebuilt, inspected, deleted, and restored later.

The prototype deliberately separates three jobs:

1. `scene_agent.core` owns persistent conversation/scene memory and replays the
   active function history into a deterministic scene graph.
2. `scene_agent.renderer` makes low-resolution multi-view CUDA renders for
   rapid training rewards. It is a surrogate renderer, not Roblox Studio.
3. `scene_agent.transformer` is the real 12-layer, 99.2M-parameter causal
   Transformer. It generates one grammar-checked scene function at a time.
   `scene_agent.policy` remains only as a small RL smoke-test policy.

## Why the model never writes XML directly

The model emits `SceneFunction` records such as `add_rig_root` and
`add_rig_part`. The validator
rebuilds the complete scene from those records and `rbxmx.py` emits the XML.
Deleting a function simply marks it inactive; replay removes its part and any
function with a now-missing dependency is skipped. Restoring it is reversible.

## Install on a CUDA host

Create a fresh Linux CUDA instance, then install the PyTorch build matching the
driver/CUDA version shown by that host. After PyTorch is installed:

```bash
pip install -r requirements.txt
python demo.py --project runs/castle
python train.py --project runs/train --steps 64 --grader local
python -m unittest discover -s tests -v
```

`demo.py` creates:

```text
runs/castle/
  session.json                 persistent conversation + function history
  castle.rbxmx                 Studio-import candidate
  renders/front.ppm            quick GPU surrogate renders
  renders/side.ppm
  renders/top.ppm
```

The `.rbxmx` output must be opened and checked in Roblox Studio before use.
Use Studio screenshots as the high-quality/ground-truth grade and the CUDA
surrogate for fast step-by-step rewards.

## Roblox Studio interpreter

`studio_plugin/SceneInterpreter.plugin.lua` is the model-only Studio bridge.
Install it as a local Studio plugin, open **Scene Agent → Interpreter**, copy
the complete `runs/<project>/session.json` into its text area, then select
**Validate and rebuild GeneratedScene**. The plugin validates and replays every
active function into `Workspace.GeneratedScene`; it keeps the old model if a
document fails validation and creates Studio undo waypoints.

It does not run arbitrary Luau and does not need a network connection. The
Python session document is the only allowed input.

See [LUA_COMMANDS.md](LUA_COMMANDS.md) for the complete safe command catalogue,
including the two-angle joint schema. These are constrained JSON model calls;
they are not permission to execute arbitrary Luau.

## Grammar-constrained language

`scene_agent.grammar` is the authority for model output. It provides a strict
Python validator, a readable BNF grammar, and `JSON_SCHEMA` for a
schema-constrained inference server. Call `validate_generated_call()` before
`SceneSession.apply_generated_call()`; the latter also validates every call as
a defense-in-depth check.

## Training loop

`scene_agent.environment.RenderInTheLoopEnv` performs:

```text
policy proposes one function
  -> schema/dependency validation
  -> persistent scene replay
  -> CUDA render
  -> grader reward - previous reward
  -> next policy step
```

`LocalQwenVLGrader` is now connected to the training loop. It downloads and
runs `Qwen/Qwen2.5-VL-3B-Instruct` locally through Transformers, receives the
three CUDA-rendered views plus metadata, and must return structured JSON. Use
`--grader heuristic` only for a dependency-free smoke-test. No render, prompt,
or grade is sent to an external AI API. Because Roblox is not a differentiable
renderer, train the policy with preference/RL rewards, not image loss
backpropagated through Studio.

`train.py` is an end-to-end REINFORCE smoke-test: one sampled action triggers
one validated update, one CUDA render, and one grade. Its neural policy is
small on purpose; it validates the architecture, not final model quality.

## Complete 100M scene cycle

After supervised training, use `run_scene_cycle.py` for the persistent
generation/refinement loop:

```text
saved session + original goal
  -> 100M Transformer generates one grammar-constrained function
  -> validated function mutates/rebuilds the saved scene
  -> CUDA front/side/top renders
  -> local Qwen visual grade with full conversation/history
  -> ordered repair queue
  -> local Qwen director writes one next objective to session
  -> repeat, or finish
```

```bash
python run_scene_cycle.py --checkpoint runs/transformer/checkpoint-10000.pt \
  --tokenizer runs/transformer/scene_tokenizer.json \
  --session runs/my_bridge --goal "Build a low-poly stone bridge with rails." \
  --max-steps 20 --grader qwen
```

Add `--explore-seed 42` only when creating diverse, reproducible training
conversations. Omit it for stable evaluation. The script shares one local Qwen
checkpoint between the visual grader and conversation director to conserve GPU
memory. It saves each step's multi-view renders and the full conversation
to the session directory. It also appends `trajectory.jsonl`, including the
exact context before each call, the emitted function, grade, and a review
candidate flag. It does not update neural-network weights automatically:
filter candidate traces with the visual grade and human/Studio review before
using them in later supervised or RL runs.

### Fast final-quality preview renderer

`run_scene_cycle.py` defaults to `--renderer beauty`: a CUDA ray-cast preview
renderer with perspective/orthographic cameras, depth, rotations, primitive
surface normals, directional lighting, material response, and a ground plane.
It additionally writes `beauty.ppm` for a human-facing perspective preview.
Use `--renderer surrogate` for the older, low-resolution reward renderer when
running many cheap RL rollouts. Neither renderer claims to be Roblox Studio;
the Studio interpreter and a Studio screenshot remain the final visual check.

## 100M Transformer generator

`scene_agent/transformer.py` now implements the actual generator:

```text
16,384-token BPE vocabulary + 2,048 positions
  → 768-wide token states
  → 12 Transformer blocks, each with 12 causal-attention heads
  → RMSNorm and nonlinear 768 → 3072 → 768 feed-forward layers
  → tied vocabulary output head
  = 99,192,576 trainable parameters
```

Train it with supervised function examples before attempting RL:

```bash
python train_transformer.py --data data/scene_functions.jsonl --output runs/transformer --steps 10000
```

Each JSONL record must contain a complete model-facing context and one valid
next function:

```json
{"context":"user request + current scene + grades", "target":"{\"op\":\"add_part\",...}"}
```

Generate a known-correct seed dataset with `python make_training_data.py`.
See [TRAINING.md](TRAINING.md) for the required request-author, render, grader,
evaluation, and curriculum pipeline. Do not use a visual grader as the only
source of truth for supervised targets.

The trainer learns only the target function tokens; it masks the context from
the loss. It trains a 16k BPE tokenizer from the dataset on first run, saves it
beside checkpoints, and refuses a smaller vocabulary because that would change
the 100M architecture.

Generate exactly one next function from a trained checkpoint with automatic
grammar validation/retry:

```bash
python generate_function.py --checkpoint runs/transformer/checkpoint-10000.pt \
  --tokenizer runs/transformer/scene_tokenizer.json \
  --context "Build a glass window; active scene contains a brick wall."
```

All conversation turns remain in `session.json`, and
`SceneSession.model_context()` returns them all by default along with the
current replayed scene. A production model adapter can explicitly ask for a
bounded/retrieved view when it reaches its context limit; that does not discard
the original history.

## Function schema

The model creates one visible primary/root body part, then adds later parts
with their Motor6D skeleton connection in the same reversible function:

```json
{"op":"add_rig_root","name":"RobotTorso","shape":"Block","position":[0,6,0],"size":[4,5,2.5],"color":[0.3,0.6,0.9],"material":"Metal"}
{"op":"add_rig_part","name":"LeftArm","shape":"Block","position":[-3,6,0],"size":[1,4,1],"color":[0.3,0.6,0.9],"parent_part_id":"fn_000001","joint_name":"LeftShoulder","joint_position":[-2,7,0],"role":"limb","motion_type":"universal","axes":[[1,0,0],[0,1,0]],"angle_ranges":[[-45,45],[120,-120]]}
{"op":"set_part_properties","target_id":"<function-id>","properties":{"position":[2,5,0],"transparency":0.25}}
{"op":"delete_function","target_id":"<function-id>"}
{"op":"restore_function","target_id":"<function-id>"}
```

Add more primitives in `SceneGraph.apply()` instead of allowing arbitrary
Luau. That is the safety boundary which makes each generated piece reversible.
