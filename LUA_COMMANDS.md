# Scene Agent command catalogue

The model does **not** receive arbitrary Lua execution. It can emit exactly one
JSON command from this list per turn. Python validates the command before it is
saved, and the Studio plugin validates the complete saved `session.json` before
it creates Roblox instances. `scene_agent.function_language.call_to_luau()` is
only a readable preview of these calls, not executable source code.

## Part creation

### `add_part`

Creates one ordinary visible `Part`. It can be anchored and collidable.

```json
{"op":"add_part","name":"Rock","shape":"Block","position":[0,1,0],"rotation":[0,0,0],"size":[4,2,3],"color":[0.4,0.4,0.4],"material":"Slate","transparency":0,"anchored":true,"can_collide":true}
```

Allowed shapes are `Ball`, `Block`, `Cylinder`, `Wedge`, and `CornerWedge`.
Materials must be one of the grammar's Roblox `Enum.Material` names.

### `add_rig_root`

Creates the one visible primary/root `Part` for a moving character or creature.
It is always unanchored and non-colliding. It may be a torso, shell, chassis,
segment, or any other central body piece; it is not a hidden humanoid root.

```json
{"op":"add_rig_root","name":"CrabShell","shape":"Ball","position":[0,3,0],"size":[6,3,5],"color":[0.2,0.5,0.8],"material":"SmoothPlastic"}
```

### `add_rig_part`

Creates one visible child `Part` **and** its parent `Motor6D` in one reversible
function. `parent_part_id` can target the root or any earlier rig child, so the
generator can make branches, chains, tails, wings, many legs, or non-humanoid
robot mechanisms.

```json
{"op":"add_rig_part","name":"TailSegment","shape":"Block","position":[0,3,4],"size":[1,1,3],"color":[0.2,0.5,0.8],"parent_part_id":"fn_000001","joint_name":"TailBase","joint_position":[0,3,2],"role":"limb","motion_type":"universal","axes":[[1,0,0],[0,1,0]],"angle_ranges":[[-50,50],[135,-135]]}
```

Joint fields:

| Field | Meaning |
| --- | --- |
| `motion_type` | `fixed` has no moving angle; `hinge` has one; `universal` has two independent rotation angles. |
| `axes` | Zero, one, or two local 3D rotation-axis vectors, matching `motion_type`. The validator normalizes them; universal axes cannot be parallel. |
| `angle_ranges` | One `[min,max]` pair per axis, in degrees within `-180..180`. `min <= max` is an ordinary interval. `min > max` wraps across `+/-180`; `[135,-135]` permits 135° through 180°, then -180° through -135°. |

This deliberately supports two-angle 3D joints without prescribing a humanoid
layout. Twist is not generated in this version; that avoids unstable three-axis
Euler animation and can be added later as a separate, explicit degree of
freedom.

When Studio imports the session, the plugin writes the motion data to
`Motor6D` attributes: `SceneAgentMotionType`, `SceneAgentAxis1`,
`SceneAgentAxis2`, `SceneAgentRange1Min`, `SceneAgentRange1Max`, and their
second-axis equivalents. A later animation tool reads these attributes before
setting a `Motor6D.Transform`.

## Reversible edits

### `set_part_properties`

Changes one existing part. Valid properties: `name`, `shape`, `position`,
`rotation`, `size`, `color`, `material`, `transparency`, `anchored`, and
`can_collide`. Rigged parts cannot be made anchored or collidable.

```json
{"op":"set_part_properties","target_id":"fn_000002","properties":{"color":[1,0.5,0.1],"transparency":0.2}}
```

The compact aliases `set_material`, `set_color`, `set_transparency`,
`set_size`, `set_position`, `set_rotation`, `set_anchored`, and
`set_can_collide` are accepted by Python, but the generator is trained to emit
the canonical `set_part_properties` form.

### `delete_function` and `restore_function`

`delete_function` marks a function inactive. Its dependent functions disappear
when the graph is rebuilt. `restore_function` activates it again.

```json
{"op":"delete_function","target_id":"fn_000004"}
{"op":"restore_function","target_id":"fn_000004"}
```

### `finish`

Signals that the current request needs no further scene change.

```json
{"op":"finish"}
```

## Roblox Lua used by the Studio interpreter

The plugin itself uses only these Roblox APIs after validation:

- `Instance.new("Part")` for visible geometry;
- `Instance.new("Motor6D")` for each rig connection;
- `Model.PrimaryPart` for the visible rig root;
- `Part.CFrame`, `Part.Size`, `Part.Color`, `Part.Material`, `Anchored`, and
  `CanCollide` for allowed part properties;
- `Motor6D.C0`, `Motor6D.C1`, and documented `Motor6D` attributes for the rest
  joint pose and generated movement contract.

The interpreter intentionally does not execute Lua supplied by the model.
