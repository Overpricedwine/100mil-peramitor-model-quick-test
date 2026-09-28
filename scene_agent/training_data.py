"""Deterministic seed data for supervised scene-function training.

This produces *known-correct* construction traces. A local request-author
model may paraphrase the supplied scene specification later, but it never
creates the target function trace. That separation prevents a grader from
teaching the generator its own mistakes.
"""

from __future__ import annotations

import json
import random
from typing import Any, Iterable


def _part(
    name: str, position: list[float], size: list[float], color: list[float],
    material: str = "Plastic", shape: str = "Block", rotation: list[float] | None = None,
    transparency: float = 0,
) -> dict[str, Any]:
    return {
        "op": "add_part", "name": name, "shape": shape,
        "position": position, "rotation": rotation or [0, 0, 0], "size": size,
        "color": color, "material": material, "transparency": transparency,
        "anchored": True, "can_collide": True,
    }


def make_seed_scene(index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a concise spec plus a valid sequence of canonical functions."""
    rng = random.Random(index)
    kind = (
        "tower", "bridge", "arch", "house", "vehicle", "furniture",
        "crate", "street_lamp", "tree", "sword", "fountain", "rock_cluster",
        "robot", "dog", "bird", "crab",
    )[index % 16]
    hue = [round(rng.uniform(0.15, 0.9), 2) for _ in range(3)]
    accent = [round(1 - channel * 0.55, 2) for channel in hue]
    if kind == "tower":
        height = rng.choice([12, 16, 20])
        functions = [
            _part("TowerBase", [0, 1, 0], [8, 2, 8], hue, "Brick"),
            _part("TowerBody", [0, (height + 2) / 2, 0], [5, height, 5], hue, "Brick"),
            _part("TowerRoof", [0, height + 3, 0], [7, 2, 7], accent, "Metal", "Wedge"),
            _part("Beacon", [0, height + 5, 0], [2, 2, 2], accent, "Neon", "Ball"),
        ]
        spec = {"kind": kind, "height": height, "primary_color": hue, "accent_color": accent,
                "parts": ["brick base", "brick tower body", "metal wedge roof", "neon beacon"]}
    elif kind == "bridge":
        span = rng.choice([16, 20, 24])
        functions = [
            _part("BridgeDeck", [0, 5, 0], [span, 2, 6], hue, "Wood"),
            _part("LeftPillar", [-span / 2 + 2, 2.5, 0], [3, 5, 4], accent, "Concrete"),
            _part("RightPillar", [span / 2 - 2, 2.5, 0], [3, 5, 4], accent, "Concrete"),
            _part("LeftRail", [0, 7, -2.5], [span, 2, 0.5], accent, "Metal"),
            _part("RightRail", [0, 7, 2.5], [span, 2, 0.5], accent, "Metal"),
        ]
        spec = {"kind": kind, "span": span, "deck_color": hue, "support_color": accent,
                "parts": ["wooden deck", "two concrete pillars", "two metal rails"]}
    elif kind == "arch":
        width = rng.choice([10, 14, 18])
        functions = [
            _part("LeftSupport", [-width / 2, 4, 0], [3, 8, 3], hue, "Granite"),
            _part("RightSupport", [width / 2, 4, 0], [3, 8, 3], hue, "Granite"),
            _part("TopBeam", [0, 8, 0], [width + 3, 3, 3], hue, "Granite"),
            _part("CenterLight", [0, 6, 0], [1.5, 1.5, 1.5], accent, "Neon", "Ball"),
        ]
        spec = {"kind": kind, "width": width, "stone_color": hue, "light_color": accent,
                "parts": ["two stone supports", "top stone beam", "neon center light"]}
    elif kind == "house":
        width = rng.choice([10, 14, 18])
        functions = [
            _part("HouseBase", [0, 1, 0], [width, 2, 10], hue, "Brick"),
            _part("HouseBody", [0, 5, 0], [width - 2, 6, 8], hue, "WoodPlanks"),
            _part("HouseRoof", [0, 9, 0], [width + 1, 3, 10], accent, "ClayRoofTiles", "Wedge"),
            _part("Door", [0, 3, -4.1], [2, 4, 0.3], accent, "Wood"),
        ]
        spec = {"kind": kind, "width": width, "primary_color": hue, "accent_color": accent,
                "parts": ["brick base", "wooden house body", "wedge roof", "front door"]}
    elif kind == "vehicle":
        functions = [
            _part("VehicleBody", [0, 2, 0], [10, 3, 5], hue, "Metal"),
            _part("VehicleCabin", [1, 4.5, 0], [4, 2, 4], accent, "Glass"),
            _part("FrontLeftWheel", [-3, 1, -2.7], [2, 2, 1], [0.08, 0.08, 0.08], "Rubber", "Cylinder", [90, 0, 0]),
            _part("FrontRightWheel", [-3, 1, 2.7], [2, 2, 1], [0.08, 0.08, 0.08], "Rubber", "Cylinder", [90, 0, 0]),
            _part("RearLeftWheel", [3, 1, -2.7], [2, 2, 1], [0.08, 0.08, 0.08], "Rubber", "Cylinder", [90, 0, 0]),
            _part("RearRightWheel", [3, 1, 2.7], [2, 2, 1], [0.08, 0.08, 0.08], "Rubber", "Cylinder", [90, 0, 0]),
        ]
        spec = {"kind": kind, "body_color": hue, "cabin_color": accent,
                "parts": ["metal vehicle body", "glass cabin", "four rubber wheels"]}
    elif kind == "furniture":
        functions = [
            _part("TableTop", [0, 5, 0], [10, 1, 6], hue, "Wood"),
            _part("LeftLeg", [-4, 2.5, -2], [1, 5, 1], accent, "Wood"),
            _part("RightLeg", [4, 2.5, -2], [1, 5, 1], accent, "Wood"),
            _part("BackLeg", [0, 2.5, 2], [1, 5, 1], accent, "Wood"),
        ]
        spec = {"kind": kind, "top_color": hue, "leg_color": accent,
                "parts": ["wooden table top", "three visible supporting legs"]}
    elif kind == "crate":
        functions = [
            _part("CrateBody", [0, 2.5, 0], [6, 5, 6], hue, "WoodPlanks"),
            _part("FrontBrace", [0, 2.5, -3.1], [5.5, 0.6, 0.3], accent, "Metal", rotation=[0, 0, 35]),
            _part("BackBrace", [0, 2.5, 3.1], [5.5, 0.6, 0.3], accent, "Metal", rotation=[0, 0, -35]),
        ]
        spec = {"kind": kind, "primary_color": hue, "accent_color": accent,
                "parts": ["wooden crate body", "two diagonal metal braces"]}
    elif kind == "street_lamp":
        functions = [
            _part("LampBase", [0, 0.5, 0], [3, 1, 3], accent, "Concrete"),
            _part("LampPole", [0, 7, 0], [1, 12, 1], hue, "Metal", "Cylinder"),
            _part("LampHead", [0, 13, 0], [3, 1, 3], hue, "Metal"),
            _part("LampGlow", [0, 12.2, 0], [1.5, 1.5, 1.5], accent, "Neon", "Ball"),
        ]
        spec = {"kind": kind, "pole_color": hue, "light_color": accent,
                "parts": ["concrete base", "metal pole", "lamp head", "glowing ball light"]}
    elif kind == "tree":
        functions = [
            _part("Trunk", [0, 4, 0], [2, 8, 2], [0.28, 0.15, 0.06], "Wood", "Cylinder"),
            _part("CanopyLow", [0, 9, 0], [7, 5, 7], hue, "LeafyGrass", "Ball"),
            _part("CanopyLeft", [-2.5, 11, 0], [5, 5, 5], accent, "LeafyGrass", "Ball"),
            _part("CanopyRight", [2.5, 11, 0], [5, 5, 5], accent, "LeafyGrass", "Ball"),
        ]
        spec = {"kind": kind, "leaf_colors": [hue, accent],
                "parts": ["wooden trunk", "three simple leafy ball canopies"]}
    elif kind == "sword":
        functions = [
            _part("Blade", [0, 7, 0], [1, 10, 0.6], hue, "Metal"),
            _part("Tip", [0, 12.2, 0], [2, 1.5, 0.8], hue, "Metal", "Wedge"),
            _part("Guard", [0, 2.2, 0], [6, 0.7, 1], accent, "Metal"),
            _part("Grip", [0, 0.5, 0], [1.2, 3, 1.2], [0.1, 0.06, 0.03], "Leather", "Cylinder"),
        ]
        spec = {"kind": kind, "blade_color": hue, "guard_color": accent,
                "parts": ["metal blade", "wedge blade tip", "wide guard", "leather grip"]}
    elif kind == "fountain":
        functions = [
            _part("Basin", [0, 1, 0], [12, 2, 12], hue, "Concrete"),
            _part("Water", [0, 2.1, 0], [9, 0.4, 9], accent, "Glass", transparency=0.35),
            _part("CenterPillar", [0, 5, 0], [2, 6, 2], hue, "Marble"),
            _part("TopBowl", [0, 8, 0], [5, 1, 5], accent, "Marble", "Cylinder"),
            _part("WaterJet", [0, 10, 0], [0.7, 4, 0.7], accent, "Glass", "Cylinder", transparency=0.2),
        ]
        spec = {"kind": kind, "stone_color": hue, "water_color": accent,
                "parts": ["concrete basin", "transparent water surface", "center pillar", "top bowl", "water jet"]}
    elif kind == "rock_cluster":
        functions = [
            _part("MainRock", [0, 1.5, 0], [6, 3, 5], hue, "Rock", "Ball"),
            _part("LeftRock", [-4, 1, 1], [3, 2, 3], accent, "Rock", "Ball"),
            _part("RightRock", [4, 0.8, -1], [2.5, 1.6, 3], accent, "Rock", "Ball"),
            _part("GrassPatch", [0, 0.15, 0], [12, 0.3, 10], [0.16, 0.45, 0.18], "Grass"),
        ]
        spec = {"kind": kind, "rock_color": hue, "accent_color": accent,
                "parts": ["three low-poly rocks", "flat grass patch"]}
    elif kind == "robot":
        functions = [
            _part("RobotTorso", [0, 6, 0], [4, 5, 2.5], hue, "Metal"),
            _part("RobotHead", [0, 10, 0], [3.5, 3, 3], accent, "Metal"),
            _part("RobotEye", [0, 10, -1.6], [1, 0.8, 0.3], [0.2, 0.9, 1], "Neon"),
            _part("LeftLeg", [-1.2, 2, 0], [1.2, 4, 1.5], hue, "Metal"),
            _part("RightLeg", [1.2, 2, 0], [1.2, 4, 1.5], hue, "Metal"),
            _part("LeftArm", [-3, 6, 0], [1, 4, 1], accent, "Metal", rotation=[0, 0, 12]),
            _part("RightArm", [3, 6, 0], [1, 4, 1], accent, "Metal", rotation=[0, 0, -12]),
        ]
        spec = {"kind": kind, "body_color": hue, "accent_color": accent,
                "parts": ["metal torso", "square head", "neon eye", "two legs", "two angled arms"]}
    elif kind == "dog":
        functions = [
            _part("DogBody", [0, 4, 0], [7, 4, 3], hue, "SmoothPlastic"),
            _part("DogHead", [-4, 5, 0], [3, 3, 3], hue, "SmoothPlastic"),
            _part("LeftEar", [-4, 7.2, -0.9], [1, 2, 1], accent, "SmoothPlastic", "Wedge", [0, 0, 20]),
            _part("RightEar", [-4, 7.2, 0.9], [1, 2, 1], accent, "SmoothPlastic", "Wedge", [0, 0, -20]),
            _part("FrontLeftLeg", [-2.5, 1.5, -1], [1, 3, 1], accent, "SmoothPlastic"),
            _part("FrontRightLeg", [-2.5, 1.5, 1], [1, 3, 1], accent, "SmoothPlastic"),
            _part("BackLeftLeg", [2.5, 1.5, -1], [1, 3, 1], accent, "SmoothPlastic"),
            _part("BackRightLeg", [2.5, 1.5, 1], [1, 3, 1], accent, "SmoothPlastic"),
            _part("Tail", [4.2, 5, 0], [1, 4, 1], accent, "SmoothPlastic", "Cylinder", [0, 0, -55]),
        ]
        spec = {"kind": kind, "fur_color": hue, "accent_color": accent,
                "parts": ["simple body", "head", "two wedge ears", "four-legged silhouette", "raised tail"]}
    elif kind == "bird":
        functions = [
            _part("BirdBody", [0, 5, 0], [4, 5, 4], hue, "SmoothPlastic", "Ball"),
            _part("BirdHead", [0, 8, 0], [3, 3, 3], hue, "SmoothPlastic", "Ball"),
            _part("Beak", [0, 8, -2], [1.5, 1, 2], accent, "Plastic", "Wedge", [90, 0, 0]),
            _part("LeftWing", [-3, 5.5, 0], [4, 1, 3], accent, "SmoothPlastic", "Wedge", [0, 0, 20]),
            _part("RightWing", [3, 5.5, 0], [4, 1, 3], accent, "SmoothPlastic", "Wedge", [0, 0, -20]),
            _part("LeftFoot", [-0.8, 1.5, 0], [0.7, 2, 0.7], [0.9, 0.5, 0.1], "Plastic"),
            _part("RightFoot", [0.8, 1.5, 0], [0.7, 2, 0.7], [0.9, 0.5, 0.1], "Plastic"),
        ]
        spec = {"kind": kind, "body_color": hue, "accent_color": accent,
                "parts": ["round body", "round head", "wedge beak", "two wings", "two feet"]}
    else:  # crab
        functions = [
            _part("CrabShell", [0, 3, 0], [6, 3, 5], hue, "SmoothPlastic", "Ball"),
            _part("LeftEye", [-1.3, 5.2, -1.2], [0.8, 1.4, 0.8], accent, "Plastic", "Ball"),
            _part("RightEye", [1.3, 5.2, -1.2], [0.8, 1.4, 0.8], accent, "Plastic", "Ball"),
            _part("LeftClaw", [-5, 3, 0], [3, 1.5, 2], accent, "SmoothPlastic", "Wedge", [0, 0, 25]),
            _part("RightClaw", [5, 3, 0], [3, 1.5, 2], accent, "SmoothPlastic", "Wedge", [0, 0, -25]),
            _part("LeftLeg", [-3, 1.5, 1], [4, 0.7, 0.7], hue, "SmoothPlastic", rotation=[0, 0, 20]),
            _part("RightLeg", [3, 1.5, 1], [4, 0.7, 0.7], hue, "SmoothPlastic", rotation=[0, 0, -20]),
        ]
        spec = {"kind": kind, "shell_color": hue, "accent_color": accent,
                "parts": ["rounded shell", "two raised eyes", "two wedge claws", "wide side legs"]}
    offset = [rng.choice([-24, -12, 0, 12, 24]), 0, rng.choice([-24, -12, 0, 12, 24])]
    for function in functions:
        function["position"] = [round(function["position"][axis] + offset[axis], 2) for axis in range(3)]
    spec["local_origin"] = offset
    return spec, functions


def split_for_scene(index: int) -> str:
    """Hold out whole template families, not random near-duplicate variants."""
    remainder = index % 16
    if remainder == 10:
        return "validation"  # fountain
    if remainder == 11:
        return "test"  # rock cluster
    return "train"


def _history_item(function_id: str, call: dict[str, Any], active: bool = True) -> dict[str, Any]:
    return {
        "id": function_id, "op": call["op"],
        "payload": {key: value for key, value in call.items() if key not in {"op", "depends_on"}},
        "depends_on": call.get("depends_on", []), "active": active,
    }


def records_for_scene(
    spec: dict[str, Any], functions: list[dict[str, Any]], request: str, *, scene_id: int = 0,
) -> Iterable[dict[str, str]]:
    """Yield construction, completion, and reversible-revision examples.

    Context deliberately mirrors `SceneSession.model_context`: IDs are visible
    before an edit/delete, including inactive history for a later restore.
    """
    history: list[dict[str, Any]] = []
    turns = [{"role": "user", "text": request}]
    split = split_for_scene(scene_id)

    def context() -> str:
        value = {
            "user_request": request, "scene_spec": spec,
            "active_function_ids": [item["id"] for item in history if item["active"]],
            "active_functions": [item for item in history if item["active"]],
            "function_history": history, "conversation": turns,
            "instruction": "Emit exactly one next canonical scene function.",
        }
        return json.dumps(value, separators=(",", ":"))

    def record(target: dict[str, Any], stage: str) -> dict[str, str]:
        return {
            "context": context(), "target": json.dumps(target, separators=(",", ":")),
            "scene_id": str(scene_id), "split": split, "stage": stage,
        }

    for function in functions:
        yield record(function, "build")
        history.append(_history_item(f"fn_{len(history) + 1:06d}", function))

    yield record({"op": "finish"}, "finish")

    # Every synthetic scene also supplies a three-turn revision conversation:
    # edit an existing part, remove an accent, then restore it.  This teaches
    # the model to use IDs from the saved current scene rather than rebuilding.
    primary_id = history[0]["id"]
    revision_color = [round(1 - channel, 2) for channel in spec.get("primary_color", [0.35, 0.35, 0.35])]
    turns.append({"role": "user", "text": "Revise the main part: use this new accent colour."})
    edit = {"op": "set_part_properties", "target_id": primary_id, "properties": {"color": revision_color}}
    yield record(edit, "revision_edit")
    history.append(_history_item(f"fn_{len(history) + 1:06d}", edit))

    accent_id = history[-2]["id"]
    turns.append({"role": "user", "text": "Remove the final accent part for a cleaner low-poly silhouette."})
    delete = {"op": "delete_function", "target_id": accent_id}
    yield record(delete, "revision_delete")
    for item in history:
        if item["id"] == accent_id:
            item["active"] = False

    turns.append({"role": "user", "text": "Restore that accent part; it is needed after all."})
    restore = {"op": "restore_function", "target_id": accent_id}
    yield record(restore, "revision_restore")


def deterministic_request(spec: dict[str, Any]) -> str:
    """Safe fallback while a local request-author checkpoint is unavailable."""
    return f"Build a low-poly {spec['kind']} model with " + ", ".join(spec["parts"]) + "."
