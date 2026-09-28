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
    material: str = "Plastic", shape: str = "Block",
) -> dict[str, Any]:
    return {
        "op": "add_part", "name": name, "shape": shape,
        "position": position, "rotation": [0, 0, 0], "size": size,
        "color": color, "material": material, "transparency": 0,
        "anchored": True, "can_collide": True,
    }


def make_seed_scene(index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a concise spec plus a valid sequence of canonical functions."""
    rng = random.Random(index)
    kind = ("tower", "bridge", "arch", "house", "vehicle", "furniture")[index % 6]
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
            _part("LeftSupport", [-width / 2, 4, 0], [3, 8, 3], hue, "Stone"),
            _part("RightSupport", [width / 2, 4, 0], [3, 8, 3], hue, "Stone"),
            _part("TopBeam", [0, 8, 0], [width + 3, 3, 3], hue, "Stone"),
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
            _part("FrontWheel", [-3, 1, -2.7], [2, 2, 1], [0.08, 0.08, 0.08], "Rubber", "Cylinder"),
            _part("RearWheel", [3, 1, -2.7], [2, 2, 1], [0.08, 0.08, 0.08], "Rubber", "Cylinder"),
        ]
        spec = {"kind": kind, "body_color": hue, "cabin_color": accent,
                "parts": ["metal vehicle body", "glass cabin", "two visible rubber wheels"]}
    else:
        functions = [
            _part("TableTop", [0, 5, 0], [10, 1, 6], hue, "Wood"),
            _part("LeftLeg", [-4, 2.5, -2], [1, 5, 1], accent, "Wood"),
            _part("RightLeg", [4, 2.5, -2], [1, 5, 1], accent, "Wood"),
            _part("BackLeg", [0, 2.5, 2], [1, 5, 1], accent, "Wood"),
        ]
        spec = {"kind": kind, "top_color": hue, "leg_color": accent,
                "parts": ["wooden table top", "three visible supporting legs"]}
    return spec, functions


def records_for_scene(spec: dict[str, Any], functions: list[dict[str, Any]], request: str) -> Iterable[dict[str, str]]:
    """Yield one next-function target per construction step."""
    completed: list[dict[str, Any]] = []
    for function in functions:
        context = {
            "user_request": request,
            "scene_spec": spec,
            "completed_functions": completed,
            "instruction": "Emit exactly one next canonical scene function.",
        }
        yield {
            "context": json.dumps(context, separators=(",", ":")),
            "target": json.dumps(function, separators=(",", ":")),
        }
        completed.append(function)


def deterministic_request(spec: dict[str, Any]) -> str:
    """Safe fallback while a local request-author checkpoint is unavailable."""
    return f"Build a low-poly {spec['kind']} model with " + ", ".join(spec["parts"]) + "."
