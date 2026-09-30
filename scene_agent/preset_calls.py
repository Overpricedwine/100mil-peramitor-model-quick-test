"""Numeric preset calls for safe, compact scene-function generation.

The learned policy emits only a preset ID and numbers separated by ``|``.  It
never writes a JSON member name, Roblox enum, function ID, or arbitrary Luau.
This module expands that compact call into the canonical JSON function that
the existing grammar and scene interpreter already validate.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .core import ALLOWED_MATERIALS, ALLOWED_SHAPES
from .grammar import validate_generated_call


SHAPES = tuple(sorted(ALLOWED_SHAPES))
MATERIALS = tuple(sorted(ALLOWED_MATERIALS))
ROLES = ("body", "limb", "accessory")
MOTIONS = ("fixed", "hinge", "universal")

# 0=root, 1=rig child, 2=colour edit, 3=delete, 4=restore, 5=finish, 6=static part.
PRESET_NAMES = ("rig_root", "rig_part", "set_color", "delete", "restore", "finish", "static_part")


def _context(value: str | dict[str, Any]) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value


def _format(value: float | int) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("preset values must be finite")
    return f"{number:.6g}"


def _numbers(call: list[float | int]) -> str:
    return "|".join(_format(value) for value in call)


def _reference_number(function_id: str) -> int:
    if function_id.startswith("fn_") and function_id[3:].isdigit():
        return int(function_id[3:])
    raise ValueError(f"preset calls require canonical function IDs, not {function_id!r}")


def encode_preset_call(call: dict[str, Any], context: str | dict[str, Any]) -> str:
    """Encode a known-valid canonical call using only a preset ID and numbers."""
    call = validate_generated_call(call)
    operation = call["op"]
    if operation in {"add_rig_root", "add_part", "add_rig_part"}:
        common = [
            SHAPES.index(call["shape"]), MATERIALS.index(call["material"]),
            *call["position"], *call["rotation"], *call["size"], *call["color"], call["transparency"],
        ]
    if operation == "add_rig_root":
        return _numbers([0, *common])
    if operation == "add_part":
        return _numbers([6, *common, int(call["anchored"]), int(call["can_collide"])])
    if operation == "add_rig_part":
        motion = call["motion_type"]
        axes = [value for axis in call["axes"] for value in axis]
        ranges = [value for angle_range in call["angle_ranges"] for value in angle_range]
        axes += [0] * (6 - len(axes))
        ranges += [0] * (4 - len(ranges))
        return _numbers([
            1, *common, _reference_number(call["parent_part_id"]), *call["joint_position"],
            ROLES.index(call["role"]), MOTIONS.index(motion), *axes, *ranges,
        ])
    if operation == "set_part_properties":
        properties = call["properties"]
        if set(properties) != {"color"}:
            raise ValueError("numeric presets currently support colour edits only")
        return _numbers([2, _reference_number(call["target_id"]), *properties["color"]])
    if operation == "delete_function":
        return _numbers([3, _reference_number(call["target_id"])])
    if operation == "restore_function":
        return _numbers([4, _reference_number(call["target_id"])])
    if operation == "finish":
        return "5"
    raise ValueError(f"unsupported preset operation: {operation}")


def is_preset_prefix(text: str) -> bool:
    """Check whether text can still become one compact numeric preset call."""
    if not text:
        return True
    if text[0] not in "0123456":
        return False
    if any(character not in "0123456789|+-.eE" for character in text):
        return False
    fields = text.split("|")
    if len(fields) > 1 and fields[0] not in {str(index) for index in range(7)}:
        return False
    for field in fields[1:]:
        if field and not _is_number_prefix(field):
            return False
    return True


def _is_number_prefix(value: str) -> bool:
    # A permissive incremental number check; ``decode_preset_call`` performs
    # exact conversion and range checking when a call is complete.
    if value in {"-", "+", ".", "-.", "+."}:
        return True
    try:
        float(value)
    except ValueError:
        return value.endswith(("e", "E", "e+", "e-", "E+", "E-")) and _is_number_prefix(value[:-1])
    return True


def _read(text: str) -> list[float]:
    if not text or text.endswith("|"):
        raise ValueError("preset call is incomplete")
    try:
        values = [float(field) for field in text.split("|")]
    except ValueError as error:
        raise ValueError("preset fields must be numbers") from error
    if any(not math.isfinite(value) for value in values):
        raise ValueError("preset fields must be finite")
    return values


def _integer(value: float, label: str, upper: int) -> int:
    result = int(value)
    if value != result or not 0 <= result < upper:
        raise ValueError(f"{label} is outside its preset range")
    return result


def _vector(values: list[float], start: int) -> list[float]:
    return values[start:start + 3]


def _function_ids(context: dict[str, Any]) -> list[str]:
    ids = context.get("active_function_ids")
    if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
        return ids
    return [item["id"] for item in context.get("function_history", []) if item.get("active", True)]


def _reference(context: dict[str, Any], value: float) -> str:
    index = _integer(value, "function reference", 1_000_000)
    ids = _function_ids(context)
    if not 1 <= index <= len(ids):
        raise ValueError("preset references a part that is not in the current scene")
    return ids[index - 1]


def _common(values: list[float], context: dict[str, Any], *, part_name: str) -> dict[str, Any]:
    if len(values) < 16:
        raise ValueError("part preset needs 16 numeric fields")
    shape = SHAPES[_integer(values[1], "shape", len(SHAPES))]
    material = MATERIALS[_integer(values[2], "material", len(MATERIALS))]
    return {
        "name": part_name, "shape": shape, "material": material,
        "position": _vector(values, 3), "rotation": _vector(values, 6),
        "size": _vector(values, 9), "color": _vector(values, 12), "transparency": values[15],
    }


def decode_preset_call(text: str, context: str | dict[str, Any]) -> dict[str, Any]:
    """Expand one complete numeric preset into a normal grammar-checked call."""
    values = _read(text)
    preset = _integer(values[0], "preset", len(PRESET_NAMES))
    ctx = _context(context)
    spec = ctx.get("scene_spec", {})
    current_count = len(_function_ids(ctx))
    if preset == 5:
        if len(values) != 1:
            raise ValueError("finish takes no parameters")
        return validate_generated_call({"op": "finish"})
    if preset in {3, 4}:
        if len(values) != 2:
            raise ValueError("delete and restore take one reference")
        return validate_generated_call({"op": "delete_function" if preset == 3 else "restore_function",
                                        "target_id": _reference(ctx, values[1])})
    if preset == 2:
        if len(values) != 5:
            raise ValueError("colour edit needs a reference and RGB values")
        return validate_generated_call({"op": "set_part_properties", "target_id": _reference(ctx, values[1]),
                                        "properties": {"color": values[2:5]}})
    if preset == 0:
        if len(values) != 16:
            raise ValueError("rig-root preset needs 16 fields")
        name = str(spec.get("primary_part", "RigRoot"))
        return validate_generated_call({"op": "add_rig_root", **_common(values, ctx, part_name=name)})
    if preset == 6:
        if len(values) != 18:
            raise ValueError("static-part preset needs 18 fields")
        call = {"op": "add_part", **_common(values, ctx, part_name=f"StaticPart{current_count + 1}"),
                "anchored": bool(_integer(values[16], "anchored", 2)),
                "can_collide": bool(_integer(values[17], "can_collide", 2))}
        return validate_generated_call(call)
    if len(values) != 32:
        raise ValueError("rig-part preset needs 32 fields")
    parent = _reference(ctx, values[16])
    role = ROLES[_integer(values[20], "role", len(ROLES))]
    motion = MOTIONS[_integer(values[21], "motion", len(MOTIONS))]
    axis_values, range_values = values[22:28], values[28:32]
    axis_count = {"fixed": 0, "hinge": 1, "universal": 2}[motion]
    axes = [axis_values[index * 3:(index + 1) * 3] for index in range(axis_count)]
    ranges = [range_values[index * 2:(index + 1) * 2] for index in range(axis_count)]
    name = f"RigPart{current_count + 1}"
    call = {
        "op": "add_rig_part", **_common(values, ctx, part_name=name), "parent_part_id": parent,
        "joint_name": f"{name}Joint", "joint_position": _vector(values, 17), "role": role,
        "motion_type": motion, "axes": axes, "angle_ranges": ranges,
    }
    return validate_generated_call(call)


class NumericPresetLogitsProcessor:
    """Constrain decoding to a complete numeric call for one saved scene."""

    def __init__(self, tokenizer, prompt_length: int, context: str | dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.context = context
        self._pieces = [tokenizer.decode([token_id]) for token_id in range(tokenizer.vocab_size)]
        special = ("<pad>", "<bos>", "<eos>", "<scene>", "</scene>", "<function>")
        self._special_ids = {tokenizer.tokenizer.token_to_id(token) for token in special}

    def _complete(self, text: str) -> bool:
        try:
            decode_preset_call(text, self.context)
        except (TypeError, ValueError):
            return False
        return True

    def __call__(self, logits, token_ids):
        if logits.shape[0] != 1:
            raise ValueError("numeric preset decoding supports batch size 1")
        import torch

        generated = token_ids[0, self.prompt_length:].tolist()
        text = self.tokenizer.decode(generated)
        if self._complete(text):
            allowed = [self.tokenizer.eos_id]
        else:
            usable = logits[:, :len(self._pieces)]
            candidates = torch.topk(usable, min(512, usable.shape[-1]), dim=-1).indices[0].tolist()
            allowed = self._allowed(text, candidates)
            if not allowed:
                allowed = self._allowed(text, range(len(self._pieces)))
        if not allowed:
            raise ValueError("generation reached a numeric-preset dead end")
        constrained = torch.full_like(logits, float("-inf"))
        constrained[:, allowed] = logits[:, allowed]
        return constrained

    def _allowed(self, text: str, candidates) -> list[int]:
        allowed: list[int] = []
        for token_id in candidates:
            if token_id in self._special_ids:
                continue
            piece = self._pieces[token_id]
            if piece and is_preset_prefix(text + piece):
                allowed.append(token_id)
        return allowed
