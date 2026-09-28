"""The closed language boundary between a text model and Roblox Studio.

The model may emit JSON only. Validate every object with this module before it
can mutate a SceneSession or be sent to the Studio interpreter.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .core import ALLOWED_MATERIALS, ALLOWED_SHAPES, EDITABLE_PART_PROPERTIES


EDIT_ALIASES = {
    "set_material": "material", "set_color": "color", "set_transparency": "transparency",
    "set_size": "size", "set_position": "position", "set_rotation": "rotation",
    "set_anchored": "anchored", "set_can_collide": "can_collide",
}


SCENE_FUNCTION_BNF = """scene_function ::= add_part | set_properties | delete | restore | finish
add_part       ::= {"op":"add_part","name":string,"shape":shape,"position":vec3,
                    "rotation":vec3,"size":positive_vec3,"color":rgb,"material":material,
                    "transparency":unit,"anchored":boolean,"can_collide":boolean}
set_properties ::= {"op":"set_part_properties","target_id":id,"properties":property_map}
delete         ::= {"op":"delete_function","target_id":id}
restore        ::= {"op":"restore_function","target_id":id}
finish         ::= {"op":"finish"}
vec3           ::= [number,number,number]
positive_vec3  ::= [positive_number,positive_number,positive_number]
rgb            ::= [unit,unit,unit]
unit           ::= number from 0 through 1
shape          ::= one of Ball | Block | Cylinder | Wedge | CornerWedge
material       ::= one of the Roblox Enum.Material names in JSON_SCHEMA
"""


def _vector(value: Any, name: str, *, positive: bool = False, unit: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-number array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain JSON numbers")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} values must be finite")
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    if unit and any(item < 0 or item > 1 for item in result):
        raise ValueError(f"{name} values must be in [0, 1]")
    return result


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def validate_generated_call(value: Any) -> dict[str, Any]:
    """Return a normalized safe call or raise ValueError. Unknown fields fail."""
    if not isinstance(value, dict) or not isinstance(value.get("op"), str):
        raise ValueError("a scene function must be an object with string op")
    operation = value["op"]
    if operation == "add_part":
        allowed = {"op", "name", "shape", "position", "rotation", "size", "color", "material",
                   "transparency", "anchored", "can_collide", "depends_on"}
        required = allowed - {"rotation", "material", "transparency", "anchored", "can_collide", "depends_on"}
        if set(value) - allowed or required - set(value):
            raise ValueError("add_part has missing or unsupported fields")
        if (not isinstance(value["name"], str) or not value["name"].strip()
                or value["shape"] not in ALLOWED_SHAPES
                or value.get("material", "Plastic") not in ALLOWED_MATERIALS):
            raise ValueError("unknown Roblox shape or material")
        depends_on = value.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            raise ValueError("depends_on must be an array of function IDs")
        for key, default in (("anchored", True), ("can_collide", True)):
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f"{key} must be boolean")
        return {
            "op": operation, "name": value["name"], "shape": value["shape"],
            "position": _vector(value["position"], "position"),
            "rotation": _vector(value.get("rotation", [0, 0, 0]), "rotation"),
            "size": _vector(value["size"], "size", positive=True),
            "color": _vector(value["color"], "color", unit=True),
            "material": value.get("material", "Plastic"),
            "transparency": _unit(value.get("transparency", 0), "transparency"),
            "anchored": value.get("anchored", True),
            "can_collide": value.get("can_collide", True), "depends_on": depends_on,
        }
    if operation in EDIT_ALIASES:
        property_name = EDIT_ALIASES[operation]
        if set(value) != {"op", "target_id", property_name}:
            raise ValueError(f"{operation} must include only target_id and {property_name}")
        return validate_generated_call({"op": "set_part_properties", "target_id": value["target_id"],
                                        "properties": {property_name: value[property_name]}})
    if operation == "set_part_properties":
        if (set(value) != {"op", "target_id", "properties"}
                or not isinstance(value["target_id"], str) or not value["target_id"]):
            raise ValueError("set_part_properties requires target_id and properties")
        properties = value["properties"]
        if not isinstance(properties, dict) or not properties or set(properties) - EDITABLE_PART_PROPERTIES:
            raise ValueError("properties must be a non-empty supported property map")
        checked: dict[str, Any] = {}
        for key, item in properties.items():
            if key in {"position", "rotation"}:
                checked[key] = _vector(item, key)
            elif key == "size":
                checked[key] = _vector(item, key, positive=True)
            elif key == "color":
                checked[key] = _vector(item, key, unit=True)
            elif key == "transparency":
                checked[key] = _unit(item, key)
            elif key == "material":
                if not isinstance(item, str) or item not in ALLOWED_MATERIALS:
                    raise ValueError("unknown Roblox material")
                checked[key] = item
            elif key == "shape":
                if not isinstance(item, str) or item not in ALLOWED_SHAPES:
                    raise ValueError("unknown Roblox shape")
                checked[key] = item
            elif key in {"anchored", "can_collide"}:
                if not isinstance(item, bool):
                    raise ValueError(f"{key} must be boolean")
                checked[key] = item
            else:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("name must be a non-empty string")
                checked[key] = item
        return {"op": operation, "target_id": value["target_id"], "properties": checked}
    if operation in {"delete_function", "restore_function"}:
        if (set(value) != {"op", "target_id"}
                or not isinstance(value["target_id"], str) or not value["target_id"]):
            raise ValueError(f"{operation} requires only target_id")
        return {"op": operation, "target_id": value["target_id"]}
    if operation == "finish" and set(value) == {"op"}:
        return {"op": "finish"}
    raise ValueError(f"unsupported scene function: {operation}")


def parse_generated_json(text: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_non_standard_number(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    return validate_generated_call(json.loads(
        text, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_non_standard_number
    ))


def _vector_schema(item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "minItems": 3, "maxItems": 3, "items": item_schema}


NUMBER_VECTOR = _vector_schema({"type": "number"})
POSITIVE_VECTOR = _vector_schema({"type": "number", "exclusiveMinimum": 0})
COLOR_VECTOR = _vector_schema({"type": "number", "minimum": 0, "maximum": 1})
PART_PROPERTY_SCHEMA = {
    "type": "object", "minProperties": 1, "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1}, "shape": {"enum": sorted(ALLOWED_SHAPES)},
        "position": NUMBER_VECTOR, "rotation": NUMBER_VECTOR, "size": POSITIVE_VECTOR,
        "color": COLOR_VECTOR, "material": {"enum": sorted(ALLOWED_MATERIALS)},
        "transparency": {"type": "number", "minimum": 0, "maximum": 1},
        "anchored": {"type": "boolean"}, "can_collide": {"type": "boolean"},
    },
}

# This exact union is directly usable by guided-JSON backends. The aliases are
# accepted by the Python validator for convenience, but the model emits only
# canonical operations so its token grammar remains compact.
JSON_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": ["op", "name", "shape", "position", "rotation", "size", "color", "material",
                         "transparency", "anchored", "can_collide"],
            "properties": {
                "op": {"const": "add_part"}, "name": {"type": "string", "minLength": 1},
                "shape": {"enum": sorted(ALLOWED_SHAPES)}, "position": NUMBER_VECTOR,
                "rotation": NUMBER_VECTOR, "size": POSITIVE_VECTOR, "color": COLOR_VECTOR,
                "material": {"enum": sorted(ALLOWED_MATERIALS)},
                "transparency": {"type": "number", "minimum": 0, "maximum": 1},
                "anchored": {"type": "boolean"}, "can_collide": {"type": "boolean"},
                "depends_on": {"type": "array", "items": {"type": "string"}},
            },
        },
        {
            "type": "object", "additionalProperties": False,
            "required": ["op", "target_id", "properties"],
            "properties": {"op": {"const": "set_part_properties"}, "target_id": {"type": "string"},
                           "properties": PART_PROPERTY_SCHEMA},
        },
        *[
            {"type": "object", "additionalProperties": False, "required": ["op", "target_id"],
             "properties": {"op": {"const": operation}, "target_id": {"type": "string"}}}
            for operation in ("delete_function", "restore_function")
        ],
        {"type": "object", "additionalProperties": False, "required": ["op"],
         "properties": {"op": {"const": "finish"}}},
    ],
}
