from __future__ import annotations

from typing import Any


def call_to_luau(call: dict[str, Any]) -> str:
    """Display a generated function as readable, constrained Luau-like code.

    The host should execute the JSON call through SceneSession.apply_generated_call,
    rather than execute this display text as arbitrary Luau.
    """
    operation = call["op"]
    if operation == "add_part":
        x, y, z = call["position"]
        sx, sy, sz = call["size"]
        r, g, b = call["color"]
        rx, ry, rz = call.get("rotation", (0, 0, 0))
        return (
            "scene:add_part({\n"
            f'  name = "{call["name"]}", shape = "{call["shape"]}",\n'
            f"  position = Vector3.new({x}, {y}, {z}),\n"
            f"  size = Vector3.new({sx}, {sy}, {sz}),\n"
            f"  color = Color3.new({r}, {g}, {b}),\n"
            f'  material = Enum.Material.{call.get("material", "Plastic")},\n'
            f"  transparency = {call.get('transparency', 0)},\n"
            f"  rotation = Vector3.new({rx}, {ry}, {rz}),\n"
            f"  anchored = {str(call.get('anchored', True)).lower()},\n"
            f"  canCollide = {str(call.get('can_collide', True)).lower()},\n"
            "})"
        )
    if operation == "set_part_properties":
        return f'scene:set_part_properties("{call["target_id"]}", {call["properties"]})'
    if operation.startswith("set_"):
        property_name = operation.removeprefix("set_")
        return f'scene:{operation}("{call["target_id"]}", {call[property_name]!r})'
    if operation in {"delete_function", "restore_function"}:
        return f'scene:{operation}("{call["target_id"]}")'
    if operation == "finish":
        return "scene:finish()"
    raise ValueError(f"unsupported operation {operation}")
