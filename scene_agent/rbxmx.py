from __future__ import annotations

import math
from html import escape
from pathlib import Path

from .core import Part, SceneGraph


PART_TYPE_VALUES = {"Ball": 0, "Block": 1, "Cylinder": 2, "Wedge": 3, "CornerWedge": 4}
MATERIAL_VALUES = {
    "Plastic": 256, "SmoothPlastic": 272, "Neon": 288, "Wood": 512, "WoodPlanks": 528,
    "Marble": 784, "Basalt": 788, "Slate": 800, "CrackedLava": 804, "Concrete": 816,
    "Limestone": 820, "Granite": 832, "Pavement": 836, "Brick": 848, "Pebble": 864,
    "Cobblestone": 880, "Rock": 896, "Sandstone": 912, "CorrodedMetal": 1040,
    "DiamondPlate": 1056, "Foil": 1072, "Metal": 1088, "Grass": 1280, "LeafyGrass": 1284,
    "Sand": 1296, "Fabric": 1312, "Snow": 1328, "Mud": 1344, "Ground": 1360,
    "Asphalt": 1376, "Salt": 1392, "Ice": 1536, "Glacier": 1552, "Glass": 1568,
    "ForceField": 1584, "Cardboard": 2304, "Carpet": 2305, "CeramicTiles": 2306,
    "ClayRoofTiles": 2307, "RoofShingles": 2308, "Leather": 2309, "Plaster": 2310,
    "Rubber": 2311,
}


def _rotation_matrix(rotation: tuple[float, float, float]) -> tuple[float, ...]:
    """Roblox CFrame rotation matrix using X, then Y, then Z Euler degrees."""
    x, y, z = (math.radians(value) for value in rotation)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return (
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx,
    )


def _part_xml(part: Part) -> str:
    # Minimal Part subset. Always validate the resulting file in Studio.
    x, y, z = part.position
    sx, sy, sz = part.size
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _rotation_matrix(part.rotation)
    r, g, b = (round(component * 255) for component in part.color)
    packed = (255 << 24) | (r << 16) | (g << 8) | b
    return f'''  <Item class="Part" referent="RBX{part.function_id}">
    <Properties>
      <bool name="Anchored">{str(part.anchored).lower()}</bool>
      <bool name="CanCollide">{str(part.can_collide).lower()}</bool>
      <string name="Name">{escape(part.name)}</string>
      <Color3uint8 name="Color3">{packed}</Color3uint8>
      <token name="Material">{MATERIAL_VALUES[part.material]}</token>
      <token name="shape">{PART_TYPE_VALUES[part.shape]}</token>
      <float name="Transparency">{part.transparency}</float>
      <vector3 name="size"><X>{sx}</X><Y>{sy}</Y><Z>{sz}</Z></vector3>
      <CoordinateFrame name="CFrame"><X>{x}</X><Y>{y}</Y><Z>{z}</Z><R00>{r00}</R00><R01>{r01}</R01><R02>{r02}</R02><R10>{r10}</R10><R11>{r11}</R11><R12>{r12}</R12><R20>{r20}</R20><R21>{r21}</R21><R22>{r22}</R22></CoordinateFrame>
    </Properties>
  </Item>'''


def to_rbxmx(graph: SceneGraph) -> str:
    body = "\n".join(_part_xml(part) for part in graph.parts)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<roblox version="4">
<Item class="Model" referent="RBXSceneAgent">
  <Properties><string name="Name">GeneratedScene</string></Properties>
{body}
</Item>
</roblox>
'''


def write_rbxmx(graph: SceneGraph, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_rbxmx(graph), encoding="utf-8")
    return path
