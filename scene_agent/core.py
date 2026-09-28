from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable


ALLOWED_SHAPES = {"Ball", "Block", "Cylinder", "Wedge", "CornerWedge"}
ALLOWED_MATERIALS = {
    "Plastic", "SmoothPlastic", "Neon", "Wood", "WoodPlanks", "Marble", "Basalt", "Slate",
    "CrackedLava", "Concrete", "Limestone", "Granite", "Pavement", "Brick", "Pebble",
    "Cobblestone", "Rock", "Sandstone", "CorrodedMetal", "DiamondPlate", "Foil", "Metal",
    "Grass", "LeafyGrass", "Sand", "Fabric", "Snow", "Mud", "Ground", "Asphalt", "Salt",
    "Ice", "Glacier", "Glass", "ForceField", "Cardboard", "Carpet", "CeramicTiles",
    "ClayRoofTiles", "RoofShingles", "Leather", "Plaster", "Rubber",
}
EDITABLE_PART_PROPERTIES = {
    "name", "shape", "position", "rotation", "size", "color", "material", "transparency",
    "anchored", "can_collide",
}


@dataclass
class SceneFunction:
    """One model-generated, independently reversible piece of scene code."""

    id: str
    op: str
    payload: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    active: bool = True
    created_at: float = field(default_factory=time.time)

    @classmethod
    def add_part(
        cls,
        name: str,
        shape: str,
        position: tuple[float, float, float],
        size: tuple[float, float, float],
        color: tuple[float, float, float],
        depends_on: Iterable[str] = (),
        rotation: tuple[float, float, float] = (0, 0, 0),
        material: str = "Plastic",
        transparency: float = 0.0,
        anchored: bool = True,
        can_collide: bool = True,
    ) -> "SceneFunction":
        if shape not in ALLOWED_SHAPES:
            raise ValueError(f"shape must be one of {sorted(ALLOWED_SHAPES)}")
        if any(value <= 0 for value in size):
            raise ValueError("part sizes must be positive")
        if any(not 0 <= value <= 1 for value in color):
            raise ValueError("colors use normalized RGB values in [0, 1]")
        if material not in ALLOWED_MATERIALS:
            raise ValueError(f"unknown Roblox material: {material}")
        if not 0 <= transparency <= 1:
            raise ValueError("transparency must be in [0, 1]")
        if len(rotation) != 3:
            raise ValueError("rotation must contain X, Y, and Z degrees")
        return cls(
            id=uuid.uuid4().hex,
            op="add_part",
            payload={
                "name": name,
                "shape": shape,
                "position": list(position),
                "rotation": list(rotation),
                "size": list(size),
                "color": list(color),
                "material": material,
                "transparency": transparency,
                "anchored": bool(anchored),
                "can_collide": bool(can_collide),
            },
            depends_on=list(depends_on),
        )

    @classmethod
    def set_part_properties(cls, target_id: str, properties: dict[str, Any]) -> "SceneFunction":
        unknown = set(properties) - EDITABLE_PART_PROPERTIES
        if unknown:
            raise ValueError(f"unsupported part properties: {sorted(unknown)}")
        if not properties:
            raise ValueError("set_part_properties requires at least one property")
        return cls(id=uuid.uuid4().hex, op="set_part_properties",
                   payload={"target_id": target_id, "properties": properties})

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneFunction":
        return cls(**value)


@dataclass
class Part:
    function_id: str
    name: str
    shape: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    material: str = "Plastic"
    transparency: float = 0.0
    anchored: bool = True
    can_collide: bool = True


@dataclass
class SceneGraph:
    parts: list[Part] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        if not self.parts:
            return {"part_count": 0, "bounds": [0, 0, 0], "names": [], "shapes": []}
        minima = [min(p.position[i] - p.size[i] / 2 for p in self.parts) for i in range(3)]
        maxima = [max(p.position[i] + p.size[i] / 2 for p in self.parts) for i in range(3)]
        return {
            "part_count": len(self.parts),
            "bounds": [round(maxima[i] - minima[i], 3) for i in range(3)],
            "names": [p.name for p in self.parts],
            "shapes": [p.shape for p in self.parts],
            "materials": [p.material for p in self.parts],
            "parts": [
                {
                    "function_id": p.function_id, "name": p.name, "shape": p.shape,
                    "position": p.position, "rotation": p.rotation, "size": p.size,
                    "color": p.color, "material": p.material,
                    "transparency": p.transparency, "anchored": p.anchored,
                    "can_collide": p.can_collide,
                }
                for p in self.parts
            ],
            "skipped_functions": self.skipped,
        }


@dataclass
class ConversationTurn:
    role: str
    text: str
    created_at: float = field(default_factory=time.time)


class SceneSession:
    """Disk-backed complete conversation and replayable function history."""

    FORMAT_VERSION = 1

    def __init__(self, root: Path, title: str, functions: list[SceneFunction] | None = None,
                 turns: list[ConversationTurn] | None = None) -> None:
        self.root = root
        self.title = title
        self.functions = functions or []
        self.turns = turns or []

    @property
    def path(self) -> Path:
        return self.root / "session.json"

    @classmethod
    def open(cls, root: str | Path, title: str = "Untitled scene") -> "SceneSession":
        root = Path(root)
        path = root / "session.json"
        if not path.exists():
            return cls(root, title)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            root=root,
            title=raw["title"],
            functions=[SceneFunction.from_dict(item) for item in raw["functions"]],
            turns=[ConversationTurn(**item) for item in raw["turns"]],
        )

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "title": self.title,
            "functions": [asdict(item) for item in self.functions],
            "turns": [asdict(item) for item in self.turns],
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def record_turn(self, role: str, text: str) -> None:
        if role not in {"system", "user", "assistant", "grader"}:
            raise ValueError("role must be system, user, assistant, or grader")
        self.turns.append(ConversationTurn(role, text))
        self.save()

    def add_function(self, function: SceneFunction) -> SceneFunction:
        if any(existing.id == function.id for existing in self.functions):
            raise ValueError(f"duplicate function id {function.id}")
        known = {item.id for item in self.functions}
        missing = set(function.depends_on) - known
        if missing:
            raise ValueError(f"function has unknown dependencies: {sorted(missing)}")
        self.functions.append(function)
        self.save()
        return function

    def delete_function(self, target_id: str) -> None:
        function = self._find(target_id)
        function.active = False
        self.save()

    def restore_function(self, target_id: str) -> None:
        function = self._find(target_id)
        function.active = True
        self.save()

    def apply_generated_call(self, call: dict[str, Any]) -> str | None:
        """Apply precisely one validated model output.

        The generator never receives direct access to the filesystem or Studio;
        this method is its only mutation boundary.
        """
        # Local import avoids a module cycle: grammar imports the public enums above.
        from .grammar import validate_generated_call

        call = validate_generated_call(call)
        operation = call.get("op")
        if operation == "add_part":
            function = SceneFunction.add_part(
                name=str(call["name"]),
                shape=str(call["shape"]),
                position=tuple(call["position"]),
                size=tuple(call["size"]),
                color=tuple(call["color"]),
                depends_on=call.get("depends_on", ()),
                rotation=tuple(call.get("rotation", (0, 0, 0))),
                material=str(call.get("material", "Plastic")),
                transparency=float(call.get("transparency", 0)),
                anchored=bool(call.get("anchored", True)),
                can_collide=bool(call.get("can_collide", True)),
            )
            return self.add_function(function).id
        if operation == "set_part_properties":
            return self.add_function(SceneFunction.set_part_properties(
                str(call["target_id"]), dict(call["properties"])
            )).id
        aliases = {
            "set_material": "material", "set_color": "color", "set_transparency": "transparency",
            "set_size": "size", "set_position": "position", "set_rotation": "rotation",
            "set_anchored": "anchored", "set_can_collide": "can_collide",
        }
        if operation in aliases:
            property_name = aliases[operation]
            return self.add_function(SceneFunction.set_part_properties(
                str(call["target_id"]), {property_name: call[property_name]}
            )).id
        if operation == "delete_function":
            self.delete_function(str(call["target_id"]))
            return str(call["target_id"])
        if operation == "restore_function":
            self.restore_function(str(call["target_id"]))
            return str(call["target_id"])
        if operation == "finish":
            return None
        raise ValueError(f"unsupported generated operation: {operation}")

    def _find(self, target_id: str) -> SceneFunction:
        for function in self.functions:
            if function.id == target_id:
                return function
        raise KeyError(f"no function with id {target_id}")

    def rebuild(self) -> SceneGraph:
        graph = SceneGraph()
        valid_ids: set[str] = set()
        parts_by_id: dict[str, Part] = {}
        part_order: list[str] = []
        for function in self.functions:
            if not function.active:
                graph.skipped[function.id] = "deleted"
                continue
            missing = [item for item in function.depends_on if item not in valid_ids]
            if missing:
                graph.skipped[function.id] = f"missing active dependency: {', '.join(missing)}"
                continue
            if function.op == "set_part_properties":
                target_id = str(function.payload.get("target_id", ""))
                target = parts_by_id.get(target_id)
                if target is None:
                    graph.skipped[function.id] = f"target part is unavailable: {target_id}"
                    continue
                try:
                    properties = dict(function.payload["properties"])
                    unknown = set(properties) - EDITABLE_PART_PROPERTIES
                    if unknown:
                        raise ValueError(f"unsupported properties: {sorted(unknown)}")
                    candidate = replace(target, **self._normalise_part_properties(properties))
                    self._validate_part(candidate)
                    parts_by_id[target_id] = candidate
                except (KeyError, TypeError, ValueError) as error:
                    graph.skipped[function.id] = str(error)
                    continue
                valid_ids.add(function.id)
                continue
            if function.op != "add_part":
                graph.skipped[function.id] = f"unsupported operation: {function.op}"
                continue
            payload = function.payload
            try:
                part = Part(
                    function_id=function.id,
                    name=str(payload["name"]),
                    shape=str(payload["shape"]),
                    position=tuple(float(x) for x in payload["position"]),
                    rotation=tuple(float(x) for x in payload.get("rotation", (0, 0, 0))),
                    size=tuple(float(x) for x in payload["size"]),
                    color=tuple(float(x) for x in payload["color"]),
                    material=str(payload.get("material", "Plastic")),
                    transparency=float(payload.get("transparency", 0)),
                    anchored=bool(payload.get("anchored", True)),
                    can_collide=bool(payload.get("can_collide", True)),
                )
                self._validate_part(part)
            except (KeyError, TypeError, ValueError) as error:
                graph.skipped[function.id] = str(error)
                continue
            parts_by_id[function.id] = part
            part_order.append(function.id)
            valid_ids.add(function.id)
        graph.parts = [parts_by_id[function_id] for function_id in part_order if function_id in parts_by_id]
        return graph

    @staticmethod
    def _normalise_part_properties(properties: dict[str, Any]) -> dict[str, Any]:
        normalised = dict(properties)
        for key in ("position", "rotation", "size", "color"):
            if key in normalised:
                normalised[key] = tuple(float(value) for value in normalised[key])
        for key in ("transparency",):
            if key in normalised:
                normalised[key] = float(normalised[key])
        for key in ("anchored", "can_collide"):
            if key in normalised:
                normalised[key] = bool(normalised[key])
        return normalised

    @staticmethod
    def _validate_part(part: Part) -> None:
        if part.shape not in ALLOWED_SHAPES or any(len(vector) != 3 for vector in
                                                    (part.position, part.rotation, part.size, part.color)):
            raise ValueError("invalid part properties")
        if any(value <= 0 for value in part.size):
            raise ValueError("non-positive size")
        if any(not 0 <= value <= 1 for value in part.color):
            raise ValueError("color values must be in [0, 1]")
        if part.material not in ALLOWED_MATERIALS:
            raise ValueError(f"unknown Roblox material: {part.material}")
        if not 0 <= part.transparency <= 1:
            raise ValueError("transparency must be in [0, 1]")

    def model_context(self, max_recent_turns: int | None = None) -> dict[str, Any]:
        """Return the complete conversation by default; callers may explicitly bound it."""
        graph = self.rebuild()
        turns = self.turns if max_recent_turns is None else self.turns[-max_recent_turns:]
        return {
            "scene_request": self.title,
            "scene": graph.metadata(),
            "active_function_ids": [item.id for item in self.functions if item.active],
            "active_functions": [asdict(item) for item in self.functions if item.active],
            "conversation": [asdict(turn) for turn in turns],
            "history_location": str(self.path),
        }
