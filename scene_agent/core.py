from __future__ import annotations

import json
import math
import time
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
RIG_PART_ROLES = {"body", "limb", "accessory"}
JOINT_MOTION_TYPES = {"fixed": 0, "hinge": 1, "universal": 2}


def normalise_joint_motion(
    motion_type: str, axes: Iterable[Iterable[float]], angle_ranges: Iterable[Iterable[float]],
) -> tuple[str, tuple[tuple[float, float, float], ...], tuple[tuple[float, float], ...]]:
    """Validate a generic rig joint without assuming a humanoid skeleton.

    ``universal`` has two independent angular coordinates.  Each range is on a
    circular -180..180 degree dial: a lower bound greater than its upper bound
    intentionally means the legal interval wraps through +/-180 degrees.
    """
    if motion_type not in JOINT_MOTION_TYPES:
        raise ValueError(f"motion_type must be one of {sorted(JOINT_MOTION_TYPES)}")
    expected = JOINT_MOTION_TYPES[motion_type]
    if (not isinstance(axes, (list, tuple)) or not isinstance(angle_ranges, (list, tuple))):
        raise ValueError("joint axes and angle_ranges must be arrays")
    try:
        axes_result = tuple(tuple(float(value) for value in axis) for axis in axes)
        ranges_result = tuple(tuple(float(value) for value in angle_range) for angle_range in angle_ranges)
    except (TypeError, ValueError) as error:
        raise ValueError("joint axes and angle_ranges must be numeric arrays") from error
    if len(axes_result) != expected or len(ranges_result) != expected:
        raise ValueError(f"{motion_type} joints require exactly {expected} axes and angle ranges")
    normalised_axes: list[tuple[float, float, float]] = []
    for index, axis in enumerate(axes_result, start=1):
        raw_axis = axes[index - 1]
        if (not isinstance(raw_axis, (list, tuple)) or len(axis) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_axis)
                or not all(math.isfinite(value) for value in axis)):
            raise ValueError(f"axis {index} must be a finite three-number vector")
        length = math.sqrt(sum(value * value for value in axis))
        if length < 1e-6:
            raise ValueError(f"axis {index} must not be zero")
        normalised_axes.append(tuple(value / length for value in axis))
    if expected == 2:
        dot = sum(normalised_axes[0][index] * normalised_axes[1][index] for index in range(3))
        if abs(dot) > 0.95:
            raise ValueError("universal joint axes must not be parallel")
    for index, angle_range in enumerate(ranges_result, start=1):
        raw_range = angle_ranges[index - 1]
        if (not isinstance(raw_range, (list, tuple)) or len(angle_range) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_range)
                or not all(math.isfinite(value) for value in angle_range)):
            raise ValueError(f"angle range {index} must contain two finite numbers")
        if any(value < -180 or value > 180 for value in angle_range):
            raise ValueError("joint angles must be within -180 through 180 degrees")
    return motion_type, tuple(normalised_axes), ranges_result


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
            # The session assigns a short stable ID on insertion.  A model must
            # be able to copy this ID later to edit/delete the same function.
            id="",
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
        return cls(id="", op="set_part_properties",
                   payload={"target_id": target_id, "properties": properties})

    @classmethod
    def add_rig_root(
        cls, name: str, shape: str, position: tuple[float, float, float],
        size: tuple[float, float, float], color: tuple[float, float, float],
        *, rotation: tuple[float, float, float] = (0, 0, 0), material: str = "SmoothPlastic",
        transparency: float = 0.0,
    ) -> "SceneFunction":
        """Create the one visible root part of a generated character rig."""
        function = cls.add_part(
            name, shape, position, size, color, rotation=rotation, material=material,
            transparency=transparency, anchored=False, can_collide=False,
        )
        function.op = "add_rig_root"
        function.payload["role"] = "root"
        return function

    @classmethod
    def add_rig_part(
        cls, name: str, shape: str, position: tuple[float, float, float],
        size: tuple[float, float, float], color: tuple[float, float, float], *,
        parent_part_id: str, joint_name: str, joint_position: tuple[float, float, float],
        role: str = "limb", rotation: tuple[float, float, float] = (0, 0, 0),
        material: str = "SmoothPlastic", transparency: float = 0.0,
        motion_type: str = "hinge", axes: Iterable[Iterable[float]] = ((1, 0, 0),),
        angle_ranges: Iterable[Iterable[float]] = ((-45, 45),),
    ) -> "SceneFunction":
        """Create one part and its Motor6D connection as one reversible action."""
        if role not in RIG_PART_ROLES:
            raise ValueError(f"role must be one of {sorted(RIG_PART_ROLES)}")
        if not parent_part_id or not joint_name.strip():
            raise ValueError("rig parts require a parent part ID and joint name")
        motion_type, axes, angle_ranges = normalise_joint_motion(motion_type, axes, angle_ranges)
        function = cls.add_part(
            name, shape, position, size, color, depends_on=[parent_part_id], rotation=rotation,
            material=material, transparency=transparency, anchored=False, can_collide=False,
        )
        function.op = "add_rig_part"
        function.payload.update({
            "role": role, "parent_part_id": parent_part_id, "joint_name": joint_name,
            "joint_position": list(joint_position), "joint_type": "Motor6D",
            "motion_type": motion_type, "axes": [list(axis) for axis in axes],
            "angle_ranges": [list(angle_range) for angle_range in angle_ranges],
        })
        return function

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
    is_rig_root: bool = False
    rig_role: str | None = None


@dataclass
class Joint:
    """A constrained Motor6D created together with its child rig part."""

    function_id: str
    name: str
    parent_part_id: str
    child_part_id: str
    position: tuple[float, float, float]
    joint_type: str = "Motor6D"
    motion_type: str = "hinge"
    axes: tuple[tuple[float, float, float], ...] = ((1.0, 0.0, 0.0),)
    angle_ranges: tuple[tuple[float, float], ...] = ((-45.0, 45.0),)


@dataclass
class SceneGraph:
    parts: list[Part] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def rig_validation(self) -> dict[str, Any]:
        """Deterministic checks the visual grader should never have to guess."""
        roots = [part for part in self.parts if part.is_rig_root]
        rigged_parts = [part for part in self.parts if part.rig_role is not None]
        issues: list[str] = []
        if rigged_parts and len(roots) != 1:
            issues.append("rig requires exactly one visible primary part")
        part_ids = {part.function_id for part in self.parts}
        child_ids: set[str] = set()
        for joint in self.joints:
            if joint.parent_part_id not in part_ids or joint.child_part_id not in part_ids:
                issues.append(f"joint {joint.name} references an unavailable part")
            if joint.child_part_id in child_ids:
                issues.append(f"part {joint.child_part_id} has more than one parent joint")
            child_ids.add(joint.child_part_id)
        for part in rigged_parts:
            if part.anchored or part.can_collide:
                issues.append(f"rigged part {part.name} must be unanchored and non-colliding")
            if not part.is_rig_root and part.function_id not in child_ids:
                issues.append(f"rigged part {part.name} has no parent joint")
        return {
            "is_rigged": bool(rigged_parts), "root_count": len(roots),
            "joint_count": len(self.joints), "valid": not issues, "issues": issues,
        }

    def metadata(self) -> dict[str, Any]:
        if not self.parts:
            return {"part_count": 0, "joint_count": 0, "bounds": [0, 0, 0], "names": [], "shapes": []}
        minima = [min(p.position[i] - p.size[i] / 2 for p in self.parts) for i in range(3)]
        maxima = [max(p.position[i] + p.size[i] / 2 for p in self.parts) for i in range(3)]
        return {
            "part_count": len(self.parts),
            "bounds": [round(maxima[i] - minima[i], 3) for i in range(3)],
            "names": [p.name for p in self.parts],
            "shapes": [p.shape for p in self.parts],
            "materials": [p.material for p in self.parts],
            "joint_count": len(self.joints),
            "rig_roots": [p.function_id for p in self.parts if p.is_rig_root],
            "rig_validation": self.rig_validation(),
            "joints": [
                {"function_id": joint.function_id, "name": joint.name,
                 "parent_part_id": joint.parent_part_id, "child_part_id": joint.child_part_id,
                 "position": joint.position, "joint_type": joint.joint_type,
                 "motion_type": joint.motion_type, "axes": joint.axes,
                 "angle_ranges": joint.angle_ranges}
                for joint in self.joints
            ],
            "parts": [
                {
                    "function_id": p.function_id, "name": p.name, "shape": p.shape,
                    "position": p.position, "rotation": p.rotation, "size": p.size,
                    "color": p.color, "material": p.material,
                    "transparency": p.transparency, "anchored": p.anchored,
                    "can_collide": p.can_collide, "is_rig_root": p.is_rig_root,
                    "rig_role": p.rig_role,
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
        if not function.id:
            function.id = self._next_function_id()
        if any(existing.id == function.id for existing in self.functions):
            raise ValueError(f"duplicate function id {function.id}")
        known = {item.id for item in self.functions}
        missing = set(function.depends_on) - known
        if missing:
            raise ValueError(f"function has unknown dependencies: {sorted(missing)}")
        self.functions.append(function)
        self.save()
        return function

    def _next_function_id(self) -> str:
        """Readable, deterministic IDs make editing actions learnable/copyable."""
        number = 1
        existing = {item.id for item in self.functions}
        while (candidate := f"fn_{number:06d}") in existing:
            number += 1
        return candidate

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
        if operation == "add_rig_root":
            function = SceneFunction.add_rig_root(
                name=str(call["name"]), shape=str(call["shape"]), position=tuple(call["position"]),
                size=tuple(call["size"]), color=tuple(call["color"]),
                rotation=tuple(call.get("rotation", (0, 0, 0))), material=str(call.get("material", "SmoothPlastic")),
                transparency=float(call.get("transparency", 0)),
            )
            return self.add_function(function).id
        if operation == "add_rig_part":
            function = SceneFunction.add_rig_part(
                name=str(call["name"]), shape=str(call["shape"]), position=tuple(call["position"]),
                size=tuple(call["size"]), color=tuple(call["color"]),
                parent_part_id=str(call["parent_part_id"]), joint_name=str(call["joint_name"]),
                joint_position=tuple(call["joint_position"]), role=str(call.get("role", "limb")),
                rotation=tuple(call.get("rotation", (0, 0, 0))), material=str(call.get("material", "SmoothPlastic")),
                transparency=float(call.get("transparency", 0)),
                motion_type=str(call.get("motion_type", "hinge")), axes=call.get("axes", ((1, 0, 0),)),
                angle_ranges=call.get("angle_ranges", ((-45, 45),)),
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
        joints: list[Joint] = []
        rig_root_id: str | None = None
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
                    if candidate.rig_role is not None and (candidate.anchored or candidate.can_collide):
                        raise ValueError("rigged parts must remain unanchored and non-colliding")
                    parts_by_id[target_id] = candidate
                except (KeyError, TypeError, ValueError) as error:
                    graph.skipped[function.id] = str(error)
                    continue
                valid_ids.add(function.id)
                continue
            if function.op not in {"add_part", "add_rig_root", "add_rig_part"}:
                graph.skipped[function.id] = f"unsupported operation: {function.op}"
                continue
            payload = function.payload
            try:
                if function.op == "add_rig_root":
                    if rig_root_id is not None:
                        raise ValueError("a rig may have only one active root part")
                    if payload.get("role") != "root":
                        raise ValueError("rig root role must be root")
                if function.op == "add_rig_part":
                    parent_id = str(payload.get("parent_part_id", ""))
                    if parent_id not in parts_by_id:
                        raise ValueError(f"rig parent is unavailable: {parent_id}")
                    if payload.get("role") not in RIG_PART_ROLES:
                        raise ValueError("rig part has an invalid role")
                    if payload.get("joint_type") != "Motor6D" or not str(payload.get("joint_name", "")).strip():
                        raise ValueError("rig part requires a named Motor6D joint")
                    normalise_joint_motion(
                        str(payload.get("motion_type", "hinge")), payload.get("axes", ((1, 0, 0),)),
                        payload.get("angle_ranges", ((-45, 45),)),
                    )
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
                    anchored=bool(payload.get("anchored", function.op == "add_part")),
                    can_collide=bool(payload.get("can_collide", function.op == "add_part")),
                    is_rig_root=function.op == "add_rig_root",
                    rig_role=payload.get("role") if function.op in {"add_rig_root", "add_rig_part"} else None,
                )
                self._validate_part(part)
                if function.op in {"add_rig_root", "add_rig_part"} and (part.anchored or part.can_collide):
                    raise ValueError("rigged parts must be unanchored and non-colliding")
            except (KeyError, TypeError, ValueError) as error:
                graph.skipped[function.id] = str(error)
                continue
            parts_by_id[function.id] = part
            part_order.append(function.id)
            if function.op == "add_rig_root":
                rig_root_id = function.id
            elif function.op == "add_rig_part":
                motion_type, axes, angle_ranges = normalise_joint_motion(
                    str(payload.get("motion_type", "hinge")), payload.get("axes", ((1, 0, 0),)),
                    payload.get("angle_ranges", ((-45, 45),)),
                )
                joints.append(Joint(
                    function_id=function.id, name=str(payload["joint_name"]),
                    parent_part_id=str(payload["parent_part_id"]), child_part_id=function.id,
                    position=tuple(float(x) for x in payload["joint_position"]),
                    motion_type=motion_type, axes=axes, angle_ranges=angle_ranges,
                ))
            valid_ids.add(function.id)
        graph.parts = [parts_by_id[function_id] for function_id in part_order if function_id in parts_by_id]
        graph.joints = joints
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
            # Restores must be able to refer to a deleted function, so never
            # hide history from the model's saved scene context.
            "function_history": [asdict(item) for item in self.functions],
            "conversation": [asdict(turn) for turn in turns],
            "history_location": str(self.path),
        }
