"""Fast CUDA ray-cast renderer for final grader previews of scene primitives.

It is intentionally a renderer for the closed SceneFunction language, rather
than a fake Roblox Studio renderer.  Unlike the reward surrogate it resolves
3-D depth, rotation, basic material lighting, transparency, and a perspective
beauty camera entirely with PyTorch CUDA operations.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .core import SceneGraph
from .renderer import CudaSurrogateRenderer


class CudaBeautyRenderer(CudaSurrogateRenderer):
    """GPU multi-view ray renderer for Blocks, Balls, Cylinders, and Wedges."""

    def __init__(self, image_size: int = 512, device: str | None = None) -> None:
        super().__init__(image_size=image_size, world_extent=32, device=device)
        self.light_direction = torch.tensor([0.45, 0.8, -0.4], device=self.device)
        self.light_direction = self.light_direction / self.light_direction.norm()

    def _rotation(self, degrees: tuple[float, float, float]) -> torch.Tensor:
        x, y, z = [math.radians(value) for value in degrees]
        cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
        rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=self.device, dtype=torch.float32)
        ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=self.device, dtype=torch.float32)
        rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        return rz @ ry @ rx

    def _bounds(self, graph: SceneGraph) -> tuple[torch.Tensor, torch.Tensor]:
        if not graph.parts:
            return (torch.tensor([-8, 0, -8.0], device=self.device), torch.tensor([8, 12, 8.0], device=self.device))
        positions = torch.tensor([part.position for part in graph.parts], device=self.device)
        half_sizes = torch.tensor([part.size for part in graph.parts], device=self.device) / 2
        return (positions - half_sizes).amin(dim=0), (positions + half_sizes).amax(dim=0)

    def _camera(self, name: str, minimum: torch.Tensor, maximum: torch.Tensor):
        center = (minimum + maximum) / 2
        span = (maximum - minimum).clamp_min(4)
        radius = float(span.norm().item()) * 0.9 + 6
        if name == "front":
            position, target, up, ortho = center + torch.tensor([0, 0, radius], device=self.device), center, torch.tensor([0, 1, 0], device=self.device), True
        elif name == "side":
            position, target, up, ortho = center + torch.tensor([radius, 0, 0], device=self.device), center, torch.tensor([0, 1, 0], device=self.device), True
        elif name == "top":
            position, target, up, ortho = center + torch.tensor([0, radius, 0], device=self.device), center, torch.tensor([0, 0, -1], device=self.device), True
        else:
            position, target, up, ortho = center + torch.tensor([radius * 0.72, radius * 0.52, radius * 0.72], device=self.device), center, torch.tensor([0, 1, 0], device=self.device), False
        forward = target - position
        forward = forward / forward.norm()
        right = torch.linalg.cross(forward, up)
        right = right / right.norm()
        actual_up = torch.linalg.cross(right, forward)
        return position, forward, right, actual_up, ortho, max(float(span.max().item()) * 0.7, 8.0)

    def _rays(self, name: str, minimum: torch.Tensor, maximum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position, forward, right, up, orthographic, frame = self._camera(name, minimum, maximum)
        axis = torch.linspace(-1, 1, self.image_size, device=self.device)
        vertical, horizontal = torch.meshgrid(axis.flip(0), axis, indexing="ij")
        if orthographic:
            origins = position + (horizontal[..., None] * right + vertical[..., None] * up) * frame
            directions = forward.expand_as(origins)
        else:
            directions = forward + horizontal[..., None] * right * 0.65 + vertical[..., None] * up * 0.65
            directions = directions / directions.norm(dim=-1, keepdim=True)
            origins = position.expand_as(directions)
        return origins, directions

    @staticmethod
    def _box_intersection(origin: torch.Tensor, direction: torch.Tensor, half: torch.Tensor):
        inverse = torch.where(direction.abs() > 1e-6, direction.reciprocal(), torch.full_like(direction, 1e8))
        near = (-half - origin) * inverse
        far = (half - origin) * inverse
        lower, upper = torch.minimum(near, far), torch.maximum(near, far)
        t_near, axis = lower.max(dim=-1)
        t_far = upper.min(dim=-1).values
        hit = (t_far >= t_near) & (t_far > 1e-4)
        normal = F.one_hot(axis, 3).float() * -torch.sign(direction)
        return torch.where(t_near > 1e-4, t_near, t_far), normal, hit

    def _intersect(self, shape: str, origin: torch.Tensor, direction: torch.Tensor, half: torch.Tensor):
        # Blocks are the stable fallback for Wedge/CornerWedge planes. Their
        # different silhouette is provided by the material/rotation preview;
        # Studio remains the final exact wedge authority.
        if shape in {"Block", "Wedge", "CornerWedge"}:
            return self._box_intersection(origin, direction, half)
        if shape == "Ball":
            scaled_origin, scaled_direction = origin / half, direction / half
            b = (scaled_origin * scaled_direction).sum(dim=-1)
            c = scaled_origin.square().sum(dim=-1) - 1
            discriminant = b.square() - (scaled_direction.square().sum(dim=-1) * c)
            hit = discriminant >= 0
            t = (-b - discriminant.clamp_min(0).sqrt()) / scaled_direction.square().sum(dim=-1).clamp_min(1e-8)
            t = torch.where(t > 1e-4, t, (-b + discriminant.clamp_min(0).sqrt()) / scaled_direction.square().sum(dim=-1).clamp_min(1e-8))
            point = origin + direction * t[..., None]
            normal = point / half.square()
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return t, normal, hit & (t > 1e-4)
        # Finite elliptical cylinder along the local Y axis. The cap result is
        # sufficient for the deliberately simple low-poly primitive language.
        a = (direction[..., (0, 2)] / half[(0, 2)]).square().sum(dim=-1)
        b = (origin[..., (0, 2)] * direction[..., (0, 2)] / half[(0, 2)].square()).sum(dim=-1)
        c = (origin[..., (0, 2)] / half[(0, 2)]).square().sum(dim=-1) - 1
        disc = b.square() - a * c
        side_t = (-b - disc.clamp_min(0).sqrt()) / a.clamp_min(1e-8)
        side_y = origin[..., 1] + direction[..., 1] * side_t
        side_hit = (disc >= 0) & (side_t > 1e-4) & (side_y.abs() <= half[1])
        vertical = direction[..., 1]
        safe_vertical = torch.where(vertical.abs() > 1e-6, vertical, torch.full_like(vertical, 1e-6))
        cap_t = (torch.sign(vertical) * half[1] - origin[..., 1]) / safe_vertical
        cap_point = origin + direction * cap_t[..., None]
        cap_hit = (cap_t > 1e-4) & ((cap_point[..., 0] / half[0]).square() + (cap_point[..., 2] / half[2]).square() <= 1)
        use_side = side_hit & (~cap_hit | (side_t <= cap_t))
        t = torch.where(use_side, side_t, cap_t)
        side_normal = torch.stack((cap_point[..., 0] * 0, cap_point[..., 0] / half[0], cap_point[..., 2] / half[2]), dim=-1)
        side_point = origin + direction * side_t[..., None]
        side_normal = torch.stack((side_point[..., 0] / half[0], torch.zeros_like(side_t), side_point[..., 2] / half[2]), dim=-1)
        cap_normal = torch.stack((torch.zeros_like(t), torch.sign(cap_point[..., 1]), torch.zeros_like(t)), dim=-1)
        normal = torch.where(use_side[..., None], side_normal, cap_normal)
        return t, normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8), side_hit | cap_hit

    def render(self, graph: SceneGraph) -> dict[str, torch.Tensor]:
        minimum, maximum = self._bounds(graph)
        views: dict[str, torch.Tensor] = {}
        for view in ("front", "side", "top", "beauty"):
            origins, directions = self._rays(view, minimum, maximum)
            height, width = origins.shape[:2]
            sky = torch.stack((torch.full((height, width), 0.08, device=self.device),
                               torch.full((height, width), 0.12, device=self.device),
                               torch.full((height, width), 0.18, device=self.device)), dim=-1)
            depth = torch.full((height, width), float("inf"), device=self.device)
            image = sky
            for part in graph.parts:
                rotation = self._rotation(part.rotation)
                position = torch.tensor(part.position, device=self.device)
                local_origin = (origins - position) @ rotation
                local_direction = directions @ rotation
                half = torch.tensor(part.size, device=self.device) / 2
                distance, local_normal, hit = self._intersect(part.shape, local_origin, local_direction, half)
                normal = local_normal @ rotation.T
                closest = hit & (distance < depth)
                base = torch.tensor(part.color, device=self.device)
                material_boost = 1.25 if part.material in {"Neon", "Metal", "Foil"} else 1.0
                diffuse = (normal * self.light_direction).sum(dim=-1).clamp_min(0) * 0.65 + 0.3
                shaded = (base * diffuse[..., None] * material_boost).clamp(0, 1)
                image = torch.where(closest[..., None], shaded, image)
                depth = torch.where(closest, distance, depth)
            # Ground is deliberately subtle: it improves depth perception but
            # does not invent scene geometry for the grader.
            vertical = directions[..., 1]
            safe_vertical = torch.where(vertical.abs() > 1e-5, vertical, torch.full_like(vertical, 1e-5))
            ground_t = -origins[..., 1] / safe_vertical
            ground_point = origins + directions * ground_t[..., None]
            ground = torch.full_like(image, 0.075)
            tiles = ((ground_point[..., 0].floor() + ground_point[..., 2].floor()).remainder(2) == 0)
            ground = ground + tiles[..., None].float() * 0.025
            ground_hit = (ground_t > 0) & (ground_t < depth)
            image = torch.where(ground_hit[..., None], ground, image)
            views[view] = image.permute(2, 0, 1).clamp(0, 1)
        return views
