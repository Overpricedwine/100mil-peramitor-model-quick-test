from __future__ import annotations

from pathlib import Path

import torch

from .core import SceneGraph


class CudaSurrogateRenderer:
    """Fast orthographic multi-view rasterizer for reward shaping, using CUDA when present."""

    def __init__(self, image_size: int = 256, world_extent: float = 32.0, device: str | None = None) -> None:
        self.image_size = image_size
        self.world_extent = world_extent
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def render(self, graph: SceneGraph) -> dict[str, torch.Tensor]:
        axis = torch.linspace(-self.world_extent, self.world_extent, self.image_size, device=self.device)
        vertical, horizontal = torch.meshgrid(axis.flip(0), axis, indexing="ij")
        images = {name: torch.full((3, self.image_size, self.image_size), 0.055, device=self.device)
                  for name in ("front", "side", "top")}
        projections = {
            "front": (0, 1, 0, 1, 2),  # horizontal/vertical position, size, and rotation axis
            "side": (2, 1, 2, 1, 0),
            "top": (0, 2, 0, 2, 1),
        }
        for part in graph.parts:
            color = torch.tensor(part.color, device=self.device).view(3, 1, 1)
            for view, (h_pos, v_pos, h_size, v_size, rotation_axis) in projections.items():
                radians = torch.deg2rad(torch.tensor(part.rotation[rotation_axis], device=self.device))
                cosine, sine = torch.cos(radians), torch.sin(radians)
                delta_h = horizontal - part.position[h_pos]
                delta_v = vertical - part.position[v_pos]
                local_h = cosine * delta_h + sine * delta_v
                local_v = -sine * delta_h + cosine * delta_v
                half_h, half_v = part.size[h_size] / 2, part.size[v_size] / 2
                rectangle = (local_h.abs() <= half_h) & (local_v.abs() <= half_v)
                if part.shape in {"Ball", "Cylinder"}:
                    inside = (local_h / half_h).square() + (local_v / half_v).square() <= 1
                elif part.shape == "Wedge":
                    inside = rectangle & (local_v >= local_h / max(half_h, 0.001) * half_v - half_v)
                elif part.shape == "CornerWedge":
                    inside = rectangle & (local_v >= local_h.abs() / max(half_h, 0.001) * half_v - half_v)
                else:
                    inside = rectangle
                blend = color * (1 - part.transparency) + images[view] * part.transparency
                images[view] = torch.where(inside.unsqueeze(0), blend, images[view])
        return images

    def render_to_directory(self, graph: SceneGraph, directory: str | Path) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        result: dict[str, Path] = {}
        for name, tensor in self.render(graph).items():
            path = directory / f"{name}.ppm"
            self._write_ppm(tensor, path)
            result[name] = path
        return result

    @staticmethod
    def _write_ppm(image: torch.Tensor, path: Path) -> None:
        pixels = (image.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        height, width, _ = pixels.shape
        with path.open("wb") as output:
            output.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            output.write(pixels.tobytes())
