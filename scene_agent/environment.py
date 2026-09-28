from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import SceneFunction, SceneSession
from .grader import Grade, HeuristicGrader, SceneGrader
from .policy import PolicyChoice
from .renderer import CudaSurrogateRenderer


@dataclass
class StepResult:
    reward: float
    grade: Grade
    function_id: str | None
    done: bool


class RenderInTheLoopEnv:
    def __init__(self, session: SceneSession, renderer: CudaSurrogateRenderer,
                 grader: SceneGrader | None = None, max_parts: int = 24) -> None:
        self.session = session
        self.renderer = renderer
        self.grader = grader or HeuristicGrader()
        self.max_parts = max_parts
        self.previous_score = 0.0

    def scene_features(self) -> torch.Tensor:
        metadata = self.session.rebuild().metadata()
        values = [metadata["part_count"] / self.max_parts, *(dimension / 64 for dimension in metadata["bounds"])]
        values.extend([0.0] * (12 - len(values)))
        return torch.tensor([values], device=self.renderer.device, dtype=torch.float32)

    def step(self, choice: PolicyChoice) -> StepResult:
        operation = choice.operation
        if operation == 3:
            return StepResult(0.0, Grade(self.previous_score, [], []), None, True)
        active = [function for function in self.session.functions if function.active]
        function_id: str | None = None
        if operation == 2 and active:
            target = active[choice.x % len(active)]
            function_id = self.session.apply_generated_call({"op": "delete_function", "target_id": target.id})
        elif len(self.session.rebuild().parts) < self.max_parts:
            coordinate = lambda value: (value / 16 - 0.5) * 32
            position = (coordinate(choice.x), max(0.5, coordinate(choice.y)), coordinate(choice.z))
            side = 1.5 + choice.scale * 0.45
            palette = ((0.75, 0.08, 0.06), (0.55, 0.55, 0.62), (0.28, 0.42, 0.8),
                       (0.25, 0.65, 0.32), (0.72, 0.5, 0.16), (0.8, 0.8, 0.78))
            function_id = self.session.apply_generated_call({
                "op": "add_part", "name": f"Generated_{len(self.session.functions) + 1}",
                "shape": "Cylinder" if operation == 1 else "Block", "position": list(position),
                "size": [side, side, side], "color": list(palette[choice.color % len(palette)]),
            })
        graph = self.session.rebuild()
        grade = self.grader.grade(self.session.title, graph, self.renderer.render(graph))
        reward = (grade.score - self.previous_score) / 100.0
        self.previous_score = grade.score
        self.session.record_turn("grader", f"score={grade.score:.1f}; feedback={grade.feedback}")
        return StepResult(reward, grade, function_id, False)
