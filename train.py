from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scene_agent.core import SceneSession
from scene_agent.environment import RenderInTheLoopEnv
from scene_agent.grader import HeuristicGrader, LocalQwenVLGrader
from scene_agent.policy import CudaFunctionPolicy
from scene_agent.renderer import CudaSurrogateRenderer


def encode_prompt(text: str, device: torch.device, length: int = 192) -> torch.Tensor:
    values = list(text.encode("utf-8"))[:length]
    values.extend([0] * (length - len(values)))
    return torch.tensor([values], device=device, dtype=torch.long)


def train(project: Path, steps: int, learning_rate: float, grader_kind: str, model_id: str) -> None:
    session = SceneSession.open(project, title="A castle with two red towers")
    if not session.turns:
        session.record_turn("user", session.title)
    renderer = CudaSurrogateRenderer(image_size=128)
    grader = LocalQwenVLGrader(model_id) if grader_kind == "local" else HeuristicGrader()
    env = RenderInTheLoopEnv(session, renderer, grader=grader)
    policy = CudaFunctionPolicy().to(renderer.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    running_baseline = 0.0

    for step in range(steps):
        prompt = encode_prompt(session.title, renderer.device)
        choice = policy.sample(prompt, env.scene_features())
        result = env.step(choice)  # validation, GPU render, and grade happen here.
        advantage = result.reward - running_baseline
        running_baseline = 0.95 * running_baseline + 0.05 * result.reward
        loss = -choice.log_probability.mean() * advantage
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        print(f"step={step:04d} reward={result.reward:+.3f} score={result.grade.score:5.1f} loss={loss.item():+.4f}")
        if result.done:
            break

    project.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "title": session.title}, project / "policy.pt")
    session.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one-function-at-a-time scene policy with CUDA renders.")
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grader", choices=("local", "heuristic"), default="local")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    arguments = parser.parse_args()
    train(arguments.project, arguments.steps, arguments.learning_rate, arguments.grader, arguments.model_id)
