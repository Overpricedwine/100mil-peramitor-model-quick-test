"""Run the complete persistent scene-generation and local-review cycle.

This is inference/refinement, not a weight-update command.  It creates useful
reviewed traces which can later be ranked and added to the next training set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from generate_function import generate_valid_function, load_generator
from scene_agent.conversation import LocalQwenConversationDirector, build_review_context, continue_after_grade
from scene_agent.core import SceneSession
from scene_agent.grader import HeuristicGrader, LocalQwenVLGrader
from scene_agent.beauty_renderer import CudaBeautyRenderer
from scene_agent.renderer import CudaSurrogateRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, render, grade, and refine one Roblox scene session.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True, help="Directory containing persistent session.json")
    parser.add_argument("--goal", required=True, help="Original user model request for a new session")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--grader", choices=("qwen", "heuristic"), default="qwen")
    parser.add_argument("--renderer", choices=("beauty", "surrogate"), default="beauty",
                        help="Use beauty for final grading, surrogate for cheaper RL rollouts")
    parser.add_argument("--device", default=None)
    parser.add_argument("--explore-seed", type=int,
                        help="Optional reproducible variation for grader/director repair selection")
    parser.add_argument("--trajectory", type=Path,
                        help="JSONL destination for graded candidate traces (default: session/trajectory.jsonl)")
    parser.add_argument("--accept-score", type=float, default=80.0,
                        help="Mark traces at or above this score as candidates for human review")
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    session = SceneSession.open(args.session, args.goal)
    goal = session.title
    trajectory_path = args.trajectory or args.session / "trajectory.jsonl"
    if not session.turns:
        session.record_turn("user", goal)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator, tokenizer = load_generator(args.checkpoint, args.tokenizer, device)
    renderer = CudaBeautyRenderer(device=str(device)) if args.renderer == "beauty" else CudaSurrogateRenderer(device=str(device))
    if args.grader == "qwen":
        grader = LocalQwenVLGrader()
        # Reuse the grader's Qwen weights for the text-only director. Loading
        # two independent 3B checkpoints would waste roughly another 6 GB.
        director = LocalQwenConversationDirector(model=grader.model, processor=grader.processor)
    else:
        grader, director = HeuristicGrader(), None

    previous_grade = None
    for step in range(1, args.max_steps + 1):
        action_context = session.model_context()
        function = generate_valid_function(generator, tokenizer, action_context)
        if function["op"] == "finish":
            session.record_turn("assistant", json.dumps(function, separators=(",", ":")))
            print(f"step={step}: generator finished")
            break
        try:
            function_id = session.apply_generated_call(function)
        except (KeyError, ValueError) as error:
            # JSON/schema validity is not enough for an edit: its target may
            # have been deleted or never existed. Keep the immutable history
            # accurate and give the next turn the exact rejection reason.
            session.record_turn("system", f"Rejected generated function: {error}")
            print(f"step={step}: rejected semantic scene call: {error}")
            continue
        session.record_turn("assistant", json.dumps(function, separators=(",", ":")))
        graph = session.rebuild()
        render_directory = args.session / "renders" / f"step_{step:03d}"
        renderer.render_to_directory(graph, render_directory)
        review_context = build_review_context(session, goal, function_id, previous_grade)
        seed = None if args.explore_seed is None else args.explore_seed + step * 2
        grade = grader.grade(goal, graph, renderer.render(graph), review_context=review_context, suggestion_seed=seed)
        score_before = float(previous_grade["score"]) if previous_grade is not None else 0.0
        score_delta = grade.score - score_before
        previous_grade = grade.as_dict()
        session.record_turn("grader", json.dumps(previous_grade, separators=(",", ":")))
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory = {
            # These two fields are directly compatible with the supervised
            # JSONL reader after human/rule-based acceptance filtering.
            "context": json.dumps(action_context, separators=(",", ":")),
            "target": json.dumps(function, separators=(",", ":")),
            "step": step, "function_id": function_id, "grade": previous_grade,
            "score_delta": score_delta,
            "candidate_for_review": (grade.score >= args.accept_score and score_delta >= 0
                                     and grade.status != "repair"),
        }
        with trajectory_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trajectory, separators=(",", ":")) + "\n")
        print(f"step={step}: {function['op']} id={function_id}; score={grade.score:.1f}; status={grade.status}")

        if grade.status == "done":
            print("grader marked the request complete")
            break
        if director is not None:
            decision = continue_after_grade(
                session, director, original_goal=goal, last_function_id=function_id,
                grade_result=previous_grade,
                selection_seed=None if args.explore_seed is None else args.explore_seed + step * 2 + 1,
            )
            if decision.action == "finish":
                print("conversation director marked the request complete")
                break
    else:
        print(f"stopped at --max-steps={args.max_steps}; session is saved and can be resumed")

    print(f"session={session.path}\ntrajectory={trajectory_path}")


if __name__ == "__main__":
    main()
