from __future__ import annotations

import argparse
from pathlib import Path

from scene_agent.core import SceneFunction, SceneSession
from scene_agent.rbxmx import write_rbxmx
from scene_agent.renderer import CudaSurrogateRenderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/castle")
    args = parser.parse_args()
    project = Path(args.project)
    session = SceneSession.open(project, title="A red castle with two towers")
    if not session.functions:
        session.record_turn("user", "Build a red castle with two matching towers.")
        base = session.add_function(SceneFunction.add_part(
            "CastleBase", "Block", (0, 0.5, 0), (24, 1, 16), (0.42, 0.18, 0.12)
        ))
        session.add_function(SceneFunction.add_part(
            "LeftTower", "Block", (-8, 5, 0), (4, 10, 4), (0.75, 0.08, 0.06), [base.id]
        ))
        session.add_function(SceneFunction.add_part(
            "RightTower", "Block", (8, 5, 0), (4, 10, 4), (0.75, 0.08, 0.06), [base.id]
        ))
        session.record_turn("assistant", "Added base and two reversible tower functions.")

    graph = session.rebuild()
    write_rbxmx(graph, project / "castle.rbxmx")
    renderer = CudaSurrogateRenderer(image_size=256)
    renderer.render_to_directory(graph, project / "renders")
    session.save()
    print(f"device={renderer.device}; active_parts={len(graph.parts)}; project={project}")


if __name__ == "__main__":
    main()
