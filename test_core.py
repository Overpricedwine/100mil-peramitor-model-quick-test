import tempfile
import unittest
from pathlib import Path

from scene_agent.core import SceneFunction, SceneSession


class SceneSessionTests(unittest.TestCase):
    def test_delete_and_restore_replays_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = SceneSession.open(Path(directory), "test")
            base = session.add_function(SceneFunction.add_part(
                "Base", "Block", (0, 1, 0), (4, 2, 4), (1, 0, 0)
            ))
            tower = session.add_function(SceneFunction.add_part(
                "Tower", "Block", (0, 5, 0), (2, 6, 2), (1, 0, 0), [base.id]
            ))
            self.assertEqual(2, len(session.rebuild().parts))
            session.delete_function(base.id)
            graph = session.rebuild()
            self.assertEqual(0, len(graph.parts))
            self.assertIn(tower.id, graph.skipped)
            session.restore_function(base.id)
            self.assertEqual(2, len(session.rebuild().parts))

    def test_persists_all_conversation_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = SceneSession.open(Path(directory), "test")
            session.record_turn("user", "first request")
            loaded = SceneSession.open(Path(directory))
            self.assertEqual("first request", loaded.turns[0].text)

    def test_generated_delete_call_is_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = SceneSession.open(Path(directory), "test")
            function_id = session.apply_generated_call({
                "op": "add_part", "name": "Door", "shape": "Block",
                "position": [0, 1, 0], "size": [1, 2, 1], "color": [1, 1, 1],
            })
            session.apply_generated_call({"op": "delete_function", "target_id": function_id})
            self.assertEqual(0, len(session.rebuild().parts))
            session.apply_generated_call({"op": "restore_function", "target_id": function_id})
            self.assertEqual(1, len(session.rebuild().parts))

    def test_primary_rig_part_creates_one_motor_joint_per_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = SceneSession.open(Path(directory), "rig")
            root_id = session.apply_generated_call({
                "op": "add_rig_root", "name": "Torso", "shape": "Block",
                "position": [0, 5, 0], "size": [4, 5, 2], "color": [0.2, 0.5, 0.8],
            })
            arm_id = session.apply_generated_call({
                "op": "add_rig_part", "name": "LeftArm", "shape": "Block",
                "position": [-3, 5, 0], "size": [1, 4, 1], "color": [0.2, 0.5, 0.8],
                "parent_part_id": root_id, "joint_name": "LeftShoulder",
                "joint_position": [-2, 6, 0], "role": "limb",
                "motion_type": "universal", "axes": [[1, 0, 0], [0, 1, 0]],
                "angle_ranges": [[-45, 45], [120, -120]],
            })
            graph = session.rebuild()
            self.assertEqual([root_id], graph.metadata()["rig_roots"])
            self.assertEqual(1, len(graph.joints))
            self.assertEqual(root_id, graph.joints[0].parent_part_id)
            self.assertEqual(arm_id, graph.joints[0].child_part_id)
            self.assertEqual("universal", graph.joints[0].motion_type)
            self.assertEqual(((120.0, -120.0)), graph.joints[0].angle_ranges[1:])


if __name__ == "__main__":
    unittest.main()
