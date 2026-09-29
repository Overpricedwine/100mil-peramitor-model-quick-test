import unittest

from scene_agent.constrained_decoding import is_json_object_prefix
from scene_agent.grammar import parse_generated_json


class GrammarTests(unittest.TestCase):
    def test_prefix_rejects_second_document_and_bad_escape(self):
        self.assertTrue(is_json_object_prefix('{"op":"add'))
        self.assertFalse(is_json_object_prefix('{"op":"finish"}{'))
        self.assertFalse(is_json_object_prefix('{"op":"bad\\q'))

    def test_strict_json_rejects_duplicates_and_nonfinite_numbers(self):
        with self.assertRaises(ValueError):
            parse_generated_json('{"op":"finish","op":"finish"}')
        with self.assertRaises(ValueError):
            parse_generated_json(
                '{"op":"add_part","name":"x","shape":"Block","position":[NaN,0,0],'
                '"size":[1,1,1],"color":[0,0,0]}'
            )

    def test_string_boolean_is_not_coerced(self):
        with self.assertRaises(ValueError):
            parse_generated_json(
                '{"op":"add_part","name":"x","shape":"Block","position":[0,0,0],'
                '"size":[1,1,1],"color":[0,0,0],"anchored":"false"}'
            )

    def test_rig_child_requires_a_real_parent_and_joint_position(self):
        with self.assertRaises(ValueError):
            parse_generated_json(
                '{"op":"add_rig_part","name":"arm","shape":"Block","position":[0,0,0],'
                '"size":[1,1,1],"color":[0,0,0],"joint_name":"Shoulder",'
                '"joint_position":[0,0,0]}'
            )

    def test_universal_joint_accepts_two_axes_and_a_wraparound_range(self):
        call = parse_generated_json(
            '{"op":"add_rig_part","name":"tail","shape":"Block","position":[0,0,0],'
            '"size":[1,1,1],"color":[0,0,0],"parent_part_id":"fn_000001",'
            '"joint_name":"Tail","joint_position":[0,0,0],"motion_type":"universal",'
            '"axes":[[1,0,0],[0,1,0]],"angle_ranges":[[-45,45],[120,-120]]}'
        )
        self.assertEqual("universal", call["motion_type"])
        self.assertEqual([120.0, -120.0], call["angle_ranges"][1])

    def test_hinge_rejects_a_second_angle_range(self):
        with self.assertRaises(ValueError):
            parse_generated_json(
                '{"op":"add_rig_part","name":"tail","shape":"Block","position":[0,0,0],'
                '"size":[1,1,1],"color":[0,0,0],"parent_part_id":"fn_000001",'
                '"joint_name":"Tail","joint_position":[0,0,0],"motion_type":"hinge",'
                '"axes":[[1,0,0]],"angle_ranges":[[-45,45],[-20,20]]}'
            )


if __name__ == "__main__":
    unittest.main()
