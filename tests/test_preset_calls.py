import json
import unittest

from scene_agent.preset_calls import decode_preset_call, encode_preset_call, is_preset_prefix
from scene_agent.training_data import make_seed_scene, records_for_scene


class PresetCallTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.functions = make_seed_scene(0)
        records = list(records_for_scene(self.spec, self.functions, "Build this rig.", scene_id=0))
        self.contexts = [json.loads(record["context"]) for record in records]

    def test_root_round_trip_uses_only_numbers(self):
        text = encode_preset_call(self.functions[0], self.contexts[0])
        self.assertTrue(all(character in "0123456789|+-.eE" for character in text))
        rebuilt = decode_preset_call(text, self.contexts[0])
        self.assertEqual("add_rig_root", rebuilt["op"])
        self.assertEqual(self.functions[0]["position"], rebuilt["position"])

    def test_rig_child_round_trip_uses_context_reference(self):
        text = encode_preset_call(self.functions[1], self.contexts[1])
        rebuilt = decode_preset_call(text, self.contexts[1])
        self.assertEqual("add_rig_part", rebuilt["op"])
        self.assertEqual("fn_000001", rebuilt["parent_part_id"])

    def test_every_generated_record_has_a_numeric_preset(self):
        records = list(records_for_scene(self.spec, self.functions, "Build this rig.", scene_id=0))
        for record in records:
            call = json.loads(record["target"])
            text = encode_preset_call(call, record["context"])
            rebuilt = decode_preset_call(text, record["context"])
            self.assertEqual(call["op"], rebuilt["op"])

    def test_prefix_rejects_words_and_unknown_preset(self):
        self.assertTrue(is_preset_prefix("1|0|-2.5"))
        self.assertFalse(is_preset_prefix("add_rig_root"))
        self.assertFalse(is_preset_prefix("7|0"))


if __name__ == "__main__":
    unittest.main()
