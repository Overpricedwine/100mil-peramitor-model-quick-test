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


if __name__ == "__main__":
    unittest.main()
