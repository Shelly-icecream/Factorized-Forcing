import unittest

from stage3.conditions import CONDITION_NAMES
from stage3.parser import CaptionParser


class Stage3ParserTests(unittest.TestCase):
    def test_expanded_actions_do_not_duplicate_contained_verbs(self):
        result = CaptionParser().parse(
            "A man is holding a cup and walking in a park."
        )
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.action, "is holding a cup and walking")
        self.assertEqual(sum(span.field == "action" for span in result.spans), 1)

    def test_background_keeps_preposition_and_modifiers(self):
        for location in ("on a busy street", "through a very dense forest",
                         "in a snowy forest", "on a snow-covered street"):
            with self.subTest(location=location):
                result = CaptionParser().parse(f"A man is walking {location}.")
                self.assertEqual(result.parse_status, "ok")
                self.assertEqual(result.background, location)

    def test_person_stops_at_background_before_action(self):
        for description, person in (("A man in a room", "A man"),
                                    ("A man with long hair in a busy room", "A man with long hair"),
                                    ("A man in a red coat in a room", "A man in a red coat")):
            with self.subTest(description=description):
                caption = f"{description} is walking."
                result = CaptionParser().parse(caption)
                self.assertEqual(result.parse_status, "ok")
                self.assertEqual(result.person, person)
                self.assertIn(result.background, ("in a room", "in a busy room"))
                self.assertEqual(result.conditions["PAB"], caption)
                self.assertNotIn("room", result.conditions["P"])

    def test_background_prefix_does_not_cross_action(self):
        result = CaptionParser().parse("A man in a red coat is walking on a busy street.")
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.person, "A man in a red coat")
        self.assertEqual(result.action, "is walking")
        self.assertEqual(result.background, "on a busy street")

    def test_controlled_caption_and_exact_full_condition(self):
        caption = "A girl in a red coat is running through a snowy forest, cinematic lighting."
        result = CaptionParser().parse(caption)
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.person, "A girl in a red coat")
        self.assertEqual(result.action, "is running")
        self.assertEqual(result.background, "through a snowy forest")
        self.assertEqual(result.conditions["PAB"], caption)
        self.assertNotIn("remaining", result.to_dict())
        self.assertEqual(tuple(result.conditions), CONDITION_NAMES)
        spans = result.spans
        self.assertEqual(len(spans), 3)
        self.assertEqual(len({(span.start, span.end) for span in spans}), 3)
        self.assertTrue(all(spans[index].end <= spans[index + 1].start for index in range(2)))

    def test_long_action_and_clothing_are_not_background(self):
        result = CaptionParser().parse("A man wearing a blue coat is riding a horse on a beach.")
        self.assertEqual(result.parse_status, "ok")
        self.assertIn("blue coat", result.person)
        self.assertEqual(result.action, "is riding a horse")
        self.assertEqual(result.background, "on a beach")

    def test_person_keeps_demographic_and_appearance_modifiers(self):
        result = CaptionParser().parse(
            "A portrait of an East Asian woman with long wavy hair and expressive eyes "
            "is walking through a park."
        )
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(
            result.person,
            "an East Asian woman with long wavy hair and expressive eyes",
        )

    def test_scene_leading_caption_keeps_subject_and_background(self):
        result = CaptionParser().parse(
            "A mystical forest filled with ancient trees and glowing luminescent flora "
            "surrounds a lost and bewildered lady."
        )
        self.assertEqual(result.person, "a lost and bewildered lady")
        self.assertIn("mystical forest", result.background)
        self.assertNotIn("trees", result.person)

    def test_fields_do_not_depend_on_span_character_order(self):
        result = CaptionParser().parse(
            "Near a river, a tree made of candy stands tall."
        )
        self.assertEqual(result.person, "a tree")
        self.assertEqual(result.action, "stands")
        self.assertEqual(result.background, "Near a river")

    def test_group_subject_and_irregular_progressives(self):
        result = CaptionParser().parse(
            "A serene beach scene at Lake Baikal, showcasing several people "
            "enjoying the tranquil waters."
        )
        self.assertEqual(result.person, "several people")
        self.assertIn("beach", result.background)
        self.assertIn("swimming", " ".join(
            "swimming" if match[0] == 0 else "" for match in CaptionParser()._action_matches("swimming and sunbathing")
        ))

    def test_conjoined_subjects_and_negated_inputs(self):
        self.assertEqual(
            CaptionParser().parse("A girl and a boy are running in a room.").parse_status,
            "ok",
        )
        result = CaptionParser().parse("A girl is not running in a room.")
        self.assertEqual(result.failure_reason, "negation_not_supported")
        self.assertEqual(result.conditions, {})

    def test_dictionary_variants_and_context_guards(self):
        parser = CaptionParser()
        result = parser.parse("An elderly woman was slowly walking through a snowy forest.")
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.action, "was slowly walking")
        self.assertEqual(result.background, "through a snowy forest")

        result = parser.parse("A child is riding a brown horse on a beach.")
        self.assertEqual(result.parse_status, "ok")
        self.assertIn("horse", result.action)
        self.assertNotIn("horse", result.background)

        result = parser.parse("A person is painting a forest in a room.")
        self.assertEqual(result.parse_status, "ok")
        self.assertIn("forest", result.action)
        self.assertEqual(result.background, "in a room")

    def test_nonhuman_subjects_are_supported_in_person_field(self):
        parser = CaptionParser()
        result = parser.parse("A robot is walking in a factory.")
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.person, "A robot")
        result = parser.parse("A camera pans over a mountain.")
        self.assertNotEqual(result.parse_status, "ok")
        self.assertEqual(result.person, "")

    def test_camera_does_not_replace_scene_subject(self):
        result = CaptionParser().parse(
            "Several people are walking on a beach. The camera captures the scene."
        )
        self.assertEqual(result.person, "Several people")

    def test_following_action_only_disambiguates_person(self):
        result = CaptionParser().parse(
            "A person is near a wall. A dog is running through a forest."
        )
        self.assertEqual(result.person, "A dog")
        self.assertEqual(result.action, "is running")

    def test_subject_conjunction_remains_ambiguous(self):
        result = CaptionParser().parse("A girl and a boy are running in a room.")
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.person, "A girl and a boy")
        self.assertEqual(result.action, "are running")
        result = CaptionParser().parse("A man, a woman, and a child are walking in a park.")
        self.assertEqual(result.person, "A man, a woman, and a child")

    def test_action_objects_are_not_subjects(self):
        result = CaptionParser().parse("A child is riding a brown horse on a beach.")
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.person, "A child")

    def test_word_boundaries_and_irregular_forms(self):
        parser = CaptionParser()
        self.assertNotEqual(parser.parse("A woman is running in a room.").failure_reason, "multiple_subjects")
        self.assertNotEqual(parser.parse("A human is walking in a room.").failure_reason, "multiple_subjects")
        self.assertNotEqual(parser.parse("A cameraman is walking in a room.").parse_status, "ok")
        result = parser.parse("A girl ran through a forest.")
        self.assertEqual(result.parse_status, "ok")
        self.assertEqual(result.action, "ran")

    def test_possessive_subject_reference_is_not_a_second_subject(self):
        result = CaptionParser().parse(
            "A man is walking in a room. The man's posture is upright."
        )
        self.assertNotEqual(result.failure_reason, "multiple_subjects")
        self.assertEqual(result.person, "A man")

    def test_multiple_actions_are_kept_in_all_action_conditions(self):
        result = CaptionParser().parse(
            "A man reaches for a joystick in an electronics store. "
            "He is standing near a display and selects the joystick."
        )
        self.assertEqual(result.parse_status, "ok")
        self.assertIsNone(result.failure_reason)
        self.assertIn("reaches", result.action)
        self.assertIn("standing", result.action)
        self.assertIn("selects", result.action)
        self.assertEqual(sum(span.field == "action" for span in result.spans), 3)
        self.assertEqual(result.conditions["PAB"], result.caption)
        for verb in ("reaches", "standing", "selects"):
            for name in ("A", "PA", "AB", "PAB"):
                self.assertIn(verb, result.conditions[name])
            for name in ("empty", "P", "B", "PB"):
                self.assertNotIn(verb, result.conditions[name])
        self.assertTrue(all(a.end <= b.start for a, b in zip(result.spans, result.spans[1:])))

    def test_missing_action_still_fails(self):
        result = CaptionParser().parse("A man in a park.")
        self.assertEqual(result.failure_reason, "action_missing")
        self.assertEqual(result.conditions, {})



try:
    import torch
    from stage3.decomposition import decompose_conditions, reconstruct_full_condition
    from stage3.prediction import linear_flow_difference_to_score, linear_flow_velocity_to_score
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the lightweight test environment")
class Stage3TensorTests(unittest.TestCase):
    def test_inclusion_exclusion_reconstructs_full_condition(self):
        outputs = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        components = decompose_conditions(outputs)
        reconstructed = reconstruct_full_condition(components)
        self.assertTrue(torch.equal(reconstructed, outputs[7]))

    def test_linear_flow_difference_does_not_repeat_common_term(self):
        noisy = torch.tensor([[3.0]])
        t = torch.tensor([0.25])
        first = torch.tensor([[1.0]])
        second = torch.tensor([[2.0]])
        difference = linear_flow_velocity_to_score(first, noisy, t) - linear_flow_velocity_to_score(second, noisy, t)
        direct = linear_flow_difference_to_score(first - second, t)
        self.assertTrue(torch.allclose(difference, direct))


if __name__ == "__main__":
    unittest.main()
