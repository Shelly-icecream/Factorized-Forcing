import json
from pathlib import Path
import tempfile
import unittest

from prompt_processing.parser import parse_jsonl
from prompt_processing.spacy_parser import SpacyCaptionParser, SPACY_PARSER_VERSION


class SpacyParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.parser = SpacyCaptionParser()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    def test_word_senses_do_not_become_actions(self):
        for caption in (
            "A woman in a dress is walking in a clean room.",
            "A close-up shot of a man walking in a park.",
            "A man in a blue work uniform is walking in a park.",
        ):
            with self.subTest(caption=caption):
                result = self.parser.parse(caption)
                self.assertEqual(result.parse_status, "ok")
                self.assertIn("walking", result.action)
                for word in ("dress", "clean", "close", "work"):
                    self.assertNotIn(word, result.action)

    def test_lemma_recognizes_unlisted_inflections(self):
        for verb in ("jogging", "clapping", "hopping"):
            with self.subTest(verb=verb):
                result = self.parser.parse(f"A man is {verb} in a park.")
                self.assertEqual(result.parse_status, "ok")
                self.assertIn(verb, result.action)

    def test_camera_motion_is_shared_context(self):
        caption = "A man is walking in a park. The camera pans slowly."
        result = self.parser.parse(caption)
        self.assertEqual(result.parse_status, "ok")
        self.assertNotIn("pans", result.action)
        self.assertIn("pans", result.conditions["empty"])
        self.assertEqual(result.conditions["PAB"], caption)
        self.assertEqual(self.parser.parse("A camera pans over a mountain.").conditions, {})

    def test_standalone_background_and_unknown_subject(self):
        for caption, person, background in (
            ("A man is walking. The kitchen is warm and inviting.", "man", "kitchen"),
            ("Thanos is walking in a park.", "Thanos", "park"),
            ("A ghost is dancing in a room.", "ghost", "room"),
        ):
            with self.subTest(caption=caption):
                result = self.parser.parse(caption)
                self.assertEqual(result.parse_status, "ok")
                self.assertIn(person, result.person)
                self.assertIn(background, result.background)

    def test_action_objects_and_all_conditions(self):
        caption = "A man is holding a cup in a park. He smiles."
        result = self.parser.parse(caption)
        self.assertEqual(result.parse_status, "ok")
        self.assertIn("a cup", result.action)
        for word in ("holding", "cup", "smiles"):
            for name in ("A", "PA", "AB", "PAB"):
                self.assertIn(word, result.conditions[name])
            for name in ("empty", "P", "B", "PB"):
                self.assertNotIn(word, result.conditions[name])
        self.assertEqual(result.conditions["PAB"], caption)
        self.assertTrue(all(a.end <= b.start for a, b in zip(result.spans, result.spans[1:])))

    def test_offsets_use_original_case_and_unicode(self):
        caption = "An Élite woman is walking in a park."
        result = self.parser.parse(caption)
        self.assertEqual(result.parse_status, "ok")
        for span in result.spans:
            self.assertEqual(caption[span.start:span.end], span.text)

    def test_negation_and_multiple_backgrounds_still_fail(self):
        self.assertEqual(self.parser.parse("A man is not walking in a park.").failure_reason,
                         "negation_not_supported")
        self.assertEqual(self.parser.parse("A man is walking on a street near a park.").failure_reason,
                         "multiple_backgrounds")

    def test_generic_background_and_scenic_modifiers_do_not_make_valid_factors(self):
        result = self.parser.parse("A man is walking. The background is blurred.")
        self.assertEqual(result.failure_reason, "background_missing")
        result = self.parser.parse("A scene of rolling hills in a meadow.")
        self.assertNotEqual(result.parse_status, "ok")
        self.assertEqual(result.action, "")

    def test_scenery_nouns_do_not_replace_the_action_subject(self):
        result = self.parser.parse(
            "A village with towering trees. A gnome is playing in a park."
        )
        self.assertEqual(result.parse_status, "ok")
        self.assertIn("gnome", result.person)
        self.assertNotIn("trees", result.person)

    def test_action_adverbs_and_conjunctions_do_not_swallow_background(self):
        for caption in (
            "A man walks in a park slowly.",
            "A man is walking in a park and smiling.",
        ):
            with self.subTest(caption=caption):
                result = self.parser.parse(caption)
                self.assertEqual(result.parse_status, "ok")
                self.assertEqual(result.background, "in a park")
                self.assertNotIn("park", result.action)

    def test_workers_use_spacy_and_preserve_order(self):
        captions = ["A man is jogging in a park.", "A camera pans over a mountain."]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "captions.txt"
            single, parallel = Path(directory) / "single.jsonl", Path(directory) / "parallel.jsonl"
            source.write_text("\n".join(captions), encoding="utf-8")
            a = parse_jsonl(source, single, self.parser, "text")
            b = parse_jsonl(source, parallel, self.parser, "text", workers=2, chunksize=1)
            self.assertEqual(a, b)
            self.assertEqual(single.read_bytes(), parallel.read_bytes())
            rows = [json.loads(line) for line in single.read_text().splitlines()]
            self.assertTrue(all(row["parser_version"] == SPACY_PARSER_VERSION for row in rows))
            self.assertEqual(rows[0]["parse_status"], "ok")


if __name__ == "__main__":
    unittest.main()
