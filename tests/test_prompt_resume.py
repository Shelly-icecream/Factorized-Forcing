import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prompt_processing.parser import CaptionParser, parse_jsonl


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'input.txt'
        self.output = Path(self.tmp.name) / 'output.jsonl'
        self.captions = ['A man walks in a park.', 'A girl runs in a room.', 'A robot stands in a factory.']
        self.source.write_text('\n'.join(self.captions), encoding='utf-8')
        self.parser = CaptionParser()

    def prefix(self, size=1):
        self.output.write_text(''.join(json.dumps(self.parser.parse(c).to_dict())+'\n'
                                       for c in self.captions[:size]), encoding='utf-8')

    def test_skip_prefix_and_repeated_resume(self):
        self.prefix()
        with patch.object(self.parser, 'parse', wraps=self.parser.parse) as parse:
            counts = parse_jsonl(self.source, self.output, self.parser, 'text')
            self.assertEqual(parse.call_count, 2)
        self.assertEqual(sum(counts.values()), 3)
        before = self.output.read_bytes()
        with patch.object(self.parser, 'parse', side_effect=AssertionError('must skip')):
            parse_jsonl(self.source, self.output, self.parser, 'text')
        self.assertEqual(before, self.output.read_bytes())

    def test_incomplete_final_json_and_utf8(self):
        for fragment in (b'{"caption": "A girl', b'{"caption": "\xe4\xbd'):
            with self.subTest(fragment=fragment):
                self.prefix()
                with self.output.open('ab') as handle:
                    handle.write(fragment)
                parse_jsonl(self.source, self.output, self.parser, 'text')
                rows = [json.loads(line) for line in self.output.read_text().splitlines()]
                self.assertEqual([row['caption'] for row in rows], self.captions)

    def test_valid_final_row_without_newline(self):
        self.prefix()
        self.output.write_bytes(self.output.read_bytes().rstrip(b'\n'))
        parse_jsonl(self.source, self.output, self.parser, 'text')
        self.assertEqual(len(self.output.read_text().splitlines()), 3)

    def test_mismatch_does_not_modify_output(self):
        for change in ('caption', 'version', 'corruption', 'extra_rows'):
            with self.subTest(change=change):
                self.prefix()
                if change == 'caption':
                    self.source.write_text('Different input\n', encoding='utf-8')
                elif change == 'version':
                    item = json.loads(self.output.read_text())
                    item['parser_version'] = 'other'
                    self.output.write_text(json.dumps(item)+'\n')
                elif change == 'corruption':
                    self.output.write_text('{broken}\n')
                else:
                    self.source.write_text('', encoding='utf-8')
                before = self.output.read_bytes()
                with self.assertRaises(ValueError):
                    parse_jsonl(self.source, self.output, self.parser, 'text')
                self.assertEqual(before, self.output.read_bytes())
                self.source.write_text('\n'.join(self.captions), encoding='utf-8')

    def test_overwrite_is_explicit(self):
        self.output.write_text('old incompatible data\n')
        parse_jsonl(self.source, self.output, self.parser, 'text', resume=False)
        self.assertEqual(len(self.output.read_text().splitlines()), 3)

    def test_parallel_resume_matches_serial(self):
        self.prefix()
        parse_jsonl(self.source, self.output, self.parser, 'text', workers=2, chunksize=1)
        serial = Path(self.tmp.name) / 'serial.jsonl'
        parse_jsonl(self.source, serial, self.parser, 'text')
        self.assertEqual(self.output.read_bytes(), serial.read_bytes())

    def test_same_file_rejected_without_modification(self):
        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            parse_jsonl(self.source, self.source, self.parser, 'text')
        self.assertEqual(before, self.source.read_bytes())


if __name__ == '__main__':
    unittest.main()
