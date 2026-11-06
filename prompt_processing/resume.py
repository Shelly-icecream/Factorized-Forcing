"""Validate ordered JSONL prefixes and recover an interrupted final write."""
from collections import Counter
from contextlib import contextmanager
import json


@contextmanager
def resume_output(path):
    # Lock the actual output before validation or truncation. Older versions of
    # the writer do not take this lock and must be stopped before restarting.
    with open(path, "a+b") as raw:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            try:
                fcntl.flock(raw.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another parser is already writing {path}") from exc
        try:
            yield raw
        finally:
            if fcntl is not None:
                fcntl.flock(raw.fileno(), fcntl.LOCK_UN)


def validate_prefix(target, captions, parser_version):
    """Consume matching input rows; modify output only after prefix validation."""
    counts = Counter()
    target.seek(0)
    good_end = 0
    needs_newline = False
    row = 0
    while True:
        line = target.readline()
        if not line:
            break
        row += 1
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if not line.endswith(b"\n"):
                # readline consumed the unterminated EOF fragment; reparse its
                # input caption rather than counting a partially written result.
                break
            raise ValueError(f"invalid existing JSONL at output row {row}") from exc
        if not isinstance(item, dict) or item.get("parse_status") not in {"ok", "ambiguous", "unsupported"}:
            raise ValueError(f"invalid existing parsed result at output row {row}")
        if item.get("parser_version") != parser_version:
            raise ValueError(f"parser version mismatch at output row {row}; use the original parser or a new output")
        try:
            caption = next(captions)
        except StopIteration as exc:
            raise ValueError("existing output contains more rows than the input") from exc
        if item.get("caption") != caption:
            raise ValueError(f"input/output caption mismatch at output row {row}; refusing to append")
        counts[item["parse_status"]] += 1
        good_end = target.tell()
        needs_newline = not line.endswith(b"\n")
    target.seek(good_end)
    target.truncate()
    if needs_newline:
        target.write(b"\n")
    target.flush()
    return counts


class JsonlTextWriter:
    """Write UTF-8 text to a locked binary output, flushing completed rows."""
    def __init__(self, raw):
        self.raw = raw

    def write(self, text):
        return self.raw.write(text.encode("utf-8"))

    def flush(self):
        self.raw.flush()
