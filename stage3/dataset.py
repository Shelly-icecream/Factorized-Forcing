"""Validated parsed-caption data with offsets instead of an in-memory corpus."""

from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path

from .conditions import CONDITION_NAMES


class ParsedCaptionDataset:
    def __init__(self, path, parser_version=None):
        self.path = str(Path(path).resolve())
        self.offsets = array("Q")
        self.counts = Counter()
        digest = hashlib.sha256()
        versions = set()
        with open(self.path, "rb") as source:
            line_number = 0
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                line_number += 1
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("expected a parsed-caption object")
                    status = item.get("parse_status", "missing")
                    self.counts[status] += 1
                    if status != "ok":
                        continue
                    self._validate(item, parser_version)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
                self.offsets.append(offset)
                versions.add(item["parser_version"])
        if not self.offsets:
            raise ValueError("no valid parsed captions (parse_status=ok) in dataset")
        self.fingerprint = digest.hexdigest()
        self.parser_versions = sorted(versions)

    @staticmethod
    def _validate(item, parser_version):
        if not isinstance(item.get("caption"), str) or not item["caption"].strip():
            raise ValueError("caption must be a nonempty string")
        conditions = item.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(CONDITION_NAMES):
            raise ValueError("valid rows must contain all eight conditions")
        if not all(isinstance(value, str) for value in conditions.values()):
            raise ValueError("condition texts must be strings")
        if conditions["PAB"] != item["caption"]:
            raise ValueError("PAB must equal the original caption")
        if not isinstance(item.get("parser_version"), str):
            raise ValueError("parser_version is required")
        if parser_version is not None and item["parser_version"] != parser_version:
            raise ValueError("parser version does not match training configuration")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with open(self.path, "rb") as source:
            source.seek(self.offsets[index])
            item = json.loads(source.readline())
        return {"caption": item["caption"], "idx": index,
                "parser_version": item["parser_version"],
                "conditions": {name: item["conditions"][name] for name in CONDITION_NAMES}}


class CaptionStream:
    """Deterministic rank-local ordering with resumable epoch and position."""

    def __init__(self, dataset, rank=0, world_size=1, seed=0):
        from torch.utils.data.distributed import DistributedSampler
        self.dataset = dataset
        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                          seed=seed, shuffle=True, drop_last=False)
        self.epoch = 0
        self.position = 0
        self._iterator = None

    def __next__(self):
        if self._iterator is None:
            from itertools import islice
            self.sampler.set_epoch(self.epoch)
            self._iterator = iter(islice(iter(self.sampler), self.position, None))
        try:
            index = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.position = 0
            self._iterator = None
            return next(self)
        self.position += 1
        return self.dataset[index]

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        if state["epoch"] < 0 or not 0 <= state["position"] <= len(self.sampler):
            raise ValueError("invalid data stream position")
        self.epoch, self.position = state["epoch"], state["position"]
        self._iterator = None
