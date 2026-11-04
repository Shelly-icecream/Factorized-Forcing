"""CPU-only syntax-assisted parsing; the JSON vocabulary remains authoritative."""

from .parser import CaptionParser

SPACY_PARSER_VERSION = "spacy_en_v1_multi_action"
DEFAULT_SPACY_MODEL = "en_core_web_sm"


class SpacyCaptionParser(CaptionParser):
    def __init__(self, rules=None, parser_version=SPACY_PARSER_VERSION,
                 model=DEFAULT_SPACY_MODEL):
        super().__init__(rules, parser_version)
        try:
            import spacy
            self.nlp = spacy.load(model, exclude=["ner"])
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "spaCy English pipeline is required. Install stage3/requirements-spacy.txt "
                "and run: python -m spacy download en_core_web_sm"
            ) from exc
        self.model = model
        self._doc = None
        self._action_lemmas = {phrase.split()[0].casefold()
                               for phrase, _ in self.rules["action"]}
        self._background_lemmas = {phrase.casefold() for phrase, rule in self.rules["background"]
                                   if not rule.endswith("weather")}

    def parse(self, caption):
        # Preserve original case for tagging, and original character offsets for spans.
        self._doc = self.nlp(caption)
        try:
            return super().parse(caption)
        finally:
            self._doc = None

    def _chunk(self, token):
        return next((chunk for chunk in self._doc.noun_chunks if chunk.root == token),
                    self._doc[token.i:token.i + 1])

    @staticmethod
    def _subjects(verb):
        subjects = [child for child in verb.children
                    if child.dep_ in {"nsubj", "nsubjpass"}]
        if not subjects and verb.dep_ in {"conj", "xcomp", "advcl"} and verb.head != verb:
            return SpacyCaptionParser._subjects(verb.head)
        return subjects

    def _camera_action(self, verb):
        production = {"camera", "shot", "view", "video", "scene", "image", "frame", "film"}
        if any(subject.lemma_.casefold() in production for subject in self._subjects(verb)):
            return True
        if verb.dep_ in {"acl", "relcl"} and verb.head.lemma_.casefold() in production:
            return True
        return False

    def _action_matches(self, text):
        if self._doc is None:
            return super()._action_matches(text)
        matches = []
        for token in self._doc:
            if token.pos_ != "VERB" or token.lemma_.casefold() not in self._action_lemmas:
                continue
            if self._camera_action(token) or token.dep_ == "amod":
                continue
            if token.dep_ == "acl" and token.head.lemma_.casefold() in self._background_lemmas:
                continue
            indices = {token.i}
            for child in token.children:
                if child.dep_ in {"aux", "auxpass", "neg", "prt"}:
                    indices.add(child.i)
                elif child.dep_ in {"dobj", "obj", "dative", "attr", "oprd"}:
                    indices.update(t.i for t in self._chunk(child))
            # A trailing adverb may follow a location ("walks in a park slowly").
            # Only adjacent adverbs belong in this contiguous action span.
            adverbs = {child.i for child in token.children if child.dep_ == "advmod"}
            while min(indices) - 1 in adverbs or max(indices) + 1 in adverbs:
                if min(indices) - 1 in adverbs:
                    indices.add(min(indices) - 1)
                if max(indices) + 1 in adverbs:
                    indices.add(max(indices) + 1)
            start, end = min(indices), max(indices) + 1
            # Never swallow another clause or subject across an argument boundary.
            if any(t.dep_ in {"nsubj", "nsubjpass"} for t in self._doc[start:end]):
                start, end = token.i, token.i + 1
            matches.append((self._doc[start].idx,
                            self._doc[end - 1].idx + len(self._doc[end - 1]),
                            "action_spacy_" + token.lemma_.casefold()))
        import re
        merged = []
        for current in sorted(matches):
            if merged and (current[0] < merged[-1][1] or re.fullmatch(
                r"(?:\s|,|\band\b|\bwhile\b|\bthen\b)+",
                text[merged[-1][1]:current[0]],
            )):
                previous = merged[-1]
                merged[-1] = (previous[0], max(previous[1], current[1]), "action_spacy_combined")
            else:
                merged.append(current)
        return merged

    def _background_matches(self, text, actions):
        if self._doc is None:
            return super()._background_matches(text, actions)
        # Known location noun phrases also count without a leading preposition,
        # e.g. "The kitchen is warm". Do not infer locations from arbitrary nouns.
        matches = []
        for chunk in self._doc.noun_chunks:
            known = self._matches(chunk.text.casefold(), self.rules["background"])
            if chunk.root.lemma_.casefold() == "background":
                continue
            if not any(chunk.start_char + end == chunk.root.idx + len(chunk.root)
                       and not rule.endswith("weather") for _, end, rule in known):
                continue
            if chunk.root.dep_ not in {"pobj", "nsubj", "attr", "nsubjpass"}:
                continue
            start, end = chunk.start_char, chunk.end_char
            if chunk.root.dep_ == "pobj":
                prep = chunk.root.head
                if prep.lemma_.casefold() not in {
                    "in", "on", "at", "through", "near", "by", "under", "over",
                    "inside", "outside", "along", "across", "within", "beside", "behind",
                }:
                    continue
                if prep.head.pos_ == "NOUN" and prep.head.lemma_.casefold() in {
                    "portrait", "photo", "photograph", "view", "shot",
                }:
                    continue
                start = prep.idx
            if not any(self._overlap((start, end), action) for action in actions):
                matches.append((start, end, "background_spacy_noun_phrase"))
        # Preserve the established location selection. Syntax recovers noun phrases
        # only when rules found no concrete scene; repeated descriptive mentions
        # must not create new backgrounds merely because they are grammatical subjects.
        legacy = []
        for old in super()._background_matches(text, actions):
            phrase = text[old[0]:old[1]]
            if phrase.strip().casefold() in {"background", "the background", "in the background"}:
                continue
            overlap = next((match for match in matches if self._overlap(old, match)), None)
            legacy.append(overlap or old)
        return sorted(set(legacy)) if legacy else sorted(set(matches))

    def _action_subjects(self, actions):
        subjects = []
        for verb in self._doc:
            if verb.pos_ != "VERB" or not any(start <= verb.idx < end for start, end, _ in actions):
                continue
            subjects.extend(self._subjects(verb))
            # Reduced/relative clauses: "a ghost dancing", "a man who walks".
            if verb.dep_ in {"acl", "relcl"} and verb.head.pos_ in {"NOUN", "PROPN"}:
                subjects.append(verb.head)
        return subjects

    def _subject_matches(self, text, actions, backgrounds):
        candidates = super()._subject_matches(text, actions, backgrounds)
        if self._doc is None:
            return candidates
        actors = self._action_subjects(actions)

        def supported(candidate):
            for actor in actors:
                # Include noun modifiers and "group of people", but never a
                # separate verbal clause hanging off the nominal subject.
                tokens = [actor]
                for token in actor.subtree:
                    path = []
                    node = token
                    while node != actor and node.head != node:
                        path.append(node)
                        node = node.head
                    if node == actor and not any(t.pos_ == "VERB" for t in path):
                        tokens.append(token)
                if any(candidate[0] <= token.idx < candidate[1]
                       and token.pos_ in {"NOUN", "PROPN", "PRON"}
                       and token.lemma_.casefold() != "camera" for token in tokens):
                    return True
            return False

        candidates = [candidate for candidate in candidates if supported(candidate)]
        if candidates:
            return candidates
        # Recover unknown nouns/names only as subjects of accepted actions.
        recovered = []
        for subject in actors:
            if subject.pos_ not in {"NOUN", "PROPN"} or subject.lemma_.casefold() in {
                "camera", "scene", "video", "image", "shot", "view", "frame", "film",
            }:
                continue
            chunk = self._chunk(subject)
            span = (chunk.start_char, chunk.end_char, "person_spacy_subject")
            if not any(self._overlap(span, other) for other in actions + backgrounds):
                recovered.append(span)
        return sorted(set(recovered))

    def _expand_person(self, caption, start, end, action_start, backgrounds=()):
        start, expanded_end = super()._expand_person(caption, start, end, action_start, backgrounds)
        if self._doc is not None:
            for chunk in self._doc.noun_chunks:
                if chunk.start_char <= end - 1 < chunk.end_char:
                    candidate_end = max(expanded_end, chunk.end_char)
                    if candidate_end <= action_start and not any(
                        self._overlap((start, candidate_end), background) for background in backgrounds
                    ):
                        expanded_end = candidate_end
                    break
        return start, expanded_end


def make_parser(backend="spacy", rules_path=None, parser_version=None,
                model=DEFAULT_SPACY_MODEL):
    if backend not in {"rules", "spacy"}:
        raise ValueError("backend must be rules or spacy")
    if backend == "rules":
        from .parser import DEFAULT_PARSER_VERSION
        version = parser_version or DEFAULT_PARSER_VERSION
        return (CaptionParser.from_json(rules_path, version) if rules_path
                else CaptionParser(parser_version=version))
    import json
    rules = None
    if rules_path:
        with open(rules_path, encoding="utf-8") as handle:
            rules = json.load(handle)
    return SpacyCaptionParser(rules, parser_version or SPACY_PARSER_VERSION, model)
