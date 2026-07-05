from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    family: str
    char_start: int
    char_end: int
    boundary_char: int
    text: str


_CONTRAST = re.compile(
    r"""(?ix)
    (?P<span>
      (?:(?<=^)|(?<=[.!?;:\n]))\s*
      [^.!?\n]{0,140}?
      (?:
        \b(?:is|are|was|were|does|do|did|means?|involves?|requires?)\s+
        (?:not(?!\s+only\b)|never)\b
        |\b(?:isn['’]t|aren['’]t|wasn['’]t|weren['’]t|doesn['’]t|don['’]t|didn['’]t)\b
      )
      [^.!?\n]{0,100}?
      (?P<pivot>
        \bbut\b|\brather\b|\binstead\b
        |(?:\bit\b|\bthis\b|\bthat\b)\s+(?:is|means)\b
        |(?:\bit\b|\bthis\b|\bthat\b)['’]s\b
      )
    )
    """
)
_ENUMERATIVE = re.compile(
    r"""(?ix)
    (?P<span>
      (?:(?<=^)|(?<=[.!?;\n]))\s*
      (?:
        there\s+(?:are|is)\s+
        |(?:this|it)\s+(?:happens?|works?|matters?|occurs?)\s+(?:for|because\s+of)\s+
        |[^.!?\n:]{1,60}?\s+(?:is\s+so|happens?|occurs?|arises?)\s+(?:for|because\s+of)\s+
        |(?:the\s+)?(?:answer|explanation|reason|cause|process)\s+(?:has|involves?)\s+
      )
      (?:(?:two|three|four|five|several|a\s+few|multiple|\d+)\s+)?
      (?:main\s+|key\s+|basic\s+|distinct\s+)?
      (?:reasons?|causes?|factors?|parts?|steps?|things?|considerations?|mechanisms?)
      \s*(?P<boundary>:\s*|\.\s+(?=(?:first(?:ly)?|one)\b))
    )
    """
)
_REFRAMING = re.compile(
    r"""(?ix)
    (?P<span>
      \b(?:rather\s+than|instead\s+of)\b
      [^.!?;\n]{1,120}?
      (?P<pivot>,|;)\s*
    )
    |
    (?P<scaled>
      \b(?:is|are|was|were)\s+
      (?:not\s+so\s+much|less)\s+
      [^.!?;\n]{1,100}?
      (?P<scaled_pivot>\bas\b|\b(?:and\s+)?more\b)
    )
    """
)
_CONCESSIVE = re.compile(
    r"""(?ix)
    (?P<span>
      \b(?:although|though|even\s+though|while|admittedly)\b
      [^.!?;\n]{1,140}?
      (?P<pivot>,|;)\s*
    )
    |
    (?P<appearance>
      \b(?:it|this|that)\s+(?:may|might|can|could)\s+
      (?:seem|appear)\b
      [^.!?;\n]{1,120}?
      (?P<appearance_pivot>\bbut\b|\bhowever\b|,|;)
    )
    """
)


def detect(text: str) -> list[Candidate]:
    hits: list[Candidate] = []
    for match in _CONTRAST.finditer(text):
        start, end = match.span("span")
        pivot_end = match.end("pivot")
        hits.append(
            Candidate("contrastive_negation", start, end, pivot_end, text[start:end])
        )
    for match in _ENUMERATIVE.finditer(text):
        start, end = match.span("span")
        hits.append(
            Candidate("enumerative_preamble", start, end, end, text[start:end])
        )
    for match in _REFRAMING.finditer(text):
        group = "span" if match.group("span") is not None else "scaled"
        pivot = "pivot" if group == "span" else "scaled_pivot"
        start, end = match.span(group)
        hits.append(
            Candidate(
                "contrastive_negation",
                start,
                end,
                match.end(pivot),
                text[start:end],
            )
        )
    for match in _CONCESSIVE.finditer(text):
        group = "span" if match.group("span") is not None else "appearance"
        pivot = "pivot" if group == "span" else "appearance_pivot"
        start, end = match.span(group)
        hits.append(
            Candidate(
                "concessive_qualification",
                start,
                end,
                match.end(pivot),
                text[start:end],
            )
        )
    ordered = sorted(hits, key=lambda item: (item.char_start, item.char_end))
    deduplicated: list[Candidate] = []
    for hit in ordered:
        if any(
            hit.family == prior.family
            and hit.char_start < prior.char_end
            and prior.char_start < hit.char_end
            for prior in deduplicated
        ):
            continue
        deduplicated.append(hit)
    return deduplicated


def token_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise AssertionError("Token IDs and offsets differ in length")
    return ids, offsets


def align_candidate(tokenizer: Any, text: str, hit: Candidate) -> dict[str, int]:
    ids, offsets = token_offsets(tokenizer, text)
    overlapping = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > hit.char_start and start < hit.char_end
    ]
    if not overlapping:
        raise ValueError("Detected character span maps to no tokens")
    token_start, token_end = overlapping[0], overlapping[-1] + 1
    boundary = next(
        (index for index, (start, _) in enumerate(offsets) if start >= hit.boundary_char),
        len(ids),
    )
    decoded = tokenizer.decode(
        ids[token_start:token_end],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    reencoded = tokenizer.encode(decoded, add_special_tokens=False)
    if list(reencoded) != ids[token_start:token_end]:
        raise AssertionError("Aligned token slice does not exactly re-tokenize")
    if offsets[token_start][0] > hit.char_start or offsets[token_end - 1][1] < hit.char_end:
        raise AssertionError("Aligned token slice does not cover character span")
    return {
        "token_start": token_start,
        "token_end": token_end,
        "boundary_token": boundary,
    }
