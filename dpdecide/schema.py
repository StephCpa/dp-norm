"""Typed decision rows, prompt rendering and privacy units.

The row format and the prompt are byte-compatible with eve-rlcd (github.com/anthony-maio/eve-rlcd,
MIT): the same JSONL files and the same checkpoints work in both. Nothing from eve-rlcd is imported.
"""
from __future__ import annotations

import hashlib
import json
import string
from dataclasses import asdict, dataclass
from pathlib import Path

LETTERS: list[str] = list(string.ascii_uppercase)
MAX_CHOICES = 26
NEG = -1e4  # finite mask value for letters beyond a row's option count
PRIMITIVES = ("choice", "score", "noul")


@dataclass
class Question:
    primitive: str
    context: str
    question: str
    choices: list[str]
    ordered: bool = False
    answer: int | None = None
    source: str = ""
    id: str = ""

    @property
    def k(self) -> int:
        return len(self.choices)

    def validate(self) -> "Question":
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {self.primitive!r}")
        n = len(self.choices)
        if n < 2 or n > MAX_CHOICES:
            raise ValueError(f"need 2..{MAX_CHOICES} choices, got {n}")
        if self.answer is not None and not (0 <= self.answer < n):
            raise ValueError(f"answer {self.answer} out of range for {n} choices")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        keep = {k: d[k] for k in ("primitive", "context", "question", "choices", "ordered", "answer", "source", "id")
                if k in d}
        return cls(**keep).validate()


def render_prompt(q: Question) -> str:
    """Identical to eve-rlcd's render_prompt."""
    lines = ["User: Context:", q.context.strip(), "", f"Question: {q.question.strip()}", "Options:"]
    for letter, choice in zip(LETTERS, q.choices):
        lines.append(f"{letter}) {choice}")
    lines.append("Answer with the letter only.")
    lines.append("Assistant: The answer is")
    return "\n".join(lines)


def read_jsonl(path) -> list[Question]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(Question.from_dict(json.loads(line)))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{path}:{i}: {e}") from e
    return out


def write_jsonl(path, questions: list[Question]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.validate().to_json() + "\n")


def select_rows(rows: list, start: int, limit: int) -> list:
    """eve-rlcd's slice convention: rows from start in file order, then the first limit (0 = all)."""
    rows = rows[start:]
    return rows[:limit] if limit > 0 else rows


def context_key(q: Question) -> str:
    return hashlib.sha256(q.context.strip().encode("utf-8")).hexdigest()


def unit_key(q: Question, unit: str) -> str:
    """The privacy unit a row belongs to. 'context': every row sharing a context (a triage ticket's
    three questions, BoolQ questions on one passage) is one unit, which is what a person's record
    is. 'row': each row on its own (only correct when contexts never repeat)."""
    if unit == "context":
        return context_key(q)
    if unit == "row":
        return q.id or hashlib.sha256(q.to_json().encode("utf-8")).hexdigest()
    raise ValueError(f"unknown unit {unit!r}")


def group_units(questions: list[Question], unit: str, max_rows: int) -> tuple[list[list[int]], int]:
    """Indices grouped by privacy unit, in first-appearance order. Units longer than max_rows keep
    their first max_rows rows (file order); the number of dropped rows is returned. The cap bounds
    a unit's contribution to any per-unit sum (DP temperature scaling, label-DP budget split)."""
    groups: dict[str, list[int]] = {}
    for i, q in enumerate(questions):
        groups.setdefault(unit_key(q, unit), []).append(i)
    units, dropped = [], 0
    for idx in groups.values():
        units.append(idx[:max_rows])
        dropped += max(0, len(idx) - max_rows)
    return units, dropped


def parse_k(spec: str):
    """'' -> None (all); '2' -> {2}; '2-5' -> {2,3,4,5}; '2,4' -> {2,4}."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def filter_k(rows: list, spec: str) -> list:
    ks = parse_k(spec)
    return rows if ks is None else [q for q in rows if q.k in ks]
