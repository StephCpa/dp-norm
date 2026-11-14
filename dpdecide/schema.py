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


def parse_partition(spec: str) -> tuple[str, float, float]:
    """'salt:lo:hi' -> (salt, lo, hi) with 0 <= lo < hi <= 1."""
    salt, lo, hi = spec.rsplit(":", 2)
    lo, hi = float(lo), float(hi)
    if not (0.0 <= lo < hi <= 1.0) or not salt:
        raise ValueError(f"bad partition {spec!r}; want SALT:LO:HI with 0 <= LO < HI <= 1")
    return salt, lo, hi


def unit_bucket(key: str, salt: str) -> float:
    """A unit's position in [0, 1), a function of its own key and a public salt only."""
    h = hashlib.sha256(f"{salt}\x00{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def select_partition(rows: list, unit: str, spec: str) -> list:
    """Rows whose privacy unit hashes into [lo, hi). Unlike positional slicing (--start/--limit), adding or
    removing one unit of the raw file changes only that unit's membership, so unit-level adjacency on the
    raw file is preserved, and two disjoint ranges with the same salt and unit definition never share a
    unit (parallel composition is valid)."""
    salt, lo, hi = parse_partition(spec)
    return [q for q in rows if lo <= unit_bucket(unit_key(q, unit), salt) < hi]


def partition_record(data: str, unit: str, spec: str, start: int, limit: int) -> dict:
    """Public description of how the private slice was cut from the raw file (released in the ledgers)."""
    import os
    if spec:
        salt, lo, hi = parse_partition(spec)
        return dict(scheme="hash", data=os.path.basename(data), unit=unit, salt=salt, lo=lo, hi=hi,
                    start=start, limit=limit)
    return dict(scheme="positional", data=os.path.basename(data), unit=unit, start=start, limit=limit)


def partitions_disjoint(a: dict | None, b: dict | None) -> bool:
    """True only when disjointness is provable from the public records: both hash partitions of the same
    file, unit definition, salt and positional pre-slice, with non-overlapping ranges."""
    if not a or not b or a.get("scheme") != "hash" or b.get("scheme") != "hash":
        return False
    same = all(a.get(k) == b.get(k) for k in ("data", "unit", "salt", "start", "limit"))
    return same and (a["hi"] <= b["lo"] or b["hi"] <= a["lo"])


def context_key(q: Question) -> str:
    return hashlib.sha256(q.context.strip().encode("utf-8")).hexdigest()


def unit_key(q: Question, unit: str) -> str:
    """The privacy unit a row belongs to. 'context': every row sharing a context (a triage ticket's
    three questions, BoolQ questions on one passage) is one unit, which is what a person's record
    is. 'row': each row on its own (only correct when contexts never repeat)."""
    if unit == "context":
        return context_key(q)
    if unit == "row":
        # Never hash the label: under label DP (neighbours differ only in labels) a key that depends on the
        # answer makes partition membership and unit grouping depend on the protected label before RR
        # (reaudit of df9bb65, P0). Without an id, the key is a hash of the inputs only; rows with identical
        # inputs then share one unit. A supplied id must itself be label-independent (stated precondition).
        if q.id:
            return q.id
        inputs = json.dumps([q.primitive, q.context, q.question, list(q.choices), q.source], ensure_ascii=False)
        return hashlib.sha256(inputs.encode("utf-8")).hexdigest()
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
