"""Synthetic triage tickets for CPU smoke tests: each ticket is one context with three typed
questions (department choice, priority score, paging noul), ~30% of tickets cue two departments."""
from __future__ import annotations

import random

from dpdecide.schema import Question

DEPTS = ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"]
PRIOS = ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]
DEPT_CUES = {
    "BILLING": ["charged twice on the invoice", "refund not received", "card declined at renewal"],
    "INFRASTRUCTURE": ["API latency spiked", "load balancer returning errors", "database replica lagging"],
    "SECURITY": ["suspicious login from new country", "leaked API key reported", "phishing email received"],
    "PRODUCT_SUPPORT": ["cannot find export button", "dashboard widget looks wrong", "how to invite teammates"],
}
PRIO_CUES = {"P3_LOW": ["whenever convenient"], "P2_NORMAL": ["this week please"],
             "P1_HIGH": ["blocking several users"], "P0_CRITICAL": ["all customers affected"]}
TIERS = ["free tier", "startup plan", "enterprise tier one", "enterprise tier two"]


def tickets(n: int, seed: int = 0, prefix: str = "t") -> list[Question]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d = rng.choice(DEPTS)
        p = rng.randrange(4)
        cues = [rng.choice(DEPT_CUES[d])]
        label = d
        if rng.random() < 0.3:
            other = rng.choice([x for x in DEPTS if x != d])
            cues.append(rng.choice(DEPT_CUES[other]))
            if rng.random() < 0.4:
                label = other
        rng.shuffle(cues)
        text = (f"Ticket number {prefix}{i} from a {rng.choice(TIERS)} customer . Report : {' ; '.join(cues)} . "
                f"Note : {rng.choice(PRIO_CUES[PRIOS[p]])} .")
        esc = p >= 2
        if rng.random() < 0.1:
            esc = not esc
        rid = f"{prefix}{i}"
        out += [Question("choice", text, "Which department should handle this ticket?", list(DEPTS), False,
                         DEPTS.index(label), "triage", rid + "-dept"),
                Question("score", text, "What is the priority of this ticket?", list(PRIOS), True, p, "triage",
                         rid + "-prio"),
                Question("noul", text, "Should an on-call engineer be paged immediately?", ["true", "false"], False,
                         0 if esc else 1, "triage", rid + "-esc")]
    return out
