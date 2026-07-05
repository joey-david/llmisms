from __future__ import annotations

import hashlib

from .records import FILES, write_json, write_records


TOPICS = {
    "conceptual_ambiguity": [
        ("fairness", "a decision can be fair even when outcomes differ"),
        ("intelligence", "expertise and intelligence come apart"),
        ("creativity", "constraints sometimes improve creativity"),
        ("privacy", "privacy can matter without secrecy"),
        ("identity", "a person's identity persists through change"),
        ("trust", "trust differs from mere predictability"),
        ("freedom", "more options can reduce practical freedom"),
        ("objectivity", "objectivity can coexist with perspective"),
        ("authenticity", "authenticity does not require spontaneity"),
        ("understanding", "understanding differs from accurate prediction"),
        ("responsibility", "responsibility can be shared without vanishing"),
        ("meaning", "meaning can emerge without deliberate design"),
        ("rationality", "rational choices can lead to poor group outcomes"),
        ("progress", "technical progress can coexist with social loss"),
        ("simplicity", "a simple explanation can still be deep"),
    ],
    "causal_explanation": [
        ("metal rusts faster near the sea", "chemistry"),
        ("crowds sometimes move in waves", "collective behavior"),
        ("sleep loss impairs judgment", "cognition"),
        ("cities develop heat islands", "urban climate"),
        ("prices rise after supply shocks", "economics"),
        ("rumors spread rapidly online", "social dynamics"),
        ("leaves change color in autumn", "biology"),
        ("batteries degrade over repeated use", "materials"),
        ("languages split into dialects", "linguistics"),
        ("traffic jams appear without accidents", "transport"),
        ("some memories become distorted", "psychology"),
        ("coastal fog forms in summer", "weather"),
        ("teams become overconfident", "organization"),
        ("bread becomes stale", "food science"),
        ("antibiotic resistance spreads", "evolution"),
    ],
    "advice_tradeoffs": [
        ("choose between a stable job and a risky project", "career"),
        ("decide whether to move closer to work", "housing"),
        ("handle a disagreement with a close collaborator", "relationships"),
        ("decide when to replace an old laptop", "purchasing"),
        ("balance focused work with availability", "productivity"),
        ("choose how much emergency savings to keep", "finance"),
        ("decide whether to learn a new programming language", "learning"),
        ("respond to repeated meeting overload", "work"),
        ("choose between repairing and replacing an appliance", "household"),
        ("decide whether to accept a leadership role", "career"),
        ("balance exercise intensity with recovery", "health"),
        ("choose whether to travel during peak season", "travel"),
        ("handle a project whose requirements keep changing", "engineering"),
        ("decide whether to publish preliminary work", "research"),
        ("balance convenience with digital privacy", "technology"),
    ],
    "direct_factual": [
        ("How does a heat pump move heat into a house?", "physics"),
        ("Why does salt lower the freezing point of water?", "chemistry"),
        ("How do vaccines create immune memory?", "biology"),
        ("What determines the length of a solar day?", "astronomy"),
        ("How does a compiler turn source code into a program?", "computing"),
        ("Why do central banks change interest rates?", "economics"),
        ("How does GPS calculate a receiver's location?", "engineering"),
        ("Why do some materials conduct electricity?", "physics"),
        ("How does a bill become law in France?", "civics"),
        ("What causes ocean tides?", "astronomy"),
        ("How does public-key encryption protect a message?", "computing"),
        ("Why does yeast make bread rise?", "biology"),
        ("How do historians date ancient artifacts?", "history"),
        ("What makes an earthquake produce a tsunami?", "geology"),
        ("How does a telescope resolve distant objects?", "optics"),
    ],
}


def build_prompts() -> list[dict]:
    rows: list[dict] = []
    for stratum, topics in TOPICS.items():
        for topic_index, (topic, domain) in enumerate(topics):
            if stratum == "conceptual_ambiguity":
                variants = (
                    f"Explain what is at stake in the claim that {domain}.",
                    f"How should someone interpret the idea that {domain}?",
                    f"What subtlety is easy to miss when discussing {topic}?",
                    f"What does {topic} mean in this context: {domain}?",
                )
            elif stratum == "causal_explanation":
                variants = (
                    f"Explain why {topic}.",
                    f"What process makes it the case that {topic}?",
                    f"What explains the fact that {topic}?",
                    f"What is happening when {topic}, viewed through {domain}?",
                )
            elif stratum == "advice_tradeoffs":
                variants = (
                    f"How should someone {topic}?",
                    f"What should a person think through when they need to {topic}?",
                    f"Give practical guidance to someone trying to {topic}.",
                    f"What is a sound way to reason about this {domain} decision: {topic}?",
                )
            else:
                variants = (
                    topic,
                    f"What is the explanation for this question: {topic}",
                    f"What facts are needed to answer this question: {topic}",
                    f"What mechanism answers this {domain} question: {topic}",
                )
            for variant_index, text in enumerate(variants):
                prompt_id = f"{stratum}-{topic_index:02d}-{variant_index}"
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "stratum": stratum,
                        "text": text,
                        "topic_index": topic_index,
                        "variant_index": variant_index,
                    }
                )
    assert len(rows) == 240
    assert len({row["text"] for row in rows}) == 240
    return rows


def write_corpus(output, limit: int | None = None) -> list[dict]:
    rows = build_prompts()
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError(f"limit must be between 1 and {len(rows)}")
        # Round-robin selection keeps validation subsets as balanced as possible.
        buckets = {
            stratum: [row for row in rows if row["stratum"] == stratum]
            for stratum in TOPICS
        }
        selected = []
        index = 0
        while len(selected) < limit:
            for stratum in TOPICS:
                if len(selected) == limit:
                    break
                selected.append(buckets[stratum][index])
            index += 1
        rows = selected
    write_records(output / FILES["prompts"], rows)
    write_json(
        output / "corpus.json",
        {
            "count": len(rows),
            "sha256": hashlib.sha256(
                "\n".join(row["text"] for row in rows).encode()
            ).hexdigest(),
            "strata": {s: sum(r["stratum"] == s for r in rows) for s in TOPICS},
            "frozen": True,
            "single_turn": True,
            "language": "en",
        },
    )
    return rows
