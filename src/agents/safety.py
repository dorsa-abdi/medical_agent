import re


class SafetyAgent:

    PATTERNS = {
        "severe breathing difficulty": [
            r"\b(can'?t|cannot|unable to)\s+breathe\b",
            r"\b(gasping|choking)\b",
            r"\bblue (lips|skin)\b",
        ],
        "possible heart emergency": [
            r"\b(chest (pain|pressure|tightness|squeezing))\b",
            r"\bpain (spreading|radiating) to (my |the )?(arm|jaw|neck)\b",
        ],
        "possible stroke signs": [
            r"\bface (droop|drooping)\b",
            r"\bone[- ]sided (weakness|numbness)\b",
            r"\b(slurred speech|can'?t speak|cannot speak)\b",
        ],
        "loss of consciousness": [r"\b(unconscious|passed out|fainted and.*not waking)\b"],
        "severe bleeding": [r"\b(heavy|severe|uncontrolled) bleeding\b"],
        "immediate self-harm risk": [
            r"\b(kill myself|end my life|suicide|hurt myself)\b",
            r"\boverdose\b",
        ],
        "severe allergic reaction": [
            r"\b(swollen|swelling) (tongue|throat)\b",
            r"\banaphylaxis\b",
        ],
    }

    def screen(self, text: str) -> list[str]:
        normalized = text.casefold()
        return [
            label
            for label, patterns in self.PATTERNS.items()
            if any(re.search(pattern, normalized) for pattern in patterns)
        ]
