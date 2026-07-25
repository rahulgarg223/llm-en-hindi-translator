"""
Prompt templates for English -> Hindi translation using an
instruction-tuned LLM (few-shot / prompt-based approach).
"""

SYSTEM_PROMPT = (
    "You are a professional English-to-Hindi translator. "
    "Translate the given text into fluent, natural, grammatically correct Hindi "
    "(Devanagari script). Preserve proper nouns, numbers, and technical terms "
    "appropriately. Match the register (formal/informal) of the source text. "
    "When transliterating personal names, use the minimal, most common Hindi "
    "spelling — do not insert extra vowel sounds between consonants that are "
    "pronounced together (e.g. 'Garg' -> 'गर्ग', not 'गार्ग'). "
    "Output ONLY the Hindi translation — no explanations, no transliteration, "
    "no English text, no quotation marks."
)


FEW_SHOT_EXAMPLES = [
    {
        "en": "The meeting is scheduled for 5 PM tomorrow.",
        "hi": "बैठक कल शाम 5 बजे निर्धारित है।",
    },
    {
        "en": "We regret to inform you that your application has been declined.",
        "hi": "हमें आपको यह सूचित करते हुए खेद है कि आपका आवेदन अस्वीकार कर दिया गया है।",
    },
    {
        "en": "It's raining cats and dogs outside.",
        "hi": "बाहर मूसलधार बारिश हो रही है।",
    },
    {
        "en": "Hey, are you free this weekend? Let's catch up.",
        "hi": "अरे, क्या तुम इस वीकेंड फ्री हो? चलो मिलते हैं।",
    },
    {
        "en": "Ravi bought 3 kg of mangoes from the market on Monday.",
        "hi": "रवि ने सोमवार को बाजार से 3 किलो आम खरीदे।",
    },
]


def build_prompt(source_text: str, num_shots: int = 5) -> str:
    """
    Builds the full few-shot prompt string to send as the user turn.
    num_shots controls how many of the curated examples to include
    (useful for zero-shot vs few-shot comparison experiments).
    """
    lines = []
    for ex in FEW_SHOT_EXAMPLES[:num_shots]:
        lines.append(f"EN: {ex['en']}\nHI: {ex['hi']}")

    examples_block = "\n\n".join(lines)

    if examples_block:
        prompt = (
            f"{examples_block}\n\n"
            f"Now translate:\n"
            f"EN: {source_text}\n"
            f"HI:"
        )
    else:
        # zero-shot fallback
        prompt = f"Translate to Hindi:\nEN: {source_text}\nHI:"

    return prompt


def build_chat_messages(source_text: str, num_shots: int = 5) -> list:
    """
    Returns a messages list in the standard chat format
    (system + user turn), ready to pass to an HF chat model.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(source_text, num_shots)},
    ]
