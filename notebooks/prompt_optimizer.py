# prompt + few-shot + CoT
BASE_SYSTEM_PROMPT = """
You are an expert rater of perceived emotions in short text.
Emotions: anger, fear, joy, sadness, surprise.

Task: For each emotion, output 0 or 1 indicating whether the emotion is present (1) or absent (0).

Think step-by-step to find emotion cues, but output ONLY the final JSON.

Return ONLY a JSON object with exactly these keys and integer values:
{"anger":0,"fear":0,"joy":0,"sadness":0,"surprise":0}

Text: "{TEXT}"
"""

# Chain-of-Thought scaffold (kept concise)
COT_PROMPT = """Steps:
1) Identify explicit or implicit emotion cues.
2) For each emotion, decide presence (0/1) and briefly reason internally.
3) Output ONLY the final JSON (no text before or after).
"""

# Few-shot examples in the 0/1 schema
FEW_SHOT_PROMPT = """Examples:
Text: "Colorado, middle of nowhere."
→ {"anger":0,"fear":1,"joy":0,"sadness":0,"surprise":1}

Text: "Hondas are notoriously great cars for long trips for their dependability and great gas mileage."
→ {"anger":0,"fear":0,"joy":1,"sadness":0,"surprise":0}

Text: "Not only was I not able to move, I smacked my head against the guy sitting in front me and things just got awkward."
→ {"anger":1,"fear":1,"joy":0,"sadness":0,"surprise":0}
"""

PROMPTS = {
    "BASE_SYSTEM_PROMPT": BASE_SYSTEM_PROMPT,
    "FEW_SHOT_PROMPT": BASE_SYSTEM_PROMPT + FEW_SHOT_PROMPT,
    "COT_PROMPT": BASE_SYSTEM_PROMPT + COT_PROMPT,
    "COMBINED_PROMPT": BASE_SYSTEM_PROMPT + COT_PROMPT + FEW_SHOT_PROMPT,
}