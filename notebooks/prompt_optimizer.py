# prompt + few-shot + CoT
BASE_SYSTEM_PROMPT = """
You are an expert rater of perceived emotions in short text. Emotions: anger, fear, joy, sadness, surprise.

Task A: For each emotion, output 0/1 (absent/present).
Task B: Also output intensity 0–3 if present.

Think step-by-step to find emotion cues, but only output final JSON.

Rules:
- y: 0/1 presence
- i: 0–3 intensity (0 if y=0)
Return ONLY JSON object: {"anger":{"y":0,"i":0}, ...}

Text: "{TEXT}"
"""

COT_PROMPT = """ Steps:
1) Identify explicit/implicit emotion cues.
2) For each emotion, decide presence (0/1), with a brief reason.
3) If present, assign intensity 1–3; else 0.
"""

FEW_SHOT_PROMPT = """Examples:
Text: "Colorado, middle of nowhere."
→ {"anger":{"y":0,"i":0},"fear":{"y":1,"i":1},"joy":{"y":0,"i":0},"sadness":{"y":0,"i":0},"surprise":{"y":1,"i":1}}

Text: "Hondas are notoriously great cars for long trips for their dependability and great gas mileage."
→ {"anger":{"y":0,"i":0},"fear":{"y":0,"i":0},"joy":{"y":1,"i":2},"sadness":{"y":0,"i":0},"surprise":{"y":0,"i":0}}

Text: "Not only was I not able to move, I smacked my head against the guy sitting in front me and things just got awkward."
→ {"anger":{"y":1,"i":1},"fear":{"y":1,"i":1},"joy":{"y":0,"i":0},"sadness":{"y":0,"i":0},"surprise":{"y":0,"i":0}}
"""

PROMPTS = {
    "BASE_SYSTEM_PROMPT": BASE_SYSTEM_PROMPT,
    "FEW_SHOT_PROMPT": BASE_SYSTEM_PROMPT + FEW_SHOT_PROMPT,
    "COT_PROMPT": BASE_SYSTEM_PROMPT + COT_PROMPT,
    "COMBINED_PROMPT": BASE_SYSTEM_PROMPT + COT_PROMPT + FEW_SHOT_PROMPT,
}