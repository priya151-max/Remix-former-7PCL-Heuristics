"""
Metadata questions shown to the user.

The trained model was fitted on Derm7pt, so the questions are the Derm7pt
metadata fields, worded for a non-expert. Option values are the exact strings of
the training vocabulary (validated against ``model/metadata_vocab.json``).

- Basic questions (sex, body site, elevation) are asked from everybody.
- Clinician fields (diagnostic difficulty, planned management) are optional.
- ``seven_point_score`` is deliberately NOT asked: it is a derived checklist score
  that users cannot know. It is left blank (all-zero one-hot block).
"""
from typing import Dict, List

from .metadata_encoder import MetadataEncoder

NOT_SURE = ""  # blank -> all-zero one-hot block

FIELDS: List[dict] = [
    {
        "key": "sex", "group": "basic", "required": True,
        "label": "Sex of the patient",
        "help": "Sex has only a small influence on which lesions are more common.",
        "options": [
            {"value": "female", "label": "Female"},
            {"value": "male", "label": "Male"},
        ],
    },
    {
        "key": "location", "group": "basic", "required": True,
        "label": "Where on the body is the lesion?",
        "help": "Body site gives context, for example sun-exposed skin versus the back or legs.",
        "options": [
            {"value": "head neck", "label": "Head or neck"},
            {"value": "chest", "label": "Chest"},
            {"value": "back", "label": "Back"},
            {"value": "abdomen", "label": "Abdomen (belly)"},
            {"value": "upper limbs", "label": "Arm or shoulder"},
            {"value": "lower limbs", "label": "Leg"},
            {"value": "acral", "label": "Palm, sole or nail area"},
            {"value": "buttocks", "label": "Buttocks"},
            {"value": "genital areas", "label": "Genital area"},
        ],
    },
    {
        "key": "elevation", "group": "basic", "required": True,
        "label": "How does the lesion feel?",
        "help": "Run a fingertip gently over the lesion. Is it level with the surrounding skin, or raised?",
        "options": [
            {"value": "flat", "label": "Flat", "hint": "Level with the skin"},
            {"value": "palpable", "label": "Slightly raised", "hint": "Can be felt with a fingertip"},
            {"value": "nodular", "label": "A raised bump", "hint": "Clearly protruding lump"},
        ],
    },
    {
        "key": "level_of_diagnostic_difficulty", "group": "clinician", "required": False,
        "label": "How hard is this lesion to diagnose?",
        "help": "Clinician's judgement, as recorded in the Derm7pt dataset. Leave blank if unknown.",
        "options": [
            {"value": "low", "label": "Low"},
            {"value": "medium", "label": "Medium"},
            {"value": "high", "label": "High"},
        ],
    },
    {
        "key": "management", "group": "clinician", "required": False,
        "label": "Planned management",
        "help": "The clinician's plan, as recorded in Derm7pt. Leave blank if unknown; "
                "it is not needed for a self-check.",
        "options": [
            {"value": "clinical follow up", "label": "Follow-up (monitor over time)"},
            {"value": "excision", "label": "Excision (surgical removal)"},
            {"value": "no further examination", "label": "No further examination"},
        ],
    },
]

FIELD_BY_KEY: Dict[str, dict] = {f["key"]: f for f in FIELDS}


def validate_against_vocab(enc: MetadataEncoder) -> None:
    """Every option value must be a real category of the trained encoder."""
    for f in FIELDS:
        cats = set(enc.categories[f["key"]])
        for o in f["options"]:
            if o["value"] not in cats:
                raise ValueError(
                    f"Form option {o['value']!r} for {f['key']} is not in the trained "
                    f"vocabulary {sorted(cats)}.")


def clean_values(raw: Dict[str, str]) -> Dict[str, str]:
    """Keep only known fields with valid option values; everything else is blank."""
    out: Dict[str, str] = {}
    for f in FIELDS:
        v = (raw.get(f["key"]) or "").strip()
        valid = {o["value"] for o in f["options"]}
        out[f["key"]] = v if v in valid else NOT_SURE
    return out


def missing_required(values: Dict[str, str], allow_not_sure: bool = True) -> List[str]:
    return [f["key"] for f in FIELDS
            if f["required"] and f["key"] not in values]


def readable(values: Dict[str, str]) -> List[dict]:
    """[{field, label, answer}] for display; blank -> 'Not provided'."""
    rows = []
    for f in FIELDS:
        v = values.get(f["key"], NOT_SURE)
        lab = next((o["label"] for o in f["options"] if o["value"] == v), "Not provided")
        rows.append({"key": f["key"], "question": f["label"], "answer": lab,
                     "provided": bool(v), "group": f["group"]})
    return rows
