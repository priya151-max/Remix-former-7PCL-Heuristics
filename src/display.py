"""
Human-readable names and plain-language descriptions used by the web UI,
the explanation engine and the PDF report.

These are presentation strings only. All numbers come from the model.
"""

CLASS_INFO = {
    "BCC": {
        "name": "Basal cell carcinoma",
        "short": "A common, usually slow-growing type of skin cancer.",
        "group": "Malignant (cancer)",
        "tone": "warn",
        "color": "#c2410c",
    },
    "Nev": {
        "name": "Melanocytic nevus (mole)",
        "short": "A common mole: a harmless cluster of pigment cells.",
        "group": "Usually benign",
        "tone": "ok",
        "color": "#0f766e",
    },
    "Mel": {
        "name": "Melanoma",
        "short": "A serious skin cancer of the pigment cells; early detection matters.",
        "group": "Malignant (most serious)",
        "tone": "danger",
        "color": "#b91c1c",
    },
    "SK": {
        "name": "Seborrheic keratosis",
        "short": "A very common, harmless, wart-like or 'stuck-on' growth.",
        "group": "Benign",
        "tone": "ok",
        "color": "#4f46e5",
    },
    "Misc": {
        "name": "Other lesion types",
        "short": "Less common lesions such as lentigo, vascular lesions or dermatofibroma.",
        "group": "Mixed group",
        "tone": "neutral",
        "color": "#64748b",
    },
}

ATTRIBUTE_INFO = {
    "PN": {
        "name": "Pigment network",
        "positive": "Irregular (atypical) pigment network",
        "plain": "A honeycomb-like grid of brown lines seen with the dermatoscope. "
                 "Regular and even is typical of harmless moles; thick, dark or "
                 "uneven lines are a warning sign.",
        "points": 2, "kind": "major",
    },
    "BWV": {
        "name": "Blue-whitish veil",
        "positive": "Blue-whitish veil present",
        "plain": "An irregular area of blue colour with a milky-white haze on top, "
                 "most often linked to melanoma.",
        "points": 2, "kind": "major",
    },
    "VS": {
        "name": "Vascular structures",
        "positive": "Blood-vessel pattern present",
        "plain": "Small blood vessels visible in the lesion. The shape matters: "
                 "tree-like branching vessels point towards basal cell carcinoma, "
                 "irregular linear or dotted ones can be a melanoma sign.",
        "points": 2, "kind": "major",
    },
    "PIG": {
        "name": "Irregular pigmentation",
        "positive": "Irregular blotches of pigment",
        "plain": "Uneven patches of black, brown or grey colour spread unevenly "
                 "through the lesion.",
        "points": 1, "kind": "minor",
    },
    "STR": {
        "name": "Streaks",
        "positive": "Irregular streaks",
        "plain": "Finger-like or radial lines at the edge of the lesion. Irregular "
                 "streaks are a warning sign.",
        "points": 1, "kind": "minor",
    },
    "DaG": {
        "name": "Dots and globules",
        "positive": "Irregular dots and globules",
        "plain": "Small round dots and larger globules of dark colour. Even and "
                 "regular is common in moles; varying in size and spread is a warning sign.",
        "points": 1, "kind": "minor",
    },
    "RS": {
        "name": "Regression structures",
        "positive": "Regression structures present",
        "plain": "White scar-like areas or blue-grey 'peppered' spots where part of "
                 "the lesion may have faded.",
        "points": 1, "kind": "minor",
    },
}

REMIX_TOKEN_INFO = {
    "global_clinical": "Whole clinical photo",
    "local_clinical": "Close-up crops of the clinical photo",
    "global_dermoscopy": "Whole dermoscopy image",
    "local_dermoscopy": "Dermoscopy image regions",
    "metadata": "Patient information",
}

# Plain-language wording for the seven checklist states (per-attribute vocab
# is read from the checkpoint-verified config; this only supplies friendlier text).
STATE_LABELS = {
    "absent": "absent", "present": "present", "typical": "typical (regular)",
    "atypical": "atypical (irregular)", "regular": "regular", "irregular": "irregular",
    "arborizing": "tree-like branching", "comma": "comma-shaped", "dotted": "dotted",
    "hairpin": "hairpin-shaped", "linear irregular": "irregular lines",
    "within regression": "within regression areas", "wreath": "wreath-like",
    "diffuse irregular": "diffuse, irregular", "diffuse regular": "diffuse, regular",
    "localized irregular": "localised, irregular", "blue areas": "blue areas",
    "combinations": "mixed types", "white areas": "white areas",
}
