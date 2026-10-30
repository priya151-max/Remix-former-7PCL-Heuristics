"""
Configuration constants.

All image/architecture constants are copied VERBATIM from notebook
`7pcl-heuristic3 (1).ipynb` (cell 2, lines 82-161), which produced the deployed
checkpoint. Nothing here is guessed.

Runtime model configuration (class names, attribute names, graph nodes,
dimensions, graph matrices, priors, heuristic lambda) is NOT stored here --
it is read from the checkpoint at load time by src/checkpoint_loader.py.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "RemixFormerPP_FULL_best_model.pth"
METADATA_VOCAB_PATH = PROJECT_ROOT / "model" / "metadata_vocab.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------
# Notebook constants (7pcl-heuristic3 (1).ipynb, cell 2)
# --------------------------------------------------------------------------
USE_GRAD_CHECKPOINTING = False   # inference only; notebook value was True (training)
DERM_WINDOW_CHUNK = 4
LOCAL_CROP_CHUNK = 2

CLINICAL_SIZE = 896
GLOBAL_CLINICAL_SIZE = 384
LOCAL_PATCH_SIZE = 224
DERMOSCOPY_SIZE = 1024
DERM_WINDOW_SIZE = 256
NUM_DERM_WINDOWS = 16
LSM_K = 4

SWIN_EMBED_DIM = 96
SWIN_DEPTHS = (2, 2, 6, 2)
SWIN_HEADS = (3, 6, 12, 24)
SWIN_WINDOW = 7
SWIN_MLP_RATIO = 4.0

VIT_DIM = 384
VIT_DEPTH = 12
VIT_HEADS = 6
MSTA_TEMPLATES = 32

NUM_DIAG_CLASSES = 5
LSM_TRAIN_TEMPERATURE = 0.7

# --------------------------------------------------------------------------
# Upstream RemixFormer++ configuration
#
# meta_dim and checklist_specs were stored in the ORIGINAL upstream checkpoint
# (RemixFormerPP_FULL_best_model.pt), not in the end-to-end package we deploy.
# The values below were recovered from the executed notebook outputs
# (7pcl-heuristic3 (1).ipynb cells 5, 6 and 28) and are independently verified
# at load time against the tensor shapes in upstream_remix_state_dict:
#   metadata.net.0.weight  -> [768, 30]      => META_DIM == 30
#   checklist_heads.<k>.weight -> [n_k, 3840] => num_classes per attribute
# If verification fails, the loader raises instead of proceeding.
# --------------------------------------------------------------------------
META_DIM = 30

METADATA_COLUMNS = [
    "seven_point_score",
    "level_of_diagnostic_difficulty",
    "elevation",
    "location",
    "sex",
    "management",
]

CHECKLIST_COLUMNS = {
    "PN": "pigment_network",
    "BWV": "blue_whitish_veil",
    "VS": "vascular_structures",
    "PIG": "pigmentation",
    "STR": "streaks",
    "DaG": "dots_and_globules",
    "RS": "regression_structures",
}

# Recovered from notebook output. `classes` are the sorted train-split label
# strings; their ORDER defines the softmax index order of each checklist head
# and is also used to build the graph node features, so it must not change.
CHECKLIST_SPECS = {
    "PN":  {"column": "pigment_network",        "classes": ["absent", "atypical", "typical"]},
    "BWV": {"column": "blue_whitish_veil",      "classes": ["absent", "present"]},
    "VS":  {"column": "vascular_structures",    "classes": ["absent", "arborizing", "comma", "dotted",
                                                            "hairpin", "linear irregular",
                                                            "within regression", "wreath"]},
    "PIG": {"column": "pigmentation",           "classes": ["absent", "diffuse irregular",
                                                            "diffuse regular", "localized irregular"]},
    "STR": {"column": "streaks",                "classes": ["absent", "irregular", "regular"]},
    "DaG": {"column": "dots_and_globules",      "classes": ["absent", "irregular", "regular"]},
    "RS":  {"column": "regression_structures",  "classes": ["absent", "blue areas",
                                                            "combinations", "white areas"]},
}
for _name, _spec in CHECKLIST_SPECS.items():
    _spec["num_classes"] = len(_spec["classes"])

# --------------------------------------------------------------------------
# Deployment deviation flag
#
# The notebook builds ClinicalBranch with CLSSwinBackbone(pretrained=True),
# which downloads ImageNet Swin-T weights via timm. Those weights are fully
# overwritten by upstream_remix_state_dict during checkpoint loading, so the
# download is skipped by default at inference. Set to True to reproduce the
# notebook construction path exactly.
# --------------------------------------------------------------------------
LOAD_IMAGENET_PRETRAINED = False

# --------------------------------------------------------------------------
# Heuristic constants (7pcl-heuristic3 (1).ipynb cell 54)
# lam itself is read from the checkpoint, NOT hardcoded.
# --------------------------------------------------------------------------
EPS = 1e-4
HEURISTIC_GRAPH_WEIGHT = 0.35
HEURISTIC_CONTRADICTION_WEIGHT = 0.20
HEURISTIC_INTERACTION_WEIGHT = 0.10

# --------------------------------------------------------------------------
# Attribute abbreviation expansions.
# Source: the `semantic_text` dictionary in notebook cell 43 -- these are the
# project's own descriptions, not externally invented clinical definitions.
# --------------------------------------------------------------------------
ATTRIBUTE_DESCRIPTIONS = {
    "PN": "pigment network in a skin lesion",
    "BWV": "blue whitish veil in a skin lesion",
    "VS": "vascular structures in a skin lesion",
    "PIG": "irregular pigmentation in a skin lesion",
    "STR": "streaks in a skin lesion",
    "DaG": "dots and globules in a skin lesion",
    "RS": "regression structures in a skin lesion",
}
CLASS_DESCRIPTIONS = {
    "BCC": "basal cell carcinoma skin lesion diagnosis",
    "Nev": "benign melanocytic nevus skin lesion diagnosis",
    "Mel": "melanoma skin lesion diagnosis",
    "SK": "seborrheic keratosis skin lesion diagnosis",
    "Misc": "other miscellaneous skin lesion diagnosis",
}

REMIX_TOKEN_NAMES = [
    "global_clinical",
    "local_clinical",
    "global_dermoscopy",
    "local_dermoscopy",
    "metadata",
]

DISCLAIMER = (
    "Research prototype only. This application provides experimental machine learning "
    "predictions and is not a medical diagnostic tool. Results must not be used as a "
    "substitute for professional clinical examination, dermoscopic assessment, "
    "biopsy, or medical advice."
)
