# label_config.py
LABEL_MAP = {
    "X": 0,          # Special token for padding/unknown
    "I-Claim": 1,
    "I-Premise": 2,
    "O": 3
}

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}