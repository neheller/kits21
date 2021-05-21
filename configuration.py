from pathlib import Path
import os

TRAINING_DIR = Path(__file__).parent / "data"
TESTING_DIR = Path(os.environ["KITS21_TEST_DIR"]).resolve(strict=True) if "KITS21_TEST_DIR" in os.environ.keys() else \
    None
SRC_DIR = Path(os.environ["KITS21_SERVER_DATA"]).resolve(strict=True) if "KITS21_SERVER_DATA" in os.environ.keys() \
    else None
CACHE_FILE = Path(__file__).parent / "annotation" / "cache.json"

# This is how we construct the hec regions from the labels. (1, 5, 6) means that labels 1, 5 and 6 will be merged and
# evaluated jointly in the corresponding hec region
KITS_HEC_LABEL_MAPPING = {
    'kidney_and_mass': (1, 5, 6),
    'mass': (5, 6),
    'tumor': (6, ),
    'ureter_and_vessels': (2, 3, 4),
    'vessels': (3, 4),
    'arteries': (3, )
}

# I know dict keys are always in the same order now and we could just use KITS_HEC_LABEL_MAPPING.keys() but old habits
# die hard and we want to be sure metrics are computed in the right order because we rely on indexing for gauged score
# computation
HEC_NAME_LIST = ['kidney_and_mass', 'mass', 'tumor', 'ureter_and_vessels', 'vessels', 'arteries']

# just for you as a reference. This tells you which metric is at what index. This is not used anywhere
METRIC_NAME_LIST = ["1-Dice", "1-Jaccard", "SRVD", "AVD", "ASSD", "RMSD"]

HEC_CONSTRUCTION_ORDER = (2, 4, 3, 1, 5, 6)  # from https://github.com/neheller/kits21/blob/master/annotation/import.py


KITS_LABEL_NAMES = {
    1: "kidney",
    2: "ureter",
    3: "artery",
    4: "vein",
    5: "cyst",
    6: "tumor"
}
