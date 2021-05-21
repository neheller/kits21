KITS_CLASS_LABEL_MAP = {
    1: 'kidney',
    2: 'ureter',
    3: 'artery',
    4: 'vein',
    5: 'cyst',
    6: 'tumor'
}

KITS_LABEL_CLASS_MAP = {j: i for i, j in KITS_CLASS_LABEL_MAP.items()}

KITS_HEC_LABEL_MAPPING = {
    'kidney_and_mass': (1, 5, 6),
    'mass': (5, 6),
    'tumor': (6, ),
    'ureter_and_vessels': (2, 3, 4),
    'vessels': (3, 4),
    'arteries': (3, )
}

HEC_NAME_LIST = ['kidney_and_mass', 'mass', 'tumor', 'ureter_and_vessels', 'vessels', 'arteries']
METRIC_NAME_LIST = ["1-Dice", "1-Jaccard", "SRVD", "AVD", "ASSD", "RMSD"]

HEC_CONSTRUCTION_ORDER = (2, 4, 3, 1, 5, 6)  # from https://github.com/neheller/kits21/blob/master/annotation/import.py
