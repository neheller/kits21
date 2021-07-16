# This is how we construct the hec regions from the labels. (1, 5, 6) means that labels 1, 5 and 6 will be merged and
# evaluated jointly in the corresponding hec region
KITS_HEC_LABEL_MAPPING = {
    'kidney_and_mass': (1, 2, 3),
    'mass': (2, 3),
    'tumor': (2, ),
}

HEC_NAME_LIST = list(KITS_HEC_LABEL_MAPPING.keys())

# just for you as a reference. This tells you which metric is at what index. This is not used anywhere
METRIC_NAME_LIST = ["Dice", "SD"]

LABEL_AGGREGATION_ORDER = (1, 3, 2)  # this means that we first place the kidney, then the cyst and finally the tumor.
# The order matters!
# If parts of a later label (example tumor) overlap with a prior label (kidney or cyst) the prior label is overwritten

KITS_LABEL_NAMES = {
    1: "kidney",
    2: "tumor",
    3: "cyst"
}

# values are determined by kits21/evaluation/compute_tolerances.py
HEC_SD_TOLERANCES_MM = {
    'kidney_and_mass': 1.0330772532390826,
    'mass': 1.1328796488598762,
    'tumor': 1.1498198361434828,
}

# this determines which reference file we use for evaluation
GT_SEGM_FNAME = 'aggregated_MAJ_seg.nii.gz'

# how many groups of sampled segmentations?
NUMBER_OF_GROUPS = 5
