from multiprocessing import Pool
from typing import Union, Tuple, List

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfolders, subfiles, save_json, isfile, \
    load_json, isdir
from numpy.lib.utils import source
from surface_distance import compute_surface_distances

from kits21.configuration.labels import HEC_NAME_LIST, KITS_HEC_LABEL_MAPPING
from kits21.configuration.paths import TRAINING_DIR, TESTING_DIR
from kits21.evaluation.metrics import construct_HEC_from_segmentation


def determine_tolerance_label(segmentation_1: np.ndarray, segmentation_2: np.ndarray,
                              label: Union[int, Tuple[int, ...]], spacing: Tuple[float, float, float]) \
        -> float:
    mask_1 = construct_HEC_from_segmentation(segmentation_1, label)
    mask_2 = construct_HEC_from_segmentation(segmentation_2, label)

    mask1_empty = np.sum(mask_2) == 0
    mask2_empty = np.sum(mask_1) == 0

    if mask1_empty or mask2_empty:
        return np.nan
    else:
        dist = compute_surface_distances(mask_2, mask_1, spacing)
        distances_gt_to_pred = dist["distances_gt_to_pred"]
        distances_pred_to_gt = dist["distances_pred_to_gt"]
        return (np.percentile(distances_gt_to_pred, 95) + np.percentile(distances_pred_to_gt, 95)) / 2


def determine_tolerances_one_sample(fname_1: str, fname_2: str) -> List[float]:
    img_1 = sitk.ReadImage(fname_1)
    img_2 = sitk.ReadImage(fname_2)

    # we need to invert the spacing because SimpleITK is weird
    spacing_1 = list(img_1.GetSpacing())[::-1]

    img_1_npy = sitk.GetArrayFromImage(img_1)
    img_2_npy = sitk.GetArrayFromImage(img_2)

    tolerances = []
    for hec in HEC_NAME_LIST:
        tolerances.append(determine_tolerance_label(
            img_1_npy, img_2_npy,
            KITS_HEC_LABEL_MAPPING[hec],
            tuple(spacing_1)
        ))
    return tolerances


def determine_tolerances_case(case_folder):
    segmentation_samples_folder = join(case_folder, 'segmentation_samples')
    if not isdir(segmentation_samples_folder):
        return
    groups = subfolders(segmentation_samples_folder, join=False, prefix='group')
    tolerances = []
    for g in groups:
        nii_files = subfiles(join(segmentation_samples_folder, g), suffix='.nii.gz')
        for ref_idx in range(len(nii_files)):
            for pred_idx in range(ref_idx + 1, len(nii_files)):
                tolerances.append(determine_tolerances_one_sample(nii_files[pred_idx], nii_files[ref_idx]))

    save_json({"tolerances": {HEC_NAME_LIST[i]: j for i, j in enumerate(np.mean(tolerances, 0))}}, join(case_folder, 'tolerances.json'))


def compute_tolerances_for_SD(num_proceses: int = 12, overwrite_existing=False, source_dir=TRAINING_DIR):
    p = Pool(num_proceses)
    case_folders = subfolders(source_dir, prefix='case_')
    if not overwrite_existing:
        c = []
        for cs in case_folders:
            if not isfile(join(cs, 'tolerances.json')):
                c.append(cs)
        print(len(c), 'out of', len(case_folders), 'to go...')
        case_folders = c
    for c in case_folders:
        assert isdir(join(c, 'segmentation_samples')), "please generate the segmentation samples first (kits21/annotation/sample_segmentations.py)"
    r = p.starmap_async(determine_tolerances_case, ([i] for i in case_folders))
    _ = r.get()
    p.close()
    p.join()

    # load and aggregate
    case_folders = subfolders(source_dir, prefix='case_')
    tolerances = {i: [] for i in HEC_NAME_LIST}
    for c in case_folders:
        if isfile(join(source_dir, c, 'tolerances.json')):
            tolerances_here = load_json(join(source_dir, c, 'tolerances.json'))
            for i, hec in enumerate(HEC_NAME_LIST):
                tolerances[hec].append(tolerances_here['tolerances'][hec])
    tolerances = {i: np.nanmean(j) for i, j in tolerances.items()}
    print(tolerances)
    return tolerances


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_processes', required=False, default=12, type=int)
    parser.add_argument('-testing', required=False, default=False, type=bool)
    args = parser.parse_args()
    source_dir = TRAINING_DIR
    if args.testing:
        source_dir = TESTING_DIR
    compute_tolerances_for_SD(args.num_processes, False, source_dir)
