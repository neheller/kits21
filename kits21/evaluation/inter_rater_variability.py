from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfolders, subfiles, save_json, isfile, isdir
import SimpleITK as sitk

from kits21.configuration.labels import HEC_NAME_LIST, KITS_HEC_LABEL_MAPPING
from kits21.configuration.paths import TRAINING_DIR
from kits21.evaluation.metrics import compute_metrics_for_case, compute_metrics_for_label


def compute_inter_rater_variability_for_case(case_folder):
    """
    We are running this with many tolerance thresholds so that we can determine a good tolerance for evaluating the
    test set
    :param casename:
    :return:
    """
    segmentation_samples_folder = join(case_folder, 'segmentation_samples')
    if not isdir(segmentation_samples_folder):
        return
    thresholds = np.linspace(0.1, 5, 50)
    dice_scores = {i: [] for i in HEC_NAME_LIST}
    nsds = {i: [] for i in HEC_NAME_LIST}
    groups = subfolders(segmentation_samples_folder, join=False, prefix='group')
    for g in groups:
        nii_files = subfiles(join(segmentation_samples_folder, g), suffix='.nii.gz')
        for ref_idx in range(len(nii_files)):
            for pred_idx in range(ref_idx + 1, len(nii_files)):
                img_pred = sitk.ReadImage(nii_files[pred_idx])
                img_ref = sitk.ReadImage(nii_files[ref_idx])

                img_pred_npy = sitk.GetArrayFromImage(img_pred)
                img_gt_npy = sitk.GetArrayFromImage(img_ref)
                spacing_pred = list(img_pred.GetSpacing())[::-1]

                for i, hec in enumerate(HEC_NAME_LIST):
                    d, n = compute_metrics_for_label(img_pred_npy, img_gt_npy, KITS_HEC_LABEL_MAPPING[hec],
                                                     tuple(spacing_pred), nsd_tolerance_mm=tuple(thresholds))
                    dice_scores[hec].append(d)
                    nsds[hec].append(n)
    dice_averages = {i: float(np.mean(j)) for i, j in dice_scores.items()}
    nsd_averages = {i: list(np.mean(j, 0).astype(float)) for i, j in nsds.items()}
    save_json({"dice": dice_averages, "nsd": nsd_averages, "nsd_thresholds": list(thresholds)}, join(case_folder, 'inter_rater_variability.json'))


def compute_all_inter_rater_variabilities(num_proceses: int = 10, overwrite_existing=False):
    p = Pool(num_proceses)
    case_folders = subfolders(TRAINING_DIR, prefix='case_')
    if not overwrite_existing:
        c = []
        for cs in case_folders:
            if not isfile(join(cs, 'inter_rater_variability.json')):
                c.append(cs)
        print(len(c), 'out of', len(case_folders), 'to go...')
        case_folders = c
    r = p.starmap_async(compute_inter_rater_variability_for_case, ([i] for i in case_folders))
    _ = r.get()
    p.close()
    p.join()