import os.path
from multiprocessing import Pool
from typing import Tuple, Union, List

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, isdir, subdirs
from medpy.metric import dc
from surface_distance import compute_surface_distances

from kits21.configuration.labels import KITS_HEC_LABEL_MAPPING, HEC_NAME_LIST, HEC_SD_TOLERANCES_MM, GT_SEGM_FNAME
from kits21.configuration.paths import TRAINING_DIR
from time import time


def construct_HEC_from_segmentation(segmentation: np.ndarray, label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if not isinstance(label, (tuple, list)):
        return segmentation == label
    else:
        mask = np.zeros(segmentation.shape, dtype=bool)
        for l in label:
            mask[segmentation == l] = True
        return mask


def compute_metrics_for_label(segmentation_predicted: np.ndarray, segmentation_reference: np.ndarray,
                              label: Union[int, Tuple[int, ...]], spacing: Tuple[float, float, float],
                              sd_tolerance_mm: Union[float, Tuple[float, ...]]) \
        -> Tuple[float, float]:
    """
    :param segmentation_predicted: segmentation map (np.ndarray) with int values representing the predicted segmentation
    :param segmentation_reference:  segmentation map (np.ndarray) with int values representing the gt segmentation
    :param label: can be int or tuple of ints. If tuple of ints, a HEC is constructed from the labels in the tuple.
    :param spacing: important to know for volume and surface distance computation
    :param sd_tolerance_mm
    :return:
    """
    assert all([i == j] for i, j in zip(segmentation_predicted.shape, segmentation_reference.shape)), \
        "predicted and gt segmentation must have the same shape"

    # build a bool mask from the segmentation_predicted, segmentation_reference and provided label(s)
    mask_pred = construct_HEC_from_segmentation(segmentation_predicted, label)
    mask_gt = construct_HEC_from_segmentation(segmentation_reference, label)

    gt_empty = np.sum(mask_gt) == 0
    pred_empty = np.sum(mask_pred) == 0

    # dice and jaccard are not defined if both are empty ( 0/0 situation)
    if gt_empty and pred_empty:
        dice = 1
    else:
        dice = dc(mask_pred, mask_gt)

    if gt_empty and pred_empty:
        sds = [1] * len(sd_tolerance_mm) if isinstance(sd_tolerance_mm, (tuple, list, np.ndarray)) else 1
    elif gt_empty or pred_empty:
        sds = [0] * len(sd_tolerance_mm) if isinstance(sd_tolerance_mm, (tuple, list, np.ndarray)) else 0
    else:
        dist = compute_surface_distances(mask_gt, mask_pred, spacing)
        distances_gt_to_pred = dist["distances_gt_to_pred"]
        distances_pred_to_gt = dist["distances_pred_to_gt"]
        surfel_areas_gt = dist["surfel_areas_gt"]
        surfel_areas_pred = dist["surfel_areas_pred"]
        if not isinstance(sd_tolerance_mm, (tuple, list, np.ndarray)):
            sd_tolerance_mm = (sd_tolerance_mm, )
        sds = []
        for th in sd_tolerance_mm:
            overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= th])
            overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= th])
            sds.append((overlap_gt + overlap_pred) / (np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred)))

    if isinstance(sds, (tuple, list, np.ndarray)) and len(sds) == 1:
        sds = sds[0]

    return dice, sds


def compute_metrics_for_case(fname_pred: str, fname_ref: str) -> np.ndarray:
    """
    Takes two .nii.gz segmentation maps and computes the KiTS metrics for all HECs. The return value of this function
    is an array of size num_HECs x num_metrics (currently 3x2).

    The order of metrics in the tuple follows the order on the KiTS website (https://kits21.kits-challenge.org/):
    -> Dice (1 is best)
    -> Surface Dice (1 is best)

    :param fname_pred: filename of the predicted segmentation
    :param fname_ref: filename of the ground truth segmentation
    :return: np.ndarray of shape 3x2 (labels x metrics). Labels are HECs in the order given by HEC_NAME_LIST
    """
    img_pred = sitk.ReadImage(fname_pred)
    img_ref = sitk.ReadImage(fname_ref)

    # we need to invert the spacing because SimpleITK is weird
    spacing_pred = list(img_pred.GetSpacing())[::-1]
    spacing_ref = list(img_ref.GetSpacing())[::-1]

    if not all([i == j] for i, j in zip(spacing_pred, spacing_ref)):
        # no need to make this an error. We can evaluate successfullt as long as the shapes match.
        print("WARNING: predited and reference segmentation do not have the same spacing!")

    img_pred_npy = sitk.GetArrayFromImage(img_pred)
    img_gt_npy = sitk.GetArrayFromImage(img_ref)

    metrics = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
    for i, hec in enumerate(HEC_NAME_LIST):
        metrics[i] = compute_metrics_for_label(img_pred_npy, img_gt_npy, KITS_HEC_LABEL_MAPPING[hec],
                                               tuple(spacing_pred), sd_tolerance_mm=HEC_SD_TOLERANCES_MM[hec])
    return metrics


def evaluate_predictions(folder_with_predictions: str, num_processes: int = 8, write_csv_file: bool = True,) \
        -> Tuple[np.ndarray, List[str]]:
    """

    :param folder_with_predictions: your predictions must be located in this folder. Predictions must be named
    case_XXXXX.nii.gz
    :param num_processes: number of CPU processes to use for metric computation. Watch out for RAM usage!
    :param write_csv_file: if True, writes metrics to folder_with_predictions/evaluation.csv
    :return: metrics (num_predictions x num_HECs x num_metrics)
    """
    start = time()
    p = Pool(num_processes)

    predicted_segmentation_files = subfiles(folder_with_predictions, suffix='.nii.gz', join=True)
    caseids = [os.path.basename(i)[:-7] for i in predicted_segmentation_files]

    params = []
    for c in caseids:
        params.append(
            [join(folder_with_predictions, c + '.nii.gz'),
             join(TRAINING_DIR, c, GT_SEGM_FNAME)]
        )
    metrics = p.starmap(compute_metrics_for_case, params)
    metrics = np.vstack([i[None] for i in metrics])
    p.close()
    p.join()
    end = time()
    print('Evaluation took %f s. Num_processes: %d' % (np.round(end - start, 2), num_processes))

    if write_csv_file:
        # let's write a csv file
        # for each case, metrics are a 3x2 array (num_HECs x num_metrics).
        with open(join(folder_with_predictions, 'evaluation.csv'), "w") as f:
            f.write("caseID,Dice_kidney,Dice_masses,Dice_tumor,SD_kidney,SD_masses,SD_tumor\n")
            for i, c in enumerate(caseids):
                f.write("%s,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f\n" % (
                    c,
                    metrics[i, 0, 0], metrics[i, 1, 0], metrics[i, 2, 0],
                    metrics[i, 0, 1], metrics[i, 1, 1], metrics[i, 2, 1],
                ))
            mean_metrics = metrics.mean(0)
            f.write("average,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f" % (
                mean_metrics[0, 0], mean_metrics[1, 0], mean_metrics[2, 0],
                mean_metrics[0, 1], mean_metrics[1, 1], mean_metrics[2, 1],
            ))
    return metrics, predicted_segmentation_files


def evaluate_predictions_on_samples(folder_with_predictions: str, num_processes: int = 8, write_csv_file: bool = True,) \
        -> Tuple[np.ndarray, List[str]]:
    start = time()
    p = Pool(num_processes)

    predicted_segmentation_files = subfiles(folder_with_predictions, suffix='.nii.gz', join=True)
    caseids = [os.path.basename(i)[:-7] for i in predicted_segmentation_files]

    metrics = p.map(evaluate_predicted_file_on_samples, predicted_segmentation_files)
    metrics = np.vstack([i[None] for i in metrics])
    p.close()
    p.join()
    end = time()
    print('Evaluation on samples took %f s. Num_processes: %d' % (np.round(end - start, 2), num_processes))

    if write_csv_file:
        # let's write a csv file
        # for each case, metrics are a 3x2 array (num_HECs x num_metrics).
        with open(join(folder_with_predictions, 'evaluation_samples.csv'), "w") as f:
            f.write("caseID,Dice_kidney,Dice_masses,Dice_tumor,SD_kidney,SD_masses,SD_tumor\n")
            for i, c in enumerate(caseids):
                f.write("%s,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f\n" % (
                    c,
                    metrics[i, 0, 0], metrics[i, 1, 0], metrics[i, 2, 0],
                    metrics[i, 0, 1], metrics[i, 1, 1], metrics[i, 2, 1],
                ))
            mean_metrics = metrics.mean(0)
            f.write("average,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f" % (
                mean_metrics[0, 0], mean_metrics[1, 0], mean_metrics[2, 0],
                mean_metrics[0, 1], mean_metrics[1, 1], mean_metrics[2, 1],
            ))
    return metrics, predicted_segmentation_files


def evaluate_predicted_file_on_samples(filename_predicted: str):
    assert os.path.basename(filename_predicted).startswith("case_") and filename_predicted.endswith('.nii.gz'), \
        "filename_predicted must benamed case_xxxxx.nii.gz where xxxxx is the case id"
    caseid = os.path.basename(filename_predicted)[:-7]
    samples_folder = join(TRAINING_DIR, caseid, 'segmentation_samples')
    if not isdir(samples_folder):
        raise RuntimeError('segmentation_samples folder missing. Please run kits21/annotation/sample_segmentations.py')
    groups = subdirs(samples_folder, prefix='group')
    metrics = []
    for g in groups:
        nii_files = subfiles(g, suffix='.nii.gz')
        for n in nii_files:
            metrics.append(compute_metrics_for_case(filename_predicted, n)[None])
    return np.mean(np.vstack(metrics), 0)


def sort_by_worst_Dice(evaluation_csv_file: str, n_worst: int = 20):
    loaded = np.loadtxt(evaluation_csv_file, dtype=str, skiprows=1, delimiter=',')
    casenames = loaded[:, 0]
    metrics = loaded[:, 1:].astype(float)
    dice_scores = metrics[:, :3]
    for i, hec in enumerate(HEC_NAME_LIST):
        print(hec)
        argsorted = np.argsort(dice_scores[:, i])
        for a in argsorted[:n_worst]:
            print(casenames[a], dice_scores[a, i])
        print()
