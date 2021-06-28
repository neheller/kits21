from multiprocessing import Pool
from typing import Tuple, Union, Dict, List
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isdir, subfiles, join
from kits21.configuration.paths import TRAINING_DIR
from medpy.metric import dc, jc
from medpy.metric.binary import __surface_distances

from kits21.configuration.labels import KITS_HEC_LABEL_MAPPING, HEC_NAME_LIST, HEC_NSD_TOLERANCES_MM, GT_SEGM_FNAME
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances


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
                              nsd_tolerance_mm: float) \
        -> Tuple[float, float]:
    """
    :param segmentation_predicted: segmentation map (np.ndarray) with int values representing the predicted segmentation
    :param segmentation_reference:  segmentation map (np.ndarray) with int values representing the gt segmentation
    :param label: can be int or tuple of ints. If tuple of ints, a HEC is constructed from the labels in the tuple.
    :param spacing: important to know for volume and surface distance computation
    :param nsd_tolerance_mm
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
        dice = np.nan
    else:
        dice = dc(mask_pred, mask_gt)

    if gt_empty and pred_empty:
        nsd = np.nan
    else:
        dist = compute_surface_distances(mask_gt, mask_pred, spacing)
        nsd = compute_surface_dice_at_tolerance(dist, nsd_tolerance_mm)

    return dice, nsd


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
                                               tuple(spacing_pred), nsd_tolerance_mm=HEC_NSD_TOLERANCES_MM[hec])
    return metrics


def evaluate_predictions(folder_with_predictions: str, num_processes: int = 3) \
        -> Tuple[np.ndarray, List[str]]:
    """

    :param folder_with_predictions: your predictions must be located in this folder. Predictions must be named
    case_XXXXX.nii.gz
    :param num_processes: number of CPU processes to use for metric computation. Watch out for RAM usage!
    :param strict: if True, will throw an error if not all 210 training cases have been predicted. If False, it will
    evaluate only the available predictions and ignore the missing ones
    :return: metrics (num_predictions x num_HECs x num_metrics)
    """
    p = Pool(num_processes)

    predicted_segmentation_files = subfiles(folder_with_predictions, suffix='.nii.gz', join=False)
    caseids = [i[:-7] for i in predicted_segmentation_files]

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
    return metrics, predicted_segmentation_files


if __name__ == '__main__':
    from time import time

    img_pred = '/home/fabian/http_git/kits21/data/case_00000/segmentation_samples/sample_0001.nii.gz'
    res = []
    p = Pool(3)
    start = time()
    for ref_id in range(10):
        img_ref = '/home/fabian/http_git/kits21/data/case_00000/segmentation_samples/sample_%04.0d.nii.gz' % ref_id
        res.append(p.starmap_async(compute_metrics_for_case, ((
                                                                  img_pred, img_ref
                                                              ), )))
    res = np.vstack([np.array(i.get()) for i in res])
    p.close()
    p.join()
    end = time()
    print("This took %s seconds" % np.round((end - start), 4))
