from multiprocessing import Pool
from typing import Tuple, Union, Dict
import SimpleITK as sitk
import numpy as np
from medpy.metric import dc, jc
from medpy.metric.binary import __surface_distances

from kits21.configuration.labels import KITS_HEC_LABEL_MAPPING, HEC_NAME_LIST


def compute_metrics(segmentation_predicted: np.ndarray, segmentation_reference: np.ndarray,
                    label: Union[int, Tuple[int, ...]], spacing: Tuple[float, float, float] = (1, 1, 1)) \
        -> Tuple[float, float, float, float, float, float]:
    """

    :param segmentation_predicted: segmentation map (np.ndarray) with int values representing the predicted segmentation
    :param segmentation_reference:  segmentation map (np.ndarray) with int values representing the gt segmentation
    :param label: can be int or tuple of ints. If tuple of ints, a HEC is constructed from the labels in the tuple.
    :param spacing: important to know for volume and surface distance computation
    :return:
    """
    assert all([i == j] for i, j in zip(segmentation_predicted.shape, segmentation_reference.shape)), \
        "predicted and gt segmentation must have the same shape"

    # build a bool mask from the segmentation_predicted, segmentation_reference and provided label(s)
    if not isinstance(label, (tuple, list)):
        mask_pred = segmentation_predicted == label
        mask_gt = segmentation_reference == label
    else:
        mask_pred = np.zeros(segmentation_predicted.shape, dtype=bool)
        mask_gt = np.zeros(segmentation_reference.shape, dtype=bool)
        for l in label:
            mask_pred[segmentation_predicted == l] = True
            mask_gt[segmentation_reference == l] = True

    gt_empty = np.sum(mask_gt) == 0
    pred_empty = np.sum(mask_pred) == 0

    # dice and jaccard are not defined if both are empty ( 0/0 situation)
    if gt_empty and pred_empty:
        dice = np.nan
        jaccard = np.nan
    else:
        dice = dc(mask_pred, mask_gt)
        jaccard = jc(mask_pred, mask_gt)

    volume_per_voxel = np.prod(spacing)
    volume_pred = np.sum(mask_pred) * volume_per_voxel
    volume_gt = np.sum(mask_gt) * volume_per_voxel

    if gt_empty and pred_empty:
        srvd = np.nan
    else:
        srvd = float(2 * np.abs(volume_pred - volume_gt) / (volume_pred + volume_gt))

    avd = float(np.abs(volume_pred - volume_gt))

    if gt_empty and pred_empty:
        # both are empty and we correctly didn't predict anything
        assd = np.nan
        rmsd = np.nan
    elif gt_empty or pred_empty:
        # worst possible values
        assd = 10
        rmsd = 10
    else:
        dist_pred_to_gt = __surface_distances(mask_pred, mask_gt, spacing)
        dist_gt_to_pred = __surface_distances(mask_gt, mask_pred, spacing)

        # for assd and rmsd we need to cap the distances at 10cm spacing is in mm so the threshold is 100
        dist_pred_to_gt_capped = np.clip(dist_pred_to_gt, a_max=100, a_min=None)
        dist_gt_to_pred_capped = np.clip(dist_gt_to_pred, a_max=100, a_min=None)

        del dist_pred_to_gt, dist_gt_to_pred

        assd = (np.sum(dist_pred_to_gt_capped) + np.sum(dist_gt_to_pred_capped)) / \
               (len(dist_pred_to_gt_capped) + len(dist_gt_to_pred_capped))

        rmsd = np.sqrt(
            (np.sum(dist_pred_to_gt_capped ** 2) + np.sum(dist_gt_to_pred_capped ** 2)) /
            (len(dist_pred_to_gt_capped) + len(dist_gt_to_pred_capped))
        )
    return 1 - dice, 1 - jaccard, srvd, avd, assd, rmsd


def evaluate_case(fname_pred: str, fname_ref: str) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    """
    Takes two .nii.gz segmentation maps and computes the KiTS metrics for all HECs. The return value of this function
    is a dictionoary mapping each HEC (by its name as defined in KITS_HEC_LABEL_MAPPING) to a tuple of metrics.
    The order of metrics in the tuple follows the order on the KiTS website (https://kits21.kits-challenge.org/):
    -> 1 - Dice (0 is best)
    -> 1 - Jaccard (0 is best)
    -> Symmetric Relative Volume Difference (0 is best)
    -> Absolute Volume Difference (in mm^3, 0 is best)
    -> Average Symmetric Surface Distance (in mm, 0 is best)
    -> RMS Symmetric Surface Distance (in mm, 0 is best)

    :param fname_pred: filename of the predicted segmentation
    :param fname_ref: filename of the ground truth segmentation
    :return: np.ndarray of shape 6x6 (labels x metrics). Labels are HECs in the order given by
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

    metrics = np.zeros((6, 6), dtype=float)
    for i, hec in enumerate(HEC_NAME_LIST):
        metrics[i] = compute_metrics(img_pred_npy, img_gt_npy, KITS_HEC_LABEL_MAPPING[hec], tuple(spacing_pred))

    return metrics


def compute_gauged_score(computed_metrics: np.ndarray, corresponding_deltas: np.ndarray):
    assert all(i == j for i, j in zip((6, 6), corresponding_deltas.shape[1:])), "corresponding_deltas must have shape " \
                                                                            "(N, 6, 6) (n_references x labels x metrics)"
    assert all(i == j for i, j in zip((6, 6), computed_metrics.shape[1:])), "computed_metrics must have shape " \
                                                                            "(N, 6, 6) (n_references x labels x metrics)"
    scaled_errors = computed_metrics / corresponding_deltas
    gauged_score = 100 - 10 * np.mean(scaled_errors)
    return gauged_score


if __name__ == '__main__':
    from time import time

    img_pred = '/home/fabian/http_git/kits21/data/case_00000/segmentation_samples/sample_0001.nii.gz'
    res = []
    p = Pool(3)
    start = time()
    for ref_id in range(10):
        img_ref = '/home/fabian/http_git/kits21/data/case_00000/segmentation_samples/sample_%04.0d.nii.gz' % ref_id
        res.append(p.starmap_async(evaluate_case, ((
            img_pred, img_ref
                                                   ), )))
    res = np.vstack([np.array(i.get()) for i in res])
    p.close()
    p.join()
    end = time()
    print("This took %s seconds" % np.round((end - start), 4))
