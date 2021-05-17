from typing import Tuple, Union, Dict

import SimpleITK as sitk
import numpy as np
from medpy.metric import dc, jc
from medpy.metric.binary import __surface_distances

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


def compute_metrics(segmentation_predicted: np.ndarray, segmentation_reference: np.ndarray,
                    label: Union[int, Tuple[int, ...]], spacing: Tuple[float, float, float] = (1, 1, 1)) \
        -> Tuple[float, float, float, float, float, float]:
    assert all([i==j] for i, j in zip(segmentation_predicted.shape, segmentation_reference.shape)), \
        "predicted and gt segmentation must have the same shape"

    if not isinstance(label, tuple):
        mask_pred = segmentation_predicted == label
        mask_gt = segmentation_reference == label
    else:
        mask_pred = np.zeros(segmentation_predicted.shape, dtype=bool)
        mask_gt = np.zeros(segmentation_reference.shape, dtype=bool)
        for l in label:
            mask_pred[segmentation_predicted==l] = True
            mask_gt[segmentation_reference==l] = True

    dice = dc(mask_pred, mask_gt)
    jaccard = jc(mask_pred, mask_gt)

    volume_per_voxel = np.prod(spacing)

    volume_pred = np.sum(mask_pred) * volume_per_voxel
    volume_gt = np.sum(mask_gt) * volume_per_voxel

    srvd = float(2 * np.abs(volume_pred - volume_gt) / (volume_pred + volume_gt))
    avd = float(np.abs(volume_pred - volume_gt))

    dist_pred_to_gt = __surface_distances(mask_pred, mask_gt, spacing)
    dist_gt_to_pred = __surface_distances(mask_gt, mask_pred, spacing)

    # for assd the distances seem to be capped at 10cm? spacing is in mm so the threshold is 100
    dist_pred_to_gt_capped = np.clip(dist_pred_to_gt, a_max=100, a_min=None)
    dist_gt_to_pred_capped = np.clip(dist_gt_to_pred, a_max=100, a_min=None)

    assd = (np.sum(dist_pred_to_gt_capped) + np.sum(dist_gt_to_pred_capped)) / \
           (len(dist_pred_to_gt_capped) + len(dist_gt_to_pred_capped))

    # rmsd does not seem to use capped distances?
    rmsd = np.sqrt(np.sum(dist_pred_to_gt ** 2) + np.sum(dist_gt_to_pred ** 2)) / \
           (len(dist_pred_to_gt_capped) + len(dist_gt_to_pred_capped))
    # I believe that this should be the correct formula. Website says the one above is it. Let's discuss
    # rmsd = np.sqrt((np.sum(dist_pred_to_gt ** 2) + np.sum(dist_gt_to_pred ** 2)) / (len(dist_pred_to_gt_capped) + len(dist_gt_to_pred_capped)))
    return dice, jaccard, srvd, avd, assd, rmsd


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
    :return:
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

    metrics = {}
    for hec in KITS_HEC_LABEL_MAPPING.keys():
        metrics[hec] = compute_metrics(img_pred_npy, img_gt_npy, KITS_HEC_LABEL_MAPPING[hec], tuple(spacing_pred))

    return metrics


if __name__ == '__main__':
    from time import time
    img_pred = '/home/fabian/http_git/kits21/data/case_00002/aggregated_AND_seg.nii.gz'
    img_ref = '/home/fabian/http_git/kits21/data/case_00002/aggregated_OR_seg.nii.gz'
    start = time()
    ret = evaluate_case(img_pred, img_ref)
    end = time()
    print("This took %s seconds" % np.round((end - start), 4))
    print("Metrics")
    print(ret)

    # this runs in 76.2658s on a Ryzen 5800X CPU (single threaded)