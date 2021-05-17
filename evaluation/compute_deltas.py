import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool

from evaluation.metrics import KITS_HEC_LABEL_MAPPING, evaluate_case


def _convert_idx_to_xy(idx, shape_x):
    x = idx % shape_x
    y = idx // shape_x
    return x, y


def _convert_xy_to_idx(x, y, shape_x):
    assert x < shape_x
    return y * shape_x + x


def compute_deltas(folder_with_segmentations: str, num_processes: int = 8):
    num_metrics = 6
    n_labels = len(KITS_HEC_LABEL_MAPPING)
    segmentation_files = subfiles(folder_with_segmentations, suffix='.nii.gz', join=True)[:4]
    num_segs = len(segmentation_files)

    p = Pool(num_processes)

    indexes = []
    results = []
    for seg_source in range(num_segs):
        for seg_target in range(seg_source + 1, num_segs):
            indexes.append(_convert_xy_to_idx(seg_source, seg_target, num_segs))
            results.append(p.starmap_async(evaluate_case,
                                           ((
                                               segmentation_files[seg_source], segmentation_files[seg_target]
                                            ), )))
    results = [i.get() for i in results]
    p.close()
    p.join()

    # now assign results to correct index
    metrics = np.zeros((num_segs, num_segs, n_labels, num_metrics))
    for i, r in zip(indexes, results):
        x, y = _convert_idx_to_xy(i, num_segs)
        for j, k in enumerate(KITS_HEC_LABEL_MAPPING.keys()):
            metrics[x, y, j] = r[0][k]
            metrics[y, x, j] = r[0][k]

    # delta has shape (n_segmentations x n_labels x n_metrics)
    deltas = np.zeros((num_segs, n_labels, num_metrics))
    for n in range(num_segs):
        deltas[n] = np.sum(metrics[n], axis=0) / (num_segs - 1)
    return deltas