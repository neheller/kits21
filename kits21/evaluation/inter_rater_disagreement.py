from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfolders, subfiles, save_json, isfile, \
    load_json, isdir

from kits21.configuration.labels import HEC_NAME_LIST
from kits21.configuration.paths import TRAINING_DIR
from kits21.evaluation.metrics import compute_metrics_for_case


def compute_inter_rater_disagreement_for_case(case_folder):
    """
    We are running this with many tolerance thresholds so that we can determine a good tolerance for evaluating the
    test set
    :param casename:
    :return:
    """
    segmentation_samples_folder = join(case_folder, 'segmentation_samples')
    if not isdir(segmentation_samples_folder):
        return
    dice_scores = {i: [] for i in HEC_NAME_LIST}
    sds = {i: [] for i in HEC_NAME_LIST}
    groups = subfolders(segmentation_samples_folder, join=False, prefix='group')
    for g in groups:
        nii_files = subfiles(join(segmentation_samples_folder, g), suffix='.nii.gz')
        for ref_idx in range(len(nii_files)):
            for pred_idx in range(ref_idx + 1, len(nii_files)):
                metrics = compute_metrics_for_case(nii_files[pred_idx], nii_files[ref_idx])

                for i, hec in enumerate(HEC_NAME_LIST):
                    dice_scores[hec].append(metrics[i, 0])
                    sds[hec].append(metrics[i, 1])
    dice_averages = {i: float(np.mean(j)) for i, j in dice_scores.items()}
    sd_averages = {i: np.mean(j) for i, j in sds.items()}
    save_json({"dice": dice_averages, "sd": sd_averages}, join(case_folder, 'inter_rater_disagreement.json'))


def compute_all_inter_rater_disagreement(num_proceses: int = 10, overwrite_existing=False):
    p = Pool(num_proceses)
    case_folders = subfolders(TRAINING_DIR, prefix='case_')
    if not overwrite_existing:
        c = []
        for cs in case_folders:
            if not isfile(join(cs, 'inter_rater_disagreement.json')):
                c.append(cs)
        print(len(c), 'out of', len(case_folders), 'to go...')
        case_folders = c
    r = p.starmap_async(compute_inter_rater_disagreement_for_case, ([i] for i in case_folders))
    _ = r.get()
    p.close()
    p.join()


def aggregate_inter_rater_disagreement():
    case_folders = subfolders(TRAINING_DIR, prefix='case_')
    dice_scores = {i: [] for i in HEC_NAME_LIST}
    sds = {i: [] for i in HEC_NAME_LIST}
    for c in case_folders:
        if isfile(join(TRAINING_DIR, c, 'inter_rater_disagreement.json')):
            inter_rater_disagreement = load_json(join(TRAINING_DIR, c, 'inter_rater_disagreement.json'))
            for i, hec in enumerate(HEC_NAME_LIST):
                dice_scores[hec].append(inter_rater_disagreement["dice"][hec])
                sds[hec].append(inter_rater_disagreement["sd"][hec])
    dice = {i: np.mean(j) for i, j in dice_scores.items()}
    sd = {i: np.mean(j) for i, j in sds.items()}
    return dice, sd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_processes', required=False, default=12, type=int)
    args = parser.parse_args()
    compute_all_inter_rater_disagreement(args.num_processes)
    dice, sd = aggregate_inter_rater_disagreement()
    print(dice, sd)
