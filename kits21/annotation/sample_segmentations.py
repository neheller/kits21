import argparse
import os
import shutil
from multiprocessing import Pool

import decorator

from kits21.configuration.labels import LABEL_AGGREGATION_ORDER
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

from kits21.configuration.paths import TRAINING_DIR

"""In order to find the inter rater variability we need to evaluate sampled segmentations against each other. The 
problem with that is that annotation-X is not the same annotator between instances and labels, meaning that we need 
to take care not to underestimate the inter-rater variability by comparing segmentations that contain the same intance 
annotations. Example: a kidney label generated from instance-1_annotation-1 + instance-2_annotation-2 cannot be 
evaluated against a label generated from instance-1_annotation-1 + instance-2_annotation-3 because the instance-1 
annotator is reused, resulting in an underestimation of the inter-rater variability"""


def get_number_of_instances(segmentations_folder: str, label_name: str = 'kidney'):
    nii_files = subfiles(segmentations_folder, suffix='.nii.gz', prefix=label_name, join=False)
    instance_strings = [i.split('_')[1] for i in nii_files]
    instance_idx = [int(i.split('-')[-1]) for i in instance_strings]
    return list(np.unique(instance_idx))


def get_annotations(segmentations_folder: str, label_name: str = 'kidney', instance_idx: int = 1):
    nii_files = subfiles(segmentations_folder, suffix='.nii.gz', prefix=label_name + '_instance-%s' % instance_idx, join=False)
    annotation_strings = [i.split('_')[-1][:-7] for i in nii_files]
    annotation_idx = [int(i.split('-')[-1]) for i in annotation_strings]
    return list(np.unique(annotation_idx))


def build_segmentation(kidney_files, tumor_files, cyst_files, output_file: str) -> None:
    labelid_files_mapping = {
        i: j if j is not None else list() for i, j in {
            1: kidney_files,
            3: tumor_files,
            2: cyst_files,
        }.items()}

    seg = None
    seg_itk = None

    for current_label in LABEL_AGGREGATION_ORDER:
        files = labelid_files_mapping[current_label]
        for f in files:
            if seg is None:
                seg_itk = sitk.ReadImage(f)
                seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
                seg[seg == 1] = current_label
            else:
                new_seg = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.uint8)
                seg[new_seg == 1] = current_label

    seg = seg.astype(np.uint8)
    seg = sitk.GetImageFromArray(seg)
    seg.CopyInformation(seg_itk)
    sitk.WriteImage(seg, output_file)


def generate_samples(segmentations_folder: str, samples_output_folder: str, num_groups=3, seed=1234):
    """
    We do this the stupid way, because the smart way is above my head right now.

    Why groups? We can only determine the inter-rater variability within each group because otherwise we might see the
    same annotation more than once. So we compute the inter-rater variability within each group, then average across
    groups

    :param segmentations_folder:
    :param samples_output_folder:
    :return:
    """
    instances_kidney = get_number_of_instances(segmentations_folder, 'kidney')
    instances_cyst = get_number_of_instances(segmentations_folder, 'cyst')
    instances_tumor = get_number_of_instances(segmentations_folder, 'tumor')

    anno_kidney = [get_annotations(segmentations_folder, 'kidney', i) for i in instances_kidney]
    anno_cyst = [get_annotations(segmentations_folder, 'cyst', i) for i in instances_cyst]
    anno_tumor = [get_annotations(segmentations_folder, 'tumor', i) for i in instances_tumor]

    num_kidney_seg_per_group = min([len(i) for i in anno_kidney]) if len(anno_kidney) > 0 else np.nan
    num_cyst_seg_per_group = min([len(i) for i in anno_cyst]) if len(anno_cyst) > 0 else np.nan
    num_tumor_seg_per_group = min([len(i) for i in anno_tumor]) if len(anno_tumor) > 0 else np.nan

    n_seg_per_group = int(np.nanmin((num_kidney_seg_per_group, num_cyst_seg_per_group, num_tumor_seg_per_group)))

    rs = np.random.RandomState(seed)
    for n in range(num_groups):
        output_folder = join(samples_output_folder, 'group_%s' % str(n))
        maybe_mkdir_p(output_folder)

        random_offsets_kidney = [rs.randint(0, len(i)) for i in anno_kidney]
        random_offsets_tumor = [rs.randint(0, len(i)) for i in anno_tumor]
        random_offsets_cyst = [rs.randint(0, len(i)) for i in anno_cyst]

        for i in range(n_seg_per_group):
            output_filename = 'kidney'
            kidney_files = []
            for ik, inst_k in enumerate(instances_kidney):
                anno = anno_kidney[ik][(random_offsets_kidney[ik] + i) % len(anno_kidney[ik])]
                kidney_files.append(join(segmentations_folder, 'kidney_instance-%s_annotation-%s.nii.gz' % (inst_k, anno)))
                output_filename += "_i%sa%s" % (inst_k, anno)

            output_filename += '_cyst'
            cyst_files = []
            for ic, inst_c in enumerate(instances_cyst):
                anno = anno_cyst[ic][(random_offsets_cyst[ic] + i) % len(anno_cyst[ic])]
                cyst_files.append(join(segmentations_folder, 'cyst_instance-%s_annotation-%s.nii.gz' % (inst_c, anno)))
                output_filename += "_i%sa%s" % (inst_c, anno)

            output_filename += '_tumor'
            tumor_files = []
            for it, inst_t in enumerate(instances_tumor):
                anno = anno_tumor[it][(random_offsets_tumor[it] + i) % len(anno_tumor[it])]
                tumor_files.append(join(segmentations_folder, 'tumor_instance-%s_annotation-%s.nii.gz' % (inst_t, anno)))
                output_filename += "_i%sa%s" % (inst_t, anno)

            output_filename += ".nii.gz"
            build_segmentation(kidney_files, tumor_files, cyst_files, join(output_folder, output_filename))


def generate_samples_for_all_cases(num_processes: int, num_groups_per_case: int = 5) -> None:
    """
    THIS WILL DELETE PREVIOUSLY EXISTING SAMPLES! BEWARE!

    :param num_processes:
    :param num_groups_per_case:
    :return:
    """
    cases = subfolders(TRAINING_DIR, prefix='case_', join=False)
    case_ids = [int(i.split('_')[-1]) for i in cases]
    p = Pool(num_processes)
    res = []
    for case, caseid in zip(cases, case_ids):
        if isdir(join(TRAINING_DIR, case, 'segmentations')) and \
                len(subfiles(join(TRAINING_DIR, case, 'segmentations'), suffix='.nii.gz')) > 0:
            if isdir(join(TRAINING_DIR, case, 'segmentation_samples')):
                shutil.rmtree(join(TRAINING_DIR, case, 'segmentation_samples'))
            if isfile(join(TRAINING_DIR, case, 'inter_rater_variability.json')):
                os.remove(join(TRAINING_DIR, case, 'inter_rater_variability.json'))
            res.append(p.starmap_async(
                generate_samples, ((join(TRAINING_DIR, case, 'segmentations'),
                                    join(TRAINING_DIR, case, 'segmentation_samples'),
                                    num_groups_per_case,
                                    caseid), )
            ))
    _ = [i.get() for i in res]
    p.close()
    p.join()