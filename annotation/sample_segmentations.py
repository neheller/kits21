from multiprocessing import Pool
from typing import Callable

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.Task134_cryoET import flood_fill_hull, morphology_valid_region
from skimage.morphology import binary_closing, ball
from skimage.transform import resize

from configuration import HEC_CONSTRUCTION_ORDER


def build_segmentation_replace_ureter_with_kidney_label_in_convhull(kidney_files, tumor_files, cyst_files, ureter_files,
                                                                    artery_files, vein_files, output_file: str
                                                                    ) -> None:
    """
    create a convex hull around the kidneys. Replace ureter label within the convex hull with kidney label

    :param kidney_files:
    :param tumor_files:
    :param cyst_files:
    :param ureter_files:
    :param artery_files:
    :param vein_files:
    :param output_file:
    :return:
    """
    labelid_files_mapping = {
        i: j if j is not None else list() for i, j in {
            1: kidney_files,
            6: tumor_files,
            5: cyst_files,
            2: ureter_files,
            3: artery_files,
            4: vein_files
        }.items()}

    seg = None
    seg_itk = None

    # compute convex hulls of kidneys
    kidneys = [sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.uint8) for f in kidney_files]
    kidney_convhulls = [flood_fill_hull(i, subsampling=2)[0][None] for i in kidneys]
    kidney_convhulls = np.vstack(kidney_convhulls).sum(0) > 0
    del kidneys
    for current_label in HEC_CONSTRUCTION_ORDER:
        files = labelid_files_mapping[current_label]
        for f in files:
            if seg is None:
                seg_itk = sitk.ReadImage(f)
                new_seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
                seg = np.zeros_like(new_seg)
                seg[new_seg == 1] = current_label
                if current_label == 2:
                    seg[kidney_convhulls & (new_seg == 1)] = 1
            else:
                new_seg = sitk.GetArrayFromImage(sitk.ReadImage(f))
                seg[new_seg == 1] = current_label
                if current_label == 2:
                    seg[kidney_convhulls & (new_seg == 1)] = 1

    seg = seg.astype(np.uint8)
    seg = sitk.GetImageFromArray(seg)
    seg.CopyInformation(seg_itk)
    sitk.WriteImage(seg, output_file)


def build_segmentation_replace_ureter_with_kidney_label_in_closed_kidney(kidney_files, tumor_files, cyst_files,
                                                                         ureter_files, artery_files, vein_files,
                                                                         output_file: str,
                                                                         radius: int = 11) -> None:
    """
    perform morphological closing on the kidney labels (with sphere of given radius, radius is in mm(!)). Then replace
    all ureter pixels within the closed kidney label with kidney. (closed kidney label is only used to convert
    the ureter labels, the actual kidney label remains unchanged)

    :param kidney_files:
    :param tumor_files:
    :param cyst_files:
    :param ureter_files:
    :param artery_files:
    :param vein_files:
    :param output_file:
    :param radius: radius of closing operation in mm (!), not pixels
    :return:
    """
    labelid_files_mapping = {
        i: j if j is not None else list() for i, j in {
            1: kidney_files,
            6: tumor_files,
            5: cyst_files,
            2: ureter_files,
            3: artery_files,
            4: vein_files
        }.items()}

    seg = None
    seg_itk = None

    # compute convex hulls of kidneys
    kindey_0 = sitk.ReadImage(kidney_files[0])
    spacing = list(kindey_0.GetSpacing())[::-1]
    del kindey_0
    strel = ball(1 / min(spacing) * radius)
    new_size_ball = [np.round(radius / spacing[i] * 2 + 1).astype(int) for i in range(3)]
    strel = resize(strel.astype(float), output_shape=new_size_ball, order=0, anti_aliasing=False).astype(np.uint8)
    print(kidney_files[0].split('/')[-3], kidney_files[0].split('/')[-1], spacing, strel.shape, np.unique(strel))

    kidneys = [sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.uint8) for f in kidney_files]
    kidney_closed = [morphology_valid_region(i, strel, binary_closing)[None] for i in kidneys]
    kidney_closed = np.vstack(kidney_closed).sum(0) > 0
    del kidneys
    for current_label in HEC_CONSTRUCTION_ORDER:
        files = labelid_files_mapping[current_label]
        for f in files:
            if seg is None:
                seg_itk = sitk.ReadImage(f)
                new_seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
                seg = np.zeros_like(new_seg)
                seg[new_seg == 1] = current_label
                if current_label == 2:
                    seg[kidney_closed & (new_seg == 1)] = 1
            else:
                new_seg = sitk.GetArrayFromImage(sitk.ReadImage(f))
                seg[new_seg == 1] = current_label
                if current_label == 2:
                    seg[kidney_closed & (new_seg == 1)] = 1

    seg = seg.astype(np.uint8)
    seg = sitk.GetImageFromArray(seg)
    seg.CopyInformation(seg_itk)
    sitk.WriteImage(seg, output_file)


def build_segmentation_ureter_on_top_of_kidney(kidney_files, tumor_files, cyst_files, ureter_files, artery_files,
                                               vein_files, output_file: str
                                               ) -> None:
    """
    the default replaces ureter label with the kidney segmentation by placing kidney after ureter has been set. Here
    the manual ureter segmentations are retained and instead overwrite existing kidney voxels
    :param kidney_files:
    :param tumor_files:
    :param cyst_files:
    :param ureter_files:
    :param artery_files:
    :param vein_files:
    :param output_file:
    :return:
    """
    labelid_files_mapping = {
        i: j if j is not None else list() for i, j in {
            1: kidney_files,
            6: tumor_files,
            5: cyst_files,
            2: ureter_files,
            3: artery_files,
            4: vein_files
        }.items()}

    seg = None
    seg_itk = None
    order = (1, 2, 4, 3, 5, 6)
    for current_label in order:
        files = labelid_files_mapping[current_label]
        for f in files:
            if seg is None:
                seg_itk = sitk.ReadImage(f)
                seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
                seg[seg == 1] = current_label
            else:
                new_seg = sitk.GetArrayFromImage(sitk.ReadImage(f))
                seg[new_seg == 1] = current_label

    seg = seg.astype(np.uint8)
    seg = sitk.GetImageFromArray(seg)
    seg.CopyInformation(seg_itk)
    sitk.WriteImage(seg, output_file)


def build_segmentation_default(kidney_files, tumor_files, cyst_files, ureter_files, artery_files, vein_files,
                               output_file: str
                               ) -> None:
    """
    nick's default

    :param kidney_files:
    :param tumor_files:
    :param cyst_files:
    :param ureter_files:
    :param artery_files:
    :param vein_files:
    :param output_file:
    :return:
    """
    labelid_files_mapping = {
        i: j if j is not None else list() for i, j in {
            1: kidney_files,
            6: tumor_files,
            5: cyst_files,
            2: ureter_files,
            3: artery_files,
            4: vein_files
        }.items()}

    seg = None
    seg_itk = None

    for current_label in HEC_CONSTRUCTION_ORDER:
        files = labelid_files_mapping[current_label]
        for f in files:
            if seg is None:
                seg_itk = sitk.ReadImage(f)
                seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
                seg[seg == 1] = current_label
            else:
                new_seg = sitk.GetArrayFromImage(sitk.ReadImage(f))
                seg[new_seg == 1] = current_label

    seg = seg.astype(np.uint8)
    seg = sitk.GetImageFromArray(seg)
    seg.CopyInformation(seg_itk)
    sitk.WriteImage(seg, output_file)


def _get_instances(list_of_fnames):
    return np.unique([i.split('_')[1].split("-")[1] for i in list_of_fnames])


def generate_segmentations(kits_data_base_dir: str,
                           num_segmentations: int = 100,
                           n_processes: int = 5,
                           skip_existing: bool = False,
                           verbose: bool = False,
                           segmentation_build_fn: Callable = build_segmentation_default,
                           custom_suffix: str = None) -> None:
    """
    This function randomly samples valid segmentations from the provided annotations for each instance
    (3 annotations per instance).
    For each image, the random state is initialized with the case id (which is read from the folder name of the cases).
    This ensures that the code always generates the same segmentations if run multiple times.
    Sampled segmentations will be saved in kits_data_base_dir/CASE/segmentation_samples. They are called
    sample_XXXX.nii.gz where XXXX is a number from 0 to num_segmentations-1.

    :param kits_data_base_dir: kits21/data
    :param num_segmentations: how many segmentations are generated per case? default 100
    :param n_processes: how many CPU cores should be used to generate segmentations? Generating segmentations can
    take a lot of RAM, so be careful! Guidance: 5 processes are the maximum for 32GB RAM
    :param skip_existing: skips existing samples
    :param verbose: print a warning !=3 annotations are available for some instances. (each should have been
    annotated by 3 annotators)
    :param segmentation_build_fn: experimental. When changing this, please make sure to set skip_existing=True so
    that everything can be overwritten. We strongly recommend you stick to the default because this is what will be
    used on the test set as well (the default may change, so update your repo frequently!)
    :param custom_suffix: if changing segmentation_build_fn you can set a custom_suffix to distinghuish the newly
    created sampled from the old ones. If custom_suffix is set, the samples will be saved as
    sample_XXXX_[CUSTOM_SUFFIX].nii.gz
    :return:
    """
    p = Pool(n_processes)
    results = []
    case_dirs = subdirs(kits_data_base_dir, join=False, prefix='case_')
    for c in case_dirs:
        segmentation_dir = join(kits_data_base_dir, c, 'segmentations')
        if not isdir(segmentation_dir):
            # segmentations for this case not published yet, skip
            continue

        case_id = int(c.split("_")[-1])

        files = subfiles(segmentation_dir, suffix='.nii.gz', join=False)
        kidney_files = [i for i in files if i.startswith('kidney_instance')]
        kidney_instances = _get_instances(kidney_files)
        kidney_files_per_instance = []
        for inst in kidney_instances:
            kidney_files_per_instance.append(
                [join(segmentation_dir, i) for i in kidney_files if i.find('_instance-%s_' % inst) != -1])

        artery_files = [i for i in files if i.startswith('artery_instance')]
        artery_instances = _get_instances(artery_files)
        artery_files_per_instance = []
        for inst in artery_instances:
            artery_files_per_instance.append(
                [join(segmentation_dir, i) for i in artery_files if i.find('_instance-%s_' % inst) != -1])

        tumor_files = [i for i in files if i.startswith('tumor_instance')]
        tumor_instances = _get_instances(tumor_files)
        tumor_files_per_instance = []
        for inst in tumor_instances:
            tumor_files_per_instance.append(
                [join(segmentation_dir, i) for i in tumor_files if i.find('_instance-%s_' % inst) != -1])

        cyst_files = [i for i in files if i.startswith('cyst_instance')]
        cyst_instances = _get_instances(cyst_files)
        cyst_files_per_instance = []
        for inst in cyst_instances:
            cyst_files_per_instance.append(
                [join(segmentation_dir, i) for i in cyst_files if i.find('_instance-%s_' % inst) != -1])

        ureter_files = [i for i in files if i.startswith('ureter_instance')]
        ureter_instances = _get_instances(ureter_files)
        ureter_files_per_instance = []
        for inst in ureter_instances:
            ureter_files_per_instance.append(
                [join(segmentation_dir, i) for i in ureter_files if i.find('_instance-%s_' % inst) != -1])

        vein_files = [i for i in files if i.startswith('vein_instance')]
        vein_instances = _get_instances(vein_files)
        vein_files_per_instance = []
        for inst in vein_instances:
            vein_files_per_instance.append(
                [join(segmentation_dir, i) for i in vein_files if i.find('_instance-%s_' % inst) != -1])

        if verbose:
            for annos in kidney_files_per_instance + tumor_files_per_instance + cyst_files_per_instance + \
                         ureter_files_per_instance + artery_files_per_instance + vein_files_per_instance:
                if len(annos) != 3:
                    print("seems like not all three annotators annotated this: ", annos)

        rs = np.random.RandomState(case_id)
        output_folder = join(kits_data_base_dir, c, 'segmentation_samples')
        maybe_mkdir_p(output_folder)
        for i in range(num_segmentations):
            if custom_suffix is not None:
                output_file = join(output_folder, 'sample_%04d_%s.nii.gz' % (i, custom_suffix))
            else:
                output_file = join(output_folder, 'sample_%04d.nii.gz' % i)
            if isfile(output_file) and skip_existing:
                continue
            kidney_files = [rs.choice(i) for i in kidney_files_per_instance] if len(
                kidney_files_per_instance) > 0 else []
            tumor_files = [rs.choice(i) for i in tumor_files_per_instance] if len(
                tumor_files_per_instance) > 0 else []
            cyst_files = [rs.choice(i) for i in cyst_files_per_instance] if len(
                cyst_files_per_instance) > 0 else []
            ureter_files = [rs.choice(i) for i in ureter_files_per_instance] if len(
                ureter_files_per_instance) > 0 else []
            artery_files = [rs.choice(i) for i in artery_files_per_instance] if len(
                artery_files_per_instance) > 0 else []
            vein_files = [rs.choice(i) for i in vein_files_per_instance] if len(
                vein_files_per_instance) > 0 else []
            results.append(p.starmap_async(segmentation_build_fn,
                                           ((kidney_files,
                                             tumor_files,
                                             cyst_files,
                                             ureter_files,
                                             artery_files,
                                             vein_files,
                                             output_file),)
                                           ))
            save_json({
                'kidney': [i.split('/')[-1] for i in kidney_files],
                'tumor': [i.split('/')[-1] for i in tumor_files],
                'cyst': [i.split('/')[-1] for i in cyst_files],
                'ureter': [i.split('/')[-1] for i in ureter_files],
                'artery': [i.split('/')[-1] for i in artery_files],
                'vein': [i.split('/')[-1] for i in vein_files],
            }, join(output_folder, 'sample_%04d.json' % i))
    [i.get() for i in results]
    p.close()
    p.join()


if __name__ == "__main__":
    kits_data_base_dir = '/data'
    num_segmentations = 100
    n_processes = 6
    generate_segmentations(kits_data_base_dir, num_segmentations, n_processes, skip_existing=False, verbose=True)
