import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
from multiprocessing import Pool

from evaluation.metrics import KITS_LABEL_CLASS_MAP


def build_segmentation(kidney_files, tumor_files, cyst_files, ureter_files, artery_files, vein_files, output_file: str,
                       random_seed=None) -> None:
    label_file_mapping = {i: j for i, j in {
        'tumor': tumor_files,
        'cyst': cyst_files,
        'ureter': ureter_files,
        'artery': artery_files,
        'vein': vein_files
    }.items() if j is not None}

    remaining_labels = list(label_file_mapping.keys())

    rs = np.random.RandomState(random_seed)

    seg_itk = sitk.ReadImage(kidney_files[0])
    seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
    seg[seg == 1] = KITS_LABEL_CLASS_MAP['kidney']
    for f in kidney_files[1:]:
        seg[sitk.GetArrayFromImage(sitk.ReadImage(f)) == 1] = KITS_LABEL_CLASS_MAP['kidney']

    while len(remaining_labels) > 0:
        random_label = rs.choice(remaining_labels)
        remaining_labels.remove(random_label)
        for f in label_file_mapping[random_label]:
            seg[sitk.GetArrayFromImage(sitk.ReadImage(f)) == 1] = KITS_LABEL_CLASS_MAP[random_label]

    seg = seg.astype(np.uint8)
    seg = sitk.GetImageFromArray(seg)
    seg.CopyInformation(seg_itk)
    sitk.WriteImage(seg, output_file)


def _get_instances(list_of_fnames):
    return np.unique([i.split('_')[1].split("-")[1] for i in list_of_fnames])


def generate_segmentations(kits_data_base_dir: str, num_segmentations: int = 100,
                           n_processes: int = 8) -> None:
    p = Pool(n_processes)
    results = []
    case_dirs = subdirs(kits_data_base_dir, join=False, prefix='case_')
    for c in case_dirs:
        segmentation_dir = join(kits_data_base_dir, c, 'segmentations')
        if not isdir(segmentation_dir):
            # segmentations not published yet
            continue

        case_id = int(c.split("_")[-1])

        files = subfiles(segmentation_dir, suffix='.nii.gz', join=False)
        kidney_files = [i for i in files if i.startswith('kidney_instance')]
        kidney_instances = _get_instances(kidney_files)
        kidney_files_per_instance = []
        for inst in kidney_instances:
            kidney_files_per_instance.append([join(segmentation_dir, i) for i in kidney_files if i.find('_instance-%s_' % inst) != -1])

        artery_files = [i for i in files if i.startswith('artery_instance')]
        artery_instances = _get_instances(artery_files)
        artery_files_per_instance = []
        for inst in artery_instances:
            artery_files_per_instance.append([join(segmentation_dir, i) for i in artery_files if i.find('_instance-%s_' % inst) != -1])

        tumor_files = [i for i in files if i.startswith('tumor_instance')]
        tumor_instances = _get_instances(tumor_files)
        tumor_files_per_instance = []
        for inst in tumor_instances:
            tumor_files_per_instance.append([join(segmentation_dir, i) for i in tumor_files if i.find('_instance-%s_' % inst) != -1])

        cyst_files = [i for i in files if i.startswith('cyst_instance')]
        cyst_instances = _get_instances(cyst_files)
        cyst_files_per_instance = []
        for inst in cyst_instances:
            cyst_files_per_instance.append([join(segmentation_dir, i) for i in cyst_files if i.find('_instance-%s_' % inst) != -1])

        ureter_files = [i for i in files if i.startswith('ureter_instance')]
        ureter_instances = _get_instances(ureter_files)
        ureter_files_per_instance = []
        for inst in ureter_instances:
            ureter_files_per_instance.append([join(segmentation_dir, i) for i in ureter_files if i.find('_instance-%s_' % inst) != -1])

        vein_files = [i for i in files if i.startswith('vein_instance')]
        vein_instances = _get_instances(vein_files)
        vein_files_per_instance = []
        for inst in vein_instances:
            vein_files_per_instance.append([join(segmentation_dir, i) for i in vein_files if i.find('_instance-%s_' % inst) != -1])

        rs = np.random.RandomState(case_id)
        output_folder = join(kits_data_base_dir, c, 'segmentation_samples')
        maybe_mkdir_p(output_folder)
        for i in range(num_segmentations):
            output_file = join(output_folder, 'sample_%04d.nii.gz' % i)
            results.append(p.starmap_async(build_segmentation,
                                           (([rs.choice(i) for i in kidney_files_per_instance],
                                             [rs.choice(i) for i in artery_files_per_instance] if len(artery_files_per_instance) > 0 else [],
                                             [rs.choice(i) for i in tumor_files_per_instance] if len(tumor_files_per_instance) > 0 else [],
                                             [rs.choice(i) for i in cyst_files_per_instance] if len(cyst_files_per_instance) > 0 else [],
                                             [rs.choice(i) for i in ureter_files_per_instance] if len(ureter_files_per_instance) > 0 else [],
                                             [rs.choice(i) for i in vein_files_per_instance] if len(vein_files_per_instance) > 0 else [],
                                             output_file,
                                             case_id + i),)
                                           ))
    [i.get() for i in results]
    p.close()
    p.join()


if __name__ == "__main__":
    kits_data_base_dir = '/home/fabian/http_git/kits21/data'
    num_segmentations = 100
    n_processes = 6
    generate_segmentations(kits_data_base_dir, num_segmentations, n_processes)