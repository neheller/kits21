import argparse
from pathlib import Path
import shutil

import numpy as np
import nibabel as nib

from kits21.annotation.postprocessing import delineation_to_seg, load_json, write_json
from kits21.configuration.labels import KITS_LABEL_NAMES, HEC_CONSTRUCTION_ORDER
from kits21.configuration.paths import TESTING_DIR, SRC_DIR, TRAINING_DIR, CACHE_FILE


def _check_testing_dir_available() -> None:
    assert TESTING_DIR is not None, "KITS21_TEST_DIR does not exist on your system. You are probably not supposed " \
                                    "to run this code :-)"


def _check_src_dir_available() -> None:
    assert SRC_DIR is not None, "KITS21_SERVER_DATA does not exist on your system. You are probably not supposed " \
                                "to run this code :-)"


def get_case_dir(case):
    _check_src_dir_available()
    # TODO remove hardcoding -- test both to find it
    page = int(case // 50)
    tst = "training_data"
    if case >= 210:
        tst = "testing_data"
    return (SRC_DIR / tst / "cases_{:05d}".format(page) / "case_{:05d}".format(case)).resolve(strict=True)


def get_all_case_dirs():
    # TODO set this number dynamically
    return [get_case_dir(i) for i in range(300)]


def get_region_dir(case_dir, region):
    return (case_dir / region).resolve(strict=True)


def get_all_region_dirs(case_dir):
    return [r for r in case_dir.glob("*")]


def get_instance_dir(region_dir, instance):
    return (region_dir / "{:02d}".format(instance)).resolve(strict=True)


def get_all_instance_dirs(region_dir):
    return [i for i in region_dir.glob("*")]


def get_delineation(instance_dir, delineation):
    return (instance_dir / "delineation{}".format(delineation)).resolve(strict=True)


def get_all_delineations(instance_dir):
    return [d for d in instance_dir.glob("delineation*")]


def get_most_recent_save(parent_dir):
    try:
        return sorted([s for s in parent_dir.glob("*")])[-1]
    except Exception as e:
        print()
        print("Error finding most recent save in", str(parent_dir))
        raise e


def update_raw(delineation_path, case_id, in_test_set):
    # Get parent directory (create if necessary)
    destination_parent = TRAINING_DIR / case_id
    if in_test_set:
        _check_testing_dir_available()
        destination_parent = TESTING_DIR / case_id
    if not destination_parent.exists():
        destination_parent.mkdir()
    destination_parent = destination_parent / "raw"
    if not destination_parent.exists():
        destination_parent.mkdir()

    # Get source directory
    src = delineation_path.parent.parent.parent.parent

    # Copy all annotation files to destination
    shutil.copytree(str(src), str(destination_parent), dirs_exist_ok=True)


def get_localization(delineation_path):
    return get_most_recent_save(delineation_path.parent.parent / "localization")


def get_artery_localization(delineation_path):
    pth = delineation_path.parent.parent.parent.parent / "artery" / "00" / "localization"
    if not pth.exists():
        return None
    return get_most_recent_save(pth)


def get_image_path(case_id, in_test_set):
    if in_test_set:
        _check_testing_dir_available()
        return (TESTING_DIR / case_id / "imaging.nii.gz").resolve(strict=True)
    else:
        return (TRAINING_DIR / case_id / "imaging.nii.gz").resolve(strict=True)


def save_segmentation(case_id, region_type, delineation_path, n1img, in_test_set):
    # Create name of destination file
    annotation_num = int(delineation_path.parent.name[-1])
    instance_num = int(delineation_path.parent.parent.name)
    filename = "{}_instance-{}_annotation-{}.nii.gz".format(region_type, instance_num+1, annotation_num)

    # Get parent directory (create if necessary)
    destination_parent = TRAINING_DIR / case_id
    if in_test_set:
        _check_testing_dir_available()
        destination_parent = TESTING_DIR / case_id
    if not destination_parent.exists():
        destination_parent.mkdir()
    destination_parent = destination_parent / "segmentations"
    if not destination_parent.exists():
        destination_parent.mkdir()
    destination = destination_parent / filename

    # Save file
    nib.save(n1img, str(destination))


def run_import(delineation_path):
    # Useful values
    region_type = delineation_path.parent.parent.parent.name
    case_id = delineation_path.parent.parent.parent.parent.name
    in_test_set = False
    if delineation_path.parent.parent.parent.parent.parent.parent.name == "testing_data":
        in_test_set = True

    # Copy updated raw data
    update_raw(delineation_path, case_id, in_test_set)

    # Kidneys require hilum information from the localization
    localization = None
    if region_type == "kidney":
        localization = get_localization(delineation_path)

    # Path to underlying CT scan stored as .nii.gz
    image_path = get_image_path(case_id, in_test_set)

    # Compute and save segmentation based on delineation
    seg_nib = delineation_to_seg(region_type, image_path, delineation_path, localization)
    save_segmentation(case_id, region_type, delineation_path, seg_nib, in_test_set)


def aggregate(parent, region, idnum, agg, affine, agtype="maj"):

    seg_files = [x for x in parent.glob("{}*.nii.gz".format(region))]
    instances = [int(x.stem.split("_")[1].split("-")[1]) for x in seg_files]
    unq_insts = sorted(list(set(instances)))

    reg_agg = None
    for inst in unq_insts:
        inst_agg = None
        n_anns = 0
        for tins, tfnm in zip(instances, seg_files):
            if tins != inst:
                continue
            seg_nib = nib.load(str(tfnm))
            n_anns += 1
            if inst_agg is None:
                inst_agg = np.asanyarray(seg_nib.dataobj)
                affine = seg_nib.affine
            else:
                inst_agg = inst_agg + np.asanyarray(seg_nib.dataobj)

        if agtype == "maj":
            inst = np.greater(inst_agg, n_anns/2).astype(inst_agg.dtype)
        elif agtype == "or":
            inst = np.greater(inst_agg, 0).astype(inst_agg.dtype)
        elif agtype == "and":
            inst = np.equal(inst_agg, n_anns).astype(inst_agg.dtype)

        if reg_agg is None:
            reg_agg = np.copy(inst)
        else:
            reg_agg = np.logical_or(reg_agg, inst).astype(reg_agg.dtype)

    # If no info here, just return what we started with
    if reg_agg is None:
        return agg, affine

    if agg is None:
        agg = idnum*reg_agg
    else:
        agg = np.where(np.logical_not(np.equal(reg_agg, 0)), idnum*reg_agg, agg)

    return agg, affine


def aggregate_case(case_id):
    segs = Path(__file__).resolve().parent.parent / "data" / case_id / "segmentations"

    affine = None
    agg = None
    for labelid in HEC_CONSTRUCTION_ORDER:
        label_name = KITS_LABEL_NAMES[labelid]
        agg, affine = aggregate(segs, label_name, labelid, agg, affine, agtype="or")
    if agg is not None:
        nib.save(
            nib.Nifti1Image(agg.astype(np.int32), affine),
            str(Path(__file__).resolve().parent.parent / "data" / case_id / "aggregated_OR_seg.nii.gz")
        )

    affine = None
    agg = None
    for labelid in HEC_CONSTRUCTION_ORDER:
        label_name = KITS_LABEL_NAMES[labelid]
        agg, affine = aggregate(segs, label_name, labelid, agg, affine, agtype="and")
    if agg is not None:
        nib.save(
            nib.Nifti1Image(agg.astype(np.int32), affine),
            str(Path(__file__).resolve().parent.parent / "data" / case_id / "aggregated_AND_seg.nii.gz")
        )

    affine = None
    agg = None
    for labelid in HEC_CONSTRUCTION_ORDER:
        label_name = KITS_LABEL_NAMES[labelid]
        agg, affine = aggregate(segs, label_name, labelid, agg, affine, agtype="maj")
    if agg is not None:
        nib.save(
            nib.Nifti1Image(agg.astype(np.int32), affine),
            str(Path(__file__).resolve().parent.parent / "data" / case_id / "aggregated_MAJ_seg.nii.gz")
        )


def main(args):
    cache = load_json(CACHE_FILE)
    cli = True
    if args.case is not None:
        case_dirs = [get_case_dir(args.case)]
    else:
        cli = False
        case_dirs = get_all_case_dirs()

    for case_dir in case_dirs:
        print(case_dir.name)
        reaggregate = args.reaggregate
        if cli and args.region is not None:
            region_dirs = [get_region_dir(case_dir, args.region)]
        else:
            cli = False
            region_dirs = get_all_region_dirs(case_dir)

        for region_dir in region_dirs:
            if cli and args.instance is not None:
                instance_dirs = [get_instance_dir(region_dir, args.instance - 1)]
            else:
                cli = False
                instance_dirs = get_all_instance_dirs(region_dir)

            for instance_dir in instance_dirs:
                if cli and args.delineation is not None:
                    delineations = [get_delineation(instance_dir, args.delineation)]
                else:
                    delineations = get_all_delineations(instance_dir)

                for delineation in delineations:
                    dln_file = get_most_recent_save(delineation)
                    cache_key = str(delineation.relative_to(delineation.parent.parent.parent.parent))
                    if args.regenerate or cache_key not in cache or cache[cache_key] != dln_file.name:
                        run_import(dln_file)
                        cache[cache_key] = dln_file.name
                        write_json(CACHE_FILE, cache)
                        reaggregate = True

        if reaggregate:
            aggregate_case(case_dir.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--case", help="The index of the case to import", type=int)
    parser.add_argument("-r", "--region", help="The type of region to import", type=str)
    parser.add_argument("-i", "--instance", help="The index of the instance of that region to import", type=int)
    parser.add_argument("-d", "--delineation", help="The index of the delineation of that instance to import "
                                                    "(1, 2, or 3)", type=int)
    parser.add_argument("--regenerate", help="Regenerate segmentations regardless of cached values",
                        action="store_true")
    parser.add_argument("--reaggregate", help="Reaggregate segmentations regardless of whether it was changed",
                        action="store_true")
    if __name__ == "__main__":
        cl_args = parser.parse_args()
        main(cl_args)
