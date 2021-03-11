import sys
import argparse
from pathlib import Path
import os
import json
import shutil

import nibabel as nib

from annotation.postprocessing import delineation_to_seg, load_json, write_json 


TRAINING_DIR = Path(__file__).parent.parent / "data"
TESTING_DIR = Path(os.environ["KITS21_TEST_DIR"]).resolve(strict=True)
SRC_DIR = Path(os.environ["KITS21_SERVER_DATA"]).resolve(strict=True)
CACHE_FILE = Path(__file__).parent / "cache.json"


def get_case_dir(case):
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
        raise(e)


def update_raw(delineation_path, case_id, in_test_set):
    # Get parent directory (create if necessary)
    destination_parent = TRAINING_DIR / case_id
    if in_test_set:
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


def get_image_path(case_id, in_test_set):
    if in_test_set:
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


def main(args):
    cache = load_json(CACHE_FILE)
    cli = True
    if args.case is not None:
        case_dirs = [get_case_dir(args.case)]            
    else:
        cli = False
        case_dirs = get_all_case_dirs()

    for case_dir in case_dirs:
        if cli and args.region is not None:
            region_dirs = [get_region_dir(case_dir, args.region)]
        else:
            cli = False
            region_dirs = get_all_region_dirs(case_dir)

        for region_dir in region_dirs:
            if cli and args.instance is not None:
                instance_dirs = [get_instance_dir(region_dir, args.instance)]
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


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case", help="The index of the case to import", type=int)
parser.add_argument("-r", "--region", help="The type of region to import", type=str)
parser.add_argument("-i", "--instance", help="The index of the instance of that region to import", type=int)
parser.add_argument("-d", "--delineation", help="The index of the delineation of that instance to import (1, 2, or 3)", type=int)
parser.add_argument("--regenerate", help="Regenerate segmentations regardless of cached values", action="store_true")
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)