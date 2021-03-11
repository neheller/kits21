import sys
import argparse
from pathlib import Path
import os
import json

from annotation.postprocessing import delineation_to_seg, load_json, write_json 


IMAGE_ROOT = Path(os.environ["KITS21_MASTER"]).resolve(strict=True)
SRC_DIR = Path(os.environ["KITS21_SERVER_DATA"]).resolve(strict=True)
CACHE_FILE = Path(__file__).parent / "cache.json"


def get_case_dir(case):
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
    return (instance_dir / "delination{}".format(delineation)).resolve(strict=True)


def get_all_delineations(instance_dir):
    return [d for d in instance_dir.glob("delineation*")]


def get_most_recent_save(parent_dir):
    return sorted([s for s in parent_dir.glob("*")])[-1]


def update_raw(delineation_path):
    # TODO
    # shutil.copytree(..., dirs_exist_ok=True) will be helpful here
    pass


def get_localization(delineation_path):
    # TODO
    return None


def get_image_path(case_id):
    image_path = IMAGE_ROOT / case_id / "imaging.nii.gz"
    return image_path


def save_segmentation(case_id, region_type, delineation_path, n1img):
    # TODO
    # Get destination based on delination path
    pass


def run_import(delineation_path):
    print("Running import for {}!".format(str(delineation_path)))
    sys.exit()
    # Useful values
    region_type = delineation_path.parent.parent.parent.name
    case_id = delineation_path.parent.parent.parent.parent.name

    # Copy updated raw data
    update_raw(delineation_path)

    # Kidneys require hilum information from the localization
    localization = None
    if region_type == "kidney":
        localization = get_localization(delineation)
    
    # Path to underlying CT scan stored as .nii.gz
    image_path = get_image_path(case_id)

    # Compute and save segmentation based on delineation
    seg_nib = delineation_to_seg(region_dir.name, image_path, delineation, localization)
    save_segmentation(case_id, region_type, delineation, seg_nib)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case", help="The index of the case to import", type=int)
parser.add_argument("-r", "--region", help="The type of region to import", type=int)
parser.add_argument("-i", "--instance", help="The index of the instance of that region to import", type=int)
parser.add_argument("-d", "--delineation", help="The index of the delineation of that instance to import (1, 2, or 3)", type=int)
parser.add_argument("--regenerate", help="Regenerate segmentations regardless of cached values", action="store_true")
if __name__ == "__main__":
    cache = load_json(CACHE_FILE)
    args = parser.parse_args()
    cli = True
    if args.case:
        case_dirs = [get_case_dir(args.case)]            
    else:
        cli = False
        case_dirs = get_all_case_dirs()

    for case_dir in case_dirs:
        if cli and args.region:
            region_dirs = [get_region_dir(case_dirs, args.region)]
        else:
            cli = False
            region_dirs = get_all_region_dirs(case_dir)

        for region_dir in region_dirs:
            if cli and args.instance:
                instance_dirs = [get_instance_dir(region_dir, args.instance)]
            else:
                cli = False
                instance_dirs = get_all_instance_dirs(region_dir)

            for instance_dir in instance_dirs:
                if cli and args.delineation:
                    delineations = [get_delineation(instance_dir, args.delineation)]
                else:
                    delineations = get_all_delineations(instance_dir)

                for delineation in delineations:
                    dln_file = get_most_recent_save(delineation)
                    print(dln_file)
                    cache_key = str(delineation.relative_to(SRC_DIR))
                    if args.regenerate or cache_key not in cache or cache[cache_key] != dln_file.name:
                        run_import(dln_file)
                        cache[cache_key] = dln_file.name
                        write_json(CACHE_FILE, cache)
