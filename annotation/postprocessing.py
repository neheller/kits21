from pathlib import Path
import json

import nibabel as nib


def load_json(json_path):
    with json_path.open() as f:
        return json.loads(f.read())


def write_json(json_path, data):
    with json_path.open("w") as f:
        return f.write(json.dumps(data, indent=2))


def get_containing_box(dln):
    # TODO
    pass


def get_cropped_scan(cbox, img_nib):
    # TODO
    pass


def generate_cropped_drawing(cbox, dln):
    # TODO
    pass


def generate_segmentation(region_type, cropped_img, cropped_drw, lzn=None):
    # TODO
    pass


def inflate_seg_to_image_size(cbox, cropped_seg):
    # TODO
    pass


def delineation_to_seg(region_type, image_path, delineation_path, localization_path=None):
    # Read and parse delination and (maybe) localization from file
    lzn = None
    if region_type == "kidney":
        assert localization_path is not None
        lzn = load_json(localization_path)
    dln = load_json(delineation_path)

    # Read CT scan
    img_nib = nib.load(str(image_path))

    # Crop image to the smallest possible box for memory/computational efficiency
    cbox = get_containing_box(dln)
    cropped_img = get_cropped_scan(cbox, img_nib)

    # Generate the drawing made by the annotator
    cropped_drw = generate_cropped_drawing(cbox, dln)

    # Apply heuristics to infer segmentation based on drawing and image
    cropped_seg = generate_segmentation(region_type, cropped_img, cropped_drw, lzn)

    # Undo cropping to get final segmentation
    seg = inflate_seg_to_image_size(cbox, cropped_seg)

    # Return the seg in nifti format
    return nib.Nifti1Image(seg, img_nib.affine)
