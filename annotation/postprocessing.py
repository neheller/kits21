import sys
from pathlib import Path
import json

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
import torch
import torch.nn.functional
from scipy import signal

#pylint: disable=no-member


def load_json(json_path):
    with json_path.open() as f:
        return json.loads(f.read())


def write_json(json_path, data):
    with json_path.open("w") as f:
        return f.write(json.dumps(data, indent=2))


def get_containing_box(dln, shape):
    annotated_frames = set([])
    maxs = [0, 0]
    mins = [np.inf, np.inf]
    max_sz = 0
    for ann in dln["annotations"]:
        annotated_frames.add(ann["frame"])
        for pt in ann["spatial_payload"]:
            if pt[0] > maxs[0]:
                maxs[0] = pt[0]
            if pt[1] > maxs[1]:
                maxs[1] = pt[1]
            if pt[0] < mins[0]:
                mins[0] = pt[0]
            if pt[1] < mins[1]:
                mins[1] = pt[1]
        if ann["line_size"] > max_sz:
            max_sz = ann["line_size"]

    afrms = sorted(list(annotated_frames))
    last = afrms[0]
    min_step = np.inf
    for afrm in afrms[1:]:
        if afrm - last < min_step:
            min_step = afrm - last
        last = afrm

    return {
        "xmin": max(0, int(np.floor(mins[0] - max_sz))),
        "xmax": min(shape[2], int(np.ceil(maxs[0] + max_sz))),
        "ymin": max(0, int(np.floor(mins[1] - max_sz))),
        "ymax": min(shape[1], int(np.ceil(maxs[1] + max_sz))),
        "zmin": max(0, min(afrms) - min_step),
        "zmax": min(shape[0], max(afrms) + min_step),
        "step": min_step,
        "xdim": shape[2],
        "ydim": shape[1],
        "zdim": shape[0]
    }


def get_cropped_scan(cbox, img_nib):
    return img_nib.get_fdata()[
        cbox["zmin"]:cbox["zmax"] + 1,
        cbox["ymin"]:cbox["ymax"] + 1,
        cbox["xmin"]:cbox["xmax"] + 1
    ]


def generate_cropped_drawing_interior(cbox, dln):
    ret = np.zeros((
        cbox["zmax"] - cbox["zmin"] + 1,
        cbox["ymax"] - cbox["ymin"] + 1,
        cbox["xmax"] - cbox["xmin"] + 1
    ), dtype=np.int)

    for i in range(ret.shape[0]):
        with Image.new("L", (ret.shape[2]*10, ret.shape[1]*10)) as im:
            draw = ImageDraw.Draw(im)
            drew = False
            for stroke in dln["annotations"]:
                if i + cbox["zmin"] == stroke["frame"]:
                    drew = True
                    draw.line(
                        [
                            (
                                int(round((x[0] - cbox["xmin"])*10)), 
                                int(round((x[1] - cbox["ymin"])*10))
                            ) 
                            for x in stroke["spatial_payload"]
                        ],
                        fill=128,
                        width=int(round(stroke["line_size"]*10))
                    )
            if drew:
                rszd = im.resize((ret.shape[2], ret.shape[1]), Image.BILINEAR)
                ImageDraw.floodfill(rszd, (0,0), 128, thresh=64.1)
                ret[i,:,:] = np.less(np.array(rszd), 64.1).astype(np.int)
    
    return ret


def interpolate_drawings(cropped_drw, step):
    # TODO
    return cropped_drw


def get_blur_kernel_d(affine):
    kerx = signal.gaussian(5, std=1/np.abs(affine[0,2])).reshape(5, 1)
    kerxy = np.outer(kerx, kerx).reshape(1, 5, 5)
    kerz = signal.gaussian(5, std=1/np.abs(affine[2,0])).reshape(5, 1, 1)
    kerxyz = np.outer(kerz, kerxy)
    kerxyz /= np.sum(kerxyz)
    return torch.from_numpy(kerxyz.reshape(1,1,5,5,5)).to("cuda:0")


def get_threshold(region_type):
    # TODO tune this
    if region_type == "ureter":
        return -50
    return -30


def add_renal_hilum(thresholded_d, blr_d, lzn):
    # TODO
    return thresholded_d


def generate_segmentation(region_type, cropped_img, cropped_drw, step=1, affine=None, lzn=None):
    # Interpolate drawings
    interpolated_drw = interpolate_drawings(cropped_drw, step)

    # Send tensors to GPU
    img_d = torch.from_numpy(cropped_img).to("cuda:0")
    drw_d = torch.from_numpy(interpolated_drw).to("cuda:0")

    # Apply a 3d blur convolution
    blur_kernel_d = get_blur_kernel_d(affine)
    blr_d = torch.nn.functional.conv3d(
        img_d.reshape((1,1)+cropped_img.shape), 
        blur_kernel_d, stride=1, padding=2
    ).reshape(cropped_img.shape)

    # Apply threshold
    threshold = get_threshold(region_type)
    thresholded_d = torch.logical_and(
        torch.greater(blr_d, threshold),
        torch.greater(drw_d, 0)
    ).int()

    # If region is kidney, add hilum, redraw, and get new threshold
    if region_type == "kidney":
        thresholded_d = add_renal_hilum(thresholded_d, blr_d, lzn)

    # Bring result back to cpu memory
    return thresholded_d.to("cpu").numpy()


def inflate_seg_to_image_size(cbox, cropped_seg):
    seg_np = np.zeros((cbox["zdim"], cbox["ydim"], cbox["xdim"]), dtype=np.int)
    seg_np[
        cbox["zmin"]:cbox["zmax"] + 1,
        cbox["ymin"]:cbox["ymax"] + 1,
        cbox["xmin"]:cbox["xmax"] + 1,
    ] = cropped_seg
    return seg_np


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
    cbox = get_containing_box(dln, img_nib.shape)
    cropped_img = get_cropped_scan(cbox, img_nib)

    # Generate the drawing made by the annotator
    cropped_drw = generate_cropped_drawing_interior(cbox, dln)

    # Apply heuristics to infer segmentation based on drawing and image
    cropped_seg = generate_segmentation(
        region_type, cropped_img, cropped_drw, cbox["step"], img_nib.affine, lzn
    )

    # Undo cropping to get final segmentation
    seg = inflate_seg_to_image_size(cbox, cropped_seg)

    # Return the seg in nifti format
    return nib.Nifti1Image(seg, img_nib.affine)
